"""Runnable preparation, software smoke, training and paired-query comparison.

No command re-labels software tests as original-mesh accuracy or GPU speed.
Native Newton integration is the registered callback in deployment.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_geometry import make_hierarchy
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_hierarchy import evaluate_deterministic
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
from experiments.compact_pino_v21.training.state import seed_everything
from .dataset import canonical_digest,dataset_binding,load_henmo_dataset,model_contract
from .training import HENMOTrainer,HENMOTrainingConfig,SCHEMA,_tensor_digest
from .metrics import prediction_metrics,summarize_prediction_metrics


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _model_config(path):
    return HENMOConfig(**json.loads(Path(path).read_text()))


def _training_config(path):
    return HENMOTrainingConfig(**json.loads(Path(path).read_text())).validate() if path else HENMOTrainingConfig()


def _device(device):
    result = torch.device(device)
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU timing will be labelled GPU')
    return result


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _place_model(config,device,dtype):
    # Module.to(dtype=<real>) discards the imaginary part of spectral weights.
    # The dedicated float/double methods change floating tensors only, preserving
    # the encoder's explicit complex Fourier parameterization.
    model = HENMOOperator(config).to(device=device)
    return model.double() if dtype == 'float64' else model.float()


def _environment():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count())


def _new_output_directory(path):
    """Never combine fresh receipts with a previous or partially failed run."""
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('training/smoke output must be a new or empty directory; resume into a new directory')
    path.mkdir(parents=True,exist_ok=True)
    return path


def config_command(args):
    if args.profile == 'cpu_smoke':
        config = HENMOConfig.cpu_smoke()
    elif args.profile == 'e16_compatible':
        config = HENMOConfig.e16_compatible()
    else:
        config = HENMOConfig(native_material_resolution=int(args.profile.removeprefix('pilot')))
    _write(args.output,asdict(config))
    print(json.dumps(dict(config=str(Path(args.output).resolve()),fingerprint=config.fingerprint(),
                         status='CONFIGURATION_ONLY_NOT_A_TRAINED_MODEL')))


def smoke_command(args):
    from .fixtures import synthetic_dataset
    output = _new_output_directory(args.output)
    torch.set_num_threads(args.threads)
    training = HENMOTrainingConfig(max_steps=3,max_attempts=6)
    seed_everything(training.seed)
    data = synthetic_dataset()
    config = HENMOConfig.cpu_smoke(response_zero_init=False)
    start = time.perf_counter()
    with HENMOOperator(config).double() as model:
        binding = dataset_binding(data,model_contract(model))
        trainer = HENMOTrainer(model,training,binding,records=data.train)
        first = trainer.step()
        receipt = trainer.save_checkpoint(output/'resume.pt')
        second = trainer.step()
        tensors = {k:v.detach().clone() for k,v in model.state_dict().items()}
        with HENMOOperator(config).double() as restored:
            resumed = HENMOTrainer(restored,training,binding,records=data.train)
            resumed.load_checkpoint(output/'resume.pt',expected_sha256=receipt['sha256'])
            again = resumed.step()
            exact = all(torch.equal(value,tensors[name]) for name,value in restored.state_dict().items())
            if not exact or not first['committed'] or not second['committed'] or not again['committed']:
                raise RuntimeError('real M/J update or exact next-step restore failed')
        model.eval()
        case = data.train[0]['case']
        cache = model.encode_geometry(case['blocks'],case['fixed_moments'],inference=True)
        with torch.inference_mode():
            prediction = model.evaluate_operand(cache,case['operand'],parameter_scales=case['parameter_scales'])
        report = dict(schema='henmo.software-smoke.v1',status='PASS',environment=_environment(),
            scope='CPU analytic software fixture; no production teacher or blind geometry',
            actual_committed_updates=trainer.step_count,resumed_next_update_exact=exact,
            initial_and_restored_query_order_equal=second['record_ids']==again['record_ids'],
            config=asdict(config),checkpoint=receipt,steps=[first,second],
            prediction=prediction_metrics(prediction,data.train[0]),diagnostics=prediction.diagnostics,
            wall_s=time.perf_counter()-start,physics_promotion=False,gpu_speedup_proven=False)
        _write(output/'smoke.json',report)
    print(json.dumps(dict(status='PASS',report=str((output/'smoke.json').resolve()),
                         actual_committed_updates=2,resumed_next_update_exact=True)))


def audit_command(args):
    config = _model_config(args.config)
    started = time.perf_counter()
    data,coverage = load_henmo_dataset(args.manifest,mode=args.mode,
        allowed_native_resolutions=(config.native_material_resolution,),cache_directory=args.cache)
    report = dict(coverage,manifest_sha256=data.manifest_sha256,snapshot_id=data.snapshot_id,
                  material_layout='prepared',config=asdict(config),load_s=time.perf_counter()-started)
    _write(args.output,report)
    print(json.dumps(dict(report=str(Path(args.output).resolve()),loaded_train=len(data.train),
                         loaded_development=len(data.development),missing_evidence=coverage['missing'])))


def _initialize(model,args,output):
    if not args.initialize_from:
        if args.transfer_rules or args.initialize_sha256:
            raise ValueError('transfer rules/hash require --initialize-from')
        return None
    if not args.transfer_rules or not args.initialize_sha256:
        raise ValueError('initialization requires explicit source SHA256 and semantic rules')
    from .migration import ModuleSemantics,TransferRule,transfer_from_checkpoint
    raw = json.loads(Path(args.transfer_rules).read_text())
    rules = [TransferRule(row['source_prefix'],row['target_prefix'],
        ModuleSemantics(**row['source_semantics']),ModuleSemantics(**row['target_semantics'])) for row in raw]
    state_key = getattr(args,'initialize_state_key','model')
    receipt = transfer_from_checkpoint(model,args.initialize_from,rules,
        expected_sha256=args.initialize_sha256,state_key=None if state_key == 'none' else state_key,
        source_label='explicit_cli_initialization')
    _write(output/'initialization.json',receipt)
    return receipt


def train_command(args):
    output = _new_output_directory(args.output)
    config,training = _model_config(args.config),_training_config(args.training_config)
    if config.response_path == 'deterministic':
        raise ValueError('deterministic baseline has no learning update')
    device = _device(args.device)
    seed_everything(training.seed,training.deterministic)
    data,coverage = load_henmo_dataset(args.manifest,mode=args.mode,cache_directory=args.cache,
                                      allowed_native_resolutions=(config.native_material_resolution,))
    _write(output/'coverage.json',coverage)
    model = _place_model(config,device,args.dtype)
    try:
        if args.resume and args.initialize_from:
            raise ValueError('exact resume and semantic initialization are different operations')
        _initialize(model,args,output)
        trainer = HENMOTrainer(model,training,dataset_binding(data,model_contract(model)),records=data.train)
        resume_receipt = (trainer.load_checkpoint(args.resume,expected_sha256=args.resume_sha256)
                          if args.resume else None)
        initial_checkpoint = trainer.save_checkpoint(output/'initial.pt')
        start_step = trainer.step_count
        initial = trainer.evaluate(data.development)
        _write(output/f'development_step_{trainer.step_count:06d}.json',initial)
        selected = None
        best = float('inf')
        initial_score = initial['source_macro_raw_moment_rmse']
        if initial['complete'] and initial_score is not None:
            best = initial_score
            selected = trainer.save_checkpoint(output/'selected.pt')
        while trainer.step_count < training.max_steps and trainer.attempts < training.max_attempts:
            step = trainer.step()
            with (output/'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(step,ensure_ascii=False,allow_nan=False)+'\n')
            if not step['committed']:
                continue
            if trainer.step_count % args.evaluate_every == 0 or trainer.step_count == training.max_steps:
                evaluation = trainer.evaluate(data.development)
                _write(output/f'development_step_{trainer.step_count:06d}.json',evaluation)
                score = evaluation['source_macro_raw_moment_rmse']
                if evaluation['complete'] and score is not None and score < best:
                    best = score
                    selected = trainer.save_checkpoint(output/'selected.pt')
                trainer.save_checkpoint(output/'last.pt')
                print(json.dumps(dict(step=trainer.step_count,attempts=trainer.attempts,
                                      development_source_macro_raw_moment_rmse=score)),flush=True)
        final = trainer.save_checkpoint(output/'last.pt')
        _write(output/'run.json',dict(schema='henmo.training-run.v1',config=asdict(config),
            training_config=asdict(training),environment=_environment(),last=final,selected=selected,
            device=str(device),dtype=args.dtype,initial=initial_checkpoint,resumed_from=resume_receipt,
            state_tensor_dtypes=sorted({str(value.dtype) for value in model.state_dict().values()}),
            selection_scope=dict(start_step=start_step,end_step=trainer.step_count,
                includes_starting_state=True,
                scope='current_continuation_only' if args.resume else 'entire_current_run'),
            selection_protocol='HENMO_starting_incumbent_v1_not_a_replay_of_V21_104_208_312_only',
            selected_step=None if selected is None else selected['step_count'],
            selected_has_new_updates=selected is not None and selected['step_count'] > start_step,
            best_source_macro_raw_moment_rmse=None if selected is None else best,
            update_budget_reached=trainer.step_count == training.max_steps,
            scope=data.scope,initialization='explicit_transfer' if args.initialize_from else 'fresh_or_exact_resume',
            selection='lowest complete development source-macro raw Moment RMSE including starting state; not promotion',
            physics_promotion=False,long_training_promotion=False))
    finally:
        model.close()


def _load_comparison_checkpoint(model,path,data,expected_sha256=None):
    """Evaluation is bound to the same admitted snapshot as this checkpoint.

    A new independently held-out snapshot needs a separately registered source
    isolation/physical acceptance run, rather than bypassing this strict check.
    """
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix+'.json').read_text())
    digest = _sha(path)
    if receipt.get('schema') != 'henmo.checkpoint-receipt.v1' or receipt.get('sha256') != digest:
        raise ValueError('comparison checkpoint receipt SHA256 mismatch')
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError('comparison checkpoint is not the explicitly requested state')
    state = torch.load(path,map_location='cpu',weights_only=True)
    binding = dataset_binding(data,model_contract(model))
    if (state.get('schema') != SCHEMA or state.get('architecture') != model_contract(model)
            or state.get('binding') != binding or state.get('binding_sha256') != canonical_digest(binding)):
        raise ValueError('comparison requires the checkpoint exact architecture and same admitted snapshot binding; '
                         'new independent snapshots require separately registered source-isolation/physical evaluation')
    if (receipt.get('binding_sha256') != state['binding_sha256']
            or receipt.get('model_tensor_sha256') != _tensor_digest(state['model'])):
        raise ValueError('comparison checkpoint tensor or binding receipt mismatch')
    reference = model.state_dict()
    if set(reference) != set(state['model']):
        raise ValueError('comparison checkpoint parameter inventory differs')
    for name,value in state['model'].items():
        if (not isinstance(value,torch.Tensor) or value.shape != reference[name].shape
                or value.dtype != reference[name].dtype
                or (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())):
            raise ValueError('comparison requires matching finite tensor shapes and dtype: '+name)
    model.load_state_dict(state['model'],strict=True)
    return dict(sha256=digest,binding_sha256=state['binding_sha256'],
                model_tensor_sha256=receipt['model_tensor_sha256'],step_count=state['step_count'],
                evaluation_scope='same_admitted_snapshot_development_only')


def compare_command(args):
    config = _model_config(args.config)
    device = _device(args.device)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    data,coverage = load_henmo_dataset(args.manifest,mode=args.mode,cache_directory=args.cache,
                                      allowed_native_resolutions=(config.native_material_resolution,))
    load_s = time.perf_counter()-started
    records = data.development
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError('no admitted development queries for paired comparison')
    seed_everything(830251)
    with _place_model(config,device,args.dtype).eval() as model:
        checkpoint_info = None
        if args.checkpoint:
            checkpoint_info = _load_comparison_checkpoint(model,args.checkpoint,data,args.checkpoint_sha256)
        rows = {arm:[] for arm in ('deterministic_gl2','deterministic_gl3','henmo')}
        timings,preparations = [],[]
        # One material hierarchy is shared algebraically, but query caches start
        # empty for each arm. Grouping allows one geometry build per material
        # without retaining every geometry's hierarchy in memory at once.
        groups = {}
        for record in records:
            groups.setdefault(id(record['case']['blocks']),[]).append(record)
        for group in groups.values():
            begin = time.perf_counter()
            tree = make_hierarchy(group[0]['case']['blocks'],config)
            build_s = time.perf_counter()-begin
            preparations.append(dict(material_identity=tree.material_identity,hierarchy_build_s=build_s,
                                     record_ids=[record['record_id'] for record in group]))
            for arm in rows:
                tree.clear_cache()
                if arm == 'henmo':
                    model.invalidate_caches(force=True)
                for query_index,record in enumerate(group):
                    case,meta = record['case'],record['case']['meta']
                    operand = case['operand']
                    encode_s = 0.
                    if arm == 'henmo':
                        _sync(device); begin = time.perf_counter()
                        cache = model.encode_geometry(tree,case['fixed_moments'],meta['density'],inference=True)
                        _sync(device); encode_s = time.perf_counter()-begin
                    _sync(device); begin = time.perf_counter()
                    diagnostics = {}
                    try:
                        with torch.no_grad():
                            if arm == 'henmo':
                                # evaluate_operand includes preparation of the finite surface.
                                result = model.evaluate_operand(cache,operand,with_jacobian=meta['derivative_valid'],
                                    parameter_scales=case.get('parameter_scales') if meta['derivative_valid'] else None)
                            else:
                                surface = prepare_surface(operand.vertices,operand.faces,
                                                          execution_context=model.surface_execution)
                                result = evaluate_deterministic(tree,surface,
                                    case['blocks'].pitch*config.bandwidth_voxels,operand.operation,case['fixed_moments'],
                                    operand.vertex_velocity if meta['derivative_valid'] else None,
                                    full_cell_rule=arm.rsplit('_',1)[1],chunk_size=config.chunk_size)
                        # The native solver consumes CPU moments/Jacobians, so
                        # completion includes that output transfer for every arm.
                        result = replace(result,moments=result.moments.detach().cpu(),
                            jacobian=None if result.jacobian is None else result.jacobian.detach().cpu())
                        _sync(device); elapsed = time.perf_counter()-begin
                        row = prediction_metrics(result,record)
                        diagnostics = result.diagnostics
                        row['solver_status'] = diagnostics.get('solver_status','DETERMINISTIC_REFERENCE_ONLY')
                    except Exception as exc:
                        _sync(device); elapsed = time.perf_counter()-begin
                        row = dict(record_id=record['record_id'],source_group=meta['source_group'],status='FAILED',
                            derivative_label_qualified=meta['derivative_valid'],moment=None,jacobian=None,
                            error=dict(type=type(exc).__name__,message=str(exc)))
                    rows[arm].append(row)
                    timings.append(dict(record_id=record['record_id'],arm=arm,query_s=elapsed,
                        geometry_encode_s=encode_s,includes_finite_surface_preparation=True,
                        includes_moment_jacobian_cpu_transfer=True,
                        material_query_index=query_index,status=row['status'],
                        device=str(device) if arm == 'henmo' else 'cpu',
                        source_group=meta['source_group'],diagnostics=diagnostics))
            tree.clear_cache()
        report = dict(schema='henmo.paired-query-comparison.v1',scope='admitted development; query comparison only',
            environment=_environment(),config=asdict(config),manifest_sha256=data.manifest_sha256,dtype=args.dtype,
            state_tensor_dtypes=sorted({str(value.dtype) for value in model.state_dict().values()}),
            checkpoint=checkpoint_info,model_trained=checkpoint_info is not None and checkpoint_info['step_count'] > 0,
            loaded_data_preparation_s=load_s,arms={k:summarize_prediction_metrics(v) for k,v in rows.items()},
            loaded_data_scope='manifest/material/label audit loading; not an inference-only cold-start benchmark',
            shared_material_preparation=preparations,
            shared_hierarchy_build_s=sum(row['hierarchy_build_s'] for row in preparations),
            candidate_geometry_encode_s=sum(row['geometry_encode_s'] for row in timings),
            arm_query_s={arm:sum(row['query_s'] for row in timings if row['arm']==arm) for arm in rows},
            timings=timings,missing_coverage=coverage['missing'],
            comparator_note='deterministic GL2/GL3 are smooth-response cubature; labels are independently admitted native values',
            latency_scope='one observed query per arm including finite surface preparation; shared hierarchy and candidate encoding reported separately; not full-solve acceleration evidence',
            cache_scope='material query caches cleared at each arm start, then reused within that material/arm; runtime compilation and warmup are not controlled',
            physics_promotion=False,acceleration_proven=False)
        _write(args.output,report)
    print(json.dumps(dict(report=str(Path(args.output).resolve()),queries=len(records),acceleration_proven=False)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    cfg = commands.add_parser('config')
    cfg.add_argument('--profile',choices=('pilot256','pilot512','pilot1024','e16_compatible','cpu_smoke'),default='pilot512')
    cfg.add_argument('--output',required=True); cfg.set_defaults(run=config_command)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--output',required=True); smoke.add_argument('--threads',type=int,default=2)
    smoke.set_defaults(run=smoke_command)
    for name,handler in (('audit-data',audit_command),('train',train_command),('compare',compare_command)):
        sub = commands.add_parser(name)
        sub.add_argument('--manifest',required=True); sub.add_argument('--config',required=True)
        sub.add_argument('--mode',choices=('smoke','trial','long'),default='trial',
                         help='smoke accepts explicitly synthetic fixtures only; trial/long require native source admission')
        sub.add_argument('--cache'); sub.add_argument('--output',required=True)
        sub.set_defaults(run=handler)
        if name != 'audit-data':
            sub.add_argument('--device',default='cuda:0')
            sub.add_argument('--dtype',choices=('float32','float64'),default='float32')
        if name == 'train':
            sub.add_argument('--training-config'); sub.add_argument('--resume'); sub.add_argument('--resume-sha256')
            sub.add_argument('--initialize-from'); sub.add_argument('--initialize-sha256'); sub.add_argument('--transfer-rules')
            sub.add_argument('--initialize-state-key',default='model',
                             help='checkpoint mapping key; use none for a bare state dict')
            sub.add_argument('--evaluate-every',type=int,default=104)
        if name == 'compare':
            sub.add_argument('--checkpoint'); sub.add_argument('--checkpoint-sha256'); sub.add_argument('--limit',type=int)
            sub.add_argument('--threads',type=int,default=2)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args,'evaluate_every',1) < 1 or getattr(args,'threads',1) < 1:
        raise ValueError('positive evaluation interval and thread count required')
    if getattr(args,'limit',None) is not None and args.limit < 1:
        raise ValueError('positive query limit required')
    if getattr(args,'resume_sha256',None) and not getattr(args,'resume',None):
        raise ValueError('--resume-sha256 requires --resume')
    if getattr(args,'checkpoint_sha256',None) and not getattr(args,'checkpoint',None):
        raise ValueError('--checkpoint-sha256 requires --checkpoint')
    args.run(args)


if __name__ == '__main__':
    main()
