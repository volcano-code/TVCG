"""Actual Compact V2.1 updates with exposure audit and boundary-only exact resume.

This is a NEW runner, not a reconstruction of the unavailable local 17:56
diagnostic_training/train.py. Defaults are Token16 / seed 830251 / lr 3e-5;
normalized Moment and qualified-Jacobian supervision remain the defaults.
Optional case-bound physical residual targets add separate loss memberships.
No benchmark/test input or automatic long-training promotion is supported.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v22_inference_cache import (
    InferenceGeometryCache, geometry_feature_key,
)
from ..p0_diagnostics.baseline_cases import jsonable, moment_scale, sha256
from ..p0_diagnostics.exposure_audit import ExposureRecorder, inventory_digest, read_events, summarize_events
from ..p0_diagnostics import promotion_gate as screen_gate
from ..p0_diagnostics.screen_exports import write_screen_bundle
from .admission import authorize, canonical_digest
from .losses import physical_loss
from .manifest import load_training_manifest
from .state import StatefulShuffle, capture_rng, observed_optimizer_step, restore_rng, seed_everything
from .residual_matching import residual_metrics, validate_residual_target
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import validate_full_cell_rule

CHECKPOINT_SCHEMA = 'compact_pino.training-state.v1'
RESUME_SCHEMA = 'compact_pino.training-resume.v1'


@dataclass(frozen=True)
class TrainConfig:
    token_grid: int = 16
    seed: int = 830251
    learning_rate: float = 3e-5
    batch_size: int = 1
    jacobian_loss_weight: float = 1.
    auxiliary_loss_weight: float = 0.
    chunk_size: int = 65536
    amp: str = 'off'
    deterministic: bool = True
    sampler: str = 'shuffle_without_replacement'
    efficiency: bool = True
    static_cache_bytes: int = 128*1024**2
    query_cache_bytes: int = 128*1024**2
    evaluation_cache_bytes: int = 64*1024**2
    full_cell_rule: str = 'tetra4'
    residual_loss_weight: float = 0.
    constraint_loss_weight: float = 0.

    def validate(self):
        if self.token_grid not in (8,16) or type(self.seed) is not int:
            raise ValueError('Token8/Token16 and an integer seed required')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('positive learning rate required')
        if type(self.batch_size) is not int or self.batch_size < 1 or type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError('positive integer batch/chunk sizes required')
        if not np.isfinite(self.jacobian_loss_weight) or self.jacobian_loss_weight < 0 or self.auxiliary_loss_weight != 0:
            raise ValueError('nonnegative J weight; auxiliary losses are not implemented')
        if self.amp not in ('off','fp16') or type(self.deterministic) is not bool:
            raise ValueError('AMP must be off/fp16 and deterministic must be Boolean')
        if self.sampler != 'shuffle_without_replacement':
            raise ValueError('only the explicit complete-pass shuffle sampler is implemented')
        if type(self.efficiency) is not bool or any(type(value) is not int or value < 0 for value in
                (self.static_cache_bytes,self.query_cache_bytes,self.evaluation_cache_bytes)):
            raise ValueError('Boolean efficiency and nonnegative integer cache byte budgets required')
        validate_full_cell_rule(self.full_cell_rule)
        for value in (self.residual_loss_weight, self.constraint_loss_weight):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError('physical loss weights must be finite nonnegative numbers')
        return self


def _physical_target_coverage(data, config):
    """Fail before allocating a model if an enabled objective has no targets."""
    coverage = dict(train_valid=0, development_valid=0, train_constraint_valid=0, profiles={})
    for role, records in (('train', data.train), ('development', data.development)):
        for record in records:
            case = record['case']
            if config.full_cell_rule != 'tetra4' and getattr(case['blocks'], 'material', None) is None:
                raise ValueError('gl2/gl3 require a block pack with material provenance; rebuild into a fresh cache directory')
            target = record.get('residual_target')
            valid = isinstance(target, dict) and target.get('valid') is True
            if valid:
                validate_residual_target(target, case['native_moments'], coefficients=case['coefficients'])
                coverage[role + '_valid'] += 1
                coverage['profiles'][target['profile']] = coverage['profiles'].get(target['profile'], 0) + 1
            if role == 'train':
                positive = valid and record.get('residual_weight', record['moment_weight']) > 0
                constraint_positive = positive and len(target['multipliers']) > 0
                coverage['train_constraint_valid'] += int(constraint_positive)
                coverage.setdefault('train_positive', 0)
                coverage['train_positive'] += int(positive)
                effective = (record['moment_weight'] > 0 or
                    case['meta']['derivative_valid'] and record['jacobian_weight'] > 0 and config.jacobian_loss_weight > 0 or
                    positive and config.residual_loss_weight > 0 or
                    constraint_positive and config.constraint_loss_weight > 0)
                if not effective:
                    raise ValueError('training row has no enabled positive-weight objective: ' + record['record_id'])
    if config.residual_loss_weight > 0 and not coverage.get('train_positive', 0):
        raise ValueError('residual_loss_weight enabled but no admitted positive-weight training targets')
    if config.constraint_loss_weight > 0 and not coverage['train_constraint_valid']:
        raise ValueError('constraint_loss_weight enabled but no admitted positive-weight constraint targets')
    return coverage


def _physical_metrics(values, record):
    target = record.get('residual_target')
    if not isinstance(target, dict) or target.get('valid') is not True:
        return dict(availability='UNAVAILABLE_PROFILE', reason=(target or {}).get('reason', 'NO_RESIDUAL_TARGET'))
    case = record['case']
    return residual_metrics(values, case['native_moments'], target, coefficients=case['coefficients'])


def _physical_exposure_at_checkpoint(events, checkpoint, config):
    """Bind participation to this checkpoint, not to a later training update."""
    receipt = json.loads(Path(checkpoint['receipt_path']).read_text())
    if (receipt['checkpoint']['sha256'] != checkpoint['sha256'] or
            receipt['exposure_receipt']['checkpoint_sha256'] != checkpoint['sha256']):
        raise ValueError('physical exposure checkpoint receipt mismatch')
    if receipt['exposure_receipt'] not in events:
        raise ValueError('physical exposure receipt is absent from the recorded checkpoint lineage')
    identifier = receipt['exposure_receipt']['checkpoint_id']
    counts = summarize_events(events, selected_checkpoint=identifier)['selected_lineage']['counts']
    requested = {'residual': config.residual_loss_weight > 0, 'constraint': config.constraint_loss_weight > 0}
    participation = {name: dict(requested=enabled, committed_occurrences=counts['committed_'+name+'_occurrences'])
                     for name, enabled in requested.items()}
    exercised = all(not value['requested'] or value['committed_occurrences'] > 0
                    for value in participation.values())
    return dict(checkpoint_sha256=checkpoint['sha256'], checkpoint_id=identifier,
                objectives=participation, enabled_objectives_exercised=exercised)


def _json_write(path, value):
    path = Path(path)
    with path.open('x',encoding='utf8') as stream:
        json.dump(jsonable(value),stream,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def _tensor_digest(state):
    digest = hashlib.sha256()
    for key,value in sorted(state.items()):
        if isinstance(value,torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode()); digest.update(str((array.dtype,array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _evaluation_steps(protocol,max_steps,*,required):
    steps=protocol.get('evaluation_steps')
    if steps is None:
        if required:
            raise ValueError('real training requires predeclared evaluation_steps in the frozen protocol')
        return None
    if (not isinstance(steps,list) or not steps
            or any(type(step) is not int or step <= 0 for step in steps)
            or steps != sorted(set(steps))):
        raise ValueError('evaluation_steps must be strictly increasing positive integers')
    if not any(step <= max_steps for step in steps):
        raise ValueError('the declared absolute budget must contain at least one registered evaluation step')
    return tuple(steps)


def _read_resume(path):
    path = Path(path).resolve(strict=True)
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != RESUME_SCHEMA:
        raise ValueError('a runner-issued resume receipt is required')
    checkpoint = Path(receipt['checkpoint']['path'])
    checkpoint = checkpoint if checkpoint.is_absolute() else path.parent/checkpoint
    if sha256(checkpoint) != receipt['checkpoint']['sha256']:
        raise ValueError('checkpoint SHA256 mismatch')
    if receipt['exposure_receipt'].get('checkpoint_sha256') != receipt['checkpoint']['sha256']:
        raise ValueError('checkpoint hash differs from its exposure lineage receipt')
    events = read_events(receipt['journal_paths'])
    if receipt['exposure_receipt'] not in events:
        raise ValueError('checkpoint exposure receipt missing from parent journals')
    summarize_events(events,selected_checkpoint=receipt['exposure_receipt']['checkpoint_id'])
    receipt['checkpoint_path'] = str(checkpoint)
    receipt['receipt_path'] = str(path)
    return receipt


def _execution_sources(model):
    root=Path(__file__).resolve().parents[3]
    paths={Path(__file__).with_name(name) for name in ('runner.py','state.py','losses.py','manifest.py','admission.py',
                                                    'residual_matching.py')}
    adapter=root/'research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter'
    paths.update(adapter/name for name in ('compact_pino_v21.py','v21_response.py','v21_response_surface.py',
                                          'v21_blocks.py','unified_factory.py','v21_material.py','v21_native.py',
                                          'v22_geometry_inputs.py','v22_query_cache.py','v22_inference_cache.py'))
    paths.update(root/name for name in ('src/spin/SpinMomentProblem.cpp','include/spin/MomentInertiaAlgebra.h'))
    paths.add(adapter/'v21_material_native.cpp')
    paths.update(Path(__file__).parent.parent/'p0_diagnostics'/name
                 for name in ('baseline_cases.py','exposure_audit.py','export_cases.py','handoff_authority.py',
                              'promotion_gate.py','screen_exports.py'))
    for module in model.modules():
        name=inspect.getsourcefile(type(module))
        if name and Path(name).resolve().is_relative_to(root):
            paths.add(Path(name).resolve())
    return {str(path.relative_to(root)):sha256(path) for path in sorted(paths)}


def _gradient_stats(model):
    groups = ('shared_encoder.pino_encoder.','geometry_attention.','response.')
    result = {}; all_finite = True
    for prefix in groups:
        gradients = [p.grad.detach() for name,p in model.named_parameters()
                     if name.startswith(prefix) and p.grad is not None]
        finite = all(bool(torch.isfinite(grad).all()) for grad in gradients)
        all_finite = all_finite and finite
        norm = (float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients])))
                if gradients and finite else 0. if not gradients else None)
        result[prefix] = dict(parameters_with_grad=len(gradients),finite=finite,norm=norm)
    # Include any trainable parameters outside the three reported groups.
    all_finite = all_finite and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    return all_finite,result


class Trainer:
    def __init__(self, data, output, *, config=None, mode='smoke', device='cpu',
                 max_steps=2, max_attempts=4, authorization_path=None,
                 resume=None, run_id=None, evaluation_protocol_path=None, registered_protocol_path=None,
                 initial_weights=None, initial_weights_sha256=None):
        self.segment_started = time.perf_counter()
        self.elapsed_at_resume = 0.
        self.config = (config or TrainConfig()).validate()
        self.config_sha256 = canonical_digest(asdict(self.config))
        if resume is not None and initial_weights is not None:
            raise ValueError('initial weights start a new run and are mutually exclusive with exact resume')
        if initial_weights is None and initial_weights_sha256 is not None:
            raise ValueError('initial weight SHA requires its explicit checkpoint path')
        self.initial_weights = None
        self.data,self.mode = data,mode
        if (mode == 'smoke') != (data.scope == 'synthetic'):
            raise ValueError('synthetic smoke and real training data cannot be interchanged')
        self.residual_target_coverage = _physical_target_coverage(data, self.config)
        self.resume_receipt = _read_resume(resume) if resume is not None else None
        resumed_hash = self.resume_receipt['checkpoint']['sha256'] if self.resume_receipt else None
        self.authorization = authorize(mode=mode,manifest_sha256=data.manifest_sha256,
            config_sha256=self.config_sha256,max_steps=max_steps,max_attempts=max_attempts,
            authorization_path=authorization_path,resume_checkpoint_sha256=resumed_hash)
        self.protocol = None; self.protocol_path = None; self.registered_protocol_path = None
        self.protocol_sha256 = None; self.selection_state = None; self.evaluation_steps=None
        if evaluation_protocol_path is not None:
            if registered_protocol_path is None:
                raise ValueError('registered protocol provenance required before evaluation')
            self.protocol_path = Path(evaluation_protocol_path).resolve(strict=True)
            self.registered_protocol_path = Path(registered_protocol_path).resolve(strict=True)
            self.protocol = screen_gate._read_json(self.protocol_path)
            expected_ids = screen_gate._protocol(self.protocol)
            if self.protocol['registered_protocol_sha256'] != sha256(self.registered_protocol_path):
                raise ValueError('registered evaluation protocol SHA256 mismatch')
            if expected_ids != {row['record_id'] for row in data.development}:
                raise ValueError('frozen development IDs differ from protocol or include unloaded records')
            if any(row['role']=='development' and not row['loader_eligible'] for row in data.records):
                raise ValueError('evaluation cannot silently drop declared development cases')
            self.protocol_sha256 = sha256(self.protocol_path)
            self.evaluation_steps=_evaluation_steps(self.protocol,max_steps,required=mode!='smoke')
        if mode != 'smoke':
            if self.protocol is None:
                raise ValueError('real training requires its frozen evaluation protocol before the first update')
            if self.authorization.get('evaluation_protocol_sha256') != self.protocol_sha256:
                raise ValueError('authorization must bind the frozen evaluation protocol')
        self.max_steps,self.max_attempts = max_steps,max_attempts
        self.device = torch.device(device)
        if self.config.amp == 'fp16' and self.device.type != 'cuda':
            raise ValueError('fp16 runner AMP is CUDA-only; CPU smoke uses full precision')
        if (self.device.type == 'cuda' and self.config.deterministic
                and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8')):
            raise ValueError('deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before process launch')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        seed_everything(self.config.seed,self.config.deterministic)
        self.model = CompactPINOv21(token_grid=self.config.token_grid,efficiency=self.config.efficiency,
            static_cache_bytes=self.config.static_cache_bytes,query_cache_bytes=self.config.query_cache_bytes,
            full_cell_rule=self.config.full_cell_rule)
        if self.config.amp == 'fp16' and any(parameter.is_complex() for parameter in self.model.parameters()):
            raise ValueError('fp16 AMP is not qualified for the complex Fourier parameters; use amp=off')
        self.model.to(self.device)
        self.execution_sources = _execution_sources(self.model)
        if initial_weights is not None:
            initial_path=Path(initial_weights).resolve(strict=True)
            if not initial_weights_sha256 or sha256(initial_path) != initial_weights_sha256:
                raise ValueError('explicit initial checkpoint SHA256 mismatch')
            initial_state=torch.load(initial_path,map_location=self.device,weights_only=True)
            if initial_state.get('schema') != CHECKPOINT_SCHEMA:
                raise ValueError('initial weights currently accept only this runner checkpoint format')
            self.model.load_state_dict(initial_state['model'],strict=True)
            self.initial_weights=dict(path=str(initial_path),sha256=initial_weights_sha256,
                                      scope='weights_only_new_optimizer_rng_sampler_and_run')
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.learning_rate,
                                          betas=(.9,.999),eps=1e-8,weight_decay=0.)
        # Explicit constant scheduler; its state is still saved/restored.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=1.)
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.config.amp=='fp16')
        self.sampler = StatefulShuffle(len(data.train),self.config.seed)
        self.global_step,self.attempts = 0,0
        self.run_id = run_id or ('compact-v21-'+uuid4().hex)
        self.runtime = dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                            numpy=str(np.__version__),device=str(self.device),deterministic=self.config.deterministic,
                            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                            cuda_device_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
                            cuda_compute_capability=list(torch.cuda.get_device_capability(self.device)) if self.device.type=='cuda' else None,
                            torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        self.journal_paths = []
        if self.resume_receipt:
            state = torch.load(self.resume_receipt['checkpoint_path'],map_location=self.device,weights_only=True)
            self._restore(state,run_id)
            self.journal_paths = list(self.resume_receipt['journal_paths'])
        if self.global_step > max_steps or self.attempts > max_attempts:
            raise ValueError('resume counters exceed the requested absolute budget')
        self.model.train()
        self.journal = self.output/'exposure.jsonl'
        self.audit = ExposureRecorder(self.journal,inventory=data.inventory,run_id=self.run_id,
            resume_from=self.resume_receipt['exposure_receipt'] if self.resume_receipt else None,
            start_step=self.global_step,durable=True,
            initial_checkpoint_sha256=self.initial_weights['sha256'] if self.initial_weights else None)
        self.journal_paths.append(str(self.journal))
        self.metrics_stream = (self.output/'steps.jsonl').open('x',encoding='utf8')
        self.last_checkpoint = None
        self.closed = False
        _json_write(self.output/'config.json',asdict(self.config))
        _json_write(self.output/'run.json',dict(schema='compact_pino.training-run.v1',mode=mode,
            scope=data.scope,run_id=self.run_id,config=asdict(self.config),config_sha256=self.config_sha256,
            training_manifest_sha256=data.manifest_sha256,runtime=self.runtime,
            execution_sources=self.execution_sources,initial_weights=self.initial_weights,
            source_admission=data.source_admission,
            residual_target_coverage=self.residual_target_coverage,
            authorization=self.authorization,registered_train_count=data.inventory['registered_count'],
            eligible_train_records=len(data.train),eligible_development_records=len(data.development),
            admission_note='Loader membership and completed updates do not establish model promotion.'))

    def elapsed_seconds(self):
        return self.elapsed_at_resume + time.perf_counter()-self.segment_started

    def _bind_self_references(self,checkpoint):
        if self.selection_state is not None:
            for key in ('epoch0_checkpoint','best_checkpoint'):
                if self.selection_state.get(key) == 'SELF':
                    self.selection_state[key] = dict(checkpoint)

    def _restore(self,state,requested_run_id):
        if state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('unsupported training state schema')
        for name,expected in (('config_sha256',self.config_sha256),
                              ('training_manifest_sha256',self.data.manifest_sha256),
                              ('inventory_sha256',inventory_digest(self.data.inventory)),
                              ('evaluation_protocol_sha256',self.protocol_sha256),
                              ('execution_sources',self.execution_sources),
                              ('runtime',self.runtime)):
            if state.get(name) != expected:
                raise ValueError(f'exact-resume identity mismatch: {name}')
        if requested_run_id is not None and requested_run_id != state['run_id']:
            raise ValueError('exact resume preserves run_id')
        self.run_id = state['run_id']
        self.initial_weights = state.get('initial_weights')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.sampler.load_state_dict(state['sampler'])
        self.global_step,self.attempts = state['global_step'],state['attempts']
        self.elapsed_at_resume = max(float(state['cumulative_runner_seconds']),
                                    float(self.resume_receipt['cumulative_runner_seconds']))
        self.selection_state = copy.deepcopy(state.get('selection'))
        self._bind_self_references(dict(path=self.resume_receipt['checkpoint_path'],
            sha256=self.resume_receipt['checkpoint']['sha256'],receipt_path=self.resume_receipt['receipt_path'],
            model_tensor_sha256=self.resume_receipt['model_tensor_sha256']))
        if self.selection_state:
            for key in ('epoch0_evaluation','best_evaluation','no_learning_evaluation'):
                entry=self.selection_state.get(key)
                if entry is not None and sha256(entry['path']) != entry['sha256']:
                    raise ValueError('selection evaluation asset changed on resume: '+key)
            for key in ('epoch0_checkpoint','best_checkpoint'):
                entry=self.selection_state[key]
                if entry is not None and sha256(entry['path']) != entry['sha256']:
                    raise ValueError('selection checkpoint asset changed on resume: '+key)
        if self.global_step != self.resume_receipt['exposure_receipt']['optimizer_step']:
            raise ValueError('checkpoint update count contradicts its exposure receipt')
        if (self.resume_receipt['exposure_receipt']['run_id'] != self.run_id
                or self.resume_receipt['exposure_receipt']['inventory_sha256'] != state['inventory_sha256']):
            raise ValueError('checkpoint run/inventory contradicts exposure lineage')
        restore_rng(state['rng'])

    def _predictions(self,records,*,evaluation=False,geometry_cache=None):
        if geometry_cache is not None and not evaluation:
            raise ValueError('persistent feature cache is restricted to one fixed-weight evaluation')
        caches = {} if geometry_cache is None else geometry_cache
        predictions = []
        for record in records:
            case = record['case']; meta=case['meta']
            key = geometry_feature_key(case['blocks'],case['fixed_moments'],meta['density'])
            encoded = caches.get(key)
            if encoded is None:
                encoded = self.model.encode_geometry(case['blocks'],case['fixed_moments'],
                                                     density=meta['density'],inference=evaluation)
                caches[key] = encoded
            with_j = meta['derivative_valid'] and (evaluation or
                     self.config.jacobian_loss_weight>0 and record['jacobian_weight']>0)
            predictions.append(self.model.evaluate_operand(encoded,case['operand'],
                with_jacobian=with_j,chunk_size=self.config.chunk_size))
        # Training caches are local to this forward. Evaluation callers may
        # retain a bounded graph-free cache until this fixed-weight pass ends.
        return predictions

    def step(self):
        if self.closed or self.global_step >= self.max_steps or self.attempts >= self.max_attempts:
            raise RuntimeError('training closed or absolute budget exhausted')
        wall=self.authorization.get('max_wall_seconds')
        if wall is not None and self.elapsed_seconds() >= wall:
            raise RuntimeError('cumulative runner wall budget exhausted')
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        records = [self.data.train[i] for i in self.sampler.next_batch(self.config.batch_size)]
        self.attempts += 1
        batch = f'attempt-{self.attempts}'; window=batch
        start = time.perf_counter()
        context = torch.autocast(device_type='cuda',dtype=torch.float16) if self.config.amp=='fp16' else nullcontext()
        try:
            with context:
                predictions = self._predictions(records)
        except Exception as exc:
            self.audit.forward(batch,window,[r['record_id'] for r in records],success=False,
                               reason=f'{type(exc).__name__}: {exc}')
            self.audit.optimizer_step(window,committed=False,reason='forward_failed_no_update')
            raise
        self.audit.forward(batch,window,[r['record_id'] for r in records])
        try:
            loss,info = physical_loss(predictions,records,jacobian_loss_weight=self.config.jacobian_loss_weight,
                residual_loss_weight=self.config.residual_loss_weight,
                constraint_loss_weight=self.config.constraint_loss_weight)
        except Exception:
            self.audit.optimizer_step(window,committed=False,reason='loss_construction_failed_no_update')
            raise
        self.audit.loss(batch,**{key:info[key] for key in
            ('moment_mask','moment_weight','jacobian_mask','jacobian_weight',
             'residual_mask','residual_weight','constraint_mask','constraint_weight')})
        if not torch.isfinite(loss):
            self.audit.backward(batch,success=False,reason='nonfinite_loss_backward_not_run')
            self.audit.optimizer_step(window,committed=False,reason='nonfinite_loss_no_update')
            raise FloatingPointError('nonfinite training loss')
        active = any(prediction.diagnostics.get('active_nodes',0)>0 and (mm or jm or rm or cm)
                     for prediction,mm,jm,rm,cm in zip(predictions,info['moment_mask'],info['jacobian_mask'],
                                                       info['residual_mask'],info['constraint_mask']))
        if not loss.requires_grad or not active:
            self.audit.backward(batch,success=False,reason='NO_PARAMETER_DEPENDENCE_backward_not_run')
            self.audit.optimizer_step(window,committed=False,reason='NO_PARAMETER_DEPENDENCE')
            _,gradients=_gradient_stats(self.model)
            return self._step_record(records,loss,info,start,False,gradients,'NO_PARAMETER_DEPENDENCE')
        try:
            self.scaler.scale(loss).backward()
        except Exception as exc:
            self.audit.backward(batch,success=False,reason=f'{type(exc).__name__}: {exc}')
            self.audit.optimizer_step(window,committed=False,reason='backward_failed_no_update')
            raise
        self.audit.backward(batch,success=True)
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        finite,gradients = _gradient_stats(self.model)
        if not finite and not self.scaler.is_enabled():
            self.audit.optimizer_step(window,committed=False,reason='nonfinite_unscaled_gradient_no_update')
            raise FloatingPointError('nonfinite gradients without AMP')
        try:
            committed = observed_optimizer_step(self.optimizer,self.scaler)
        except Exception as exc:
            self.audit.optimizer_step(window,committed=False,
                reason=f'optimizer_failed_state_not_checkpointed: {type(exc).__name__}: {exc}')
            raise
        self.audit.optimizer_step(window,committed=committed,
                                  reason=None if committed else 'AMP_OVERFLOW_SKIP')
        if committed:
            self.global_step += 1
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return self._step_record(records,loss,info,start,committed,gradients,
                                 None if committed else 'AMP_OVERFLOW_SKIP')

    def _step_record(self,records,loss,info,start,committed,gradients,skip_reason):
        record = dict(attempt=self.attempts,global_step=self.global_step,record_ids=[r['record_id'] for r in records],
                      committed=committed,amp_skip=skip_reason=='AMP_OVERFLOW_SKIP',skip_reason=skip_reason,
                      loss=float(loss.detach()),**info,
                      gradients=gradients,seconds=time.perf_counter()-start,
                      cumulative_runner_seconds=self.elapsed_seconds(),
                      sampler=dict(epoch=self.sampler.epoch,cursor=self.sampler.cursor,count=self.sampler.count))
        self.metrics_stream.write(json.dumps(jsonable(record),allow_nan=False)+'\n')
        self.metrics_stream.flush()
        return record

    def save_checkpoint(self):
        """Save complete boundary state, then journal its hash, then issue receipt."""
        if self.closed:
            raise RuntimeError('cannot save a closed trainer')
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('boundary checkpoint requires cleared accumulated gradients')
        identifier = f'step-{self.global_step}-attempt-{self.attempts}-{uuid4().hex[:12]}'
        destination = self.output/(identifier+'.pt')
        state = dict(schema=CHECKPOINT_SCHEMA,run_id=self.run_id,config=asdict(self.config),
            config_sha256=self.config_sha256,training_manifest_sha256=self.data.manifest_sha256,
            inventory_sha256=inventory_digest(self.data.inventory),runtime=self.runtime,
            execution_sources=self.execution_sources,initial_weights=self.initial_weights,
            evaluation_protocol_sha256=self.protocol_sha256,selection=copy.deepcopy(self.selection_state),
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),scaler=self.scaler.state_dict(),rng=capture_rng(),
            sampler=self.sampler.state_dict(),global_step=self.global_step,attempts=self.attempts,
            cumulative_runner_seconds=self.elapsed_seconds())
        with tempfile.NamedTemporaryFile(dir=self.output,delete=False) as stream:
            staging = Path(stream.name)
            try:
                torch.save(state,stream)
                stream.flush(); os.fsync(stream.fileno())
                os.replace(staging,destination)
            finally:
                staging.unlink(missing_ok=True)
        digest = sha256(destination)
        event = self.audit.checkpoint(identifier,digest,optimizer_step=self.global_step)
        receipt = dict(schema=RESUME_SCHEMA,checkpoint=dict(path=destination.name,sha256=digest),
                       exposure_receipt=event,journal_paths=self.journal_paths,
                       cumulative_runner_seconds=self.elapsed_seconds(),
                       wall_budget_scope='cumulative runner sessions through checkpoint receipt; initial data loading excluded; checked at update boundaries',
                       model_tensor_sha256=_tensor_digest(state['model']))
        receipt_path = destination.with_suffix('.resume.json')
        _json_write(receipt_path,receipt)
        self.last_checkpoint = dict(path=str(destination),sha256=digest,receipt_path=str(receipt_path),
                                    model_tensor_sha256=receipt['model_tensor_sha256'])
        self._bind_self_references(self.last_checkpoint)
        return dict(self.last_checkpoint)

    def evaluate_development(self):
        """Fixed development only. Returned rows can feed the separate screen.

        The normalized diagnostic summary is not the screen's primary metric;
        the frozen evaluation protocol decides raw/scaled aggregation explicitly.
        """
        saved_rng = capture_rng(); was_training=self.model.training
        rows=[]; grouped={}
        feature_cache = InferenceGeometryCache(self.config.evaluation_cache_bytes if self.config.efficiency else 0)
        self.model.eval()
        try:
            for record in self.data.development:
                case=record['case']; meta=case['meta']
                with torch.no_grad():
                    prediction=self._predictions([record],evaluation=True,geometry_cache=feature_cache)[0]
                values=prediction.moments.detach().cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError('nonfinite development prediction; do not omit failed rows')
                jac=None if prediction.jacobian is None else prediction.jacobian.detach().cpu().numpy()
                if jac is not None and not np.isfinite(jac).all():
                    raise FloatingPointError('nonfinite development Jacobian')
                scale=moment_scale(meta['editable_mass'],meta['length_scale'])
                rmse=float(np.sqrt(np.mean(((values-case['native_moments'])/scale)**2)))
                grouped.setdefault(meta['source_group'],[]).append(rmse)
                rows.append(dict(record_id=record['record_id'],status='available',moments=values,
                                 jacobian=jac,jacobian_status='available' if jac is not None else 'unavailable',
                                 chart_id=meta['chart_id'],failure_reason=None if jac is not None else
                                 'label_unqualified' if not meta['derivative_valid'] else 'response_nonregular',
                                 diagnostics=prediction.diagnostics,
                                 physical_residual=_physical_metrics(values, record)))
        finally:
            feature_cache_stats = feature_cache.snapshot()
            feature_cache.clear()
            self.model.train(was_training); restore_rng(saved_rng)
        return jsonable(dict(rows=rows,source_groups={key:float(np.mean(value)) for key,value in grouped.items()},
            normalized_source_macro_mean_query_rmse=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
            promotion_status='NOT_EVALUATED',scope='synthetic' if self.mode=='smoke' else 'development',
            efficiency=dict(evaluation_features=feature_cache_stats,operator=self.model.efficiency_statistics())))

    def _raw_selection_score(self,evaluation):
        if self.protocol is None:
            raise ValueError('no frozen selection protocol')
        if sha256(self.protocol_path) != self.protocol_sha256:
            raise ValueError('evaluation protocol changed after training started')
        by_id={row['record_id']:row for row in self.data.development}
        predictions={row['record_id']:row for row in evaluation['rows']}
        if set(predictions) != set(by_id):
            raise ValueError('selection must cover every frozen development query')
        errors={rid:np.asarray(predictions[rid]['moments'])-item['case']['native_moments']
                for rid,item in by_id.items()}
        labels={rid:dict(source_group_id=item['case']['meta']['source_group']) for rid,item in by_id.items()}
        result=screen_gate._metric(errors,labels,by_id,self.protocol['primary_aggregation'])
        if result['value'] is None:
            raise ValueError('incomplete selection metric')
        return result

    def evaluate_no_learning(self):
        from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import evaluate_packed
        from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
        rows=[]
        for item in self.data.development:
            case=item['case']; operand=case['operand']; meta=case['meta']
            result=evaluate_packed(case['blocks'],prepare_surface(operand.vertices,operand.faces),
                case['blocks'].pitch,operand.operation,case['fixed_moments'],
                operand.vertex_velocity if meta['derivative_valid'] else None,model=None,
                chunk_size=self.config.chunk_size,full_cell_rule=self.config.full_cell_rule)
            rows.append(dict(record_id=item['record_id'],status='available',
                moments=result.moments.detach().cpu().numpy(),
                jacobian=None if result.jacobian is None else result.jacobian.detach().cpu().numpy(),
                jacobian_status='unavailable' if result.jacobian is None else 'available',
                chart_id=meta['chart_id'],failure_reason='derivative_unavailable' if result.jacobian is None else None,
                diagnostics=result.diagnostics,
                physical_residual=_physical_metrics(result.moments.detach().cpu().numpy(), item)))
        return jsonable(dict(rows=rows,model_weights_used=False))

    def _selection_evaluate(self,*,initialize=False):
        if not initialize and self.evaluation_steps is not None and self.global_step not in self.evaluation_steps:
            raise ValueError('unregistered update position cannot enter candidate selection')
        evaluation=self.evaluate_development()
        score=self._raw_selection_score(evaluation)
        path=self.output/f'development-step-{self.global_step}-attempt-{self.attempts}.json'
        _json_write(path,evaluation)
        entry=dict(path=str(path),sha256=sha256(path))
        if initialize:
            no_learning=self.evaluate_no_learning()
            no_path=self.output/'no-learning.json'; _json_write(no_path,no_learning)
            self.selection_state=dict(epoch0_checkpoint='SELF',epoch0_evaluation=entry,
                epoch0_score=score['value'],epoch0_global_step=self.global_step,
                best_checkpoint=None,best_evaluation=None,best_score=None,
                best_global_step=None,primary_aggregation=self.protocol['primary_aggregation'],
                primary_units='raw_undoubled_ten_moments',tie_break='earliest_strict_minimum',
                no_learning_evaluation=dict(path=str(no_path),sha256=sha256(no_path)))
        elif (self.global_step > self.selection_state['epoch0_global_step'] and
              (self.selection_state['best_score'] is None or score['value'] < self.selection_state['best_score'])):
            self.selection_state.update(best_checkpoint='SELF',best_evaluation=entry,
                                        best_score=score['value'],best_global_step=self.global_step)
        self.selection_state['last_evaluated_position']=[self.global_step,self.attempts]
        return score

    def _export_screen(self):
        selected=self.selection_state
        if selected['best_checkpoint'] is None:
            return dict(bundle_path=None,report_path=None,decision='INSUFFICIENT_EVIDENCE',
                        reason='NO_POST_UPDATE_CANDIDATE',screen_run=False)
        bundle=write_screen_bundle(self.data,protocol_path=self.protocol_path,
            registered_protocol_path=self.registered_protocol_path,
            epoch0_evaluation=json.loads(Path(selected['epoch0_evaluation']['path']).read_text()),
            candidate_evaluation=json.loads(Path(selected['best_evaluation']['path']).read_text()),
            no_learning_evaluation=json.loads(Path(selected['no_learning_evaluation']['path']).read_text()),
            run_id=self.run_id,config=asdict(self.config),
            epoch0_checkpoint_path=selected['epoch0_checkpoint']['path'],
            candidate_checkpoint_path=selected['best_checkpoint']['path'],output_directory=self.output/'screen')
        report=screen_gate.load_and_screen(bundle)
        _json_write(self.output/'screen_report.json',report)
        return dict(bundle_path=str(bundle),report_path=str(self.output/'screen_report.json'),decision=report['decision'])

    def run(self,*,checkpoint_every=100,evaluate_every=0):
        if checkpoint_every < 1 or evaluate_every < 0:
            raise ValueError('positive checkpoint frequency and nonnegative eval frequency required')
        if self.evaluation_steps is not None and evaluate_every:
            expected={step for step in self.evaluation_steps if step<=self.max_steps}
            if expected != set(range(evaluate_every,self.max_steps+1,evaluate_every)):
                raise ValueError('evaluate-every conflicts with frozen evaluation_steps; use zero to follow the explicit schedule')
        start=time.perf_counter(); update_count=0; start_global_step=self.global_step; selection_evaluations=0
        stop_reason='STEP_BUDGET'
        try:
            if self.protocol is not None and self.selection_state is None:
                self._selection_evaluate(initialize=True)
            initial=self.save_checkpoint()
            while self.global_step < self.max_steps and self.attempts < self.max_attempts:
                wall=self.authorization.get('max_wall_seconds')
                if wall is not None and self.elapsed_seconds() >= wall:
                    stop_reason='WALL_BUDGET'; break
                row=self.step(); update_count+=1
                evaluated=False
                scheduled=(self.global_step in self.evaluation_steps if self.evaluation_steps is not None else
                           bool(evaluate_every and self.global_step % evaluate_every == 0))
                if self.protocol is not None and row['committed'] and scheduled:
                    self._selection_evaluate(); evaluated=True; selection_evaluations+=1
                if evaluated or self.attempts % checkpoint_every == 0:
                    self.save_checkpoint()
            if self.attempts >= self.max_attempts and self.global_step < self.max_steps:
                stop_reason='ATTEMPT_BUDGET'
            if (self.protocol is not None and self.evaluation_steps is None
                    and self.selection_state['last_evaluated_position'] != [self.global_step,self.attempts]):
                self._selection_evaluate()
                if self.global_step>self.selection_state['epoch0_global_step']:
                    selection_evaluations+=1
            last=self.save_checkpoint()
            events=read_events(self.journal_paths)
            physics_exposure=_physical_exposure_at_checkpoint(events,last,self.config)
            candidate=self.selection_state['best_checkpoint'] if self.selection_state else None
            candidate_exposure=_physical_exposure_at_checkpoint(events,candidate,self.config) if candidate else None
            candidate_exercised=candidate_exposure is None or candidate_exposure['enabled_objectives_exercised']
            screening=(self._export_screen() if candidate_exercised else
                       dict(decision='INSUFFICIENT_EVIDENCE',reason='ENABLED_PHYSICAL_OBJECTIVE_NOT_EXPOSED',
                            screen_run=False)) if self.protocol is not None else None
            _json_write(self.output/'last.json',last)
            if self.selection_state is not None and self.selection_state['best_checkpoint'] is not None:
                _json_write(self.output/'best.json',dict(self.selection_state['best_checkpoint'],
                    primary_score=self.selection_state['best_score'],best_global_step=self.selection_state['best_global_step'],
                    primary_aggregation=self.protocol['primary_aggregation']))
            summary=dict(schema='compact_pino.training-result.v1',scope=self.data.scope,
                status='TRAINING_BUDGET_FINISHED_NOT_PROMOTION',stop_reason=stop_reason,
                global_step=self.global_step,attempts=self.attempts,initial_checkpoint=initial,
                last_checkpoint=last,config_sha256=self.config_sha256,
                training_manifest_sha256=self.data.manifest_sha256,seconds=time.perf_counter()-start,
                update_records=update_count,segment_committed_updates=self.global_step-start_global_step,
                no_parameter_updates=(self.global_step==start_global_step),
                cumulative_runner_seconds=self.elapsed_seconds(),
                physical_supervision=physics_exposure,
                physical_objectives_exercised=physics_exposure['enabled_objectives_exercised'],
                selected_candidate_physical_supervision=candidate_exposure,
                residual_target_coverage=self.residual_target_coverage,
                best_checkpoint=self.selection_state['best_checkpoint'] if self.selection_state else None,
                epoch0_checkpoint=self.selection_state['epoch0_checkpoint'] if self.selection_state else None,
                screen=screening,promotion_status='NOT_AUTHORIZATION',
                candidate_evaluations_this_segment=selection_evaluations,
                evaluation_steps=list(self.evaluation_steps) if self.evaluation_steps is not None else None,
                candidate_status=('NO_POST_UPDATE_CANDIDATE' if not self.selection_state or self.selection_state['best_checkpoint'] is None
                    else 'CONTINUED_EXISTING_BEST_NO_NEW_EVALUATION' if selection_evaluations==0
                    else 'EVALUATED_REGISTERED_CANDIDATES' if self.evaluation_steps is not None
                    else 'EVALUATED_SOFTWARE_FALLBACK_CANDIDATES'),
                selection_status='FROZEN_RAW_SOURCE_MACRO' if self.selection_state else 'NOT_PERFORMED_SOFTWARE_ONLY')
            _json_write(self.output/'result.json',summary)
            self.close()
            return summary
        except BaseException:
            self.close(interrupted=True)
            raise

    def close(self,*,interrupted=False):
        if not self.closed:
            self.audit.close(interrupted=interrupted)
            self.metrics_stream.close()
            self.closed=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',required=True,choices=['smoke','trial','long'])
    parser.add_argument('--config',help='explicit JSON TrainConfig; defaults remain software proposal, not old Pilot equivalence')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-steps',type=int,required=True,help='absolute committed optimizer-step ceiling, including resumed history')
    parser.add_argument('--max-attempts',type=int,required=True,help='absolute attempt ceiling including AMP skips')
    parser.add_argument('--authorization',help='required budget/protocol authorization for real trial or long')
    parser.add_argument('--evaluation-protocol',help='frozen screen/selection protocol, required before real training')
    parser.add_argument('--registered-protocol',help='original protocol file bound by evaluation-protocol')
    parser.add_argument('--resume',help='runner-issued .resume.json, never an unbound weight file')
    parser.add_argument('--initial-weights',help='new run from a common runner checkpoint model only; optimizer/RNG/sampler reset')
    parser.add_argument('--initial-weights-sha256',help='required exact digest with --initial-weights')
    parser.add_argument('--cache-directory')
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--evaluate-every',type=int,default=0)
    args=parser.parse_args()
    config=TrainConfig(**json.loads(Path(args.config).read_text())) if args.config else TrainConfig()
    data=load_training_manifest(args.manifest,mode=args.mode,cache_directory=args.cache_directory)
    trainer=Trainer(data,args.output,config=config,mode=args.mode,device=args.device,
                    max_steps=args.max_steps,max_attempts=args.max_attempts,
                    authorization_path=args.authorization,resume=args.resume,
                    evaluation_protocol_path=args.evaluation_protocol,registered_protocol_path=args.registered_protocol,
                    initial_weights=args.initial_weights,initial_weights_sha256=args.initial_weights_sha256)
    # Protocol files are constructor inputs so they are checked before updates.
    result=trainer.run(checkpoint_every=args.checkpoint_every,evaluate_every=args.evaluate_every)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
