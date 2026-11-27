"""Registered HENMO callback and native-verified hybrid orchestration.

The deployed Newton equations are not reimplemented here.  A caller supplies
one registered solver adapter, used unchanged with either evaluator. Candidate
queries construct the native finite operand only; native material Booleans are
reserved for explicit correction and the existing independent verifier.

Registration hashes bind supplied evidence. They do not create teacher
qualification, certify a learned response, or authorize training promotion.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from collections.abc import Mapping
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.compact_pino_v21.runtime_efficiency.deployment import RuntimePredictor, sha256
from experiments.compact_pino_v21.runtime_efficiency import acceptance
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_config import HENMOConfig
from .dataset import canonical_digest, model_contract


REGISTRATION_SCHEMA = 'henmo.deployment-registration.v1'


def _hash_array(h, name, value):
    array = np.ascontiguousarray(value)
    h.update(name.encode()); h.update(str((array.dtype.str, array.shape)).encode())
    if array.size:
        data = memoryview(array).cast('B')
        for begin in range(0, len(data), 1024*1024):
            h.update(data[begin:begin+1024*1024])


def geometry_digest(request):
    h = hashlib.sha256(b'henmo.original-design-mesh.v1')
    _hash_array(h, 'vertices', np.asarray(request.design_vertices, dtype='<f8'))
    _hash_array(h, 'faces', np.asarray(request.design_faces, dtype='<i8'))
    return h.hexdigest()


def _configuration_value(value):
    """Serialize actual solver settings, never an object's address or repr.

    A callback may use iteration/line-search controls beyond the acceptance
    tolerances. They are bound too, including nested structures. Unknown native
    settings objects must expose a plain configuration instead of being silently
    dropped from the task identity.
    """
    if isinstance(value,Enum):
        return _configuration_value(value.value)
    if isinstance(value,np.ndarray):
        return _configuration_value(value.tolist())
    if isinstance(value,np.generic):
        return _configuration_value(value.item())
    if isinstance(value,Mapping):
        if any(not isinstance(key,str) for key in value):
            raise ValueError('native settings mapping keys must be strings')
        return {key:_configuration_value(item) for key,item in value.items()}
    if is_dataclass(value) and not isinstance(value,type):
        return _configuration_value(asdict(value))
    if isinstance(value,(list,tuple)):
        return [_configuration_value(item) for item in value]
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    raise ValueError('native settings contain an unsupported configuration value: '+type(value).__name__)


def _all_settings(settings):
    if is_dataclass(settings) or isinstance(settings,Mapping):
        raw=settings
    elif hasattr(settings,'__dict__') and vars(settings):
        raw=vars(settings)
    else:
        raise ValueError('native settings need an explicit complete dataclass or mapping configuration')
    result=_configuration_value(raw)
    # Validate finiteness as part of the frozen contract, rather than allowing
    # NaN/Infinity to produce a platform-specific or ambiguous identity.
    canonical_digest(result)
    return result


def _task_values(request):
    settings = request.settings
    return dict(mode=request.mode, objective=request.spinning_top_objective,
                center_of_mass_weight=float(request.center_of_mass_weight),
                density=float(request.material_density),
                target_axes=np.asarray(request.target_axes).tolist(),
                target_origin=np.asarray(request.target_origin).tolist(),
                fixed_moments=np.asarray(request.fixed_shell_moments).tolist(),
                complete_solver_settings=_all_settings(settings),
                equation_profile=settings.equation_profile,
                equation_tolerance=settings.equation_tolerance,
                physical_relative_tolerance=settings.physical_relative_tolerance)


def task_digest(request):
    return canonical_digest(dict(geometry_sha256=geometry_digest(request), **_task_values(request)))


def _source(source):
    # Sparse hierarchy inputs are bound to their original material storage.
    if hasattr(source, 'material_identity'):
        source.assert_source_unchanged()
        return source.blocks if source.blocks is not None else source.material
    return source


def _source_arrays(source):
    source = _source(source)
    if hasattr(source, 'full_runs'):
        return [(name, getattr(source, name)) for name in ('origin','full_runs','partial_cells','boundary_nodes')]
    if hasattr(source, 'block_voxels'):
        arrays = [(name, getattr(source, name)) for name in ('origin','nodes','ids','offsets','moments','centers')]
        if source.material is not None:
            arrays.extend(('provenance.'+name,value) for name,value in _source_arrays(source.material))
        return arrays
    raise TypeError('deployment requires an admitted PreparedMaterial, BlockMeasure, or hierarchy')


def material_digest(source):
    """Hash actual static bytes once at setup, without expanding native voxels."""
    source = _source(source)
    h = hashlib.sha256(b'henmo.deployment-material.v1')
    h.update(str((source.identity, source.side, source.pitch,
                  getattr(source,'block_voxels',None), getattr(source,'density',None),
                  getattr(source,'source',None), getattr(source,'partial_rule',None))).encode())
    for name,array in _source_arrays(source):
        _hash_array(h,name,array)
    return h.hexdigest()


def registration_digest(registration):
    if isinstance(registration, (str, Path)):
        registration = json.loads(Path(registration).read_text())
    return canonical_digest(registration)


def candidate_registration(backend, source, checkpoint, config, *, admission_id):
    """Create a DRAFT inventory for review, never an admitted registration.

    Work must reconcile these fields with its frozen source/teacher registry.
    Copying this draft and changing its status is not independent qualification.
    """
    if not isinstance(config,HENMOConfig) or not isinstance(admission_id,str) or not admission_id:
        raise ValueError('explicit HENMO config and source-admission ID required')
    weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
    source = _source(source)
    return dict(schema=REGISTRATION_SCHEMA,status='DRAFT_NOT_ADMITTED',admission_id=admission_id,
                geometry_sha256=geometry_digest(backend.request),task_sha256=task_digest(backend.request),
                material_sha256=material_digest(source),material_identity=source.identity,
                native_resolution=int(source.side),density=float(backend.request.material_density),
                fixed_moments=np.asarray(backend.fixed,dtype=float).tolist(),
                checkpoint_sha256=sha256(checkpoint),training_binding_sha256=weights.get('binding_sha256'),
                architecture_sha256=canonical_digest(weights.get('architecture')),
                config_sha256=config.fingerprint())


def _model_versions(model):
    return tuple((kind,name,id(value),value._version,value.data_ptr(),value.dtype,value.device)
                 for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers()))
                 for name,value in items)


def _storage_state(source):
    s = _source(source)
    return (id(s),s.identity,s.side,s.pitch,
            tuple((name,id(a),a.shape,a.dtype.str,a.__array_interface__['data'][0],a.flags.writeable)
                  for name,a in _source_arrays(s)))


def _request_state(backend):
    request=backend.request
    identities=[]
    for name in ('design_vertices','design_faces'):
        raw=getattr(request,name);a=np.asarray(raw)
        if isinstance(raw,np.ndarray):
            identities.append((name,id(a),a.shape,a.dtype.str,a.__array_interface__['data'][0],a.flags.writeable))
        else:
            # Nonarray requests remain supported; their geometry must be hashed
            # on use because there is no immutable NumPy storage to pin.
            identities.append((name,canonical_digest(a.tolist())))
    return (id(request),canonical_digest(_task_values(request)),
            np.asarray(backend.fixed,dtype=float).tobytes(),tuple(identities))


class _QualifiedModel:
    """Enforce HENMO's refinement/event qualification before native export."""
    def __init__(self,operator):
        self.operator=operator

    def __getattr__(self,name):
        return getattr(self.operator,name)

    def evaluate_operand(self,*args,**kwargs):
        prediction=self.operator.evaluate_operand(*args,**kwargs)
        status=prediction.diagnostics.get('solver_status')
        if not isinstance(status,str):
            raise ValueError('HENMO proposal omitted its solver qualification')
        if status.startswith('NATIVE_REQUIRED_'):
            raise ValueError(status)
        if status not in ('PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE','MOMENT_ONLY_REQUIRES_NATIVE_ACCEPTANCE'):
            raise ValueError('unrecognized HENMO proposal qualification: '+status)
        return prediction


class HENMORuntimePredictor(RuntimePredictor):
    """Reuse raw-to-unit J conversion once; forbid stale checkpoint/task use."""
    def __init__(self,backend,model,cache,source,*,checkpoint_sha256,metadata,
                 registration,trace_sink=None,profile_sync=False):
        super().__init__(backend,_QualifiedModel(model),cache,checkpoint_sha256=checkpoint_sha256,
                         metadata=metadata,trace_sink=trace_sink,profile_sync=profile_sync)
        self.operator,self.source=model,source
        self.registration=deepcopy(registration)
        # The native evaluator and material index consume immutable static bytes.
        for _,array in _source_arrays(source):
            array.flags.writeable=False
        for name in ('design_vertices','design_faces','target_axes','target_origin','fixed_shell_moments'):
            array=getattr(backend.request,name)
            if isinstance(array,np.ndarray):
                array.flags.writeable=False
        if isinstance(backend.fixed,np.ndarray):
            backend.fixed.flags.writeable=False
        self._versions=_model_versions(model)
        self._storage=_storage_state(source)
        self._request=_request_state(backend)
        self._config=model.config.fingerprint()

    def _check_frozen(self):
        if self._closed:
            raise RuntimeError('predictor is closed')
        if _model_versions(self.operator)!=self._versions or self.operator.config.fingerprint()!=self._config:
            raise ValueError('frozen deployment checkpoint/config changed; rebuild the predictor')
        if _storage_state(self.source)!=self._storage:
            raise ValueError('registered material storage changed; rebuild the predictor')
        if _request_state(self.backend)!=self._request:
            raise ValueError('registered native geometry/task changed; rebuild the predictor')

    def evaluate(self,query,*,with_jacobian=True):
        self._check_frozen()
        packet=super().evaluate(query,with_jacobian=with_jacobian)
        packet['evaluation_map_identity']=type(self.operator).__name__+'-'+self.digest+'-'+self._config
        return packet

    def close(self):
        if self._closed:
            return
        try:
            super().close()
        finally:
            self.operator.close()


def _new_model(config):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_operator import HENMOOperator
    return HENMOOperator(config)


def make_predictor(backend,source,checkpoint,registration,*,expected_registration_sha256,
                   config=None,device='cuda:0',metadata=None,trace_sink=None,profile_sync=False):
    """Load an exact frozen HENMO state for an independently registered task.

    The expected registration digest must come from Work's frozen registry, not
    from trusting an arbitrary newly loaded file. Setup timing includes byte
    verification and feature encoding; earlier mesh/material creation is not
    silently counted as free or included in that timing.
    """
    start=perf_counter()
    if isinstance(registration,(str,Path)):
        registration=json.loads(Path(registration).read_text())
    registration=deepcopy(registration)
    if (not isinstance(registration,dict) or registration.get('schema')!=REGISTRATION_SCHEMA
            or registration.get('status')!='ADMITTED' or not registration.get('admission_id')):
        raise ValueError('an admitted frozen deployment registration is required')
    if registration_digest(registration)!=expected_registration_sha256:
        raise ValueError('frozen deployment registration digest mismatch')
    acceptance._check_native_request(backend,backend.request)
    checkpoint=Path(checkpoint)
    if sha256(checkpoint)!=registration.get('checkpoint_sha256'):
        raise ValueError('registered checkpoint hash mismatch')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (weights.get('schema')!='henmo.training-state.v1'
            or weights.get('binding_sha256')!=canonical_digest(weights.get('binding'))
            or weights.get('binding_sha256')!=registration.get('training_binding_sha256')
            or canonical_digest(weights.get('architecture'))!=registration.get('architecture_sha256')):
        raise ValueError('registered checkpoint training/architecture binding mismatch')
    arch=weights['architecture']
    if config is None:
        config=HENMOConfig(**arch['config'])
    if not isinstance(config,HENMOConfig) or config.fingerprint()!=registration.get('config_sha256'):
        raise ValueError('registered HENMO scientific config mismatch')
    s=_source(source)
    if (geometry_digest(backend.request)!=registration.get('geometry_sha256')
            or task_digest(backend.request)!=registration.get('task_sha256')):
        raise ValueError('registered original geometry/native task mismatch')
    if (s.identity!=registration.get('material_identity')
            or s.side!=registration.get('native_resolution') or s.side!=config.native_material_resolution
            or material_digest(s)!=registration.get('material_sha256')):
        raise ValueError('registered material content/identity/resolution mismatch')
    fixed=np.asarray(registration.get('fixed_moments'),dtype=float)
    if (fixed.shape!=(10,) or not np.isfinite(fixed).all() or fixed[0]<0
            or not np.array_equal(fixed,backend.fixed)
            or not np.array_equal(fixed,backend.request.fixed_shell_moments)):
        raise ValueError('registered fixed material differs from native task')
    density=registration.get('density')
    source_density=getattr(s,'density',getattr(getattr(s,'material',None),'density',density))
    if (not isinstance(density,(int,float)) or isinstance(density,bool) or not np.isfinite(density)
            or density<=0 or density!=backend.request.material_density or density!=source_density):
        raise ValueError('registered material density differs from native task')
    model=_new_model(config)
    try:
        if model_contract(model)!=arch:
            raise ValueError('checkpoint does not belong to this HENMO implementation/config')
        expected=model.state_dict()
        if set(expected)!=set(weights['model']):
            raise ValueError('checkpoint parameter inventory mismatch')
        for name,value in weights['model'].items():
            if (not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape
                    or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('checkpoint tensor mismatch: '+name)
        model.load_state_dict(weights['model'],strict=True)
        model.to(device).eval()
        with torch.no_grad():
            cache=model.encode_geometry(source,fixed,density=density,inference=True)
        if torch.device(device).type=='cuda':
            torch.cuda.synchronize(device)
        if metadata is None:
            metadata={}
        metadata.update(arm='HENMO_V1PLUS',registration_sha256=expected_registration_sha256,
                        registration=registration,checkpoint=str(checkpoint.resolve()),
                        checkpoint_sha256=registration['checkpoint_sha256'],step_count=weights.get('step_count'),
                        config=asdict(config),setup_elapsed_s=perf_counter()-start,
                        setup_started_perf_counter=start,
                        original_material_preprocessing_included=False,
                        query_condition='same finite native operand before material Boolean',
                        native_acceptance_required=True,solver_equations_replaced=False)
        return HENMORuntimePredictor(backend,model,cache,source,
                    checkpoint_sha256=registration['checkpoint_sha256'],metadata=metadata,
                    registration=registration,trace_sink=trace_sink,profile_sync=profile_sync)
    except BaseException:
        model.close()
        raise


@dataclass(frozen=True)
class SolverCandidate:
    query: np.ndarray
    status: str
    objective: float

    @classmethod
    def parse(cls,value):
        if isinstance(value,cls):
            query,status,objective=value.query,value.status,value.objective
        elif isinstance(value,dict):
            query,status,objective=value.get('query'),value.get('status'),value.get('objective')
        else:
            raise TypeError('registered solver must return SolverCandidate or its three-field mapping')
        q=np.asarray(query,dtype=float)
        if (q.shape!=(10,) or not np.isfinite(q).all() or not np.isfinite(np.linalg.norm(q))
                or np.linalg.norm(q)==0 or not isinstance(status,str) or not status
                or isinstance(objective,bool) or not isinstance(objective,(int,float))
                or not np.isfinite(objective)):
            raise ValueError('registered solver returned an incomplete/nonfinite candidate')
        return cls(q.copy(),status,float(objective))


class RegisteredSolver:
    """Explicit adapter to Work's solver, with checked source and task identities.

    callback(evaluator, initial_query) -> SolverCandidate. The callback retains
    the registered objective, constraints, iteration budget and line search.
    It must call the supplied evaluator for every M/J request. No surrogate
    packet or callback-reported accepted flag can certify a returned result.
    """
    def __init__(self,callback,*,source_path,expected_source_sha256,expected_task_sha256):
        if not callable(callback):
            raise TypeError('registered solver callback is required')
        self.callback=callback
        self.source_path=Path(source_path).resolve()
        target=callback if inspect.isfunction(callback) else callback.__call__
        actual=inspect.getsourcefile(target)
        if actual is None or Path(actual).resolve()!=self.source_path or sha256(self.source_path)!=expected_source_sha256:
            raise ValueError('registered solver source identity mismatch')
        self.source_sha256=expected_source_sha256
        if not isinstance(expected_task_sha256,str) or len(expected_task_sha256)!=64:
            raise ValueError('registered solver task digest required')
        self.task_sha256=expected_task_sha256

    def __call__(self,evaluator,query):
        if sha256(self.source_path)!=self.source_sha256:
            raise ValueError('registered solver source changed')
        return SolverCandidate.parse(self.callback(evaluator,np.array(query,dtype=float,copy=True)))


class _CountedEvaluator:
    def __init__(self,exact):
        self.exact,self.calls=exact,0
    def __getattr__(self,name):
        return getattr(self.exact,name)
    def __call__(self,query):
        self.calls+=1
        return self.exact(query)
    def evaluate_moments(self,query):
        self.calls+=1
        return self.exact.evaluate_moments(query)


def solve_hybrid(predictor,solver,initial_queries,*,max_proposals=1,max_native_attempts=1,
                 verify_direct=True,timing_context=None):
    """Bounded optional multi-start scheduling; native verify decides acceptance.

    Starts execute sequentially through the existing callback. This is not a
    GPU-parallel throughput claim. The native budget counts solver attempts,
    not assumed Newton steps or Boolean calls. Every failure/verification is
    timed. Full end-to-end timing needs a caller-provided read-start receipt
    from this process, begun before make_predictor and original mesh access.
    Required timing_context keys: regime ('cold' or 'material_cache_reuse'),
    read_start_perf_counter, and True for includes_original_mesh_read,
    includes_material_preparation_or_cache_open, includes_backend_initialization,
    includes_checkpoint_and_geometry_encoding. Omitted/incomplete receipts
    leave timing explicitly partial and ineligible for end-to-end comparison.
    """
    if not isinstance(solver,RegisteredSolver):
        raise TypeError('a source/task-bound RegisteredSolver is required')
    if any(type(n) is not int or n<1 for n in (max_proposals,max_native_attempts)) or type(verify_direct) is not bool:
        raise ValueError('positive proposal/native budgets and Boolean verify_direct required')
    predictor._check_frozen()
    if solver.task_sha256!=predictor.registration['task_sha256']:
        raise ValueError('solver and predictor are bound to different native tasks')
    starts=[SolverCandidate.parse(dict(query=q,status='INITIAL',objective=0.)).query for q in initial_queries]
    if not starts:
        raise ValueError('at least one frozen initial query is required')
    start=perf_counter();before=predictor.calls
    proposal_records=[];candidates=[];verification_records=[];native_records=[]
    exact=_CountedEvaluator(predictor.backend)
    def result(accepted,candidate=None,verification=None):
        now=perf_counter()
        required=('includes_original_mesh_read','includes_material_preparation_or_cache_open',
                  'includes_backend_initialization','includes_checkpoint_and_geometry_encoding')
        receipt={} if timing_context is None else dict(timing_context)
        missing=[key for key in required if receipt.get(key) is not True]
        if receipt.get('regime') not in ('cold','material_cache_reuse'):
            missing.append('regime')
        read_start=receipt.get('read_start_perf_counter')
        setup_start=predictor.metadata.get('setup_started_perf_counter')
        if (isinstance(read_start,bool) or not isinstance(read_start,(int,float))
                or not np.isfinite(read_start) or setup_start is None or read_start>setup_start
                or read_start<0 or read_start>now):
            missing.append('read_start_before_factory')
        complete=not missing
        fixture=bool(verification is not None and
                     verification.get('certified_integral_evidence',{}).get('fixture_only'))
        return dict(schema='henmo.hybrid-result.v1',accepted=accepted,
                    query=None if candidate is None else candidate.query.tolist(),
                    native_verification=verification,proposals=proposal_records,
                    native_attempts=native_records,verifications=verification_records,
                    neural_query_calls=predictor.calls-before,native_solver_evaluator_calls=exact.calls,
                    native_verification_calls=len(verification_records),
                    workflow_seconds=now-start,
                    factory_setup_seconds=predictor.metadata.get('setup_elapsed_s'),
                    original_material_preprocessing_included=complete,
                    timing_scope='caller_observed_end_to_end' if complete else 'PARTIAL_WORKFLOW_ONLY',
                    timing_context=receipt,timing_missing_components=missing,
                    end_to_end_seconds=now-read_start if complete else None,
                    eligible_for_end_to_end_comparison=complete and not fixture,
                    acceleration_proven=False,
                    proposal_execution='sequential_registered_solver',gpu_parallel_multistart=False,
                    solver_source_sha256=solver.source_sha256,
                    acceptance_source='existing_original_mesh_verify_only',
                    guaranteed_polish_steps=None)
    def verify(candidate,kind,index):
        ts=perf_counter()
        row=dict(kind=kind,index=index,query=candidate.query.tolist())
        try:
            predictor._check_frozen()
            out=acceptance.verify(predictor.backend,candidate.query,predictor.backend.request,candidate.status)
            row.update(accepted=out['accepted'],result=out)
            return out
        except Exception as exc:
            row.update(accepted=False,error=type(exc).__name__+': '+str(exc))
            return None
        finally:
            row['seconds']=perf_counter()-ts
            verification_records.append(row)
    for index,q in enumerate(starts[:max_proposals]):
        ts=perf_counter();row=dict(initial_index=index,initial_query=q.tolist())
        try:
            candidate=solver(predictor,q)
            candidates.append((candidate.objective,index,candidate))
            row.update(status=candidate.status,objective=candidate.objective,query=candidate.query.tolist())
        except Exception as exc:
            row.update(status='PROPOSAL_FAILED',error=type(exc).__name__+': '+str(exc))
        row['seconds']=perf_counter()-ts;proposal_records.append(row)
    candidates.sort(key=lambda item:(item[0],item[1]))
    # A failed neural proposal does not silently become success. Its original
    # initial state is an explicit native fallback under the same attempt cap.
    if not candidates:
        candidates=[(0.,i,SolverCandidate(q,'INITIAL_FALLBACK',0.)) for i,q in enumerate(starts[:max_proposals])]
    for _,index,candidate in candidates[:max_native_attempts]:
        if verify_direct and candidate.status=='CONVERGED':
            checked=verify(candidate,'direct',index)
            if checked is not None and checked['accepted']:
                return result(True,candidate,checked)
        ts=perf_counter();row=dict(initial_index=index,warm_start=candidate.query.tolist())
        polished=None
        try:
            predictor._check_frozen()
            polished=solver(exact,candidate.query)
            row.update(status=polished.status,query=polished.query.tolist(),objective=polished.objective)
        except Exception as exc:
            row.update(status='NATIVE_FAILED',error=type(exc).__name__+': '+str(exc))
        row['seconds']=perf_counter()-ts;native_records.append(row)
        if polished is not None:
            checked=verify(polished,'corrected',index)
            if checked is not None and checked['accepted']:
                return result(True,polished,checked)
    return result(False)
