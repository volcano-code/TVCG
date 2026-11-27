"""No-update, same-query path and objective diagnostics for HENMO.

The CLI verifies checkpoint/snapshot binding before calling these functions.
Execution overrides are labelled experiments, never silent checkpoint migration.
The deterministic references integrate the same material and finite operand;
they are smooth cubature references, not exact original-mesh Boolean oracles.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict, replace
import inspect
import math
import time

import numpy as np
import torch

from experiments.compact_pino_v21.training.losses import physical_loss
from experiments.compact_pino_v21.training.state import capture_rng, restore_rng
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_geometry import make_hierarchy
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_hierarchy import evaluate_deterministic
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_response import HENMOBoundaryResponse
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface

from .metrics import prediction_metrics, summarize_prediction_metrics


MOMENT_ORDER = ['m', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'yz', 'xz', 'xy']
MODULE_GROUPS = ('shared_encoder', 'geometry_attention', 'local_geometry', 'response')


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Keep a failed prediction inspectable without emitting non-JSON NaN.
        return {'nonfinite': repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported diagnostic value: ' + type(value).__name__)


def _sync(model):
    devices = {parameter.device for parameter in model.parameters() if parameter.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)


class _DiagnosticStageTrace:
    """Bounded phase totals for the operator's genuinely nested scopes.

    QueryTrace intentionally rejects nesting, while material_rule in this
    operator contains distance and response scopes. Inclusive durations here
    overlap; exclusive durations subtract child intervals. CUDA sync and
    detailed tracing add overhead, so these totals are diagnosis, not speedup.
    """
    def __init__(self, synchronize=None, clock=None):
        self.synchronize = synchronize
        self.clock = clock or time.perf_counter
        self.stack = []
        self.totals = {}
        self.sync_errors = []

    @contextmanager
    def stage(self, name):
        if self.synchronize is not None:
            self.synchronize()
        frame = [self.clock(), 0.]
        self.stack.append(frame)
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if self.synchronize is not None:
                    self.synchronize()
            except BaseException as exc:
                self.sync_errors.append(dict(type=type(exc).__name__, message=str(exc)))
                if error is None:
                    raise
            finally:
                elapsed = self.clock()-frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                summary = self.totals.setdefault(name, dict(calls=0, inclusive_s=0., exclusive_s=0.))
                summary['calls'] += 1
                summary['inclusive_s'] += elapsed
                summary['exclusive_s'] += max(0., elapsed-frame[1])

    def report(self):
        return dict(phases=copy.deepcopy(self.totals), sync_errors=list(self.sync_errors),
            scope='nested diagnostic phases; inclusive intervals overlap; CUDA synchronization overhead included; not a throughput benchmark')


def _validate_records(records, *, training_only=False):
    records = list(records)
    if not records:
        raise ValueError('at least one admitted diagnostic query is required')
    ids = set()
    for record in records:
        rid = record.get('record_id')
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError('diagnostic record IDs must be distinct and nonempty')
        ids.add(rid)
        if record.get('loader_eligible') is not True or 'case' not in record:
            raise ValueError('diagnostics require admitted loaded queries')
        if record.get('role') not in ('train', 'development'):
            raise ValueError('only declared train/development queries may enter diagnostics')
        if training_only and (record.get('role') != 'train'
                              or record['case']['meta'].get('split') not in ('train', 'synthetic')):
            raise ValueError('gradient diagnostics require admitted training queries; held-out rows are forbidden')
    return records


def _validate_grid(model, grid):
    if (type(grid) is not int or grid < 1 or grid & (grid-1)
            or grid > model.config.leaf_grid or model.config.native_material_resolution % grid):
        raise ValueError('coarse grid must be a power of two dividing the material side and not exceed leaf_grid')


@contextmanager
def _diagnostic_session(model):
    """Retain scientific/model state; invalidate temporary learned graphs.

    We never load a copied state_dict or write original parameters, so their
    version counters remain unchanged. autograd.grad does not accumulate .grad.
    The response-zero control swaps a separate module and restores the original.
    Learned caches are discarded at both boundaries, not restored as live graphs.
    """
    config, response = model.config, model.response
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    # The actual encoder has no running-stat updates, but defend against future
    # modules that do: only copy buffers back if they genuinely changed.
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    counts = copy.deepcopy(getattr(model, '_counts', None))
    model.invalidate_caches(force=True)
    try:
        yield
    finally:
        model.response = response
        model.config = config
        for module, training in modes:
            module.training = training
        with torch.no_grad():
            for buffer, original in buffers:
                if not torch.equal(buffer, original):
                    buffer.copy_(original)
        model.invalidate_caches(force=True)
        if counts is not None:
            model._counts = counts
        restore_rng(rng)


def _zero_response(response):
    if not isinstance(response, HENMOBoundaryResponse):
        raise ValueError('same-path zero-correction control requires HENMOBoundaryResponse')
    result = copy.deepcopy(response)
    with torch.no_grad():
        result.output.weight.zero_()
        result.output.bias.zero_()
    return result


def _raw_payload(prediction, record):
    case = record['case']
    qualified = bool(case['meta']['derivative_valid'])
    return dict(
        moment_order=MOMENT_ORDER,
        predicted_moments=_json_value(prediction.moments),
        predicted_jacobian=None if prediction.jacobian is None else _json_value(prediction.jacobian),
        native_moments=_json_value(case['native_moments']),
        native_jacobian=_json_value(case['native_jacobian']) if qualified else None,
        parameter_scales=_json_value(case['parameter_scales']) if qualified else None,
        native_direction=_json_value(case.get('direction')) if qualified else None,
        qualification=dict(derivative_label_qualified=qualified,
            derivative_dimension=case['meta'].get('derivative_dimension') if qualified else None,
            derivative_reason=case['meta'].get('derivative_reason'),
            moment_weight=float(record.get('moment_weight', 0.)),
            jacobian_weight=float(record.get('jacobian_weight', 0.))),
        diagnostics=_json_value(prediction.diagnostics),
    )


def _base_row(record):
    meta = record['case']['meta']
    return dict(record_id=record['record_id'], source_group=meta['source_group'],
        geometry_id=meta['geometry_id'], role=record['role'],
        derivative_label_qualified=bool(meta['derivative_valid']),
        status='FAILED', moment=None, jacobian=None)


def _model_prediction(model, record, *, inference, coarse_grid, chunk_size, diagnostic_nodes=0):
    case, meta = record['case'], record['case']['meta']
    trace = _DiagnosticStageTrace(lambda: _sync(model)) if diagnostic_nodes else None
    _sync(model)
    before = time.perf_counter()
    with trace.stage('geometry_encode') if trace is not None else nullcontext():
        cache = model.encode_geometry(case['blocks'], case['fixed_moments'],
                                      density=meta['density'], inference=inference)
    _sync(model)
    encode_s = time.perf_counter()-before
    kwargs = dict(with_jacobian=bool(meta['derivative_valid']), chunk_size=chunk_size,
                  coarse_grid=coarse_grid)
    if meta['derivative_valid']:
        kwargs['parameter_scales'] = case['parameter_scales']
    if 'bandwidth' in record:
        kwargs['bandwidth'] = record['bandwidth']
    if diagnostic_nodes and 'diagnostic_nodes' in inspect.signature(model.evaluate_operand).parameters:
        kwargs['diagnostic_nodes'] = diagnostic_nodes
    if trace is not None and 'trace' in inspect.signature(model.evaluate_operand).parameters:
        kwargs['trace'] = trace
    before = time.perf_counter()
    prediction = model.evaluate_operand(cache, case['operand'], **kwargs)
    # Materialize output on CPU within the recorded query interval. Keep the
    # original prediction graph for gradient diagnostics.
    with trace.stage('output_cpu_transfer') if trace is not None else nullcontext():
        prediction.moments.detach().cpu()
        if prediction.jacobian is not None:
            prediction.jacobian.detach().cpu()
    _sync(model)
    timing = dict(geometry_encode_s=encode_s, query_s=time.perf_counter()-before)
    if trace is not None:
        timing['stage_profile'] = trace.report()
    return prediction, timing


def run_path_diagnostics(model, records, *, coarse_grids=(32, 64), chunk_size=65536,
                         include_zero_correction=True, diagnostic_nodes=128):
    """Evaluate fixed checkpoints through labelled paths without any update.

    Call once per already-verified checkpoint. Rows remain paired by record ID.
    Caller-side selection must preserve held-out identities and source binding.
    The output includes failures rather than dropping hard queries from metrics.
    """
    records = _validate_records(records)
    coarse_grids = tuple(coarse_grids)
    if not coarse_grids or len(set(coarse_grids)) != len(coarse_grids):
        raise ValueError('distinct nonempty coarse grids required')
    for grid in coarse_grids:
        _validate_grid(model, grid)
    if type(chunk_size) is not int or chunk_size < 1 or type(diagnostic_nodes) is not int or diagnostic_nodes < 0:
        raise ValueError('positive chunk size and nonnegative diagnostic node budget required')
    if type(include_zero_correction) is not bool:
        raise ValueError('include_zero_correction must be Boolean')
    source_config = model.config
    routes = [(f'coarse{grid}', 'coarse', grid, False) for grid in coarse_grids]
    if include_zero_correction:
        routes += [(f'coarse{grid}_zero_correction', 'coarse', grid, True) for grid in coarse_grids]
    routes += [(f'adaptive{grid}', 'adaptive', grid, False) for grid in coarse_grids]
    routes += [('deterministic_gl2', 'gl2', None, False),
               ('deterministic_gl3', 'gl3', None, False)]
    results = {}
    with _diagnostic_session(model):
        original_response = model.response
        model.eval()
        for name, execution, grid, zero_correction in routes:
            model.response = original_response
            model.config = source_config
            model.invalidate_caches(force=True)
            model.eval()
            setup_error = None
            if execution in ('coarse', 'adaptive'):
                model.config = replace(source_config, execution=execution, coarse_grid=grid)
                if zero_correction:
                    try:
                        model.response = _zero_response(original_response).eval()
                    except Exception as exc:
                        setup_error = exc
            rows = []
            current_source = current_tree = None
            for record in records:
                row = _base_row(record)
                row.update(route=name, execution_override=execution, coarse_grid_override=grid,
                           learning_correction_disabled=zero_correction,
                           adaptive_starts_at_leaf=execution == 'adaptive' and grid == source_config.leaf_grid,
                           execution_device=str(next(model.parameters()).device) if execution in ('coarse', 'adaptive') else 'cpu',
                           detailed_diagnostic_node_budget=diagnostic_nodes if execution in ('coarse', 'adaptive') else 0)
                prediction = None
                started = time.perf_counter()
                try:
                    if setup_error is not None:
                        raise setup_error
                    with torch.no_grad():
                        if execution in ('coarse', 'adaptive'):
                            prediction, timing = _model_prediction(model, record, inference=True,
                                coarse_grid=grid, chunk_size=chunk_size, diagnostic_nodes=diagnostic_nodes)
                        else:
                            case, meta = record['case'], record['case']['meta']
                            before = time.perf_counter()
                            if current_source is not case['blocks']:
                                current_tree = make_hierarchy(case['blocks'], source_config)
                                current_source = case['blocks']
                            else:
                                current_tree.assert_source_unchanged()
                            build_s = time.perf_counter()-before
                            before = time.perf_counter()
                            operand = case['operand']
                            surface = prepare_surface(operand.vertices, operand.faces,
                                                      execution_context=model.surface_execution)
                            bandwidth = record.get('bandwidth', current_tree.pitch*source_config.bandwidth_voxels)
                            prediction = evaluate_deterministic(current_tree, surface, bandwidth,
                                operand.operation, case['fixed_moments'],
                                operand.vertex_velocity if meta['derivative_valid'] else None,
                                full_cell_rule=execution, chunk_size=chunk_size)
                            timing = dict(hierarchy_build_s=build_s, geometry_encode_s=0.,
                                          query_s=time.perf_counter()-before)
                    row.update(_raw_payload(prediction, record))
                    row.update(prediction_metrics(prediction, record))
                    row['solver_status'] = prediction.diagnostics.get('solver_status')
                    row['timing'] = timing
                except Exception as exc:
                    # Retain the raw output even when validation/metric building
                    # rejects it. Failure rows still make their arm incomplete.
                    if prediction is not None:
                        row.update(_raw_payload(prediction, record))
                    row['status'] = 'FAILED'
                    row['error'] = dict(type=type(exc).__name__, message=str(exc))
                row['observed_wall_s'] = time.perf_counter()-started
                rows.append(row)
            results[name] = summarize_prediction_metrics(rows)
            results[name]['execution_override'] = dict(execution=execution, coarse_grid=grid,
                learning_correction_disabled=zero_correction,
                adaptive_starts_at_leaf=execution == 'adaptive' and grid == source_config.leaf_grid)
    return dict(schema='henmo.path-diagnostics.v1', config=asdict(source_config),
        record_ids=[r['record_id'] for r in records], routes=results,
        actual_parameter_updates=0, moment_order=MOMENT_ORDER,
        deterministic_note='GL2/GL3 are same-material smooth finite-operand cubature, not hard mesh truth; zero correction retains coarse nodes and support',
        timing_scope='single observed diagnostic query including finite surface preparation and output CPU transfer; encoding/build separate; uncontrolled warmup/cache order; not end-to-end acceleration evidence',
        state_restoration='original response/config/module modes/RNG retained; no optimizer or original parameter writes; temporary learned caches invalidated',
        physics_promotion=False, acceleration_proven=False)


def gradient_alignment(named_parameters, moment_gradients, jacobian_gradients, *, jacobian_loss_weight=1.):
    """Real Euclidean inner products, including complex Fourier parameters.

    Missing gradients count as zero in the joint vector and are explicitly
    counted. Zero vectors produce cosine=None, not fabricated agreement.
    """
    if (not math.isfinite(jacobian_loss_weight) or jacobian_loss_weight < 0
            or isinstance(jacobian_loss_weight, bool)):
        raise ValueError('finite nonnegative Jacobian loss weight required')
    named_parameters = list(named_parameters)
    if len(named_parameters) != len(moment_gradients) or len(named_parameters) != len(jacobian_gradients):
        raise ValueError('gradient arrays must align with named parameters')
    rows = {}
    for label in ('total', *MODULE_GROUPS):
        selected = [(i, name, parameter) for i, (name, parameter) in enumerate(named_parameters)
                    if label == 'total' or name.split('.', 1)[0] == label]
        if not selected:
            rows[label] = dict(status='NO_TRAINABLE_PARAMETERS', parameter_tensors=0,
                               moment_gradient_norm=0., jacobian_gradient_norm=0., cosine=None)
            continue
        mm, jj, dot = [], [], []
        m_available = j_available = 0
        for i, _, parameter in selected:
            gm, gj = moment_gradients[i], jacobian_gradients[i]
            for gradient in (gm, gj):
                if gradient is not None and (gradient.shape != parameter.shape or not bool(torch.isfinite(gradient).all())):
                    raise ValueError('gradient shape mismatch or nonfinite gradient')
            if gm is not None:
                m_available += 1
                gm64 = gm.detach().to(torch.complex128 if gm.is_complex() else torch.float64)
                mm.append(float(gm64.abs().square().sum().cpu()))
            if gj is not None:
                j_available += 1
                gj64 = gj.detach().to(torch.complex128 if gj.is_complex() else torch.float64)
                jj.append(float(gj64.abs().square().sum().cpu()))
            if gm is not None and gj is not None:
                dot.append(float((gm64.conj()*gj64).real.sum().cpu()))
        m2, j2, inner = math.fsum(mm), math.fsum(jj), math.fsum(dot)
        mn, jn = math.sqrt(m2), math.sqrt(j2)
        cosine = max(-1., min(1., inner/(mn*jn))) if mn and jn else None
        lam = float(jacobian_loss_weight)
        rows[label] = dict(status='OBSERVED' if cosine is not None else 'ZERO_GRADIENT_DIRECTION',
            parameter_tensors=len(selected), moment_gradient_tensors=m_available,
            jacobian_gradient_tensors=j_available, moment_gradient_norm=mn,
            jacobian_gradient_norm=jn, weighted_jacobian_gradient_norm=lam*jn,
            real_inner_product=inner, cosine=cosine,
            combined_gradient_norm=math.sqrt(max(0., m2+lam*lam*j2+2*lam*inner)),
            norm_balance_ratio=mn/jn if jn else None)
    return rows


def _gradients(loss, parameters, *, retain_graph=False):
    if loss is None or not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def run_gradient_diagnostics(model, records, *, coarse_grid, chunk_size=65536,
                             jacobian_loss_weight=1., diagnostic_nodes=0):
    """Separate M/J gradients at the same fixed train query, without updating.

    Uses the production physical_loss masks, scales and denominators. J is
    measured unweighted, then its requested coefficient is applied to the
    reported vector. No weight calibration is automatically adopted.
    """
    records = _validate_records(records, training_only=True)
    _validate_grid(model, coarse_grid)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive chunk size required')
    if type(diagnostic_nodes) is not int or diagnostic_nodes < 0:
        raise ValueError('nonnegative diagnostic node budget required')
    if isinstance(jacobian_loss_weight, bool) or not math.isfinite(jacobian_loss_weight) or jacobian_loss_weight < 0:
        raise ValueError('finite nonnegative Jacobian loss weight required')
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not named or model.config.response_path == 'deterministic':
        raise ValueError('gradient diagnostics require a trainable neural response')
    source_config = model.config
    rows = []
    with _diagnostic_session(model):
        model.config = replace(source_config, training_execution='coarse', training_coarse_grids=(coarse_grid,))
        model.train()
        for record in records:
            model.invalidate_caches(force=True)
            row = _base_row(record)
            row['coarse_grid_override'] = coarse_grid
            before = time.perf_counter()
            prediction = m_loss = j_loss = gm = gj = None
            try:
                with torch.enable_grad():
                    prediction, timing = _model_prediction(model, record, inference=False,
                        coarse_grid=coarse_grid, chunk_size=chunk_size, diagnostic_nodes=diagnostic_nodes)
                    row.update(_raw_payload(prediction, record))
                    row.update(prediction_metrics(prediction, record))
                    m_loss = j_loss = None
                    m_info = j_info = None
                    if record['moment_weight'] > 0:
                        m_loss, m_info = physical_loss([prediction], [record], jacobian_loss_weight=0.)
                    valid_j = (record['case']['meta']['derivative_valid']
                               and prediction.jacobian is not None and record['jacobian_weight'] > 0)
                    if valid_j:
                        j_record = dict(record, moment_weight=0.)
                        j_loss, j_info = physical_loss([prediction], [j_record], jacobian_loss_weight=1.)
                    parameters = [value for _, value in named]
                    gm = _gradients(m_loss, parameters, retain_graph=j_loss is not None and j_loss.requires_grad)
                    gj = _gradients(j_loss, parameters)
                    row['gradient_alignment'] = gradient_alignment(named, gm, gj,
                                                    jacobian_loss_weight=jacobian_loss_weight)
                    row['losses'] = dict(moment=None if m_loss is None else float(m_loss.detach()),
                        jacobian_unweighted=None if j_loss is None else float(j_loss.detach()),
                        jacobian_coefficient=jacobian_loss_weight,
                        moment_membership=m_info, jacobian_membership=j_info)
                    row['jacobian_gradient_status'] = 'QUALIFIED' if valid_j else (
                        'LABEL_UNQUALIFIED' if not record['case']['meta']['derivative_valid'] else
                        'RESPONSE_J_UNAVAILABLE' if prediction.jacobian is None else 'ZERO_EXAMPLE_WEIGHT')
                    row['timing'] = timing
            except Exception as exc:
                row['status'] = 'FAILED'
                row['error'] = dict(type=type(exc).__name__, message=str(exc))
            finally:
                # Do not retain the previous query's encoder/J graph while the
                # next forward allocates its own graph. This also covers a
                # failed loss or derivative evaluation halfway through a row.
                prediction = m_loss = j_loss = gm = gj = None
                model.invalidate_caches(force=True)
            _sync(model)
            row['observed_wall_s'] = time.perf_counter()-before
            rows.append(row)
    requested_j = sum(bool(record['case']['meta']['derivative_valid']) and record['jacobian_weight'] > 0
                      for record in records)
    observed_j = sum(row['status'] == 'OK' and row.get('jacobian_gradient_status') == 'QUALIFIED'
                     for row in rows)
    nonzero_joint = sum(row['status'] == 'OK'
                        and row['gradient_alignment']['total']['moment_gradient_norm'] > 0
                        and row['gradient_alignment']['total']['jacobian_gradient_norm'] > 0
                        for row in rows)
    return dict(schema='henmo.gradient-diagnostics.v1', config=asdict(source_config),
        execution_override=dict(training_execution='coarse', coarse_grid=coarse_grid),
        record_ids=[r['record_id'] for r in records], rows=rows,
        complete=all(row['status'] == 'OK' for row in rows), actual_parameter_updates=0,
        qualified_derivative_objectives=requested_j, observed_derivative_objectives=observed_j,
        derivative_objective_complete=(observed_j == requested_j) if requested_j else None,
        nonzero_joint_gradient_queries=nonzero_joint,
        jacobian_loss_weight=jacobian_loss_weight,
        inner_product='real part of Hermitian inner product for complex Fourier parameters',
        norm_balance_ratio_note='descriptive same-query gradient norm ratio; not an automatically recommended loss weight',
        timing_scope='diagnostic forward, two gradient computations and auditing; not training or inference benchmark',
        state_restoration='original parameters/gradients/config/module modes/RNG retained; temporary learned caches invalidated',
        physics_promotion=False)
