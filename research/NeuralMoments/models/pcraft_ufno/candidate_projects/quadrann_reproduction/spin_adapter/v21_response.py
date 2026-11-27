"""Positive local response to the actual finite native query operand.

For static positive material atoms (x_i,w_i), all ten variable moments are one
sum of w_i r_i phi(x_i). The learned profile only changes r_i inside a band of
specified physical half-width. A nearest feature of the finite closed operand
supplies signed distance and its velocity. For a regular closest point the
envelope theorem gives ds/dq = n_motion dot sum_j bary_j * dV_j/dq. This
includes edge, vertex and cap motion. The local derivative is contracted with
the derivative of the same retention used for M. Both M and J retain the
learning graph. Changing the native construction/topology requires rebuilding
the stencil; native rank and chart eligibility are the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .v21_response_surface import FiniteSurface, prepare_surface, _numpy, _points


MOMENT_ORDER = ('1', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'yz', 'xz', 'xy')


def _size(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('chunk_size must be a positive integer')
    return n


def _weighted_basis(nodes, weights):
    x, y, z = nodes.T
    basis = np.empty((len(nodes), 10), dtype=np.float64)
    basis[:, 0] = weights
    basis[:, 1] = weights*x
    basis[:, 2] = weights*y
    basis[:, 3] = weights*z
    basis[:, 4] = basis[:, 1]*x
    basis[:, 5] = basis[:, 2]*y
    basis[:, 6] = basis[:, 3]*z
    basis[:, 7] = basis[:, 2]*z
    basis[:, 8] = basis[:, 1]*z
    basis[:, 9] = basis[:, 1]*y
    return basis


@dataclass(frozen=True)
class PositiveMeasure:
    nodes: np.ndarray
    weights: np.ndarray
    total_moments: np.ndarray


def prepare_measure(nodes, weights) -> PositiveMeasure:
    """Freeze a query-independent node rule. Weight includes material density once."""
    xyz = _points(nodes).copy()
    w = np.array(_numpy(weights), dtype=np.float64, order='C', copy=True)
    if w.shape != (len(xyz),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('one strictly positive finite weight required per static atom')
    total = np.zeros(10, dtype=np.float64)
    for i in range(0, len(w), 65536):
        total += _weighted_basis(xyz[i:i+65536], w[i:i+65536]).sum(axis=0)
    if not np.isfinite(total).all():
        raise ValueError('static positive rule has nonfinite moments')
    for a in (xyz, w, total):
        a.flags.writeable = False
    return PositiveMeasure(xyz, w, total)


@dataclass(frozen=True)
class ResponseStencil:
    surface: FiniteSurface
    bandwidth: float
    active_indices: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray
    signed_distance: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    motion_normal: np.ndarray
    interior_moments: np.ndarray
    exterior_moments: np.ndarray
    total_moments: np.ndarray
    total_nodes: int
    interior_nodes: int
    ambiguous_nodes: int

    def distance_motion(self, vertex_velocity):
        """Contract supplied native vertex motions into ds/dq at all band atoms."""
        self.surface.check_tensor_source()
        velocity = _checked_motion(self.surface, vertex_velocity)
        if velocity is None:
            raise ValueError('a native vertex motion is required')
        if self.ambiguous_nodes:
            raise ValueError('signed-distance derivative undefined at active closest-feature tie')
        return _contract_motion(self, slice(None), velocity)


def prepare_stencil(measure: PositiveMeasure, surface: FiniteSurface, *, bandwidth, chunk_size=65536):
    """Stream points through the operand BVH. Keep only compact-band witnesses."""
    chunk_size = _size(chunk_size)
    surface.check_tensor_source()
    width = float(bandwidth)
    if not math.isfinite(width) or width <= 0:
        raise ValueError('response half-bandwidth must be positive in physical units')
    low = surface.vertices.min(axis=0) - width - surface.tolerance
    high = surface.vertices.max(axis=0) + width + surface.tolerance
    fields = [[] for _ in range(5)]
    inside_moments = np.zeros(10)
    outside_moments = np.zeros(10)
    interior = ambiguous = 0
    for i in range(0, len(measure.nodes), chunk_size):
        pts = measure.nodes[i:i+chunk_size]
        # Outside the expanded operand box, no angular or surface query is necessary.
        within = np.all((pts >= low) & (pts <= high), axis=1)
        if not within.all():
            rejected = i + np.flatnonzero(~within)
            outside_moments += _weighted_basis(measure.nodes[rejected], measure.weights[rejected]).sum(0)
        potential = np.flatnonzero(within)
        if not len(potential):
            continue
        d, face, bary, normal, status = surface._query(pts[potential])
        for k in np.flatnonzero((status != 0) & (np.abs(d) < width + surface.tolerance)):
            d[k], face[k], bary[k], normal[k], status[k] = surface._refine_active_tie(pts[potential[k]], d[k])
        full = i + potential[d >= width]
        inside_moments += _weighted_basis(measure.nodes[full], measure.weights[full]).sum(axis=0)
        exterior = i + potential[d <= -width]
        outside_moments += _weighted_basis(measure.nodes[exterior], measure.weights[exterior]).sum(axis=0)
        interior += len(full)
        active = np.abs(d) < width
        if active.any():
            fields[0].append(i + potential[active])
            for output, values in zip(fields[1:], (d, face, bary, normal)):
                output.append(values[active])
            ambiguous += np.count_nonzero(status[active])
    def join(parts, shape, dtype):
        return np.concatenate(parts) if parts else np.empty(shape, dtype=dtype)
    ids = join(fields[0], (0,), np.int64)
    return ResponseStencil(surface, width, ids, measure.nodes[ids], measure.weights[ids],
                           join(fields[1], (0,), np.float64), join(fields[2], (0,), np.int64),
                           join(fields[3], (0, 3), np.float64), join(fields[4], (0, 3), np.float64),
                           inside_moments, outside_moments, measure.total_moments, len(measure.nodes), interior, int(ambiguous))


def _base_profile(s):
    """C1 compact cubic in signed distance divided by physical half-width."""
    t = (s + 1.) * .5
    return t*t*(3. - 2.*t)


def _projection_version(layer):
    # A cached W_h h is valid only for this exact first-layer weight. Loading a
    # state_dict, an optimizer update, replacement, or device/dtype conversion
    # invalidates it. The bias is applied per point and is not cached.
    weight = layer.weight
    return (id(weight), weight._version, weight.data_ptr(), tuple(weight.shape),
            weight.dtype, weight.device, weight.requires_grad)


@dataclass(frozen=True)
class ProjectedRetentionContext:
    """A graph-preserving first-layer projection, scoped to one weight version.

    The source token features must be independent of signed cut distance. A
    training projection may be gathered by many chunks in ONE forward/backward
    graph; it cannot be carried across optimizer updates or reused after that
    graph has been consumed. The owning geometry cache enforces the latter
    lifecycle. No parameter or feature is detached here.
    """
    values: torch.Tensor
    owner: int
    version: tuple
    context_dim: int

    @property
    def shape(self):
        return (len(self.values), self.context_dim)

    @property
    def ndim(self):
        return 2

    @property
    def device(self):
        return self.values.device

    def gather(self, indices):
        if isinstance(indices, np.ndarray) and not indices.flags.writeable:
            indices = indices.copy()
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.values.device)
        if indices.ndim != 1:
            raise ValueError('projected context indices must be one-dimensional')
        return ProjectedRetentionContext(self.values.index_select(0, indices), self.owner,
                                         self.version, self.context_dim)


class LocalRetention(nn.Module):
    """A bounded local response conditioned only on static material features.

    Inputs to forward are active-node coordinates, signed distance divided by
    the fixed outer half-width, and optional static encoder features. The legacy
    ``bounded_cubic`` profile retains its parameter layout and interpretation.
    ``monotone_warp`` instead moves and sharpens the cubic's coordinate, allowing
    saturation inside that SAME prepared band. Its two coefficients are static
    with respect to the cut, so the distance derivative below is the complete
    local chain rule. The response is not an exact hard-cut integration rule.

    ``response_type`` belongs in the owning model/checkpoint configuration, not
    the state_dict: the two interpretations must never be silently interchanged.
    ``zero_init`` resets only the final layer and reproduces the unlearned cubic.
    """
    def __init__(self, context_dim=0, hidden_dim=24, *, response_type='bounded_cubic', zero_init=False):
        super().__init__()
        if not isinstance(context_dim, int) or context_dim < 0 or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError('invalid local retention dimensions')
        if response_type not in ('bounded_cubic', 'monotone_warp'):
            raise ValueError('response_type must be bounded_cubic or monotone_warp')
        if type(zero_init) is not bool:
            raise ValueError('zero_init must be a Boolean')
        self.context_dim = context_dim
        self.response_type = response_type
        # Ordinary Python option: no new state_dict entry or model parameter.
        # False retains the independent autograd reference path for comparisons.
        self.use_analytic_slope = True
        self.local = nn.Sequential(nn.Linear(3 + context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        if zero_init:
            nn.init.zeros_(self.local[-1].weight)
        else:
            nn.init.normal_(self.local[-1].weight, std=.015)
        nn.init.zeros_(self.local[-1].bias)

    def project_context(self, context):
        """Compute W_h h once per token without changing parameter layout."""
        first = self.local[0]
        if not isinstance(context, torch.Tensor) or context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError('token context must be a [tokens,context_dim] tensor')
        if context.device != first.weight.device or context.dtype != first.weight.dtype:
            raise ValueError('token context must match response parameter device and dtype')
        return ProjectedRetentionContext(F.linear(context, first.weight[:, 3:], None),
                                         id(self), _projection_version(first), self.context_dim)

    def _coefficients(self, points, context):
        if isinstance(context, ProjectedRetentionContext):
            first = self.local[0]
            if context.owner != id(self) or context.version != _projection_version(first):
                raise ValueError('foreign or stale projected retention context')
            if (context.shape != (len(points), self.context_dim)
                    or context.values.shape != (len(points), first.out_features)):
                raise ValueError('projected context must match active points and hidden width')
            if context.values.device != points.device or context.values.dtype != points.dtype:
                raise ValueError('projected context must match point device and dtype')
            hidden = F.linear(points, first.weight[:, :3], first.bias) + context.values
            return self.local[2](self.local[1](hidden)).tanh().unbind(dim=1)
        if context is None:
            if self.context_dim:
                raise ValueError('static local encoder features required')
            features = points
        else:
            if context.shape != (len(points), self.context_dim):
                raise ValueError('static encoder features must match active point count and context_dim')
            features = torch.cat((points, context.to(points)), dim=1)
        return self.local(features).tanh().unbind(dim=1)

    def _response(self, points, normalized_distance, context, operation, *, slope):
        bias, steepness = self._coefficients(points, context)
        bias, steepness = bias.to(normalized_distance), steepness.to(normalized_distance)
        s = normalized_distance.clamp(-1., 1.)
        if self.response_type == 'monotone_warp':
            # The first output controls sharpness, the second a bounded shift.
            # _coefficients has already applied tanh to both raw outputs.
            a = torch.expm1(math.log(3.)*bias)
            delta = .25*steepness
            y = s + delta*(1.-s)*(1.+s)
            q = y*(1. + a*(1.-y)*(1.+y))
            t = q.clamp(-1., 1.)
            if operation == 'intersection':
                retained, sign = _base_profile(t), 1.
            elif operation == 'difference':
                # Evaluate the small complement directly, never as 1 - inside.
                retained, sign = _base_profile(-t), -1.
            else:
                raise ValueError('operation must be intersection or difference')
            if not slope:
                return retained
            # dy/ds >= 1/2. For a > 1/2 the coordinate q can turn back near an
            # endpoint, but only where |q| >= 1 and P' is zero. Thus the complete
            # saturated response remains monotone and C1 for all allowed a.
            # Keep this expression connected to a/delta/context for J training.
            derivative = (sign*.75*(1.-t)*(1.+t)
                          * (1. + a*(1.-3.*y*y)) * (1.-2.*delta*s))
            return retained, derivative
        p = _base_profile(s)
        complement = _base_profile(-s)
        # |coefficient| <= 1, so 0 <= p + p(1-p)*coefficient <= 1.
        coefficient = .5*(bias + steepness*(2.*p - 1.))
        # The two expressions sum to one. Evaluate the small side directly
        # when a retained mass is many orders below total editable mass.
        if operation == 'intersection':
            retained, sign = p*(1. + complement*coefficient), 1.
        elif operation == 'difference':
            retained, sign = complement*(1. - p*coefficient), -1.
        else:
            raise ValueError('operation must be intersection or difference')
        if not slope:
            return retained
        # This is the derivative of THAT SAME response, not a learned J head.
        # Keep all Torch operations connected to bias, steepness and context so
        # a Jacobian loss still updates the encoder and both local layers.
        # Factoring 1-s*s avoids cancellation near either saturated endpoint.
        base_slope = .75*(1. - s)*(1. + s)
        derivative = sign*base_slope*(1. + (complement-p)*coefficient + p*complement*steepness)
        return retained, derivative

    def forward(self, points, normalized_distance, context=None, *, operation='intersection'):
        return self._response(points, normalized_distance, context, operation, slope=False)

    def retention_and_slope(self, points, normalized_distance, context=None, *, operation='intersection'):
        """Return r and dr/ds for s=d/h with cut-independent point features.

        Saturated/clamped values have zero slope. Both supported profiles retain
        the parameter graph. This closed form is only used for this exact class;
        custom models retain AD fallback.
        """
        return self._response(points, normalized_distance, context, operation, slope=True)


@dataclass(frozen=True)
class ResponseResult:
    moments: torch.Tensor
    jacobian: torch.Tensor | None
    diagnostics: dict


def _checked_motion(surface, vertex_velocity):
    if vertex_velocity is None:
        return None
    a = vertex_velocity
    if not isinstance(a, torch.Tensor) or not a.requires_grad:
        a = np.asarray(_numpy(a), dtype=np.float64)
        finite = np.isfinite(a).all()
    else:
        finite = bool(torch.isfinite(a.detach()).all())
    if a.ndim != 3 or a.shape[:2] != surface.vertices.shape or a.shape[2] < 1 or not finite:
        raise ValueError('native vertex_velocity must be finite [N_vertices,3,D]')
    return a


def _contract_motion(stencil, region, velocity):
    f = stencil.surface.faces[stencil.face_indices[region]]
    n, bary = stencil.motion_normal[region], stencil.barycentric[region]
    # Accumulate the three participating vertices without forming [nodes,3,3,D].
    if isinstance(velocity, torch.Tensor):
        n = _as_tensor(n, dtype=velocity.dtype, device=velocity.device)
        bary = _as_tensor(bary, dtype=velocity.dtype, device=velocity.device)
        ids = torch.as_tensor(f, device=velocity.device)
        return sum(bary[:, k:k+1]*torch.einsum('ni,nid->nd', n, velocity[ids[:, k]]) for k in range(3))
    result = np.zeros((len(f), velocity.shape[-1]))
    for k in range(3):
        result += bary[:, k:k+1]*np.einsum('ni,nid->nd', n, velocity[f[:, k]])
    return result


def _as_tensor(x, *, device, dtype=torch.float64):
    # Static bytes are not mutated by torch. Avoid writable aliases to the frozen measure.
    if isinstance(x, np.ndarray) and not x.flags.writeable:
        x = np.array(x, copy=True)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _evaluate_torch_band(stencil, region, model, context, device, velocity, live_vertices, need_graph, operation,
                         fused_contraction=False):
    analytic = type(model) is LocalRetention and model.use_analytic_slope
    # AD fallback still works inside no_grad/inference_mode. The analytic path
    # does not build a temporary local graph merely to produce inference J.
    local_grad = need_graph or (velocity is not None and not analytic)
    with torch.inference_mode(False), torch.set_grad_enabled(local_grad):
        d = _as_tensor(stencil.signed_distance[region], device=device)
        if live_vertices is not None:
            bary = _as_tensor(stencil.barycentric[region], device=device)
            n = _as_tensor(stencil.motion_normal[region], device=device)
            face = torch.as_tensor(stencil.surface.faces[stencil.face_indices[region]], device=device)
            displacement = (live_vertices - live_vertices.detach()).to(device=device, dtype=torch.float64)
            delta = sum(bary[:, j:j+1]*displacement[face[:, j]] for j in range(3))
            # Exact first local differential at this prepared geometry. Actual
            # geometry finite differences rebuild the closest features instead.
            d = d + (n*delta).sum(1)
        if velocity is not None and not analytic and not d.requires_grad:
            d = d.requires_grad_()
        scale = d/stencil.bandwidth
        slope = None
        if model is None:
            retained = _base_profile(scale if operation == 'intersection' else -scale)
        else:
            param = next(model.parameters())
            points = _as_tensor(stencil.nodes[region], dtype=param.dtype, device=device)
            features = None
            if context is not None:
                if callable(context):
                    features = context(points)
                else:
                    ids = stencil.active_indices[region]
                    if isinstance(context, ProjectedRetentionContext):
                        features = context.gather(ids)
                    else:
                        features = context[_as_tensor(ids, dtype=torch.long, device=context.device)] if isinstance(context, torch.Tensor) else context[ids]
                if not isinstance(features, ProjectedRetentionContext):
                    features = _as_tensor(features, device=device, dtype=param.dtype)
            if analytic and velocity is not None:
                retained, normalized_slope = model.retention_and_slope(points, scale, features, operation=operation)
                slope = normalized_slope/stencil.bandwidth
            else:
                retained = model(points, scale, features, operation=operation)
        basis = _as_tensor(_weighted_basis(stencil.nodes[region], stencil.weights[region]), device=device)
        moments = None if velocity is not None and fused_contraction else basis.T @ retained
        jac = None
        if velocity is not None:
            if slope is None:
                # The reference/custom model path uses one VJP, not ten
                # separate Moment reverse passes.
                slope, = torch.autograd.grad(retained.sum(), d, create_graph=need_graph, retain_graph=need_graph)
            ds = _as_tensor(_contract_motion(stencil, region, velocity), device=device)
            differentiated = slope[:, None]*ds
            if fused_contraction:
                # One shared FP64 reduction for the value and its own local
                # differential. No independent J head or detached training path.
                combined = basis.T @ torch.cat((retained[:, None], differentiated), dim=1)
                moments, jac = combined[:, 0], combined[:, 1:]
            else:
                jac = basis.T @ differentiated
        if not need_graph:
            moments = moments.detach()
            jac = jac.detach() if jac is not None else None
    return moments, jac


def evaluate_stencil(stencil: ResponseStencil, *, operation='intersection', fixed_moments=None,
                     vertex_velocity=None, model=None, context=None, chunk_size=65536,
                     fused_contraction=False) -> ResponseResult:
    """Evaluate both operations from the same inside retention and motion.

    An encoder callback receives active static node positions on the model's
    device/dtype and returns [K,context_dim] or ProjectedRetentionContext.
    Alternatively pass features for all_input_nodes. A static feature may carry a training
    graph. An ambiguous active distance leaves M available and J undefined.
    The active mass bounds smoothing's absolute mass discrepancy from a hard
    indicator on these same atoms. Error from replacing continuous material by
    the static quadrature requires an independent refinement or reference.
    """
    if operation not in ('intersection', 'difference'):
        raise ValueError('operation must be intersection or difference')
    if type(fused_contraction) is not bool:
        raise ValueError('fused_contraction must be a Boolean')
    stencil.surface.check_tensor_source()
    chunk_size = _size(chunk_size)
    if model is None and context is not None:
        raise ValueError('static encoder features require a learned retention model')
    if context is not None and not callable(context) and (context.ndim != 2 or context.shape[0] != stencil.total_nodes):
        raise ValueError('context array must correspond to the supplied static nodes')
    velocity = _checked_motion(stencil.surface, vertex_velocity)
    live = stencil.surface.tensor_vertices
    if model is not None:
        device = next(model.parameters()).device
    elif live is not None:
        device = live.device
    elif isinstance(vertex_velocity, torch.Tensor):
        device = vertex_velocity.device
    elif isinstance(fixed_moments, torch.Tensor):
        device = fixed_moments.device
    else:
        device = torch.device('cpu')
    fixed = _as_tensor(np.zeros(10) if fixed_moments is None else fixed_moments, device=device)
    if fixed.shape != (10,) or not bool(torch.isfinite(fixed.detach()).all()):
        raise ValueError('fixed_moments must contain ten finite undoubled moments')
    differentiable_motion = isinstance(velocity, torch.Tensor) and velocity.requires_grad
    status = 'MOTION_NOT_SUPPLIED' if velocity is None else 'DEFINED_FOR_SUPPLIED_NATIVE_MOTION'
    if stencil.ambiguous_nodes:
        status = 'NONREGULAR_ACTIVE_DISTANCE'
        velocity = None
        live = None
    need_graph = torch.is_grad_enabled()
    use_numpy = model is None and live is None and not differentiable_motion
    sign = 1. if operation == 'intersection' else -1.
    saturated = stencil.interior_moments if operation == 'intersection' else stencil.exterior_moments
    count = stencil.interior_nodes if operation == 'intersection' else stencil.total_nodes-stencil.interior_nodes-len(stencil.nodes)
    if count == stencil.total_nodes:
        saturated = stencil.total_moments
    variable = _as_tensor(saturated, device=device)
    jac = None if velocity is None else torch.zeros((10, velocity.shape[2]), dtype=torch.float64, device=device)
    if not len(stencil.nodes):
        # Keep ordinary training/backward valid on a locally constant map.
        if live is not None:
            variable = variable + live.sum().to(device)*0.
        if model is not None and need_graph:
            zero = next(model.parameters()).reshape(-1)[0]*0.
            variable = variable + zero
            jac = jac + zero if jac is not None else None
    for i in range(0, len(stencil.nodes), chunk_size):
        region = slice(i, i+chunk_size)
        if use_numpy:
            s = stencil.signed_distance[region]/stencil.bandwidth
            basis = _weighted_basis(stencil.nodes[region], stencil.weights[region])
            mm = np.einsum('ki,k->i', basis, _base_profile(sign*s))
            variable = variable + _as_tensor(mm, device=device)
            if velocity is not None:
                # Analytic derivative of the same cubic profile; signed distance
                # is positive inside, so outward normal motion grows the mass.
                slope = sign*.75*(1. - s*s)/stencil.bandwidth
                ds = _contract_motion(stencil, region, velocity)
                jj = np.einsum('ki,kj->ij', basis, slope[:, None]*ds, optimize=False)
                jac = jac + _as_tensor(jj, device=device)
        else:
            mm, jj = _evaluate_torch_band(stencil, region, model, context, device, velocity, live, need_graph, operation,
                                         fused_contraction=fused_contraction)
            variable = variable + mm
            if jj is not None:
                jac = jac + jj
    diagnostics = dict(bandwidth=stencil.bandwidth, total_nodes=stencil.total_nodes,
                       active_nodes=len(stencil.nodes), inside_saturated_nodes=stencil.interior_nodes,
                       outside_saturated_nodes=stencil.total_nodes-len(stencil.nodes)-stencil.interior_nodes,
                       ambiguous_nodes=stencil.ambiguous_nodes, jacobian_status=status,
                       active_mass=float(stencil.weights.sum()), smoothing='compact_cubic_signed_euclidean',
                       learned_shift_and_steepness=model is not None, operation=operation, moment_order=MOMENT_ORDER)
    diagnostics['fused_contraction'] = bool(fused_contraction and not use_numpy and velocity is not None)
    diagnostics['response_type'] = 'unlearned_cubic' if model is None else getattr(model, 'response_type', 'custom')
    if diagnostics['response_type'] == 'monotone_warp':
        diagnostics['smoothing'] = 'compact_monotone_warp_signed_euclidean'
    return ResponseResult(fixed + variable, jac, diagnostics)


def evaluate_response(measure: PositiveMeasure, surface: FiniteSurface, *, bandwidth,
                      operation='intersection', fixed_moments=None, vertex_velocity=None,
                      model=None, context=None, chunk_size=65536, fused_contraction=False) -> ResponseResult:
    stencil = prepare_stencil(measure, surface, bandwidth=bandwidth, chunk_size=chunk_size)
    return evaluate_stencil(stencil, operation=operation, fixed_moments=fixed_moments,
                            vertex_velocity=vertex_velocity, model=model, context=context, chunk_size=chunk_size,
                            fused_contraction=fused_contraction)
