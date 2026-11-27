"""Finite-operand boundary response with an explicit, trainable tangent chain.

Material points and positive weights are fixed with respect to the cut.  One
bounded retention field supplies all ten raw moments.  The optional small
cross-attention reads descriptors of the *same* finite triangulated operand;
its key, value, softmax, and response derivatives are propagated together.
There is no independently learned Jacobian and no input Hessian in inference.

The compact C1 profile is a surrogate of hard cutting, not an assertion that
every physical contact event is C1.  Its outer support must be included in the
hierarchy's conservative full/empty test.  Surface/closest-feature validity and
the native coefficient chart remain the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _tensor(value, *, dtype, device):
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=dtype, device=device)


def _finite(value, name):
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(name + ' must be finite')


def _positive_scale(value, name):
    if isinstance(value, Tensor) and value.requires_grad:
        raise ValueError(name + ' must be cut-independent')
    if isinstance(value, bool):
        raise ValueError(name + ' must be positive')
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + ' must be positive and finite')
    return number


def _basis(points):
    x, y, z = points.unbind(dim=1)
    return torch.stack((torch.ones_like(x), x, y, z, x*x, y*y, z*z,
                        y*z, x*z, x*y), dim=1)


@dataclass(frozen=True)
class CutTokens:
    """All facets contribute; groups are fixed by face index, never geometry rank.

    Seven features per group: area-weighted centroid (3), oriented area vector
    (3), and scalar area (1), in a fixed material coordinate frame.  Cap faces
    are included exactly like other faces.  This compact conditioning is not a
    lossless representation of the operand; signed distances still use its
    complete surface.  Changing face topology requires rebuilding descriptors.
    """
    features: Tensor
    motion: Tensor | None
    face_groups: Tensor


def finite_cut_tokens(surface, vertex_velocity=None, *, origin=(0., 0., 0.),
                      length_scale=1., max_tokens=16, device=None,
                      dtype=torch.float64) -> CutTokens:
    """Compute finite-face descriptors and their full first native differential.

    ``vertex_velocity`` is [vertices,3,D], with D the valid raw/restricted chart
    dimension. The frame is fixed per material, not recomputed from the moving
    cut's extrema. A live tensor surface preserves its ordinary AD value graph.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    scale = _positive_scale(length_scale, 'length_scale')
    surface.check_tensor_source()
    live = getattr(surface, 'tensor_vertices', None)
    vertices = _tensor(surface.vertices if live is None else live, dtype=dtype, device=device)
    frame = _tensor(origin, dtype=dtype, device=vertices.device)
    if frame.shape != (3,) or frame.requires_grad:
        raise ValueError('origin must be a fixed three-coordinate material frame')
    _finite(frame, 'origin')
    faces = torch.as_tensor(surface.faces.copy(), dtype=torch.long, device=vertices.device)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('finite operand requires triangle faces')
    coordinates = (vertices-frame)/scale
    tri = coordinates[faces]
    centroid = tri.mean(dim=1)
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    area_vector = .5*torch.linalg.cross(e1, e2, dim=1)
    area = torch.linalg.vector_norm(area_vector, dim=1)
    if not bool((area.detach() > 0).all()):
        raise ValueError('degenerate operand facet has no regular area derivative')
    groups_count = min(max_tokens, len(faces))
    group = torch.arange(len(faces), device=vertices.device)*groups_count//len(faces)

    def aggregate(value):
        return value.new_zeros((groups_count,) + tuple(value.shape[1:])).index_add(0, group, value)

    total_area = aggregate(area)
    first = aggregate(area[:, None]*centroid)
    group_center = first/total_area[:, None]
    features = torch.cat((group_center, aggregate(area_vector), total_area[:, None]), dim=1)
    tangent = None
    if vertex_velocity is not None:
        velocity = _tensor(vertex_velocity, dtype=dtype, device=vertices.device)
        if velocity.ndim != 3 or velocity.shape[:2] != vertices.shape or velocity.shape[2] < 1:
            raise ValueError('vertex_velocity must be [vertices,3,D]')
        _finite(velocity, 'vertex_velocity')
        vt = velocity[faces]/scale
        dc = vt.mean(dim=1)
        de1, de2 = vt[:, 1]-vt[:, 0], vt[:, 2]-vt[:, 0]
        dav = .5*(torch.linalg.cross(de1, e2[:, :, None].expand_as(de1), dim=1)
                   + torch.linalg.cross(e1[:, :, None].expand_as(de2), de2, dim=1))
        da = (area_vector[:, :, None]*dav).sum(dim=1)/area[:, None]
        dtotal = aggregate(da)
        dfirst = aggregate(da[:, None, :]*centroid[:, :, None] + area[:, None, None]*dc)
        dcenter = (dfirst-group_center[:, :, None]*dtotal[:, None, :])/total_area[:, None, None]
        tangent = torch.cat((dcenter, aggregate(dav), dtotal[:, None, :]), dim=1)
    _finite(features, 'cut descriptors')
    if tangent is not None:
        _finite(tangent, 'cut descriptor motion')
    return CutTokens(features, tangent, group)


def _silu_tangent(value, tangent):
    sig = value.sigmoid()
    return F.silu(value), tangent*(sig*(1.+value*(1.-sig)))[:, :, None]


class HENMOBoundaryResponse(nn.Module):
    """Small point/context query with optional one-head finite-cut attention.

    Geometry context is static with respect to cut parameters but can retain a
    trainable encoder graph. Coordinates for the neural head are normalized by
    an explicit fixed material frame; integration always uses physical points.
    Zero initialization reproduces the unlearned compact cubic. After the last
    projection first learns, gradients can reach the conditioning layers.

    ``compact_warp`` can saturate anywhere inside the outer support, unlike the
    more restrictive ``bounded_residual`` ablation. Neither profile asserts
    monotonicity with respect to arbitrary changes of a finite cut. ``profile``
    changes the interpreted function and must be saved in the scientific config.
    """
    semantic_version = 'henmo.boundary-positive-atoms.cut-attention.v1'

    def __init__(self, context_dim=0, hidden_dim=32, cut_feature_dim=7, *,
                 use_cut_attention=True, zero_init=True, profile='compact_warp'):
        super().__init__()
        if (type(context_dim) is not int or context_dim < 0 or type(hidden_dim) is not int
                or hidden_dim < 1 or type(cut_feature_dim) is not int or cut_feature_dim < 1):
            raise ValueError('invalid boundary-response dimensions')
        if type(use_cut_attention) is not bool or type(zero_init) is not bool:
            raise ValueError('attention and zero_init must be Boolean')
        if profile not in ('compact_warp','bounded_residual'):
            raise ValueError('profile must be compact_warp or bounded_residual')
        self.context_dim, self.hidden_dim = context_dim, hidden_dim
        self.cut_feature_dim, self.use_cut_attention = cut_feature_dim, use_cut_attention
        self.profile=profile
        self.geometry_query = nn.Linear(3+context_dim, hidden_dim)
        # A common key bias cancels under softmax, so it is deliberately absent.
        self.cut_key = nn.Linear(cut_feature_dim, hidden_dim, bias=False) if use_cut_attention else None
        self.cut_value = nn.Linear(cut_feature_dim, hidden_dim)
        self.mix = nn.Linear(2*hidden_dim+1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        if zero_init:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _inputs(self, nodes, signed_distance, context, cut_tokens, point_origin, point_scale):
        param = self.geometry_query.weight
        points = _tensor(nodes, dtype=param.dtype, device=param.device)
        distance = _tensor(signed_distance, dtype=param.dtype, device=param.device)
        if points.ndim != 2 or points.shape[1] != 3 or distance.shape != (len(points),):
            raise ValueError('nodes [N,3] and signed_distance [N] required')
        frame = _tensor(point_origin, dtype=param.dtype, device=param.device)
        if frame.shape != (3,) or frame.requires_grad:
            raise ValueError('point_origin must be fixed [3]')
        scale = _positive_scale(point_scale, 'point_scale')
        normalized = (points-frame)/scale
        if context is None:
            if self.context_dim:
                raise ValueError('static geometry context is required')
            features = normalized
        else:
            ctx = _tensor(context, dtype=param.dtype, device=param.device)
            if ctx.shape != (len(points), self.context_dim):
                raise ValueError('context must be [N,context_dim]')
            _finite(ctx, 'context')
            features = torch.cat((normalized, ctx), dim=1)
        if cut_tokens is None:
            raise ValueError('finite-cut descriptors are required for the learned response')
        cut = _tensor(cut_tokens, dtype=param.dtype, device=param.device)
        if cut.ndim != 2 or cut.shape[1] != self.cut_feature_dim or not len(cut):
            raise ValueError('cut_tokens must be [S,cut_feature_dim] with S>0')
        for value, name in ((points, 'nodes'), (distance, 'distance'), (frame, 'point_origin'),
                            (cut, 'cut_tokens')):
            _finite(value, name)
        return features, distance, cut

    def _evaluate(self, nodes, signed_distance, *, bandwidth, context=None, cut_tokens=None,
                  distance_motion=None, cut_motion=None, operation='intersection',
                  point_origin=(0., 0., 0.), point_scale=1.):
        if operation not in ('intersection', 'difference'):
            raise ValueError('operation must be intersection or difference')
        width = _positive_scale(bandwidth, 'bandwidth')
        features, distance, cut = self._inputs(nodes, signed_distance, context, cut_tokens,
                                                point_origin, point_scale)
        s = (distance/width).clamp(-1., 1.)
        interior = (distance.abs() < width).to(distance.dtype)
        ds = dcut = None
        if distance_motion is not None:
            ds = _tensor(distance_motion, dtype=distance.dtype, device=distance.device)
            if ds.ndim != 2 or ds.shape[0] != len(distance) or ds.shape[1] < 1:
                raise ValueError('distance_motion must be [N,D]')
            if cut_motion is None:
                raise ValueError('cut_motion is required: omitting the cut-conditioning chain is invalid')
            dcut = _tensor(cut_motion, dtype=distance.dtype, device=distance.device)
            if dcut.shape != tuple(cut.shape)+(ds.shape[1],):
                raise ValueError('cut_motion must be [S,cut_feature_dim,D]')
            _finite(ds, 'distance_motion'); _finite(dcut, 'cut_motion')
            ds = ds/width*interior[:, None]
        elif cut_motion is not None:
            raise ValueError('cut_motion requires distance_motion')

        q = F.silu(self.geometry_query(features))
        value = self.cut_value(cut)
        dvalue = None if ds is None else torch.einsum('sfd,hf->shd', dcut, self.cut_value.weight)
        if self.use_cut_attention:
            key = self.cut_key(cut)
            attention = torch.softmax(q @ key.T / math.sqrt(self.hidden_dim), dim=1)
            attended = attention @ value
            if ds is not None:
                dkey = torch.einsum('sfd,hf->shd', dcut, self.cut_key.weight)
                dscore = torch.einsum('nh,shd->nsd', q, dkey)/math.sqrt(self.hidden_dim)
                dattention = attention[:, :, None]*(dscore-(attention[:, :, None]*dscore).sum(dim=1, keepdim=True))
                dattended = (torch.einsum('nsd,sh->nhd', dattention, value)
                             + torch.einsum('ns,shd->nhd', attention, dvalue))
        else:
            attended = value.mean(dim=0).expand(len(q), -1)
            if ds is not None:
                dattended = dvalue.mean(dim=0).expand(len(q), -1, -1)
        mixed_input = torch.cat((q, attended, s[:, None]), dim=1)
        hidden = self.mix(mixed_input)
        if ds is None:
            coefficient = torch.tanh(self.output(F.silu(hidden)).squeeze(1))
        else:
            # q is independent of the cut, while its parameter/encoder graph is
            # retained. No model parameter is detached in the tangent formulas.
            dinput = torch.cat((dattended.new_zeros((len(q), self.hidden_dim, ds.shape[1])),
                                dattended, ds[:, None, :]), dim=1)
            dhidden = torch.einsum('nfd,hf->nhd', dinput, self.mix.weight)
            activated, dactivated = _silu_tangent(hidden, dhidden)
            raw = self.output(activated).squeeze(1)
            draw = torch.einsum('nhd,h->nd', dactivated, self.output.weight[0])
            coefficient = raw.tanh()
            dcoefficient = (1.-coefficient.square())[:, None]*draw
        if self.profile=='compact_warp':
            amplitude=2.*coefficient
            warped=(s+(1.-s)*(1.+s)*amplitude).clamp(-1.,1.)
            t=(warped+1.)*.5
            tc=(1.-warped)*.5
            retained=(t.square()*(3.-2.*t) if operation=='intersection'
                      else tc.square()*(3.-2.*tc))
            tangent=None
            if ds is not None:
                du=((1.-2.*s*amplitude)[:,None]*ds
                    + (2.*(1.-s)*(1.+s))[:,None]*dcoefficient)
                # H'(clamp(u)) is zero at both saturation boundaries, so this
                # is also the complete derivative of the clamped profile.
                tangent=.75*((1.-warped)*(1.+warped))[:,None]*du
                if operation=='difference':
                    tangent=-tangent
            return retained,tangent
        t = (s+1.)*.5
        base = t.square()*(3.-2.*t)
        # Evaluate the small complement directly, avoiding 1-r cancellation.
        tc = (1.-s)*.5
        complement = tc.square()*(3.-2.*tc)
        inside = base*(1.+complement*coefficient)
        retained = inside if operation == 'intersection' else complement*(1.-base*coefficient)
        tangent = None
        if ds is not None:
            db = .75*(1.-s)*(1.+s)
            tangent = ((1.+(complement-base)*coefficient)[:, None]*db[:, None]*ds
                       + (base*complement)[:, None]*dcoefficient)
            if operation == 'difference':
                tangent = -tangent
        return retained, tangent

    def forward(self, nodes, signed_distance, *, bandwidth, context=None, cut_tokens=None,
                operation='intersection', point_origin=(0., 0., 0.), point_scale=1.):
        return self._evaluate(nodes, signed_distance, bandwidth=bandwidth, context=context,
                              cut_tokens=cut_tokens, operation=operation, point_origin=point_origin,
                              point_scale=point_scale)[0]

    def retention_and_jacobian(self, nodes, signed_distance, *, distance_motion, cut_motion, **kwargs):
        return self._evaluate(nodes, signed_distance, distance_motion=distance_motion,
                              cut_motion=cut_motion, **kwargs)


@dataclass(frozen=True)
class BoundaryResult:
    moments: Tensor
    jacobian: Tensor | None
    retention: Tensor
    diagnostics: dict


def integrate_boundary_response(nodes, weights, signed_distance, *, bandwidth,
                                model=None, context=None, cut_tokens=None,
                                distance_motion=None, cut_motion=None,
                                operation='intersection', point_origin=(0., 0., 0.),
                                point_scale=1.) -> BoundaryResult:
    """Compact active atoms, then reduce all ten physical moments once in FP64.

    With model=None this is the same compact cubic without learned correction.
    A complete or rejected atom never enters the neural network. The caller
    adds fixed material and disjoint complete-block moments exactly once.
    """
    if operation not in ('intersection', 'difference'):
        raise ValueError('invalid Boolean operation')
    width = _positive_scale(bandwidth, 'bandwidth')
    param = None if model is None else next(model.parameters())
    device = None if param is None else param.device
    xyz = _tensor(nodes, dtype=torch.float64, device=device)
    w = _tensor(weights, dtype=torch.float64, device=xyz.device)
    d = _tensor(signed_distance, dtype=torch.float64, device=xyz.device)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or w.shape != (len(xyz),) or d.shape != w.shape:
        raise ValueError('positive atoms require nodes [N,3], weights/distance [N]')
    for value, name in ((xyz, 'nodes'), (w, 'weights'), (d, 'distance')):
        _finite(value, name)
    if not bool((w > 0).all()):
        raise ValueError('material atom weights must be strictly positive')
    motion = None
    if distance_motion is not None:
        motion = _tensor(distance_motion, dtype=torch.float64, device=xyz.device)
        if motion.ndim != 2 or motion.shape[0] != len(xyz) or motion.shape[1] < 1:
            raise ValueError('distance_motion must be [N,D]')
        _finite(motion, 'distance_motion')
    active = (d.abs() < width)
    ids = torch.nonzero(active, as_tuple=False).flatten()
    retained = (d >= width).to(torch.float64) if operation == 'intersection' else (d <= -width).to(torch.float64)
    tangent = None if motion is None else d.new_zeros(motion.shape)
    if len(ids):
        if model is None:
            sign = 1. if operation == 'intersection' else -1.
            s = sign*d[ids]/width
            t = (s+1.)*.5
            r = t.square()*(3.-2.*t)
            dr = None if motion is None else sign*.75*(1.-s.square())[:, None]*motion[ids]/width
        else:
            ctx = None
            if context is not None:
                all_context = _tensor(context, dtype=param.dtype, device=param.device)
                if all_context.shape != (len(xyz), model.context_dim):
                    raise ValueError('context must match the entire supplied atom set')
                ctx = all_context[ids]
            args = dict(bandwidth=width, context=ctx, cut_tokens=cut_tokens,
                        operation=operation, point_origin=point_origin, point_scale=point_scale)
            if motion is None:
                r, dr = model(xyz[ids], d[ids], **args), None
            else:
                r, dr = model.retention_and_jacobian(xyz[ids], d[ids], distance_motion=motion[ids],
                                                    cut_motion=cut_motion, **args)
        retained = retained.index_copy(0, ids, r.to(torch.float64))
        if tangent is not None:
            tangent = tangent.index_copy(0, ids, dr.to(torch.float64))
    elif model is not None and torch.is_grad_enabled():
        # Training on saturated samples remains a valid zero-gradient update.
        zero = sum((p.sum()*0. for p in model.parameters()), retained.new_zeros(()))
        retained = retained+zero
        tangent = tangent+zero if tangent is not None else None
    weighted = _basis(xyz)*w[:, None]
    if tangent is None:
        moments, jacobian = weighted.T @ retained, None
    else:
        reduced = weighted.T @ torch.cat((retained[:, None], tangent), dim=1)
        moments, jacobian = reduced[:, 0], reduced[:, 1:]
    return BoundaryResult(moments, jacobian, retained,
                          dict(total_nodes=len(xyz), active_nodes=len(ids),
                               attention_elements=0 if not len(ids) or model is None or not model.use_cut_attention
                               else len(ids)*len(cut_tokens),
                               derivative='same_response_explicit_tangent',
                               response_profile='unlearned_cubic' if model is None else model.profile,
                               target='compact_smoothed_material_response',
                               hard_cut_equivalence=False))
