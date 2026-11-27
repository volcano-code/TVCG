"""Float64 closest-point BVH for one moving, oriented native operand.

The pseudonormal sign uses incident faces at an edge and corner angles at a
vertex (Baerentzen and Aanaes, 2005). Its input is an embedded solid boundary
with correct component nesting supplied by the native construction. Closure
and directed edge incidence are checked here; this is not an embedding audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np
import torch

from .v23_surface_runtime import (
    SurfaceExecutionContext, get_surface_execution_context,
    shutdown_default_surface_execution,
)

try:
    from numba import njit
except ImportError:  # Small fixtures can also run in a NumPy/Torch installation.
    def njit(**kwargs):
        return lambda func: func


# Numba embeds the import path in cached kernels. Legacy top-level and full
# research-package imports share this file and must never share incompatible
# compiled entries. The legacy import accelerates in memory.
_DISK_CACHE = __name__.startswith('research.')


def _numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _points(x):
    a = np.asarray(_numpy(x), dtype=np.float64, order='C')
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('expected finite [N,3] Cartesian points')
    return a


def _make_tree(triangles):
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    centers = (low + high) * .5
    bounds, children, ranges, order = [], [], [], []

    def build(indices):
        idx = len(bounds)
        bounds.append([*low[indices].min(axis=0), *high[indices].max(axis=0)])
        children.append([-1, -1])
        ranges.append([0, 0])
        if len(indices) <= 6:
            ranges[idx] = [len(order), len(order) + len(indices)]
            order.extend(indices)
        else:
            subset = centers[indices]
            axis = np.ptp(subset, axis=0).argmax()
            by_axis = indices[np.argsort(subset[:, axis], kind='stable')]
            middle = len(indices) // 2
            children[idx] = [build(by_axis[:middle]), build(by_axis[middle:])]
        return idx

    build(np.arange(len(triangles)))
    return (np.asarray(bounds), np.asarray(children, np.int64), np.asarray(ranges, np.int64),
            np.asarray(order, np.int64))


def _exact_witness(point, triangle):
    """Rational squared point–triangle distance for exceptional floating ties."""
    def sub(x, y):
        return tuple(a-b for a, b in zip(x, y))

    def dot(x, y):
        return sum(a*b for a, b in zip(x, y))

    a, b, c = triangle
    ab, ac, ap = sub(b, a), sub(c, a), sub(point, a)
    uu, uv, vv = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    up, vp = dot(ab, ap), dot(ac, ap)
    determinant = uu*vv - uv*uv
    beta, gamma = (up*vv - vp*uv)/determinant, (vp*uu - up*uv)/determinant
    if beta >= 0 and gamma >= 0 and beta+gamma <= 1:
        bary = (1-beta-gamma, beta, gamma)
        witness = tuple(a[k] + beta*ab[k] + gamma*ac[k] for k in range(3))
        delta = sub(point, witness)
        return dot(delta, delta), bary, witness
    candidates = []
    for i in range(3):
        u, v = triangle[i], triangle[(i+1)%3]
        edge = sub(v, u)
        t = min(1, max(0, dot(sub(point, u), edge)/dot(edge, edge)))
        witness = tuple(u[k] + t*edge[k] for k in range(3))
        bary = [0, 0, 0]
        bary[i], bary[(i+1)%3] = 1-t, t
        delta = sub(point, witness)
        candidates.append((dot(delta, delta), tuple(bary), witness))
    return min(candidates, key=lambda w: w[0])


@njit(cache=_DISK_CACHE, nogil=True)
def _box_dist2(x, y, z, box):
    dx = max(box[0] - x, 0., x - box[3])
    dy = max(box[1] - y, 0., y - box[4])
    dz = max(box[2] - z, 0., z - box[5])
    return dx*dx + dy*dy + dz*dz


@njit(cache=_DISK_CACHE, nogil=True)
def _triangle_witness(x, y, z, triangle, directions):
    # The last two directions are gradients of the barycentric coordinates.
    ax, ay, az = triangle[0]
    dx, dy, dz = x - ax, y - ay, z - az
    u = dx*directions[1, 0] + dy*directions[1, 1] + dz*directions[1, 2]
    v = dx*directions[2, 0] + dy*directions[2, 1] + dz*directions[2, 2]
    if u >= 0. and v >= 0. and u + v <= 1.:
        nx, ny, nz = directions[0]
        h = dx*nx + dy*ny + dz*nz
        return h*h, 1. - u - v, u, v, x-h*nx, y-h*ny, z-h*nz
    # Outside the face projection the closest point is on one of its segments.
    best = math.inf
    ba = bb = bc = cx = cy = cz = 0.
    for i in range(3):
        ax, ay, az = triangle[i]
        bx, by, bz = triangle[(i + 1) % 3]
        ex, ey, ez = bx - ax, by - ay, bz - az
        t = ((x-ax)*ex + (y-ay)*ey + (z-az)*ez) / (ex*ex + ey*ey + ez*ez)
        t = min(1., max(0., t))
        qx, qy, qz = ax + t*ex, ay + t*ey, az + t*ez
        delta2 = (x-qx)**2 + (y-qy)**2 + (z-qz)**2
        if delta2 < best:
            best, cx, cy, cz = delta2, qx, qy, qz
            if i == 0:
                ba, bb, bc = 1.-t, t, 0.
            elif i == 1:
                ba, bb, bc = 0., 1.-t, t
            else:
                ba, bb, bc = t, 0., 1.-t
    return best, ba, bb, bc, cx, cy, cz


@njit(cache=_DISK_CACHE, nogil=True)
def _same_feature(face_a, weights_a, face_b, weights_b):
    # A common projection can be reported by every incident triangle. Cartesian
    # closeness alone cannot identify it across a very thin gap between shells.
    for i in range(3):
        a_has_match = weights_a[i] <= 2e-13
        b_has_match = weights_b[i] <= 2e-13
        for j in range(3):
            a_has_match |= face_a[i] == face_b[j] and weights_b[j] > 2e-13
            b_has_match |= face_b[i] == face_a[j] and weights_a[j] > 2e-13
        if not a_has_match or not b_has_match:
            return False
    return True


@njit(cache=_DISK_CACHE, nogil=True)
def _closest(x, y, z, triangles, directions, faces, bounds, children, ranges, order, tol, stack):
    size = 1
    stack[0] = 0
    best2 = cutoff = math.inf
    fid = -1
    ba = bb = bc = cx = cy = cz = 0.
    ambiguous = False
    while size:
        size -= 1
        node = stack[size]
        if _box_dist2(x, y, z, bounds[node]) > cutoff:
            continue
        left, right = children[node]
        if left >= 0:
            dl = _box_dist2(x, y, z, bounds[left])
            dr = _box_dist2(x, y, z, bounds[right])
            if dl < dr:
                if dr <= cutoff:
                    stack[size] = right
                    size += 1
                if dl <= cutoff:
                    stack[size] = left
                    size += 1
            else:
                if dl <= cutoff:
                    stack[size] = left
                    size += 1
                if dr <= cutoff:
                    stack[size] = right
                    size += 1
            continue
        for k in range(ranges[node, 0], ranges[node, 1]):
            f = order[k]
            dist2, a, b, c, qx, qy, qz = _triangle_witness(x,y,z,triangles[f],directions[f])
            if dist2 > cutoff:
                continue
            dist = math.sqrt(dist2)
            if fid >= 0 and abs(dist - math.sqrt(best2)) <= tol:
                # Multiple triangles sharing the same closest edge/vertex are
                # a single geometric witness. Distinct closest points are a tie.
                if ((qx-cx)**2 + (qy-cy)**2 + (qz-cz)**2 > (8*tol)**2
                        or not _same_feature(faces[f], (a,b,c), faces[fid], (ba,bb,bc))):
                    ambiguous = True
                if dist2 >= best2:
                    continue
            else:
                ambiguous = False
            best2, ba, bb, bc, cx, cy, cz, fid = dist2, a, b, c, qx, qy, qz, f
            cutoff = (dist + tol)**2
    return math.sqrt(best2), fid, ba, bb, bc, cx, cy, cz, ambiguous


@njit(cache=_DISK_CACHE, nogil=True)
def _query_batch(points, triangles, directions, edge_normals, vertex_normals, faces,
                 sharp_edges, sharp_vertices, bounds, children, ranges, order, tol):
    count = len(points)
    signed = np.empty(count)
    face = np.empty(count, dtype=np.int64)
    bary = np.empty((count, 3))
    motion_normal = np.empty((count, 3))
    status = np.zeros(count, dtype=np.uint8)
    stack = np.empty(3 + int(math.ceil(math.log2(len(triangles)))), dtype=np.int64)
    for i in range(count):
        x, y, z = points[i]
        d, f, a, b, c, qx, qy, qz, tied = _closest(
            x,y,z,triangles,directions,faces,bounds,children,ranges,order,tol,stack)
        bary[i, 0], bary[i, 1], bary[i, 2], face[i] = a, b, c, f
        feature = 2e-13
        near = int(a <= feature) + int(b <= feature) + int(c <= feature)
        primitive_sharp = False
        if near >= 2:
            j = 0 if a >= b and a >= c else (1 if b >= c else 2)
            v = faces[f, j]
            normal = vertex_normals[v]
            primitive_sharp = sharp_vertices[v]
        elif near == 1:
            j = 0 if a <= feature else (1 if b <= feature else 2)
            normal = edge_normals[f, j]
            primitive_sharp = sharp_edges[f, j]
        else:
            normal = directions[f, 0]
        dx, dy, dz = x-qx, y-qy, z-qz
        dot = dx*normal[0] + dy*normal[1] + dz*normal[2]
        inside_sign = 1. if dot < 0. else -1.
        signed[i] = inside_sign*d
        if tied or (d <= tol and primitive_sharp):
            status[i] = 1
        if d > tol:
            scale = -inside_sign/d
            motion_normal[i, 0], motion_normal[i, 1], motion_normal[i, 2] = dx*scale, dy*scale, dz*scale
            if abs(dot) <= tol:
                status[i] |= 2  # pseudonormal sign needs a winding cross-check
        else:
            motion_normal[i, :] = directions[f, 0, :]
    return signed, face, bary, motion_normal, status


@dataclass(frozen=True)
class FiniteSurface:
    vertices: np.ndarray
    faces: np.ndarray
    tensor_vertices: torch.Tensor | None
    _data: tuple
    tolerance: float
    execution_context: SurfaceExecutionContext | None = field(default=None, repr=False, compare=False)

    def check_tensor_source(self):
        if self.tensor_vertices is not None and not np.array_equal(_numpy(self.tensor_vertices), self.vertices):
            raise ValueError('stale surface stencil: differentiable vertex positions changed')

    def _query(self, points):
        p = _points(points)
        args = (*self._data, self.tolerance)
        runtime = (self.execution_context if self.execution_context is not None
                   else get_surface_execution_context())
        result = runtime.query(_query_batch, p, *args)
        signed, face, bary, normal, status = result
        for index in np.flatnonzero(status & 2):
            # For an embedded oriented solid the generalized winding is either
            # zero or one off the boundary. This path is rare, not the bulk test.
            r = self._data[0] - p[index]
            a, b, c = r[:, 0], r[:, 1], r[:, 2]
            lengths = np.linalg.norm(r, axis=2)
            denom = (lengths.prod(axis=1) + (a*b).sum(1)*lengths[:, 2]
                     + (b*c).sum(1)*lengths[:, 0] + (c*a).sum(1)*lengths[:, 1])
            winding = np.arctan2((a*np.cross(b, c)).sum(1), denom).sum()/(2*math.pi)
            if min(abs(winding), abs(winding-1.)) > 1e-5:
                raise ValueError('oriented operand has unresolved winding at a quadrature point')
            is_inside = winding > .5
            if (signed[index] > 0) != is_inside:
                signed[index] *= -1
                normal[index] *= -1
            status[index] &= np.uint8(1)
        return signed, face, bary, normal, status

    def signed_distance(self, points):
        """Return finite-operand Euclidean distance, positive inside, in world units."""
        self.check_tensor_source()
        return self._query(points)[0]

    def _refine_active_tie(self, point, approximate_distance):
        # Two distances often differ quadratically near a triangle edge. Never
        # decide regularity by shrinking a floating tolerance. Rank close
        # witnesses exactly for the binary64 positions of this finite operand.
        tri, directions, edge_normals, vertex_normals, _, sharp_edges, sharp_vertices = self._data[:7]
        lo, hi = tri.min(axis=1), tri.max(axis=1)
        delta = np.maximum(np.maximum(lo-point, point-hi), 0.)
        near = np.flatnonzero(np.linalg.norm(delta, axis=1) <= abs(approximate_distance)+8*self.tolerance)
        if not len(near):
            raise ValueError('distance refinement could not locate an operand triangle')
        def as_fraction(xyz):
            return tuple(Fraction(float(value)) for value in xyz)
        exact_vertices = {idx: as_fraction(self.vertices[idx]) for idx in np.unique(self.faces[near])}
        p = as_fraction(point)
        witnesses = []
        for f in near:
            triangle = tuple(exact_vertices[idx] for idx in self.faces[f])
            dist2, weights, nearest = _exact_witness(p, triangle)
            witnesses.append((dist2, f, weights, nearest))
        dist2, f, weights, nearest = min(witnesses, key=lambda w: w[0])
        tied = any(d == dist2 and q != nearest for d, _, _, q in witnesses)
        bary = np.asarray(weights, dtype=np.float64)
        zero = [i for i, w in enumerate(weights) if w == 0]
        if len(zero) == 2:
            vertex = self.faces[f, np.argmax(bary)]
            pseudo, sharp = vertex_normals[vertex], sharp_vertices[vertex]
        elif len(zero) == 1:
            pseudo, sharp = edge_normals[f, zero[0]], sharp_edges[f, zero[0]]
        else:
            pseudo, sharp = directions[f, 0], False
        if dist2 == 0:
            return 0., f, bary, directions[f, 0], tied or sharp
        d = math.sqrt(float(dist2))
        ray = np.array([float(a-b) for a, b in zip(p, nearest)])
        sign = 1. if ray@pseudo < 0. else -1.
        return sign*d, f, bary, -sign*ray/d, tied


def prepare_surface(vertices, faces, *, execution_context=None) -> FiniteSurface:
    """Build one spatial hierarchy for the entire indexed operand, caps included.

    An explicit execution_context is borrowed, never closed by the surface.
    Without one, each query uses the currently active scope (or process default),
    so a cached geometric stencil does not retain a closed earlier solve scope.
    """
    if execution_context is not None and not isinstance(execution_context, SurfaceExecutionContext):
        raise TypeError('execution_context must be a SurfaceExecutionContext')
    v = _points(vertices).copy()
    f = _numpy(faces)
    if f.dtype.kind not in 'iu' or f.ndim != 2 or f.shape[1] != 3 or len(f) < 4 or len(v) < 4:
        raise ValueError('finite closed operand requires indexed [F,3] integer triangles')
    if np.min(f) < 0 or np.max(f) >= len(v):
        raise ValueError('surface triangle index out of range')
    f = np.array(f, dtype=np.int64, order='C', copy=True)
    tri = v[f]
    ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cross = np.cross(ab, ac)
    area = np.linalg.norm(cross, axis=1)
    scale = max(np.abs(v).max(), np.ptp(v, axis=0).max())
    eps = np.finfo(float).eps
    if (not np.isfinite(area).all() or (area <= 32*eps*np.linalg.norm(ab,axis=1)*np.linalg.norm(ac,axis=1)).any()):
        raise ValueError('degenerate or numerically singular operand triangle')
    face_normal = cross / area[:, None]
    normals = np.stack([face_normal, np.cross(ac, face_normal)/area[:, None],
                        np.cross(face_normal, ab)/area[:, None]], axis=1)
    # Each local edge is indexed by the vertex opposite it.
    edges = f[:, [[1, 2], [2, 0], [0, 1]]].reshape(-1, 2)
    sorted_edges = np.sort(edges, axis=1)
    idx = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    if len(idx) % 2:
        raise ValueError('surface has a boundary edge')
    first, second = idx[::2], idx[1::2]
    if (not np.array_equal(edges[first], edges[second, ::-1])
            or np.any(np.all(sorted_edges[first[1:]] == sorted_edges[first[:-1]], axis=1))):
        raise ValueError('surface is open, nonmanifold, or inconsistently oriented')
    # Translation prevents a potentially large common coordinate offset from
    # corrupting signed volume. Component nesting is a native operand premise.
    if np.einsum('ij,ij->', tri[:, 0]-v.mean(0), cross) <= 0.:
        raise ValueError('closed operand requires positive oriented volume')
    edge_normal = np.empty((len(edges), 3))
    combined = face_normal[first//3] + face_normal[second//3]
    lengths = np.linalg.norm(combined, axis=1)
    if (lengths <= 32*eps).any():
        raise ValueError('operand has an oppositely folded edge')
    combined /= lengths[:, None]
    edge_normal[first] = edge_normal[second] = combined
    edge_normal = edge_normal.reshape(-1, 3, 3)
    sharp_edges = np.linalg.norm(edge_normal-face_normal[:, None, :], axis=-1) > 1e-9
    vertex_normal = np.zeros_like(v)
    sharp_vertex = np.zeros(len(v), dtype=bool)
    for j in range(3):
        a = tri[:, (j+1)%3] - tri[:, j]
        b = tri[:, (j+2)%3] - tri[:, j]
        angle = np.arctan2(area, (a*b).sum(1))
        np.add.at(vertex_normal, f[:, j], angle[:, None]*face_normal)
        np.logical_or.at(sharp_vertex, f[:, j], sharp_edges[:, (j+1)%3] | sharp_edges[:, (j+2)%3])
    lengths = np.linalg.norm(vertex_normal, axis=1)
    if (lengths[np.unique(f)] <= 32*eps).any():
        raise ValueError('undefined angle-weighted normal at an operand vertex')
    vertex_normal /= np.where(lengths > 0, lengths, 1.)[:, None]
    arrays = (tri, normals, edge_normal, vertex_normal, f, sharp_edges, sharp_vertex, *_make_tree(tri))
    for array in (v, *arrays):
        array.flags.writeable = False
    tensor = vertices if isinstance(vertices, torch.Tensor) and vertices.requires_grad else None
    return FiniteSurface(v, f, tensor, arrays, 128*eps*scale, execution_context)
