"""Sufficient certificate for the incidence of two embedded triangular boundaries.

For every candidate triangle pair exclude a vertex touching the opposing face
and each mixed edge/edge contact. Separate disjoint feature boxes first.
Remaining predicates need a strict interval sign. For a connected parameter
chart and continuous source construction this preserves the indexed geometric
crossings of the two boundaries, including all caps and all material shells.
Output triangulation inside a physical polygon is a distinct event.
"""
import numpy as np
if __package__:
    from .bounds import Interval, cross, dot, orient
else:
    from bounds import Interval, cross, dot, orient


class DesignTriangles:
    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(vertices,dtype=np.float64)
        self.faces = _triangles(faces,len(self.vertices))
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError('invalid fixed vertices')
        tri = self.vertices[self.faces]
        self.lo,self.hi = tri.min(axis=1),tri.max(axis=1)
        self.global_lo,self.global_hi = self.lo.min(axis=0),self.hi.max(axis=0)
        self.order = self.lo[:,0].argsort(kind='stable')
        self.sorted_x = self.lo[self.order,0]

    def candidates(self, lo, hi):
        if np.any(self.global_lo>hi) or np.any(self.global_hi<lo):
            return np.empty(0,dtype=np.intp)
        ids = self.order[:np.searchsorted(self.sorted_x,hi[0],side='right')]
        mask = np.all(self.hi[ids]>=lo,axis=1) & np.all(self.lo[ids]<=hi,axis=1)
        return ids[mask]


def _triangles(faces, n):
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=n):
        raise ValueError('invalid indexed triangles')
    return faces


def _overlap(a, b, lo, hi):
    return np.all((b>=lo)&(a<=hi),axis=-1)


def _separated(signs):
    return np.all(signs==1,axis=1)|np.all(signs==-1,axis=1)


def _chunk(fixed, moving):
    """Return witness position and reason on the first unresolved contact."""
    ds = dot(cross(moving[1]-moving[0],moving[2]-moving[0]),fixed-moving[0]).sign()
    idx = np.flatnonzero(~_separated(ds))
    if not len(idx):
        return 0,None
    fixed,ds = fixed[idx],ds[idx]
    f_normal = cross(fixed[:,1]-fixed[:,0],fixed[:,2]-fixed[:,0])
    ms = dot(f_normal[:,None], moving-fixed[:,0,None]).sign()
    active = ~_separated(ms)
    idx,fixed,ds,ms = idx[active],fixed[active],ds[active],ms[active]
    if not len(idx):
        return 0,None
    flo,fhi = fixed.lo.min(axis=1),fixed.hi.max(axis=1)
    mlo,mhi = moving.lo.min(axis=0),moving.hi.max(axis=0)
    for i in range(3):
        bad = (ds[:,i]==0) & _overlap(fixed.lo[:,i],fixed.hi[:,i],mlo,mhi)
        if bad.any():
            return len(idx),(int(idx[np.flatnonzero(bad)[0]]),'fixed_vertex_moving_face',i)
        bad = (ms[:,i]==0) & _overlap(moving.lo[i],moving.hi[i],flo,fhi)
        if bad.any():
            return len(idx),(int(idx[np.flatnonzero(bad)[0]]),'moving_vertex_fixed_face',i)
    for i in range(3):
        k=(i+1)%3
        edge_min = np.minimum(fixed.lo[:,i],fixed.lo[:,k])
        edge_max = np.maximum(fixed.hi[:,i],fixed.hi[:,k])
        for j in range(3):
            l=(j+1)%3
            mixed = (ds[:,i]*ds[:,k]<=0) & (ms[:,j]*ms[:,l]<=0)
            mixed &= _overlap(edge_min,edge_max,np.minimum(moving.lo[j],moving.lo[l]),
                               np.maximum(moving.hi[j],moving.hi[l]))
            positions=np.flatnonzero(mixed)
            if len(positions):
                s=orient(fixed[positions,i],fixed[positions,k],moving[j],moving[l]).sign()
                if np.any(s==0):
                    p=positions[np.flatnonzero(s==0)[0]]
                    return len(idx),(int(idx[p]),'mixed_edge_edge',(i,j))
    return len(idx),None


def certify_incidence(design, source, source_faces):
    if not isinstance(source,Interval) or source.lo.ndim != 2 or source.lo.shape[1]!=3:
        raise ValueError('finite operand requires enclosing 3D vertex intervals')
    faces = _triangles(source_faces,len(source.lo))
    counts = dict(fixed_faces=len(design.faces),moving_faces=len(faces),aabb_candidates=0,
                  narrow_phase_face_pairs=0)
    for mi,tri in enumerate(faces):
        moving = source[tri]
        candidates=design.candidates(moving.lo.min(axis=0),moving.hi.max(axis=0))
        counts['aabb_candidates']+=len(candidates)
        for offset in range(0,len(candidates),8192):
            ids = candidates[offset:offset+8192]
            fixed = Interval.point(design.vertices[design.faces[ids]])
            n,unresolved=_chunk(fixed,moving)
            counts['narrow_phase_face_pairs']+=n
            if unresolved is not None:
                i,kind,local=unresolved
                return dict(certified=False,reason=kind,design_face=int(ids[i]),source_face=mi,
                            feature=local,**counts)
    return dict(certified=True,**counts)
