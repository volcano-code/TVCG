"""Independent, bounded float64 cubature and six-face topology. No oracle imports."""
from collections import deque
from time import monotonic
import numpy as np

class Pending(RuntimeError): pass

def check_budget(side):
    # Includes dense masks, label queue, coordinates and moment intermediates.
    estimate=64*side**3+64*1024**2
    if estimate>1024**3 or side>64:
        raise Pending(f'bounded adapter limit: side={side}, estimated workspace={estimate} bytes; no full-grid work started')
    return estimate

def canonical(rows, side):
    if not isinstance(rows,list):raise ValueError('RLE must be a JSON array')
    prev=None
    for row in rows:
        if not isinstance(row,list) or len(row)!=4 or any(type(v) is not int for v in row):raise ValueError('integer RLE required; no coercion')
        x,y,a,b=row
        if not (0<=x<side and 0<=y<side and 0<=a<b<=side):raise ValueError('out-of-range or empty RLE')
        if prev is not None:
            if tuple(row[:3])<=tuple(prev[:3]):raise ValueError('duplicate or unordered RLE')
            if row[:2]==prev[:2] and a<=prev[3]:raise ValueError('overlapping or unmerged adjacent RLE')
        prev=row
    return rows

def bitmap(rows,side):
    canonical(rows,side);check_budget(side)
    out=np.zeros((side,)*3,dtype=bool)
    for x,y,a,b in rows:out[x,y,a:b]=True
    return out

def encode(mask):
    out=[]
    for x,y in np.ndindex(mask.shape[:2]):
        d=np.diff(np.r_[False,mask[x,y],False].astype(np.int8))
        out.extend([x,y,int(a),int(b)] for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
    return out

def prism_moments(rows,origin,h,rho):
    # Product antiderivatives, independent from per-cell central moments below.
    result=np.zeros(10,np.float64)
    for x,y,a,b in rows:
        lo=np.asarray(origin,np.float64)+h*np.array([x,y,a],np.float64)
        hi=np.asarray(origin,np.float64)+h*np.array([x+1,y+1,b],np.float64)
        p0=hi-lo;p1=(hi**2-lo**2)/2;p2=(hi**3-lo**3)/3
        v=np.prod(p0);c=p1/p0;s=p2/p0
        result+=rho*v*np.array([1,*c,*s,c[1]*c[2],c[0]*c[2],c[0]*c[1]],np.float64)
    return result

def cell_moments(mask,origin,h,rho):
    # Slabbed, direct retained-cell computation (no deleted subtraction).
    result=np.zeros(10,np.float64)
    for i in range(mask.shape[0]):
        yz=np.argwhere(mask[i]); xyz=np.column_stack((np.full(len(yz),i),yz))
        c=np.asarray(origin,np.float64)+(xyz+.5)*h
        if not len(c):continue
        x,y,z=c.T
        result+=rho*h**3*np.array([len(c),x.sum(),y.sum(),z.sum(),(x*x+h*h/12).sum(),(y*y+h*h/12).sum(),(z*z+h*h/12).sum(),(y*z).sum(),(x*z).sum(),(x*y).sum()])
    return result

def mesh_occupancy(vertices,faces,grid,deadline):
    """Odd/even z-ray centre occupancy of a closed triangulated original mesh.

    Float64 represented-lattice operation, not mesh validity/manufacturing certification.
    Rays on shared triangle edges are deduplicated; centre-on-surface is pending.
    """
    n=grid['side'];check_budget(n);h=grid['h'];o=np.asarray(grid['origin'])
    v=np.asarray(vertices,float);f=np.asarray(faces)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu':raise ValueError('invalid outer mesh')
    if not len(f) or (f<0).any() or (f>=len(v)).any():raise ValueError('invalid outer faces')
    if len(f)*n*n>5_000_000:raise Pending('mesh ray budget exceeded before reconstruction')
    eps=1e-10*h
    if (v<o-eps).any() or (v>o+n*h+eps).any():raise Pending('whole outer mesh extends outside registered lattice; extended adapter required')
    edges={}
    for tri in f:
        for a,b in zip(tri,np.roll(tri,1)):
            k=tuple(sorted((int(a),int(b))));edges[k]=edges.get(k,0)+1
    if any(c!=2 for c in edges.values()):raise Pending('outer mesh is not a closed two-face edge mesh')
    tri=v[f];a=tri[:,0];u=tri[:,1]-a;w=tri[:,2]-a
    det=u[:,0]*w[:,1]-u[:,1]*w[:,0];active=abs(det)>1e-14*h*h
    a,u,w,det=a[active],u[active],w[active],det[active]
    out=np.zeros((n,)*3,bool);zs=o[2]+(np.arange(n)+.5)*h
    for x,y in np.ndindex(n,n):
        if monotonic()>deadline:raise Pending('mesh reconstruction time budget exceeded')
        dx=o[0]+(x+.5)*h-a[:,0];dy=o[1]+(y+.5)*h-a[:,1]
        s=(dx*w[:,1]-dy*w[:,0])/det;t=(u[:,0]*dy-u[:,1]*dx)/det
        ok=(s>=-1e-12)&(t>=-1e-12)&(s+t<=1+1e-12)
        hits=np.sort((a[:,2]+s*u[:,2]+t*w[:,2])[ok]);unique=[]
        for z in hits:
            if not unique or abs(z-unique[-1])>eps:unique.append(float(z))
        if len(unique)%2:raise Pending('ambiguous odd mesh-ray intersections')
        if any(np.any(abs(zs-z)<=eps) for z in unique):raise Pending('voxel centre lies on original surface')
        for lo,hi in zip(unique[::2],unique[1::2]):out[x,y]|=(zs>lo)&(zs<hi)
    return out

def components(mask,deadline):
    # scipy default structure is six face neighbours, stated explicitly here.
    from scipy.ndimage import label
    if monotonic()>deadline:raise Pending('connectivity time budget exceeded')
    structure=np.zeros((3,3,3),np.uint8);structure[1,1,:]=1;structure[1,:,1]=1;structure[:,1,1]=1
    labels,count=label(mask,structure=structure)
    if monotonic()>deadline:raise Pending('connectivity time budget exceeded')
    sizes=np.bincount(labels.ravel())[1:]
    return int(count),int(sizes.max(initial=0))
