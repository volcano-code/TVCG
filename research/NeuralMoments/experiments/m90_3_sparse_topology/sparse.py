"""Independent sparse six-face topology; no dense volume or frozen Oracle import."""
from time import monotonic
import numpy as np

class Pending(RuntimeError): pass

def _tick(deadline):
    if monotonic()>deadline:raise Pending('bounded topology deadline reached')

def validated(runs,side):
    if type(side) is not int or not 1<=side<=2048:raise ValueError('integer side in [1,2048] required')
    a=np.asarray(runs)
    if a.ndim!=2 or a.shape[1]!=4 or a.dtype.kind not in 'iu':raise ValueError('integer (N,4) RLE required, no coercion')
    if len(a)>2_000_000:raise Pending('RLE interval budget exceeded')
    if len(a):
        if np.any(a[:,:3]<0) or np.any(a[:,:2]>=side) or np.any(a[:,3]>side) or np.any(a[:,2]>=a[:,3]):raise ValueError('invalid/out-of-range RLE')
        a=a.astype(np.int64,copy=False)
        keys=(a[:,0]*side+a[:,1])*(side+1)+a[:,2]
        if np.any(keys[1:]<=keys[:-1]):raise ValueError('unordered/duplicate RLE')
        same=np.all(a[1:,:2]==a[:-1,:2],axis=1)
        if np.any(same&(a[1:,2]<=a[:-1,3])):raise ValueError('overlap or unmerged touching intervals')
    return a.astype(np.int64,copy=False)

def subtract(original,deleted,side,*,deadline=None):
    """Strict D subset O difference; no clipping, sorting or repair."""
    deadline=monotonic()+110 if deadline is None else deadline
    a,b=validated(original,side),validated(deleted,side);out=[];j=0
    for i,(x,y,lo,hi) in enumerate(a):
        if i%1024==0:_tick(deadline)
        key=int(x)*side+int(y);cursor=int(lo)
        while j<len(b):
            bx,by,bl,bh=map(int,b[j]);bk=bx*side+by
            if bk<key or (bk==key and bl<lo):raise ValueError('deleted cells outside original support')
            if bk>key or bl>=hi:break
            if bh>hi:raise ValueError('deleted cells outside original support')
            if cursor<bl:out.append((int(x),int(y),cursor,bl))
            cursor=bh;j+=1
        if cursor<hi:out.append((int(x),int(y),cursor,int(hi)))
    if j!=len(b):raise ValueError('deleted cells outside original support')
    _tick(deadline)
    return np.asarray(out,np.int64).reshape(-1,4)

def components(runs,side,*,deadline=None):
    """Each z interval is one node; link strict z overlaps on adjacent xy columns."""
    deadline=monotonic()+110 if deadline is None else deadline
    a=validated(runs,side);N=len(a)
    if not N:return {'components':0,'largest_voxels':0,'total_voxels':0,'intervals':0,'dense_n_cubed_allocated':False}
    keys=a[:,0]*side+a[:,1]
    offsets=np.zeros(side*side+1,np.int64)
    offsets[1:]=np.cumsum(np.bincount(keys,minlength=side*side))
    parent=np.arange(N,dtype=np.int64);size=np.ones(N,dtype=np.int64)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    def join(i,j):
        i,j=root(i),root(j)
        if i==j:return
        if size[i]<size[j]:i,j=j,i
        parent[j]=i;size[i]+=size[j]
    def link(key,other):
        i,end=int(offsets[key]),int(offsets[key+1]);j,jend=int(offsets[other]),int(offsets[other+1])
        while i<end and j<jend:
            if max(a[i,2],a[j,2])<min(a[i,3],a[j,3]):join(i,j)
            ai,bj=a[i,3],a[j,3]
            if ai<=bj:i+=1
            if bj<=ai:j+=1
    for index,k in enumerate(np.flatnonzero(np.diff(offsets))):
        if index%1024==0:_tick(deadline)
        k=int(k)
        if k%side:link(k,k-1)
        if k>=side:link(k,k-side)
    counts=np.zeros(N,np.int64)
    for i in range(N):
        if i%4096==0:_tick(deadline)
        counts[root(i)]+=a[i,3]-a[i,2]
    _tick(deadline)
    return {'components':int(np.count_nonzero(counts)),'largest_voxels':int(counts.max(initial=0)),'total_voxels':int(counts.sum()),'intervals':N,'dense_n_cubed_allocated':False}

def mesh_runs(vertices,faces,origin,h,side,*,max_projected_tests=2_000_000,deadline=None):
    """Closed mesh parity rays at lattice centres, sparse triangle-event traversal.

    Original mesh must fit the lattice; larger/ambiguous cases are pending.
    This is float64 represented-lattice evidence, not exact CAD/manufacturing.
    """
    start=monotonic();deadline=start+110 if deadline is None else deadline
    if vertices is None or faces is None:raise Pending('missing original outer mesh')
    validated(np.empty((0,4),np.int64),side)
    v=np.asarray(vertices);f=np.asarray(faces);o=np.asarray(origin,float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not len(f):raise ValueError('invalid mesh arrays')
    if len(f)>1_000_000 or len(v)>1_000_000:raise Pending('mesh input budget exceeded')
    if o.shape!=(3,) or not np.isfinite(o).all() or type(h) not in (int,float) or not np.isfinite(h) or h<=0:raise ValueError('invalid physical grid')
    if (f<0).any() or (f>=len(v)).any():raise ValueError('invalid face indices')
    v=v.astype(np.float64,copy=False);eps=h*1e-10
    if (v<o-eps).any() or (v>o+side*h+eps).any():raise Pending('whole mesh extends outside lattice; explicit padded registration required')
    tri=v[f];u=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0]
    det=u[:,0]*w[:,1]-u[:,1]*w[:,0]
    active=abs(det)>h*h*1e-14
    # Reject degenerate triangles; no repair. Vertical valid faces simply cast no z crossings.
    if np.any(np.linalg.norm(np.cross(u,w),axis=1)<=h*h*1e-14):raise Pending('degenerate outer triangle')
    left=np.maximum(0,np.ceil((tri[active,:,:2].min(axis=1)-o[:2])/h-.5-1e-12)).astype(np.int64)
    right=np.minimum(side-1,np.floor((tri[active,:,:2].max(axis=1)-o[:2])/h-.5+1e-12)).astype(np.int64)
    widths=np.maximum(0,right-left+1);projected=int(np.sum(widths[:,0]*widths[:,1],dtype=np.int64))
    estimate=128*projected+128*len(f)+64*1024**2
    if projected>max_projected_tests or estimate>768*1024**2:raise Pending(f'preflight budget: projected_tests={projected}, estimated_bytes={estimate}')
    edges={}
    for i,face in enumerate(f):
        if i%1024==0:_tick(deadline)
        for a,b in zip(face,np.roll(face,1)):
            key=(min(int(a),int(b)),max(int(a),int(b)));edges[key]=edges.get(key,0)+1
    if any(c!=2 for c in edges.values()):raise Pending('outer mesh not closed with exactly two faces per edge')
    del edges
    # Z-parallel faces do not emit crossings, but can contain voxel centres.
    # Test them in their dominant-normal projection, without a dense volume.
    parallel_tests=0
    for idx in np.flatnonzero(~active):
        _tick(deadline)
        normal=np.cross(u[idx],w[idx]);axis=int(np.argmax(abs(normal)))
        axes=[a for a in range(3) if a!=axis]
        depth_lo=(tri[idx,:,axis].min()-o[axis])/h-.5
        depth_hi=(tri[idx,:,axis].max()-o[axis])/h-.5
        if np.ceil(depth_lo-1e-10)>np.floor(depth_hi+1e-10):continue
        lower=np.maximum(0,np.ceil((tri[idx][:,axes].min(0)-o[axes])/h-.5-1e-12)).astype(int)
        upper=np.minimum(side-1,np.floor((tri[idx][:,axes].max(0)-o[axes])/h-.5+1e-12)).astype(int)
        if np.any(upper<lower):continue
        tests=int(np.prod(upper-lower+1));parallel_tests+=tests
        if projected+parallel_tests>max_projected_tests:raise Pending('parallel-surface projection budget exceeded')
        i,j=np.meshgrid(np.arange(lower[0],upper[0]+1),np.arange(lower[1],upper[1]+1),indexing='ij');i=i.ravel();j=j.ravel()
        a=tri[idx,0];b,c=axes
        dx=o[b]+(i+.5)*h-a[b];dy=o[c]+(j+.5)*h-a[c]
        determinant=u[idx,b]*w[idx,c]-u[idx,c]*w[idx,b]
        first=(dx*w[idx,c]-dy*w[idx,b])/determinant
        second=(u[idx,b]*dy-u[idx,c]*dx)/determinant
        inside=(first>=-1e-12)&(second>=-1e-12)&(first+second<=1+1e-12)
        depth=(a[axis]+first[inside]*u[idx,axis]+second[inside]*w[idx,axis]-o[axis])/h-.5
        if np.any(abs(depth-np.rint(depth))<=1e-10):raise Pending('original vertical surface lies on a voxel centre')
    ids=np.flatnonzero(active);event_keys=[];event_z=[]
    for k,idx in enumerate(ids):
        if k%256==0:_tick(deadline)
        lo,hi=left[k],right[k]
        if np.any(hi<lo):continue
        ix,iy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij');ix=ix.ravel();iy=iy.ravel()
        a=tri[idx,0];dx=o[0]+(ix+.5)*h-a[0];dy=o[1]+(iy+.5)*h-a[1]
        s=(dx*w[idx,1]-dy*w[idx,0])/det[idx];t=(u[idx,0]*dy-u[idx,1]*dx)/det[idx]
        valid=(s>=-1e-12)&(t>=-1e-12)&(s+t<=1+1e-12)
        if valid.any():
            event_keys.append(ix[valid]*side+iy[valid]);event_z.append(a[2]+s[valid]*u[idx,2]+t[valid]*w[idx,2])
    if not event_keys:raise Pending('no outer mesh ray crossings on registered lattice')
    keys=np.concatenate(event_keys);zs=np.concatenate(event_z);del event_keys,event_z
    order=np.lexsort((zs,keys));keys,zs=keys[order],zs[order];del order
    # Identical hits at shared triangle edges describe a single crossing.
    keep=np.r_[True,(keys[1:]!=keys[:-1])|(abs(zs[1:]-zs[:-1])>eps)]
    keys,zs=keys[keep],zs[keep]
    q=(zs-o[2])/h-.5
    if np.any(abs(q-np.rint(q))<=1e-10):raise Pending('original surface lies on a voxel centre; no arbitrary endpoint policy')
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1];ends=np.r_[starts[1:],len(keys)]
    if np.any((ends-starts)%2):raise Pending('odd/ambiguous original mesh ray parity')
    rows=[]
    for k,(a,b) in enumerate(zip(starts,ends)):
        if k%1024==0:_tick(deadline)
        key=int(keys[a]);x,y=divmod(key,side)
        for low,high in zip(zs[a:b:2],zs[a+1:b:2]):
            lo=int(np.floor((low-o[2])/h-.5))+1;hi=int(np.ceil((high-o[2])/h-.5))
            if not 0<=lo<=side or not 0<=hi<=side:raise Pending('outside reconstructed interval')
            if lo<hi:
                # Separate surfaces can leave a subvoxel gap with no empty centre.
                if rows and rows[-1][:2]==[x,y] and rows[-1][3]==lo:rows[-1][3]=hi
                else:rows.append([x,y,lo,hi])
    result=validated(np.asarray(rows,np.int64).reshape(-1,4),side)
    return result,{'method':'float64 sparse triangle-event parity at voxel centres','projected_tests':projected,'parallel_surface_tests':parallel_tests,'estimated_workspace_bytes':estimate+128*parallel_tests,'crossings_after_shared_edge_dedup':len(keys),'intervals':len(result),'seconds':monotonic()-start,'dense_n_cubed_allocated':False,'mesh_repair':False,'surface_policy':'centre on surface or odd parity => pending'}
