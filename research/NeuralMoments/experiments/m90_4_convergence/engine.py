"""Isolated sparse geometry diagnostics. All prisms stay on the registered grid."""
import ctypes,hashlib,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'m90_3_sparse_topology'))
from sparse import validated,subtract,components,mesh_runs,Pending
from test_sparse import box_mesh
ROOT=HERE.parents[3]
REPORT=ROOT/'reports/m90_4_convergence_20261116'
_lib=None
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def library():
 global _lib
 if _lib is None:
  import json
  receipt=json.loads((REPORT/'native_build.json').read_text())
  for p,d in receipt['inputs'].items():
   if digest(p)!=d:raise ValueError('native source hash mismatch')
  lib=REPORT/'build/mesh68_sparse.so'
  if digest(lib)!=receipt['library_sha256']:raise ValueError('native library hash mismatch')
  _lib=ctypes.CDLL(str(lib))
 return _lib
def mesh68_runs(vertices,faces,origin,h,n,phase):
 if vertices is None or faces is None:raise Pending('missing original outer mesh')
 v=np.ascontiguousarray(vertices,np.float64);f=np.asarray(faces);o=np.ascontiguousarray(origin,np.float64)
 validated(np.empty((0,4),np.int64),n)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not len(f):raise ValueError('invalid mesh')
 if len(f)>1000000 or len(v)>1000000:raise Pending('mesh input budget')
 if o.shape!=(3,) or not np.isfinite(o).all() or not np.isfinite(h) or h<=0:raise ValueError('invalid grid')
 if (f<0).any() or (f>=len(v)).any():raise ValueError('invalid face indices')
 if (v<o-1e-10*h).any() or (v>o+n*h+1e-10*h).any():raise Pending('mesh outside original lattice')
 if len(phase)!=2 or not np.isfinite(phase).all():raise ValueError('invalid phase')
 edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
 if np.any(np.unique(edges,axis=0,return_counts=True)[1]!=2):raise Pending('outer mesh not closed')
 f=np.ascontiguousarray(f,np.int64)
 cap=min(2_000_000,max(n*n*4,1));out=np.empty((cap,4),np.int64);stats=np.zeros(3,np.int64)
 fn=library().m904_outer
 fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p]
 result=fn(v.ctypes.data,f.ctypes.data,len(f),o.ctypes.data,h,n,*phase,out.ctypes.data,cap,stats.ctypes.data)
 if result!=0:raise Pending('native ray parity/capacity failure: '+str(result))
 a=validated(out[:int(stats[1])].copy(),n)
 return a,dict(method='frozen Mesh68 Tree/ray + sparse native outer classification',raw_native_sample_count=int(stats[0]),canonical_intervals=len(a),odd_columns=int(stats[2]),dense_n_cubed_allocated=False,phase_xy=phase)
def exact_runs(v,f,o,h,n,phase):
 # Translate the sampling geometry oppositely to simulate xy ray offsets.
 # This never translates integration cells, source reference, or physical frame.
 delta=np.array([*phase,0.])*h
 return mesh_runs(None if v is None else np.asarray(v)-delta,f,o,h,n)
def cuboid(lo,hi):
 lo,hi=np.array(lo,float),np.array(hi,float);m=np.prod(hi-lo);c=(lo+hi)/2;s=(lo*lo+lo*hi+hi*hi)/3
 return m*np.array([1,*c,*s,c[1]*c[2],c[0]*c[2],c[0]*c[1]])
def box_runs(lo,hi,o,h,n):
 lo=np.maximum(0,np.ceil((np.asarray(lo)-o)/h-.5)).astype(int)
 hi=np.minimum(n,np.floor((np.asarray(hi)-o)/h-.5).astype(int)+1)
 if np.any(hi<=lo):return np.empty((0,4),np.int64)
 x,y=np.meshgrid(np.arange(lo[0],hi[0]),np.arange(lo[1],hi[1]),indexing='ij')
 return np.column_stack((x.ravel(),y.ravel(),np.full(x.size,lo[2]),np.full(x.size,hi[2]))).astype(np.int64)
def moments(runs,o,h,rho):
 # Independent antiderivatives, chunked float64; no Oracle moment calls.
 total=np.zeros(10,np.float64);o=np.asarray(o,float)
 for i in range(0,len(runs),16384):
  r=np.asarray(runs[i:i+16384],np.float64)
  low=o+r[:,:3]*h;high=low.copy();high[:,:2]+=h;high[:,2]=o[2]+r[:,3]*h
  a=high-low;b=(high**2-low**2)/2;c=(high**3-low**3)/3
  terms=[a[:,0]*a[:,1]*a[:,2],b[:,0]*a[:,1]*a[:,2],a[:,0]*b[:,1]*a[:,2],a[:,0]*a[:,1]*b[:,2],c[:,0]*a[:,1]*a[:,2],a[:,0]*c[:,1]*a[:,2],a[:,0]*a[:,1]*c[:,2],a[:,0]*b[:,1]*b[:,2],b[:,0]*a[:,1]*b[:,2],b[:,0]*b[:,1]*a[:,2]]
  total+=rho*np.array([np.sum(t,dtype=np.float64) for t in terms])
 return total
def membership(a,b,n):
 a,b=validated(a,n),validated(b,n);i=j=common=0
 while i<len(a) and j<len(b):
  ak=int(a[i,0])*n+int(a[i,1]);bk=int(b[j,0])*n+int(b[j,1])
  if ak<bk:i+=1;continue
  if bk<ak:j+=1;continue
  common+=max(0,int(min(a[i,3],b[j,3])-max(a[i,2],b[j,2])))
  ai,bj=a[i,3],b[j,3]
  if ai<=bj:i+=1
  if bj<=ai:j+=1
 ca=int(np.sum(a[:,3]-a[:,2]));cb=int(np.sum(b[:,3]-b[:,2]));union=ca+cb-common
 return dict(count_a=ca,count_b=cb,intersection_voxels=common,union_voxels=union,xor_voxels=ca+cb-2*common,jaccard_distance=(ca+cb-2*common)/union if union else 0.)
def save_rle(path,r,n):
 with path.open('xb') as f:f.write(np.asarray(validated(r,n),dtype='<u4').tobytes())
 payload=path.read_bytes()
 if len(payload)%16:raise ValueError('truncated saved RLE')
 result=validated(np.frombuffer(payload,dtype='<u4').reshape(-1,4),n)
 return result,dict(path=str(path),sha256=digest(path),bytes=len(payload))
