"""R3.0.3 explicit geometry adapter: no inferred masks or coordinate changes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np

SCHEMA = "tvcg.r3.geometry-adapter.v1"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _npz(path,key):
    with np.load(path,allow_pickle=False) as z:
        if key not in z.files: raise KeyError(f"{path}: missing {key}; found {z.files}")
        return np.array(z[key],copy=True)

def _mesh(v,f):
    v=np.asarray(v,dtype=np.float64); f=np.asarray(f)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
        raise ValueError("vertices must be finite [N,3]")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)<4 or f.dtype.kind not in "iu":
        raise ValueError("faces must be integer triangles [F,3]")
    if (f<0).any() or (f>=len(v)).any(): raise ValueError("face index outside vertices")
    if any(len(set(map(int,row)))!=3 for row in f): raise ValueError("degenerate index triangle")
    return v,f.astype(np.int64,copy=False)

def _mask(path,key,shape,name):
    a=_npz(path,key)
    if a.shape!=shape: raise ValueError(f"{name} shape {a.shape} != {shape}")
    if a.dtype!=np.bool_: raise ValueError(f"{name} must be bool; thresholding forbidden")
    return a

@dataclass(frozen=True)
class GeometryBundle:
    vertices: np.ndarray
    faces: np.ndarray
    fixed_shell_mask: np.ndarray
    design_domain_mask: np.ndarray
    grid_origin: tuple
    unit_token: str
    mesh_sha256: str
    shell_mask_sha256: str
    design_mask_sha256: str
    grid_shape: tuple
    def validate_partition(self):
        if np.any(self.fixed_shell_mask & self.design_domain_mask):
            raise ValueError("fixed shell and design domain overlap")
        if not self.fixed_shell_mask.any(): raise ValueError("fixed shell mask is empty")
        if not self.design_domain_mask.any(): raise ValueError("design domain mask is empty")
        return True

def load_geometry_bundle(manifest_path):
    p=Path(manifest_path); m=json.loads(p.read_text())
    if m.get("schema")!=SCHEMA: raise ValueError("unsupported geometry adapter schema")
    required=("mesh_file","vertices_key","faces_key","shell_mask_file","shell_mask_key",
              "design_mask_file","design_mask_key","grid_shape","grid_origin","unit_token",
              "mesh_sha256","shell_mask_sha256","design_mask_sha256")
    missing=[k for k in required if k not in m]
    if missing: raise ValueError("incomplete geometry manifest: "+", ".join(missing))
    root=p.parent
    paths={k:root/m[k] for k in ("mesh_file","shell_mask_file","design_mask_file")}
    for key,path in paths.items():
        if not path.is_file(): raise FileNotFoundError(path)
        hash_key={"mesh_file":"mesh_sha256","shell_mask_file":"shell_mask_sha256",
                  "design_mask_file":"design_mask_sha256"}[key]
        actual=sha256_file(path)
        if actual!=str(m[hash_key]).lower():
            raise ValueError(f"{hash_key} mismatch: expected {m[hash_key]}, got {actual}")
    v,f=_mesh(_npz(paths["mesh_file"],m["vertices_key"]),_npz(paths["mesh_file"],m["faces_key"]))
    shape=tuple(m["grid_shape"])
    if len(shape)!=3 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError("grid_shape must be three positive integers")
    origin=tuple(float(x) for x in m["grid_origin"])
    if len(origin)!=3 or not np.isfinite(origin).all(): raise ValueError("invalid grid_origin")
    token=m["unit_token"]
    if not isinstance(token,str) or not token.strip(): raise ValueError("unit_token must be explicit")
    shell=_mask(paths["shell_mask_file"],m["shell_mask_key"],shape,"fixed_shell_mask")
    design=_mask(paths["design_mask_file"],m["design_mask_key"],shape,"design_domain_mask")
    out=GeometryBundle(v,f,shell,design,origin,token,m["mesh_sha256"],
                       m["shell_mask_sha256"],m["design_mask_sha256"],shape)
    out.validate_partition()
    return out
