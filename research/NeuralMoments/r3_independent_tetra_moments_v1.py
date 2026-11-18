"""Independent exact raw-moment integration over an explicit tetrahedral solid mesh.

This backend is independent of the native quadric evaluator. Input tetrahedra must
already represent the admitted retained material volume. No tetrahedralization,
voxelization, density inference, or material classification is performed here.
Coordinates are converted to metres using an explicit metres_per_unit factor.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

SCHEMA = "tvcg.r3.independent-tetra-moments.v1"
MOMENT_ORDER = ["V","Lx","Ly","Lz","Qxx","Qyy","Qzz","Qyz","Qxz","Qxy"]

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def integrate_tetrahedra(tetrahedra, *, density, meters_per_unit, orientation_tolerance=1e-14):
    t=np.asarray(tetrahedra,dtype=np.float64)
    if t.ndim!=3 or t.shape[1:]!=(4,3) or len(t)==0:
        raise ValueError("tetrahedra must have shape [T,4,3]")
    if not np.isfinite(t).all(): raise ValueError("tetrahedra contain non-finite coordinates")
    rho=float(density); scale=float(meters_per_unit)
    if not math.isfinite(rho) or rho<=0: raise ValueError("density must be positive finite")
    if not math.isfinite(scale) or scale<=0: raise ValueError("meters_per_unit must be positive finite")
    x=t*scale
    e1=x[:,1]-x[:,0]; e2=x[:,2]-x[:,0]; e3=x[:,3]-x[:,0]
    signed=np.einsum("ij,ij->i",e1,np.cross(e2,e3))/6.0
    if np.any(np.abs(signed)<=orientation_tolerance):
        raise ValueError("degenerate or near-zero-volume tetrahedron")
    if np.any(signed<0):
        raise ValueError("inconsistent tetrahedron orientation: negative signed volume")
    volumes=signed
    sums=x.sum(axis=1)
    first=np.sum(volumes[:,None]*sums/4.0,axis=0)
    # Exact simplex identity: integral x_i*x_j = V/20 * ((sum_v x_v_i)(sum_v x_v_j) + sum_v x_v_i*x_v_j).
    vertex_products=np.matmul(np.transpose(x,(0,2,1)),x)
    second=np.sum(volumes[:,None,None]*(sums[:,:,None]*sums[:,None,:]+vertex_products)/20.0,axis=0)
    raw=np.array([volumes.sum(),*first,*np.diag(second),second[1,2],second[0,2],second[0,1]],dtype=np.float64)
    raw*=rho
    if not np.isfinite(raw).all() or raw[0]<=0: raise ValueError("integrated raw moments are invalid")
    return raw

def derive_mass_properties(raw_moments):
    m=np.asarray(raw_moments,dtype=np.float64)
    if m.shape!=(10,) or not np.isfinite(m).all() or m[0]<=0:
        raise ValueError("raw moments must be finite [10] with positive mass-volume component")
    mass,V1,V2,V3,Qxx,Qyy,Qzz,Qyz,Qxz,Qxy=m
    com=np.array([V1,V2,V3])/mass
    Q=np.array([[Qxx,Qxy,Qxz],[Qxy,Qyy,Qyz],[Qxz,Qyz,Qzz]])
    I_origin=np.trace(Q)*np.eye(3)-Q
    I_center=I_origin-mass*((com@com)*np.eye(3)-np.outer(com,com))
    return {"mass":float(mass),"center_of_mass":com.tolist(),
      "second_moment_tensor":Q.tolist(),"inertia_origin":I_origin.tolist(),
      "inertia_center":I_center.tolist()}

def integrate_from_manifest(manifest_path):
    p=Path(manifest_path); m=json.loads(p.read_text()); root=p.parent
    if m.get("schema")!=SCHEMA: raise ValueError("unsupported independent tetra moment schema")
    required=("case_id","tetrahedra_file","tetrahedra_sha256","tetrahedra_key",
      "geometry_hash","coordinate_frame","length_unit","meters_per_unit","density","density_source")
    missing=[k for k in required if k not in m]
    if missing: raise ValueError("manifest missing: "+", ".join(missing))
    f=root/m["tetrahedra_file"]
    if not f.is_file(): raise FileNotFoundError(f)
    actual=sha256_file(f)
    if actual!=str(m["tetrahedra_sha256"]).lower(): raise ValueError("tetrahedra SHA-256 mismatch")
    with np.load(f,allow_pickle=False) as z:
        if m["tetrahedra_key"] not in z.files: raise KeyError("tetrahedra array key missing")
        t=np.array(z[m["tetrahedra_key"]],copy=True)
    raw=integrate_tetrahedra(t,density=m["density"],meters_per_unit=m["meters_per_unit"])
    props=derive_mass_properties(raw)
    return {"schema":"tvcg.r3.independent-moment-record.v1","case_id":m["case_id"],
      "source_geometry_hash":m["geometry_hash"],"backend_identity":"independent_tetra_exact_v1",
      "tetrahedra_sha256":actual,"coordinate_frame":m["coordinate_frame"],
      "length_unit":m["length_unit"],"meters_per_unit":m["meters_per_unit"],
      "density":m["density"],"density_source":m["density_source"],
      "moment_order":MOMENT_ORDER,"raw_moments":raw.tolist(),"mass_properties":props}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    a=p.parse_args(); result=integrate_from_manifest(a.manifest)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"case_id":result["case_id"],"backend_identity":result["backend_identity"],
      "raw_moments":result["raw_moments"],"mass_properties":result["mass_properties"],
      "output":str(out)},indent=2))
if __name__=="__main__":main()
