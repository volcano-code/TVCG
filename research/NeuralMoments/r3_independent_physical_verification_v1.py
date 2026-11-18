"""Independent spinning-top physical-constraint verification from mass moments.

The four equality constraints mirror SpinMomentProblem::SpinningTop in the
registered target frame: Lx=0, Ly=0, I_c,xz=0, I_c,yz=0. This module also
reports axis-offset/principal-axis diagnostics, but does not promote those
diagnostics to hard constraints unless the caller supplies explicit limits.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

SCHEMA="tvcg.r3.independent-physical-verification.v1"
MOMENT_ORDER=["M","LX","LY","LZ","QXX","QYY","QZZ","QYZ","QXZ","QXY"]

def _vec3(v,name):
    x=np.asarray(v,dtype=np.float64)
    if x.shape!=(3,) or not np.isfinite(x).all(): raise ValueError(f"{name} must be finite [3]")
    return x

def _moments(record):
    if record.get("moment_order")!=MOMENT_ORDER: raise ValueError("moment_order mismatch; expected mass-moment convention")
    x=np.asarray(record.get("raw_moments"),dtype=np.float64)
    if x.shape!=(10,) or not np.isfinite(x).all() or x[0]<=0: raise ValueError("raw mass moments must be finite [10] with positive mass")
    if record.get("backend_identity") in (None,"","native","native_solver"): raise ValueError("independent backend identity required")
    return x

def verify_spinning_top_constraints(record, task_frame, tolerances, *, expected_geometry_hash=None):
    required=("case_id","source_geometry_hash","coordinate_frame","length_unit",
      "meters_per_unit","density","geometric_volume_m3","density_source","tetrahedra_sha256","backend_sha256")
    missing=[k for k in required if k not in record]
    if missing: raise ValueError("independent record missing: "+", ".join(missing))
    if expected_geometry_hash is not None and record["source_geometry_hash"]!=expected_geometry_hash:
        raise ValueError("geometry hash mismatch")
    if not isinstance(tolerances,dict): raise ValueError("explicit tolerances object required")
    hard_names=("Lx_abs","Ly_abs","Icxz_abs","Icyz_abs","mass_relative")
    missing_tol=[k for k in hard_names if k not in tolerances]
    if missing_tol: raise ValueError("explicit hard-constraint tolerances missing: "+", ".join(missing_tol))
    limits={k:float(tolerances[k]) for k in hard_names}
    if any(not math.isfinite(v) or v<0 for v in limits.values()): raise ValueError("tolerances must be finite and nonnegative")
    m=_moments(record)
    density=float(record["density"]); volume=float(record["geometric_volume_m3"])
    scale=float(record["meters_per_unit"])
    if not math.isfinite(density) or density<=0 or not math.isfinite(volume) or volume<=0:
        raise ValueError("density and geometric volume must be positive finite")
    if not math.isfinite(scale) or scale<=0: raise ValueError("meters_per_unit must be positive finite")
    expected_mass=density*volume
    mass_rel=abs(m[0]-expected_mass)/max(abs(m[0]),abs(expected_mass),np.finfo(float).tiny)

    origin=_vec3(task_frame.get("origin"),"task_frame.origin")
    xaxis=_vec3(task_frame.get("x_axis"),"task_frame.x_axis")
    yaxis=_vec3(task_frame.get("y_axis"),"task_frame.y_axis")
    zaxis=_vec3(task_frame.get("axis"),"task_frame.axis")
    R=np.column_stack([xaxis,yaxis,zaxis])
    if np.linalg.norm(R.T@R-np.eye(3),ord="fro")>1e-10 or np.linalg.det(R)<1-1e-10:
        raise ValueError("task frame axes must be right-handed orthonormal columns")
    if task_frame.get("coordinate_frame")!=record["coordinate_frame"]:
        raise ValueError("task and independent coordinate_frame mismatch")
    if task_frame.get("length_unit")!=record["length_unit"]:
        raise ValueError("task and independent length_unit mismatch")
    if float(task_frame.get("meters_per_unit",float("nan")))!=scale:
        raise ValueError("task and independent meters_per_unit mismatch")
    if float(task_frame.get("density",float("nan")))!=density:
        raise ValueError("task and independent density mismatch")

    mass=m[0]
    first_world=m[1:4]
    com_world=first_world/mass
    com_target=R.T@(com_world-origin)
    Q_world=np.array([[m[4],m[9],m[8]],[m[9],m[5],m[7]],[m[8],m[7],m[6]]])
    Q_center_world=Q_world-mass*np.outer(com_world,com_world)
    I_center_world=np.trace(Q_center_world)*np.eye(3)-Q_center_world
    I_target=R.T@I_center_world@R

    # The original SpinMomentProblem spinning-top equalities in target frame.
    Lx_target=float((R.T@first_world)[0]-mass*(R.T@origin)[0])
    Ly_target=float((R.T@first_world)[1]-mass*(R.T@origin)[1])
    Icxz=float(I_target[0,2])
    Icyz=float(I_target[1,2])
    residuals={"Lx_abs":abs(Lx_target),"Ly_abs":abs(Ly_target),
               "Icxz_abs":abs(Icxz),"Icyz_abs":abs(Icyz)}
    passes={k:residuals[k]<=limits[k] for k in residuals}
    passes["mass_relative"]=mass_rel<=limits["mass_relative"]

    projector=np.eye(3)-np.outer(zaxis,zaxis)
    axis_offset=float(np.linalg.norm(projector@(com_world-origin)))
    eigenvalues,eigenvectors=np.linalg.eigh(I_center_world)
    alignments=np.abs(eigenvectors.T@zaxis)
    best=int(np.argmax(alignments))
    principal_axis_alignment_cosine=float(alignments[best])
    principal_axis_angle_radians=float(math.acos(min(1.0,max(-1.0,principal_axis_alignment_cosine))))
    eigengaps=np.diff(eigenvalues)
    diagnostics={
      "com_axis_offset_m":axis_offset,
      "principal_axis_alignment_cosine":principal_axis_alignment_cosine,
      "principal_axis_alignment_angle_radians":principal_axis_angle_radians,
      "nearest_principal_axis_index":best,
      "principal_inertia_eigenvalues":eigenvalues.tolist(),
      "principal_inertia_eigengaps":eigengaps.tolist(),
      "inertia_target_frame":I_target.tolist(),
      "center_of_mass_world":com_world.tolist(),
      "center_of_mass_target_relative_to_origin":com_target.tolist(),
    }
    # Optional diagnostics only become acceptance gates when an explicit limit is supplied.
    optional={}
    if "com_axis_offset_m" in tolerances:
        lim=float(tolerances["com_axis_offset_m"])
        if not math.isfinite(lim) or lim<0: raise ValueError("com_axis_offset_m tolerance invalid")
        optional["com_axis_offset_m"]={"value":axis_offset,"limit":lim,"pass":axis_offset<=lim}
    if "principal_axis_alignment_angle_radians" in tolerances:
        lim=float(tolerances["principal_axis_alignment_angle_radians"])
        if not math.isfinite(lim) or lim<0: raise ValueError("principal-axis angle tolerance invalid")
        optional["principal_axis_alignment_angle_radians"]={"value":principal_axis_angle_radians,"limit":lim,"pass":principal_axis_angle_radians<=lim}

    return {"schema":SCHEMA,"case_id":record["case_id"],"geometry_hash":record["source_geometry_hash"],
      "backend_identity":record["backend_identity"],"independent_backend_sha256":record.get("backend_sha256"),
      "density":density,"geometric_volume_m3":volume,"integrated_mass":float(mass),
      "expected_density_times_volume_mass":expected_mass,"mass_relative_error":mass_rel,
      "task_constraint_residuals":{"Lx":Lx_target,"Ly":Ly_target,"Icxz":Icxz,"Icyz":Icyz},
      "hard_constraint_passes":passes,"hard_constraints_pass":all(passes.values()),
      "diagnostics":diagnostics,"optional_diagnostic_gates":optional,
      "overall_pass":all(passes.values()) and all(x["pass"] for x in optional.values()),
      "tolerances":limits|{k:float(v) for k,v in tolerances.items() if k not in limits},
      "constraint_semantics":"SpinMomentProblem::SpinningTop equalities Lx=0, Ly=0, I_c,xz=0, I_c,yz=0; other reported values are diagnostics unless explicit limits are supplied"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--independent",required=True);p.add_argument("--task-frame",required=True)
    p.add_argument("--tolerances",required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    record=json.loads(Path(a.independent).read_text())
    frame=json.loads(Path(a.task_frame).read_text())
    tol=json.loads(Path(a.tolerances).read_text())
    result=verify_spinning_top_constraints(record,frame,tol,expected_geometry_hash=frame.get("geometry_hash"))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"case_id":result["case_id"],"overall_pass":result["overall_pass"],
      "hard_constraint_passes":result["hard_constraint_passes"],"output":str(out)},indent=2))
    raise SystemExit(0 if result["overall_pass"] else 3)
if __name__=="__main__":main()
