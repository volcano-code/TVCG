"""R3.1.2 real-geometry semantic admission gate.

This gate joins already loaded geometry assets to reviewed task semantics and a
source-backed pin/support artifact. It does not create geometry, infer support,
generate masks, or assign units. A successful report means admission metadata is
complete and internally consistent; physical mesh validity still needs the native
exact geometry checks.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA = "tvcg.r3.real-geometry-admission.v1"

def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _vec3(v,name):
    a=np.asarray(v,dtype=np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all(): raise ValueError(f"{name} must be finite 3-vector")
    return a

def _unit(v,name):
    a=_vec3(v,name); n=float(np.linalg.norm(a))
    if n<=1e-14: raise ValueError(f"{name} cannot be zero")
    return a/n

def _hash_record(root,record,path_key,hash_key):
    rel=record.get(path_key); expected=record.get(hash_key)
    if not isinstance(rel,str) or not rel: raise ValueError(f"{path_key} must be explicit")
    p=root/rel
    if not p.is_file(): raise FileNotFoundError(f"{path_key}: {p}")
    actual=_sha(p)
    if not isinstance(expected,str) or actual!=expected.lower():
        raise ValueError(f"{hash_key} mismatch for {p}")
    return p,actual

def admit_real_geometry(manifest_path: str|Path)->dict[str,Any]:
    p=Path(manifest_path); root=p.parent
    m=json.loads(p.read_text())
    if m.get("schema")!=SCHEMA: raise ValueError("unsupported real geometry admission schema")
    required=("case_id","geometry_manifest","geometry_manifest_sha256","task_record",
      "task_record_sha256","pin_support_file","pin_support_sha256","axis","axis_origin",
      "contact_point","coordinate_frame","length_unit","meters_per_unit",
      "grid_to_world_linear","grid_to_world_translation","initial_material_policy",
      "fixed_shell_semantics","design_domain_semantics","semantic_review")
    missing=[k for k in required if k not in m]
    if missing: raise ValueError("incomplete admission manifest: "+", ".join(missing))
    gm,gm_hash=_hash_record(root,m,"geometry_manifest","geometry_manifest_sha256")
    taskp,task_hash=_hash_record(root,m,"task_record","task_record_sha256")
    pinp,pin_hash=_hash_record(root,m,"pin_support_file","pin_support_sha256")
    geometry=json.loads(gm.read_text())
    task=json.loads(taskp.read_text())
    if geometry.get("schema")!="tvcg.r3.geometry-adapter.v1":
        raise ValueError("geometry manifest schema mismatch")
    if not isinstance(m["semantic_review"],dict) or m["semantic_review"].get("status")!="reviewed":
        raise ValueError("semantic_review must be explicitly reviewed")
    reviewer=m["semantic_review"].get("reviewer")
    review_source=m["semantic_review"].get("source_reference")
    if not reviewer or not review_source: raise ValueError("semantic review requires reviewer and source_reference")
    axis=_unit(m["axis"],"axis")
    origin=_vec3(m["axis_origin"],"axis_origin")
    contact=_vec3(m["contact_point"],"contact_point")
    off=contact-origin; off-=float(np.dot(off,axis))*axis
    if float(np.linalg.norm(off))>1e-9*max(1.0,float(np.linalg.norm(contact-origin))):
        raise ValueError("contact point does not lie on declared rotation axis")
    unit=m["length_unit"]
    scale=float(m["meters_per_unit"])
    if not isinstance(unit,str) or not unit.strip() or not math.isfinite(scale) or scale<=0:
        raise ValueError("length unit and positive meters_per_unit must be explicit")
    A=np.asarray(m["grid_to_world_linear"],dtype=np.float64)
    t=_vec3(m["grid_to_world_translation"],"grid_to_world_translation")
    if A.shape!=(3,3) or not np.isfinite(A).all() or abs(float(np.linalg.det(A)))<1e-15:
        raise ValueError("grid_to_world_linear must be finite invertible 3x3")
    if task.get("task_hash")!=m.get("task_hash"):
        raise ValueError("task_hash in admission manifest does not match task record")
    if not m.get("task_hash"): raise ValueError("task_hash must be explicit")
    task_axis=_unit(task["axis"],"task.axis")
    task_origin=_vec3(task["axis_origin"],"task.axis_origin")
    task_contact=_vec3(task["contact"],"task.contact")
    tol=float(m.get("task_frame_tolerance",1e-10))
    if np.linalg.norm(axis-task_axis)>tol or np.linalg.norm(origin-task_origin)>tol or np.linalg.norm(contact-task_contact)>tol:
        raise ValueError("admission axis/origin/contact differ from canonical TaskSpec")
    if m["initial_material_policy"]!="source_defined_full_material_or_explicit_state":
        if not isinstance(m["initial_material_policy"],str) or not m["initial_material_policy"].strip():
            raise ValueError("initial material policy must be explicit")
    if not isinstance(m["fixed_shell_semantics"],str) or not m["fixed_shell_semantics"].strip():
        raise ValueError("fixed_shell_semantics must cite reviewed construction semantics")
    if not isinstance(m["design_domain_semantics"],str) or not m["design_domain_semantics"].strip():
        raise ValueError("design_domain_semantics must cite reviewed construction semantics")
    # Pin/support artifact is hash-bound, but its interpretation remains a reviewed statement.
    pin_semantics=m.get("pin_support_semantics")
    if not isinstance(pin_semantics,str) or not pin_semantics.strip():
        raise ValueError("pin_support_semantics must explicitly describe the source artifact")
    return {
      "schema":"tvcg.r3.real-geometry-admission-report.v1",
      "case_id":m["case_id"],"geometry_manifest_sha256":gm_hash,
      "task_record_sha256":task_hash,"task_hash":m["task_hash"],
      "pin_support_sha256":pin_hash,"axis":axis.tolist(),
      "axis_origin":origin.tolist(),"contact_point":contact.tolist(),
      "coordinate_frame":m["coordinate_frame"],"length_unit":unit,
      "meters_per_unit":scale,"grid_to_world_linear":A.tolist(),
      "grid_to_world_translation":t.tolist(),
      "semantic_reviewer":reviewer,"semantic_source_reference":review_source,
      "metadata_admission":True,"physical_mesh_admission":False,
      "native_exact_geometry_checks_passed":False,
      "reason":"native topology/containment and independently reviewed mesh/mask correspondence remain required"
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); report=admit_real_geometry(a.manifest)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
