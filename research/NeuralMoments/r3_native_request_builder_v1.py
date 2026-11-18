"""R3.1.3 build a native SpinItFasterRequest from explicit source-backed assets.

No mesh is generated, repaired, transformed, or inferred. The manifest must point to
outer and design triangle meshes in the exact coordinate frame expected by the solver.
All physical request scalars/frames are supplied explicitly by the reviewed task JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

SCHEMA="tvcg.r3.native-request-manifest.v1"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def mesh_npz(path,vertices_key,faces_key):
    with np.load(path,allow_pickle=False) as z:
        if vertices_key not in z.files or faces_key not in z.files:
            raise KeyError(f"{path}: required keys missing; found {z.files}")
        v=np.asarray(z[vertices_key],dtype=np.float64)
        f=np.asarray(z[faces_key])
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
        raise ValueError(f"{path}: vertices must be finite [N,3]")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)<4 or f.dtype.kind not in "iu":
        raise ValueError(f"{path}: faces must be integer triangles [F,3]")
    if (f<0).any() or (f>=len(v)).any() or (f>2**31-1).any():
        raise ValueError(f"{path}: face indices out of range")
    if any(len(set(map(int,row)))!=3 for row in f):
        raise ValueError(f"{path}: degenerate triangle")
    return v.tolist(),f.astype(np.int64).tolist()

def build_request(manifest_path):
    p=Path(manifest_path); root=p.parent
    m=json.loads(p.read_text())
    if m.get("schema")!=SCHEMA: raise ValueError("unsupported native request manifest schema")
    required=("case_id","geometry_hash","task_hash","task_record","task_record_sha256",
      "outer_mesh_file","outer_mesh_sha256","outer_vertices_key","outer_faces_key",
      "design_mesh_file","design_mesh_sha256","design_vertices_key","design_faces_key",
      "pin_support_file","pin_support_sha256","request_fields","semantic_review",
      "moment_metadata")
    missing=[k for k in required if k not in m]
    if missing: raise ValueError("incomplete native request manifest: "+", ".join(missing))
    if not isinstance(m["semantic_review"],dict) or m["semantic_review"].get("status")!="reviewed":
        raise ValueError("request semantics must be reviewed")
    for k in ("reviewer","source_reference"):
        if not m["semantic_review"].get(k): raise ValueError(f"semantic_review.{k} required")
    files={}
    for path_key,hash_key in (("task_record","task_record_sha256"),
       ("outer_mesh_file","outer_mesh_sha256"),("design_mesh_file","design_mesh_sha256"),("pin_support_file","pin_support_sha256")):
        fp=root/m[path_key]
        if not fp.is_file(): raise FileNotFoundError(fp)
        actual=sha256(fp)
        if actual!=str(m[hash_key]).lower(): raise ValueError(f"{hash_key} mismatch")
        files[path_key]={"path":str(fp),"sha256":actual}
    task=json.loads((root/m["task_record"]).read_text())
    if task.get("task_hash")!=m["task_hash"]: raise ValueError("TaskSpec hash mismatch")
    outer_v,outer_f=mesh_npz(root/m["outer_mesh_file"],m["outer_vertices_key"],m["outer_faces_key"])
    design_v,design_f=mesh_npz(root/m["design_mesh_file"],m["design_vertices_key"],m["design_faces_key"])
    moment_metadata=dict(m["moment_metadata"])
    metadata_required=("coordinate_frame","length_unit","meters_per_unit","density","moment_order")
    missing_metadata=[k for k in metadata_required if k not in moment_metadata]
    if missing_metadata: raise ValueError("moment_metadata missing: "+", ".join(missing_metadata))
    if moment_metadata["moment_order"] != ["V","Lx","Ly","Lz","Qxx","Qyy","Qzz","Qyz","Qxz","Qxy"]:
        raise ValueError("moment_metadata moment_order mismatch")
    for key in ("meters_per_unit","density"):
        value=float(moment_metadata[key])
        if not math.isfinite(value) or value<=0: raise ValueError(f"moment_metadata.{key} must be positive finite")
    fields=dict(m["request_fields"])
    if float(fields.get("material_density",0)) != float(moment_metadata["density"]):
        raise ValueError("request material_density differs from moment_metadata density")
    required_fields=("fixed_shell_moments","target_origin","target_axes","mode","material_density",
      "original_units_per_normalized_unit","minimum_wall_thickness","spinning_top_objective","settings")
    missing_fields=[k for k in required_fields if k not in fields]
    if missing_fields: raise ValueError("request_fields missing: "+", ".join(missing_fields))
    if fields["mode"]!="SPINNING_TOP": raise ValueError("R3 request must use SPINNING_TOP")
    if fields["spinning_top_objective"] not in ("INERTIA_AND_LOWERING","LOWER_CENTER_OF_MASS"):
        raise ValueError("unsupported spinning_top_objective")
    settings=dict(fields["settings"])
    if settings.get("equation_profile")!="DUAL_MONOMIAL_RAW_V1":
        raise ValueError("native full-rank solver requires DUAL_MONOMIAL_RAW_V1")
    if settings.get("capture_evaluation_payload") is not True:
        raise ValueError("capture_evaluation_payload must be true")
    if not isinstance(fields["fixed_shell_moments"],list) or len(fields["fixed_shell_moments"])!=10:
        raise ValueError("fixed_shell_moments must be explicitly supplied as ten values")
    request={"case_id":m["case_id"],"geometry_hash":m["geometry_hash"],"task_hash":m["task_hash"],
      "outer_vertices":outer_v,"outer_faces":outer_f,"design_vertices":design_v,"design_faces":design_f,
      **fields}
    # Validate finite physical scalars and frame dimensions without changing them.
    for key in ("material_density","original_units_per_normalized_unit"):
        value=float(request[key])
        if not math.isfinite(value) or value<=0: raise ValueError(f"{key} must be positive finite")
    if len(request["target_origin"])!=3 or len(request["target_axes"])!=3:
        raise ValueError("target frame must provide origin and three axes")
    return {"schema":"tvcg.r3.native-request.v1","request":request,
      "source_assets":files,"semantic_review":m["semantic_review"],
      "moment_metadata":moment_metadata,
      "request_status":"assembled_unexecuted","native_execution":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    a=p.parse_args(); result=build_request(a.manifest)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"request_status":result["request_status"],"case_id":result["request"]["case_id"],
      "outer_vertices":len(result["request"]["outer_vertices"]),
      "design_vertices":len(result["request"]["design_vertices"]),"output":str(out)},indent=2))
if __name__=="__main__":main()
