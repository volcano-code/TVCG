"""R3.1.4 fail-closed gate for physically admitted native exact replay inputs.

A metadata-only admission report is insufficient. The report must carry native
geometry evidence and an explicit physical_mesh_admission=true result. This module
validates provenance links; it does not itself certify geometry or run the solver.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def validate_replay_admission(request_envelope: dict[str,Any], initialization: Any,
                              admission: dict[str,Any], evaluator_path: str | Path) -> dict:
    if request_envelope.get("schema") != "tvcg.r3.native-request.v1":
        raise ValueError("native replay requires a provenance-bound native-request envelope")
    if request_envelope.get("request_status") != "assembled_unexecuted":
        raise ValueError("native request must be assembled_unexecuted before replay")
    request=request_envelope.get("request")
    if not isinstance(request,dict):
        raise ValueError("native request envelope lacks request object")
    if admission.get("schema") != "tvcg.r3.physical-admission.v1":
        raise ValueError("missing native physical-admission certificate")
    if admission.get("status") != "physically_admitted" or admission.get("physical_mesh_admission") is not True:
        raise ValueError("metadata-only admission cannot authorize native exact replay")
    required=("case_id","geometry_hash","task_hash","task_record_sha256","outer_mesh_sha256",
              "design_mesh_sha256","pin_support_sha256","native_geometry_evidence_sha256",
              "reviewer","review_source")
    missing=[k for k in required if not admission.get(k)]
    if missing: raise ValueError("physical-admission certificate missing: "+", ".join(missing))
    if request.get("case_id") != admission["case_id"]:
        raise ValueError("physical-admission case_id mismatch")
    if request.get("geometry_hash") != admission["geometry_hash"]:
        raise ValueError("physical-admission geometry hash mismatch")
    if request.get("task_hash") != admission["task_hash"] or initialization.task_hash != admission["task_hash"]:
        raise ValueError("physical-admission TaskSpec hash mismatch")
    if initialization.geometry_hash != admission["geometry_hash"]:
        raise ValueError("initialization geometry hash mismatch")
    assets=request_envelope.get("source_assets")
    if not isinstance(assets,dict):
        raise ValueError("request source_assets provenance is missing")
    for key,expected in (("outer_mesh_file",admission["outer_mesh_sha256"]),
                         ("design_mesh_file",admission["design_mesh_sha256"]),
                         ("pin_support_file",admission["pin_support_sha256"]),
                         ("task_record",admission["task_record_sha256"])):
        record=assets.get(key)
        if not isinstance(record,dict) or record.get("sha256") != expected:
            raise ValueError(f"{key} hash differs from physical-admission certificate")
        asset_path=record.get("path")
        if not isinstance(asset_path,str) or not Path(asset_path).is_file():
            raise FileNotFoundError(f"request asset missing: {asset_path}")
        if sha256_file(asset_path) != expected:
            raise ValueError(f"{key} file content changed after request assembly")
    evaluator_path=Path(evaluator_path)
    if not evaluator_path.is_file():
        raise FileNotFoundError(evaluator_path)
    evaluator_hash=sha256_file(evaluator_path)
    if admission.get("evaluator_config_sha256") != evaluator_hash:
        raise ValueError("evaluator config hash differs from physical-admission certificate")
    return {
      "authorized":True,"case_id":admission["case_id"],
      "geometry_hash":admission["geometry_hash"],"task_hash":admission["task_hash"],
      "evaluator_config_sha256":evaluator_hash,
      "native_geometry_evidence_sha256":admission["native_geometry_evidence_sha256"],
      "reviewer":admission["reviewer"],"review_source":admission["review_source"],
    }
