"""R3.1.5.5 double-replay evidence auditor.

This command audits two independently executed native replay bundles and their
independent moment, physical-verification and geometry-integrity reports. It does
not execute the solver. Both replay bundles must already be produced by the native
runner and accepted by the native result gate.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from r3_native_independent_moment_compare_v1 import compare_records
from pathlib import Path
from typing import Any

SCHEMA="tvcg.r3.double-replay-evidence.v1"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def _read(path):
    if isinstance(path,dict):
        return path
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    return json.loads(p.read_text())

def _evidence_digest(value):
    if isinstance(value,dict):
        raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()
    return sha256_file(value)

def _require(d,keys,name):
    missing=[k for k in keys if k not in d]
    if missing:raise ValueError(f"{name} missing fields: "+", ".join(missing))

def _signature(bundle):
    _require(bundle,("schema","ledger","native_result","ledger_validation"),"replay bundle")
    if bundle["schema"]!="tvcg.r3.native-replay-bundle.v1":
        raise ValueError("unsupported native replay bundle schema")
    ledger=bundle["ledger"];native=bundle["native_result"]
    _require(ledger,("case_id","geometry_hash","task_hash","evaluator_identity",
      "solver_source_commit","initialization_record_hash","equation_profile","moment_metadata",
      "request_sha256","evaluator_config_sha256",
      "physical_admission","result_acceptance","run_receipt","request_sha256","evaluator_config_sha256"),"native ledger")
    _require(native,("status","event_hash","trace_hash","input_hash","output_hash",
      "function_evaluations","restart_count","restart_function_evaluations",
      "quadric_coefficients","final_total_raw_moments","final_geometry_hash",
      "final_geometry_verified"),"native result")
    if ledger["result_acceptance"].get("physically_accepted") is not True:
        raise ValueError("native replay is not physically accepted")
    if native["final_geometry_verified"] is not True:
        raise ValueError("native final geometry verification failed")
    if str(native["status"]).upper() not in ("CONVERGED","SUCCESS"):
        raise ValueError("native solver did not converge")
    return {
      "case_id":ledger["case_id"],"geometry_hash":ledger["geometry_hash"],
      "task_hash":ledger["task_hash"],"evaluator_identity":ledger["evaluator_identity"],
      "solver_source_commit":ledger["solver_source_commit"],
      "initialization_record_hash":ledger["initialization_record_hash"],
      "equation_profile":ledger["equation_profile"],"moment_metadata":ledger["moment_metadata"],
      "physical_admission":ledger["physical_admission"],
      "request_sha256":ledger["request_sha256"],"evaluator_config_sha256":ledger["evaluator_config_sha256"],
      "run_receipt":ledger["run_receipt"],
      "event_hash":native["event_hash"],"trace_hash":native["trace_hash"],
      "input_hash":native["input_hash"],"output_hash":native["output_hash"],
      "status":native["status"],"function_evaluations":native["function_evaluations"],
      "restart_count":native["restart_count"],
      "restart_function_evaluations":native["restart_function_evaluations"],
      "quadric_coefficients":native["quadric_coefficients"],
      "raw_mass_moments":native["final_total_raw_moments"],
      "final_geometry_hash":native["final_geometry_hash"],
    }

def _same(a,b,label):
    if a!=b:raise ValueError(f"Replay A/B mismatch: {label}")

def audit_double_replay(paths:dict[str,str])->dict[str,Any]:
    required=("replay_a","replay_b","independent_a","independent_b",
      "physical_a","physical_b","geometry_a","geometry_b")
    _require(paths,required,"input paths")
    loaded={k:_read(v) for k,v in paths.items()}
    a=_signature(loaded["replay_a"]);b=_signature(loaded["replay_b"])
    receipt_a=a["run_receipt"];receipt_b=b["run_receipt"]
    if not receipt_a.get("run_id") or not receipt_b.get("run_id") or receipt_a["run_id"]==receipt_b["run_id"]:
        raise ValueError("Replay A/B must have distinct nonempty execution run_id receipts")
    invariant=("case_id","geometry_hash","task_hash","evaluator_identity",
      "solver_source_commit","initialization_record_hash","equation_profile","moment_metadata",
      "physical_admission","input_hash","request_sha256","evaluator_config_sha256")
    for key in invariant:_same(a[key],b[key],key)
    scientific=("event_hash","trace_hash","output_hash","status","function_evaluations",
      "restart_count","restart_function_evaluations","quadric_coefficients",
      "raw_mass_moments","final_geometry_hash")
    replay_mismatches=[k for k in scientific if a[k]!=b[k]]
    for label in ("a","b"):
        indep=loaded[f"independent_{label}"]
        phys=loaded[f"physical_{label}"]
        geom=loaded[f"geometry_{label}"]
        _require(indep,("case_id","task_hash","source_geometry_hash","backend_identity",
          "backend_sha256","raw_moments","moment_order"),f"independent {label}")
        _require(phys,("case_id","task_hash","geometry_hash","overall_pass",
          "hard_constraints_pass","tolerance_source_reference"),f"physical {label}")
        _require(geom,("case_id","task_hash","geometry_hash","geometry_integrity_pass",
          "native_independent_equivalence"),f"geometry {label}")
        _same(indep["case_id"],a["case_id"],f"independent {label} case")
        _same(indep["task_hash"],a["task_hash"],f"independent {label} task")
        _same(indep["source_geometry_hash"],a["geometry_hash"],f"independent {label} geometry")
        _same(phys["case_id"],a["case_id"],f"physical {label} case")
        _same(phys["task_hash"],a["task_hash"],f"physical {label} task")
        _same(phys["geometry_hash"],a["geometry_hash"],f"physical {label} geometry")
        _same(geom["case_id"],a["case_id"],f"geometry {label} case")
        _same(geom["task_hash"],a["task_hash"],f"geometry {label} task")
        _same(geom["geometry_hash"],a["geometry_hash"],f"geometry {label} geometry")
        if phys["overall_pass"] is not True or phys["hard_constraints_pass"] is not True:
            raise ValueError(f"independent physical verification {label} failed")
        if geom["geometry_integrity_pass"] is not True:
            raise ValueError(f"geometry integrity {label} failed")
        if geom["native_independent_equivalence"].get("reviewed_equivalent") is not True:
            raise ValueError(f"geometry equivalence {label} not reviewed")
        if indep["backend_identity"] in ("native","native_solver",""):
            raise ValueError("independent backend identity is not independent")
        comparison=compare_records(loaded[f"replay_{label}"],indep)
        if comparison["moment_match"] is not True:
            raise ValueError(f"native/independent moment comparison {label} failed")
    # The two independent records must agree as well, but must be separately generated and traceable.
    ia,ib=loaded["independent_a"],loaded["independent_b"]
    independent_match=ia["raw_moments"]==ib["raw_moments"]
    independent_backend_match=(ia["backend_identity"]==ib["backend_identity"] and ia["backend_sha256"]==ib["backend_sha256"])
    evidence_hashes={k:_evidence_digest(v) for k,v in paths.items()}
    return {"schema":SCHEMA,"case_id":a["case_id"],"geometry_hash":a["geometry_hash"],
      "task_hash":a["task_hash"],"invariant_inputs_match":True,
      "native_replay_scientific_match":not replay_mismatches,
      "native_replay_mismatches":replay_mismatches,
      "independent_records_moment_match":independent_match,
      "distinct_native_execution_ids":True,
      "independent_backend_same_version":independent_backend_match,
      "all_physical_checks_pass":True,"all_geometry_checks_pass":True,
      "double_replay_pass":not replay_mismatches and independent_match and independent_backend_match,
      "timing_excluded":True,"evidence_file_sha256":evidence_hashes,
      "limitations":["two replay bundles must originate from separate native executions",
        "independent tetrahedralization must be independently generated or deterministically reproduced",
        "reviewed mapping certificate is evidence, not an automatic proof of non-overlap",
        "CI fixtures do not constitute a real M69 replay"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("replay-a","replay-b","independent-a","independent-b",
                 "physical-a","physical-b","geometry-a","geometry-b"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    paths={k:getattr(a,k.replace("-","_")) for k in
      ("replay_a","replay_b","independent_a","independent_b","physical_a","physical_b","geometry_a","geometry_b")}
    report=audit_double_replay(paths)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"case_id":report["case_id"],"double_replay_pass":report["double_replay_pass"],
      "native_replay_scientific_match":report["native_replay_scientific_match"],
      "output":str(out)},indent=2))
    raise SystemExit(0 if report["double_replay_pass"] else 3)
if __name__=="__main__":main()
