"""R3.0.6 native Spin-It Faster replay runner.

Loads a JSON request and explicit normalized initial quadrics, calls the existing
Python facade backed by the native pybind C++ solver, and writes a provenance-
bound replay bundle. It does not emulate Newton in Python.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

from make_it_stand.spin import (
    SpinItFasterRequest, SpinSolverConfig, SpinEvaluatorConfig,
    solve_spin_it_faster_fullrank_quadric,
)
from r3_faster_initialization_v1 import FasterInitialization
from r3_solver_event_ledger_v1 import SCHEMA as LEDGER_SCHEMA, validate_ledger
from r3_replay_admission_guard_v1 import validate_replay_admission

def _jsonable(x: Any):
    if dataclasses.is_dataclass(x): return {k:_jsonable(v) for k,v in dataclasses.asdict(x).items()}
    if isinstance(x,dict): return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if hasattr(x,"value"): return x.value
    return x

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _request(payload: dict) -> SpinItFasterRequest:
    d=dict(payload)
    for metadata_key in ("case_id","geometry_hash","task_hash"):
        d.pop(metadata_key, None)
    settings=dict(d.get("settings",{}))
    d["settings"]=SpinSolverConfig(**settings)
    return SpinItFasterRequest(**d)

def validate_native_replay_inputs(request_payload, init, evaluator_payload):
    if init.equation_profile != "dual_monomial_raw_v1":
        raise ValueError("native full-rank replay requires dual_monomial_raw_v1 profile")
    if init.mode != "spinning_top":
        raise ValueError("R3 replay currently admits only spinning_top mode")
    for key, expected in (("geometry_hash", init.geometry_hash), ("task_hash", init.task_hash)):
        if request_payload.get(key) != expected:
            raise ValueError(f"request {key} does not match initialization provenance")
    if not request_payload.get("case_id"):
        raise ValueError("request must bind a nonempty case_id")
    request = _request(request_payload)
    if request.mode != "SPINNING_TOP":
        raise ValueError("request mode must be SPINNING_TOP")
    if request.settings.equation_profile != "DUAL_MONOMIAL_RAW_V1":
        raise ValueError("request must use DUAL_MONOMIAL_RAW_V1")
    if not request.settings.capture_evaluation_payload:
        raise ValueError("capture_evaluation_payload must be true for R3 replay")
    checks = {
        "maximum_function_evaluations_per_restart": init.maximum_function_evaluations_per_restart,
        "equation_tolerance": init.equation_tolerance,
        "physical_relative_tolerance": init.physical_relative_tolerance,
        "deterministic_seed": init.deterministic_seed,
        "maximum_iterations": init.maximum_iterations,
        "maximum_restarts": init.maximum_restarts,
    }
    for field, expected in checks.items():
        actual = getattr(request.settings, field)
        if actual != expected:
            raise ValueError(f"request {field} differs from initialization record")
    evaluator_identity = evaluator_payload.get("evaluator_identity")
    evaluator_settings_payload = evaluator_payload.get("settings", evaluator_payload)
    if evaluator_identity is not None and evaluator_identity != init.evaluator_identity:
        raise ValueError("evaluator identity differs from initialization record")
    evaluator = SpinEvaluatorConfig(**evaluator_settings_payload)
    return request, evaluator

def run_replay(request_path, initialization_path, evaluator_path, output_path, admission_path):
    request_path,init_path=Path(request_path),Path(initialization_path)
    request_envelope=json.loads(request_path.read_text())
    if request_envelope.get("schema")=="tvcg.r3.native-request.v1":
        request_payload=request_envelope.get("request")
        if not isinstance(request_payload,dict): raise ValueError("native request envelope has no request object")
    else:
        request_payload=request_envelope
    init_payload=json.loads(init_path.read_text())
    init=FasterInitialization.from_dict(init_payload)
    evaluator_payload=json.loads(Path(evaluator_path).read_text())
    admission_payload=json.loads(Path(admission_path).read_text())
    admission_validation=validate_replay_admission(request_envelope,init,admission_payload,evaluator_path)
    request, evaluator = validate_native_replay_inputs(request_payload, init, evaluator_payload)
    initial=[list(x) for x in init.initial_guesses]
    result=solve_spin_it_faster_fullrank_quadric(request,initial_quadrics=initial,evaluator_settings=evaluator)
    raw=_jsonable(result)
    events=raw["events"]
    for e in events:
        e["type"]=str(e["type"]).lower()
        e["outcome"]=str(e.get("outcome","")).lower()
    terminal=str(raw["status"]).lower()
    if terminal in ("converged","success"): terminal="converged"
    ledger={
      "schema":LEDGER_SCHEMA,"case_id":str(request_payload.get("case_id","")),
      "geometry_hash":init.geometry_hash,"task_hash":init.task_hash,
      "evaluator_identity":init.evaluator_identity,"solver_source_commit":init.solver_source_commit,
      "initialization_record_hash":_sha(init_path),"equation_profile":init.equation_profile,
      "reported_function_evaluations":int(raw["function_evaluations"]),
      "reported_restart_function_evaluations":list(raw["restart_function_evaluations"]),
      "terminal_status":terminal,"events":events,
      "native_event_hash":raw["event_hash"],"native_trace_hash":raw["trace_hash"],
      "native_input_hash":raw["input_hash"],"native_output_hash":raw["output_hash"],
      "native_status":raw["status"],"native_diagnostic":raw["diagnostic"],
      "native_restart_count":raw["restart_count"],"trace":raw["trace"],
      "evaluation_timings":raw["evaluation_timings"],
      "final_geometry_verified":raw["final_geometry_verified"],
      "verification_diagnostic":raw["verification_diagnostic"],
      "independent_raw_residual":raw["independent_raw_residual"],
      "final_total_raw_moments":raw["final_total_raw_moments"],
      "independent_total_raw_moments":raw["independent_total_raw_moments"],
      "final_geometry_hash":raw["final_geometry_hash"],
      "quadric_coefficients":raw["quadric_coefficients"],
      "request_sha256":_sha(request_path),"request_envelope_schema":request_envelope.get("schema"),
      "physical_admission":admission_validation,"initialization_record_sha256":_sha(init_path),
      "evaluator_config_sha256":_sha(Path(evaluator_path)),
    }
    validation=validate_ledger(ledger)
    bundle={"schema":"tvcg.r3.native-replay-bundle.v1","ledger":ledger,
            "ledger_validation":validation,"native_result":raw}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(bundle,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return {"output":str(out),"bundle_sha256":_sha(out),"ledger_validation":validation,
            "native_status":raw["status"],"final_geometry_verified":raw["final_geometry_verified"],
            "function_evaluations":raw["function_evaluations"],"restart_count":raw["restart_count"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True);p.add_argument("--initialization",required=True)
    p.add_argument("--evaluator",required=True);p.add_argument("--admission",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(run_replay(a.request,a.initialization,a.evaluator,a.output,a.admission),indent=2))
if __name__=="__main__":main()
