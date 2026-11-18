"""R3.0.5 deterministic solver event-ledger capture and validation."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

SCHEMA = "tvcg.r3.solver-event-ledger.v1"
TIMING_KEYS = {"wall_seconds","evaluator_wall_seconds","residual_evaluation_seconds",
               "derivative_evaluation_seconds","created_at","timestamp"}

def _canonical(value: Any):
    if isinstance(value,dict):
        return {k:_canonical(v) for k,v in sorted(value.items()) if k.lower() not in TIMING_KEYS and not k.lower().endswith("_seconds")}
    if isinstance(value,list): return [_canonical(v) for v in value]
    if isinstance(value,float):
        if not math.isfinite(value): raise ValueError("non-finite number in scientific event payload")
        return value
    if value is None or isinstance(value,(str,int,bool)): return value
    raise TypeError(f"unsupported ledger value: {type(value).__name__}")

def scientific_hash(payload:dict) -> str:
    raw=json.dumps(_canonical(payload),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

def validate_ledger(payload:dict) -> dict:
    if payload.get("schema")!=SCHEMA: raise ValueError("unsupported solver event ledger schema")
    required=("case_id","geometry_hash","task_hash","evaluator_identity","solver_source_commit",
      "initialization_record_hash","equation_profile","events","reported_function_evaluations",
      "reported_restart_function_evaluations","terminal_status")
    for key in required:
        if key not in payload: raise ValueError(f"missing ledger field: {key}")
    events=payload["events"]
    if not isinstance(events,list) or not events: raise ValueError("events must be a nonempty list")
    starts=[e for e in events if e.get("type")=="attempt_start"]
    terminals=[e for e in events if e.get("type")=="attempt_terminal"]
    evaluations=[e for e in events if e.get("type")=="evaluation"]
    if not starts or len(starts)!=len(terminals): raise ValueError("each restart must have one start and terminal event")
    expected=list(range(len(starts)))
    if [e.get("restart") for e in starts]!=expected or [e.get("restart") for e in terminals]!=expected:
        raise ValueError("restart ordinals must be unique, ordered, contiguous, and zero-based")
    per_restart=[]
    for rid in expected:
        count=sum(e.get("restart")==rid for e in evaluations); per_restart.append(count)
        start=next(e for e in starts if e.get("restart")==rid)
        end=next(e for e in terminals if e.get("restart")==rid)
        if start.get("evaluation_index",0)>end.get("evaluation_index",0):
            raise ValueError(f"restart {rid}: terminal precedes start")
        if end.get("outcome")=="converged" and rid!=expected[-1]:
            raise ValueError("restart ledger continues after convergence")
    if len(evaluations)!=int(payload["reported_function_evaluations"]):
        raise ValueError("evaluation event count != reported function evaluations")
    if per_restart!=payload["reported_restart_function_evaluations"]:
        raise ValueError("per-restart evaluation counts do not match report")
    if payload.get("terminal_status")=="converged" and terminals[-1].get("outcome")!="converged":
        raise ValueError("terminal status says converged but final restart is not converged")
    for k in ("case_id","geometry_hash","task_hash","evaluator_identity","solver_source_commit",
              "initialization_record_hash","equation_profile"):
        if not isinstance(payload[k],str) or not payload[k].strip(): raise ValueError(f"{k} must be nonempty")
    return {"valid":True,"restart_count":len(starts),"evaluation_count":len(evaluations),
            "restart_evaluations":per_restart,"scientific_hash":scientific_hash(payload)}

def write_ledger(payload,path):
    result=validate_ledger(payload); out=dict(payload); out["validation"]=result
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); payload=json.loads(Path(a.input).read_text())
    print(json.dumps(write_ledger(payload,a.output),indent=2))
if __name__=="__main__": main()
