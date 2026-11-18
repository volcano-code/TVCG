"""R3.1.4 native solver result acceptance gate.

A replay bundle is emitted only for a structurally valid native result. Solver
non-convergence is preserved as a diagnostic artifact, never promoted to success.
Physical feasibility requires the native geometry verifier and finite final fields.
"""
from __future__ import annotations
import math
from typing import Any

def _finite_tree(value: Any, path="result"):
    if isinstance(value,dict):
        for k,v in value.items(): _finite_tree(v,f"{path}.{k}")
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): _finite_tree(v,f"{path}[{i}]")
    elif isinstance(value,float) and not math.isfinite(value):
        raise ValueError(f"non-finite native result at {path}")

def validate_native_result(raw:dict[str,Any]) -> dict:
    required=("status","diagnostic","function_evaluations","restart_count",
      "restart_function_evaluations","events","trace","event_hash","trace_hash",
      "input_hash","output_hash","final_geometry_verified","verification_diagnostic",
      "final_total_raw_moments","quadric_coefficients","final_geometry_hash")
    missing=[k for k in required if k not in raw]
    if missing: raise ValueError("native result missing fields: "+", ".join(missing))
    _finite_tree(raw)
    if int(raw["function_evaluations"])<0 or int(raw["restart_count"])<0:
        raise ValueError("native evaluation/restart counts cannot be negative")
    if len(raw["restart_function_evaluations"])!=int(raw["restart_count"]):
        raise ValueError("restart count does not match per-restart evaluation ledger")
    if sum(int(x) for x in raw["restart_function_evaluations"])!=int(raw["function_evaluations"]):
        raise ValueError("per-restart function evaluations do not sum to total")
    if len(raw["final_total_raw_moments"])!=10:
        raise ValueError("final_total_raw_moments must contain ten values")
    if len(raw["quadric_coefficients"])!=10:
        raise ValueError("final quadric must contain ten coefficients")
    q=[float(x) for x in raw["quadric_coefficients"]]
    norm=math.sqrt(sum(x*x for x in q))
    if abs(norm-1.0)>1e-8:
        raise ValueError(f"final quadric must have unit norm; got {norm}")
    if not raw["final_geometry_hash"]:
        raise ValueError("final geometry hash is empty")
    status=str(raw["status"]).upper()
    converged=status in ("CONVERGED","SUCCESS")
    geometry_ok=raw["final_geometry_verified"] is True
    return {
      "result_structurally_valid":True,
      "solver_converged":converged,
      "native_geometry_verified":geometry_ok,
      "physically_accepted":converged and geometry_ok,
      "status":raw["status"],
      "function_evaluations":int(raw["function_evaluations"]),
      "restart_count":int(raw["restart_count"]),
      "verification_diagnostic":raw["verification_diagnostic"],
      "acceptance_rule":"solver convergence AND native final geometry verification",
    }
