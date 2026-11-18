"""R3.0.4 Faster numerical initialization and solver-protocol provenance.

This records the actual TVCG solver inputs without claiming equivalence to the
unreleased author Matlab implementation. Initial guesses are numerical states,
not original physical objects. The adapter is intentionally serialization-only:
it never invents seeds, guesses, or solver settings.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math
from typing import Any

SCHEMA = "tvcg.r3.faster-initialization.v1"
PROFILES = {"legacy_native_normalized_v1", "dual_monomial_raw_v1"}
MODES = {"static_balance", "buoyant_stability", "fidget_spinner", "yoyo", "spinning_top"}

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _vector10(value,name):
    if not isinstance(value,(list,tuple)) or len(value)!=10:
        raise ValueError(f"{name} must contain exactly ten coefficients")
    v=tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in v): raise ValueError(f"{name} must be finite")
    n=math.sqrt(sum(x*x for x in v))
    if abs(n-1.0)>1e-12: raise ValueError(f"{name} must have unit norm; got {n}")
    return v

@dataclass(frozen=True)
class FasterInitialization:
    mode: str
    equation_profile: str
    deterministic_seed: int
    maximum_iterations: int
    maximum_restarts: int
    maximum_function_evaluations_per_restart: int
    equation_tolerance: float
    physical_relative_tolerance: float
    line_search_reduction: float
    maximum_geodesic_step: float
    initial_guesses: tuple[tuple[float,...],...]
    initialization_source: str
    solver_source_commit: str
    evaluator_identity: str
    geometry_hash: str
    task_hash: str

    @classmethod
    def from_dict(cls,d:dict[str,Any]):
        if d.get("schema")!=SCHEMA: raise ValueError("unsupported Faster initialization schema")
        required=("mode","equation_profile","deterministic_seed","maximum_iterations",
          "maximum_restarts","maximum_function_evaluations_per_restart","equation_tolerance",
          "physical_relative_tolerance","line_search_reduction","maximum_geodesic_step",
          "initial_guesses","initialization_source","solver_source_commit","evaluator_identity",
          "geometry_hash","task_hash")
        missing=[k for k in required if k not in d]
        if missing: raise ValueError("incomplete Faster initialization: "+", ".join(missing))
        if d["mode"] not in MODES: raise ValueError("unsupported solver mode")
        if d["equation_profile"] not in PROFILES: raise ValueError("unsupported equation profile")
        if type(d["deterministic_seed"]) is not int or d["deterministic_seed"]<0:
            raise ValueError("deterministic_seed must be a nonnegative integer")
        for k in ("maximum_iterations","maximum_restarts","maximum_function_evaluations_per_restart"):
            if type(d[k]) is not int or d[k]<=0: raise ValueError(f"{k} must be positive integer")
        if d["equation_profile"]=="dual_monomial_raw_v1" and d["maximum_function_evaluations_per_restart"]>60:
            raise ValueError("strict dual profile permits at most 60 function evaluations per restart")
        for k in ("equation_tolerance","physical_relative_tolerance"):
            if not math.isfinite(float(d[k])) or float(d[k])<=0: raise ValueError(f"{k} must be positive finite")
        if float(d["line_search_reduction"])!=0.5 or float(d["maximum_geodesic_step"])!=0.5:
            raise ValueError("paper protocol locks line_search_reduction and maximum_geodesic_step to 0.5")
        guesses=d["initial_guesses"]
        if not isinstance(guesses,list) or not guesses: raise ValueError("initial_guesses must be a nonempty list")
        parsed=tuple(_vector10(x,f"initial_guesses[{i}]") for i,x in enumerate(guesses))
        if len(parsed)>d["maximum_restarts"]: raise ValueError("initial guess count exceeds maximum_restarts")
        for k in ("initialization_source","solver_source_commit","evaluator_identity","geometry_hash","task_hash"):
            if not isinstance(d[k],str) or not d[k].strip(): raise ValueError(f"{k} must be explicit")
        return cls(d["mode"],d["equation_profile"],d["deterministic_seed"],
          d["maximum_iterations"],d["maximum_restarts"],d["maximum_function_evaluations_per_restart"],
          float(d["equation_tolerance"]),float(d["physical_relative_tolerance"]),
          float(d["line_search_reduction"]),float(d["maximum_geodesic_step"]),parsed,
          d["initialization_source"],d["solver_source_commit"],d["evaluator_identity"],
          d["geometry_hash"],d["task_hash"])

    def to_dict(self):
        return {"schema":SCHEMA,"mode":self.mode,"equation_profile":self.equation_profile,
          "deterministic_seed":self.deterministic_seed,"maximum_iterations":self.maximum_iterations,
          "maximum_restarts":self.maximum_restarts,
          "maximum_function_evaluations_per_restart":self.maximum_function_evaluations_per_restart,
          "equation_tolerance":self.equation_tolerance,
          "physical_relative_tolerance":self.physical_relative_tolerance,
          "line_search_reduction":self.line_search_reduction,
          "maximum_geodesic_step":self.maximum_geodesic_step,
          "initial_guesses":[list(x) for x in self.initial_guesses],
          "initialization_source":self.initialization_source,
          "solver_source_commit":self.solver_source_commit,"evaluator_identity":self.evaluator_identity,
          "geometry_hash":self.geometry_hash,"task_hash":self.task_hash}

def load_initialization(path):
    d=json.loads(Path(path).read_text())
    if not isinstance(d,dict): raise ValueError("initialization record must be a JSON object")
    return FasterInitialization.from_dict(d)

def write_initialization(record,path):
    parsed=FasterInitialization.from_dict(record)
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(parsed.to_dict(),indent=2,sort_keys=True)+"\n")
    return _sha(p)
