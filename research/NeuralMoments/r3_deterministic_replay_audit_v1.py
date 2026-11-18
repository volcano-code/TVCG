"""R3.0.7 deterministic replay comparison and independent raw-moment audit.

Independent verification consumes a separately produced 10-component raw-moment
record. It does not reuse the native solver's internal independent-residual field.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

SCHEMA = "tvcg.r3.deterministic-replay-audit.v1"
MOMENT_ORDER = ("v","l1","l2","l3","Q11","Q22","Q33","Q23","Q13","Q12")

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _finite_vector(x,n,name):
    if not isinstance(x,(list,tuple)) or len(x)!=n: raise ValueError(f"{name} must have {n} values")
    y=[float(v) for v in x]
    if not all(math.isfinite(v) for v in y): raise ValueError(f"{name} contains non-finite values")
    return y

def _native_signature(bundle):
    ledger=bundle["ledger"]; native=bundle["native_result"]
    return {
      "case_id":ledger["case_id"],"geometry_hash":ledger["geometry_hash"],
      "task_hash":ledger["task_hash"],"evaluator_identity":ledger["evaluator_identity"],
      "solver_source_commit":ledger["solver_source_commit"],
      "initialization_record_hash":ledger["initialization_record_hash"],
      "equation_profile":ledger["equation_profile"],
      "native_event_hash":native["event_hash"],"native_trace_hash":native["trace_hash"],
      "native_input_hash":native["input_hash"],"native_output_hash":native["output_hash"],
      "status":native["status"],"function_evaluations":native["function_evaluations"],
      "restart_function_evaluations":native["restart_function_evaluations"],
      "quadric_coefficients":native["quadric_coefficients"],
    }

def compare_replays(first,second):
    a,b=_native_signature(first),_native_signature(second)
    mismatches=[k for k in a if a[k]!=b[k]]
    return {"schema":SCHEMA,"deterministic":not mismatches,"mismatches":mismatches,
      "first":a,"second":b,
      "timing_excluded":True,
      "note":"Only scientific/native identity fields are compared; timing sidecars may differ."}

def derive_spinning_top_physics(raw_moments,density):
    v,l1,l2,l3,q11,q22,q33,q23,q13,q12=_finite_vector(raw_moments,10,"raw_moments")
    density=float(density)
    if not math.isfinite(density) or density<=0: raise ValueError("density must be positive finite")
    if v<=0: raise ValueError("raw volume moment v must be positive")
    c=[l1/v,l2/v,l3/v]
    mass=density*v
    q=[[q11,q12,q13],[q12,q22,q23],[q13,q23,q33]]
    trace=q11+q22+q33
    inertia_origin=[[density*((trace if i==j else 0.0)-q[i][j]) for j in range(3)] for i in range(3)]
    c2=sum(x*x for x in c)
    inertia_center=[[inertia_origin[i][j]-mass*((c2 if i==j else 0.0)-c[i]*c[j])
                      for j in range(3)] for i in range(3)]
    scale=max(1.0,abs(density)*max(abs(x) for x in raw_moments))
    constraints={"l1":density*l1/scale,"l2":density*l2/scale,
      "Ic13":inertia_center[0][2]/max(1.0,abs(inertia_center[0][0]),abs(inertia_center[2][2])),
      "Ic23":inertia_center[1][2]/max(1.0,abs(inertia_center[1][1]),abs(inertia_center[2][2]))}
    return {"mass":mass,"center_of_mass":c,"inertia_center":inertia_center,
            "normalized_spinning_top_constraints":constraints,
            "constraint_l2":math.sqrt(sum(x*x for x in constraints.values()))}

def verify_independent_moments(native_raw_moments,independent_record,*,density,
                               moment_relative_tolerance=1e-8,
                               physical_constraint_tolerance=1e-6,
                               expected_geometry_hash=None):
    independent=_finite_vector(independent_record["raw_moments"],10,"independent raw_moments")
    native=_finite_vector(native_raw_moments,10,"native raw_moments")
    if independent_record.get("moment_order")!=list(MOMENT_ORDER):
        raise ValueError("independent record moment_order does not match canonical order")
    if not independent_record.get("backend_identity") or not independent_record.get("source_geometry_hash"):
        raise ValueError("independent record must bind backend_identity and source_geometry_hash")
    if expected_geometry_hash is not None and independent_record["source_geometry_hash"] != expected_geometry_hash:
        raise ValueError("independent moment geometry hash does not match replay input geometry")
    diffs=[abs(a-b) for a,b in zip(native,independent)]
    rel=[d/max(1.0,abs(a),abs(b)) for d,a,b in zip(diffs,native,independent)]
    physics=derive_spinning_top_physics(independent,density)
    return {"independent_backend":independent_record["backend_identity"],
      "source_geometry_hash":independent_record["source_geometry_hash"],
      "moment_absolute_errors":diffs,"moment_relative_errors":rel,
      "moment_max_relative_error":max(rel),
      "moment_pass":max(rel)<=moment_relative_tolerance,
      "physical":physics,
      "physical_constraint_pass":physics["constraint_l2"]<=physical_constraint_tolerance,
      "tolerances":{"moment_relative":moment_relative_tolerance,
                    "physical_constraint":physical_constraint_tolerance}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("compare"); c.add_argument("--first",required=True); c.add_argument("--second",required=True); c.add_argument("--output",required=True)
    v=sub.add_parser("verify-moments"); v.add_argument("--native",required=True); v.add_argument("--independent",required=True); v.add_argument("--density",type=float,required=True); v.add_argument("--output",required=True)
    a=p.parse_args()
    if a.command=="compare":
        result=compare_replays(json.loads(Path(a.first).read_text()),json.loads(Path(a.second).read_text()))
    else:
        n=json.loads(Path(a.native).read_text()); i=json.loads(Path(a.independent).read_text())
        raw=n.get("native_result",n).get("final_total_raw_moments")
        if raw is None: raise ValueError("native replay bundle lacks final_total_raw_moments")
        expected=n.get("ledger",{}).get("geometry_hash")
        result=verify_independent_moments(raw,i,density=a.density,expected_geometry_hash=expected)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
