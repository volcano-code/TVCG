"""Strict native-vs-independent ten-moment comparison.

No implicit unit conversion is attempted. Native and independent records must
declare identical SI coordinate frame, meters-per-unit and material density.
Comparison uses component-wise atol + rtol because moment components have
different physical dimensions and may be close to zero.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

MOMENT_ORDER=["M","LX","LY","LZ","QXX","QYY","QZZ","QYZ","QXZ","QXY"]

def compare_records(native_bundle, independent_record, *, rtol=1e-8, atol=1e-14):
    ledger=native_bundle.get("ledger",{})
    native=native_bundle.get("native_result",{})
    native_meta=ledger.get("moment_metadata")
    if not isinstance(native_meta,dict):
        raise ValueError("native replay bundle lacks explicit moment_metadata")
    required_meta=("case_id","geometry_hash","coordinate_frame","length_unit","meters_per_unit","density")
    for src,name in ((native_meta,"native moment metadata"),(independent_record,"independent record")):
        missing=[k for k in required_meta if k not in src]
        if missing: raise ValueError(f"{name} missing metadata: "+", ".join(missing))
    if native_meta["case_id"]!=independent_record["case_id"]:
        raise ValueError("case_id mismatch")
    if native_meta["geometry_hash"]!=independent_record.get("source_geometry_hash"):
        raise ValueError("geometry hash mismatch")
    for key in ("coordinate_frame","length_unit"):
        if native_meta[key]!=independent_record[key]:
            raise ValueError(f"{key} mismatch; implicit transform/conversion is forbidden")
    for key in ("meters_per_unit","density"):
        a=float(native_meta[key]); b=float(independent_record[key])
        if not math.isfinite(a) or not math.isfinite(b) or a<=0 or b<=0 or a!=b:
            raise ValueError(f"{key} mismatch; implicit unit/density conversion is forbidden")
    if independent_record.get("moment_order")!=MOMENT_ORDER:
        raise ValueError("independent moment order mismatch")
    if native_meta.get("moment_order")!=MOMENT_ORDER:
        raise ValueError("native moment order metadata mismatch")
    if independent_record.get("backend_identity") in (None,"","native","native_solver"):
        raise ValueError("independent backend identity must be distinct from native solver")
    n=np.asarray(native.get("final_total_raw_moments"),dtype=np.float64)
    i=np.asarray(independent_record.get("raw_moments"),dtype=np.float64)
    if n.shape!=(10,) or i.shape!=(10,) or not np.isfinite(n).all() or not np.isfinite(i).all():
        raise ValueError("both raw moment vectors must be finite [10]")
    rtol=float(rtol); atol=float(atol)
    if not math.isfinite(rtol) or rtol<0 or not math.isfinite(atol) or atol<0:
        raise ValueError("rtol/atol must be finite nonnegative")
    abs_error=np.abs(n-i)
    scale=np.maximum(np.abs(n),np.abs(i))
    allowed=atol+rtol*scale
    passed=abs_error<=allowed
    denom=np.maximum(scale,np.finfo(np.float64).tiny)
    rel=abs_error/denom
    return {
      "schema":"tvcg.r3.native-independent-moment-comparison.v1",
      "case_id":native_meta["case_id"],"geometry_hash":native_meta["geometry_hash"],
      "independent_backend":independent_record["backend_identity"],
      "moment_order":MOMENT_ORDER,"native_raw_moments":n.tolist(),
      "independent_raw_moments":i.tolist(),"absolute_errors":abs_error.tolist(),
      "relative_errors":rel.tolist(),"allowed_errors":allowed.tolist(),
      "component_pass":passed.tolist(),"max_relative_error":float(rel.max()),
      "moment_match":bool(passed.all()),
      "metadata_match":True,"tolerances":{"rtol":rtol,"atol":atol},
      "interpretation":"component-wise abs(error) <= atol + rtol*max(abs(native),abs(independent)); no implicit unit conversion"
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",required=True);p.add_argument("--independent",required=True)
    p.add_argument("--rtol",type=float,default=1e-8);p.add_argument("--atol",type=float,default=1e-14)
    p.add_argument("--output",required=True);a=p.parse_args()
    result=compare_records(json.loads(Path(a.native).read_text()),
      json.loads(Path(a.independent).read_text()),rtol=a.rtol,atol=a.atol)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"case_id":result["case_id"],"moment_match":result["moment_match"],
      "max_relative_error":result["max_relative_error"],"output":str(out)},indent=2))
    raise SystemExit(0 if result["moment_match"] else 3)
if __name__=="__main__":main()
