"""Verify old-map/reconstruction controls independently of the registered new-map population."""
import argparse
import gzip
from pathlib import Path
import json

import numpy as np

from archive import HERE, NATIVE_EVIDENCE, sha, tally


def main(args):
    old=json.loads((NATIVE_EVIDENCE/"native_replay.json").read_bytes())["alternative_chart_diagnostic"]
    rows={p.stem:json.loads(p.read_bytes()) for p in (args.control/"rows").glob("*.json")}
    compared=[]
    for ref in old["rows"]:
        r=rows[ref["probe_id"]]
        if (r["context"]!=old["alternative_context"] or r["base_algebraic_finite_map"]
                or not np.array_equal(r["raw_basis"],ref["chart_basis_raw"])
                or not np.array_equal(r["query"],ref["query"])
                or not np.array_equal(r["material"]["moments"],ref["M"])
                or not np.array_equal(r["material"]["raw_restricted_J"],ref["chart_J"])):
            raise ValueError("archived external control does not reproduce its original source/target")
        radius=[s["h"] for s in r["steps"]]
        compared.append(dict(row_id=r["row_id"], query_raw_norm=float(np.linalg.norm(r["query"])),
                             original_M_identical=True,original_J_identical=True,
                             steps=radius,registered_base_moments=r["material"]["moments"],
                             both_step_reports=tally([r])))
    timings=json.loads(args.timing.read_bytes())
    if timings["reference_max_abs"]!=0 or not all(timings[k] for k in
                ("same_certified_moments","same_target_vertices","same_oriented_faces")):
        raise ValueError("large combB frozen control mismatch")
    args.evidence.mkdir(parents=True,exist_ok=True)
    artifact=args.evidence/"control_replay.json.gz"
    packed=json.dumps(dict(scope="Historical alternative map control only",rows=list(rows.values())),
                      ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    artifact.write_bytes(gzip.compress(packed,mtime=0))
    result=dict(scope="Controls outside the versioned 89-row population",
                archive=artifact.name,archive_sha256=sha(artifact),archived_nonunit_chart_controls=compared,
                registered_large_mesh_full_rank_timing=timings)
    (args.evidence/"control_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--timing",type=Path,required=True)
    p.add_argument("--evidence",type=Path,default=HERE/"evidence")
    main(p.parse_args())
