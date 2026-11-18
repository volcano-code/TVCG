"""Audit real M69 corpus records for R3.0.2 admission readiness."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

SCHEMA = "tvcg.r3.m69-corpus-audit.v1"
REQUIRED_ASSETS = ("case_file", "geometry_file", "labels_file", "label_audit_file")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _resolve(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    rel = value[6:] if value.startswith("files/") else value
    return root / "files" / rel

def audit_corpus(manifest_path: str | Path, corpus_root: str | Path) -> dict:
    manifest_path, root = Path(manifest_path), Path(corpus_root)
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != "m69.corpus.v1":
        raise ValueError("expected m69.corpus.v1 manifest")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("corpus manifest has no records")
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row = {
            "id": record.get("id"), "candidate_id": record.get("candidate_id"),
            "group_id": record.get("group_id"), "split": record.get("split"),
            "status": record.get("status"), "design_status": record.get("design_status"),
            "design_qualified": record.get("design_qualified"),
            "source_mesh_sha256": record.get("source_mesh_sha256"),
            "source_geometry_sha256": record.get("source_geometry_sha256"),
            "task_sha256": record.get("task_sha256"),
            "assets": {}, "missing_assets": [], "hash_mismatches": [],
            "surface_mesh_present": False, "pin_support_bound": False,
            "fixed_shell_mask_bound": False, "design_domain_mask_bound": False,
            "faster_beta_initialization_bound": False, "physical_admission": False,
        }
        for field in REQUIRED_ASSETS:
            rel = record.get(field)
            path = _resolve(root, rel)
            exists = bool(path and path.is_file())
            hash_field = field.replace("_file", "_sha256")
            expected = record.get(hash_field)
            actual = sha256(path) if exists else None
            row["assets"][field] = {
                "manifest_path": rel, "resolved_path": str(path) if path else None,
                "exists": exists, "expected_sha256": expected, "actual_sha256": actual,
            }
            if not exists:
                row["missing_assets"].append(field)
            elif expected and actual != expected:
                row["hash_mismatches"].append(field)
        rows.append(row)
    counts = {}
    for row in rows:
        key = str(row["status"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": SCHEMA, "manifest": str(manifest_path), "corpus_root": str(root),
        "corpus_schema": payload["schema"], "manifest_fingerprint": payload.get("fingerprint"),
        "record_count": len(rows), "status_counts": counts,
        "qualified_design_count": sum(
            r["status"] == "qualified" and bool(r["design_qualified"]) for r in rows
        ),
        "records": rows,
        "admission_summary": {
            "surface_mesh_admitted": 0, "fixed_shell_mask_admitted": 0,
            "design_domain_mask_admitted": 0, "pin_support_admitted": 0,
            "faster_beta_initialization_admitted": 0, "physical_cases_admitted": 0,
            "note": "Corpus/label presence is not physical case admission."
        },
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    report = audit_corpus(a.manifest, a.corpus_root)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "record_count": report["record_count"], "status_counts": report["status_counts"],
        "qualified_design_count": report["qualified_design_count"],
        "physical_cases_admitted": 0, "output": str(out.resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
