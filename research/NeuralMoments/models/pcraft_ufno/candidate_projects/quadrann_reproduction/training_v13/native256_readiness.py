"""Fail-closed admission and training-only derivative scales for Phase 3.2.3.

A receipt binds Work A's reviewed native teacher to exact manifest/asset hashes.
It is not an independent mesh-cut implementation or proof of physical truth.
Locked test labels are never opened by these routines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..spin_adapter.unified_factory import get_config
from .unified_corpus import file_sha256

SCHEMA = "tvcg.native-cut-admission.v1"
MOMENT_ORDER = ["1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"]
PHYSICAL_FIELDS = ("teacher_version", "retained_side", "finite_cut_extent",
                   "cap_rule", "fixed_shell", "coordinate_frame", "units",
                   "coefficient_basis", "density_rule", "rejection_rule")


def admit_native_training(corpus, config_name, contract_path=None):
    """Require a reviewed, immutable native256 train/dev release, not test labels."""
    required = get_config(config_name).native_resolution
    if required is None:
        if contract_path is not None:
            raise ValueError("native cut admission requires an explicit native-resolution candidate")
        return None
    if contract_path is None:
        raise ValueError("native256 requires --native-cut-contract from Work A")
    path = Path(contract_path).resolve(strict=True)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("invalid native cut admission schema")
    if doc.get("manifest_sha256") != corpus.manifest_sha256:
        raise ValueError("native cut contract manifest hash mismatch")
    if doc.get("native_resolution") != required:
        raise ValueError("native cut contract resolution mismatch")
    if doc.get("status") != "approved_native_discrete":
        raise ValueError("native cut contract is not approved")
    if doc.get("moment_order") != MOMENT_ORDER or doc.get("jacobian_basis") != "raw_coefficients":
        raise ValueError("native cut moment/Jacobian convention mismatch")
    for field in PHYSICAL_FIELDS:
        value = doc.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"native cut contract missing {field}")
    if not corpus.splits["train"] or not corpus.splits["val"]:
        raise ValueError("native256 requires nonempty train and development splits")
    receipts = doc.get("records")
    rows = corpus.splits["train"] + corpus.splits["val"]
    if not isinstance(receipts, dict) or set(receipts) != {r["id"] for r in rows}:
        raise ValueError("native cut receipts must exactly cover train/dev records")
    for row in rows:
        rid = row["id"]
        if str(row["source_type"]).casefold() != "trellis":
            raise ValueError(f"{rid}: native256 main train/dev must be TRELLIS")
        receipt = receipts[rid]
        if (not isinstance(receipt, dict) or receipt.get("status") != "approved_native_discrete"
                or receipt.get("geometry_sha256") != row["geometry_sha256"]
                or receipt.get("labels_sha256") != row["labels_sha256"]):
            raise ValueError(f"{rid}: native cut receipt/asset hash mismatch")
        gpath = corpus._resolve_asset(row["geometry_file"])
        lpath = corpus._resolve_asset(row["labels_file"])
        if file_sha256(gpath) != row["geometry_sha256"] or file_sha256(lpath) != row["labels_sha256"]:
            raise ValueError(f"{rid}: native cut asset hash mismatch")
        with np.load(gpath, allow_pickle=False) as g:
            sides = [np.asarray(g[k]) for k in ("side", "native_resolution") if k in g.files]
            if not sides or any(v.shape != () or v.dtype.kind not in "iu" or int(v) != required for v in sides):
                raise ValueError(f"{rid}: exact native resolution {required} required in geometry asset")
        if "native_resolution" in row and row["native_resolution"] != required:
            raise ValueError(f"{rid}: manifest native resolution mismatch")
        with np.load(lpath, allow_pickle=False) as z:
            keys = set(z.files)
            if not keys.intersection({"valid_mask", "moment_valid", "query_status"}):
                raise ValueError(f"{rid}: missing explicit Moment validity")
            if "jacobian" in keys and "derivative_valid" not in keys:
                raise ValueError(f"{rid}: missing explicit derivative_valid")
            if keys.intersection({"restricted_basis", "restricted_jacobian"}) and not {
                "restricted_valid", "restricted_dimension"
            }.issubset(keys):
                raise ValueError(f"{rid}: missing explicit restricted validity/dimension")
        # Use the existing loader's complete numerical and mask validation.
        labels = corpus.load_labels(row)
        valid = labels["valid_mask"]
        full = labels.get("jacobian_valid_mask", np.zeros(len(valid), dtype=bool))
        restricted = labels.get("restricted_valid", np.zeros(len(valid), dtype=bool))
        if np.any((full | restricted) & ~valid):
            raise ValueError(f"{rid}: derivative supervision outside Moment validity")
        if restricted.any():
            d = labels["restricted_dimension"][restricted]
            if not np.isin(d, (3, 8)).all():
                raise ValueError(f"{rid}: native256 requires integer R3/R8 dimensions")
    return {"schema": SCHEMA, "manifest_sha256": corpus.manifest_sha256,
            "contract_sha256": hashlib.sha256(raw).hexdigest(),
            "contract_path": str(path), "native_resolution": required,
            "reviewed_train_dev_records": len(rows), "teacher_version": doc["teacher_version"],
            "locked_test_labels_opened": False,
            "scope": "receipt_and_asset_verification_not_independent_native_physics"}


def derivative_scales(corpus, *, mode="train_rms", floor=1e-12):
    """Per-output RMS of valid raw derivative entries; no dev/test fitting.

    Full scale is repeated over 10 raw coefficient columns ([10,10]).
    Restricted uses a separate [10] scale over valid directional columns.
    This intentionally does not whiten arbitrary teacher bases or assume
    orthogonality. R3/R8 counts and RMS are separately reported.
    """
    if mode == "legacy":
        return {"mode": "legacy", "jacobian_scale": 1.0, "restricted_scale": 1.0}
    if mode != "train_rms" or not np.isfinite(floor) or floor <= 0:
        raise ValueError("invalid derivative normalization mode/floor")
    sums = {name: np.zeros(10, dtype=np.float64) for name in ("full", "r3", "r8")}
    counts = dict.fromkeys(sums, 0)
    queries = dict.fromkeys(sums, 0)
    for row in corpus.splits["train"]:
        labels = corpus.load_labels(row)
        valid = labels["valid_mask"]
        fm = labels.get("jacobian_valid_mask", np.zeros(len(valid), dtype=bool))
        rm = labels.get("restricted_valid", np.zeros(len(valid), dtype=bool))
        if np.any((fm | rm) & ~valid) or np.any(fm & rm):
            raise ValueError("inconsistent training derivative masks")
        if fm.any():
            j = np.asarray(labels["target_jacobian"][fm], dtype=np.float64)
            sums["full"] += np.square(j).sum(axis=(0, 2))
            counts["full"] += len(j)*10
            queries["full"] += len(j)
        if rm.any():
            dims = labels["restricted_dimension"]
            if not np.isin(dims[rm], (3, 8)).all():
                raise ValueError("train_rms supports R3 and R8 labels")
            for d in (3, 8):
                mask = rm & (dims == d)
                if mask.any():
                    j = np.asarray(labels["restricted_jacobian"][mask, :, :d], dtype=np.float64)
                    sums[f"r{d}"] += np.square(j).sum(axis=(0, 2))
                    counts[f"r{d}"] += len(j)*d
                    queries[f"r{d}"] += len(j)
    if sum(counts.values()) == 0:
        raise ValueError("train_rms requires valid training derivatives")
    if any(not np.isfinite(s).all() for s in sums.values()):
        raise ValueError("nonfinite derivative RMS accumulator")
    def rms(s, n):
        return np.sqrt(s/n) if n else np.ones(10)
    full = rms(sums["full"], counts["full"])
    restricted = rms(sums["r3"]+sums["r8"], counts["r3"]+counts["r8"])
    return {"mode": mode, "fit_split": "train", "floor": float(floor),
            "jacobian_scale": np.repeat(np.maximum(full, floor)[:, None], 10, axis=1).tolist(),
            "restricted_scale": np.maximum(restricted, floor).tolist(),
            "query_counts": queries, "columns_per_output": counts,
            "rms_by_mode": {k: rms(sums[k], counts[k]).tolist() if counts[k] else None for k in sums},
            "full_floored_outputs": np.flatnonzero(full < floor).tolist(),
            "restricted_floored_outputs": np.flatnonzero(restricted < floor).tolist()}


def resolve_readiness(corpus, config_name, contract_path=None, normalization=None):
    admission = admit_native_training(corpus, config_name, contract_path)
    mode = normalization or ("train_rms" if admission is not None else "legacy")
    return admission, derivative_scales(corpus, mode=mode)
