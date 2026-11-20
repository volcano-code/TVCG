"""Strict manifest-backed corpus for Unified M80+SQNO training.

The loader preserves partial derivative supervision and source-group splits.
It adapts native RLE geometry to the existing P-CRAFT input contract and
never fabricates missing Jacobian labels.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
from typing import Any

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    geometry_from_columns,
    conservative_coarse_grid,
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "\x00" in relative):
        raise ValueError("invalid corpus member path")
    member = PurePosixPath(relative)
    if member.is_absolute() or any(part in {".", ".."} for part in member.parts):
        raise ValueError("corpus member path must be relative and traversal-free")
    base = root.resolve()
    result = (base / Path(*member.parts)).resolve()
    if not result.is_relative_to(base) or result == base:
        raise ValueError("corpus member escapes corpus root")
    return result


class UnifiedOperatorCorpus:
    """Read immutable RLE geometry + Moment/partial-Jacobian NPZ records.

    Manifest schema:
      {"schema":"tvcg.unified-operator-corpus.v1",
       "records":[{"id","group_id","split","status","geometry_file",
       "geometry_sha256","labels_file","labels_sha256","source_type"}]}

    Geometry NPZ uses the existing native P-CRAFT RLE schema:
    columns, origin, h, side (or native_resolution), density,
    fixed_moments, original_moments.

    Labels NPZ requires queries and moments. Full and restricted derivative
    arrays are optional and retain independent boolean masks.
    """

    SCHEMA = "tvcg.unified-operator-corpus.v1"
    SPLITS = ("train", "val", "test")
    GEOMETRY_REQUIRED = {
        "columns", "origin", "h", "fixed_moments", "original_moments",
    }
    LABEL_REQUIRED = {"queries", "moments"}
    LABEL_OPTIONAL = {
        "jacobian", "derivative_valid", "restricted_basis",
        "restricted_jacobian", "restricted_valid", "restricted_dimension",
        "query_status", "event_state",
    }

    def __init__(self, manifest_path: str | Path, *, grid_size: int = 64):
        self.path = Path(manifest_path).resolve(strict=True)
        self.root = self.path.parent
        self.manifest_sha256 = file_sha256(self.path)
        self.document = json.loads(self.path.read_text())
        if self.document.get("schema") != self.SCHEMA:
            raise ValueError(f"unsupported manifest schema: expected {self.SCHEMA}")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = int(grid_size)
        self.records = [
            row for row in self.document.get("records", [])
            if row.get("status") == "qualified"
        ]
        if not self.records:
            raise ValueError("manifest has no qualified records")
        self.splits = {
            split: sorted(
                [row for row in self.records if row.get("split") == split],
                key=lambda row: row["id"],
            )
            for split in self.SPLITS
        }
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        ids: set[str] = set()
        owners: dict[str, str] = {}
        for split, rows in self.splits.items():
            for row in rows:
                for key in ("id", "group_id", "geometry_file", "geometry_sha256",
                            "labels_file", "labels_sha256", "source_type"):
                    if key not in row:
                        raise ValueError(f"manifest row missing {key}")
                rid = row["id"]
                if not isinstance(rid, str) or not rid or rid in ids:
                    raise ValueError(f"duplicate or invalid record id: {rid!r}")
                ids.add(rid)
                group = row["group_id"]
                if not isinstance(group, str) or not group:
                    raise ValueError(f"{rid}: missing source group_id")
                if group in owners and owners[group] != split:
                    raise ValueError(f"source-group leakage: {group}")
                owners[group] = split
                for key in ("geometry_sha256", "labels_sha256"):
                    value = row[key]
                    if not isinstance(value, str) or len(value) != 64:
                        raise ValueError(f"{rid}: invalid {key}")
                    try:
                        int(value, 16)
                    except ValueError as exc:
                        raise ValueError(f"{rid}: invalid {key}") from exc
                safe_member(self.root, row["geometry_file"])
                safe_member(self.root, row["labels_file"])
        if not self.splits["train"]:
            raise ValueError("empty train split")
        if not self.splits["val"]:
            raise ValueError("empty val split")
        if not self.splits["test"]:
            raise ValueError("empty test split")

    def split_summary(self) -> dict[str, Any]:
        return {
            split: {
                "records": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "group_ids": sorted({row["group_id"] for row in rows}),
            }
            for split, rows in self.splits.items()
        }

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}

    def load(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        geometry_path = safe_member(self.root, row["geometry_file"])
        labels_path = safe_member(self.root, row["labels_file"])
        if file_sha256(geometry_path) != row["geometry_sha256"]:
            raise ValueError(f"{row['id']}: geometry SHA-256 mismatch")
        if file_sha256(labels_path) != row["labels_sha256"]:
            raise ValueError(f"{row['id']}: labels SHA-256 mismatch")

        raw_geometry = self._load_npz(geometry_path)
        missing_geometry = self.GEOMETRY_REQUIRED - set(raw_geometry)
        if missing_geometry or not ({"side", "native_resolution"} & set(raw_geometry)):
            raise ValueError(f"{row['id']}: invalid RLE geometry schema; missing={sorted(missing_geometry)}")
        geometry = geometry_from_columns(raw_geometry)
        if self.grid_size > geometry["side"] or geometry["side"] % self.grid_size:
            raise ValueError(f"{row['id']}: native side must be divisible by grid_size")
        geometry["pcraft_grid"] = conservative_coarse_grid(geometry, self.grid_size)

        raw_labels = self._load_npz(labels_path)
        missing_labels = self.LABEL_REQUIRED - set(raw_labels)
        unknown_labels = set(raw_labels) - self.LABEL_REQUIRED - self.LABEL_OPTIONAL
        if missing_labels or unknown_labels:
            raise ValueError(
                f"{row['id']}: label schema mismatch; missing={sorted(missing_labels)}, "
                f"unknown={sorted(unknown_labels)}"
            )
        q = np.asarray(raw_labels["queries"])
        moments = np.asarray(raw_labels["moments"])
        if q.ndim != 2 or q.shape[1] != 10 or not len(q):
            raise ValueError(f"{row['id']}: queries must be nonempty [Q,10]")
        if moments.shape != (len(q), 10):
            raise ValueError(f"{row['id']}: moments must have shape [Q,10]")
        if not np.isfinite(q).all() or not np.isfinite(moments).all():
            raise ValueError(f"{row['id']}: queries and moments must be finite")
        if (np.linalg.norm(q.astype(np.float64), axis=1) <= 0).any():
            raise ValueError(f"{row['id']}: zero quadric query")

        labels: dict[str, np.ndarray] = {
            "queries": q,
            "moments": moments,
            "valid_mask": np.asarray(raw_labels.get("valid_mask", np.ones(len(q), dtype=bool))),
        }
        if labels["valid_mask"].dtype != np.bool_ or labels["valid_mask"].shape != (len(q),):
            raise ValueError(f"{row['id']}: valid_mask must be boolean [Q]")
        if not labels["valid_mask"].any():
            raise ValueError(f"{row['id']}: no valid Moment queries")
        if not np.isfinite(moments[labels["valid_mask"]]).all():
            raise ValueError(f"{row['id']}: active Moment targets must be finite")

        if "jacobian" in raw_labels:
            jac = np.asarray(raw_labels["jacobian"])
            if jac.shape != (len(q), 10, 10):
                raise ValueError(f"{row['id']}: jacobian must have shape [Q,10,10]")
            labels["target_jacobian"] = jac
        if "derivative_valid" in raw_labels:
            labels["jacobian_valid_mask"] = np.asarray(raw_labels["derivative_valid"])
        if "restricted_basis" in raw_labels:
            labels["restricted_basis"] = np.asarray(raw_labels["restricted_basis"])
        if "restricted_jacobian" in raw_labels:
            labels["restricted_jacobian"] = np.asarray(raw_labels["restricted_jacobian"])
        if "restricted_valid" in raw_labels:
            labels["restricted_valid"] = np.asarray(raw_labels["restricted_valid"])
        if "restricted_dimension" in raw_labels:
            labels["restricted_dimension"] = np.asarray(raw_labels["restricted_dimension"])
        if "query_status" in raw_labels:
            labels["query_status"] = np.asarray(raw_labels["query_status"])
        if "event_state" in raw_labels:
            labels["event_state"] = np.asarray(raw_labels["event_state"])

        for key in ("jacobian_valid_mask", "restricted_valid"):
            if key in labels and (
                labels[key].dtype != np.bool_ or labels[key].shape != (len(q),)
            ):
                raise ValueError(f"{row['id']}: {key} must be boolean [Q]")
        if "jacobian_valid_mask" not in labels and "target_jacobian" in labels:
            labels["jacobian_valid_mask"] = labels["valid_mask"].copy()
        if "restricted_basis" in labels or "restricted_jacobian" in labels:
            if not {"restricted_basis", "restricted_jacobian"}.issubset(labels):
                raise ValueError(f"{row['id']}: restricted basis and labels must be paired")
            basis, target = labels["restricted_basis"], labels["restricted_jacobian"]
            if basis.ndim != 3 or basis.shape[:2] != (len(q), 10) or target.shape != basis.shape:
                raise ValueError(f"{row['id']}: restricted labels must match [Q,10,D]")
            if "restricted_valid" not in labels:
                labels["restricted_valid"] = np.zeros(len(q), dtype=bool)
            if "restricted_dimension" not in labels:
                labels["restricted_dimension"] = np.full(len(q), basis.shape[2], dtype=np.int64)
            if labels["restricted_dimension"].shape != (len(q),):
                raise ValueError(f"{row['id']}: restricted_dimension must be [Q]")
        elif "restricted_valid" in labels and labels["restricted_valid"].any():
            raise ValueError(f"{row['id']}: restricted_valid set without restricted labels")

        full_mask = labels.get("jacobian_valid_mask", np.zeros(len(q), dtype=bool))
        restricted_mask = labels.get("restricted_valid", np.zeros(len(q), dtype=bool))
        if np.any(full_mask & restricted_mask):
            raise ValueError(f"{row['id']}: full and restricted derivative masks overlap")
        if "target_jacobian" in labels and not np.isfinite(labels["target_jacobian"][full_mask]).all():
            raise ValueError(f"{row['id']}: active full Jacobian targets must be finite")
        if "restricted_basis" in labels:
            dims = labels["restricted_dimension"]
            for i in np.flatnonzero(restricted_mask):
                d = int(dims[i])
                if d < 1 or d > labels["restricted_basis"].shape[2]:
                    raise ValueError(f"{row['id']}: invalid active restricted dimension")
                if not np.isfinite(labels["restricted_basis"][i, :, :d]).all():
                    raise ValueError(f"{row['id']}: active restricted basis is nonfinite")
                if not np.isfinite(labels["restricted_jacobian"][i, :, :d]).all():
                    raise ValueError(f"{row['id']}: active restricted target is nonfinite")

        # Keep arrays immutable; downstream training converts them to tensors.
        for value in labels.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        for value in geometry.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        return geometry, labels

    def to_training_batch(self, row: dict[str, Any]) -> dict[str, Any]:
        geometry, labels = self.load(row)
        batch: dict[str, Any] = {
            "geometry": geometry,
            "coefficients": labels["queries"],
            "target_moments": labels["moments"],
            "valid_mask": labels["valid_mask"],
        }
        for source, target in (
            ("target_jacobian", "target_jacobian"),
            ("jacobian_valid_mask", "jacobian_valid_mask"),
            ("restricted_basis", "restricted_basis"),
            ("restricted_jacobian", "restricted_jacobian"),
            ("restricted_valid", "restricted_valid"),
            ("restricted_dimension", "restricted_dimension"),
        ):
            if source in labels:
                batch[target] = labels[source]
        return batch

    def iter_split(self, split: str):
        if split not in self.SPLITS:
            raise ValueError(split)
        for row in self.splits[split]:
            yield row, self.to_training_batch(row)


__all__ = ["UnifiedOperatorCorpus", "file_sha256", "safe_member"]
