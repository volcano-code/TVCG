"""Manifest-backed corpus adapter for Unified M80+SQNO training.

Supports both the original internal schema and Work Phase 2.10 corpus.v1.
Work's RLE is a conditioning representation: its moments are derived from
RLE and are deliberately NOT equated to the continuous mesh teacher moments.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
from typing import Any, Iterable

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
    """Read immutable RLE geometry and Moment/partial-Jacobian NPZ records.

    Internal schema: tvcg.unified-operator-corpus.v1
    Work schema: phase2.10.corpus.v1

    Work split mapping:
      train -> train
      development -> val
      fixed_benchmark -> test (locked benchmark; not an OOD-generalization claim)
      train_partial / excluded records -> ignored

    Work full manifests use absolute local asset paths. Set
    allow_absolute_paths=True explicitly for those manifests. Portable sample
    ZIPs use relative paths and need no such option.
    """

    SCHEMA = "tvcg.unified-operator-corpus.v1"
    WORK_SCHEMA = "phase2.10.corpus.v1"
    SPLITS = ("train", "val", "test")
    GEOMETRY_REQUIRED = {"columns", "origin", "h", "fixed_moments"}
    LABEL_REQUIRED = {"queries", "moments"}
    LABEL_OPTIONAL = {
        "jacobian", "valid_mask", "moment_valid", "derivative_valid",
        "restricted_basis", "restricted_jacobian", "restricted_valid",
        "restricted_dimension", "query_status", "event_state", "event_valid",
        "restricted_event_valid", "event_scope", "query_id", "proposal_index",
        "stratum", "ambient_event_valid", "native_output_triangulation_event_valid",
        "event_margin", "continuous_reference", "continuous_absolute_errors",
        "continuous_reference_uncertainty_indicators", "finite_difference_audit",
        "chart_stable_at_initial_epsilon", "minimum_reviewed_coarse_epsilon",
        "column_valid", "column_event_valid", "column_jacobian",
        "restricted_chart_dimension", "restricted_column_valid",
        "restricted_column_event_valid", "restricted_column_jacobian",
        "restricted_chart_basis", "column_fd_steps", "restricted_column_fd_steps",
        "derivative_valid_reason", "conditioning_moments_rle1024",
        "full_moments", "queries_source_frame", "moments_source_frame",
        "queries_before_positive_l2", "continuous_reference_source_frame",
        "continuous_absolute_errors_source_frame",
        "continuous_reference_uncertainty_indicators_source_frame",
        "continuous_0p1pct_report_mask", "continuous_1pct_report_mask",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        grid_size: int = 64,
        allow_absolute_paths: bool = False,
        required_splits: Iterable[str] = ("train", "val", "test"),
    ):
        self.path = Path(manifest_path).resolve(strict=True)
        self.root = self.path.parent
        self.allow_absolute_paths = bool(allow_absolute_paths)
        self.manifest_sha256 = file_sha256(self.path)
        self.document = json.loads(self.path.read_text())
        schema = self.document.get("schema")
        if schema not in {self.SCHEMA, self.WORK_SCHEMA}:
            raise ValueError(
                f"unsupported manifest schema: {schema!r}; "
                f"expected {self.SCHEMA!r} or {self.WORK_SCHEMA!r}"
            )
        self.schema = schema
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = int(grid_size)
        self.records = self._normalize_records(self.document.get("records", []))
        if not self.records:
            raise ValueError("manifest has no admitted records")
        self.splits = {
            split: sorted(
                [row for row in self.records if row["split"] == split],
                key=lambda row: row["id"],
            )
            for split in self.SPLITS
        }
        self._validate_manifest(set(required_splits))

    def _normalize_records(self, records: Any) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError("manifest records must be a list")
        normalized = []
        for original in records:
            if not isinstance(original, dict):
                raise ValueError("manifest record must be an object")
            if original.get("status", "qualified") != "qualified":
                continue
            source_split = original.get("split")
            if source_split == "train_partial":
                continue
            if self.schema == self.WORK_SCHEMA:
                split = {
                    "train": "train",
                    "development": "val",
                    "fixed_benchmark": "test",
                }.get(source_split)
                if split is None:
                    continue
                row = dict(original)
                row.update({
                    "id": original.get("geometry_id"),
                    "group_id": original.get("source_group_id"),
                    "split": split,
                    "status": "qualified",
                    "source_split": source_split,
                })
            else:
                row = dict(original)
                row.setdefault("status", "qualified")
                row["source_split"] = source_split
            normalized.append(row)
        return normalized

    def _resolve_asset(self, value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("asset path must be a nonempty string")
        candidate = Path(value)
        if candidate.is_absolute():
            if not self.allow_absolute_paths:
                raise ValueError(
                    "absolute asset path requires allow_absolute_paths=True"
                )
            return candidate.resolve(strict=True)
        return safe_member(self.root, value).resolve(strict=True)

    def _validate_manifest(self, required_splits: set[str]) -> None:
        unknown = required_splits - set(self.SPLITS)
        if unknown:
            raise ValueError(f"unknown required splits: {sorted(unknown)}")
        ids: set[str] = set()
        owners: dict[str, str] = {}
        for row in self.records:
            for key in ("id", "group_id", "split", "geometry_file",
                        "geometry_sha256", "labels_file", "labels_sha256",
                        "source_type"):
                if key not in row or row[key] is None:
                    raise ValueError(f"manifest row missing {key}")
            rid = row["id"]
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError(f"duplicate or invalid record id: {rid!r}")
            ids.add(rid)
            group = row["group_id"]
            if not isinstance(group, str) or not group:
                raise ValueError(f"{rid}: missing source group_id")
            split = row["split"]
            if split not in self.SPLITS:
                raise ValueError(f"{rid}: invalid normalized split {split!r}")
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
            self._resolve_asset(row["geometry_file"])
            self._resolve_asset(row["labels_file"])
        for split in required_splits:
            if not self.splits[split]:
                raise ValueError(f"empty required split: {split}")

    def split_summary(self) -> dict[str, Any]:
        return {
            split: {
                "records": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "group_ids": sorted({row["group_id"] for row in rows}),
                "source_split": sorted({row.get("source_split", split) for row in rows}),
            }
            for split, rows in self.splits.items()
        }

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}

    def load(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        geometry_path = self._resolve_asset(row["geometry_file"])
        labels_path = self._resolve_asset(row["labels_file"])
        if file_sha256(geometry_path) != row["geometry_sha256"]:
            raise ValueError(f"{row['id']}: geometry SHA-256 mismatch")
        if file_sha256(labels_path) != row["labels_sha256"]:
            raise ValueError(f"{row['id']}: labels SHA-256 mismatch")

        raw_geometry = self._load_npz(geometry_path)
        missing_geometry = self.GEOMETRY_REQUIRED - set(raw_geometry)
        if missing_geometry or not ({"side", "native_resolution"} & set(raw_geometry)):
            raise ValueError(
                f"{row['id']}: invalid RLE geometry schema; "
                f"missing={sorted(missing_geometry)}"
            )
        # Work's RLE NPZ intentionally omits original_moments. Derive the
        # conditioning-grid moment identity from fixed + editable RLE cells.
        # Mesh-teacher full_moments remain a separate label/provenance field.
        geometry = geometry_from_columns(raw_geometry)
        if "native_resolution" in row and int(geometry["side"]) != int(row["native_resolution"]):
            raise ValueError(f"{row['id']}: manifest native_resolution disagrees with RLE")
        if "fixed_moments" in row:
            manifest_fixed = np.asarray(row["fixed_moments"], dtype=np.float64)
            if manifest_fixed.shape != (10,) or not np.array_equal(
                geometry["fixed_moments"], manifest_fixed
            ):
                raise ValueError(f"{row['id']}: manifest fixed_moments disagree with RLE")
        if self.grid_size > geometry["side"] or geometry["side"] % self.grid_size:
            raise ValueError(f"{row['id']}: native side must be divisible by grid_size")
        geometry["pcraft_grid"] = conservative_coarse_grid(geometry, self.grid_size)

        raw_labels = self._load_npz(labels_path)
        missing_labels = self.LABEL_REQUIRED - set(raw_labels)
        if missing_labels:
            raise ValueError(f"{row['id']}: missing label arrays {sorted(missing_labels)}")
        q = np.asarray(raw_labels["queries"])
        moments = np.asarray(raw_labels["moments"])
        if q.ndim != 2 or q.shape[1] != 10 or not len(q):
            raise ValueError(f"{row['id']}: queries must be nonempty [Q,10]")
        if moments.shape != (len(q), 10):
            raise ValueError(f"{row['id']}: moments must have shape [Q,10]")
        if "queries" in row and int(row["queries"]) != len(q):
            raise ValueError(f"{row['id']}: manifest query count disagrees with labels")
        if not np.isfinite(q).all() or not np.isfinite(moments).all():
            raise ValueError(f"{row['id']}: queries and moments must be finite")
        if (np.linalg.norm(q.astype(np.float64), axis=1) <= 0).any():
            raise ValueError(f"{row['id']}: zero quadric query")

        if "valid_mask" in raw_labels:
            valid = np.asarray(raw_labels["valid_mask"])
        elif "moment_valid" in raw_labels:
            valid = np.asarray(raw_labels["moment_valid"])
        elif "query_status" in raw_labels:
            valid = np.asarray(raw_labels["query_status"] == "accepted_moment")
        else:
            valid = np.ones(len(q), dtype=bool)
        if valid.dtype != np.bool_ or valid.shape != (len(q),):
            raise ValueError(f"{row['id']}: valid Moment mask must be boolean [Q]")
        if not valid.any():
            raise ValueError(f"{row['id']}: no valid Moment queries")
        if not np.isfinite(moments[valid]).all():
            raise ValueError(f"{row['id']}: active Moment targets must be finite")

        labels: dict[str, Any] = {
            "queries": q,
            "moments": moments,
            "valid_mask": valid,
        }
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
        if "event_valid" in raw_labels:
            labels["event_valid"] = np.asarray(raw_labels["event_valid"])
        if "restricted_event_valid" in raw_labels:
            labels["restricted_event_valid"] = np.asarray(raw_labels["restricted_event_valid"])

        for key in ("jacobian_valid_mask", "restricted_valid"):
            if key in labels and (
                labels[key].dtype != np.bool_ or labels[key].shape != (len(q),)
            ):
                raise ValueError(f"{row['id']}: {key} must be boolean [Q]")
        if "jacobian_valid_mask" not in labels and "target_jacobian" in labels:
            labels["jacobian_valid_mask"] = labels["valid_mask"].copy()
        if ("jacobian_valid_mask" in labels and "target_jacobian" not in labels
                and labels["jacobian_valid_mask"].any()):
            raise ValueError(
                f"{row['id']}: derivative_valid selects full Jacobian rows "
                "but Jacobian labels are absent"
            )
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

        if "full_moments" in raw_labels and "original_moments" in row:
            teacher_full = np.asarray(raw_labels["full_moments"])
            manifest_full = np.asarray(row["original_moments"])
            if teacher_full.shape != (10,) or manifest_full.shape != (10,):
                raise ValueError(f"{row['id']}: original/full moments must be [10]")
            if not np.allclose(teacher_full, manifest_full, rtol=1e-10, atol=1e-12):
                raise ValueError(f"{row['id']}: manifest original_moments != label full_moments")

        # Preserve non-training audit arrays without feeding them to train_step.
        labels["audit_metadata"] = {
            key: value for key, value in raw_labels.items()
            if key not in self.LABEL_REQUIRED | self.LABEL_OPTIONAL
        }
        for value in labels.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            elif isinstance(value, dict):
                for nested in value.values():
                    if isinstance(nested, np.ndarray):
                        nested.flags.writeable = False
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
