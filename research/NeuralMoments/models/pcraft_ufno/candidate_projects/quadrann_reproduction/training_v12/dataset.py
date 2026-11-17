"""Strict reader for the frozen M83 SQNO training corpus."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json

import numpy as np


def file_sha256(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(root, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
        or "\x00" in relative
    ):
        raise ValueError("invalid manifest path")

    p = PurePosixPath(relative)

    if p.is_absolute() or ".." in p.parts:
        raise ValueError("path escapes corpus root")

    base = Path(root).resolve()
    result = (base / Path(*p.parts)).resolve()

    if not result.is_relative_to(base) or result == base:
        raise ValueError("path escapes corpus root")

    return result


class M83Corpus:
    REQUIRED_GEOMETRY = {
        "columns", "origin", "h", "side",
        "native_resolution", "density",
        "original_moments", "fixed_moments",
        "bounds",
    }

    REQUIRED_LABELS = {
        "queries", "moments", "jacobian",
        "derivative_valid", "query_status",
    }

    SPLITS = ("train", "val", "test")

    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent

        self.manifest_sha256 = file_sha256(self.path)
        self.doc = json.loads(self.path.read_text())

        self.rows = [
            r for r in self.doc["records"]
            if r.get("status") == "qualified"
        ]

        self.splits = {
            s: sorted(
                [r for r in self.rows if r.get("split") == s],
                key=lambda r: r["id"],
            )
            for s in self.SPLITS
        }

        if any(not self.splits[s] for s in self.SPLITS):
            raise ValueError("empty train/val/test split")

        self._validate_groups()

    def _validate_groups(self):
        owners = {}

        for split, rows in self.splits.items():
            for row in rows:
                gid = row.get("group_id")

                if not gid:
                    raise ValueError(
                        f"{row['id']}: missing group_id"
                    )

                if gid in owners and owners[gid] != split:
                    raise ValueError(
                        f"group leakage: {gid}"
                    )

                owners[gid] = split

    def split_summary(self):
        result = {}

        for split, rows in self.splits.items():
            groups = sorted({
                r["group_id"] for r in rows
            })

            result[split] = {
                "records": len(rows),
                "groups": len(groups),
                "group_ids": groups,
            }

        return result

    def load(self, row):
        gp = safe_member(self.root, row["geometry_file"])
        lp = safe_member(self.root, row["labels_file"])

        if file_sha256(gp) != row["geometry_sha256"]:
            raise ValueError(
                f"{row['id']}: geometry SHA256 mismatch"
            )

        if file_sha256(lp) != row["labels_sha256"]:
            raise ValueError(
                f"{row['id']}: labels SHA256 mismatch"
            )

        with np.load(gp, allow_pickle=False) as z:
            if set(z.files) != self.REQUIRED_GEOMETRY:
                missing = self.REQUIRED_GEOMETRY - set(z.files)
                extra = set(z.files) - self.REQUIRED_GEOMETRY
                raise ValueError(
                    f"geometry schema mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            geometry = {
                k: np.array(z[k], copy=True)
                for k in z.files
            }

        with np.load(lp, allow_pickle=False) as z:
            missing = self.REQUIRED_LABELS - set(z.files)

            if missing:
                raise ValueError(
                    f"missing label fields: {missing}"
                )

            labels = {
                k: np.array(z[k], copy=True)
                for k in z.files
            }

        n = len(labels["queries"])

        expected = {
            "queries": (n, 10),
            "moments": (n, 10),
            "jacobian": (n, 10, 10),
            "derivative_valid": (n,),
            "query_status": (n,),
        }

        for name, shape in expected.items():
            if labels[name].shape != shape:
                raise ValueError(
                    f"{row['id']}: invalid {name} shape"
                )

        for name in (
            "queries", "moments", "jacobian",
        ):
            if not np.isfinite(labels[name]).all():
                raise ValueError(
                    f"{row['id']}: nonfinite {name}"
                )

        for name in (
            "original_moments", "fixed_moments",
        ):
            if geometry[name].shape != (10,):
                raise ValueError(
                    f"{row['id']}: invalid {name}"
                )

        for value in labels.values():
            value.flags.writeable = False

        for value in geometry.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

        return geometry, labels

    def iter_split(self, split):
        if split not in self.SPLITS:
            raise ValueError(split)

        for row in self.splits[split]:
            geometry, labels = self.load(row)
            yield row, geometry, labels
