from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.unified_corpus import (
    UnifiedOperatorCorpus,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns


def sha(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class UnifiedCorpusTests(unittest.TestCase):
    def _write_corpus(self, root: Path, *, restricted=False):
        columns = np.asarray([[1, 1, 0, 4]], dtype=np.int64)
        origin = np.zeros(3, dtype=np.float64)
        h = np.asarray(1.0)
        fixed = np.zeros(10, dtype=np.float64)
        provisional = {
            "columns": columns, "origin": origin, "h": h,
            "side": np.asarray(4, dtype=np.int64),
            "density": np.asarray(1.0), "fixed_moments": fixed,
        }
        partial = geometry_from_columns(provisional)
        provisional["original_moments"] = partial["design_moments"] if "design_moments" in partial else partial["mass"] @ partial["moment_basis"]
        geometry_path = root / "geometry.npz"
        np.savez(geometry_path, **provisional)

        q = np.zeros((2, 10), dtype=np.float64)
        q[:, 0] = 1.0
        moments = np.zeros((2, 10), dtype=np.float64)
        moments[:, 0] = 1.0
        labels = {
            "queries": q,
            "moments": moments,
            "valid_mask": np.asarray([True, True]),
            "jacobian": np.full((2, 10, 10), np.nan, dtype=np.float64),
            "derivative_valid": np.asarray([True, False]),
        }
        labels["jacobian"][0] = np.eye(10)
        if restricted:
            labels["restricted_basis"] = np.full((2, 10, 3), np.nan)
            labels["restricted_jacobian"] = np.full((2, 10, 3), np.nan)
            labels["restricted_basis"][1] = np.eye(10)[:, :3]
            labels["restricted_jacobian"][1] = np.ones((10, 3))
            labels["restricted_valid"] = np.asarray([False, True])
            labels["restricted_dimension"] = np.asarray([0, 3], dtype=np.int64)
        labels_path = root / "labels.npz"
        np.savez(labels_path, **labels)

        records = []
        for split in ("train", "val", "test"):
            records.append({
                "id": split,
                "group_id": f"group-{split}",
                "split": split,
                "status": "qualified",
                "source_type": "synthetic_test",
                "geometry_file": "geometry.npz",
                "geometry_sha256": sha(geometry_path),
                "labels_file": "labels.npz",
                "labels_sha256": sha(labels_path),
            })
        manifest = {
            "schema": UnifiedOperatorCorpus.SCHEMA,
            "records": records,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_loads_native_rle_and_partial_full_jacobian(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_corpus(Path(tmp))
            corpus = UnifiedOperatorCorpus(path, grid_size=2)
            row = corpus.splits["train"][0]
            batch = corpus.to_training_batch(row)
            self.assertEqual(batch["coefficients"].shape, (2, 10))
            self.assertEqual(batch["target_moments"].shape, (2, 10))
            self.assertEqual(batch["target_jacobian"].shape, (2, 10, 10))
            self.assertEqual(batch["jacobian_valid_mask"].tolist(), [True, False])
            self.assertEqual(batch["geometry"]["pcraft_grid"].shape, (4, 2, 2, 2))

    def test_loads_restricted_derivative_without_fabricating_full_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_corpus(Path(tmp), restricted=True)
            corpus = UnifiedOperatorCorpus(path, grid_size=2)
            batch = corpus.to_training_batch(corpus.splits["train"][0])
            self.assertEqual(batch["jacobian_valid_mask"].tolist(), [True, False])
            self.assertEqual(batch["restricted_valid"].tolist(), [False, True])
            self.assertEqual(batch["restricted_dimension"].tolist(), [0, 3])
            self.assertTrue(np.isnan(batch["target_jacobian"][1]).all())

    def test_rejects_source_group_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_corpus(Path(tmp))
            manifest = json.loads(path.read_text())
            manifest["records"][1]["group_id"] = manifest["records"][0]["group_id"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source-group leakage"):
                UnifiedOperatorCorpus(path, grid_size=2)

    def test_rejects_geometry_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_corpus(Path(tmp))
            corpus = UnifiedOperatorCorpus(path, grid_size=2)
            row = dict(corpus.splits["train"][0])
            row["geometry_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "geometry SHA-256 mismatch"):
                corpus.load(row)

    def test_rejects_derivative_mask_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_corpus(Path(tmp), restricted=True)
            labels_path = Path(tmp) / "labels.npz"
            with np.load(labels_path) as z:
                values = {k: np.array(z[k], copy=True) for k in z.files}
            values["restricted_valid"] = np.asarray([True, True])
            np.savez(labels_path, **values)
            manifest = json.loads(path.read_text())
            manifest["records"] = [
                {**row, "labels_sha256": sha(labels_path)}
                for row in manifest["records"]
            ]
            path.write_text(json.dumps(manifest))
            corpus = UnifiedOperatorCorpus(path, grid_size=2)
            with self.assertRaisesRegex(ValueError, "masks overlap"):
                corpus.load(corpus.splits["train"][0])


    def _write_work_corpus(self, root: Path):
        columns = np.asarray([[1, 1, 0, 4]], dtype=np.int64)
        geometry_path = root / "work_geometry.npz"
        np.savez(
            geometry_path,
            columns=columns,
            origin=np.zeros(3, dtype=np.float64),
            h=np.asarray(1.0),
            native_resolution=np.asarray(4, dtype=np.int64),
            density=np.asarray(1.0),
            fixed_moments=np.zeros(10, dtype=np.float64),
        )
        q = np.zeros((2, 10), dtype=np.float64)
        q[:, 0] = 1.0
        moments = np.zeros((2, 10), dtype=np.float64)
        moments[:, 0] = 1.0
        jac = np.full((2, 10, 10), np.nan, dtype=np.float64)
        jac[0] = np.eye(10)
        basis = np.full((2, 10, 8), np.nan, dtype=np.float64)
        restricted_jac = np.full((2, 10, 8), np.nan, dtype=np.float64)
        basis[1, :, :3] = np.eye(10)[:, :3]
        restricted_jac[1, :, :3] = 0.5
        labels_path = root / "work_labels.npz"
        np.savez(
            labels_path,
            query_id=np.asarray(["q0", "q1"]),
            queries=q,
            moments=moments,
            full_moments=np.asarray([2.0] + [0.0] * 9),
            jacobian=jac,
            derivative_valid=np.asarray([True, False]),
            restricted_valid=np.asarray([False, True]),
            restricted_dimension=np.asarray([0, 3], dtype=np.int16),
            restricted_basis=basis,
            restricted_jacobian=restricted_jac,
            query_status=np.asarray(["accepted_moment", "accepted_moment"]),
            event_valid=np.asarray([1, -1], dtype=np.int8),
        )
        records = []
        for geometry_id, source_split, source_group in (
            ("train-a", "train", "train-family"),
            ("dev-a", "development", "dev-family"),
            ("bench-a", "fixed_benchmark", "benchmark-family"),
        ):
            records.append({
                "geometry_id": geometry_id,
                "source_group_id": source_group,
                "source_type": "Objaverse",
                "split": source_split,
                "geometry_file": geometry_path.name,
                "geometry_sha256": sha(geometry_path),
                "labels_file": labels_path.name,
                "labels_sha256": sha(labels_path),
                "native_resolution": 4,
                "fixed_moments": [0.0] * 10,
                "original_moments": [2.0] + [0.0] * 9,
                "queries": 2,
            })
        manifest_path = root / "work_manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": "phase2.10.corpus.v1",
            "records": records,
            "excluded_partial_count": 76,
        }))
        return manifest_path

    def test_adapts_work_manifest_splits_and_mesh_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_work_corpus(Path(tmp))
            corpus = UnifiedOperatorCorpus(path, grid_size=2)
            self.assertEqual(
                {key: len(value) for key, value in corpus.splits.items()},
                {"train": 1, "val": 1, "test": 1},
            )
            self.assertEqual(corpus.splits["val"][0]["source_split"], "development")
            self.assertEqual(corpus.splits["test"][0]["source_split"], "fixed_benchmark")
            batch = corpus.to_training_batch(corpus.splits["train"][0])
            self.assertEqual(batch["coefficients"].shape, (2, 10))
            self.assertEqual(batch["valid_mask"].tolist(), [True, True])
            self.assertEqual(batch["jacobian_valid_mask"].tolist(), [True, False])
            self.assertEqual(batch["restricted_valid"].tolist(), [False, True])
            self.assertEqual(batch["restricted_dimension"].tolist(), [0, 3])
            # Mesh-teacher original moments are separate from coarse RLE moments.
            self.assertAlmostEqual(float(batch["geometry"]["original_moments"][0]), 4.0)
            self.assertAlmostEqual(float(corpus.splits["train"][0]["original_moments"][0]), 2.0)

    def test_work_sample_can_require_only_its_declared_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_work_corpus(Path(tmp))
            # The portable one-record ZIPs are interface fixtures, not trainable
            # corpora; permit a split-scoped validation invocation explicitly.
            manifest = json.loads(path.read_text())
            manifest["records"] = [manifest["records"][0]]
            path.write_text(json.dumps(manifest))
            corpus = UnifiedOperatorCorpus(
                path, grid_size=2, required_splits=("train",)
            )
            self.assertEqual(len(corpus.splits["train"]), 1)
            self.assertEqual(len(corpus.splits["val"]), 0)


if __name__ == "__main__":
    unittest.main()
