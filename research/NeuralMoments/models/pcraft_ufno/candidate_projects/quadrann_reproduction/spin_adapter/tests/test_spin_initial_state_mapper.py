from __future__ import annotations

import hashlib
import json
import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_initial_state_mapper import (
    SPIN_INITIAL_STATE_SCHEMA,
    map_spin_initial_state_record,
)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def fixture():
    task = {
        "schema": "m69.fixed-axis-task.v1",
        "axis": [0.0, 0.0, 1.0],
        "axis_origin": [0.0, 0.0, 0.0],
        "contact": [0.0, 0.0, 0.0],
        "density": 1.0,
        "polar_retention": 0.90,
        "min_removal_fraction": 0.10,
        "max_centroid_rise": 1e-7,
        "com_tolerance": 1e-4,
        "alignment_tolerance": 1e-4,
        "principal_gap_min": -1e-8,
        "require_connected": True,
        "mass_tolerance": 1e-10,
        "inertia_tolerance": 1e-10,
        "moment_psd_tolerance": 1e-9,
        "solver_tolerance": 1e-9,
        "objective": "minimum_retained_mass",
        "material_policy": "single_density_removal_only",
        "inertia_reference": "original_fixed_axis",
    }
    state = {
        "beta_anchor": [1.0] + [0.0] * 9,
        "original_moments": [1.0] + [0.0] * 9,
        "length_scale": 1.0,
        "active_mask": [False] * 10,
        "axis": [0.0, 0.0, 1.0],
        "axis_origin": [0.0, 0.0, 0.0],
        "contact": [0.0, 0.0, 0.0],
        "coordinate_frame": "task_frame",
        "fixed_material_sha256": "1" * 64,
        "design_domain_sha256": "2" * 64,
    }
    return {
        "schema": SPIN_INITIAL_STATE_SCHEMA,
        "status": "AUDITED_INITIAL_STATE",
        "evidence": "AUDITED_REAL_GEOMETRY",
        "source_geometry_sha256": "3" * 64,
        "state_payload_sha256": digest(state),
        "task_payload": task,
        "task_payload_sha256": digest(task),
        "initial_state": state,
    }


class SpinInitialStateMapperTests(unittest.TestCase):
    def test_valid_explicit_record_maps(self):
        mapped = map_spin_initial_state_record(fixture())
        self.assertEqual(mapped.context.beta_anchor.shape, (10,))
        self.assertEqual(mapped.context.original_moments.shape, (10,))
        self.assertEqual(mapped.source_evidence, "AUDITED_REAL_GEOMETRY")

    def test_reconstructed_demo_is_rejected(self):
        record = fixture()
        record["evidence"] = "RECONSTRUCTED_ASSET"
        with self.assertRaisesRegex(ValueError, "cannot be admitted"):
            map_spin_initial_state_record(record)

    def test_task_hash_mismatch_is_rejected(self):
        record = fixture()
        record["task_payload"]["density"] = 2.0
        with self.assertRaisesRegex(ValueError, "TaskSpec payload SHA-256 mismatch"):
            map_spin_initial_state_record(record)

    def test_state_hash_mismatch_is_rejected(self):
        record = fixture()
        record["initial_state"]["length_scale"] = 2.0
        with self.assertRaisesRegex(ValueError, "initial-state payload SHA-256 mismatch"):
            map_spin_initial_state_record(record)

    def test_frame_mismatch_is_rejected(self):
        record = fixture()
        record["initial_state"]["coordinate_frame"] = "model_frame"
        record["state_payload_sha256"] = digest(record["initial_state"])
        with self.assertRaisesRegex(ValueError, "task_frame"):
            map_spin_initial_state_record(record)

    def test_axis_mismatch_is_rejected(self):
        record = fixture()
        record["initial_state"]["axis"] = [1.0, 0.0, 0.0]
        record["state_payload_sha256"] = digest(record["initial_state"])
        with self.assertRaisesRegex(ValueError, "axis does not match"):
            map_spin_initial_state_record(record)


if __name__ == "__main__":
    unittest.main()
