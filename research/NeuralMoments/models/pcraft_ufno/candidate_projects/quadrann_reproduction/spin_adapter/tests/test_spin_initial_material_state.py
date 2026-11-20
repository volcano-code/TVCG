from __future__ import annotations

import unittest

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_initial_material_state import (
    SCHEMA,
    validate_initial_material_state,
)


def fixture():
    dims = [2, 2, 2]
    minimum = [-1.0, -1.0, 0.0]
    maximum = [1.0, 1.0, 2.0]
    spacing = (np.asarray(maximum) - np.asarray(minimum)) / np.asarray(dims)
    mask = [1] * 8
    moments = [8.0, 0.0, 0.0, 8.0, 8.0 / 3.0, 8.0 / 3.0, 32.0 / 3.0, 0.0, 0.0, 0.0]
    return {
        "initial_state": {
            "schema": SCHEMA,
            "scope": "INITIAL_SOURCE_DOMAIN_NOT_FINAL_OPTIMIZED_MATERIAL",
            "domain_artifact_id": "fixture-domain",
            "grid_dimensions": dims,
            "grid_minimum_model": minimum,
            "grid_maximum_model": maximum,
            "grid_spacing_model": spacing.tolist(),
            "cell_order": "x + nx*(y + ny*z)",
            "source_material_mask": mask,
            "source_carveable_mask": [0, 0, 0, 0, 1, 1, 1, 1],
            "fixed_material_mask": [1, 1, 1, 1, 0, 0, 0, 0],
            "design_domain_mask": [0, 0, 0, 0, 1, 1, 1, 1],
            "initial_geometric_moments_unit_density_model": moments,
            "initial_geometric_moments_unit_density_target": moments,
            "model_to_target_linear": np.eye(3).tolist(),
            "model_to_target_translation": [0.0, 0.0, 0.0],
            "moment_semantics": "geometric_volume_moments_at_unit_density_in_source_coordinate_units",
            "physical_si_scale_registered": False,
        }
    }


class SpinInitialMaterialStateTests(unittest.TestCase):
    def test_valid_export_and_moments(self):
        result = validate_initial_material_state(fixture())
        self.assertEqual(result.grid_dimensions, (2, 2, 2))
        self.assertAlmostEqual(result.moments_model[0], 8.0)
        self.assertAlmostEqual(result.moments_model[3], 8.0)
        self.assertFalse(result.si_scale_registered)

    def test_rejects_mask_partition_mismatch(self):
        data = fixture()
        data["initial_state"]["fixed_material_mask"][0] = 0
        with self.assertRaisesRegex(ValueError, "fixed_material_mask"):
            validate_initial_material_state(data)

    def test_rejects_wrong_cell_order(self):
        data = fixture()
        data["initial_state"]["cell_order"] = "z-fast"
        with self.assertRaisesRegex(ValueError, "cell ordering"):
            validate_initial_material_state(data)

    def test_rejects_moment_tampering(self):
        data = fixture()
        data["initial_state"]["initial_geometric_moments_unit_density_model"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "independent recomputation"):
            validate_initial_material_state(data)

    def test_rejects_bad_transform(self):
        data = fixture()
        data["initial_state"]["model_to_target_linear"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_initial_material_state(data)


if __name__ == "__main__":
    unittest.main()
