from __future__ import annotations

import unittest

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    architecture_manifest,
    build_model,
    get_config,
)


class UnifiedFactoryContractTests(unittest.TestCase):
    def test_named_configs_are_explicit_and_fingerprinted(self):
        reference = architecture_manifest("reference_v1")
        full = architecture_manifest("full_v1_candidate")
        self.assertNotEqual(reference["config_fingerprint"], full["config_fingerprint"])
        self.assertTrue(reference["modules"]["shared_pcraft"])
        self.assertTrue(reference["modules"]["m80"])
        self.assertTrue(reference["modules"]["sqno"])
        self.assertFalse(reference["modules"]["geometry_physics_cross_attention"])
        self.assertTrue(full["modules"]["gno"])
        self.assertTrue(full["modules"]["transolver_physics_slicing"])
        self.assertTrue(full["modules"]["geometry_physics_cross_attention"])
        self.assertTrue(full["modules"]["cross_scale_attention"])
        self.assertTrue(full["modules"]["query_conditioned_fusion"])
        self.assertTrue(full["modules"]["moment_jacobian_uq"])
        self.assertTrue(full["modules"]["event_validity"])
        self.assertEqual(
            full["scientific_status"],
            "candidate_configuration_not_performance_validated",
        )

    def test_config_override_does_not_mutate_registry(self):
        original = get_config("reference_v1")
        changed = get_config("reference_v1", sqno_hidden_dim=12)
        self.assertEqual(original.sqno_hidden_dim, 32)
        self.assertEqual(changed.sqno_hidden_dim, 12)
        self.assertNotEqual(original.fingerprint(), changed.fingerprint())

    def test_unknown_config_rejected(self):
        with self.assertRaises(KeyError):
            get_config("not-a-config")

    def test_smoke_factory_builds_shared_encoder_model(self):
        model = build_model(
            "reference_v1",
            config_overrides={
                "pcraft_preset": "smoke",
                "sqno_hidden_dim": 8,
                "sqno_final_hidden_dim": 16,
                "sqno_num_layers": 1,
                "sqno_neighbors": 2,
                "sqno_chunk_intervals": 16,
                "sqno_chunk_nodes": 128,
                "query_chunk": 2,
                "uq_enabled": False,
                "event_enabled": False,
            },
        )
        self.assertIs(model.sqno.pcraft, None)
        self.assertTrue(model.sqno.external_pcraft_required)
        self.assertEqual(
            model.shared_encoder.config.interval_width,
            model.sqno.pcraft_width,
        )
        contract = model.architecture_contract()
        self.assertEqual(contract["metadata"]["named_config"], "reference_v1")


if __name__ == "__main__":
    unittest.main()
