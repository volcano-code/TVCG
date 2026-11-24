from __future__ import annotations

import unittest
from dataclasses import asdict

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    architecture_manifest,
    build_model,
    get_config,
)


class UnifiedFactoryContractTests(unittest.TestCase):
    @staticmethod
    def _small_overrides():
        return {
            "pcraft_preset": "smoke",
            "pcraft_grid_size": 8,
            "multiscale_grids": (8, 16),
            "sqno_hidden_dim": 8,
            "sqno_final_hidden_dim": 16,
            "sqno_num_layers": 1,
            "sqno_neighbors": 2,
            "sqno_chunk_intervals": 16,
            "sqno_chunk_nodes": 128,
            "query_chunk": 2,
        }

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

    def test_invalid_quadrature_orders_rejected_before_build(self):
        for order in (True, 1, 2.0, "2"):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "quadrature_order"):
                get_config("compact_v1_candidate", quadrature_order=order)

    def test_existing_config_fingerprints_and_metadata_remain_unchanged(self):
        expected = {
            "reference_v1": "579733085cf25b7b9a3387e85f5abaa00c62176dbbbc125de52d0b5ac2489769",
            "full_v1_candidate": "10d79334834607823243b34d393c96e3d018b583e10a1b498bbfac7507a1a63e",
        }
        added_fields = {
            "attention_mixing", "pcraft_grid_size", "multiscale_grids",
            "pcraft_gno_enabled", "pcraft_transolver_enabled",
        }
        for name, fingerprint in expected.items():
            with self.subTest(name=name):
                cfg = get_config(name)
                manifest = architecture_manifest(name)
                self.assertEqual(cfg.attention_mixing, "legacy")
                self.assertEqual(cfg.fingerprint(), fingerprint)
                self.assertEqual(manifest["config_fingerprint"], fingerprint)
                self.assertFalse(added_fields.intersection(manifest["config"]))
                self.assertEqual(manifest["config"], cfg.config_payload())

    def test_gno_transolver_arms_change_only_the_two_native_switches(self):
        base = architecture_manifest("ablation_g1_t1")
        base_native = dict(base["pcraft_config"])
        base_config = dict(base["config"])
        for key in ("name", "pcraft_gno_enabled", "pcraft_transolver_enabled"):
            base_config.pop(key)
        for gno, transolver in ((False, False), (True, False), (False, True), (True, True)):
            name = f"ablation_g{int(gno)}_t{int(transolver)}"
            with self.subTest(name=name):
                manifest = architecture_manifest(name)
                expected_native = dict(base_native, gno_enabled=gno, transolver_enabled=transolver)
                self.assertEqual(manifest["pcraft_config"], expected_native)
                config = dict(manifest["config"])
                for key in ("name", "pcraft_gno_enabled", "pcraft_transolver_enabled"):
                    config.pop(key)
                self.assertEqual(config, base_config)
                self.assertEqual(manifest["effective_geometry"]["grids"], [32, 64, 128])
                self.assertEqual(manifest["effective_attention_mixing"], "gated")

    def test_no_cross_scale_is_a_single_switch_with_the_same_mixing_contract(self):
        base = architecture_manifest("ablation_g1_t1")
        no_cross = architecture_manifest("ablation_no_cross_scale")
        expected = dict(base["config"], name="ablation_no_cross_scale", cross_scale_attention=False)
        self.assertEqual(no_cross["config"], expected)
        self.assertEqual(no_cross["pcraft_config"], base["pcraft_config"])
        model = build_model("ablation_no_cross_scale", config_overrides=self._small_overrides())
        self.assertIsNone(model.cross_scale_attention)
        self.assertIsNotNone(model.geometry_physics_attention)
        self.assertIsNotNone(model.parallel_interaction_logit)
        self.assertEqual(model.attention_mixing, "gated")

    def test_scale_ablations_report_effective_scales_and_retain_other_settings(self):
        base = architecture_manifest("ablation_g1_t1")
        for name, grids in (("ablation_two_scales", (64, 128)), ("ablation_one_scale", (64,))):
            with self.subTest(name=name):
                manifest = architecture_manifest(name)
                self.assertEqual(manifest["config"], dict(base["config"], name=name, multiscale_grids=grids))
                self.assertEqual(manifest["effective_geometry"]["grids"], list(grids))
                self.assertEqual(manifest["pcraft_config"]["grid_size"], 64)
                self.assertEqual(manifest["pcraft_config"]["multiscale_enabled"], len(grids) > 1)
                self.assertTrue(manifest["modules"]["cross_scale_attention"])
                self.assertEqual(manifest["effective_attention_mixing"], "gated")
        one = architecture_manifest("ablation_one_scale")
        self.assertFalse(one["modules"]["global_scale_gate"])
        self.assertFalse(one["modules"]["local_scale_gate"])
        # This tuple is retained only to satisfy the native configuration's
        # validation; the manifest must not present it as three active scales.
        self.assertEqual(one["pcraft_config"]["multiscale_grids"], (32, 64, 128))

    def test_compact_build_removes_shared_extras_and_preserves_sqno_and_query_heads(self):
        model = build_model("compact_v1_candidate", config_overrides=self._small_overrides())
        native = model.shared_encoder
        self.assertFalse(native.config.gno_enabled)
        self.assertFalse(native.config.transolver_enabled)
        for name in ("gno_kernel", "gno_value", "gno_update", "transolver", "uq_head", "event_validity_head"):
            self.assertFalse(hasattr(native, name), name)
        self.assertIsNone(model.sqno.pcraft)
        self.assertIsNotNone(model.sqno.gnn)
        self.assertTrue(hasattr(native, "query_head"))
        self.assertIsNotNone(model.query_fusion_gate)
        self.assertIsNotNone(model.geometry_physics_attention)
        self.assertIsNone(model.cross_scale_attention)
        self.assertIsNone(model.parallel_interaction_logit)
        self.assertEqual(model.attention_mixing, "sum")
        self.assertEqual(native.config.multiscale_grids, (8, 16))
        self.assertEqual(model.sqno.quadrature_order, get_config("full_v1_candidate").quadrature_order)

    def test_each_factorial_model_instantiates_only_requested_interactions(self):
        for gno, transolver in ((False, False), (True, False), (False, True), (True, True)):
            name = f"ablation_g{int(gno)}_t{int(transolver)}"
            with self.subTest(name=name):
                model = build_model(name, config_overrides=self._small_overrides())
                self.assertEqual(hasattr(model.shared_encoder, "gno_kernel"), gno)
                self.assertEqual(hasattr(model.shared_encoder, "transolver"), transolver)
                self.assertIsNotNone(model.geometry_physics_attention)
                self.assertIsNotNone(model.cross_scale_attention)
                self.assertIsNotNone(model.parallel_interaction_logit)
                self.assertIsNone(model.sqno.pcraft)

    def test_manifest_uses_exact_build_overrides_without_mutating_registry(self):
        overrides = self._small_overrides()
        overrides["multiscale_grids"] = [8, 16]
        manifest = architecture_manifest("compact_v1_candidate", config_overrides=overrides)
        model = build_model("compact_v1_candidate", config_overrides=overrides)
        self.assertEqual(manifest["pcraft_config"], asdict(model.shared_encoder.config))
        self.assertEqual(manifest["effective_geometry"]["grids"], [8, 16])
        self.assertEqual(manifest["config"], model.architecture_metadata["config"])
        self.assertEqual(manifest["config_fingerprint"], model.architecture_metadata["config_fingerprint"])
        self.assertNotEqual(manifest["config_fingerprint"], get_config("compact_v1_candidate").fingerprint())
        self.assertEqual(get_config("compact_v1_candidate").multiscale_grids, (64, 128))

    def test_one_scale_override_builds_an_actual_single_scale_encoder(self):
        overrides = dict(self._small_overrides(), multiscale_grids=(8,))
        model = build_model("ablation_one_scale", config_overrides=overrides)
        manifest = architecture_manifest("ablation_one_scale", config_overrides=overrides)
        self.assertFalse(model.shared_encoder.config.multiscale_enabled)
        self.assertEqual(model.shared_encoder.config.grid_size, 8)
        self.assertFalse(hasattr(model.shared_encoder, "pino_scale_score"))
        self.assertFalse(hasattr(model.shared_encoder, "local_scale_gate"))
        self.assertEqual(manifest["effective_geometry"]["grids"], [8])

    def test_invalid_geometry_overrides_fail_before_model_allocation(self):
        for overrides in (
            {"pcraft_grid_size": True},
            {"pcraft_grid_size": 10},
            {"multiscale_grids": ()},
            {"multiscale_grids": (64, 32)},
            {"multiscale_grids": (64, 64)},
            {"pcraft_grid_size": 32, "multiscale_grids": (64,)},
            {"pcraft_grid_size": 32, "multiscale_grids": (64, 128)},
            {"pcraft_gno_enabled": 1},
            {"pcraft_transolver_enabled": "false"},
            {"attention_mixing": "unknown"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                architecture_manifest("ablation_g1_t1", config_overrides=overrides)

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
