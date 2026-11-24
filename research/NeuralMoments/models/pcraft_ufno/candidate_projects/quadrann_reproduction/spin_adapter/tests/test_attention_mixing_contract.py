"""Small real-model contracts for independently selectable attention mixing."""
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.jacobian_diagnostics import (
    compare_jacobians, finite_difference_jacobian,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_m80_sqno import UnifiedM80SQNO
from spin_adapter.tests.test_unified_m80_sqno import make_geometry, make_unified, queries


def _model(*, mixing="gated", gno=False, geometry=True, scale=True,
           variant="independent", fusion="global_scalar"):
    base = make_unified(gno_enabled=gno)
    return UnifiedM80SQNO(base.shared_encoder, base.sqno,
        cross_attention_enabled=geometry, cross_attention_chunk=2,
        cross_scale_attention_enabled=scale, interaction_variant=variant,
        attention_mixing=mixing, fusion_mode=fusion, query_fusion_hidden=8)


def _activate(model):
    torch.manual_seed(1842)
    if model.geometry_physics_attention is not None:
        torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight, std=.02)
        torch.nn.init.normal_(model.geometry_physics_attention.output[-1].bias, std=.001)
    if model.cross_scale_attention is not None:
        torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight, std=.02)
    if model.query_fusion_gate is not None:
        torch.nn.init.normal_(model.query_fusion_gate[-1].weight, std=.4)
    if model.parallel_interaction_logit is not None:
        with torch.no_grad():
            model.parallel_interaction_logit.copy_(torch.logit(torch.tensor(.23, dtype=torch.float64)))


def _inference(model, raw):
    with torch.no_grad():
        cache = model.encode_geometry(make_geometry(), inference=True)
        return model.evaluate_queries(cache, raw)


def _assert_same_outputs(a, b):
    assert set(a) == set(b)
    for name, value in a.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, b[name], rtol=0, atol=0)
        else:
            assert value == b[name]


def test_attention_mixing_defaults_to_legacy_and_rejects_unknown_modes():
    assert make_unified().attention_mixing == "legacy"
    with pytest.raises(ValueError, match="attention_mixing"):
        _model(mixing="automatic")
    # Selecting the new mixing mode does not weaken historical variant rules.
    with pytest.raises(ValueError, match="requires shared_encoder"):
        _model(mixing="legacy", variant="parallel_gno_attention", gno=False)
    with pytest.raises(ValueError, match="both attention branches"):
        _model(mixing="legacy", variant="parallel_gno_attention", gno=True, scale=False)


@pytest.mark.parametrize("gno", [False, True])
@pytest.mark.parametrize("mixing", ["legacy", "sum", "gated"])
def test_no_attention_registers_no_unused_mixing_gate(gno, mixing):
    model = _model(mixing=mixing, gno=gno, geometry=False, scale=False)
    assert model.parallel_interaction_logit is None
    assert "parallel_interaction_logit" not in model.state_dict()
    assert all(name != "parallel_interaction_logit" for name, _ in model.named_parameters())
    assert model.architecture_contract()["attention_mixing"] == mixing


@pytest.mark.parametrize("geometry,scale,coefficient", [(True, False, .23), (False, True, .77)])
def test_single_attention_preserves_its_fixed_gate_slot(geometry, scale, coefficient):
    gated = _model(geometry=geometry, scale=scale).eval()
    _activate(gated)
    summed = _model(mixing="sum", geometry=geometry, scale=scale).eval()
    summed.load_state_dict({name: value for name, value in gated.state_dict().items()
                           if name != "parallel_interaction_logit"}, strict=True)
    assert summed.parallel_interaction_logit is None
    raw = queries()[1:2] * .7
    a, b = _inference(gated, raw), _inference(summed, raw)
    alpha = a["fusion_weight"]
    core_m = alpha * a["m80_moments"] + (1 - alpha) * a["sqno_moments"]
    core_j = alpha * a["m80_jacobian_coefficients"] + (1 - alpha) * a["sqno_jacobian_coefficients"]
    residual_m, residual_j = b["moments"] - core_m, b["jacobian_coefficients"] - core_j
    assert residual_m.norm() > 1e-6 and residual_j.norm() > 1e-8
    torch.testing.assert_close(a["moments"], core_m + coefficient * residual_m,
                               rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(a["jacobian_coefficients"], core_j + coefficient * residual_j,
                               rtol=2e-6, atol=2e-8)
    # Renormalizing the sole surviving branch to weight 1 would change output.
    assert (a["moments"] - b["moments"]).norm() > 1e-6
    assert a["parallel_attention_gate"].item() == pytest.approx(.23)
    assert "parallel_attention_gate" not in b


def test_legacy_full_and_independent_gated_are_identical_with_the_same_state():
    legacy = make_unified(gno_enabled=True, cross_attention_enabled=True,
        cross_attention_chunk=2, cross_scale_attention_enabled=True,
        interaction_variant="parallel_gno_attention", fusion_mode="query_conditioned",
        query_fusion_hidden=8).eval()
    _activate(legacy)
    independent = _model(gno=True, fusion="query_conditioned").eval()
    independent.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.attention_mixing == "legacy"
    assert independent.interaction_variant == "independent"
    assert set(legacy.state_dict()) == set(independent.state_dict())
    raw = queries()[1:2] * torch.tensor([[.7], [1.4]], dtype=torch.float64)
    _assert_same_outputs(_inference(legacy, raw), _inference(independent, raw))


def test_gno_off_gated_attention_has_complete_raw_jacobian_with_nonzero_heads():
    model = _model(gno=False, fusion="query_conditioned").eval()
    _activate(model)
    assert not model.shared_encoder.config.gno_enabled
    cache = model.encode_geometry(make_geometry())
    raw = queries()[1:2] * .7
    with torch.no_grad():
        reported = model.evaluate_queries(cache, raw)
        geometry_correction, _, _ = model.geometry_physics_attention.forward_with_query_jacobian(
            raw, cache.shared_cache.interval_features, cache.shared_cache.interval_mass)
        scale_correction, _ = model.cross_scale_attention(
            raw, cache.shared_cache.scale_features, cache.shared_cache.scale_ids)
    assert geometry_correction.norm() > 0 and scale_correction.norm() > 0
    assert reported["fusion_gate_jacobian_norm"].min() > 0
    full_ad = torch.autograd.functional.jacobian(
        lambda c: model.evaluate_queries(cache, c, create_graph=True)["moments"], raw)
    torch.testing.assert_close(reported["jacobian_coefficients"], full_ad[:, :, 0, :],
                               rtol=2e-6, atol=2e-8)
    for step in (2e-4, 1e-4):
        fd = finite_difference_jacobian(
            lambda c: model.evaluate_queries(cache, c)["moments"], raw, step=step)
        report = compare_jacobians(reported["jacobian_coefficients"], fd)
        assert report["relative_frobenius"] < 2e-4
        assert report["max_absolute"] < 1e-3
    with torch.no_grad():
        scaled = model.evaluate_queries(cache, raw * 2)
    torch.testing.assert_close(reported["moments"], scaled["moments"], rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(reported["jacobian_coefficients"] / 2,
                               scaled["jacobian_coefficients"], rtol=2e-6, atol=2e-8)
    assert model.geometry_encode_calls == 1


@pytest.mark.parametrize("gno", [False, True])
def test_gated_joint_backward_reaches_enabled_modules_and_gate(gno):
    model = _model(gno=gno, fusion="query_conditioned").train()
    _activate(model)
    cache = model.encode_geometry(make_geometry())
    out = model.evaluate_queries(cache, queries()[1:2] * .7, create_graph=True)
    (out["moments"].square().mean() + .01 * out["jacobian_coefficients"].square().mean()).backward()
    modules = (model.shared_encoder, model.sqno, model.geometry_physics_attention,
               model.cross_scale_attention, model.query_fusion_gate)
    for module in modules:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert model.parallel_interaction_logit.grad is not None
    assert torch.isfinite(model.parallel_interaction_logit.grad)
    assert model.parallel_interaction_logit.grad.abs() > 0


def test_explicit_gated_checkpoint_roundtrip_and_mode_mismatch(tmp_path):
    model = _model(fusion="query_conditioned").eval()
    _activate(model)
    raw = queries()[1:2] * .7
    before = _inference(model, raw)
    path = tmp_path / "gated.pt"
    model.save_checkpoint(path, data_fingerprint="synthetic-attention-mixing")
    restored = _model(fusion="query_conditioned").eval()
    payload = restored.load_checkpoint(path, expected_data_fingerprint="synthetic-attention-mixing")
    assert payload["architecture"]["attention_mixing"] == "gated"
    _assert_same_outputs(before, _inference(restored, raw))
    with pytest.raises(ValueError, match="architecture contract"):
        _model(mixing="sum", fusion="query_conditioned").load_checkpoint(path)


def test_missing_mixing_field_means_legacy_but_does_not_bypass_source_checks(tmp_path):
    kwargs = dict(gno=True, variant="parallel_gno_attention")
    model = _model(mixing="legacy", **kwargs)
    path = tmp_path / "legacy-metadata.pt"
    model.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["architecture"].pop("attention_mixing")
    torch.save(payload, path)
    _model(mixing="legacy", **kwargs).load_checkpoint(path)
    # All parameter names also exist in this target, but its explicit
    # architecture selection must not be silently treated as legacy.
    with pytest.raises(ValueError, match="architecture contract"):
        _model(mixing="gated", **kwargs).load_checkpoint(path)
    payload["source_fingerprints"]["unified_m80_sqno.py"] = "different-source"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="source fingerprint"):
        _model(mixing="legacy", **kwargs).load_checkpoint(path)


def test_sum_and_legacy_checkpoint_modes_remain_distinct_even_with_equal_parameters(tmp_path):
    model = _model(mixing="sum")
    legacy = _model(mixing="legacy")
    assert set(model.state_dict()) == set(legacy.state_dict())
    path = tmp_path / "sum.pt"
    model.save_checkpoint(path)
    with pytest.raises(ValueError, match="architecture contract"):
        legacy.load_checkpoint(path)
