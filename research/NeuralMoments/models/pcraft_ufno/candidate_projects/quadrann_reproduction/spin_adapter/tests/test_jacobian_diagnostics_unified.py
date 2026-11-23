"""Small real-model audit; this is not a trained-checkpoint accuracy result."""
import torch

from spin_adapter.jacobian_diagnostics import audit_jacobian_paths, compare_jacobians
from spin_adapter.tests.test_unified_m80_sqno import make_geometry, make_unified, queries


def test_unified_reported_jacobian_matches_moment_ad_and_fd_and_detects_missing_gate_term():
    model = make_unified(cross_attention_enabled=True, cross_attention_chunk=2,
        cross_scale_attention_enabled=True, fusion_mode="query_conditioned",
        query_fusion_hidden=8).eval()
    torch.manual_seed(1842)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight, std=.02)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].bias, std=.001)
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight, std=.02)
    torch.nn.init.normal_(model.query_fusion_gate[-1].weight, std=.4)
    # Keep the geometry cache differentiable for the independent Moment AD.
    # The existing tiny fixture has one scale token; it exercises the active
    # cross-scale correction, not a full multi-resolution training benchmark.
    cache = model.encode_geometry(make_geometry())
    raw = queries()[1:2] * torch.tensor([[.7], [1.4]], dtype=torch.float64)
    with torch.no_grad():
        reported = model.evaluate_queries(cache, raw)
        geometry_correction, _, _ = model.geometry_physics_attention.forward_with_query_jacobian(
            raw, cache.shared_cache.interval_features, cache.shared_cache.interval_mass)
        scale_correction, _ = model.cross_scale_attention(
            raw, cache.shared_cache.scale_features, cache.shared_cache.scale_ids)
    assert geometry_correction.norm() > 0
    assert scale_correction.norm() > 0
    assert (reported["fusion_gate_jacobian_norm"] > 0).all()

    def moments_and_gate(coefficients):
        result = model.evaluate_queries(cache, coefficients, create_graph=True)
        # AD re-differentiates returned Moments; it never reads the reported J.
        return result["moments"], result["fusion_weight"]

    moment_ad, gate_ad = torch.autograd.functional.jacobian(moments_and_gate, raw)
    diagonal_ad = torch.stack([moment_ad[q, :, q, :] for q in range(len(raw))])
    gate_diagonal = torch.stack([gate_ad[q, q, :] for q in range(len(raw))])
    torch.testing.assert_close(reported["jacobian_coefficients"], diagonal_ad,
                               rtol=2e-6, atol=2e-8)
    assert torch.count_nonzero(moment_ad[0, :, 1, :]) == 0
    assert torch.count_nonzero(moment_ad[1, :, 0, :]) == 0
    # These non-unit raw inputs must retain the inverse-scale chain rule.
    torch.testing.assert_close(reported["jacobian_coefficients"][0] / 2,
                               reported["jacobian_coefficients"][1], rtol=2e-6, atol=2e-8)

    omitted_gate_term = ((reported["m80_moments"] - reported["sqno_moments"]).unsqueeze(-1)
                        * gate_diagonal.unsqueeze(1))
    assert omitted_gate_term.norm() > 1e-3
    wrong_jacobian = reported["jacobian_coefficients"].detach() - omitted_gate_term
    ad_negative = compare_jacobians(wrong_jacobian, diagonal_ad)
    assert ad_negative["relative_frobenius"] > 1e-3

    # Both audits use the same model Moment values. Deliberately dropping the
    # gate chain-rule term must be diagnosed, rather than a toy function only
    # being compared against its own hand-written derivative.
    cached_moments = {}

    def moment_fn(coefficients):
        key = tuple(coefficients.flatten().tolist())
        if key not in cached_moments:
            cached_moments[key] = model.evaluate_queries(cache, coefficients)["moments"].detach()
        return cached_moments[key]

    kwargs = dict(step_sizes=(2e-4, 1e-4), query_ids=["synthetic_raw_0.7", "synthetic_raw_1.4"],
                  assume_query_independence=True)
    correct = audit_jacobian_paths(moment_fn,
        lambda coefficients: model.evaluate_queries(cache, coefficients)["jacobian_coefficients"],
        raw, **kwargs)
    incorrect = audit_jacobian_paths(moment_fn, lambda _: wrong_jacobian, raw, **kwargs)
    for step, good in correct["model_self_consistency"].items():
        bad = incorrect["model_self_consistency"][step]
        assert good["relative_frobenius"] < 2e-4
        assert good["max_absolute"] < 1e-3
        assert bad["relative_frobenius"] > 1e-3
        assert bad["relative_frobenius"] > 10 * good["relative_frobenius"]
        for query in good["per_query"]:
            assert query["relative_frobenius"] < 2e-4
    assert correct["native_physical_accuracy"] is None
    assert model.geometry_encode_calls == 1
