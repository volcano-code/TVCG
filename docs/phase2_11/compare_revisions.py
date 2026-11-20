"""Run this file with each revision on PYTHONPATH. See REPRODUCE.md."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig, PcraftNativeMomentOperator, geometry_from_columns,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_m80_sqno import UnifiedM80SQNO


def measure(head_offset=0., *, higher=True):
    torch.manual_seed(211211)
    torch.set_num_threads(1)
    cfg = NativePCRAFTConfig(grid_size=8, pino_width=2, pino_modes=1, pino_layers=2,
        interval_width=8, interval_depth=1, chunk_intervals=3, gno_neighbors=4,
        multiscale_enabled=True, multiscale_grids=(8, 16), gno_enabled=True,
        transolver_enabled=True, transolver_heads=2, transolver_slices=4,
        uq_enabled=True, event_validity_head_enabled=True)
    model = UnifiedM80SQNO(PcraftNativeMomentOperator(cfg),
        SpinQuadrANNV12(pcraft_width=8, hidden_dim=4, final_hidden_dim=8,
            neighbors=4, num_layers=2, chunk_intervals=3, chunk_nodes=19,
            query_chunk=2, fusion_chunk_size=3),
        cross_attention_enabled=True, cross_attention_chunk=3,
        cross_scale_attention_enabled=True, interaction_variant="parallel_gno_attention",
        fusion_mode="query_conditioned").train()
    # Zero-initialized heads would hide most attention and gate derivative errors.
    with torch.no_grad():
        for head in (model.geometry_physics_attention.output[-1],
                     model.cross_scale_attention.out[-1], model.query_fusion_gate[-1]):
            head.weight.normal_(std=0.08)
    columns = np.array([[x, y, 210, 340 + x % 15] for x in (180, 181, 182, 183)
                       for y in (150, 151)] + [[700, 300, 100, 295]], dtype=np.int32)
    geom = geometry_from_columns(dict(columns=columns, side=1024, h=.002,
                                      origin=np.array([-.43, -.38, -.50])))
    direction = torch.randn_like(model.geometry_physics_attention.output[-1].weight)
    with torch.no_grad():
        model.geometry_physics_attention.output[-1].weight.add_(direction, alpha=head_offset)
    q = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
    cache = model.encode_geometry(geom)
    out = model.evaluate_queries(cache, q, create_graph=True)
    results = {k: out[k] for k in ("moments", "jacobian_coefficients", "m80_moments",
               "sqno_moments", "sqno_jacobian_q", "m80_jacobian_q")}
    results["interval_features"] = cache.shared_cache.interval_features
    results["probabilities"] = cache.sqno_cache.probabilities
    actual_j = torch.stack([torch.autograd.grad(out["moments"][:, i].sum(), q,
                            retain_graph=True)[0] for i in range(10)], dim=1)
    torch.testing.assert_close(actual_j, out["jacobian_coefficients"], rtol=2e-5, atol=3e-7)
    objective = sum((out[k] * torch.randn_like(out[k])).sum()
                    for k in ("moments", "jacobian_coefficients"))
    names, params = zip(*model.named_parameters())
    first = torch.autograd.grad(objective, params, allow_unused=True, create_graph=True)
    for name, gradient in zip(names, first):
        if gradient is not None:
            results["grad/" + name] = gradient
    # A mixed Hessian column block: propagate an attention-head direction
    # through the full loss graph to every reachable parameter in the trunk.
    if not higher:
        return {k: v.detach() for k, v in results.items() if k.startswith("grad/")}
    idx = names.index("geometry_physics_attention.output.3.weight")
    probe = (first[idx] * direction).sum()
    for name, value in zip(names, torch.autograd.grad(probe, params, allow_unused=True)):
        if value is not None:
            results["hvp/" + name] = value
    return {k: v.detach().cpu() for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    values = measure()
    if args.record:
        # In the baseline torch kernel the double backward of LayerNorm is
        # sufficient for J-loss.grad, but differentiating that implementation
        # again does not reproduce the mixed derivative. This loss is affine
        # in the final attention weight, making its central difference exact
        # (up to rounding) for every other parameter at these fixed inputs.
        plus = measure(0.5, higher=False)
        minus = measure(-0.5, higher=False)
        max_reference_discrepancy = 0.
        for key in [k for k in values if k.startswith("hvp/")]:
            grad_key = key.replace("hvp/", "grad/", 1)
            numerical = plus[grad_key] - minus[grad_key]
            max_reference_discrepancy = max(max_reference_discrepancy,
                (values[key] - numerical).abs().max().item())
            values[key] = numerical
        print(json.dumps({"baseline_autograd_HVP_vs_central_difference_max": max_reference_discrepancy}))
        torch.save(values, args.record)
    if args.compare:
        expected = torch.load(args.compare, weights_only=True)
        if values.keys() != expected.keys():
            raise AssertionError(sorted(set(values).symmetric_difference(expected)))
        errors = {}
        for key, value in values.items():
            torch.testing.assert_close(value, expected[key], rtol=2e-4, atol=5e-7,
                                       msg=lambda message: f"{key}\n{message}")
            group = key.split("/")[0] if "/" in key else "output"
            errors[group] = max(errors.get(group, 0.), (value - expected[key]).abs().max().item())
        print(json.dumps(dict(compared=len(values), absolute_max_errors=errors,
                              rtol=2e-4, atol=5e-7), indent=2))


if __name__ == "__main__":
    main()
