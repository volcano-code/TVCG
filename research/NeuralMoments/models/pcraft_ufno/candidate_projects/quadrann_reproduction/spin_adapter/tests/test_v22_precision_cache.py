"""Production cached/streamed response agreement on real native256 cell packs.

These are integration checks for quadrature selection, not evidence of native
teacher accuracy. The local learned response and its trainable static context
are evaluated through the production reducers; no alternate M/J reduction is
implemented in this test module.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from spin_adapter.compact_pino_v21 import evaluate_packed, _evaluate_prepared_packed
from spin_adapter.v21_blocks import BlockMeasure, basis_at
from spin_adapter.v21_material import prepare_rle_material
from spin_adapter.v21_native import FiniteOperand
from spin_adapter.v21_response import LocalRetention, prepare_surface
from spin_adapter.v22_geometry_inputs import GeometryInputCache, material_signature
from spin_adapter.v22_query_cache import QueryPreparationCache


def _operand(operation="intersection"):
    cut = 0.5007
    low, high = [-1., -1., -1.], [cut, 2., 2.]
    vertices = np.array([[low[0], low[1], low[2]], [high[0], low[1], low[2]],
                         [high[0], high[1], low[2]], [low[0], high[1], low[2]],
                         [low[0], low[1], high[2]], [high[0], low[1], high[2]],
                         [high[0], high[1], high[2]], [low[0], high[1], high[2]]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    velocity = np.zeros((8, 3, 3))
    velocity[vertices[:, 0] == cut, 0, :] = [1., -0.4, 0.8]
    return FiniteOperand(vertices, faces, np.arange(len(faces)) % 2 == 0,
                         operation, velocity)


@pytest.fixture
def native_case(tmp_path):
    # Four cut-band full cells, plus one cell in each saturated region. The
    # production block pack is built from native runs and carries real source
    # provenance, rather than a handcrafted point cloud with invented metadata.
    columns = np.array([[70, 54, 81, 82], [127, 54, 81, 83],
                        [128, 105, 132, 134], [190, 157, 182, 183]])
    density = 1.7
    material = prepare_rle_material(columns, [0, 0, 0], 1 / 256,
                                    side=256, density=density)
    directory = tmp_path / "native256-pack"
    blocks = BlockMeasure.build(material, directory, block_voxels=8, chunk_size=5)
    fixed = basis_at(np.array([[0.1, 0.2, 0.3]]))[0] * material.moments()[0] * 0.07
    return blocks, _operand(), fixed, density, directory


def _models():
    torch.manual_seed(922173)
    response = LocalRetention(context_dim=3, hidden_dim=10).double()
    context = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.SiLU(),
                                  torch.nn.Linear(5, 3)).double()
    return response, context


def _gradients(result, parameters, mass, pitch):
    assert result.jacobian is not None
    moment_scale = torch.linspace(0.3, 1.2, 10, dtype=torch.float64)
    normalized_m = result.moments / mass
    normalized_j = result.jacobian * pitch / mass
    jacobian_loss = (normalized_j * moment_scale[:, None]).square().mean()
    # An M loss could conceal a detached J. Check J-only gradients separately.
    j_gradients = torch.autograd.grad(jacobian_loss, parameters, retain_graph=True)
    combined_loss = (normalized_m * moment_scale).square().mean() + jacobian_loss
    gradients = torch.autograd.grad(combined_loss, parameters)
    return gradients, j_gradients


@pytest.mark.parametrize("rule", ["gl2", "gl3"])
@pytest.mark.parametrize("operation", ["intersection", "difference"])
def test_refined_cache_and_streaming_preserve_m_j_and_training_gradients(native_case, rule, operation):
    blocks, operand, fixed, _, _ = native_case
    operand = replace(operand, operation=operation)
    response, context = _models()
    parameters = tuple(response.parameters()) + tuple(context.parameters())
    cache = QueryPreparationCache(max_bytes=2 * 1024**2)
    width, chunk = blocks.pitch, 7
    mass = blocks.moments[:, 0].sum()
    previous_moments = None
    for update in range(2):
        prepared = cache.get_or_prepare(blocks, operand, width, chunk_size=chunk,
                                        full_cell_rule=rule)
        assert prepared is not None and prepared.full_cell_rule == rule
        assert prepared.full.any() and prepared.band.any()
        actual = _evaluate_prepared_packed(prepared, operation, fixed,
                    operand.vertex_velocity, model=response, context=context, chunk_size=chunk)
        actual_grad, actual_j_grad = _gradients(actual, parameters, mass, blocks.pitch)
        expected = evaluate_packed(blocks, prepare_surface(operand.vertices, operand.faces),
                    width, operation, fixed, operand.vertex_velocity,
                    model=response, context=context, chunk_size=chunk, full_cell_rule=rule)
        expected_grad, expected_j_grad = _gradients(expected, parameters, mass, blocks.pitch)
        torch.testing.assert_close(actual.moments, expected.moments, rtol=0, atol=0)
        torch.testing.assert_close(actual.jacobian, expected.jacobian, rtol=0, atol=0)
        assert actual.diagnostics == expected.diagnostics
        assert actual.diagnostics["full_cell_rule"] == rule
        assert actual.diagnostics["bandwidth"] == blocks.pitch
        assert actual.diagnostics["ambiguous_nodes"] == 0
        for observed, reference in zip((*actual_grad, *actual_j_grad),
                                       (*expected_grad, *expected_j_grad)):
            assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
            torch.testing.assert_close(observed, reference, rtol=2e-13, atol=2e-15)
        # J-only loss must reach both response layers and the static context net.
        assert all(torch.linalg.vector_norm(g) > 0 for g in actual_j_grad)
        if previous_moments is not None:
            assert not torch.equal(actual.moments, previous_moments)
        previous_moments = actual.moments.detach().clone()
        if update == 0:
            with torch.no_grad():
                response.local[-1].bias[0] += 0.2
    assert cache.info()["builds"] == 1 and cache.info()["hits"] == 1


def test_switching_full_rule_misses_query_cache_and_reuses_static_geometry(native_case):
    blocks, operand, fixed, density, _ = native_case
    signature = material_signature(blocks)
    geometry_cache = GeometryInputCache(max_bytes=2 * 1024**2)
    geometry = geometry_cache.get(blocks, fixed, density, token_grid=8, grids=(8, 16))
    cache = QueryPreparationCache(max_bytes=2 * 1024**2)
    prepared = {}
    expected_counts = {"tetra4": 16, "gl2": 32, "gl3": 108}
    for rule in expected_counts:
        prepared[rule] = cache.get_or_prepare(blocks, operand, blocks.pitch,
                                               chunk_size=11, full_cell_rule=rule)
        assert prepared[rule].band_nodes == expected_counts[rule]
        assert prepared[rule].full_cell_rule == rule
        assert material_signature(blocks) == signature
        reused = geometry_cache.get(blocks, fixed, density, token_grid=8, grids=(8, 16))
        assert reused is geometry
    assert len({p.key for p in prepared.values()}) == 3
    assert cache.info()["misses"] == cache.info()["builds"] == 3
    assert cache.info()["hits"] == 0
    assert cache.get_or_prepare(blocks, operand, blocks.pitch, chunk_size=11,
                                full_cell_rule="gl2") is prepared["gl2"]
    assert cache.info()["hits"] == 1
    assert geometry_cache.snapshot()["builds"] == 1
    assert geometry_cache.snapshot()["hits"] == 3


@pytest.mark.parametrize("rule", ["gl2", "gl3"])
def test_refined_intersection_and_difference_conserve_moments_and_cancel_j(native_case, rule):
    blocks, operand, fixed, _, _ = native_case
    response, context = _models()
    cache = QueryPreparationCache()
    results = []
    with torch.no_grad():
        for operation in ("intersection", "difference"):
            query = replace(operand, operation=operation)
            prepared = cache.get_or_prepare(blocks, query, blocks.pitch,
                                             chunk_size=13, full_cell_rule=rule)
            results.append(_evaluate_prepared_packed(prepared, operation, fixed,
                query.vertex_velocity, model=response, context=context, chunk_size=13))
    inside, outside = results
    # Fixed shells occur once in each response, so they occur twice in the sum.
    expected = torch.from_numpy(2 * fixed + blocks.moments.sum(0))
    torch.testing.assert_close(inside.moments + outside.moments, expected,
                               rtol=3e-14, atol=1e-22)
    torch.testing.assert_close(inside.jacobian + outside.jacobian,
                               torch.zeros_like(inside.jacobian), rtol=0, atol=1e-19)
    assert cache.info()["misses"] == 2


@pytest.mark.parametrize("rule", ["gl2", "gl3"])
def test_production_paths_reject_refined_rule_on_a_legacy_pack(native_case, rule):
    blocks, operand, fixed, _, directory = native_case
    metadata = directory / "metadata.json"
    desc = json.loads(metadata.read_text())
    desc.pop("material_provenance")
    desc.pop("stored_full_cell_rule")
    metadata.write_text(json.dumps(desc))
    for name in ("full_runs", "partial_cells", "boundary_nodes"):
        (directory / f"material_{name}.npy").unlink()
    old = BlockMeasure.open(directory, identity=blocks.identity, block_voxels=8)
    assert old.material is None
    surface = prepare_surface(operand.vertices, operand.faces)
    default = evaluate_packed(old, surface, old.pitch, operand.operation,
                               fixed, operand.vertex_velocity)
    assert torch.isfinite(default.moments).all() and torch.isfinite(default.jacobian).all()
    with pytest.raises(ValueError, match="requires material provenance"):
        evaluate_packed(old, surface, old.pitch, operand.operation, fixed,
                        operand.vertex_velocity, full_cell_rule=rule)
    with pytest.raises(ValueError, match="requires material provenance"):
        QueryPreparationCache().get_or_prepare(old, operand, old.pitch, full_cell_rule=rule)
