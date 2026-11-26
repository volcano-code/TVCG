"""Rule ablations keep material support, conservative moments, and lazy storage."""
import itertools
import json

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from spin_adapter.v21_blocks import BlockMeasure, basis_at
from spin_adapter.v21_material import (
    FULL_CELL_RULES, PARTIAL_CELL_RULES, full_cell_cubature, load_material,
    material_identity, prepare_material, prepare_rle_material,
)


def collect(chunks):
    chunks = list(chunks)
    return np.vstack([p for p, _ in chunks]), np.concatenate([w for _, w in chunks])


def box_mesh(lo, hi):
    vertices = np.array(list(itertools.product(*zip(lo, hi))))
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for face, plane in zip(faces, hull.equations):
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    return vertices, faces


def box_moments(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    extent, center = hi - lo, (lo + hi) / 2
    result = basis_at(center[None])[0]
    result[4:7] += extent**2 / 12
    return extent.prod() * result


@pytest.mark.parametrize("rule,degree,count", [("tetra4", 2, 4), ("gl2", 3, 8), ("gl3", 5, 27)])
def test_full_cell_positive_polynomial_precision(rule, degree, count):
    material = prepare_rle_material(np.array([[0, 0, 0, 1]]), [-0.5] * 3, 1, side=1)
    chunks = list(material.iter_chunks(chunk_size=3, full_cell_rule=rule))
    assert all(0 < len(w) <= 3 for _, w in chunks)
    points, weights = collect(chunks)
    assert len(weights) == count == material.node_count_for_rule(rule)
    assert (weights > 0).all() and (np.abs(points) < 0.5).all()
    for powers in itertools.product(range(degree + 1), repeat=3):
        if sum(powers) > degree:
            continue
        exact = np.prod([0.0 if p % 2 else 1 / ((p + 1) * 2**p) for p in powers])
        measured = weights @ np.prod(points ** np.asarray(powers), axis=1)
        np.testing.assert_allclose(measured, exact, rtol=0, atol=1e-15)
    np.testing.assert_allclose(weights @ basis_at(points), material.moments(), atol=4e-16)


def test_tetra4_default_bytes_and_analytic_runs_are_unchanged():
    origin, pitch, density = np.array([-0.51, 0.17, -0.37]), 0.23, 1.7
    runs = np.array([[0, 1, 0, 3], [1, 0, 0, 2]])
    material = prepare_rle_material(runs, origin, pitch, side=4, density=density)
    identity, moments = material.identity, material.moments()
    block_ids, block_moments = material.block_moments(2)
    points, weights = collect(material.iter_chunks(chunk_size=5))
    signs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(12.0)
    cells = np.array([(i, j, k) for i, j, low, high in runs for k in range(low, high)])
    expected = origin + pitch * (cells[:, None] + (0.5 + signs)[None]).reshape(-1, 3)
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(weights, np.full(len(points), density * pitch**3 * 0.25))
    for rule in FULL_CELL_RULES:
        pp, ww = collect(material.iter_chunks(chunk_size=7, full_cell_rule=rule))
        np.testing.assert_allclose(ww @ basis_at(pp), moments, rtol=2e-15, atol=1e-16)
        assert material.identity == identity and material.node_count == 20
        np.testing.assert_array_equal(material.moments(), moments)
        ids, values = material.block_moments(2)
        np.testing.assert_array_equal(ids, block_ids)
        np.testing.assert_array_equal(values, block_moments)


def test_pack_refines_selected_full_cells_on_demand_and_reopens(tmp_path):
    material = prepare_rle_material(np.array([[0, 0, 0, 4], [3, 3, 2, 4]]),
                                    [-0.5] * 3, 0.25, side=4)
    directory = tmp_path / "pack"
    blocks = BlockMeasure.build(material, directory, block_voxels=2, chunk_size=5)
    assert len(blocks.nodes) == material.node_count == 24
    assert blocks.material.source == "rle" and blocks.material.partial_rule == "none"
    for name in ("full_runs", "partial_cells", "boundary_nodes"):
        a = getattr(blocks.material, name)
        assert isinstance(a, np.memmap) and a.mode == "r" and not a.flags.writeable
    before = {p.name: p.stat().st_size for p in directory.iterdir()}
    encoded = blocks.encoder_grid(2, density=1)
    moments = blocks.moments.copy()
    mask = blocks.ids == 0  # Exactly two full native cells in this spatial block.
    assert mask.sum() == 1
    for rule, per_cell in (("tetra4", 4), ("gl2", 8), ("gl3", 27)):
        chunks = list(blocks.chunks(mask, max_nodes=7, full_cell_rule=rule))
        assert all(0 < len(w) <= 7 for _, w in chunks)
        points, weights = collect(chunks)
        assert len(weights) == 2 * per_cell
        assert ((points >= -0.5) & (points < 0.0)).all()
        np.testing.assert_allclose(weights @ basis_at(points), blocks.moments[mask].sum(0), atol=1e-16)
    np.testing.assert_array_equal(blocks.encoder_grid(2, density=1), encoded)
    np.testing.assert_array_equal(blocks.moments, moments)
    assert {p.name: p.stat().st_size for p in directory.iterdir()} == before
    reopened = BlockMeasure.open(directory, identity=material.identity, block_voxels=2)
    expected = collect(blocks.chunks(mask, full_cell_rule="gl3"))
    actual = collect(reopened.chunks(mask, full_cell_rule="gl3"))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_legacy_pack_keeps_tetra4_and_rejects_refinement_without_provenance(tmp_path):
    material = prepare_rle_material(np.array([[0, 0, 0, 1]]), [0, 0, 0], 1, side=1)
    directory = tmp_path / "legacy"
    blocks = BlockMeasure.build(material, directory, block_voxels=1)
    expected = collect(blocks.chunks(np.ones(len(blocks.ids), dtype=bool)))
    metadata = directory / "metadata.json"
    desc = json.loads(metadata.read_text())
    desc.pop("material_provenance")
    metadata.write_text(json.dumps(desc))
    for name in ("full_runs", "partial_cells", "boundary_nodes"):
        (directory / f"material_{name}.npy").unlink()
    old = BlockMeasure.open(directory, identity=material.identity, block_voxels=1)
    assert old.material is None
    mask = np.ones(len(old.ids), dtype=bool)
    for a, b in zip(collect(old.chunks(mask)), expected):
        np.testing.assert_array_equal(a, b)
    for rule in ("gl2", "gl3"):
        with pytest.raises(ValueError, match="requires material provenance"):
            list(old.chunks(mask, full_cell_rule=rule))
    # Calling build on an existing pack does not silently rewrite legacy assets.
    assert BlockMeasure.build(material, directory, block_voxels=1).material is None
    assert "material_provenance" not in json.loads(metadata.read_text())


def test_partial_sweep_compression_preserves_real_support_and_ten_moments(tmp_path):
    boxes = [(np.array([0.06, 0.12, 0.20]), np.array([0.22, 0.88, 0.80])),
             (np.array([0.72, 0.17, 0.10]), np.array([0.94, 0.86, 0.90]))]
    vertices, faces = [], []
    for low, high in boxes:
        v, f = box_mesh(low, high)
        faces.append(f + sum(map(len, vertices)))
        vertices.append(v)
    vertices, faces = np.vstack(vertices), np.vstack(faces)
    exact = 1.4 * sum(box_moments(lo, hi) for lo, hi in boxes)
    prepared = {}
    for rule in PARTIAL_CELL_RULES:
        material = prepare_material(vertices, faces, [0, 0, 0], 1, side=1,
                                    density=1.4, partial_rule=rule)
        prepared[rule] = material
        assert material.source == "mesh" and material.partial_rule == rule
        assert material.full_cell_count == 0 and material.partial_cell_count == 1
        assert material.identity == material_identity(vertices, faces, [0, 0, 0], 1,
                                                       side=1, density=1.4, partial_rule=rule)
        points, weights = collect(material.iter_chunks())
        depth = sum(((points > lo) & (points < hi)).all(1) for lo, hi in boxes)
        assert (depth == 1).all() and (weights > 0).all()
        np.testing.assert_allclose(weights @ basis_at(points), exact, rtol=2e-13, atol=1e-15)
        for full_rule in FULL_CELL_RULES:
            other = collect(material.iter_chunks(full_cell_rule=full_rule))
            for a, b in zip(other, (points, weights)):
                np.testing.assert_array_equal(a, b)
        saved = tmp_path / f"{rule}.npz"
        material.save(saved)
        restored = load_material(saved, expected_identity=material.identity)
        assert restored.partial_rule == rule and restored.audit["partial_rule"] == rule
        np.testing.assert_array_equal(restored.boundary_nodes, material.boundary_nodes)
    assert len({m.identity for m in prepared.values()}) == 3
    raw, compressed = prepared["sweep_raw"], prepared["sweep_compressed"]
    assert len(raw.boundary_nodes) > len(compressed.boundary_nodes)
    raw_positions = {tuple(p) for p in raw.boundary_nodes[:, :3]}
    assert all(tuple(p) in raw_positions for p in compressed.boundary_nodes[:, :3])
    np.testing.assert_allclose(raw.moments(), compressed.moments(), rtol=2e-13, atol=1e-15)


def test_forced_sweeps_on_tilted_material_preserve_each_native_cell():
    vertices = np.array([[0.041, 0.067, 0.021], [0.929, 0.160, 0.099],
                         [0.107, 0.861, 0.247], [0.231, 0.259, 0.953]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    planes = ConvexHull(vertices).equations
    cell_moments = []
    for rule in ("sweep_compressed", "sweep_raw"):
        material = prepare_material(vertices, faces, [0, 0, 0], 0.25, side=4,
                                    partial_rule=rule)
        points, weights = collect(material.iter_chunks(chunk_size=37))
        assert (points @ planes[:, :3].T + planes[:, 3] <= 2e-12).all()
        values = np.zeros((64, 10))
        cells = np.floor(points / 0.25).astype(int)
        np.add.at(values, np.ravel_multi_index(cells.T, (4, 4, 4)),
                  weights[:, None] * basis_at(points))
        cell_moments.append(values)
        volume = abs(np.linalg.det(vertices[1:] - vertices[0])) / 6
        total = vertices.sum(0)
        second = (np.outer(total, total) + vertices.T @ vertices) / 20
        exact = volume * np.r_[1., total / 4, second.diagonal(),
                               second[1, 2], second[0, 2], second[0, 1]]
        np.testing.assert_allclose(values.sum(0), exact, rtol=3e-12, atol=1e-15)
    np.testing.assert_allclose(*cell_moments, rtol=3e-11, atol=1e-15)


def test_pack_provenance_checksum_rejects_changed_material_runs(tmp_path):
    material = prepare_rle_material(np.array([[0, 0, 0, 1]]), [0, 0, 0], 1, side=2)
    directory = tmp_path / "pack"
    BlockMeasure.build(material, directory, block_voxels=1)
    changed = np.load(directory / "material_full_runs.npy", mmap_mode="r+")
    changed[0, 3] = 2
    changed.flush()
    del changed
    with pytest.raises(ValueError, match="provenance checksum"):
        BlockMeasure.open(directory, identity=material.identity, block_voxels=1)


@pytest.mark.parametrize("rule", ["unknown", "GL2", None, [], 8])
def test_invalid_full_rule_rejected(rule):
    with pytest.raises(ValueError, match="full_cell_rule"):
        full_cell_cubature(rule)
