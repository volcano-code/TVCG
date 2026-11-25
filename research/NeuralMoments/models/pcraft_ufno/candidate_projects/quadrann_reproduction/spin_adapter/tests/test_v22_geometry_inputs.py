"""Static input equivalence, material mutation, and bounded-cache regressions."""
from dataclasses import replace

import numpy as np
import pytest

from spin_adapter.v21_blocks import BlockMeasure, basis_at
from spin_adapter import v22_geometry_inputs as inputs


def blocks_fixture():
    points = np.array([[.25, .25, .25], [.75, .25, .25]])
    weights = np.array([1e-8, 2e-8])
    nodes = np.column_stack((points, weights))
    indices = np.floor(points * 32).astype(int)
    ids = np.ravel_multi_index(indices.T, (32,)*3)
    return BlockMeasure(nodes, ids, np.array([0, 1, 2]), basis_at(points)*weights[:, None],
                        (indices + .5)/32, np.sqrt(3)/64, np.zeros(3), 1/256,
                        256, 8, "static-test-material")


def fixed_fixture():
    return np.r_[5e-8, np.zeros(9)]


def assert_same_inputs(left, right):
    assert (left.token_grid, left.token_count, left.grids) == (right.token_grid, right.token_count, right.grids)
    for key in ("fixed_moments", "active", "features", "stats", "coords", "log_mass"):
        np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
    for a, b in zip(left.encoder_inputs, right.encoder_inputs):
        np.testing.assert_array_equal(a, b)


def test_analytic_material_inputs_and_original_native_multiscale_channels():
    prepared = inputs.prepare_geometry_inputs(blocks_fixture(), fixed_fixture())
    np.testing.assert_array_equal(prepared.active, [1092, 3140])
    np.testing.assert_allclose(prepared.coords, [[-.5, -.5, -.5], [-.5, -.5, .5]], rtol=1e-15, atol=0.)
    np.testing.assert_allclose(prepared.features[:, :4], [[1., .25, .25, .25], [1., .75, .25, .25]])
    np.testing.assert_array_equal(prepared.features[:, -3:], np.full((2, 3), 1/16))
    np.testing.assert_allclose(prepared.log_mass, np.log([1/3, 2/3]), rtol=1e-15)
    np.testing.assert_allclose(prepared.stats[[0, 10, 20]], [1., 5/8, 3/8], rtol=1e-15)
    np.testing.assert_array_equal(prepared.stats[-6:], [.53125, .28125, .28125, .5625, .0625, .0625])
    assert prepared.grids == (32, 64)
    assert prepared.token_count == 4096
    for grid, channels in zip(prepared.grids, prepared.encoder_inputs):
        assert channels.shape == (4, grid, grid, grid)
        assert channels.dtype == np.float32
        cell = (grid//4, grid//4, grid//4)
        assert channels[(0, *cell)] == np.float32(1e-8*grid**3)
        assert channels[(1, *cell)] == 1
        assert channels[(3, *cell)] == 1
    arrays = (prepared.fixed_moments, prepared.active, prepared.features, prepared.stats,
              prepared.coords, prepared.log_mass, *prepared.encoder_inputs)
    assert prepared.nbytes == sum(a.nbytes for a in arrays)
    assert all(not a.flags.writeable for a in arrays)
    with pytest.raises(ValueError):
        prepared.features[0, 0] = 7


def test_cached_uncached_and_token8_inputs_are_identical(monkeypatch):
    blocks = blocks_fixture()
    calls = []
    original = blocks.encoder_grid

    def counted(grid, density):
        calls.append((grid, density))
        return original(grid, density)

    monkeypatch.setattr(blocks, "encoder_grid", counted)
    cached = inputs.GeometryInputCache()
    uncached = inputs.GeometryInputCache(max_bytes=0)
    kwargs = dict(token_grid=8, grids=(4, 8))
    first = cached.get(blocks, fixed_fixture(), **kwargs)
    assert cached.get(blocks, fixed_fixture(), **kwargs) is first
    assert calls == [(4, 1.), (8, 1.)]
    assert_same_inputs(first, uncached.get(blocks, fixed_fixture(), **kwargs))
    assert_same_inputs(first, inputs.prepare_geometry_inputs(blocks, fixed_fixture(), **kwargs))
    assert cached.snapshot()["hits"] == 1
    assert cached.snapshot()["builds"] == 1
    assert uncached.snapshot()["entries"] == 0
    assert uncached.snapshot()["bypasses"] == 1


@pytest.mark.parametrize("change", ["fixed", "density", "token_grid", "grids", "nodes", "moments", "origin", "centers"])
def test_changed_scientific_inputs_cannot_hit_old_cache(change):
    blocks = blocks_fixture()
    cache = inputs.GeometryInputCache()
    fixed = fixed_fixture()
    kwargs = dict(grids=(4, 8))
    first = cache.get(blocks, fixed, **kwargs)
    if change == "fixed":
        fixed[0] *= 2
    elif change == "density":
        kwargs["density"] = 2.
    elif change == "token_grid":
        kwargs["token_grid"] = 8
    elif change == "grids":
        kwargs["grids"] = (8, 4)
    elif change == "nodes":
        blocks.nodes[0, 3] *= .5
    elif change == "moments":
        blocks.moments[0] *= .5
    elif change == "origin":
        blocks.origin[0] -= .01
    elif change == "centers":
        blocks.centers[0, 0] += .07
    second = cache.get(blocks, fixed, **kwargs)
    assert second is not first
    assert cache.snapshot()["misses"] == 2
    assert_same_inputs(second, inputs.prepare_geometry_inputs(blocks, fixed, **kwargs))


def test_material_signature_covers_all_block_metadata_and_layout():
    original = blocks_fixture()
    initial = inputs.material_signature(original)
    for name in ("nodes", "ids", "offsets", "moments", "centers", "origin"):
        data = getattr(original, name).copy()
        data.flat[0] += 1
        modified = replace(original, **{name: data})
        assert inputs.material_signature(modified) != initial, name
    for name, value in (("side", 128), ("block_voxels", 16), ("pitch", 1/128),
                        ("radius", .1), ("identity", "another-material")):
        assert inputs.material_signature(replace(original, **{name: value})) != initial, name
    # A replacement with equal content is safe to share; changed content is not.
    assert inputs.material_signature(replace(original, nodes=original.nodes.copy())) == initial
    assert inputs.material_signature(replace(original, moments=original.moments.astype(np.float32))) != initial


def test_readonly_array_view_of_mutable_memory_is_content_checked():
    blocks = blocks_fixture()
    base = blocks.nodes.copy()
    blocks.nodes = base.view()
    blocks.nodes.flags.writeable = False
    cache = inputs.GeometryInputCache()
    first = cache.get(blocks, fixed_fixture(), grids=(4, 8))
    base[0, 3] *= .5
    second = cache.get(blocks, fixed_fixture(), grids=(4, 8))
    assert second is not first
    assert second.encoder_inputs[0][0].sum() < first.encoder_inputs[0][0].sum()


def test_readonly_memmap_skips_large_node_hash_and_file_writes_invalidate(tmp_path, monkeypatch):
    blocks = blocks_fixture()
    path = tmp_path/"nodes.npy"
    np.save(path, blocks.nodes)
    blocks.nodes = np.load(path, mmap_mode="r")
    original_digest = inputs._array_digest
    node_hashes = []

    def observed(value):
        if value is blocks.nodes:
            node_hashes.append(True)
        return original_digest(value)

    monkeypatch.setattr(inputs, "_array_digest", observed)
    cache = inputs.GeometryInputCache()
    first = cache.get(blocks, fixed_fixture(), grids=(4, 8))
    assert cache.get(blocks, fixed_fixture(), grids=(4, 8)) is first
    assert node_hashes == []
    writer = np.load(path, mmap_mode="r+")
    writer[0, 3] *= .5
    writer.flush()
    # File stamp invalidation is exercised without relying on scheduler delays.
    import os
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    second = cache.get(blocks, fixed_fixture(), grids=(4, 8))
    assert second is not first
    assert node_hashes == []
    assert second.encoder_inputs[0][0].sum() < first.encoder_inputs[0][0].sum()


def test_atomic_file_replacement_never_confuses_old_and_new_mappings(tmp_path):
    blocks = blocks_fixture()
    path = tmp_path/"nodes.npy"
    np.save(path, blocks.nodes)
    old_mapping = np.load(path, mmap_mode="r")
    replacement = blocks.nodes.copy()
    replacement[0, 3] *= .5
    other = tmp_path/"replacement.npy"
    np.save(other, replacement)
    other.replace(path)
    new_mapping = np.load(path, mmap_mode="r")
    old = replace(blocks, nodes=old_mapping)
    new = replace(blocks, nodes=new_mapping)
    assert inputs.material_signature(old) != inputs.material_signature(new)
    cache = inputs.GeometryInputCache()
    a = cache.get(old, fixed_fixture(), grids=(4, 8))
    b = cache.get(new, fixed_fixture(), grids=(4, 8))
    assert b is not a
    assert b.encoder_inputs[0][0].sum() < a.encoder_inputs[0][0].sum()


def test_readonly_memmap_slices_bind_offset_and_array_layout(tmp_path):
    blocks = blocks_fixture()
    rows = np.r_[blocks.nodes, blocks.nodes.copy()]
    rows[2:, 3] *= .5
    path = tmp_path/"nodes.npy"
    np.save(path, rows)
    mapping = np.load(path, mmap_mode="r")
    a = replace(blocks, nodes=mapping[:2])
    b = replace(blocks, nodes=mapping[2:])
    c = replace(blocks, nodes=np.asarray(mapping[:2]))
    assert inputs.material_signature(a) != inputs.material_signature(b)
    assert inputs.material_signature(a) == inputs.material_signature(c)
    assert inputs.material_signature(a) != inputs.material_signature(replace(blocks, nodes=mapping[:2, ::-1]))


def test_readonly_view_of_writable_memmap_uses_content_hash(tmp_path, monkeypatch):
    blocks = blocks_fixture()
    path = tmp_path/"nodes.npy"
    np.save(path, blocks.nodes)
    writer = np.load(path, mmap_mode="r+")
    view = writer.view()
    view.flags.writeable = False
    blocks.nodes = view
    assert inputs._readonly_mapping_signature(view) is None
    before = inputs.material_signature(blocks)
    writer[0, 3] *= .5
    assert inputs.material_signature(blocks) != before


def test_file_name_removal_falls_back_to_actual_live_mapping_content(tmp_path):
    blocks = blocks_fixture()
    path = tmp_path/"nodes.npy"
    np.save(path, blocks.nodes)
    blocks.nodes = np.load(path, mmap_mode="r")
    path.unlink()
    assert inputs._readonly_mapping_signature(blocks.nodes) is None
    signature = inputs.material_signature(blocks)
    assert signature == inputs.material_signature(blocks)


def test_lru_budget_and_entry_limit_are_enforced():
    blocks = blocks_fixture()
    fixed = fixed_fixture()
    size = inputs.prepare_geometry_inputs(blocks, fixed, grids=(4, 8)).nbytes
    cache = inputs.GeometryInputCache(max_bytes=2*size)
    a = cache.get(blocks, fixed, grids=(4, 8))
    fixed_b = fixed*2
    b = cache.get(blocks, fixed_b, grids=(4, 8))
    assert cache.get(blocks, fixed, grids=(4, 8)) is a
    cache.get(blocks, fixed*3, grids=(4, 8))
    assert cache.snapshot()["entries"] == 2
    assert cache.snapshot()["bytes"] == 2*size
    assert cache.snapshot()["evictions"] == 1
    assert cache.get(blocks, fixed_b, grids=(4, 8)) is not b
    assert cache.snapshot()["bytes"] <= cache.max_bytes
    limited = inputs.GeometryInputCache(max_bytes=100*size, max_entries=1)
    limited.get(blocks, fixed, grids=(4, 8))
    limited.get(blocks, fixed_b, grids=(4, 8))
    assert limited.snapshot()["entries"] == 1
    assert limited.snapshot()["evictions"] == 1


def test_oversize_and_disabled_caches_do_not_retain_payload():
    blocks = blocks_fixture()
    fixed = fixed_fixture()
    reference = inputs.prepare_geometry_inputs(blocks, fixed, grids=(4, 8))
    for limit in (0, reference.nbytes - 1):
        cache = inputs.GeometryInputCache(max_bytes=limit)
        first = cache.get(blocks, fixed, grids=(4, 8))
        second = cache.get(blocks, fixed, grids=(4, 8))
        assert first is not second
        assert_same_inputs(first, second)
        assert cache.snapshot()["entries"] == 0
        assert cache.snapshot()["bytes"] == 0
        assert cache.snapshot()["bypasses"] == 2


def test_source_mutation_during_preparation_is_rejected(monkeypatch):
    blocks = blocks_fixture()
    original = blocks.encoder_grid

    def mutate(grid, density):
        blocks.moments[0, 0] *= .5
        return original(grid, density)

    monkeypatch.setattr(blocks, "encoder_grid", mutate)
    cache = inputs.GeometryInputCache()
    with pytest.raises(RuntimeError, match="material changed"):
        cache.get(blocks, fixed_fixture(), grids=(4, 8))
    assert cache.snapshot()["source_changes"] == 1
    assert cache.snapshot()["entries"] == 0


def test_clear_releases_cached_arrays_and_optionally_counters():
    cache = inputs.GeometryInputCache()
    cache.get(blocks_fixture(), fixed_fixture(), grids=(4, 8))
    cache.clear()
    assert cache.snapshot()["bytes"] == 0
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["builds"] == 1
    cache.clear(reset_stats=True)
    assert cache.snapshot()["builds"] == 0


@pytest.mark.parametrize("kwargs", [{"max_bytes": -1}, {"max_bytes": True},
                                    {"max_bytes": 1.5}, {"max_entries": 0}])
def test_invalid_cache_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        inputs.GeometryInputCache(**kwargs)


@pytest.mark.parametrize("kwargs", [{"density": 0}, {"density": np.inf}, {"token_grid": 7},
                                    {"grids": ()}, {"grids": (3,)}, {"grids": (True,)}])
def test_invalid_static_inputs_rejected_before_cache_lookup(kwargs):
    with pytest.raises(ValueError):
        inputs.GeometryInputCache().get(blocks_fixture(), fixed_fixture(), **kwargs)
