import itertools

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from spin_adapter.rle_quadrature import native_moment_basis


def box_mesh(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    v = np.array(list(itertools.product(*zip(lo, hi))))
    hull = ConvexHull(v)
    f = hull.simplices.copy()
    for face, plane in zip(f, hull.equations):
        a, b, c = v[face]
        if np.dot(np.cross(b - a, c - a), plane[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    return v, f


def box_integral(lo, hi):
    extent = np.maximum(hi - lo, 0)
    center = (lo + hi) / 2
    values = native_moment_basis(center[None])[0]
    values[4:7] += extent**2 / 12
    return values * extent.prod()


def tetra_integral(vertices):
    vol = abs(np.linalg.det(vertices[1:] - vertices[0])) / 6
    total = vertices.sum(axis=0)
    second = (np.outer(total, total) + vertices.T @ vertices) / 20
    return (
        vol
        * np.r_[
            1.0, total / 4, second.diagonal(), second[1, 2], second[0, 2], second[0, 1]
        ]
    )


def intersect_tetra_cube(equations, lo, hi):
    planes = np.vstack([equations, np.c_[np.eye(3), -hi], np.c_[-np.eye(3), lo]])
    pts = []
    for ii in itertools.combinations(range(len(planes)), 3):
        m = planes[list(ii)]
        if abs(np.linalg.det(m[:, :3])) < 1e-12:
            continue
        x = np.linalg.solve(m[:, :3], -m[:, 3])
        if (planes[:, :3] @ x + planes[:, 3] < 1e-11).all():
            pts.append(x)
    pts = np.unique(np.round(pts, 13), axis=0)
    if len(pts) < 4:
        return np.zeros(10)
    hull = ConvexHull(pts)
    return sum(
        tetra_integral(np.vstack([pts.mean(0), pts[face]])) for face in hull.simplices
    )


def cell_moments(mat, side, h, origin):
    out = np.zeros((side**3, 10))
    for xyz, w in mat.iter_chunks(chunk_size=53):
        assert 0 < len(w) <= 53 and (w > 0).all()
        ind = np.floor((xyz - origin) / h).astype(int)
        np.add.at(
            out,
            np.ravel_multi_index(ind.T, (side,) * 3),
            w[:, None] * native_moment_basis(xyz),
        )
    return out.reshape((side, side, side, 10))


def test_tilted_tetra_all_native_cells_and_support():
    from spin_adapter.v21_material import prepare_material

    v = np.array(
        [
            [0.041, 0.067, 0.021],
            [0.929, 0.16, 0.099],
            [0.107, 0.861, 0.247],
            [0.231, 0.259, 0.953],
        ]
    )
    f = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    domain = ConvexHull(v).equations
    side, h = 5, 0.2
    material = prepare_material(v, f, np.zeros(3), h, side=side, density=1.7)
    got = cell_moments(material, side, h, np.zeros(3))
    for ijk in np.ndindex((side,) * 3):
        lo = np.asarray(ijk) * h
        exact = 1.7 * intersect_tetra_cube(domain, lo, lo + h)
        np.testing.assert_allclose(got[ijk], exact, rtol=2e-10, atol=1e-14)
    for xyz, _ in material.iter_chunks():
        assert (xyz @ domain[:, :3].T + domain[:, 3] <= 2e-12).all()
    np.testing.assert_allclose(
        material.moments(), 1.7 * tetra_integral(v), rtol=1e-11, atol=1e-15
    )


def test_void_subcell_disconnected_island_and_half_open_blocks(tmp_path):
    from spin_adapter.v21_material import prepare_material, load_material

    boxes = [
        (np.array(lo), np.array(hi), sign)
        for lo, hi, sign in [
            ((0.13, 0.21, 0.32), (3.95, 3.89, 3.76), 1),
            ((0.41, 0.7, 0.8), (3.71, 3.41, 3.29), -1),
            ((1.011, 1.012, 1.014), (1.027, 1.032, 1.046), 1),
            ((4.01, 0.12, 0.22), (4.027, 3.67, 3.89), 1),
        ]
    ]
    vs, fs = [], []
    for lo, hi, sign in boxes:
        v, f = box_mesh(lo, hi)
        fs.append((f if sign > 0 else f[:, ::-1]) + sum(map(len, vs)))
        vs.append(v)
    m = prepare_material(np.vstack(vs), np.vstack(fs), [0, 0, 0], 1.0, side=5)
    got = cell_moments(m, 5, 1, np.zeros(3))
    for ijk in np.ndindex((5,) * 3):
        low = np.asarray(ijk)
        exact = sum(
            s * box_integral(np.maximum(lo, low), np.minimum(hi, low + 1))
            for lo, hi, s in boxes
        )
        np.testing.assert_allclose(got[ijk], exact, rtol=4e-11, atol=1e-13)
    for xyz, _ in m.iter_chunks():
        depth = sum(s * ((xyz >= lo) & (xyz <= hi)).all(axis=1) for lo, hi, s in boxes)
        assert (depth == 1).all()
    assert got[4, 2, 2, 0] > 0  # All centers of this slender component lie outside it.
    ids, blocks = m.block_moments(2)
    for ijk, bm in zip(ids, blocks):
        lo, hi = ijk * 2, (ijk + 1) * 2
        expected = got[tuple(slice(a, b) for a, b in zip(lo, hi))].sum(axis=(0, 1, 2))
        np.testing.assert_allclose(bm, expected, rtol=3e-12, atol=1e-13)
        np.testing.assert_allclose(
            m.moments(box=(lo, hi)), expected, rtol=3e-12, atol=1e-13
        )
        for xyz, _ in m.iter_chunks(box=(lo, hi)):
            assert ((xyz >= lo) & (xyz < hi)).all()
    path = tmp_path / "material.npz"
    m.save(path)
    reloaded = load_material(path, expected_identity=m.identity)
    np.testing.assert_array_equal(reloaded.moments(), m.moments())
    with pytest.raises(ValueError, match="identity"):
        load_material(path, expected_identity="other density or frame")


def test_full_cells_four_points_and_rle_comparator():
    from spin_adapter.v21_material import prepare_material, prepare_rle_material

    v, f = box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    m = prepare_material(v, f, [0, 0, 0], 1 / 8, side=8, density=2.0)
    columns = np.array([(i, j, 0, 8) for i, j in np.ndindex((8, 8))])
    other = prepare_rle_material(columns, [0, 0, 0], 1 / 8, side=8, density=2.0)
    assert m.node_count == other.node_count == 4 * 8**3
    assert other.identity != m.identity
    np.testing.assert_allclose(
        m.moments(), 2 * box_integral(np.zeros(3), np.ones(3)), atol=1e-14
    )
    np.testing.assert_allclose(other.moments(), m.moments(), atol=1e-14)


def test_shared_indexed_triangle_seam_at_a_fallback_ray():
    from spin_adapter.v21_material import prepare_material, signed_mesh_moments

    # A tapered prism with opposite face diagonals and nearly vertical sides.
    # At a native integration ray the two evaluations of a shared projected
    # edge rounded down on both sides when each face chose its own anchor.
    v = np.array(
        [
            [0.00059354, -0.21087247, -0.19359931],
            [0.13743191, -0.21087247, -0.1368678],
            [0.10151455, 0.37441692, -0.09732979],
            [0.00421419, 0.37441692, -0.13766931],
            [0.00059355, -0.21087247, 0.1935406],
            [0.13743195, -0.21087247, 0.13680905],
            [0.10151459, 0.37441692, 0.09727104],
            [0.00421419, 0.37441692, 0.1376106],
        ]
    )
    f = [[1, 0, 3], [3, 2, 1], [4, 5, 6], [6, 7, 4]]
    for i in range(4):
        j = (i + 1) % 4
        f.extend([[i, j, j + 4], [i, j + 4, i + 4]])
    prepared = prepare_material(v, np.asarray(f), [-0.52] * 3, 1.04 / 256)
    np.testing.assert_allclose(
        prepared.moments(), signed_mesh_moments(v, f), atol=1e-14
    )


def test_open_mesh_rejected_and_coincident_triangle_soup_supported():
    from spin_adapter.v21_material import prepare_material

    v, f = box_mesh((0.125, 0.125, 0.125), (0.875, 0.875, 0.875))
    with pytest.raises(ValueError, match="closed oriented"):
        prepare_material(v, f[1:], [0, 0, 0], 1 / 8, side=8)
    soup = v[f].reshape(-1, 3)
    indices = np.arange(len(soup)).reshape(-1, 3)
    material = prepare_material(soup, indices, [0, 0, 0], 1 / 8, side=8)
    np.testing.assert_allclose(
        material.moments(),
        box_integral(np.full(3, 0.125), np.full(3, 0.875)),
        atol=1e-15,
    )
