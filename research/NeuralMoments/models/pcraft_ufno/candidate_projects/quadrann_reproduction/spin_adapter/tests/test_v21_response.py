"""The point rule, analytic slab, and moving vertices are independent oracles."""
import numpy as np
import pytest
import torch

from spin_adapter import v21_response as response


def box(lo, hi):
    v = np.asarray([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                    [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                    [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                    [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, faces


def extruded_l():
    polygon = [[0., 0.], [2., 0.], [2., 1.], [1., 1.], [1., 2.], [0., 2.]]
    vertices = np.array([[x, y, z] for z in [0., 1.] for x, y in polygon])
    cap = [(0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)]
    sides = []
    for a in range(6):
        b = (a + 1) % 6
        sides.extend([(a, b, b + 6), (a, b + 6, a + 6)])
    return vertices, np.array([(c, b, a) for a, b, c in cap]
                              + [(a + 6, b + 6, c + 6) for a, b, c in cap] + sides)


def test_closed_operand_sign_handles_concavity_edges_and_caps():
    surface = response.prepare_surface(*extruded_l())
    points = np.array([[.8, 1.5, .5], [1.45, .8, .5], [1.2, 1.3, .5],
                       [2.3, 1.4, .5], [.42, .27, 1.125], [.42, .27, 1.]])
    np.testing.assert_allclose(surface.signed_distance(points), [.2, .2, -.2, -.5, -.125, 0.], atol=1e-13)


def test_manufactured_plane_returns_all_smoothed_moments_and_their_cap_motion():
    # Three-point Gauss is exact on each cell for cubic retention times a quadratic.
    local_x, local_w = np.polynomial.legendre.leggauss(3)
    coords = ((np.arange(8)[:, None] + (local_x + 1) / 2) / 8).ravel()
    weights = np.broadcast_to(local_w / 16, (8, 3)).ravel()
    transverse = (local_x + 1) / 2
    x, y, z = np.meshgrid(coords, transverse, transverse, indexing='ij')
    wx, wy, wz = np.meshgrid(weights, local_w / 2, local_w / 2, indexing='ij')
    measure = response.prepare_measure(np.stack([x, y, z], -1).reshape(-1, 3), (wx * wy * wz).ravel())
    cut, width = .5, .125
    vertices, faces = box([-2, -2, -2], [cut, 2, 2])
    surface = response.prepare_surface(vertices, faces)
    velocity = np.zeros((8, 3, 2))
    velocity[vertices[:, 0] == cut, 0, :] = [1., -2.]
    fixed = np.arange(10, dtype=float) / 37
    args = dict(bandwidth=width, vertex_velocity=velocity, fixed_moments=fixed, chunk_size=31)
    inside = response.evaluate_response(measure, surface, **args)
    outside = response.evaluate_response(measure, surface, operation='difference', **args)
    variance = width**2 / 5
    mx = (cut**2 + variance) / 2
    expected = np.array([cut, mx, cut / 2, cut / 2, cut**3 / 3 + cut * variance,
                         cut / 3, cut / 3, cut / 4, mx / 2, mx / 2])
    deriv = np.array([1., cut, .5, .5, cut**2 + variance, 1/3, 1/3, .25, cut/2, cut/2])
    np.testing.assert_allclose(inside.moments, fixed + expected, atol=2e-14, rtol=1e-13)
    np.testing.assert_allclose(inside.jacobian, deriv[:, None] * [[1., -2.]], atol=2e-14)
    np.testing.assert_allclose(inside.moments + outside.moments, 2 * fixed + measure.total_moments, atol=2e-14)
    np.testing.assert_allclose(inside.jacobian + outside.jacobian, 0., atol=1e-15)
    assert inside.diagnostics['bandwidth'] == width
    assert inside.diagnostics['active_nodes'] < len(x.ravel())


def test_zero_or_full_operand_keeps_fixed_only_once_and_zero_derivative():
    nodes = np.array([[1., 2., 3.], [-2., 3., -1.], [1., 0., -1.]])
    weights = np.array([.5, 2., 1.5])
    measure = response.prepare_measure(nodes, weights)
    np.testing.assert_allclose(measure.total_moments, [4., -2., 7., -2., 10., 20., 8., -3., 4., -11.])
    fixed = np.arange(10, dtype=float) / 2
    velocity = np.ones((8, 3, 4))
    for limits, expected in [(([-5]*3, [5]*3), measure.total_moments), (([7]*3, [8]*3), 0.)]:
        out = response.evaluate_response(measure, response.prepare_surface(*box(*limits)), bandwidth=.1,
                                         fixed_moments=fixed, vertex_velocity=velocity)
        np.testing.assert_allclose(out.moments, fixed + expected)
        np.testing.assert_array_equal(out.jacobian, np.zeros((10, 4)))


def test_shape_chain_matches_rebuilding_moving_mesh_and_torch_autograd_with_encoder():
    vertices, faces = box([-1., -1., -1.], [.2, 1., 1.])
    # The third point sees a finite edge, the last point sees a finite corner.
    nodes = np.array([[.18, -.3, .31], [.23, .12, -.4], [.25, 1.03, .2], [.22, 1.03, 1.04]])
    weights = np.array([.1, .2, .3, .4])
    measure = response.prepare_measure(nodes, weights)
    rng = np.random.default_rng(12)
    velocity = rng.normal(size=(8, 3, 3)) * .15
    torch.manual_seed(43)
    encoder = torch.nn.Linear(3, 4).double()
    model = response.LocalRetention(context_dim=4, hidden_dim=8).double()
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-.3, .3)
    kwargs = dict(bandwidth=.14, model=model, context=encoder, chunk_size=2)

    def solve(v, **extra):
        return response.evaluate_response(measure, response.prepare_surface(v, faces), **kwargs, **extra)

    out = solve(vertices, vertex_velocity=velocity)
    assert out.diagnostics['ambiguous_nodes'] == 0
    fd = []
    for col in velocity.transpose(2, 0, 1):
        plus = solve(vertices + 1e-5 * col).moments
        minus = solve(vertices - 1e-5 * col).moments
        fd.append((plus - minus) / 2e-5)
    torch.testing.assert_close(out.jacobian, torch.stack(fd, 1), rtol=5e-7, atol=2e-10)
    q = torch.zeros(3, dtype=torch.double, requires_grad=True)
    def at_q(offset):
        v = torch.from_numpy(vertices) + torch.einsum('vkd,d->vk', torch.from_numpy(velocity), offset)
        return solve(v).moments
    direct = torch.autograd.functional.jacobian(at_q, q)
    torch.testing.assert_close(out.jacobian, direct, rtol=1e-12, atol=2e-14)
    outside = solve(vertices, operation='difference', vertex_velocity=velocity)
    torch.testing.assert_close(outside.jacobian, -out.jacobian, rtol=1e-12, atol=2e-14)
    out.jacobian.square().sum().backward()
    assert encoder.weight.grad is not None and encoder.weight.grad.norm().item() > 1e-9
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.norm().item() for p in model.parameters()) > 1e-8


@pytest.mark.parametrize('point', [[0, 0, 0], [1, 1, .3]])
def test_ambiguous_active_medial_axis_or_sharp_boundary_withholds_jacobian(point):
    measure = response.prepare_measure([point], [1.])
    surface = response.prepare_surface(*box([-1]*3, [1]*3))
    out = response.evaluate_response(measure, surface, bandwidth=1.5, vertex_velocity=np.ones((8, 3, 2)))
    assert torch.isfinite(out.moments).all()
    assert out.jacobian is None and out.diagnostics['ambiguous_nodes'] == 1


def test_rejects_open_degenerate_reversed_surfaces_and_signed_weights():
    vertices, faces = box([-1]*3, [1]*3)
    for bad in (faces[:-1], np.r_[faces, faces[0:1]], faces[:, ::-1], np.r_[faces[:-1], [[1, 1, 2]]]):
        with pytest.raises(ValueError):
            response.prepare_surface(vertices, bad)
    for weight in ([0.], [-.1], [np.nan]):
        with pytest.raises(ValueError):
            response.prepare_measure([[0., 0., 0.]], weight)


def test_legacy_then_canonical_package_imports_can_query_an_operand():
    # Existing callers import this package two ways. An accelerator disk cache
    # populated under the legacy name must not poison the canonical entry.
    import os
    from pathlib import Path
    import subprocess
    import sys
    response.prepare_surface(*box([-1.]*3, [1.]*3)).signed_distance([[.2, .3, .4]])
    root = Path(__file__).resolve().parents[8]
    script = '''import numpy as np
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface
v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
assert prepare_surface(v,f).signed_distance([[.2,.2,.2]])[0]>0
'''
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', script], cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_tiny_positive_difference_is_not_lost_against_large_inside_mass():
    # The second atom lies one millionth of a transition width from saturation.
    # Its retained complement remains representable even beside a 1e16-weight
    # fully retained interior atom in the opposite operation.
    measure = response.prepare_measure([[0., .2, .3], [.75 + 2.5e-7, .2, .3]], [1e16, 1.])
    surface = response.prepare_surface(*box([-2.]*3, [1., 2., 2.]))
    st = response.prepare_stencil(measure, surface, bandwidth=.25)
    out = response.evaluate_stencil(st, operation='difference')
    t = (.75 + 2.5e-7 - .75) / .5
    expected = t*t*(3 - 2*t)
    assert out.moments[0].item() == pytest.approx(expected, rel=1e-8, abs=0.)
    model = response.LocalRetention().double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    assert response.evaluate_stencil(st, operation='difference', model=model).moments[0].item() == pytest.approx(expected, rel=1e-8, abs=0.)


def test_inference_mode_still_provides_the_derivative_of_the_learned_response():
    surface = response.prepare_surface(*box([-2.]*3, [1., 2., 2.]))
    measure = response.prepare_measure([[1.03, .2, .3]], [1.])
    model = response.LocalRetention().double()
    v = np.ones((8, 3, 1))
    kwargs = dict(bandwidth=.125, vertex_velocity=v, model=model)
    reference = response.evaluate_response(measure, surface, **kwargs)
    with torch.inference_mode():
        result = response.evaluate_response(measure, surface, **kwargs)
    torch.testing.assert_close(result.jacobian, reference.jacobian)
    assert not result.jacobian.requires_grad


def test_nonregular_tensor_operand_cannot_backpropagate_a_selected_one_sided_witness():
    vertices, faces = box([-1.]*3, [1.]*3)
    live = torch.from_numpy(vertices).requires_grad_()
    measure = response.prepare_measure([[1., 1., .3]], [1.])
    model = response.LocalRetention().double()
    out = response.evaluate_response(measure, response.prepare_surface(live, faces), bandwidth=.2,
                                     model=model, vertex_velocity=np.ones((8,3,1)))
    assert out.jacobian is None and out.diagnostics['ambiguous_nodes'] == 1
    assert torch.autograd.grad(out.moments.sum(), live, allow_unused=True)[0] is None


def test_all_saturated_training_loss_has_a_zero_parameter_gradient():
    surface = response.prepare_surface(*box([-2.]*3, [1., 2., 2.]))
    measure = response.prepare_measure([[0., .2, .3]], [1.])
    model = response.LocalRetention().double()
    out = response.evaluate_response(measure, surface, bandwidth=.125, model=model, vertex_velocity=np.ones((8,3,1)))
    (out.moments.square().sum() + out.jacobian.sum()).backward()
    assert all(p.grad is None or (p.grad == 0).all() for p in model.parameters())


def test_almost_tied_neighbor_at_shared_edge_has_its_unique_face_derivative():
    # The two distance magnitudes agree to 14 decimal places, but the closest
    # top-face point and the second-place edge point differ by 2e-8 in x.
    v, f = box([-1.]*3, [1.]*3)
    surf = response.prepare_surface(v, f)
    m = response.prepare_measure([[1.-2e-8, .3, 1.03]], [1.])
    velocity = np.broadcast_to(np.eye(3), (8, 3, 3))
    result = response.evaluate_response(m, surf, bandwidth=.1, vertex_velocity=velocity)
    assert result.diagnostics['ambiguous_nodes'] == 0
    np.testing.assert_allclose(result.jacobian[0], [0., 0., 6.825], atol=2e-12)


def test_stencil_rejects_reuse_after_in_place_tensor_surface_motion():
    v, f = box([-1.]*3, [1.]*3)
    live = torch.from_numpy(v).requires_grad_()
    st = response.prepare_stencil(response.prepare_measure([[1.02, .1, .3]], [1.]),
                                  response.prepare_surface(live, f), bandwidth=.1)
    with torch.no_grad():
        live[:, 0] += .03
    with pytest.raises(ValueError, match='stale|changed'):
        response.evaluate_stencil(st)


def test_resolvable_sub_tolerance_gap_between_components_does_not_fabricate_a_jacobian():
    gap = 1e-14
    left, faces = box([-1., -1., -1.], [-gap, 1., 1.])
    right, other = box([gap, -1., -1.], [1., 1., 1.])
    surface = response.prepare_surface(np.vstack((left, right)), np.vstack((faces, other+8)))
    measure = response.prepare_measure([[0., .13, .21]], [1.])
    result = response.evaluate_response(measure, surface, bandwidth=.01, vertex_velocity=np.ones((16,3,1)))
    assert result.jacobian is None and result.diagnostics['ambiguous_nodes'] == 1
