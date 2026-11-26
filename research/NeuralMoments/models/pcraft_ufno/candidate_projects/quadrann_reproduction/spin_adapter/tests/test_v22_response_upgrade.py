"""Physical shape constraints and independent AD/FD checks of the new profile.

These tests concern the surrogate response itself; they do not assert that a
fixed positive quadrature or a smooth profile equals a native hard mesh cut.
"""
import copy
import math

import numpy as np
import pytest
import torch

from spin_adapter import v21_response as response


def constant_model(raw=(0., 0.), *, response_type='monotone_warp'):
    model = response.LocalRetention(hidden_dim=8, response_type=response_type, zero_init=True).double()
    with torch.no_grad():
        model.local[-1].bias.copy_(torch.tensor(raw, dtype=torch.float64))
    return model


def scene():
    vertices = np.array([[-1., -1., -1.], [.2, -1., -1.], [.2, 1., -1.], [-1., 1., -1.],
                         [-1., -1., 1.], [.2, -1., 1.], [.2, 1., 1.], [-1., 1., 1.]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    # Includes a face, a finite edge and a finite corner; all witnesses regular.
    nodes = np.array([[.18, -.3, .31], [.23, .12, -.4], [.25, 1.03, .2], [.22, 1.03, 1.04]])
    measure = response.prepare_measure(nodes, [.1, .2, .3, .4])
    motion = np.random.default_rng(12).normal(size=(8, 3, 3))*.15
    return vertices, faces, nodes, measure, motion


def random_model():
    torch.manual_seed(8367)
    model = response.LocalRetention(4, 9, response_type='monotone_warp').double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-.35, .35)
    return model


@pytest.mark.parametrize('kind', ['bounded_cubic', 'monotone_warp'])
def test_zero_initialization_reproduces_unlearned_cubic_and_keeps_parameter_layout(kind):
    torch.manual_seed(81)
    model = response.LocalRetention(4, 9, response_type=kind, zero_init=True).double()
    s = torch.tensor([-1.2, -1., -.99, -.7, -.1, 0., .23, .8, .999, 1., 1.2], dtype=torch.float64)
    xyz = torch.randn(len(s), 3, dtype=torch.float64)
    context = torch.randn(len(s), 4, dtype=torch.float64)
    for operation, sign in [('intersection', 1.), ('difference', -1.)]:
        value, slope = model.retention_and_slope(xyz, s, context, operation=operation)
        t = (1.+sign*s.clamp(-1., 1.))/2
        expected = 3*t*t-2*t*t*t
        expected_slope = sign*.75*(1.-s.clamp(-1., 1.).square())
        torch.testing.assert_close(value, expected, atol=5e-16, rtol=2e-14)
        torch.testing.assert_close(slope, expected_slope, atol=5e-16, rtol=2e-14)
    expected_layout = {'local.0.weight': (9, 7), 'local.0.bias': (9,),
                       'local.2.weight': (2, 9), 'local.2.bias': (2,)}
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected_layout
    assert model.local[0].weight.abs().sum() > 0
    assert torch.equal(model.local[-1].weight, torch.zeros_like(model.local[-1].weight))
    assert torch.equal(model.local[-1].bias, torch.zeros_like(model.local[-1].bias))


def test_legacy_default_initialization_and_forward_remain_unchanged():
    torch.manual_seed(512)
    old_default = response.LocalRetention(2, 7).double()
    torch.manual_seed(512)
    explicit = response.LocalRetention(2, 7, response_type='bounded_cubic', zero_init=False).double()
    assert old_default.response_type == 'bounded_cubic'
    for key in old_default.state_dict():
        assert torch.equal(old_default.state_dict()[key], explicit.state_dict()[key])
    s = torch.linspace(-1.1, 1.1, 15, dtype=torch.float64)
    points, context = torch.randn(15, 3, dtype=torch.float64), torch.randn(15, 2, dtype=torch.float64)
    bias, steep = old_default.local(torch.cat((points, context), 1)).tanh().unbind(1)
    x = s.clamp(-1., 1.)
    t, u = (1.+x)/2, (1.-x)/2
    p, complement = t*t*(3.-2.*t), u*u*(3.-2.*u)
    expected = p*(1.+complement*.5*(bias+steep*(2.*p-1.)))
    torch.testing.assert_close(old_default(points, s, context), expected, rtol=0., atol=0.)


def test_range_monotonicity_and_same_operand_complement_for_extreme_coefficients():
    # This exercises the turning coordinate branch where a>1/2, not just the
    # near-zero initialization for which monotonicity is trivial.
    s = torch.linspace(-1.2, 1.2, 6001, dtype=torch.float64)
    xyz = torch.zeros(len(s), 3, dtype=torch.float64)
    for sharpness in (-20., -1., 0., .75, 20.):
        for shift in (-20., -.7, 0., .7, 20.):
            model = constant_model((sharpness, shift))
            inside, slope = model.retention_and_slope(xyz, s)
            outside, other_slope = model.retention_and_slope(xyz, s, operation='difference')
            assert torch.all((inside >= 0.) & (inside <= 1.))
            assert torch.all((outside >= 0.) & (outside <= 1.))
            assert (inside.diff() >= -4e-16).all()
            assert (outside.diff() <= 4e-16).all()
            assert (slope >= -1e-15).all()
            torch.testing.assert_close(inside+outside, torch.ones_like(s), rtol=0., atol=4e-16)
            torch.testing.assert_close(slope+other_slope, torch.zeros_like(s), rtol=0., atol=0.)
            assert torch.equal(slope[s.abs() >= 1.], torch.zeros_like(slope[s.abs() >= 1.]))


def test_c1_continuity_at_inner_saturation_and_fixed_outer_band():
    # a=1, delta=0: the positive inner saturation is the golden-ratio root.
    raw = np.arctanh(np.log(2.)/np.log(3.))
    model = constant_model((float(raw), 0.))
    root = (math.sqrt(5.)-1.)/2.
    eps = 1e-7
    for boundary, expected in [(-1., 0.), (-root, 0.), (root, 1.), (1., 1.)]:
        s = torch.tensor([boundary-eps, boundary, boundary+eps], dtype=torch.float64)
        value, slope = model.retention_and_slope(torch.zeros(3, 3, dtype=torch.float64), s)
        assert (value-expected).abs().max() < 1e-12
        assert slope.abs().max() < 3e-6
        assert abs(slope[1].item()) < 2e-14


def test_new_profile_can_fully_retain_old_heads_pointwise_capacity_counterexample():
    # Solve P(s)=.9 by an independent bisection; then enforce the known old
    # upper-bound coefficients. This is a pointwise example, not a claim about
    # total mesh error or native Newton convergence.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        t = (mid+1.)/2
        if 3*t*t-2*t*t*t < .9:
            lo = mid
        else:
            hi = mid
    s = torch.tensor([(lo+hi)/2], dtype=torch.float64)
    points = torch.zeros(1, 3, dtype=torch.float64)
    legacy = constant_model((20., 20.), response_type='bounded_cubic')
    improved = constant_model((20., 0.))
    assert legacy(points, s).item() == pytest.approx(.981, abs=3e-15)
    assert improved(points, s).item() == 1.
    assert improved(points, s, operation='difference').item() == 0.


@pytest.mark.parametrize('operation', ['intersection', 'difference'])
@pytest.mark.parametrize('projected', [False, True])
def test_analytic_distance_slope_and_parameter_gradients_match_autograd(operation, projected):
    model = random_model()
    ad_model = copy.deepcopy(model)
    encoder = torch.nn.Linear(5, 4).double()
    ad_encoder = copy.deepcopy(encoder)
    inputs = torch.randn(5, 5, dtype=torch.float64)
    ids = torch.tensor([0, 2, 1, 4, 0, 3, 4, 2, 3, 1, 1])
    xyz = torch.randn(len(ids), 3, dtype=torch.float64)
    s = torch.tensor([-1.2, -1., -.83, -.34, -.1, 0., .22, .51, .89, 1., 1.2], dtype=torch.float64)
    ad_s = s.clone().requires_grad_()
    table, ad_table = encoder(inputs), ad_encoder(inputs)
    context = model.project_context(table).gather(ids) if projected else table[ids]
    value, slope = model.retention_and_slope(xyz, s, context, operation=operation)
    reference = ad_model(xyz, ad_s, ad_table[ids], operation=operation)
    ad_slope, = torch.autograd.grad(reference.sum(), ad_s, create_graph=True)
    torch.testing.assert_close(value, reference, rtol=2e-13, atol=3e-15)
    torch.testing.assert_close(slope, ad_slope, rtol=3e-12, atol=2e-14)
    params = tuple(model.parameters())+tuple(encoder.parameters())
    ad_params = tuple(ad_model.parameters())+tuple(ad_encoder.parameters())
    gradients = torch.autograd.grad(slope.square().sum(), params)
    ad_gradients = torch.autograd.grad(ad_slope.square().sum(), ad_params)
    for actual, expected in zip(gradients, ad_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-13)
    assert sum(g.abs().sum() for g in gradients[-2:]) > 1e-6


def test_tiny_difference_is_evaluated_directly_instead_of_subtracting_large_inside():
    model = constant_model()
    s = torch.tensor([1.-1e-9], dtype=torch.float64)
    points = torch.zeros(1, 3, dtype=torch.float64)
    inside, _ = model.retention_and_slope(points, s)
    outside, slope = model.retention_and_slope(points, s, operation='difference')
    t = (1.-s.item())/2.
    expected = t*t*(3.-2.*t)
    assert 1.-inside.item() == 0.  # Direct subtraction would erase this mass.
    assert outside.item() == pytest.approx(expected, rel=1e-13, abs=0.)
    assert slope.item() < 0.


def test_shape_chain_matches_rebuilt_surface_fd_and_live_vertex_autograd():
    vertices, faces, nodes, measure, motion = scene()
    model = random_model()
    encoder = torch.nn.Linear(3, 4).double()
    kwargs = dict(model=model, context=encoder, bandwidth=.14, chunk_size=2)

    def evaluate(v, **extra):
        return response.evaluate_response(measure, response.prepare_surface(v, faces), **kwargs, **extra)

    out = evaluate(vertices, vertex_velocity=motion)
    assert out.diagnostics['ambiguous_nodes'] == 0
    assert out.diagnostics['response_type'] == 'monotone_warp'
    for step in (2e-5, 5e-6, 1e-6):
        columns = []
        for direction in motion.transpose(2, 0, 1):
            columns.append((evaluate(vertices+step*direction).moments
                            -evaluate(vertices-step*direction).moments)/(2*step))
        torch.testing.assert_close(out.jacobian, torch.stack(columns, 1), rtol=2e-6, atol=8e-10)
    def moments(q):
        moving = torch.from_numpy(vertices)+torch.einsum('vkd,d->vk', torch.from_numpy(motion), q)
        return evaluate(moving).moments
    direct = torch.autograd.functional.jacobian(moments, torch.zeros(3, dtype=torch.float64, requires_grad=True))
    torch.testing.assert_close(out.jacobian, direct, rtol=2e-12, atol=2e-14)
    # The optimized inference path and independent local AD agree under the
    # complete ten-moment reduction, including finite edge/corner movement.
    model.use_analytic_slope = False
    reference = evaluate(vertices, vertex_velocity=motion)
    model.use_analytic_slope = True
    with torch.inference_mode():
        inference = evaluate(vertices, vertex_velocity=motion)
    torch.testing.assert_close(inference.jacobian, reference.jacobian.detach(), rtol=2e-12, atol=2e-14)
    assert not inference.jacobian.requires_grad


def test_jacobian_only_optimizer_updates_zero_head_then_reaches_static_encoder():
    vertices, faces, nodes, measure, motion = scene()
    torch.manual_seed(8368)
    model = response.LocalRetention(4, 9, response_type='monotone_warp', zero_init=True).double()
    encoder = torch.nn.Linear(3, 4).double()
    optimizer = torch.optim.SGD(list(model.parameters())+list(encoder.parameters()), lr=1e-4)
    surface = response.prepare_surface(vertices, faces)
    initial_head = model.local[-1].weight.detach().clone()
    initial_encoder = encoder.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = response.evaluate_response(measure, surface, bandwidth=.14, model=model,
            context=encoder, vertex_velocity=motion, chunk_size=2)
        result.jacobian.square().sum().backward()
        assert torch.isfinite(model.local[-1].weight.grad).all()
        assert model.local[-1].weight.grad.abs().sum() > 1e-8
        if step == 0:
            # A zero last layer cannot yet send gradient to preceding layers.
            assert encoder.weight.grad.abs().sum() == 0
        else:
            assert encoder.weight.grad.abs().sum() > 1e-10
        optimizer.step()
    assert not torch.equal(model.local[-1].weight, initial_head)
    assert not torch.equal(encoder.weight, initial_encoder)


@pytest.mark.parametrize('kwargs', [{'response_type': 'hard'}, {'response_type': None}, {'zero_init': 1}])
def test_response_options_are_explicit_and_reject_unknown_semantics(kwargs):
    with pytest.raises(ValueError):
        response.LocalRetention(**kwargs)
