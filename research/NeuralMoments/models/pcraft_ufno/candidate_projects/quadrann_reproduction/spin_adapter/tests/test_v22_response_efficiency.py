"""Independent AD and legacy-layout checks for the two response fast paths."""
import copy
import warnings

import numpy as np
import pytest
import torch

from spin_adapter import v21_response as response


def legacy_forward(model, points, distance, context, operation):
    """Frozen V2.1 formula, independent of the new value/slope helper."""
    features = points if context is None else torch.cat((points, context.to(points)), 1)
    bias, steep = model.local(features).tanh().to(distance).unbind(1)
    s = distance.clamp(-1., 1.)
    t, u = (s+1.)/2., (1.-s)/2.
    p, q = t*t*(3.-2.*t), u*u*(3.-2.*u)
    coefficient = .5*(bias+steep*(2.*p-1.))
    return p*(1.+q*coefficient) if operation == 'intersection' else q*(1.-p*coefficient)


def randomized_model(context_dim=7, hidden_dim=13, dtype=torch.float64):
    torch.manual_seed(8351)
    model = response.LocalRetention(context_dim, hidden_dim).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-.65, .65)
    return model


@pytest.mark.parametrize('operation', ['intersection', 'difference'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_closed_form_matches_legacy_autograd_near_clamp_and_saturation(operation, dtype):
    model = randomized_model(dtype=dtype)
    # Includes both clamped sides, endpoints, near endpoints, and ordinary band.
    eps = 2e-6 if dtype == torch.float32 else 1e-12
    s = torch.tensor([-1.2, -1., -1.+eps, -.91, -.25, 0., .37, .93, 1.-eps, 1., 1.2],
                     dtype=dtype, requires_grad=True)
    xyz = torch.randn(len(s), 3, dtype=dtype)
    context = torch.randn(len(s), 7, dtype=dtype)
    old = legacy_forward(model, xyz, s, context, operation)
    old_slope, = torch.autograd.grad(old.sum(), s, create_graph=True)
    new, slope = model.retention_and_slope(xyz, s, context, operation=operation)
    tolerance = dict(atol=3e-7, rtol=3e-6) if dtype == torch.float32 else dict(atol=8e-15, rtol=2e-12)
    torch.testing.assert_close(new, old, **tolerance)
    torch.testing.assert_close(slope, old_slope, **tolerance)
    assert torch.all((new >= 0.) & (new <= 1.))
    assert torch.equal(slope[[0, 1, -2, -1]], torch.zeros(4, dtype=dtype))
    other, other_slope = model.retention_and_slope(
        xyz, s, context, operation='difference' if operation == 'intersection' else 'intersection')
    torch.testing.assert_close(new+other, torch.ones_like(new), **tolerance)
    torch.testing.assert_close(slope+other_slope, torch.zeros_like(slope), **tolerance)


@pytest.mark.parametrize('operation', ['intersection', 'difference'])
@pytest.mark.parametrize('projected', [False, True])
def test_jacobian_and_joint_loss_gradients_match_independent_ad(operation, projected):
    model = randomized_model()
    legacy = copy.deepcopy(model)
    encoder = torch.nn.Linear(4, 7).double()
    legacy_encoder = copy.deepcopy(encoder)
    torch.manual_seed(172)
    inputs = torch.randn(5, 4, dtype=torch.float64)
    tokens = torch.tensor([0, 1, 1, 4, 2, 2, 2, 0, 3, 4, 1, 0])
    xyz = torch.randn(len(tokens), 3, dtype=torch.float64)
    distance = torch.linspace(-.93, .91, len(tokens), dtype=torch.float64)
    old_distance = distance.clone().requires_grad_()
    table, old_table = encoder(inputs), legacy_encoder(inputs)
    context = model.project_context(table).gather(tokens) if projected else table[tokens]
    value, slope = model.retention_and_slope(xyz, distance, context, operation=operation)
    old = legacy_forward(legacy, xyz, old_distance, old_table[tokens], operation)
    old_slope, = torch.autograd.grad(old.sum(), old_distance, create_graph=True)
    basis = torch.from_numpy(response._weighted_basis(xyz.numpy(), np.linspace(.03, .2, len(tokens))))
    motion = torch.randn(len(tokens), 8, dtype=torch.float64)
    moments, jac = basis.T@value, basis.T@(slope[:, None]*motion)
    old_moments, old_jac = basis.T@old, basis.T@(old_slope[:, None]*motion)
    torch.testing.assert_close(moments, old_moments, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(jac, old_jac, rtol=3e-12, atol=2e-14)
    for only_j in (True, False):
        loss = jac.square().sum() + (0. if only_j else moments.square().sum())
        old_loss = old_jac.square().sum() + (0. if only_j else old_moments.square().sum())
        params = tuple(model.parameters())+tuple(encoder.parameters())
        old_params = tuple(legacy.parameters())+tuple(legacy_encoder.parameters())
        gradients = torch.autograd.grad(loss, params, retain_graph=True)
        old_gradients = torch.autograd.grad(old_loss, old_params, retain_graph=True)
        for actual, expected in zip(gradients, old_gradients):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-13)
        assert sum(g.abs().sum() for g in gradients[-2:]) > 1e-6


def test_projection_preserves_v21_parameter_layout_and_state_dict_loading():
    model = randomized_model()
    expected = {'local.0.weight': (13, 10), 'local.0.bias': (13,),
                'local.2.weight': (2, 13), 'local.2.bias': (2,)}
    assert {name:tuple(value.shape) for name, value in model.state_dict().items()} == expected
    assert model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True).missing_keys == []
    points = torch.randn(8, 3, dtype=torch.float64)
    s = torch.linspace(-.9, .9, 8, dtype=torch.float64)
    context = torch.randn(3, 7, dtype=torch.float64)
    indices = torch.tensor([0, 1, 2, 1, 0, 2, 2, 0])
    projection = model.project_context(context)
    assert projection.values.shape == (3, 13)
    torch.testing.assert_close(model(points, s, projection.gather(indices)),
                               legacy_forward(model, points, s, context[indices], 'intersection'),
                               rtol=1e-13, atol=2e-15)


@pytest.mark.parametrize('mutation', ['update', 'load', 'replace', 'convert', 'freeze'])
def test_projection_rejects_changed_parameter_version_or_owner(mutation):
    model = randomized_model()
    context = torch.randn(2, 7, dtype=torch.float64)
    projected = model.project_context(context)
    points, s = torch.randn(2, 3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    foreign = copy.deepcopy(model)
    with pytest.raises(ValueError, match='foreign or stale'):
        foreign(points, s, projected)
    if mutation == 'update':
        with torch.no_grad():
            model.local[0].weight.add_(.01)
    elif mutation == 'load':
        model.load_state_dict(copy.deepcopy(model.state_dict()))
    elif mutation == 'replace':
        model.local[0].weight = torch.nn.Parameter(model.local[0].weight.detach().clone())
    elif mutation == 'convert':
        model.float()
        points, s = points.float(), s.float()
    else:
        model.local[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match='foreign or stale'):
        model(points, s, projected)


def scene():
    vertices = np.array([[-1., -1., -1.], [.2, -1., -1.], [.2, 1., -1.], [-1., 1., -1.],
                         [-1., -1., 1.], [.2, -1., 1.], [.2, 1., 1.], [-1., 1., 1.]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    nodes = np.array([[.18, -.3, .31], [.23, .12, -.4], [.25, 1.03, .2], [.22, 1.03, 1.04]])
    measure = response.prepare_measure(nodes, [.1, .2, .3, .4])
    surface = response.prepare_surface(vertices, faces)
    stencil = response.prepare_stencil(measure, surface, bandwidth=.14)
    return stencil, nodes, np.random.default_rng(12).normal(size=(8, 3, 10))*.15


@pytest.mark.parametrize('guard', [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize('operation', ['intersection', 'difference'])
def test_preprojected_inference_keeps_j_without_temporary_autograd_graph(guard, operation, monkeypatch):
    stencil, nodes, motion = scene()
    model = randomized_model().eval()
    context = torch.randn(len(nodes), 7, dtype=torch.float64)
    model.use_analytic_slope = False
    reference = response.evaluate_stencil(stencil, model=model, context=context,
        vertex_velocity=motion, operation=operation, chunk_size=2)
    model.use_analytic_slope = True
    with guard():
        projection = model.project_context(context)
        def forbidden_ad(*args, **kwargs):
            raise AssertionError('closed form must not call autograd.grad internally')
        monkeypatch.setattr(torch.autograd, 'grad', forbidden_ad)
        actual = response.evaluate_stencil(stencil, model=model, context=projection,
            vertex_velocity=motion, operation=operation, chunk_size=2)
        zero = response.evaluate_stencil(stencil, model=model, context=projection,
            vertex_velocity=np.zeros_like(motion), operation=operation, chunk_size=2)
    torch.testing.assert_close(actual.moments, reference.moments.detach(), rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(actual.jacobian, reference.jacobian.detach(), rtol=3e-12, atol=2e-14)
    assert not actual.moments.requires_grad and not actual.jacobian.requires_grad
    assert torch.equal(zero.jacobian, torch.zeros_like(zero.jacobian))


def test_custom_response_still_uses_autograd_fallback(monkeypatch):
    class CustomResponse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = randomized_model()

        def forward(self, points, normalized_distance, context, *, operation):
            return legacy_forward(self.inner, points, normalized_distance, context, operation)

    stencil, nodes, motion = scene()
    custom = CustomResponse()
    context = torch.randn(len(nodes), 7, dtype=torch.float64)
    ad = torch.autograd.grad
    calls = []
    def counted_ad(*args, **kwargs):
        calls.append(1)
        return ad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, 'grad', counted_ad)
    result = response.evaluate_stencil(stencil, model=custom, context=context,
                                      vertex_velocity=motion, chunk_size=2)
    assert len(calls) == 2
    reference = response.evaluate_stencil(stencil, model=custom.inner, context=context,
                                         vertex_velocity=motion, chunk_size=2)
    torch.testing.assert_close(result.jacobian, reference.jacobian, rtol=3e-12, atol=2e-14)


@pytest.mark.parametrize('projected', [False, True])
def test_frozen_stencil_arrays_do_not_create_writable_torch_aliases(projected):
    stencil, nodes, motion = scene()
    model = randomized_model()
    context = torch.randn(len(nodes), 7, dtype=torch.float64)
    if projected:
        context = model.project_context(context)
    velocity = torch.from_numpy(motion).requires_grad_()
    for field in vars(stencil).values():
        if isinstance(field, np.ndarray):
            field.flags.writeable = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = response.evaluate_stencil(stencil, model=model, context=context,
                                          vertex_velocity=velocity, chunk_size=2)
    assert not any('not writable' in str(w.message) for w in caught)
    motion_gradient, = torch.autograd.grad(result.jacobian.square().sum(), velocity)
    assert torch.isfinite(motion_gradient).all() and motion_gradient.abs().sum() > 0


def test_projected_small_difference_does_not_subtract_from_large_retained_mass():
    stencil, _, motion = scene()
    # A single atom near the retained-side saturation; use its same finite face.
    surface = stencil.surface
    eps = 1e-8
    atom = np.array([[.2-.14+eps*.14, .12, -.4]])
    measure = response.prepare_measure(atom, [1.])
    rule = response.prepare_stencil(measure, surface, bandwidth=.14)
    model = randomized_model()
    context = torch.ones(1, 7, dtype=torch.float64)
    actual = response.evaluate_stencil(rule, model=model, context=model.project_context(context),
                                       vertex_velocity=motion, operation='difference')
    assert 0. < actual.moments[0].item() < 1e-14
    assert torch.isfinite(actual.jacobian).all()
    reference = copy.deepcopy(model)
    reference.use_analytic_slope = False
    old = response.evaluate_stencil(rule, model=reference, context=context,
                                    vertex_velocity=motion, operation='difference')
    torch.testing.assert_close(actual.moments, old.moments, rtol=1e-12, atol=0.)
    torch.testing.assert_close(actual.jacobian, old.jacobian, rtol=2e-8, atol=1e-20)
