"""Dense reference and mixed-derivative checks for the streamed query Jacobian."""
import copy
import math

import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.geometry_physics_attention import GeometryPhysicsCrossAttention


def dense(module, c, f, m, *, fused_norm=False):
    unit = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    token = module.query_encoder(unit.to(f))
    q = module.query(token).reshape(len(c), module.heads, module.dim)
    k = module.key(f).reshape(len(f), module.heads, module.dim)
    v = module.value(f).reshape_as(k)
    prior = m.to(f).clamp_min(1e-30).log() - m.to(f).sum().clamp_min(1e-30).log()
    z = torch.einsum("qhd,nhd->qhn", q, k) / math.sqrt(module.dim) + prior
    weights = z.softmax(-1)
    ctx = torch.einsum("qhn,nhd->qhd", weights, v).flatten(1)
    # Elementary LayerNorm keeps the dense oracle differentiable through three
    # orders, including the Hessian of a loss on an already differentiated output.
    joined = torch.cat((token, ctx), 1)
    centered = joined - joined.mean(-1, keepdim=True)
    layer = module.output[0]
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
    output = (module.output(joined) if fused_norm else module.output[1:](
        normalized * layer.weight + layer.bias)) * module.output_scale
    entropy = -(weights * z.log_softmax(-1)).sum(-1)
    jac = torch.stack([torch.autograd.grad(row.sum(), c, retain_graph=True, create_graph=True)[0]
                       for row in output.unbind(-1)], 1)
    return output, jac, dict(attention_entropy=entropy.mean(-1),
                             attention_effective_support=entropy.exp().mean(-1),
                             attention_query_norm=c.norm(dim=-1),
                             attention_mass_sum=m.sum().expand(len(c)))


def fixture():
    gen = torch.Generator().manual_seed(211)
    torch.manual_seed(211)
    model = GeometryPhysicsCrossAttention(12, heads=3, hidden=17, chunk_intervals=7, chunk_queries=2).double()
    with torch.no_grad():
        model.output[-1].weight.normal_(std=0.7, generator=gen)
        model.output[-1].bias.normal_(std=0.3, generator=gen)
    c = torch.randn(5, 10, dtype=torch.float64, generator=gen)
    f = torch.randn(29, 12, dtype=torch.float64, generator=gen)
    # Force unequal total mass across tiles, including the one-element tail.
    m = torch.exp(torch.linspace(-7, 5, len(f), dtype=torch.float64))
    return model, c.requires_grad_(), f.requires_grad_(), m.requires_grad_()


def relative_l2(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("objective", ["correction", "jacobian", "entropy"])
def test_gradients_vs_dense_autograd(objective):
    mod, c, f, m = fixture()
    args = (c, f, m)
    actual = mod.forward_with_query_jacobian(*args)
    reference = dense(mod, *args)
    for got, want in zip(actual[:2], reference[:2]):
        torch.testing.assert_close(got, want, atol=2e-12, rtol=2e-11)
    for name in reference[2]:
        torch.testing.assert_close(actual[2][name], reference[2][name], rtol=2e-12, atol=2e-12)

    def loss(outputs):
        pred, jac, diag = outputs
        if objective == "correction":
            return pred.square().sum()
        if objective == "jacobian":
            return jac.square().sum()
        return diag["attention_entropy"].square().sum() + diag["attention_effective_support"].sum()

    targets = (*args, *mod.parameters())
    results = [torch.autograd.grad(loss(out), targets, allow_unused=True) for out in (actual, reference)]
    for got, want in zip(*results):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            torch.testing.assert_close(got, want, atol=3e-11, rtol=3e-10)
    if objective == "jacobian":
        # Final additive bias is mathematically absent from the Jacobian.
        active = list(results[0][:-1])
        assert all(g is not None and g.norm() > 1e-11 for g in active)


def test_double_backward_of_jacobian_loss():
    mod, c, f, m = fixture()
    targets = (c, f, m, *(p for name, p in mod.named_parameters() if name != "output.3.bias"))
    actual = mod.forward_with_query_jacobian(c, f, m)
    ref = dense(mod, c, f, m)
    probes = tuple(torch.randn_like(target) for target in targets)

    def hessian_vector(out):
        grad = torch.autograd.grad(out[1].square().sum(), targets, create_graph=True)
        return torch.autograd.grad(sum((v * p).sum() for v, p in zip(grad, probes)), targets)

    analytic_hvp = hessian_vector(actual)
    for got, want in zip(analytic_hvp, hessian_vector(ref)):
        torch.testing.assert_close(got, want, rtol=2e-8, atol=1e-10)

    # Independent centered difference of dense gradients using the original
    # native nn.LayerNorm path, including the final output weights in the probe.
    epsilon = 1e-6
    def displaced_gradient(step):
        with torch.no_grad():
            for param, direction in zip(targets, probes):
                param.add_(direction, alpha=step)
        _, jac, _ = dense(mod, c, f, m, fused_norm=True)
        return torch.autograd.grad(jac.square().sum(), targets)

    plus = displaced_gradient(epsilon)
    minus = displaced_gradient(-2 * epsilon)
    for got, gp, gm in zip(analytic_hvp, plus, minus):
        torch.testing.assert_close(got, (gp - gm) / (2 * epsilon), rtol=3e-7, atol=1e-9)


def test_float32_network_with_float64_physics_vs_double_reference():
    mod, c, f, m = fixture()
    work = copy.deepcopy(mod).float()
    f32 = f.detach().float().requires_grad_()
    pred, jac, _ = work.forward_with_query_jacobian(c, f32, m)
    expected, expected_jac, _ = dense(mod, c, f, m)
    assert pred.dtype == torch.float32 and jac.dtype == c.dtype
    assert relative_l2(pred.double(), expected) < 3e-6
    assert relative_l2(jac, expected_jac) < 3e-6
    got = torch.autograd.grad(jac.square().sum(), (f32, work.key.weight, work.query.weight))
    want = torch.autograd.grad(expected_jac.square().sum(), (f, mod.key.weight, mod.query.weight))
    for a, b in zip(got, want):
        assert relative_l2(a.double(), b) < 6e-6


def test_no_saved_interval_activations_and_no_inner_reverse_mode(monkeypatch):
    mod, c, f, m = fixture()
    saved = []
    def pack(x):
        saved.append(x)
        return x

    def forbidden(*args, **kwargs):
        pytest.fail("analytic operator invoked torch.autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        pred, jac, _ = mod.forward_with_query_jacobian(c, f, m)
    # Only original inputs and one scalar log-prior per interval may be saved.
    # Score, K/V and per-interval derivative tiles never enter the saved graph.
    for x in saved:
        if x.ndim and (len(f) in x.shape or mod.chunk_intervals in x.shape):
            assert x is f or (x.ndim == 1 and len(x) == len(m))
    (pred.square().sum() + jac.square().sum()).backward()
    assert mod.key.weight.grad.norm() > 0
