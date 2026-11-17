import torch

from quadrann_model import QuadrANN


def make_model():
    return QuadrANN(
        dim=2,
        num_gnn_layers=3,
        k_neighbors=16,
        pe_abs=3,
        pe_rel=5,
        hidden_dim=128,
        final_hidden_dim=256,
    )


def test_cpu_forward_and_positive_weights():
    model = make_model().eval()
    x = torch.rand(2, 128, 2)

    with torch.no_grad():
        weights = model(x)

    assert weights.shape == (2, 128)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()
    assert torch.allclose(
        weights.sum(-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_cpu_backward():
    model = make_model().train()
    x = torch.rand(2, 128, 2)

    weights = model(x)
    loss = (weights - 1.0 / 128).square().mean()
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_cuda_forward_and_backward():
    if not torch.cuda.is_available():
        return

    model = make_model().cuda().train()
    x = torch.rand(2, 128, 2, device="cuda")

    weights = model(x)

    assert weights.is_cuda
    assert torch.isfinite(weights).all()
    assert torch.allclose(
        weights.sum(-1),
        torch.ones(2, device="cuda"),
        atol=1e-6,
    )

    loss = (weights - 1.0 / 128).square().mean()
    loss.backward()

    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
