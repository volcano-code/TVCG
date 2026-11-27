import torch
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.cut_conditioned_attention import CutConditionedCrossAttention
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v24_config import CompactPINOv24Config

def test_zero_init_is_exact_identity_and_zero_delta():
    torch.manual_seed(7)
    m = CutConditionedCrossAttention(width=32, heads=4)
    x, c = torch.randn(2,9,32), torch.randn(2,5,32)
    active = torch.tensor([[1]*9,[1,1,1,0,0,0,0,0,0]], dtype=torch.bool)
    y, delta, stats = m(x,c,active)
    assert torch.equal(y,x)
    assert torch.equal(delta,torch.zeros_like(delta))
    assert stats.active_queries == 12
    assert stats.attention_elements == (9*5 + 3*5)

def test_nonactive_tokens_are_exact_passthrough_after_gate_opens():
    m = CutConditionedCrossAttention(width=32, heads=4)
    with torch.no_grad(): m.residual_gate.fill_(0.3)
    x,c = torch.randn(1,8,32), torch.randn(1,4,32)
    active = torch.tensor([[1,1,0,0,1,0,0,0]],dtype=torch.bool)
    y,delta,_ = m(x,c,active)
    assert torch.equal(y[:,~active[0]],x[:,~active[0]])
    assert torch.equal(delta[:,~active[0]],torch.zeros_like(delta[:,~active[0]]))

def test_cut_tokens_receive_gradient():
    m = CutConditionedCrossAttention(width=32, heads=4)
    with torch.no_grad(): m.residual_gate.fill_(0.25)
    x = torch.randn(1,7,32,requires_grad=True)
    c = torch.randn(1,3,32,requires_grad=True)
    active = torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool)
    y,delta,_ = m(x,c,active)
    (y.square().mean()+delta.square().mean()).backward()
    assert c.grad is not None and torch.isfinite(c.grad).all() and c.grad.abs().sum()>0
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_zero_init_delta_projection_learns_against_nonzero_target():
    m = CutConditionedCrossAttention(width=32, heads=4)
    x,c = torch.randn(1,7,32), torch.randn(1,3,32)
    active = torch.ones(1,7,dtype=torch.bool)
    _,delta,_ = m(x,c,active)
    (delta-torch.ones_like(delta)).square().sum().backward()
    grad = m.response_delta[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0

def test_cut_mask_blocks_invalid_tokens():
    torch.manual_seed(9)
    m = CutConditionedCrossAttention(width=32,heads=4).eval()
    with torch.no_grad(): m.residual_gate.fill_(0.5)
    x,c = torch.randn(1,6,32), torch.randn(1,4,32)
    active = torch.ones(1,6,dtype=torch.bool)
    mask = torch.tensor([[1,1,0,0]],dtype=torch.bool)
    y1,d1,_ = m(x,c,active,mask)
    c2=c.clone(); c2[:,2:]=1e6
    y2,d2,_ = m(x,c2,active,mask)
    assert torch.allclose(y1,y2,atol=1e-6,rtol=1e-6)
    assert torch.allclose(d1,d2,atol=1e-6,rtol=1e-6)

def test_config_is_128_context_not_native_resolution():
    cfg=CompactPINOv24Config()
    assert cfg.native_material_resolution==256
    assert cfg.pcraft_context_grids==(64,128)
    assert len(cfg.fingerprint())==64
