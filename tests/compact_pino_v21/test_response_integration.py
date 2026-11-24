import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import PackedPrediction, evaluate_packed
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure


def test_fully_retained_uses_block_moments_and_only_adds_shell_once(tmp_path):
    class Sample:
        side = 8
        pitch = .125
        origin = np.zeros(3)
        identity = 'tiny-fixture'

        def iter_chunks(self, chunk_size):
            yield np.array([[.25,.4,.7]]), np.array([.4])
    class Outer:
        def signed_distance(self, x):
            return np.full(len(x), 2.)
    g = BlockMeasure.build(Sample(), tmp_path/'pack', block_voxels=4)
    shell = np.arange(10, dtype=float)*.001
    velocity = np.ones((8,3,3))
    y = evaluate_packed(g, Outer(), .005, 'intersection', shell, velocity)
    z = evaluate_packed(g, Outer(), .005, 'difference', shell, velocity)
    assert isinstance(y, PackedPrediction)
    np.testing.assert_allclose(y.moments, g.moments.sum(0)+shell)
    np.testing.assert_allclose(z.moments, shell)
    assert torch.equal(y.jacobian, torch.zeros((10,3),dtype=torch.float64))
    assert y.diagnostics['band_nodes'] == 0


def test_physical_jacobian_backpropagates_to_pcraft_and_restore_reproduces(tmp_path):
    from experiments.compact_pino_v21.fixtures import mesh_parts, setup, manufactured_operand
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import prepare_material
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand

    torch.manual_seed(830251)
    parts, h = setup('long_column')
    v, f = mesh_parts(parts)
    mat = prepare_material(v, f, np.zeros(3), 1/256.)
    blocks = BlockMeasure.build(mat, tmp_path/'column')
    sv, sf, vel = manufactured_operand(h)
    op = FiniteOperand(sv, sf, np.zeros(len(sf),dtype=bool),'intersection',vel)
    model = CompactPINOv21()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    cache = model.encode_geometry(blocks, np.zeros(10))
    pred = model.evaluate_operand(cache,op)
    # Isolate the mixed derivative: no Moment loss in this gradient check.
    loss = (pred.jacobian[:,0]/mat.moments()[0]).square().sum()
    loss.backward()
    for branch in (model.shared_encoder.pino_encoder, model.shared_encoder.interval_embedding,
                   model.geometry_attention, model.response):
        signal = sum(float(p.grad.abs().square().sum()) for p in branch.parameters() if p.grad is not None)
        assert np.isfinite(signal) and signal > 0
    opt.step()
    with pytest.raises(ValueError, match="stale"):
        model.evaluate_operand(cache,op)
    model.eval()
    post = model.evaluate_operand(model.encode_geometry(blocks,np.zeros(10),inference=True),op)
    path = tmp_path/'roundtrip.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},path)
    other = CompactPINOv21().eval()
    opt2 = torch.optim.AdamW(other.parameters(),lr=3e-5)
    state = torch.load(path,weights_only=True)
    other.load_state_dict(state['model'])
    opt2.load_state_dict(state['optimizer'])
    roundtrip = other.evaluate_operand(other.encode_geometry(blocks,np.zeros(10),inference=True),op)
    torch.testing.assert_close(post.moments,roundtrip.moments,rtol=0,atol=0)
    torch.testing.assert_close(post.jacobian,roundtrip.jacobian,rtol=0,atol=0)
    assert len(opt2.state) == len(opt.state) > 0


def test_zero_editable_mass_is_routed_to_native_physics():
    z = torch.zeros(10, dtype=torch.float64)
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import native_proposal_status
    assert native_proposal_status(PackedPrediction(z+1, torch.eye(10), {}), np.ones(10)) == 'NATIVE_REQUIRED_EMPTY_EDITABLE'
    assert native_proposal_status(PackedPrediction(z+1, None, {}), np.zeros(10)) == 'NATIVE_REQUIRED_DERIVATIVE'
    assert native_proposal_status(PackedPrediction(z+1, torch.eye(10), {}), np.zeros(10)) == 'PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE'


def test_fully_saturated_learned_packed_response_has_exact_zero_gradient(tmp_path):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import LocalRetention
    class Point:
        side=8
        pitch=.125
        origin=np.zeros(3)
        identity='constant-endpoint'
        def iter_chunks(self,chunk_size):
            yield np.array([[.3,.4,.5]]),np.array([1.])
    class Far:
        def signed_distance(self,p):
            return np.full(len(p),3.)
    packed = BlockMeasure.build(Point(), tmp_path/'positive', block_voxels=4)
    model = LocalRetention()
    y = evaluate_packed(packed,Far(),.01,'intersection',np.zeros(10),np.ones((4,3,10)),model=model)
    (y.moments.sum()+y.jacobian.sum()).backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.parameters())
