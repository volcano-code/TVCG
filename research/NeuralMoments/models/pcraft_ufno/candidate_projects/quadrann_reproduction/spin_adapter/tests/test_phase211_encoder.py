"""Deterministic native encoder, cache, raw-coefficient and adjoint controls."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research.NeuralMoments.models.pcraft_ufno.spin_pino import native_pcraft as native
from research.NeuralMoments.models.pcraft_ufno.spin_pino.geometry_tiles import sample_native_intervals
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.prepared_quadrature import prepared_quadrature
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.rle_quadrature import rle_gauss_quadrature
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_m80_sqno import UnifiedM80SQNO


def raw():
    return dict(columns=np.array([[1,1,1,4],[1,2,1,5],[2,1,2,5],[2,2,0,3],
                                  [5,5,5,7],[5,6,4,8],[6,5,6,8]]),
                origin=np.array([-.3,.5,-.7]),side=16,h=.17,density=1.3)


def model():
    cfg=native.NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=3,min_native_resolution=1,
        multiscale_enabled=True,multiscale_grids=(8,16),gno_enabled=True,gno_neighbors=3,
        gno_topology_filter=True,transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    return UnifiedM80SQNO(native.PcraftNativeMomentOperator(cfg),
        SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=12,num_layers=2,
                        neighbors=3,gnn_chunk_size=3,fusion_chunk_size=3,query_chunk=2),
        fusion_mode="query_conditioned")


def test_map_sampling_adjoint_and_second_derivative():
    torch.manual_seed(984)
    fmap=torch.randn(1,3,4,5,6,dtype=torch.double,requires_grad=True)
    coords=torch.randn(13,3,dtype=torch.double)
    grad=torch.randn(13,3,dtype=torch.double,requires_grad=True)
    sampled=sample_native_intervals(fmap,coords)
    original=F.grid_sample(fmap,coords.reshape(1,-1,1,1,3),mode="bilinear",
                           padding_mode="border",align_corners=True).reshape(3,-1).T
    torch.testing.assert_close(sampled,original,atol=0,rtol=0)
    adjoint,=torch.autograd.grad(sampled,fmap,grad_outputs=grad,create_graph=True)
    expected,=torch.autograd.grad(original,fmap,grad_outputs=grad)
    torch.testing.assert_close(adjoint,expected,atol=0,rtol=0)
    direction=torch.randn_like(fmap)
    second,=torch.autograd.grad(adjoint,grad,grad_outputs=direction)
    torch.testing.assert_close(second,sample_native_intervals(direction,coords))


def test_prepared_quadrature_and_parameter_free_cache():
    values=raw()
    prepared=native.PreparedNativeGeometry.prepare(values)
    original=rle_gauss_quadrature(prepared,order=3)
    xyz,weights,masses=prepared_quadrature(prepared,3,np.float64)
    np.testing.assert_array_equal(xyz,original["xyz"])
    np.testing.assert_array_equal(weights,original["weights"])
    np.testing.assert_array_equal(masses,np.bincount(original["interval_ids"],weights))
    values["columns"][0]=0
    assert prepared["columns"][0,0]==1
    torch.manual_seed(124)
    net=model()
    cache=net.encode_geometry(prepared)
    with torch.no_grad():
        next(net.parameters()).add_(.01)
    with pytest.raises(ValueError,match="stale"):
        net.evaluate_queries(cache,torch.ones(1,10))
    with patch.object(native,"native_interval_components",side_effect=AssertionError("rebuilt")), \
         patch.object(native,"conservative_coarse_grid",side_effect=AssertionError("rebuilt")):
        refreshed=net.encode_geometry(prepared)
    assert refreshed.geometry_id==cache.geometry_id


def test_training_checkpoint_vs_whole_geometry_outputs_and_gradients():
    torch.manual_seed(171)
    staged=model().train()
    eager=deepcopy(staged).eval()
    eager.sqno.gnn_chunk_size=eager.sqno.fusion_chunk_size=32
    geometry=raw()
    q=torch.randn(2,10,dtype=torch.double)
    directions=torch.randn(2,10,dtype=torch.double)

    def evaluate(net,chunk):
        c=net.encode_geometry(geometry,chunk=chunk)
        coeff=q.clone().requires_grad_()
        out=net.evaluate_queries(c,coeff,create_graph=True)
        moments,raw_j=out["moments"],out["jacobian_coefficients"]
        loss=moments.square().sum()+raw_j.square().sum()
        params=tuple(net.parameters())
        grads=torch.autograd.grad(loss,(*params,coeff),retain_graph=True,allow_unused=True)
        direct,=torch.autograd.grad((moments*directions).sum(),coeff,create_graph=True)
        mixed=torch.autograd.grad((direct*directions).sum(),params,allow_unused=True)
        from_reported=torch.einsum("qo,qoi->qi",directions,raw_j)
        torch.testing.assert_close(direct,from_reported,atol=1e-8,rtol=2e-5)
        return [c.shared_cache.latent,c.sqno_cache.logits,moments,raw_j,*grads,*mixed]

    refs=evaluate(eager,32)
    values=evaluate(staged,3)
    assert sum(v is not None for v in values)>150
    for x,y in zip(refs,values,strict=True):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x,y,rtol=4e-4,atol=2e-5)


def test_transolver_checkpoint_restores_rng_with_dropout():
    torch.manual_seed(191)
    module=native.PhysicsStateSlicing(8,6,2).train()
    module.attn_dropout.p=.25
    module.ffn_dropout.p=.2
    features=torch.randn(19,8,requires_grad=True)
    mass=torch.rand(19)+.1
    state=torch.get_rng_state()
    def run():
        out,_=module(features,mass,point_chunk_size=5,adaptive=True,min_slices=3,
                     adaptive_temperature=True)
        grads=torch.autograd.grad(out.square().sum(),(features,*module.parameters()))
        return (out,*grads,torch.rand(6))
    staged=run()
    torch.set_rng_state(state)
    with patch.object(native,"checkpoint_stage",side_effect=lambda name,fn,*args: fn(*args)):
        reference=run()
    for a,b in zip(staged,reference,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_standalone_sqno_rejects_own_and_exporting_encoder_updates():
    torch.manual_seed(305)
    net=model().train()
    geometry=net.prepare_geometry(raw())
    shared=net.shared_encoder.encode(geometry,retain_interval_features=True)
    payload=net.shared_encoder.export_shared_geometry_features(shared)
    branch=net.sqno
    q=torch.randn(2,10,dtype=torch.double)
    cache=branch.encode_geometry(geometry,pcraft_features=payload)
    opt=torch.optim.SGD(branch.weight_head.parameters(),lr=.01)
    loss=branch.evaluate_queries(cache,q)["moments"].square().sum()
    loss.backward()
    opt.step()
    with pytest.raises(ValueError,match="stale"):
        branch.evaluate_queries(cache,q)
    cache=branch.encode_geometry(geometry,pcraft_features=payload)
    with torch.no_grad():
        net.shared_encoder.interval_embedding[0].weight.add_(.01)
    with pytest.raises(ValueError,match="stale"):
        branch.evaluate_queries(cache,q)
    with pytest.raises(ValueError,match="stale"):
        branch.encode_geometry(geometry,pcraft_features=payload)


def test_no_grad_unified_cache_rejects_training_and_higher_order_evaluation():
    net=model().eval()
    q=torch.ones(1,10,dtype=torch.double)
    with torch.no_grad():
        cache=net.encode_geometry(raw())
    assert not cache.shared_cache.latent.requires_grad
    assert torch.isfinite(net.evaluate_queries(cache,q)["moments"]).all()
    with pytest.raises(ValueError,match="inference-only"):
        net.evaluate_queries(cache,q,create_graph=True)
    net.train()
    with pytest.raises(ValueError,match="inference-only"):
        net.evaluate_queries(cache,q)


def test_frozen_geometry_allows_head_training_from_no_grad_cache():
    net=model().eval()
    net.shared_encoder.requires_grad_(False)
    net.shared_encoder.query_head.requires_grad_(True)
    net.sqno.requires_grad_(False)
    with torch.no_grad():
        cache=net.encode_geometry(raw())
    net.train()
    q=torch.ones(1,10,dtype=torch.double)
    result=net.evaluate_queries(cache,q,create_graph=True)
    (result["moments"].square().sum()+result["jacobian_q"].square().sum()).backward()
    assert net.shared_encoder.query_head[-1].weight.grad.abs().sum()>0
    assert net.query_fusion_gate[-1].bias.grad.abs().sum()>0
    assert net.shared_encoder.interval_embedding[0].weight.grad is None


def test_external_graph_free_encoder_rejects_training_unless_frozen():
    net=model()
    geometry=net.prepare_geometry(raw())
    q=torch.ones(1,10,dtype=torch.double)
    for frozen in (False,True):
        net.shared_encoder.requires_grad_(not frozen)
        with torch.no_grad():
            shared=net.shared_encoder.encode(geometry,retain_interval_features=True)
            payload=net.shared_encoder.export_shared_geometry_features(shared)
        cache=net.sqno.encode_geometry(geometry,pcraft_features=payload)
        if not frozen:
            with pytest.raises(ValueError,match="inference-only"):
                net.sqno.evaluate_queries(cache,q)
        else:
            net.sqno.evaluate_queries(cache,q)["moments"].square().sum().backward()
            assert net.sqno.fusion[1].weight.grad.abs().sum()>0
