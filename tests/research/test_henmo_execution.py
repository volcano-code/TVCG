"""Batch/serial physical and training equivalence for frontier execution."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from spin_adapter.henmo_execution import evaluate_stencil_batch
from spin_adapter.henmo_response import (
    HENMOBoundaryResponse, finite_cut_tokens, integrate_boundary_response,
)
from spin_adapter.v21_response import prepare_measure, prepare_stencil, prepare_surface


def box(lo,hi):
    vertices=np.asarray([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],
        [hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
        [hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]],dtype=float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices,faces


def inputs(dtype=torch.float64,*,surface=None):
    if surface is None:
        surface=prepare_surface(*box([-1]*3,[.2,1,1]))
    nodes=[np.array([[.12,-.3,.31],[.23,.12,-.4],[-.7,.1,.2]]),
           np.array([[.08,.2,.1],[.16,-.2,.3],[.19,.3,-.2],[.25,.1,.3],[.28,-.1,-.3],[.7,.2,.3]]),
           np.array([[-.7,.1,.2],[.7,.2,.3]])]
    stencils=[prepare_stencil(prepare_measure(points,np.linspace(.1,.4,len(points))),surface,bandwidth=.15)
              for points in nodes]
    contexts=[torch.linspace(-.4,.5,max(1,len(st.nodes)*3),dtype=dtype)[:len(st.nodes)*3].reshape(-1,3).requires_grad_()
              for st in stencils]
    motion=np.random.default_rng(41).normal(size=(8,3,3))*.03
    cut=finite_cut_tokens(surface,motion,max_tokens=4,dtype=dtype)
    torch.manual_seed(71)
    model=HENMOBoundaryResponse(context_dim=3,hidden_dim=7,zero_init=False).to(dtype=dtype)
    return stencils,contexts,motion,cut,model


def serial(stencils,contexts,model,cut,motion,operation):
    output=[]
    for stencil,context in zip(stencils,contexts):
        signed=torch.tensor(stencil.signed_distance.copy(),dtype=torch.float64)
        live=stencil.surface.tensor_vertices
        if live is not None and not stencil.ambiguous_nodes and len(stencil.nodes):
            faces=torch.tensor(stencil.surface.faces[stencil.face_indices].copy())
            displacement=live-live.detach()
            bary=torch.tensor(stencil.barycentric.copy())
            delta=sum(bary[:,i:i+1]*displacement[faces[:,i]] for i in range(3))
            signed=signed+(torch.tensor(stencil.motion_normal.copy())*delta).sum(1)
        ds=None if motion is None or stencil.ambiguous_nodes else stencil.distance_motion(motion)
        result=integrate_boundary_response(stencil.nodes,stencil.weights,signed,bandwidth=stencil.bandwidth,
            model=model,context=context,cut_tokens=cut.features,distance_motion=ds,
            cut_motion=cut.motion if ds is not None else None,operation=operation)
        base=stencil.interior_moments if operation=='intersection' else stencil.exterior_moments
        output.append((result.moments+torch.tensor(base.copy()),result.jacobian))
    return output


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('operation',['intersection','difference'])
def test_padded_batch_matches_serial_moments_jacobian_and_all_learning_gradients(dtype,operation):
    stencils,contexts,motion,cut,model=inputs(dtype)
    reference_model=copy.deepcopy(model)
    reference_contexts=[value.detach().clone().requires_grad_() for value in contexts]
    with patch.object(model,'retention_and_jacobian',wraps=model.retention_and_jacobian) as call:
        actual=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion,operation)
        assert call.call_count==1
    expected=serial(stencils,reference_contexts,reference_model,cut,motion,operation)
    tolerance=dict(rtol=3e-6,atol=5e-8) if dtype==torch.float32 else dict(rtol=3e-12,atol=3e-13)
    loss=0; expected_loss=0
    for prediction,(moment,jacobian) in zip(actual,expected):
        torch.testing.assert_close(prediction.moments,moment,**tolerance)
        torch.testing.assert_close(prediction.jacobian,jacobian,**tolerance)
        assert prediction.moments.dtype==prediction.jacobian.dtype==torch.float64
        loss=loss+prediction.moments.square().sum()+prediction.jacobian.square().sum()
        expected_loss=expected_loss+moment.square().sum()+jacobian.square().sum()
    loss.backward(); expected_loss.backward()
    for first,second in zip(model.parameters(),reference_model.parameters()):
        assert (first.grad is None)==(second.grad is None)
        if first.grad is not None:
            torch.testing.assert_close(first.grad,second.grad,**tolerance)
    for first,second in zip(contexts,reference_contexts):
        assert (first.grad is None)==(second.grad is None)
        if first.grad is not None:
            torch.testing.assert_close(first.grad,second.grad,**tolerance)
    assert actual.diagnostics['network_calls']==1
    assert actual.diagnostics['active_stencils']==2
    assert actual.diagnostics['padded_atoms']>=actual.diagnostics['active_atoms']
    assert actual.diagnostics['index_add_used'] is False


def test_wave_budget_and_stencil_segmentation_preserve_values_and_tangents():
    stencils,contexts,motion,cut,model=inputs()
    full=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion)
    small=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion,max_padded_atoms=3)
    assert small.diagnostics['network_calls']>1
    assert all(wave['padded_atoms']<=3 for wave in small.diagnostics['waves'])
    for first,second in zip(full,small):
        torch.testing.assert_close(first.moments,second.moments,rtol=2e-12,atol=3e-13)
        torch.testing.assert_close(first.jacobian,second.jacobian,rtol=2e-12,atol=3e-13)


def test_no_grad_and_inference_modes_preserve_explicit_jacobian():
    stencils,contexts,motion,cut,model=inputs()
    reference=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion)
    for mode in (torch.no_grad,torch.inference_mode):
        with mode():
            actual=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion)
        for first,second in zip(actual,reference):
            torch.testing.assert_close(first.moments,second.moments)
            torch.testing.assert_close(first.jacobian,second.jacobian)
            assert not first.moments.requires_grad and not first.jacobian.requires_grad


def test_all_saturated_batch_skips_network_and_has_valid_zero_learning_gradient():
    stencils,contexts,motion,cut,model=inputs()
    with patch.object(model,'retention_and_jacobian',side_effect=AssertionError('network must not run')):
        result=evaluate_stencil_batch(stencils[-1:],contexts[-1:],model,cut.features,cut.motion,motion)
    assert result.diagnostics['network_calls']==0 and result.diagnostics['padded_atoms']==0
    assert torch.count_nonzero(result[0].jacobian)==0
    (result[0].moments.sum()+result[0].jacobian.sum()).backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in model.parameters())


def test_only_ambiguous_stencil_withholds_jacobian():
    surface=prepare_surface(*box([-1]*3,[1]*3))
    stencils=[prepare_stencil(prepare_measure([point],[1.]),surface,bandwidth=1.5)
              for point in ([0.,0.,0.],[.9,.2,.3])]
    assert stencils[0].ambiguous_nodes and not stencils[1].ambiguous_nodes
    contexts=[torch.ones((1,3),dtype=torch.float64) for _ in stencils]
    motion=np.broadcast_to(np.eye(3),(8,3,3))
    cut=finite_cut_tokens(surface,motion,max_tokens=4,dtype=torch.float64)
    model=HENMOBoundaryResponse(context_dim=3,hidden_dim=7,zero_init=False).double()
    actual=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion)
    assert actual[0].jacobian is None and actual[1].jacobian is not None
    assert torch.isfinite(actual[0].moments).all() and torch.isfinite(actual[1].jacobian).all()


def test_live_finite_surface_ad_matches_batched_explicit_tangents():
    vertices,faces=box([-1]*3,[.2,1,1])
    stencils,contexts,motion,cut,model=inputs()
    expected=evaluate_stencil_batch(stencils,contexts,model,cut.features,cut.motion,motion)
    expected_j=sum((item.jacobian for item in expected),torch.zeros((10,3),dtype=torch.float64))
    def evaluate(offset):
        live=torch.tensor(vertices)+torch.einsum('vkd,d->vk',torch.tensor(motion),offset)
        surface=prepare_surface(live,faces)
        active,ctx,_,_,_=inputs(surface=surface)
        desc=finite_cut_tokens(surface,max_tokens=4,dtype=torch.float64)
        result=evaluate_stencil_batch(active,ctx,model,desc.features)
        return sum((row.moments for row in result),torch.zeros(10,dtype=torch.float64))
    origin=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    independent=torch.autograd.functional.jacobian(evaluate,origin)
    torch.testing.assert_close(expected_j,independent,rtol=2e-10,atol=3e-12)


def test_zero_context_width_and_moment_only_batch():
    stencils,_,_,cut,_=inputs()
    model=HENMOBoundaryResponse(context_dim=0,hidden_dim=7,zero_init=False).double()
    result=evaluate_stencil_batch(stencils,None,model,cut.features)
    assert len(result)==3 and all(row.jacobian is None for row in result)
    assert result.diagnostics['network_calls']==1


def test_rejects_cross_surface_or_bandwidth_batch_and_missing_tangent_chain():
    stencils,contexts,motion,cut,model=inputs()
    different=prepare_stencil(prepare_measure([[.1,.2,.3]],[1.]),
        prepare_surface(*box([-1]*3,[.2,1,1])),bandwidth=.15)
    with pytest.raises(ValueError,match='exact finite surface'):
        evaluate_stencil_batch([stencils[0],different],contexts[:2],model,cut.features)
    from dataclasses import replace
    with pytest.raises(ValueError,match='bandwidth'):
        evaluate_stencil_batch([stencils[0],replace(stencils[1],bandwidth=.2)],contexts[:2],model,cut.features)
    with pytest.raises(ValueError,match='cut_motion'):
        evaluate_stencil_batch(stencils,contexts,model,cut.features,vertex_velocity=motion)
    with pytest.raises(ValueError,match='contexts'):
        evaluate_stencil_batch(stencils,contexts[:1],model,cut.features)
