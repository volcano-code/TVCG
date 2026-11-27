"""Independent AD, rebuilt finite geometry, and physical conservation oracles."""
import numpy as np
import pytest
import torch
from unittest.mock import patch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.henmo_response import (
    HENMOBoundaryResponse, finite_cut_tokens, integrate_boundary_response,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import (
    prepare_measure, prepare_stencil, prepare_surface,
)


def _box(lo, hi):
    vertices = np.array([[lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
                         [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
                         [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
                         [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]], dtype=float)
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices, faces


def _case(dim=3, attention=True, profile='compact_warp'):
    torch.manual_seed(1703)
    model = HENMOBoundaryResponse(context_dim=4, hidden_dim=8,
                                  use_cut_attention=attention, zero_init=False,profile=profile).double()
    encoder = torch.nn.Linear(3, 4).double()
    points = torch.tensor([[.1,.2,.3],[-.4,.1,.2],[.2,-.3,.4],[-.2,-.1,.3]], dtype=torch.float64)
    weights = torch.tensor([.1,.2,.3,.4], dtype=torch.float64)
    distance = torch.tensor([-.12,.07,.09,-.03], dtype=torch.float64)
    motion = torch.randn(4, dim, dtype=torch.float64)*.3
    cut = torch.randn(3, 7, dtype=torch.float64)*.4
    cut_motion = torch.randn(3, 7, dim, dtype=torch.float64)*.2
    return model, encoder, points, weights, distance, motion, cut, cut_motion


@pytest.mark.parametrize('dimension', [3,8,10])
@pytest.mark.parametrize('attention', [False,True])
@pytest.mark.parametrize('profile',['compact_warp','bounded_residual'])
def test_explicit_tangent_matches_independent_ad_and_trains_all_relevant_branches(dimension, attention, profile):
    model, encoder, nodes, weights, d, ds, cut, dc = _case(dimension, attention, profile)
    context = encoder(nodes)
    args = dict(bandwidth=.2, model=model, context=context, cut_tokens=cut,
                point_origin=(-.3,-.2,-.4), point_scale=1.3)
    # Analytic inference must not invoke reverse AD to obtain ten rows.
    with patch('torch.autograd.grad', side_effect=AssertionError('reverse AD in explicit inference')):
        out = integrate_boundary_response(nodes, weights, d, distance_motion=ds,
                                           cut_motion=dc, **args)

    def value(offset):
        kwargs = dict(args, cut_tokens=cut+torch.einsum('sfd,d->sf',dc,offset))
        return integrate_boundary_response(nodes,weights,d+ds@offset,**kwargs).moments

    reference = torch.autograd.functional.jacobian(value, torch.zeros(dimension,dtype=torch.float64))
    torch.testing.assert_close(out.jacobian,reference,atol=3e-14,rtol=3e-12)
    target = torch.linspace(-.03,.04,10*dimension,dtype=torch.float64).reshape(10,dimension)
    (out.jacobian-target).square().sum().backward()
    required = [encoder.weight,model.geometry_query.weight,model.cut_value.weight,
                model.mix.weight,model.output.weight]
    if attention:
        required.append(model.cut_key.weight)
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item()>1e-12


def test_cut_conditioning_chain_cannot_be_omitted_when_distance_does_not_move():
    model, encoder, nodes, weights, d, ds, cut, dc = _case()
    args = dict(bandwidth=.2,model=model,context=encoder(nodes),cut_tokens=cut)
    with pytest.raises(ValueError,match='cut_motion is required'):
        integrate_boundary_response(nodes,weights,d,distance_motion=ds,**args)
    out = integrate_boundary_response(nodes,weights,d,distance_motion=torch.zeros_like(ds),cut_motion=dc,**args)
    assert out.jacobian.abs().sum()>1e-6
    finite = []
    for col in range(3):
        plus=integrate_boundary_response(nodes,weights,d,**dict(args,cut_tokens=cut+1e-5*dc[:,:,col]))
        minus=integrate_boundary_response(nodes,weights,d,**dict(args,cut_tokens=cut-1e-5*dc[:,:,col]))
        finite.append((plus.moments-minus.moments)/2e-5)
    torch.testing.assert_close(out.jacobian,torch.stack(finite,1),atol=1e-11,rtol=2e-6)


def test_same_operand_complement_and_saturation_preserve_all_physical_moments():
    model,encoder,nodes,weights,_,_,cut,dc=_case()
    d=torch.tensor([-2.,-.08,.13,2.],dtype=torch.float64)
    ds=torch.ones((4,3),dtype=torch.float64)
    args=dict(bandwidth=.2,model=model,context=encoder(nodes),cut_tokens=cut,
              distance_motion=ds,cut_motion=dc)
    inside=integrate_boundary_response(nodes,weights,d,**args)
    outside=integrate_boundary_response(nodes,weights,d,operation='difference',**args)
    reference=prepare_measure(nodes.numpy(),weights.numpy()).total_moments
    np.testing.assert_allclose((inside.moments+outside.moments).detach().numpy(),reference,atol=2e-16)
    torch.testing.assert_close(inside.jacobian+outside.jacobian,torch.zeros_like(inside.jacobian),atol=1e-15,rtol=0)
    assert bool(((inside.retention>=0)&(inside.retention<=1)).all())
    assert inside.retention[0]==0 and inside.retention[-1]==1
    assert inside.diagnostics['active_nodes']==2
    assert inside.diagnostics['attention_elements']==2*len(cut)
    all_full=integrate_boundary_response(nodes,weights,torch.ones_like(d)*2,**args)
    torch.testing.assert_close(all_full.jacobian,torch.zeros((10,3),dtype=torch.float64),rtol=0,atol=0)
    assert all_full.diagnostics['attention_elements']==0


def test_normalized_neural_frame_does_not_replace_physical_moment_coordinates():
    model,encoder,nodes,weights,d,ds,cut,dc=_case()
    context=encoder(nodes)
    args=dict(bandwidth=.2,model=model,context=context,cut_tokens=cut,distance_motion=ds,cut_motion=dc)
    old=integrate_boundary_response(nodes,weights,d,**args)
    scale=3.;shift=torch.tensor([2.,-4.,.7],dtype=torch.float64)
    new=integrate_boundary_response(nodes*scale+shift,weights,d*scale,
           **dict(args,bandwidth=.2*scale,distance_motion=ds*scale,point_origin=shift,point_scale=scale))
    torch.testing.assert_close(new.retention,old.retention,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(new.moments[0],old.moments[0],rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(new.moments[1:4],scale*old.moments[1:4]+shift*old.moments[0],rtol=1e-13,atol=1e-14)


def test_zero_init_matches_unlearned_baseline_and_two_j_updates_open_conditioning():
    model,encoder,nodes,weights,d,ds,cut,dc=_case()
    torch.nn.init.zeros_(model.output.weight);torch.nn.init.zeros_(model.output.bias)
    context=encoder(nodes).detach()
    args=dict(bandwidth=.2,model=model,context=context,cut_tokens=cut,distance_motion=ds,cut_motion=dc)
    baseline=integrate_boundary_response(nodes,weights,d,bandwidth=.2,distance_motion=ds)
    learned=integrate_boundary_response(nodes,weights,d,**args)
    torch.testing.assert_close(learned.moments,baseline.moments,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(learned.jacobian,baseline.jacobian,rtol=1e-13,atol=1e-14)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    initial=model.output.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result=integrate_boundary_response(nodes,weights,d,**args)
        loss=(result.moments-baseline.moments.detach()*.97).square().sum()+(result.jacobian-.01).square().sum()
        loss.backward()
        assert model.output.weight.grad.abs().sum()>0
        if step:
            assert model.cut_key.weight.grad.abs().sum()>1e-12
        optimizer.step()
    assert not torch.equal(initial,model.output.weight)


def test_compact_warp_can_represent_full_or_empty_retention_inside_wide_support():
    model,encoder,nodes,weights,d,ds,cut,dc=_case()
    assert model.profile=='compact_warp'
    with torch.no_grad():
        model.output.weight.zero_()
    context=encoder(nodes)
    args=dict(bandwidth=.2,model=model,context=context,cut_tokens=cut,distance_motion=ds,cut_motion=dc)
    for bias,expected in ((3.,1.),(-3.,0.)):
        with torch.no_grad():model.output.bias.fill_(bias)
        out=integrate_boundary_response(nodes,weights,torch.zeros_like(d),**args)
        torch.testing.assert_close(out.retention,torch.full_like(d,expected),rtol=0,atol=0)
        torch.testing.assert_close(out.jacobian,torch.zeros_like(out.jacobian),rtol=0,atol=0)
    # The old bounded residual remains an explicit, semantically different arm.
    residual,_,_,_,_,_,_,_=_case(profile='bounded_residual')
    residual.load_state_dict(model.state_dict())
    legacy=integrate_boundary_response(nodes,weights,torch.zeros_like(d),**dict(args,model=residual))
    assert bool((legacy.retention>.25).all())
    assert bool((legacy.retention<.75).all())
    with pytest.raises(ValueError,match='profile'):
        HENMOBoundaryResponse(profile='unknown')


def test_finite_cut_tokens_include_caps_and_full_descriptor_derivatives():
    vertices,faces=_box([-1.,-1.,-1.],[.2,1.,1.])
    velocity=np.random.default_rng(70).normal(size=(8,3,3))*.2
    args=dict(origin=(-1.,-1.,-1.),length_scale=2.,max_tokens=4)
    tokens=finite_cut_tokens(prepare_surface(vertices,faces),velocity,**args)
    assert tokens.features.shape==(4,7)
    assert tokens.motion.shape==(4,7,3)
    assert len(tokens.face_groups)==len(faces)
    assert set(tokens.face_groups.tolist())==set(range(4))
    for step in (1e-4,1e-5):
        columns=[]
        for col in range(3):
            plus=finite_cut_tokens(prepare_surface(vertices+step*velocity[:,:,col],faces),**args).features
            minus=finite_cut_tokens(prepare_surface(vertices-step*velocity[:,:,col],faces),**args).features
            columns.append((plus-minus)/(2*step))
        torch.testing.assert_close(tokens.motion,torch.stack(columns,2),rtol=2e-7,atol=3e-10)
    # Independent AD through descriptor values, not through the tangent code.
    def at_q(q):
        moving=torch.as_tensor(vertices)+torch.einsum('vkd,d->vk',torch.as_tensor(velocity),q)
        return finite_cut_tokens(prepare_surface(moving,faces),**args).features
    ref=torch.autograd.functional.jacobian(at_q,torch.zeros(3,dtype=torch.float64))
    torch.testing.assert_close(tokens.motion,ref,rtol=2e-12,atol=2e-14)


def test_complete_response_matches_rebuilt_face_edge_and_cap_queries():
    vertices,faces=_box([-1.,-1.,-1.],[.2,1.,1.])
    nodes=np.array([[.18,-.3,.31],[.23,.12,-.4],[.25,1.03,.2],[.22,1.03,1.04]])
    weights=np.array([.1,.2,.3,.4])
    measure=prepare_measure(nodes,weights)
    velocity=np.random.default_rng(12).normal(size=(8,3,3))*.15
    torch.manual_seed(43)
    model=HENMOBoundaryResponse(context_dim=3,hidden_dim=8,zero_init=False).double()
    encoder=torch.nn.Linear(3,3).double()
    context=encoder(torch.tensor(nodes))
    def solve(v,motion=None):
        surface=prepare_surface(v,faces)
        stencil=prepare_stencil(measure,surface,bandwidth=.14)
        assert stencil.ambiguous_nodes==0 and len(stencil.nodes)==len(nodes)
        tokens=finite_cut_tokens(surface,motion,max_tokens=4)
        return integrate_boundary_response(nodes,weights,stencil.signed_distance,bandwidth=.14,
                  model=model,context=context,cut_tokens=tokens.features,
                  distance_motion=None if motion is None else stencil.distance_motion(motion),
                  cut_motion=tokens.motion)
    out=solve(vertices,velocity)
    for step in (1e-4,1e-5):
        fd=torch.stack([(solve(vertices+step*velocity[:,:,i]).moments-solve(vertices-step*velocity[:,:,i]).moments)/(2*step)
                         for i in range(3)],1)
        torch.testing.assert_close(out.jacobian,fd,rtol=1e-5,atol=3e-9)


@pytest.mark.parametrize('bad_weights', [[.1,-.2,.3,.4],[.1,0.,.3,.4],[.1,np.nan,.3,.4]])
def test_rejects_nonphysical_material_weights(bad_weights):
    _,_,nodes,_,d,_,_,_=_case()
    with pytest.raises(ValueError):
        integrate_boundary_response(nodes,bad_weights,d,bandwidth=.2)
