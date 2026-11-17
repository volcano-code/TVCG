from dataclasses import replace
import numpy as np
import pytest
import torch
from task69 import TaskSpec, evaluate_moments
from research.NeuralMoments.experiments.pino_material_v1.model import PINOMaterialV1
from research.NeuralMoments.experiments.pino_material_v1.test_contracts import fixture, queries
from .material import bounded_material
from .physics import rotation_objective

torch.set_num_threads(2)
D=torch.float64

def setup():
    torch.manual_seed(51)
    model=PINOMaterialV1(width=2,modes=2,layers=2).eval()
    base=replace(fixture(),lower=(-1.,-.8,-1.2),upper=(1.,.8,1.2))
    original=model.encode(base).full.detach().clone()
    logits=torch.linspace(-.3,1.5,512,dtype=D).reshape(8,8,8)
    task=TaskSpec(axis=(3,2,1),axis_origin=(.2,-.15,.3),contact=(.2,-.15,.3),principal_gap_min=.4,min_removal_fraction=.9)
    return model,base,original,logits,task

@pytest.mark.parametrize('smooth',[False,True])
def test_density_bounds_fixed_support_and_constant_preservation(smooth):
    _,base,_,_,_=setup()
    logits=torch.zeros_like(base.density,requires_grad=True)
    g=bounded_material(base,logits,smoothing=smooth)
    torch.testing.assert_close(g.density,base.density*.5*base.editable)
    assert torch.equal(g.fixed_density,base.fixed_density)
    assert torch.equal(g.fixed,base.fixed)
    g.density.sum().backward()
    assert torch.count_nonzero(logits.grad[base.fixed.bool()])==0
    assert logits.grad[base.editable.bool()].min()>0

@pytest.mark.parametrize('axis',[(0,0,1),(1,2,3)])
def test_all_constraints_match_frozen_task69(axis):
    model,base,original,logits,task=setup();task=replace(task,axis=axis)
    moments=model(bounded_material(base,logits),queries()[2:])
    value=rotation_objective(moments,original,task,base.length)
    expected=evaluate_moments(moments[0].detach().numpy(),original.numpy(),task,base.length)
    for key,number in expected['constraint_values'].items():
        assert value['constraints'][key].item()==pytest.approx(number,abs=1e-11,rel=1e-11)
    assert 'complete_pass' not in value

def test_original_normalizer_must_be_constant():
    model,base,original,logits,task=setup()
    m=model(base,queries()[2:])
    with pytest.raises(ValueError):rotation_objective(m,original.requires_grad_(),task,base.length)

def test_invalid_density_logits_and_empty_material():
    model,base,original,logits,task=setup()
    with pytest.raises(ValueError):bounded_material(base,logits*float('nan'))
    with pytest.raises(ValueError):bounded_material(replace(base,density=base.density.requires_grad_()),logits)
    with pytest.raises(ValueError):rotation_objective(torch.zeros(1,10,dtype=D),original,task,base.length)

@pytest.mark.parametrize('name',['mass','com','alignment','polar','principal','height','minimum_removal'])
def test_each_active_loss_reaches_pino_query_and_material_and_matches_fd(name):
    model,base,original,logits,task=setup()
    logits.requires_grad_(); c=queries()[2:].clone().requires_grad_()
    def run(z,q):return rotation_objective(model(bounded_material(base,z),q),original,task,base.length)['losses'][name].sum()
    loss=run(logits,c)
    assert loss.item()>1e-10, name
    gz,gc,gw=torch.autograd.grad(loss,(logits,c,model.bandwidth.weight),retain_graph=True)
    assert all(torch.isfinite(x).all() and x.abs().max()>1e-12 for x in (gz,gc,gw)),name
    dz=torch.linspace(-.8,1.,512,dtype=D).reshape_as(logits)
    dc=torch.tensor([[.3,-.1,.2,.1,-.2,.15,.3,-.1,.2,.15]],dtype=D)
    eps=.002
    fd_z=(run(logits.detach()+eps*dz,c.detach())-run(logits.detach()-eps*dz,c.detach()))/(2*eps)
    torch.testing.assert_close((gz*dz).sum(),fd_z,atol=1e-6,rtol=5e-3)
    eps=1e-5
    fd_c=(run(logits.detach(),c.detach()+eps*dc)-run(logits.detach(),c.detach()-eps*dc))/(2*eps)
    torch.testing.assert_close((gc*dc).sum(),fd_c,atol=1e-7,rtol=1e-4)
    direction=torch.ones_like(model.bandwidth.weight)
    before=model.bandwidth.weight.detach().clone();eps=.003
    with torch.no_grad():model.bandwidth.weight.copy_(before+eps*direction)
    plus=run(logits.detach(),c.detach()).detach()
    with torch.no_grad():model.bandwidth.weight.copy_(before-eps*direction)
    minus=run(logits.detach(),c.detach()).detach()
    with torch.no_grad():model.bandwidth.weight.copy_(before)
    torch.testing.assert_close((gw*direction).sum().to(D),(plus-minus)/(2*eps),atol=2e-6,rtol=8e-3)

def test_reference_does_not_shrink_with_density():
    model,base,original,logits,task=setup()
    m=model(bounded_material(base,logits),queries()[:1])
    out=rotation_objective(m,original,task,base.length)
    assert out['metrics']['mass_fraction'].item()<.9
    assert out['metrics']['polar_retention'].item()<.9

def test_inactive_constraints_zero_and_gradients_finite():
    model,base,original,_,task=setup()
    task=TaskSpec(axis=(0,0,1),min_removal_fraction=0,polar_retention=.5,com_tolerance=1,alignment_tolerance=1,principal_gap_min=-2,max_centroid_rise=1)
    m=original[None].clone().requires_grad_()
    out=rotation_objective(m,original,task,base.length)
    for key in ['com','alignment','polar','principal','height','minimum_removal','no_added_mass','no_added_polar']:
        assert out['losses'][key].item()==0
    gradient=torch.autograd.grad(sum(x.sum() for x in out['losses'].values()),m)[0]
    assert torch.isfinite(gradient).all()


def test_rotational_objectives_invariant_to_physical_length_rescaling():
    model,base,original,logits,task=setup()
    m=model(bounded_material(base,logits),queries()[2:])
    result=rotation_objective(m,original,task,base.length)
    powers=m.new_tensor([3,4,4,4,5,5,5,5,5,5])
    for s in (.01,100.):
        scaled=replace(task,axis_origin=tuple(s*x for x in task.axis_origin),contact=tuple(s*x for x in task.contact))
        other=rotation_objective(m*s**powers,original*s**powers,scaled,base.length*s)
        for key,value in result['losses'].items():
            torch.testing.assert_close(other['losses'][key],value,atol=1e-12,rtol=1e-10)


def test_smoothing_does_not_add_material_or_connect_gradients_to_empty_cells():
    _,base,_,logits,_=setup()
    editable=base.editable.clone();editable[:,3:5,3:5]=0
    base=replace(base,editable=editable)
    logits=logits.clone().requires_grad_()
    result=bounded_material(base,logits)
    assert (result.density>=0).all() and (result.density<=base.density).all()
    result.density.sum().backward()
    assert torch.count_nonzero(logits.grad[~editable.bool()])==0
