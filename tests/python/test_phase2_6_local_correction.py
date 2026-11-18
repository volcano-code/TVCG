import pytest
import torch
from research.NeuralMoments.phase2_6_local_correction import (
 LocalCorrectionField,MaterialMasks,quadric_query,integrate_retained_moments,
 evaluate_hybrid_ablation,validate_local_correction_contract,
 hybrid_moments_and_jacobians,local_correction_regularization
)

def fixture():
    dtype=torch.float64
    points=torch.tensor([
      [-.8,-.5,-.4],[-.4,.2,.1],[0.,-.3,.2],[.3,.4,-.2],
      [.7,-.1,.3],[.1,.6,.5],[-.6,.3,-.1],[.5,-.5,-.4]
    ],dtype=dtype)
    fixed=torch.tensor([1,0,0,0,0,0,0,0],dtype=torch.bool)
    protected=torch.tensor([0,1,0,0,0,0,0,0],dtype=torch.bool)
    editable=~(fixed|protected)
    masks=MaterialMasks(fixed,protected,editable)
    centers=torch.tensor([[-.3,0.,0.],[.35,.2,0.]],dtype=dtype)
    beta=torch.tensor([.1,-1.,0.,0.,0.,0.,0.,0.,0.,0.],dtype=dtype)
    weights=torch.full((8,),.1,dtype=dtype)
    return points,masks,centers,beta,weights

def test_zero_initialized_local_correction_is_exact_pure_quadric_baseline():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=.8)
    pure=field.pure_quadric(beta,points,masks)
    hybrid=field(beta,points,masks)
    assert torch.allclose(pure,hybrid,atol=1e-14,rtol=0.)
    assert torch.all(hybrid[masks.fixed]==1.)
    assert torch.all(hybrid[masks.protected]==1.)

def test_local_correction_is_bounded_and_only_changes_editable_samples():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=.8,max_logit_correction=.4)
    with torch.no_grad(): field.coefficients.copy_(torch.tensor([4.,-3.],dtype=points.dtype))
    pure=field.pure_quadric(beta,points,masks)
    hybrid=field(beta,points,masks)
    assert torch.equal(hybrid[masks.fixed],torch.ones_like(hybrid[masks.fixed]))
    assert torch.equal(hybrid[masks.protected],torch.ones_like(hybrid[masks.protected]))
    assert torch.all((hybrid>=0.)&(hybrid<=1.))
    assert torch.allclose(hybrid[~masks.editable],pure[~masks.editable])
    assert torch.max(torch.abs(hybrid-pure))<=.5

def test_local_correction_gradient_flows_to_local_coefficients():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=1.2)
    loss=field(beta,points,masks).sum()
    loss.backward()
    assert field.coefficients.grad is not None
    assert torch.isfinite(field.coefficients.grad).all()
    assert torch.linalg.vector_norm(field.coefficients.grad)>0.

def test_moment_order_and_integration_match_analytic_weighted_points():
    points,masks,centers,beta,weights=fixture()
    retention=torch.ones(len(points),dtype=points.dtype)
    m=integrate_retained_moments(points,weights,retention)
    x,y,z=points.unbind(-1)
    expected=torch.stack((weights.sum(),weights@x,weights@y,weights@z,
      weights@(x*x),weights@(y*y),weights@(z*z),weights@(y*z),
      weights@(x*z),weights@(x*y)))
    assert torch.allclose(m,expected,atol=1e-12,rtol=0.)

def test_hybrid_ablation_reports_without_claiming_improvement():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=.9)
    with torch.no_grad(): field.coefficients.copy_(torch.tensor([.8,-.6],dtype=points.dtype))
    report=evaluate_hybrid_ablation(field,beta,points,weights,masks)
    assert report.pure_moments.shape==(10,)
    assert report.hybrid_moments.shape==(10,)
    assert torch.isfinite(report.correction_l2)
    assert torch.isfinite(report.correction_max_abs)

def test_invalid_overlapping_or_incomplete_masks_are_rejected():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=.8)
    bad=MaterialMasks(masks.fixed,masks.protected,masks.fixed|masks.editable)
    with pytest.raises(ValueError,match="disjoint"):
      field(beta,points,bad)

def test_local_correction_contract_fails_closed():
    with pytest.raises(ValueError,match="not admitted"):
      validate_local_correction_contract({
       "same_frame":True,"same_moment_order":True,"fixed_mask_immutable":True,
       "protected_mask_immutable":True,"editable_mask_explicit":True,
       "bounded_correction":True,"exact_binary_topology_check":False})


def test_joint_beta_local_jacobians_match_finite_difference():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=1.2)
    psi=torch.tensor([.15,-.12],dtype=points.dtype)
    moments,jbeta,jpsi=hybrid_moments_and_jacobians(
      field,beta,psi,points,weights,masks)
    assert moments.shape==(10,)
    assert jbeta.shape==(10,10)
    assert jpsi.shape==(10,2)
    theta=torch.cat((beta,psi))
    eps=1e-6
    for k in (0,1,10,11):
      d=torch.zeros_like(theta);d[k]=eps
      plus=hybrid_moments_and_jacobians(
        field,theta[:10]+d[:10],theta[10:]+d[10:],points,weights,masks)[0]
      minus=hybrid_moments_and_jacobians(
        field,theta[:10]-d[:10],theta[10:]-d[10:],points,weights,masks)[0]
      fd=(plus-minus)/(2*eps)
      analytic=(jbeta[:,k] if k<10 else jpsi[:,k-10])
      assert torch.allclose(analytic,fd,rtol=3e-4,atol=3e-6)

def test_local_correction_regularization_is_finite_and_differentiable():
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=1.2)
    with torch.no_grad():
      field.coefficients.copy_(torch.tensor([.2,-.3],dtype=points.dtype))
    loss=local_correction_regularization(field)
    loss.backward()
    assert torch.isfinite(loss)
    assert field.coefficients.grad is not None
    assert torch.isfinite(field.coefficients.grad).all()
