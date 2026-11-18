import pytest
import torch

from research.NeuralMoments.phase2_6_local_correction import (
    LocalCorrectionField, MaterialMasks
)
from research.NeuralMoments.phase2_6_hybrid_unified_optimization import (
    make_hybrid_m80_local_problem, make_hybrid_material_chart,
    hybrid_moment_function, hybrid_zero_correction_identity,
    build_hybrid_m69_kkt_system, evaluate_exact_hybrid_candidate,
    exact_hybrid_merit, make_exact_hybrid_merit_callback,
    HybridNeuralProposal, accept_hybrid_proposal_with_exact_oracle,
)
from research.NeuralMoments.phase2_4_neural_newton import (
    make_s9_chart, safeguarded_kkt_newton_direction
)

TASK={
 "schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],
 "axis_origin":[0.,0.,0.],"contact":[0.,0.,0.],"density":1.,
 "native_resolution":8,"shell_fraction":.03,"protect_radius_fraction":.04,
 "protect_height_fraction":0.,"polar_retention":.9,"min_removal_fraction":.01,
 "max_centroid_rise":1e-7,"com_tolerance":1e-4,"alignment_tolerance":1e-4,
 "principal_gap_min":-1e-8,"require_connected":True,"mass_tolerance":1e-10,
 "inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,"solver_tolerance":1e-9,
 "solver_polar_margin":1e-5,"training_polar_margin":5e-4,
 "objective":"minimum_retained_mass","inertia_reference":"original_fixed_axis",
 "material_policy":"single_density_removal_only","purpose":"research"
}

def fixture():
    dtype=torch.float64
    points=torch.tensor([
      [-.8,-.5,-.4],[-.4,.2,.1],[0.,-.3,.2],[.3,.4,-.2],
      [.7,-.1,.3],[.1,.6,.5],[-.6,.3,-.1],[.5,-.5,-.4]
    ],dtype=dtype)
    fixed=torch.tensor([1,0,0,0,0,0,0,0],dtype=torch.bool)
    protected=torch.tensor([0,1,0,0,0,0,0,0],dtype=torch.bool)
    masks=MaterialMasks(fixed,protected,~(fixed|protected))
    centers=torch.tensor([[-.3,0.,0.],[.35,.2,0.]],dtype=dtype)
    beta=torch.tensor([.1,-1.,0.,0.,0.,0.,0.,0.,0.,0.],dtype=dtype)
    weights=torch.full((8,),.1,dtype=dtype)
    return points,masks,centers,beta,weights

def base_operator(beta):
    # Smooth geometry-conditioned surrogate used only to test the hybrid
    # optimization contract; real M80 is tested by Phase 2.6 integration CI.
    x,y,z=beta[1:4]
    return torch.stack((
      1.2 + .02*beta[0] + .01*beta[4],
      .03 + .01*x, .015 + .01*y, .024 + .005*z,
      .8 + .01*beta[4], .7 + .01*beta[5], .6 + .01*beta[6],
      .02 + .001*beta[7], .015 + .001*beta[8], .01 + .001*beta[9]
    ))

RELAXED_TEST_TASK=dict(TASK, min_removal_fraction=0.0, polar_retention=0.0,
    com_tolerance=1.0, alignment_tolerance=1.0,
    principal_gap_min=-1.0, max_centroid_rise=1.0,
    moment_psd_tolerance=1.0)

def make_hybrid(task=TASK):
    points,masks,centers,beta,weights=fixture()
    field=LocalCorrectionField(centers,support_radius=.9)
    original=torch.tensor([1.2,.012,.006,.018,.8,.7,.6,.02,.015,.01],dtype=points.dtype)
    return make_hybrid_m80_local_problem(
      base_moment_fn=base_operator,local_field=field,points=points,
      quadrature_mass=weights,masks=masks,original_moments=original,
      task=task,length_scale=2.,beta_anchor=beta
    ), beta

def test_zero_psi_is_exact_m80_baseline():
    problem,beta=make_hybrid()
    chart=make_hybrid_material_chart(problem)
    state=chart.encode_zero_correction(beta)
    hybrid,base=hybrid_zero_correction_identity(problem,state)
    assert torch.allclose(hybrid,base,atol=1e-12,rtol=0.)
    assert torch.allclose(problem.local_field.coefficients,
                          torch.zeros_like(problem.local_field.coefficients))

def test_hybrid_operator_has_joint_beta_psi_jacobian():
    problem,beta=make_hybrid()
    chart=make_hybrid_material_chart(problem)
    state=chart.encode_zero_correction(beta).detach()
    state[9:]=torch.tensor([.12,-.08],dtype=state.dtype)
    state.requires_grad_(True)
    fn=hybrid_moment_function(problem)
    jac=torch.autograd.functional.jacobian(fn,state)
    assert jac.shape==(10,chart.dimension)
    assert torch.isfinite(jac).all()
    assert torch.linalg.vector_norm(jac[:,9:])>0.

def test_hybrid_kkt_dimension_includes_local_correction():
    problem,beta=make_hybrid()
    chart=make_hybrid_material_chart(problem)
    state=chart.encode_zero_correction(beta)
    active=torch.zeros(10,dtype=torch.bool)
    system=build_hybrid_m69_kkt_system(
      problem,state,active,event_tolerance=1e-12
    )[1]
    assert system.unknowns.shape==(chart.dimension,)
    assert system.residual.shape==(chart.dimension,)
    assert system.jacobian.shape==(chart.dimension,chart.dimension)
    assert system.derivative_valid

def test_hybrid_exact_oracle_requires_beta_and_psi_and_is_independent():
    problem,beta=make_hybrid(RELAXED_TEST_TASK)
    psi=torch.tensor([.2,-.1],dtype=beta.dtype)
    calls={"n":0}
    exact=problem.original_moments*0.95
    def oracle(b,p):
      calls["n"]+=1
      assert not b.requires_grad and not p.requires_grad
      return {"moments":exact,"connected":True,"topology_valid":True,
              "event_valid":True,"source":"synthetic-independent-hybrid-oracle"}
    result=evaluate_exact_hybrid_candidate(
      beta,psi,exact_material_oracle=oracle,
      original_moments=problem.original_moments,task=problem.task,length_scale=2.
    )
    assert calls["n"]==1
    assert result.source=="synthetic-independent-hybrid-oracle"
    assert torch.equal(result.moments,exact)
    assert exact_hybrid_merit(
      result,connectivity_penalty=1e6,feasibility_penalty=1e4
    )==pytest.approx(float(exact[0]),abs=1e-12)

def test_exact_hybrid_callback_counts_oracle_calls():
    problem,beta=make_hybrid(RELAXED_TEST_TASK)
    calls={"n":0}
    def oracle(b,p):
      calls["n"]+=1
      return {"moments":problem.original_moments,"connected":True,
              "topology_valid":True,"event_valid":True,"source":"independent"}
    callback=make_exact_hybrid_merit_callback(
      problem,exact_material_oracle=oracle,
      state_to_material=lambda s:(s["beta"],s["psi"])
    )
    value=callback({"beta":beta,"psi":torch.zeros(2,dtype=beta.dtype)})
    assert value==pytest.approx(float(problem.original_moments[0]),abs=1e-12)
    assert callback.evaluation_count==1
    assert calls["n"]==1

def test_local_correction_never_changes_fixed_or_protected_samples():
    problem,beta=make_hybrid()
    psi=torch.tensor([2.,-2.],dtype=beta.dtype)
    field=problem.local_field
    hybrid=field(beta,problem.points,problem.masks)
    with torch.no_grad():
      field.coefficients.copy_(psi)
    changed=field(beta,problem.points,problem.masks)
    assert torch.equal(changed[problem.masks.fixed],torch.ones(1,dtype=torch.bool))
    assert torch.equal(changed[problem.masks.protected],torch.ones(1,dtype=torch.bool))
    assert torch.all(changed>=0.) and torch.all(changed<=1.)

if __name__=="__main__":
    pytest.main([__file__,"-q"])
