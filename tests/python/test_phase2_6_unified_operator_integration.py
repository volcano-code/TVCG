import unittest
import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
 NativePCRAFTConfig,PcraftNativeMomentOperator
)
from research.NeuralMoments.m69_task_conditioned_m80_v1 import (
 M69TaskConditionedM80,MaterialMapContract
)
from research.NeuralMoments.phase2_6_unified_operator_integration import (
 make_m80_m69_problem,m80_moment_function,evaluate_exact_m69_candidate,
 exact_m69_merit,propose_m80_m69_newton_step,make_exact_m69_merit_callback
)

TASK={"schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],
"contact":[0.,0.,0.],"density":1.,"native_resolution":8,"shell_fraction":.03,
"protect_radius_fraction":.04,"protect_height_fraction":0.,"polar_retention":.9,
"min_removal_fraction":.01,"max_centroid_rise":1e-7,"com_tolerance":1e-4,
"alignment_tolerance":1e-4,"principal_gap_min":-1e-8,"require_connected":True,
"mass_tolerance":1e-10,"inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,
"solver_tolerance":1e-9,"solver_polar_margin":1e-5,"training_polar_margin":5e-4,
"objective":"minimum_retained_mass","inertia_reference":"original_fixed_axis",
"material_policy":"single_density_removal_only","purpose":"research"}

def make_geometry():
    rng=np.random.default_rng(2601)
    raw=rng.normal(size=(40,3))
    raw=raw-raw.mean(axis=0,keepdims=True)
    sample_cov=(raw.T@raw)/len(raw)
    target_cov=np.array([[.20,2e-6,4e-6],[2e-6,.15,6e-6],[4e-6,6e-6,.10]])
    eigval,eigvec=np.linalg.eigh(sample_cov)
    whiten=eigvec@np.diag(1./np.sqrt(eigval))@eigvec.T
    fixed_xyz=raw@whiten@np.linalg.cholesky(target_cov).T
    fixed_xyz+=np.array([1e-5,2e-5,.02])
    fixed_w=np.full(40,.125)
    def basis(x):
        a,b,c=x.T
        return np.stack((np.ones_like(a),a,b,c,a*a,b*b,c*c,b*c,a*c,a*b),axis=-1)
    fixed=(fixed_w@basis(fixed_xyz))
    xyz=rng.uniform(-.6,.6,(16,3))
    mass=np.full(16,.01)
    editable=mass@basis(xyz)
    extent=np.full((16,3),.08)
    grid=np.zeros((4,8,8,8),dtype=np.float32)
    grid[0]=1.;grid[1]=1.
    grid[2,1:7,1:7,1:7]=1.
    grid[3]=np.linspace(.1,.9,8,dtype=np.float32)[None,None,:]
    return {"xyz":xyz,"mass":mass,"cell_extent":extent,"moment_basis":basis(xyz),
      "fixed_moments":fixed,"original_moments":fixed+editable,
      "native_resolution":8,"origin":np.array([-1.,-1.,-1.]),"h":.25,
      "pcraft_grid":grid,"geometry_sha256":"phase26-synthetic-native-geometry"}

def make_problem():
    torch.manual_seed(2602)
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
      interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=8,
      convex_potential_enabled=True,convex_potential_components=2)
    base=PcraftNativeMomentOperator(cfg)
    model=M69TaskConditionedM80(base,freeze_backbone=True,
      conditioning_mode="conditional_norm",hidden_dim=16)
    geometry=make_geometry()
    beta=torch.tensor([.2,-.3,.1,.5,.2,.4,-.1,.3,.2,.1],dtype=torch.float64)
    contract=MaterialMapContract(True,True,True,True,True)
    return make_m80_m69_problem(model,geometry,TASK,length_scale=2.,
      beta_anchor=beta,material_map_contract=contract),geometry

class Phase26UnifiedOperatorIntegrationTests(unittest.TestCase):
 def test_actual_m80_native_encoder_and_task_conditioned_moment_forward(self):
    problem,_=make_problem()
    fn=m80_moment_function(problem)
    moment=fn(problem.beta_anchor)
    self.assertEqual(tuple(moment.shape),(10,))
    self.assertTrue(torch.isfinite(moment).all())
    self.assertGreater(float(moment[0].detach()),0.)
    self.assertFalse(problem.model.base_model.training)
    self.assertTrue(all(not p.requires_grad for p in problem.model.base_model.parameters()))

 def test_actual_m80_query_jacobian_is_finite(self):
    problem,_=make_problem()
    q=problem.beta_anchor.detach().clone().requires_grad_(True)
    moment=m80_moment_function(problem)(q)
    jac=torch.autograd.functional.jacobian(m80_moment_function(problem),q)
    self.assertEqual(tuple(jac.shape),(10,10))
    self.assertTrue(torch.isfinite(moment).all())
    self.assertTrue(torch.isfinite(jac).all())

 def test_material_map_contract_is_mandatory(self):
    p,geometry=make_problem()
    bad=MaterialMapContract(True,False,True,True,True)
    with self.assertRaisesRegex(ValueError,"query-to-material Jacobian contract"):
      make_m80_m69_problem(p.model,geometry,TASK,length_scale=2.,
        beta_anchor=p.beta_anchor,material_map_contract=bad)

 def test_independent_exact_oracle_is_not_replaced_by_model_prediction(self):
    problem,geometry=make_problem()
    calls={"count":0}
    exact_candidate=torch.as_tensor(
      geometry["fixed_moments"]+(1.0-.01*geometry["original_moments"][0]/
        (geometry["mass"]@geometry["moment_basis"])[0])*
        (geometry["mass"]@geometry["moment_basis"]),
      dtype=torch.float64)
    def oracle(beta):
      calls["count"]+=1
      # Independent exact fixture retains 99% of the editable material,
      # satisfying this synthetic TaskSpec's 1% minimum-removal requirement.
      return {"moments":exact_candidate,
        "connected":True,"topology_valid":True,"event_valid":True,
        "source":"synthetic-independent-native-moment-fixture"}
    evaluation=evaluate_exact_m69_candidate(problem.beta_anchor,
      exact_material_oracle=oracle,original_moments=problem.original_moments,
      task=problem.task,length_scale=problem.length_scale)
    self.assertEqual(calls["count"],1)
    self.assertEqual(evaluation.source,"synthetic-independent-native-moment-fixture")
    self.assertTrue(torch.equal(evaluation.moments,exact_candidate))
    merit=exact_m69_merit(evaluation,connectivity_penalty=1e6,feasibility_penalty=1e4)
    expected=float(evaluation.moments[0])+1e4*float(
      torch.clamp(evaluation.constraints.g_le_zero,min=0).sum())
    self.assertAlmostEqual(merit,expected,places=8)
    self.assertLess(float(torch.clamp(evaluation.constraints.g_le_zero,min=0).sum()),1e-8)
    self.assertTrue(evaluation.connected)
    self.assertTrue(evaluation.topology_valid)

 def test_exact_merit_penalizes_disconnected_or_invalid_topology(self):
    problem,_=make_problem()
    def oracle(beta):
      return {"moments":problem.original_moments,"connected":False,
        "topology_valid":False,"event_valid":True,"source":"synthetic-disconnected-fixture"}
    evaluation=evaluate_exact_m69_candidate(problem.beta_anchor,
      exact_material_oracle=oracle,original_moments=problem.original_moments,
      task=problem.task,length_scale=problem.length_scale)
    merit=exact_m69_merit(evaluation,connectivity_penalty=1e6,feasibility_penalty=1e4)
    self.assertGreater(merit,float(problem.original_moments[0])+1e5)


 def test_actual_m80_to_s9_m69_kkt_newton_proposal(self):
    from research.NeuralMoments.phase2_6_unified_operator_integration import (
      propose_m80_m69_newton_step
    )
    problem,geometry=make_problem()
    with torch.no_grad():
      predicted=m80_moment_function(problem)(problem.beta_anchor)
    ratio=float(predicted[0]/problem.original_moments[0])
    task=dict(problem.task)
    active=torch.zeros(10,dtype=torch.bool)
    if ratio<=1.0:
      task["min_removal_fraction"]=max(0.0,1.0-ratio)
      active[2]=True
    else:
      task["mass_tolerance"]=max(1e-10,ratio-1.0)
      active[1]=True
    problem=make_m80_m69_problem(problem.model,geometry,task,length_scale=2.,
      beta_anchor=problem.beta_anchor,
      material_map_contract=problem.material_map_contract)
    proposal=propose_m80_m69_newton_step(problem,torch.zeros(9,dtype=torch.float64),
      active,trust_radius=.05,event_tolerance=1e-12)
    self.assertEqual(tuple(proposal.jacobian.shape),(10,10))
    self.assertEqual(proposal.residual.shape,(10,))
    self.assertTrue(torch.isfinite(proposal.moments).all())
    self.assertTrue(torch.isfinite(proposal.direction.step).all())
    self.assertLessEqual(proposal.direction.primal_step_norm,.0500001)


 def test_counted_exact_merit_callback_uses_independent_oracle(self):
    problem,geometry=make_problem()
    calls={"count":0}
    exact_candidate=torch.as_tensor(
      geometry["fixed_moments"]+.99*(geometry["mass"]@geometry["moment_basis"]),
      dtype=torch.float64)
    def oracle(beta):
      calls["count"]+=1
      return {"moments":exact_candidate,"connected":True,
        "topology_valid":True,"event_valid":True,"source":"independent-test-oracle"}
    merit=make_exact_m69_merit_callback(problem,exact_material_oracle=oracle,
      state_to_beta=lambda state:state["beta"],connectivity_penalty=1e6,
      feasibility_penalty=1e4)
    value=merit({"beta":problem.beta_anchor})
    self.assertTrue(np.isfinite(value))
    self.assertEqual(calls["count"],1)
    self.assertEqual(merit.evaluation_count,1)

if __name__=="__main__": unittest.main()
