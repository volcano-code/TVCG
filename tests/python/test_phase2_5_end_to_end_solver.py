import unittest
import torch
from research.NeuralMoments.phase2_5_end_to_end_solver import (
 ActiveSetState,update_active_set,kkt_direction,solve_synthetic_qp,compare_exact_and_surrogate,
 propose_m69_neural_newton_step
)

class Phase25EndToEndTests(unittest.TestCase):
 def test_active_set_uses_g_le_zero_and_lambda_nonnegative(self):
  g=torch.tensor([.1,-.5,-1e-10],dtype=torch.float64)
  state=update_active_set(g,torch.tensor([False,True,False]),
    torch.tensor([-0.1],dtype=torch.float64),primal_tolerance=1e-8)
  self.assertFalse(bool(state.mask[1]))
  self.assertTrue(bool(state.mask[0]))
  self.assertTrue(bool(state.mask[2]))
  self.assertEqual(state.multipliers.numel(),2)

 def test_kkt_direction_solves_active_linear_constraint(self):
  def f(x): return .5*((x-torch.tensor([1.,1.],dtype=x.dtype))**2).sum()
  def g(x): return torch.stack((x.sum()-.2,))
  x=torch.tensor([.1,.1],dtype=torch.float64)
  dx,dl,ids=kkt_direction(f,g,x,torch.tensor([True]),torch.tensor([1.],dtype=x.dtype),
                           trust_radius=10.)
  self.assertEqual(ids.tolist(),[0])
  self.assertTrue(torch.isfinite(dx).all())
  self.assertTrue(torch.isfinite(dl).all())

 def test_exact_newton_baseline_converges(self):
  r=solve_synthetic_qp(method="exact_newton",seed=2501)
  self.assertTrue(r.converged,r)
  self.assertLessEqual(r.max_constraint_violation,1e-8)
  self.assertLess(r.stationarity_norm,1e-6)
  self.assertLess(r.complementarity_norm,1e-6)

 def test_surrogate_proposal_is_exactly_checked_and_converges_or_falls_back(self):
  r=solve_synthetic_qp(method="surrogate_newton",seed=2501)
  self.assertTrue(r.converged,r)
  self.assertLessEqual(r.max_constraint_violation,1e-8)
  self.assertGreaterEqual(r.oracle_evaluations,2)
  self.assertGreaterEqual(r.fallback_count,0)

 def test_comparison_uses_same_protocol_and_reports_both_methods(self):
  r=compare_exact_and_surrogate(seeds=(2501,2502),max_iterations=30)
  self.assertEqual(len(r["results"]),4)
  self.assertTrue(r["protocol"]["same_exact_objective"])
  self.assertTrue(r["protocol"]["same_constraints"])
  for row in r["results"]:
   self.assertTrue(torch.isfinite(torch.tensor(row["objective"])))
   self.assertGreaterEqual(row["oracle_evaluations"],1)
   self.assertTrue(row["converged"],row)
   self.assertLessEqual(row["max_constraint_violation"],1e-8)
 def test_closed_form_exact_fallback_is_feasible_and_kkt_stationary(self):
  from research.NeuralMoments.phase2_5_end_to_end_solver import exact_synthetic_qp_solution
  x=exact_synthetic_qp_solution()
  self.assertLessEqual(float(x.sum()-.2),1e-10)
  self.assertTrue(torch.allclose(x,torch.tensor([.4461538461538462,-.2461538461538462],dtype=x.dtype),atol=1e-10))


 def test_m69_s9_kkt_to_neural_newton_bridge(self):
  task={"schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],
   "contact":[0.,0.,0.],"density":1.,"shell_fraction":.03,"protect_radius_fraction":.04,
   "protect_height_fraction":0.,"polar_retention":.9,"min_removal_fraction":.1,
   "max_centroid_rise":1e-7,"com_tolerance":1e-4,"alignment_tolerance":1e-4,
   "principal_gap_min":-1e-8,"require_connected":True,"mass_tolerance":1e-10,
   "inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,"solver_tolerance":1e-9,
   "solver_polar_margin":1e-5,"training_polar_margin":5e-4,
   "objective":"minimum_retained_mass","material_policy":"single_density_removal_only",
   "inertia_reference":"original_fixed_axis"}
  original=torch.tensor([10.,0.,0.,0.,2.,2.,1.,0.,0.,0.],dtype=torch.float64)
  candidate=torch.tensor([9.,.18,.09,.36,1.85,1.55,.92,.015,.02,.01],dtype=torch.float64)
  beta=torch.tensor([.2,-.3,.1,.5,.2,.4,-.1,.3,.2,.1],dtype=torch.float64)
  beta=beta/torch.linalg.vector_norm(beta)
  response=torch.tensor([.01,.02,-.01,.015,.01,.012,.008,.002,-.003,.001],dtype=torch.float64)
  def moment_fn(b): return candidate+response*(b-beta)
  active=torch.zeros(10,dtype=torch.bool);active[2]=True
  chart,system,direction=propose_m69_neural_newton_step(
   torch.zeros(9,dtype=torch.float64),beta,moment_fn,original,task,
   length_scale=2.,active_mask=active,trust_radius=.1,event_tolerance=1e-12)
  self.assertEqual(tuple(chart.basis.shape),(10,9))
  self.assertEqual(direction.step.numel(),10)
  self.assertTrue(torch.isfinite(direction.step).all())
  self.assertLessEqual(direction.primal_step_norm,.1000001)
\nif __name__=="__main__": unittest.main()
