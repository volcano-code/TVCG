import unittest
import torch
from research.NeuralMoments.phase2_4_neural_newton import (
 make_s9_chart,safeguarded_kkt_newton_direction,exact_merit_acceptance,
 safeguarded_newton_iteration,build_s9_m69_kkt_system
)

class Phase24NeuralNewtonTests(unittest.TestCase):
 def test_s9_chart_is_orthonormal_and_decodes_unit_quadric(self):
  beta=torch.tensor([.2,-.3,.1,.5,.2,.4,-.1,.3,.2,.1],dtype=torch.float64)
  chart=make_s9_chart(beta)
  self.assertEqual(tuple(chart.basis.shape),(10,9))
  self.assertLess(float(torch.linalg.norm(chart.basis.T@chart.basis-torch.eye(9))),1e-10)
  self.assertLess(float(torch.linalg.norm(chart.basis.T@chart.anchor)),1e-10)
  local=torch.randn(9,dtype=torch.float64)*.01
  out=chart.decode(local)
  self.assertAlmostEqual(float(torch.linalg.norm(out)),1.,places=12)
  self.assertGreater(float(out@chart.anchor),0.)
  self.assertTrue(torch.allclose(chart.encode_tangent(out),local,atol=2e-4,rtol=2e-4))

 def test_kkt_newton_solves_symmetric_saddle_point_without_blanket_damping(self):
  # min .5*x^2 subject to x=1: KKT [x+nu, x-1] = 0.
  F=torch.tensor([1.,0.],dtype=torch.float64)
  J=torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float64)
  d=safeguarded_kkt_newton_direction(F,J,primal_dimension=1,trust_radius=2.,
      initial_regularization=1e-10)
  self.assertTrue(torch.allclose(J@d.step,-F,atol=1e-8))
  self.assertLess(d.kkt_symmetry_error,1e-12)

 def test_primal_trust_radius_limits_step(self):
  F=torch.tensor([0.,-10.],dtype=torch.float64)
  J=torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float64)
  d=safeguarded_kkt_newton_direction(F,J,primal_dimension=1,trust_radius=.1,
      multiplier_radius=10.)
  self.assertTrue(d.trust_radius_limited)
  self.assertLessEqual(d.primal_step_norm,.1000001)

 def test_non_symmetric_kkt_jacobian_is_rejected(self):
  F=torch.ones(2,dtype=torch.float64)
  J=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64)
  with self.assertRaisesRegex(ValueError,"not symmetric"):
   safeguarded_kkt_newton_direction(F,J,primal_dimension=1,trust_radius=1.)

 def test_exact_merit_acceptance_backtracks_and_accepts(self):
  state={"x":0.}
  result=exact_merit_acceptance(state,torch.tensor([2.],dtype=torch.float64),
    decode=lambda s,d:{"x":s["x"]+float(d[0])},
    exact_merit=lambda s:(s["x"]-0.4)**2,
    predicted_reduction=1.)
  self.assertTrue(result.accepted)
  self.assertLess(result.actual_reduction,1.0)
  self.assertGreater(result.ratio,.05)

 def test_exact_merit_rejects_and_requests_fallback(self):
  state={"x":0.}
  result=exact_merit_acceptance(state,torch.tensor([-1.],dtype=torch.float64),
    decode=lambda s,d:{"x":s["x"]+float(d[0])},
    exact_merit=lambda s:s["x"]**2,
    predicted_reduction=1.)
  self.assertFalse(result.accepted)
  self.assertIn("fallback_required",result.reason)

 def test_newton_iteration_calls_exact_fallback_after_rejection(self):
  state={"x":0.}
  direction=safeguarded_kkt_newton_direction(
    torch.tensor([1.,0.],dtype=torch.float64),
    torch.tensor([[1.,1.],[1.,0.]],dtype=torch.float64),
    primal_dimension=1,trust_radius=1.)
  result=safeguarded_newton_iteration(state,direction,
    decode=lambda s,d:{"x":s["x"]+float(d[0])},
    exact_merit=lambda s:s["x"]**2,
    predicted_reduction=1.,
    exact_fallback=lambda s:{"x":.25})
  self.assertTrue(result.used_fallback)
  self.assertEqual(result.state,{"x":.25})


 def test_s9_chart_composes_with_m69_kkt_and_material_derivatives(self):
  from research.NeuralMoments.phase2_3_differentiable_m69_kkt import m69_constraint_slacks_torch
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
  # A smooth, deliberately synthetic material response. The test validates
  # chart/derivative wiring, not physical accuracy of this response.
  response=torch.tensor([.01,.02,-.01,.015,.01,.012,.008,.002,-.003,.001],dtype=torch.float64)
  def beta_to_moments(b): return candidate+response*(b-beta)
  probe_chart=make_s9_chart(beta)
  probe=beta_to_moments(probe_chart.decode(torch.zeros(9,dtype=torch.float64)))
  physical=m69_constraint_slacks_torch(probe,original,task,length_scale=2.,event_tolerance=1e-12)
  self.assertTrue(physical.derivative_valid,physical.invalid_reasons)
  active=torch.abs(physical.slack.detach())<1e-8
  chart,system=build_s9_m69_kkt_system(torch.zeros(9,dtype=torch.float64),beta,
    beta_to_moments,original,task,length_scale=2.,active_mask=active,event_tolerance=1e-12)
  self.assertEqual(tuple(chart.basis.shape),(10,9))
  self.assertEqual(system.unknowns.numel(),9+int(active.sum()))
  self.assertTrue(torch.isfinite(system.jacobian).all())
\nif __name__=="__main__": unittest.main()
