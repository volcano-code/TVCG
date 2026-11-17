import unittest
import torch
from research.NeuralMoments.phase2_5_end_to_end_solver import (
 ActiveSetState,update_active_set,kkt_direction,solve_synthetic_qp,compare_exact_and_surrogate
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

if __name__=="__main__": unittest.main()
