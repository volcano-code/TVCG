import unittest
import numpy as np
import torch
from research.NeuralMoments.phase2_3_differentiable_m69_kkt import (
    fixed_active_set_kkt,m69_constraint_slacks_torch,m69_kkt_from_moment_operator
)

TASK={"schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],
"contact":[0.,0.,0.],"density":1.,"shell_fraction":.03,"protect_radius_fraction":.04,
"protect_height_fraction":0.,"polar_retention":.9,"min_removal_fraction":.1,
"max_centroid_rise":1e-7,"com_tolerance":1e-4,"alignment_tolerance":1e-4,
"principal_gap_min":-1e-8,"require_connected":True,"mass_tolerance":1e-10,
"inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,"solver_tolerance":1e-9,
"solver_polar_margin":1e-5,"training_polar_margin":5e-4,
"objective":"minimum_retained_mass","material_policy":"single_density_removal_only",
"inertia_reference":"original_fixed_axis"}
ORIGINAL=torch.tensor([10.,0.,0.,0.,2.,2.,1.,0.,0.,0.],dtype=torch.float64)
# Offset centroid and anisotropic transverse covariance avoid zero-norm and
# repeated-eigenvalue derivative events.
CANDIDATE=torch.tensor([9.,.18,.09,.36,1.85,1.55,.92,.015,.02,.01],dtype=torch.float64)

class Phase23DifferentiableKKTTests(unittest.TestCase):
 def test_ten_constraint_values_and_g_le_zero_sign(self):
  r=m69_constraint_slacks_torch(CANDIDATE,ORIGINAL,TASK,length_scale=2.)
  self.assertEqual(r.slack.shape,(10,))
  self.assertEqual(r.g_le_zero.shape,(10,))
  self.assertTrue(torch.allclose(r.g_le_zero,-r.slack))
  self.assertEqual(len(r.constraint_names),10)

 def test_smooth_branch_constraint_jacobian_matches_finite_difference(self):
  x=CANDIDATE.clone().requires_grad_(True)
  def evaluate(v): return m69_constraint_slacks_torch(v,ORIGINAL,TASK,length_scale=2.)
  r=evaluate(x)
  self.assertTrue(r.derivative_valid,r.invalid_reasons)
  jac=torch.autograd.functional.jacobian(lambda v:evaluate(v).slack,x)
  eps=1e-5; fd=torch.empty_like(jac)
  for j in range(10):
   d=torch.zeros(10,dtype=torch.float64);d[j]=eps
   fd[:,j]=(fn(CANDIDATE+d).detach()-fn(CANDIDATE-d).detach())/(2*eps)
  self.assertTrue(torch.allclose(jac,fd,rtol=2e-4,atol=2e-6),
    msg=f"max_error={(jac-fd).abs().max().item()}")

 def test_active_constraint_is_differentiable_on_fixed_active_set(self):
  r=m69_constraint_slacks_torch(CANDIDATE,ORIGINAL,TASK,length_scale=2.)
  self.assertTrue(r.derivative_valid,r.invalid_reasons)
  self.assertTrue(r.active_set_event)
 def test_zero_com_alignment_is_reported_derivative_invalid(self):
  centered=CANDIDATE.clone();centered[1:4]=0.
  r=m69_constraint_slacks_torch(centered,ORIGINAL,TASK,length_scale=2.)
  self.assertFalse(r.derivative_valid)
  self.assertIn("zero_norm_com_alignment",r.invalid_reasons)

 def test_repeated_transverse_eigenvalue_is_invalid(self):
  iso=ORIGINAL.clone()
  iso[4]=iso[5]=2.;iso[7]=iso[8]=iso[9]=0.
  r=m69_constraint_slacks_torch(iso,ORIGINAL,TASK,length_scale=2.)
  self.assertFalse(r.derivative_valid)
  self.assertTrue(any("eigenvalue_multiplicity" in x for x in r.invalid_reasons))

 def test_fixed_active_set_kkt_sign_and_jacobian(self):
  # min (x-2)^2, g=x-1<=0; at x=1, lambda=2 => stationarity 0.
  x=torch.tensor([1.],dtype=torch.float64)
  result=fixed_active_set_kkt(x,lambda v:(v[0]-2.)**2,
    lambda v:torch.empty(0,dtype=v.dtype),lambda v:torch.stack((v[0]-1.,)),
    torch.tensor([True]),active_inequality_multipliers=torch.tensor([2.],dtype=x.dtype))
  self.assertTrue(torch.allclose(result.residual,torch.zeros(2,dtype=x.dtype),atol=1e-12))
  self.assertEqual(tuple(result.jacobian.shape),(2,2))
  self.assertTrue(torch.isfinite(result.jacobian).all())

 def test_kkt_jacobian_matches_finite_difference_of_residual(self):
  x=torch.tensor([.8],dtype=torch.float64)
  fn=lambda v:(v[0]-2.)**2
  g=lambda v:torch.stack((v[0]-1.,))
  result=fixed_active_set_kkt(x,fn,lambda v:torch.empty(0,dtype=v.dtype),g,
    torch.tensor([True]),active_inequality_multipliers=torch.tensor([1.]))
  z=result.unknowns.detach()
  def residual(zz):
   xx=zz[:1]
   lam=zz[1:]
   return torch.stack((2.0*(xx[0]-2.0)+lam[0],xx[0]-1.0))
  eps=1e-6
  fd=torch.empty_like(result.jacobian)
  for j in range(2):
   d=torch.zeros(2,dtype=z.dtype);d[j]=eps
   fd[:,j]=(residual(z+d)-residual(z-d))/(2*eps)
  self.assertTrue(torch.allclose(result.jacobian,fd,rtol=1e-6,atol=1e-6))

 def test_inactive_constraint_switch_boundary_is_rejected(self):
  x=torch.tensor([0.],dtype=torch.float64)
  with self.assertRaisesRegex(ValueError,"switching boundary"):
   fixed_active_set_kkt(x,lambda v:v[0]**2,lambda v:torch.empty(0,dtype=v.dtype),
    lambda v:torch.stack((v[0],-v[0])),torch.tensor([True,False]))

 def test_m69_kkt_wrapper_refuses_invalid_derivative_branch(self):
  x=torch.tensor([0.],dtype=torch.float64)
  def moment_fn(v):
   return CANDIDATE.clone()+v[0]*torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],dtype=v.dtype)
  with self.assertRaisesRegex(ValueError,"M69 derivative invalid"):
   m69_kkt_from_moment_operator(x,moment_fn,ORIGINAL,TASK,length_scale=2.,
     active_mask=torch.zeros(10,dtype=torch.bool))

if __name__=="__main__": unittest.main()
