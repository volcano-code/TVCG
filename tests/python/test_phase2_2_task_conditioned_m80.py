import unittest
import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig,PcraftNativeMomentOperator
)
from research.NeuralMoments.m69_task_conditioned_m80_v1 import (
    M69TaskConditionedM80,MaterialMapContract,TASK_FEATURE_DIM,encode_m69_task_features
)

TASK={
 "schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],
 "contact":[0.,0.,0.],"density":1.,"native_resolution":512,
 "shell_fraction":.03,"protect_radius_fraction":.04,"protect_height_fraction":0.,
 "polar_retention":.9,"min_removal_fraction":.1,"max_centroid_rise":1e-7,
 "com_tolerance":1e-4,"alignment_tolerance":1e-4,"principal_gap_min":-1e-8,
 "require_connected":True,"mass_tolerance":1e-10,"inertia_tolerance":1e-10,
 "moment_psd_tolerance":1e-9,"solver_tolerance":1e-9,"solver_polar_margin":1e-5,
 "training_polar_margin":5e-4,"objective":"minimum_retained_mass",
 "inertia_reference":"original_fixed_axis","material_policy":"single_density_removal_only",
 "purpose":"research"
}

def synthetic_geometry():
    xyz=np.array([
      [-.55,-.35,-.25],[-.25,.35,-.15],[.15,-.45,.25],[.55,.25,.35],
      [-.45,.45,.45],[.35,.5,-.4],[.5,-.2,-.45],[-.1,-.1,.55]
    ],dtype=np.float64)
    extent=np.full((len(xyz),3),.1)
    mass=np.full(len(xyz),.01)
    x,y,z=xyz.T
    basis=np.stack([np.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y],axis=-1)
    fixed=np.zeros(10)
    design=mass@basis
    grid=np.zeros((4,8,8,8),dtype=np.float32)
    grid[0]=1.;grid[1]=1.
    return {"xyz":xyz,"mass":mass,"cell_extent":extent,"moment_basis":basis,
      "fixed_moments":fixed,"original_moments":fixed+design,
      "native_resolution":8,"origin":np.array([-1.,-1.,-1.]),"h":.25,
      "pcraft_grid":grid,"geometry_sha256":"phase2.2-synthetic-geometry"}

def build_model(freeze=True):
    config=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
      interval_width=8,interval_depth=1,chunk_intervals=8,min_native_resolution=8,
      convex_potential_enabled=True,convex_potential_components=2)
    return M69TaskConditionedM80(PcraftNativeMomentOperator(config),freeze_backbone=freeze)

class Phase22TaskConditionedOperatorTests(unittest.TestCase):
 def setUp(self):
    torch.manual_seed(22)
    self.model=build_model()
    self.cache=self.model.encode_geometry(synthetic_geometry())
    self.features=torch.tensor(encode_m69_task_features(TASK,length_scale=2.0))
    self.q=torch.tensor([.2,-1.,.05,.03,.1,.08,.04,.01,-.02,.03],dtype=torch.float64)

 def test_task_feature_contract_is_31d_and_finite(self):
    self.assertEqual(TASK_FEATURE_DIM,31)
    self.assertEqual(self.features.shape,(31,))
    self.assertTrue(torch.isfinite(self.features).all())

 def test_zero_initialized_film_is_exact_m80_identity(self):
    self.model.eval()
    with torch.no_grad():
      base=self.model.base_model.predict(self.cache,self.q)
      conditioned=self.model.predict(self.cache,self.q,self.features)
    self.assertTrue(torch.allclose(base,conditioned,rtol=0,atol=1e-12))

 def test_task_conditioning_changes_prediction_after_training_update(self):
    self.model.eval()
    with torch.no_grad():
      last=self.model.task_encoder[-1]
      last.weight.normal_(mean=0.,std=.05)
      last.bias.normal_(mean=0.,std=.05)
      f1=self.features.clone()
      f2=self.features.clone();f2[12]+=0.2
      a=self.model.predict(self.cache,self.q,f1)
      b=self.model.predict(self.cache,self.q,f2)
    self.assertGreater(float(torch.max(torch.abs(a-b))),1e-12)

 def test_query_jacobian_matches_finite_difference(self):
    self.model.eval()
    m,j=self.model.moments_and_query_jacobian(self.cache,self.q,self.features)
    eps=2e-5
    fd=torch.empty((10,10),dtype=torch.float64)
    for k in range(10):
      d=torch.zeros(10,dtype=torch.float64);d[k]=eps
      plus=self.model.predict(self.cache,self.q+d,self.features)
      minus=self.model.predict(self.cache,self.q-d,self.features)
      fd[:,k]=(plus-minus)/(2*eps)
    self.assertTrue(torch.allclose(j,fd,rtol=3e-3,atol=3e-5),
      msg=f"max abs Jacobian error={float((j-fd).abs().max())}")

 def test_material_jacobian_requires_explicit_identity_contract(self):
    identity=MaterialMapContract(True,True,True,True,True,True)
    self.model.eval()
    m,j=self.model.moments_and_material_jacobian(
      self.cache,self.q,self.features,torch.eye(10,dtype=torch.float64),identity)
    self.assertEqual(tuple(j.shape),(10,10))
    bad=MaterialMapContract(True,False,True,True,True,True)
    with self.assertRaisesRegex(ValueError,"not admitted"):
      self.model.moments_and_material_jacobian(
       self.cache,self.q,self.features,torch.eye(10,dtype=torch.float64),bad)

 def test_nine_dimensional_material_map_uses_chain_rule(self):
    contract=MaterialMapContract(True,True,True,True,True,False)
    D=torch.randn(10,9,dtype=torch.float64)
    self.model.eval()
    _,jq=self.model.moments_and_query_jacobian(self.cache,self.q,self.features)
    _,ja=self.model.moments_and_material_jacobian(self.cache,self.q,self.features,D,contract)
    self.assertTrue(torch.allclose(ja,jq@D,rtol=1e-10,atol=1e-10))

 def test_frozen_backbone_keeps_only_task_encoder_trainable(self):
    trainable=[name for name,p in self.model.named_parameters() if p.requires_grad]
    self.assertTrue(trainable)
    self.assertTrue(all(name.startswith("task_encoder.") for name in trainable))
    self.assertFalse(self.model.base_model.training)

 def test_invalid_task_frame_is_rejected(self):
    bad=dict(TASK);bad["contact"]=[.1,0.,0.]
    with self.assertRaisesRegex(ValueError,"lie on the fixed axis"):
      encode_m69_task_features(bad,length_scale=2.)

if __name__=="__main__": unittest.main()
