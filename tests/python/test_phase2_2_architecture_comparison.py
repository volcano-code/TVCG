import unittest
import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig,PcraftNativeMomentOperator
)
from research.NeuralMoments.m69_task_conditioned_m80_v1 import (
    M69TaskConditionedM80,TASK_FEATURE_DIM,encode_m69_task_features,
    build_parameter_matched_variants
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
    xyz=np.array([[-.55,-.35,-.25],[-.25,.35,-.15],[.15,-.45,.25],[.55,.25,.35],
      [-.45,.45,.45],[.35,.5,-.4],[.5,-.2,-.45],[-.1,-.1,.55]],dtype=np.float64)
    extent=np.full((len(xyz),3),.1);mass=np.full(len(xyz),.01);x,y,z=xyz.T
    basis=np.stack([np.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y],axis=-1)
    fixed=np.zeros(10);design=mass@basis;grid=np.zeros((4,8,8,8),dtype=np.float32)
    grid[0]=1.;grid[1]=1.
    return {"xyz":xyz,"mass":mass,"cell_extent":extent,"moment_basis":basis,
      "fixed_moments":fixed,"original_moments":fixed+design,"native_resolution":8,
      "origin":np.array([-1.,-1.,-1.]),"h":.25,"pcraft_grid":grid,
      "geometry_sha256":"phase2.2-synthetic-geometry"}

def base_model():
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=8,min_native_resolution=8,
        convex_potential_enabled=True,convex_potential_components=2)
    return PcraftNativeMomentOperator(cfg)


class Phase22ArchitectureComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2202)
        self.base=base_model()
        self.geometry=synthetic_geometry()
        self.cache=self.base.encode(self.geometry)
        self.features=torch.tensor(encode_m69_task_features(TASK,length_scale=2.0))
        self.q=torch.tensor([.2,-1.,.05,.03,.1,.08,.04,.01,-.02,.03],dtype=torch.float64)
        self.variants,self.budget=build_parameter_matched_variants(
            self.base,tolerance=.20,freeze_backbone=True
        )

    def _activate_conditioner(self,model):
        with torch.no_grad():
            if model.conditioning_mode in ("film","conditional_norm"):
                model.task_encoder[-1].weight.normal_(0.,.04)
                model.task_encoder[-1].bias.normal_(0.,.04)
            else:
                model.cross_output.weight.normal_(0.,.04)
                model.cross_output.bias.normal_(0.,.04)

    def test_all_three_variants_are_exact_identity_at_initialization(self):
        for name,model in self.variants.items():
            model.eval()
            with torch.no_grad():
                expected=model.base_model.predict(model.base_model.encode(self.geometry),self.q)
                actual=model.predict(model.encode_geometry(self.geometry),self.q,self.features)
            self.assertTrue(torch.allclose(expected,actual,rtol=0.,atol=1e-12),name)

    def test_all_three_variants_respond_to_task_after_controlled_update(self):
        f2=self.features.clone()
        f2[12]+=0.25
        for name,model in self.variants.items():
            self._activate_conditioner(model)
            model.eval()
            with torch.no_grad():
                a=model.predict(model.encode_geometry(self.geometry),self.q,self.features)
                b=model.predict(model.encode_geometry(self.geometry),self.q,f2)
            self.assertGreater(float((a-b).abs().max()),1e-12,name)

    def test_all_variants_keep_query_jacobian_consistent_with_fd(self):
        eps=2e-5
        for name,model in self.variants.items():
            model.eval()
            cache=model.encode_geometry(self.geometry)
            _,jac=model.moments_and_query_jacobian(cache,self.q,self.features)
            fd=torch.empty((10,10),dtype=torch.float64)
            for k in range(10):
                d=torch.zeros(10,dtype=torch.float64);d[k]=eps
                plus=model.predict(cache,self.q+d,self.features)
                minus=model.predict(cache,self.q-d,self.features)
                fd[:,k]=(plus-minus)/(2*eps)
            self.assertTrue(torch.allclose(jac,fd,rtol=3e-3,atol=3e-5),
                msg=f"{name}: max_abs={float((jac-fd).abs().max())}")

    def test_convex_potential_jacobians_are_symmetric_psd(self):
        for name,model in self.variants.items():
            model.eval()
            cache=model.encode_geometry(self.geometry)
            _,jac=model.moments_and_query_jacobian(cache,self.q,self.features)
            sym=.5*(jac+jac.T)
            self.assertLess(float((jac-jac.T).abs().max()),2e-8,name)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(sym).min()),-2e-8,name)

    def test_matched_parameter_budget_is_reported_and_bounded(self):
        report=self.budget["variants"]
        counts=[report[name]["trainable_parameters"] for name in
                ("film","conditional_norm","cross_attention")]
        self.assertEqual(len(counts),3)
        self.assertLessEqual(max(counts)-min(counts),.20*self.budget["target_trainable_parameters"])
        self.assertTrue(all(report[name]["relative_gap_to_target"]<=.20 for name in report))

    def test_gradients_reach_conditioning_output_layers(self):
        for name,model in self.variants.items():
            model.train()
            cache=model.encode_geometry(self.geometry)
            pred=model.predict(cache,self.q,self.features)
            loss=pred.square().mean()
            loss.backward()
            if name=="cross_attention":
                grad=model.cross_output.weight.grad
            else:
                grad=model.task_encoder[-1].weight.grad
            self.assertIsNotNone(grad,name)
            self.assertGreater(float(grad.norm()),0.,name)

    def test_backbones_are_frozen_and_eval_in_all_variants(self):
        for name,model in self.variants.items():
            self.assertFalse(model.base_model.training,name)
            self.assertTrue(all(not p.requires_grad for p in model.base_model.parameters()),name)
            self.assertTrue(any(p.requires_grad for p in model.task_encoder.parameters()),name)


if __name__=="__main__":
    unittest.main()
