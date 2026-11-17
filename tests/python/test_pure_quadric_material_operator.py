import unittest
import numpy as np
from research.NeuralMoments.pure_quadric_material_operator_v1 import (
    QUERY_ORDER, central_difference_jacobian, classify_repeated_root_or_plane_event,
    material_jacobian_from_query_jacobian, numerical_quadric_box_moments,
    plane_cut_box_moments_and_jacobian, sphere_newton_step, sphere_retract,
    tangent_project,
)

BOUNDS=np.array([[-1.,1.],[-.8,.9],[-.6,.7]])
Q=np.array([.23,-1.,0,0,0,0,0,0,0,0.])
RHO=1.7

class PureQuadricMaterialOperatorTests(unittest.TestCase):
    def test_exact_plane_cut_moments_match_independent_quadrature(self):
        exact=plane_cut_box_moments_and_jacobian(Q,BOUNDS,RHO)
        numeric=numerical_quadric_box_moments(Q,BOUNDS,RHO,order=24)
        self.assertTrue(exact.derivative_valid)
        self.assertTrue(np.allclose(exact.moments,numeric,rtol=1e-12,atol=1e-12))

    def test_all_ten_shape_jacobian_columns_match_central_difference(self):
        exact=plane_cut_box_moments_and_jacobian(Q,BOUNDS,RHO)
        fd=central_difference_jacobian(Q,BOUNDS,RHO,step=1e-5,order=120)
        self.assertTrue(np.allclose(exact.jacobian,fd,rtol=2e-5,atol=2e-7),
                        msg=f"max abs error={np.max(np.abs(exact.jacobian-fd))}")

    def test_query_and_material_jacobian_equal_only_for_identity_contract(self):
        J=plane_cut_box_moments_and_jacobian(Q,BOUNDS,RHO).jacobian
        identity=np.eye(10)
        mapped=material_jacobian_from_query_jacobian(
            J,identity,same_frame=True,same_retained_side=True,same_masks=True,
            same_units=True,derivative_valid=True)
        self.assertTrue(np.allclose(mapped,J))
        with self.assertRaisesRegex(ValueError,"contract mismatch"):
            material_jacobian_from_query_jacobian(
                J,identity,same_frame=True,same_retained_side=False,same_masks=True,
                same_units=True,derivative_valid=True)

    def test_positive_quadric_scale_invariance_and_jacobian_homogeneity(self):
        a=plane_cut_box_moments_and_jacobian(Q,BOUNDS,RHO)
        k=3.7
        b=plane_cut_box_moments_and_jacobian(k*Q,BOUNDS,RHO)
        self.assertTrue(np.allclose(a.moments,b.moments,rtol=1e-12,atol=1e-12))
        self.assertTrue(np.allclose(b.jacobian,a.jacobian/k,rtol=1e-12,atol=1e-12))
        self.assertTrue(np.allclose(a.jacobian@Q,np.zeros(10),atol=1e-12))
        self.assertFalse(np.allclose(a.moments,numerical_quadric_box_moments(-Q,BOUNDS,RHO)))

    def test_plane_is_regular_even_though_quadric_matrix_rank_is_zero(self):
        result=classify_repeated_root_or_plane_event(Q,BOUNDS)
        self.assertTrue(result["valid"])
        self.assertEqual(result["quadratic_rank"],0)

    def test_boundary_coincidence_is_invalid_not_zero_jacobian(self):
        q=np.array([1.,-1.,0,0,0,0,0,0,0,0.])
        result=plane_cut_box_moments_and_jacobian(q,BOUNDS,RHO)
        self.assertFalse(result.derivative_valid)
        self.assertIsNone(result.jacobian)
        self.assertFalse(classify_repeated_root_or_plane_event(q,BOUNDS)["valid"])

    def test_repeated_root_singularity_is_invalid(self):
        # q=(x-0.2)^2, encoded as c=.04, bx=-.4, xx=1.
        q=np.array([.04,-.4,0,0,1.,0,0,0,0,0.])
        event=classify_repeated_root_or_plane_event(q,BOUNDS)
        self.assertFalse(event["valid"])
        self.assertEqual(event["event"],"singular_repeated_root")

    def test_tangent_projection_is_orthogonal_to_radial_direction(self):
        rng=np.random.default_rng(2101)
        q=rng.normal(size=10);g=rng.normal(size=10)
        p=tangent_project(q,g)
        self.assertAlmostEqual(float(np.dot(p,q)),0.,places=12)

    def test_sphere_retraction_preserves_unit_norm_and_sign_chart(self):
        q=np.array([.2,-.3,.1,.5,.2,.4,-.1,.3,.2,.1])
        q=q/np.linalg.norm(q)
        d=np.arange(10,dtype=float)*1e-3
        out=sphere_retract(q,d)
        self.assertAlmostEqual(float(np.linalg.norm(out)),1.,places=12)
        self.assertGreater(float(np.dot(out,q)),0.)

    def test_sphere_newton_step_is_tangent_and_retracted(self):
        q=np.zeros(10);q[0]=1.
        H=np.diag(np.arange(1.,11.))
        g=np.arange(10,dtype=float);g[0]=0.
        out,step,info=sphere_newton_step(q,g,H,damping=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(out)),1.,places=12)
        self.assertAlmostEqual(float(np.dot(step,q)),0.,places=10)
        self.assertLess(info["tangent_basis_orthogonality"],1e-12)

    def test_material_chain_rule_rejects_invalid_event(self):
        J=np.eye(10)
        with self.assertRaisesRegex(ValueError,"not admitted"):
            material_jacobian_from_query_jacobian(J,J,same_frame=True,same_retained_side=True,
                same_masks=True,same_units=True,derivative_valid=False)

if __name__=="__main__": unittest.main()
