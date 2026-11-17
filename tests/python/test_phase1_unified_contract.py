import unittest
import numpy as np

from research.NeuralMoments.optimization_contract_v1 import (
    MOMENT_ORDER, add_disjoint_material_moments, active_set_kkt_residual,
    integrate_weighted_points, moment_transform_matrix, plane_cut_box_moments,
    raw_monomials, target_frame_transform, transform_quadric_coefficients,
    transform_raw_moments, validate_contract_record, CONTRACT_VERSION,
)


class Phase1MomentContractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261117)
        self.points = rng.normal(size=(200, 3))
        self.weights = rng.uniform(0.01, 0.1, size=200)
        self.moments = integrate_weighted_points(self.points, self.weights)

    def test_moment_order_and_basis(self):
        self.assertEqual(MOMENT_ORDER, ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"))
        self.assertTrue(np.allclose(raw_monomials([[2, 3, 5]])[0],
                                    [1, 2, 3, 5, 4, 9, 25, 15, 10, 6]))

    def test_affine_moment_transform_matches_transformed_points(self):
        angle = 0.37
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1.0]])
        t = np.array([0.3, -0.2, 0.7])
        expected = integrate_weighted_points((R @ self.points.T).T + t, self.weights)
        actual = transform_raw_moments(self.moments, R, t)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_target_frame_transform_matches_spin_target_convention(self):
        angle = -0.61
        axes = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1.0]])
        origin = np.array([0.4, -0.1, 0.2])
        R, t = target_frame_transform(origin, axes)
        expected = integrate_weighted_points((R @ (self.points-origin).T).T, self.weights)
        actual = transform_raw_moments(self.moments, R, t)
        self.assertTrue(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))

    def test_quadric_coefficients_are_dual_to_moment_transform(self):
        A = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([0.2, 0.4, -0.3])
        c = np.array([0.7, 0.1, -0.3, 0.2, 0.5, 0.6, 0.9, 0.15, -0.12, 0.08])
        T = moment_transform_matrix(A, t)
        c_new = transform_quadric_coefficients(c, A, t)
        x_new = (A @ self.points.T).T + t
        m_new = integrate_weighted_points(x_new, self.weights)
        self.assertAlmostEqual(float(c @ self.moments), float(c_new @ m_new), places=10)
        # Also verify pointwise polynomial identity.
        beta_old = raw_monomials(self.points) @ c
        beta_new = raw_monomials(x_new) @ c_new
        self.assertTrue(np.allclose(beta_old, beta_new, rtol=1e-11, atol=1e-11))
        self.assertEqual(T.shape, (10, 10))

    def test_fixed_protected_editable_moments_added_once(self):
        fixed = np.ones(10)
        protected = np.full(10, 2.0)
        editable = np.full(10, 3.0)
        total = add_disjoint_material_moments(fixed, protected, editable)
        self.assertTrue(np.array_equal(total, np.full(10, 6.0)))

    def test_exact_plane_cut_box_moments_and_derivative(self):
        bounds = ((-1, 1), (-2, 2), (-0.5, 0.5))
        a = 0.23
        rho = 1.7
        result = plane_cut_box_moments(a, bounds, rho)
        eps = 1e-6
        plus = plane_cut_box_moments(a+eps, bounds, rho).moments
        minus = plane_cut_box_moments(a-eps, bounds, rho).moments
        fd = (plus-minus)/(2*eps)
        self.assertTrue(result.derivative_valid)
        self.assertTrue(np.allclose(result.derivative_da, fd, rtol=2e-8, atol=2e-8))

    def test_plane_cut_derivative_fails_closed_at_events(self):
        left = plane_cut_box_moments(-1.0, ((-1, 1), (-1, 1), (-1, 1)))
        right = plane_cut_box_moments(1.0, ((-1, 1), (-1, 1), (-1, 1)))
        self.assertFalse(left.derivative_valid)
        self.assertFalse(right.derivative_valid)
        self.assertTrue(np.array_equal(left.derivative_da, np.zeros(10)))
        self.assertTrue(np.array_equal(right.derivative_da, np.zeros(10)))

    def test_active_set_kkt_sign_and_dimensions(self):
        # min (x-2)^2 subject to x-1 <= 0: x=1, lambda=2.
        result = active_set_kkt_residual(
            objective_gradient=np.array([-2.0]),
            equality_values=np.array([]), equality_jacobian=np.zeros((0, 1)),
            inequality_values=np.array([0.0]), inequality_jacobian=np.array([[1.0]]),
            active_mask=np.array([True]), equality_multipliers=np.array([]),
            active_inequality_multipliers=np.array([2.0]),
        )
        self.assertEqual(result.residual.shape, (2,))
        self.assertTrue(np.allclose(result.residual, 0.0))
        self.assertEqual(result.inactive_primal_violation, 0.0)
        self.assertEqual(result.active_dual_violation, 0.0)

    def test_inactive_primal_and_dual_violations_are_reported(self):
        result = active_set_kkt_residual(
            objective_gradient=np.array([1.0]),
            equality_values=np.array([]), equality_jacobian=np.zeros((0, 1)),
            inequality_values=np.array([0.2, -0.1]), inequality_jacobian=np.array([[1.0], [1.0]]),
            active_mask=np.array([False, True]), equality_multipliers=np.array([]),
            active_inequality_multipliers=np.array([-0.5]),
        )
        self.assertAlmostEqual(result.inactive_primal_violation, 0.2)
        self.assertAlmostEqual(result.active_dual_violation, 0.5)

    def test_quadric_stationarity_is_projected_to_sphere_tangent(self):
        beta = np.zeros(10)
        beta[0] = 1.0
        grad = np.arange(1.0, 11.0)
        result = active_set_kkt_residual(
            objective_gradient=grad,
            equality_values=np.array([]), equality_jacobian=np.zeros((0, 10)),
            inequality_values=np.array([]), inequality_jacobian=np.zeros((0, 10)),
            active_mask=np.array([], dtype=bool), equality_multipliers=np.array([]),
            active_inequality_multipliers=np.array([]), beta=beta,
        )
        self.assertAlmostEqual(float(beta @ result.stationarity[:10]), 0.0)

    def test_contract_envelope_rejects_unselected_objective(self):
        record = {
            "schema_version": CONTRACT_VERSION,
            "geometry": {}, "task": {},
            "material": {"single_material": True, "density_kg_m3": 1000.0,
                         "regions": {"fixed": "fixed-v1", "protected": "protected-v1",
                                     "editable": "editable-v1"}},
            "moments": {"order": list(MOMENT_ORDER)},
            "objective": {"primary_profile_selected": False},
            "derivatives": {}, "kkt": {},
        }
        self.assertTrue(validate_contract_record(record))
        record["objective"]["primary_profile_selected"] = True
        with self.assertRaises(ValueError):
            validate_contract_record(record)


if __name__ == "__main__":
    unittest.main()
