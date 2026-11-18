import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.dataset import generate_splits
from research.NeuralMoments.models.pcraft_ufno.spin_pino.equivariance import (
    random_so3_rotation,
    transform_native_coefficients,
    transform_raw_moments,
    transform_native_jacobian,
    augment_with_random_so3,
)


def native_quadric_value(points, coefficients):
    x, y, z = points.unbind(-1)
    basis = torch.stack((torch.ones_like(x), x, y, z, x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y), dim=-1)
    return torch.einsum("bnd,bd->bn", basis, coefficients)


class SpinPINOSO3Tests(unittest.TestCase):
    def test_random_rotation_is_proper_orthogonal(self):
        R = random_so3_rotation(20261106)
        self.assertTrue(torch.allclose(R.T @ R, torch.eye(3, dtype=R.dtype), atol=1e-12))
        self.assertAlmostEqual(float(torch.det(R)), 1.0, places=12)

    def test_quadric_evaluation_is_rotation_covariant(self):
        R = random_so3_rotation(17, dtype=torch.float64)
        x = torch.tensor([[[0.2, -0.3, 0.1], [-0.4, 0.2, 0.3]]], dtype=torch.float64)
        c = torch.randn((1, 10), dtype=torch.float64)
        xp = torch.einsum("ij,bnj->bni", R, x)
        cp = transform_native_coefficients(c, R)
        self.assertTrue(torch.allclose(native_quadric_value(x, c), native_quadric_value(xp, cp), atol=1e-10, rtol=1e-10))

    def test_moment_and_jacobian_transform_shapes_and_finiteness(self):
        R = random_so3_rotation(29, dtype=torch.float64)
        moments = torch.randn((2, 10), dtype=torch.float64)
        J = torch.randn((2, 10, 10), dtype=torch.float64)
        rp = transform_raw_moments(moments, R)
        Jp = transform_native_jacobian(J, R)
        self.assertEqual(rp.shape, (2, 10))
        self.assertEqual(Jp.shape, (2, 10, 10))
        self.assertTrue(bool(torch.isfinite(rp).all() and torch.isfinite(Jp).all()))

    def test_so3_augmentation_is_reproducible_and_records_interpolation(self):
        data = generate_splits(
            seed=31, train_geometries_per_family=4,
            test_ellipsoid_geometries=2, queries_per_geometry=2, grid_size=8
        )["fit"]
        a = augment_with_random_so3(data, seed=900)
        b = augment_with_random_so3(data, seed=900)
        self.assertEqual(a[0].shape[0], 2 * data[0].shape[0])
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertEqual(a[4][-1]["augmentation"], "HAAR_SO3_LABEL_EXACT_VOXEL_TRILINEAR")


if __name__ == "__main__":
    unittest.main()


def _soft_quadric_moments(points, weights, coefficients, temperature=0.17):
    """Differentiable finite-quadrature oracle used only for transform identities."""
    x, y, z = points.unbind(-1)
    query_basis = torch.stack((
        torch.ones_like(x), x, y, z, x*x, y*y, z*z, 2*y*z, 2*x*z, 2*x*y
    ), dim=-1)
    moment_basis = torch.stack((
        torch.ones_like(x), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y
    ), dim=-1)
    occupancy = torch.sigmoid(-(query_basis @ coefficients) / temperature)
    return torch.einsum("n,nd->d", weights * occupancy, moment_basis)


class SpinPINOPhysicalSO3ContractTests(unittest.TestCase):
    def _quadrature(self):
        axis = torch.linspace(-0.9, 0.9, 9, dtype=torch.float64)
        points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        weights = (0.2 + torch.exp(-points.square().sum(-1))) / len(points)
        return points, weights

    def test_rotated_quadrature_moments_and_jacobian_chain_rule(self):
        torch.manual_seed(211)
        points, weights = self._quadrature()
        R = random_so3_rotation(20261118, dtype=torch.float64)
        coefficients = torch.tensor(
            [0.11, -0.07, 0.04, 0.09, 0.8, 0.6, 0.9, 0.13, -0.08, 0.06],
            dtype=torch.float64,
        )
        rotated_points = torch.einsum("ij,nj->ni", R, points)
        rotated_coefficients = transform_native_coefficients(coefficients[None], R)[0]

        moment_fn = lambda c: _soft_quadric_moments(points, weights, c)
        rotated_fn = lambda c: _soft_quadric_moments(rotated_points, weights, c)
        moments = moment_fn(coefficients)
        rotated_moments = rotated_fn(rotated_coefficients)
        T = moment_rotation_matrix(R, dtype=torch.float64)
        S = coefficient_rotation_matrix(R, dtype=torch.float64)
        self.assertTrue(torch.allclose(rotated_moments, T @ moments, atol=2e-12, rtol=2e-12))

        J = torch.autograd.functional.jacobian(moment_fn, coefficients)
        J_rotated_direct = torch.autograd.functional.jacobian(rotated_fn, rotated_coefficients)
        J_rotated_chain = transform_native_jacobian(J[None], R)[0]
        self.assertTrue(torch.allclose(J_rotated_direct, T @ J @ torch.linalg.inv(S),
                                        atol=3e-11, rtol=3e-11))
        self.assertTrue(torch.allclose(J_rotated_direct, J_rotated_chain,
                                        atol=3e-11, rtol=3e-11))

        # Independent central finite differences on the rotated query map.
        eps = 2e-6
        J_fd = torch.empty_like(J_rotated_direct)
        with torch.no_grad():
            for k in range(10):
                delta = torch.zeros_like(rotated_coefficients)
                delta[k] = eps
                J_fd[:, k] = (
                    rotated_fn(rotated_coefficients + delta)
                    - rotated_fn(rotated_coefficients - delta)
                ) / (2 * eps)
        self.assertTrue(torch.allclose(J_rotated_direct, J_fd, atol=2e-7, rtol=2e-6))

    def test_raw_moments_fixed_editable_rotation_and_recomposition(self):
        points, weights = self._quadrature()
        R = random_so3_rotation(311, dtype=torch.float64)
        coefficients = torch.tensor(
            [0.02, 0.04, -0.03, 0.01, 0.7, 0.9, 0.8, 0.1, 0.03, -0.06],
            dtype=torch.float64,
        )
        query_basis = torch.stack((
            torch.ones_like(points[:, 0]), points[:, 0], points[:, 1], points[:, 2],
            points[:, 0]**2, points[:, 1]**2, points[:, 2]**2,
            2*points[:, 1]*points[:, 2], 2*points[:, 0]*points[:, 2],
            2*points[:, 0]*points[:, 1],
        ), dim=-1)
        occupancy = torch.sigmoid(-(query_basis @ coefficients) / 0.2)
        moment_basis = torch.stack((
            torch.ones_like(points[:, 0]), points[:, 0], points[:, 1], points[:, 2],
            points[:, 0]**2, points[:, 1]**2, points[:, 2]**2,
            points[:, 1]*points[:, 2], points[:, 0]*points[:, 2],
            points[:, 0]*points[:, 1],
        ), dim=-1)
        fixed_mask = points[:, 0] < 0
        editable_mask = ~fixed_mask
        fixed = torch.einsum("n,nd->d", weights * occupancy * fixed_mask, moment_basis)
        editable = torch.einsum("n,nd->d", weights * occupancy * editable_mask, moment_basis)
        T = moment_rotation_matrix(R, dtype=torch.float64)
        fixed_rotated = transform_raw_moments(fixed[None], R)[0]
        editable_rotated = transform_raw_moments(editable[None], R)[0]
        total_rotated = transform_raw_moments((fixed + editable)[None], R)[0]
        self.assertTrue(torch.allclose(total_rotated, fixed_rotated + editable_rotated,
                                        atol=2e-12, rtol=2e-12))
        self.assertTrue(torch.allclose(total_rotated, T @ (fixed + editable),
                                        atol=2e-12, rtol=2e-12))

    def test_so3_transform_identity_inverse_and_composition(self):
        from research.NeuralMoments.models.pcraft_ufno.spin_pino.equivariance import (
            transform_raw_moments, transform_native_coefficients
        )
        R1 = random_so3_rotation(401, dtype=torch.float64)
        R2 = random_so3_rotation(402, dtype=torch.float64)
        moments = torch.randn(3, 10, dtype=torch.float64)
        coefficients = torch.randn(3, 10, dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64)
        self.assertTrue(torch.allclose(transform_raw_moments(moments, identity), moments,
                                        atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(transform_native_coefficients(coefficients, identity), coefficients,
                                        atol=1e-12, rtol=1e-12))
        recovered_m = transform_raw_moments(transform_raw_moments(moments, R1), R1.T)
        recovered_c = transform_native_coefficients(
            transform_native_coefficients(coefficients, R1), R1.T
        )
        self.assertTrue(torch.allclose(recovered_m, moments, atol=2e-12, rtol=2e-12))
        self.assertTrue(torch.allclose(recovered_c, coefficients, atol=2e-12, rtol=2e-12))
        composed = R2 @ R1
        m_sequential = transform_raw_moments(transform_raw_moments(moments, R1), R2)
        m_composed = transform_raw_moments(moments, composed)
        c_sequential = transform_native_coefficients(
            transform_native_coefficients(coefficients, R1), R2
        )
        c_composed = transform_native_coefficients(coefficients, composed)
        self.assertTrue(torch.allclose(m_sequential, m_composed, atol=3e-12, rtol=3e-12))
        self.assertTrue(torch.allclose(c_sequential, c_composed, atol=3e-12, rtol=3e-12))

    def test_rotation_transform_rejects_reflections_and_nonfinite_moments(self):
        from research.NeuralMoments.models.pcraft_ufno.spin_pino.equivariance import transform_raw_moments
        reflection = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64))
        with self.assertRaises(ValueError):
            transform_raw_moments(torch.zeros(1, 10, dtype=torch.float64), reflection)
        R = torch.eye(3, dtype=torch.float64)
        invalid = torch.zeros(1, 10, dtype=torch.float64)
        invalid[0, 3] = float("nan")
        # Public moment transforms reject invalid physical payloads.
        with self.assertRaises(ValueError):
            transform_raw_moments(invalid, R)
