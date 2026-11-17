"""RLE-native smooth convex ten-moment operator.

Raw moment order:
(mass, x, y, z, xx, yy, zz, yz, xz, xy)

Native coefficient convention:
(1, 1, 1, 1, 1, 1, 1, 2, 2, 2)

The q-coordinate Hessian is symmetric PSD.
The native coefficient Jacobian is J_c = J_q @ diag(NATIVE_B).
"""
from __future__ import annotations

import torch
from torch import Tensor

NATIVE_B = (1., 1., 1., 1., 1., 1., 1., 2., 2., 2.)


def raw_moment_basis(xyz: Tensor) -> Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")

    x, y, z = xyz.unbind(-1)

    return torch.stack((
        torch.ones_like(x),
        x, y, z,
        x*x, y*y, z*z,
        y*z, x*z, x*y,
    ), dim=-1)


class SpinMomentOperator:
    def __init__(
        self,
        temperature: float = 0.04,
        chunk_nodes: int = 65536,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if chunk_nodes < 1:
            raise ValueError("chunk_nodes must be positive")

        self.temperature = float(temperature)
        self.chunk_nodes = int(chunk_nodes)

    def _validate(
        self,
        xyz: Tensor,
        weights: Tensor,
        coefficients: Tensor,
    ):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be [N,3]")
        if weights.shape != (len(xyz),):
            raise ValueError("weights must be [N]")
        if coefficients.shape != (10,):
            raise ValueError("coefficients must be [10]")
        if not (xyz.device == weights.device == coefficients.device):
            raise ValueError("all inputs must share device")
        if not (xyz.dtype == weights.dtype == coefficients.dtype):
            raise ValueError("all inputs must share dtype")
        if not xyz.is_floating_point():
            raise TypeError("inputs must be floating point")
        if not all(torch.isfinite(v).all() for v in (xyz, weights, coefficients)):
            raise ValueError("inputs must be finite")
        if (weights < 0).any():
            raise ValueError("weights must be nonnegative")
        if torch.linalg.vector_norm(coefficients) <= 1e-12:
            raise ValueError("coefficients must be nonzero")

    def evaluate(
        self,
        xyz: Tensor,
        weights: Tensor,
        coefficients: Tensor,
        *,
        with_jacobian: bool = True,
    ) -> dict[str, Tensor]:
        self._validate(xyz, weights, coefficients)

        B = coefficients.new_tensor(NATIVE_B)

        c = coefficients / torch.linalg.vector_norm(coefficients)
        q = c * B

        eps2 = self.temperature ** 2
        norm2 = q.square().sum()

        potential = coefficients.new_zeros(())
        moments = coefficients.new_zeros(10)
        jacobian_q = coefficients.new_zeros((10, 10))

        for start in range(0, len(xyz), self.chunk_nodes):
            stop = min(start + self.chunk_nodes, len(xyz))

            basis = raw_moment_basis(xyz[start:stop])
            w = weights[start:stop]

            a = basis @ q
            s = torch.sqrt(a.square() + eps2 * norm2)

            n = a.unsqueeze(-1) * basis + eps2 * q

            grad_q = 0.5 * (
                basis + n / s.unsqueeze(-1)
            )

            potential = potential + 0.5 * (
                w * (a + s)
            ).sum()

            moments = moments + torch.einsum(
                "n,ni->i", w, grad_q
            )

            if with_jacobian:
                eye = torch.eye(
                    10, device=xyz.device, dtype=xyz.dtype
                )

                mm = basis.unsqueeze(-1) * basis.unsqueeze(-2)

                hess = 0.5 * (
                    (mm + eps2 * eye) / s[:, None, None]
                    - (n.unsqueeze(-1) * n.unsqueeze(-2))
                    / s[:, None, None] ** 3
                )

                jacobian_q = jacobian_q + torch.einsum(
                    "n,nij->ij", w, hess
                )

        jacobian_c = jacobian_q * B.unsqueeze(0)

        result = {
            "potential": potential,
            "moments": moments,
            "total_mass": weights.sum(),
            "normalized_coefficients": c,
            "convex_coordinates": q,
            "jacobian_q": jacobian_q,
            "jacobian_c": jacobian_c,
        }

        if with_jacobian:
            result["jacobian_symmetry_error"] = (
                jacobian_q - jacobian_q.T
            ).abs().max()

            result["jacobian_min_eigenvalue"] = (
                torch.linalg.eigvalsh(
                    0.5 * (jacobian_q + jacobian_q.T)
                ).min()
            )

        return result
