"""Batched Smooth-Hinge ten-moment operator for SQNO v1.2.

Preserves the existing single-query SpinMomentOperator convention:
q = normalize(c) * NATIVE_B
J_c = J_q @ diag(NATIVE_B)

Inputs:
    xyz:          [N, 3]
    weights:      [N]
    coefficients: [B, 10] or [10]

Outputs:
    moments:      [B, 10]
    jacobian_q:   [B, 10, 10]
    jacobian_c:   [B, 10, 10]

Query and node chunking preserve the autograd graph.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from .spin_moment_operator import (
    NATIVE_B,
    raw_moment_basis,
)


class BatchedSpinMomentOperator:
    def __init__(
        self,
        temperature: float = 0.04,
        chunk_nodes: int = 65536,
        chunk_queries: int = 16,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if chunk_nodes < 1 or chunk_queries < 1:
            raise ValueError("chunk sizes must be positive")

        self.temperature = float(temperature)
        self.chunk_nodes = int(chunk_nodes)
        self.chunk_queries = int(chunk_queries)

    def evaluate(
        self,
        xyz: Tensor,
        weights: Tensor,
        coefficients: Tensor,
        *,
        with_jacobian: bool = True,
    ) -> dict[str, Tensor]:

        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must be [N,3]")
        if weights.shape != (len(xyz),):
            raise ValueError("weights must be [N]")
        if coefficients.ndim != 2 or coefficients.shape[1] != 10:
            raise ValueError("coefficients must be [B,10]")
        if coefficients.shape[0] < 1:
            raise ValueError("empty query batch")

        if not (
            xyz.device == weights.device == coefficients.device
        ):
            raise ValueError("device mismatch")

        if not (
            xyz.dtype == weights.dtype == coefficients.dtype
        ):
            raise ValueError("dtype mismatch")

        if not all(
            torch.isfinite(v).all()
            for v in (xyz, weights, coefficients)
        ):
            raise ValueError("nonfinite inputs")

        if (weights < 0).any():
            raise ValueError("negative integration weights")

        if (
            torch.linalg.vector_norm(
                coefficients, dim=-1
            ) <= 1e-12
        ).any():
            raise ValueError("zero coefficient vector")

        B = coefficients.new_tensor(NATIVE_B)

        normalized = coefficients / torch.linalg.vector_norm(
            coefficients, dim=-1, keepdim=True
        )

        q_all = normalized * B

        eps2 = self.temperature ** 2
        eye = torch.eye(
            10,
            device=xyz.device,
            dtype=xyz.dtype,
        )

        all_potentials = []
        all_moments = []
        all_jacobians_q = []

        for qs in range(0, len(q_all), self.chunk_queries):
            qe = min(qs + self.chunk_queries, len(q_all))
            q = q_all[qs:qe]

            batch = len(q)

            potential = xyz.new_zeros(batch)
            moments = xyz.new_zeros(batch, 10)
            jacobian = xyz.new_zeros(batch, 10, 10)

            norm2 = q.square().sum(dim=-1)

            for start in range(
                0, len(xyz), self.chunk_nodes
            ):
                stop = min(
                    start + self.chunk_nodes, len(xyz)
                )

                w = weights[start:stop]

                # During Moment-only training, checkpoint each node block.
                # This preserves exact forward equations and recomputes
                # internal activations during backward.
                if (
                    not with_jacobian
                    and torch.is_grad_enabled()
                    and (w.requires_grad or q.requires_grad)
                ):
                    def chunk_forward(xyz_part, weights_part, q_part):
                        basis_part = raw_moment_basis(xyz_part)
                        a_part = basis_part @ q_part.T

                        norm2_part = q_part.square().sum(dim=-1)

                        s_part = torch.sqrt(
                            a_part.square()
                            + eps2 * norm2_part.unsqueeze(0)
                        )

                        nvec_part = (
                            a_part.unsqueeze(-1)
                            * basis_part[:, None, :]
                            + eps2 * q_part[None, :, :]
                        )

                        grad_q_part = 0.5 * (
                            basis_part[:, None, :]
                            + nvec_part / s_part.unsqueeze(-1)
                        )

                        potential_part = (
                            0.5 * weights_part[:, None]
                            * (a_part + s_part)
                        ).sum(dim=0)

                        moments_part = torch.einsum(
                            "n,nbi->bi",
                            weights_part,
                            grad_q_part,
                        )

                        return potential_part, moments_part

                    p_part, m_part = checkpoint(
                        chunk_forward,
                        xyz[start:stop],
                        w,
                        q,
                        use_reentrant=False,
                    )

                    potential = potential + p_part
                    moments = moments + m_part

                else:
                    basis = raw_moment_basis(xyz[start:stop])

                    # [N, B]
                    a = basis @ q.T

                    s = torch.sqrt(
                        a.square()
                        + eps2 * norm2.unsqueeze(0)
                    )

                    nvec = (
                        a.unsqueeze(-1) * basis[:, None, :]
                        + eps2 * q[None, :, :]
                    )

                    grad_q = 0.5 * (
                        basis[:, None, :]
                        + nvec / s.unsqueeze(-1)
                    )

                    potential = potential + (
                        0.5 * w[:, None] * (a + s)
                    ).sum(dim=0)

                    moments = moments + torch.einsum(
                        "n,nbi->bi", w, grad_q
                    )

                    if with_jacobian:
                        outer = (
                            basis[:, None, :, None]
                            * basis[:, None, None, :]
                        )

                        numerator = (
                            outer
                            + eps2 * eye[None, None, :, :]
                        )

                        hessian = 0.5 * (
                            numerator / s[:, :, None, None]
                            - (
                                nvec[:, :, :, None]
                                * nvec[:, :, None, :]
                            ) / s[:, :, None, None].pow(3)
                        )

                        jacobian = jacobian + torch.einsum(
                            "n,nbij->bij", w, hessian
                        )

            all_potentials.append(potential)
            all_moments.append(moments)

            if with_jacobian:
                all_jacobians_q.append(jacobian)

        potential = torch.cat(all_potentials, dim=0)
        moments = torch.cat(all_moments, dim=0)

        if with_jacobian:
            jacobian_q = torch.cat(
                all_jacobians_q, dim=0
            )
        else:
            jacobian_q = moments.new_zeros(
                len(moments), 10, 10
            )

        # Derivative with respect to normalized coefficients u:
        # q = B*u, hence J_u = J_q @ diag(B).
        jacobian_normalized = (
            jacobian_q * B[None, None, :]
        )

        # Derivative with respect to raw input coefficients c:
        # u = c / ||c||_2
        # du/dc = (I - u u^T) / ||c||_2.
        input_norm = torch.linalg.vector_norm(
            coefficients, dim=-1, keepdim=True
        )

        identity = torch.eye(
            10,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )

        projector = (
            identity.unsqueeze(0)
            - normalized.unsqueeze(-1)
            * normalized.unsqueeze(-2)
        )

        jacobian_input = torch.matmul(
            jacobian_normalized, projector
        ) / input_norm.unsqueeze(-1)

        return {
            "potential": potential,
            "moments": moments,
            "normalized_coefficients": normalized,
            "convex_coordinates": q_all,
            "jacobian_q": jacobian_q,
            "jacobian_c": jacobian_normalized,
            "jacobian_normalized": jacobian_normalized,
            "jacobian_input": jacobian_input,
            "total_mass": weights.sum(),
        }
