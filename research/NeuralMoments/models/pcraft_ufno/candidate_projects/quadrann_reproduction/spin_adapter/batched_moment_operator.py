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

from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from .spin_moment_operator import (
    NATIVE_B,
    raw_moment_basis,
)


def _integrate_block(xyz, weights, queries, eps2, eye, with_jacobian):
    """Contract each rank-one term over nodes before forming the 10x10 matrix."""
    basis = raw_moment_basis(xyz)
    score = queries @ basis.T
    smooth = (score.square() + eps2 * queries.square().sum(-1, keepdim=True)).sqrt()
    numerator = score.unsqueeze(-1) * basis + eps2 * queries[:, None, :]
    weighted_inverse = weights / smooth
    potential = 0.5 * ((score + smooth) * weights).sum(-1)
    moments = 0.5 * (weights @ basis + (weighted_inverse.unsqueeze(-1) * numerator).sum(1))

    if with_jacobian:
        basis_gram = (weighted_inverse.unsqueeze(-1) * basis).transpose(1, 2) @ basis
        negative_gram = numerator.transpose(1, 2) @ (
            (weighted_inverse / smooth.square()).unsqueeze(-1) * numerator
        )
        jacobian = 0.5 * (basis_gram + eps2 * weighted_inverse.sum(-1)[:, None, None] * eye
                          - negative_gram)
    else:
        jacobian = moments.new_zeros(len(queries), 10, 10)
    return potential, moments, jacobian


def _integration_contexts():
    from .memory_profile import profile_recompute
    return nullcontext(), profile_recompute("Moment Operator")


class BatchedSpinMomentOperator:
    def __init__(
        self,
        temperature: float = 0.04,
        chunk_nodes: int = 2048,
        chunk_queries: int = 1,
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

        # Split once. Backward concatenates adjacent block gradients in one
        # operation instead of zero-filling a full geometry for each slice.
        nodes = xyz.split(self.chunk_nodes)
        node_weights = weights.split(self.chunk_nodes)
        recompute = torch.is_grad_enabled() and any(
            value.requires_grad for value in (xyz, weights, q_all)
        )
        all_potentials = []
        all_moments = []
        all_jacobians_q = []
        from .memory_profile import profile_adjoint, profile_stage, profile_tensors
        query_blocks = q_all.split(self.chunk_queries)
        with profile_stage("Moment Operator", chunks=len(nodes) * len(query_blocks), nodes=len(xyz),
                           queries=len(q_all)):
            profile_tensors(node_xyz=xyz, node_weights=weights, convex_queries=q_all)
            for q in query_blocks:
                potential = xyz.new_zeros(len(q))
                moments = xyz.new_zeros(len(q), 10)
                jacobian = xyz.new_zeros(len(q), 10, 10)
                for points, mass in zip(nodes, node_weights):
                    arguments = (points, mass, q, eps2, eye, with_jacobian)
                    if recompute:
                        p_part, m_part, j_part = checkpoint(
                            _integrate_block, *arguments, use_reentrant=False, preserve_rng_state=False,
                            context_fn=_integration_contexts
                        )
                    else:
                        p_part, m_part, j_part = _integrate_block(*arguments)
                    potential = potential + p_part
                    moments = moments + m_part
                    if with_jacobian:
                        jacobian = jacobian + j_part

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

        profile_adjoint("Moment Operator", (moments, jacobian_input, jacobian_q, potential),
                        (xyz, weights, coefficients))
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
