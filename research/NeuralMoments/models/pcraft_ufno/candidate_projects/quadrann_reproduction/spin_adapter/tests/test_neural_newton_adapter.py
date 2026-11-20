from __future__ import annotations

import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.neural_newton_adapter import (
    ProposalPolicy,
    exact_correction_and_accept,
    make_neural_newton_proposal,
)


class NeuralNewtonAdapterTests(unittest.TestCase):
    def setUp(self):
        self.beta = torch.zeros(10, dtype=torch.float64)
        self.jac = torch.eye(10, dtype=torch.float64)
        self.residual = torch.ones(10, dtype=torch.float64)

    def test_proposal_is_bounded_and_does_not_accept(self):
        proposal = make_neural_newton_proposal(
            current_coefficients=self.beta,
            predicted_residual=self.residual,
            predicted_jacobian=self.jac,
            policy=ProposalPolicy(trust_radius=0.05, max_step_norm=0.04),
        )
        self.assertTrue(proposal.eligible_for_exact_evaluation)
        self.assertLessEqual(float(proposal.predicted_step_norm), 0.0400001)
        self.assertTrue(torch.equal(proposal.candidate_coefficients, self.beta + proposal.step))

    def test_uncertainty_and_event_risk_force_fallback(self):
        proposal = make_neural_newton_proposal(
            current_coefficients=self.beta,
            predicted_residual=self.residual,
            predicted_jacobian=self.jac,
            moment_log_std=torch.ones(10),
            event_probability=torch.tensor([0.9]),
        )
        self.assertFalse(proposal.eligible_for_exact_evaluation)
        self.assertIn("moment_uncertainty_above_policy", proposal.fallback_reasons)
        self.assertIn("event_risk_above_policy", proposal.fallback_reasons)

    def test_exact_gate_accepts_only_feasible_decreasing_trial(self):
        proposal = make_neural_newton_proposal(
            current_coefficients=self.beta,
            predicted_residual=self.residual,
            predicted_jacobian=self.jac,
            policy=ProposalPolicy(
                trust_radius=0.1,
                max_step_norm=0.1,
                armijo=0.0,
                feasibility_tolerance=1e-7,
            ),
        )
        calls = []

        def exact_evaluator(beta):
            calls.append(beta.clone())
            objective = float(((beta - 0.1) ** 2).sum())
            return {"objective": objective, "constraint_violation": 0.0}

        result = exact_correction_and_accept(proposal, exact_evaluator=exact_evaluator)
        self.assertTrue(result.accepted)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(result.coefficients, proposal.candidate_coefficients))

    def test_exact_gate_rejects_infeasible_or_non_decreasing_trial(self):
        proposal = make_neural_newton_proposal(
            current_coefficients=self.beta,
            predicted_residual=self.residual,
            predicted_jacobian=self.jac,
        )

        def infeasible(beta):
            return {"objective": float((beta ** 2).sum()), "constraint_violation": 1.0}

        result = exact_correction_and_accept(proposal, exact_evaluator=infeasible)
        self.assertFalse(result.accepted)
        self.assertTrue(torch.equal(result.coefficients, self.beta))

    def test_ineligible_proposal_never_evaluates_trial(self):
        proposal = make_neural_newton_proposal(
            current_coefficients=self.beta,
            predicted_residual=self.residual,
            predicted_jacobian=self.jac,
            event_probability=torch.tensor([1.0]),
        )
        calls = []

        def exact_evaluator(beta):
            calls.append(beta.clone())
            return {"objective": float((beta ** 2).sum()), "constraint_violation": 0.0}

        result = exact_correction_and_accept(proposal, exact_evaluator=exact_evaluator)
        self.assertFalse(result.accepted)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
