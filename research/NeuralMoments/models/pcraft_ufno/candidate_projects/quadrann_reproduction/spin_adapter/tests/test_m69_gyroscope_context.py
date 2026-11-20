from __future__ import annotations

import unittest

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_faster_bridge import (
    M69GyroscopeContext,
)


class M69GyroscopeContextTests(unittest.TestCase):
    def _context(self):
        beta = torch.zeros(10, dtype=torch.float64)
        beta[0] = 1.0
        moments = torch.zeros(10, dtype=torch.float64)
        moments[0] = 1.0
        return M69GyroscopeContext(
            beta_anchor=beta,
            original_moments=moments,
            task={"schema": "m69.fixed-axis-task.v1"},
            length_scale=1.0,
            active_mask=torch.zeros(10, dtype=torch.bool),
        )

    def test_valid_context(self):
        self._context().validate()

    def test_rejects_non_unit_quadric_anchor(self):
        context = self._context()
        context.beta_anchor[0] = 2.0
        with self.assertRaises(ValueError):
            context.validate()

    def test_rejects_wrong_task_schema(self):
        context = self._context()
        context.task = {"schema": "unknown"}
        with self.assertRaises(ValueError):
            context.validate()

    def test_rejects_invalid_active_mask(self):
        context = self._context()
        context.active_mask = torch.zeros(9, dtype=torch.bool)
        with self.assertRaises(ValueError):
            context.validate()


if __name__ == "__main__":
    unittest.main()
