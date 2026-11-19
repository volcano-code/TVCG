"""Opt-in multi-task gradient balancing primitives.

Predictive aleatoric UQ scales are intentionally not used as task weights.
These utilities operate on explicit scalar task losses and shared parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math
import torch
from torch import Tensor, nn


TASK_ORDER = ("moment", "jacobian", "event", "uq", "physics")


def fixed_weighted_objective(losses: Mapping[str, Tensor],
                             weights: Mapping[str, float]) -> Tensor:
    active = [(k, v) for k, v in losses.items() if v is not None]
    if not active:
        raise ValueError("at least one task loss is required")
    result = None
    for name, loss in active:
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid task weight for {name}")
        term = loss * weight
        result = term if result is None else result + term
    return result


def task_gradient_matrix(losses: Mapping[str, Tensor],
                         shared_parameters: Sequence[nn.Parameter],
                         *, retain_graph: bool = True) -> tuple[list[str], list[Tensor]]:
    """Flatten each task gradient over shared parameters; inactive tasks omitted."""
    params = [p for p in shared_parameters if p.requires_grad]
    if not params:
        raise ValueError("shared_parameters must contain trainable parameters")
    names, rows = [], []
    for name, loss in losses.items():
        if loss is None or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                    allow_unused=True)
        pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                  for p, g in zip(params, grads)]
        row = torch.cat(pieces)
        if not torch.isfinite(row).all():
            raise FloatingPointError(f"nonfinite gradient for task {name}")
        names.append(name); rows.append(row)
    if not rows:
        raise ValueError("no active task has gradients on shared parameters")
    return names, rows


def pcgrad_project(gradients: Sequence[Tensor], *, eps: float = 1e-12,
                   shuffle: bool = False, generator: torch.Generator | None = None
                   ) -> tuple[Tensor, dict]:
    """Symmetric PCGrad projection; returns mean projected gradient and diagnostics."""
    if not gradients:
        raise ValueError("gradients must not be empty")
    base = [g.clone() for g in gradients]
    if any(g.shape != base[0].shape for g in base):
        raise ValueError("all task gradients must have identical flattened shapes")
    if any(not torch.isfinite(g).all() for g in base):
        raise FloatingPointError("task gradients must be finite")
    projected = []
    order = list(range(len(base)))
    for i, original in enumerate(base):
        g = original.clone()
        peers = [j for j in range(len(base)) if j != i]
        if shuffle and len(peers) > 1:
            perm = torch.randperm(len(peers), generator=generator).tolist()
            peers = [peers[k] for k in perm]
        count = 0
        for j in peers:
            dot = torch.dot(g, base[j])
            if dot < 0:
                denom = torch.dot(base[j], base[j]).clamp_min(eps)
                g = g - (dot / denom) * base[j]
                count += 1
        projected.append(g); order[i] = count
    cosine = []
    for i in range(len(base)):
        row = []
        for j in range(len(base)):
            denom = base[i].norm() * base[j].norm()
            row.append(float(torch.dot(base[i], base[j]) / denom.clamp_min(eps)))
        cosine.append(row)
    combined = torch.stack(projected).mean(0)
    return combined, {"task_count": len(base), "projection_counts": order,
                      "cosine_matrix": cosine,
                      "gradient_norms": [float(g.norm()) for g in base],
                      "projected_norms": [float(g.norm()) for g in projected]}


@dataclass
class GradNormBalancer:
    """Stateful GradNorm weight update; absent tasks do not update."""
    task_names: tuple[str, ...]
    alpha: float = 1.5
    learning_rate: float = 0.025
    min_weight: float = 0.05
    max_weight: float = 20.0
    initial_losses: dict[str, float] | None = None
    weights: dict[str, float] | None = None

    def __post_init__(self):
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be nonempty and unique")
        if not (self.alpha >= 0 and self.learning_rate > 0 and
                0 < self.min_weight < self.max_weight):
            raise ValueError("invalid GradNorm hyperparameters")
        if self.weights is None:
            self.weights = {k: 1.0 for k in self.task_names}
        if self.initial_losses is None:
            self.initial_losses = {}

    def update(self, losses: Mapping[str, float],
               shared_gradient_norms: Mapping[str, float]) -> dict[str, float]:
        active = [k for k in self.task_names if k in losses and k in shared_gradient_norms]
        if not active:
            return dict(self.weights)
        for k in active:
            value = float(losses[k]); norm = float(shared_gradient_norms[k])
            if not math.isfinite(value) or value < 0 or not math.isfinite(norm) or norm < 0:
                raise ValueError(f"invalid GradNorm statistic for {k}")
            self.initial_losses.setdefault(k, max(value, 1e-12))
        rates = {k: max(float(losses[k]), 1e-12) / self.initial_losses[k] for k in active}
        mean_rate = sum(rates.values()) / len(rates)
        raw = {}
        for k in active:
            target = mean_rate ** self.alpha
            relative = rates[k] / max(mean_rate, 1e-12)
            # Multiplicative controller: reduce weight on tasks with excessive
            # gradient norm relative to their inverse training-rate target.
            desired = max(shared_gradient_norms[k], 1e-12) * relative ** self.alpha
            current = max(self.weights[k], 1e-12)
            raw[k] = current * math.exp(self.learning_rate *
                max(-2.0, min(2.0, math.log((desired + 1e-12) /
                    (shared_gradient_norms[k] * current + 1e-12)))))
        mean = sum(raw.values()) / len(raw)
        for k in active:
            self.weights[k] = min(self.max_weight, max(self.min_weight, raw[k] / max(mean, 1e-12)))
        return dict(self.weights)


class LearnedTaskWeights(nn.Module):
    """Homoscedastic task weighting; separate from predictive UQ heads."""
    def __init__(self, task_names: Sequence[str], min_log_variance: float = -5.,
                 max_log_variance: float = 5.):
        super().__init__()
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be nonempty and unique")
        if not min_log_variance < max_log_variance:
            raise ValueError("invalid log variance bounds")
        self.task_names = tuple(task_names)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self.log_variances = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in self.task_names})

    def forward(self, losses: Mapping[str, Tensor]) -> Tensor:
        active = [(k, v) for k, v in losses.items() if v is not None]
        if not active:
            raise ValueError("at least one active task loss is required")
        total = None
        for name, loss in active:
            if name not in self.log_variances:
                raise KeyError(f"unknown task {name}")
            s = self.log_variances[name].clamp(self.min_log_variance, self.max_log_variance)
            term = torch.exp(-s) * loss + s
            total = term if total is None else total + term
        return total
