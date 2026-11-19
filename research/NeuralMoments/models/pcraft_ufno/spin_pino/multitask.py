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


def _real_inner_product(a: Tensor, b: Tensor) -> Tensor:
    """Real Euclidean inner product, including complex-valued parameter gradients."""
    return torch.sum(torch.conj(a) * b).real


def pcgrad_project(gradients: Sequence[Tensor], *, eps: float = 1e-12,
                   shuffle: bool = False, generator: torch.Generator | None = None
                   ) -> tuple[Tensor, dict]:
    """Symmetric PCGrad projection for real or complex flattened gradients."""
    if not gradients:
        raise ValueError("gradients must not be empty")
    base = [g.clone() for g in gradients]
    if any(g.shape != base[0].shape for g in base):
        raise ValueError("all task gradients must have identical flattened shapes")
    if any(g.device != base[0].device for g in base):
        raise ValueError("all task gradients must be on the same device")
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
            dot = _real_inner_product(g, base[j])
            if dot.item() < 0:
                denom = _real_inner_product(base[j], base[j]).clamp_min(eps)
                g = g - (dot / denom).to(dtype=g.real.dtype) * base[j]
                count += 1
        projected.append(g); order[i] = count
    cosine = []
    for i in range(len(base)):
        row = []
        for j in range(len(base)):
            denom = base[i].norm() * base[j].norm()
            value = _real_inner_product(base[i], base[j]) / denom.clamp_min(eps)
            row.append(float(value))
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

    def state_dict(self) -> dict:
        return {"task_names": list(self.task_names), "alpha": self.alpha,
                "learning_rate": self.learning_rate, "min_weight": self.min_weight,
                "max_weight": self.max_weight, "initial_losses": dict(self.initial_losses),
                "weights": dict(self.weights)}

    def load_state_dict(self, state: Mapping) -> None:
        expected = tuple(state.get("task_names", ()))
        if expected != tuple(self.task_names):
            raise ValueError("GradNorm task-name contract mismatch")
        for key, current in (("alpha", self.alpha), ("learning_rate", self.learning_rate),
                             ("min_weight", self.min_weight), ("max_weight", self.max_weight)):
            if float(state.get(key, float("nan"))) != float(current):
                raise ValueError(f"GradNorm hyperparameter contract mismatch: {key}")
        initial = dict(state.get("initial_losses", {}))
        weights = dict(state.get("weights", {}))
        if set(weights) != set(self.task_names) or not set(initial).issubset(self.task_names):
            raise ValueError("GradNorm state task keys mismatch")
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in weights.values()):
            raise ValueError("GradNorm state contains invalid weights")
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in initial.values()):
            raise ValueError("GradNorm state contains invalid initial losses")
        self.initial_losses = {k: float(v) for k, v in initial.items()}
        self.weights = {k: float(v) for k, v in weights.items()}

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


def pcgrad_assign_gradients(losses: Mapping[str, Tensor | None],
                            weights: Mapping[str, float],
                            shared_parameters: Sequence[nn.Parameter],
                            private_parameters: Sequence[nn.Parameter],
                            *, eps: float = 1e-12) -> dict:
    """Compute per-task grads, PCGrad only shared params, retain weighted private-head grads.

    Each task loss is differentiated separately. Shared gradients are projected
    before summation; task-private gradients are simply summed with task weights.
    """
    shared=[p for p in shared_parameters if p.requires_grad]
    private=[p for p in private_parameters if p.requires_grad]
    if not shared: raise ValueError("at least one shared trainable parameter is required")
    if {id(p) for p in shared}&{id(p) for p in private}:
        raise ValueError("shared/private parameter groups overlap")
    if len({id(p) for p in shared+private})!=len(shared)+len(private):
        raise ValueError("duplicate parameter in gradient partition")
    active=[]
    for name,loss in losses.items():
        if loss is None: continue
        weight=float(weights.get(name,0.0))
        if not math.isfinite(weight) or weight<0: raise ValueError(f"invalid task weight for {name}")
        if weight==0 or not loss.requires_grad: continue
        active.append((name,loss*weight,weight))
    if not active: raise ValueError("no active weighted task losses")
    shared_rows=[]; private_acc={id(p):None for p in private}
    task_names=[]; norms={}; private_norms={}
    all_params=shared+private
    for index,(name,loss,weight) in enumerate(active):
        grads=torch.autograd.grad(loss,all_params,retain_graph=index<len(active)-1,
                                  allow_unused=True)
        sg=grads[:len(shared)]
        flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                        for p,g in zip(shared,sg)])
        if not torch.isfinite(flat).all(): raise FloatingPointError(f"nonfinite shared gradient for {name}")
        shared_rows.append(flat);task_names.append(name);norms[name]=float(flat.norm())
        for p,g in zip(private,grads[len(shared):]):
            if g is not None:
                current=private_acc[id(p)]
                private_acc[id(p)]=g.detach().clone() if current is None else current+g.detach()
                private_norms.setdefault(name,0.0)
                private_norms[name]+=float(g.detach().norm().square())
    combined,diagnostics=pcgrad_project(shared_rows,eps=eps)
    offset=0
    for p in shared:
        count=p.numel()
        p.grad=combined[offset:offset+count].view_as(p).to(dtype=p.dtype).detach().clone()
        offset+=count
    for p in private:
        value=private_acc[id(p)]
        p.grad=None if value is None else value
    return {"task_names":task_names,"shared_gradient_norms":norms,
            "private_gradient_norms":{k:math.sqrt(v) for k,v in private_norms.items()},
            **diagnostics}


def event_positive_class_weight(event_valid: Sequence[bool],
                                event_label_valid: Sequence[bool] | None = None,
                                *, min_weight: float = 0.25,
                                max_weight: float = 8.0) -> dict:
    """Compute bounded negative/positive ratio from labeled training rows only.

    event_valid=True is the positive/regular class according to the current
    teacher-label contract. Unknown rows are excluded, never treated as negatives.
    """
    target = torch.as_tensor(event_valid, dtype=torch.bool).reshape(-1)
    known = (torch.ones_like(target) if event_label_valid is None else
             torch.as_tensor(event_label_valid, dtype=torch.bool).reshape(-1))
    if target.shape != known.shape or not known.any():
        raise ValueError("event labels and known-mask must match and contain labeled rows")
    if not (math.isfinite(min_weight) and math.isfinite(max_weight)
            and 0 < min_weight <= max_weight):
        raise ValueError("invalid event class-weight bounds")
    positive = int((target & known).sum())
    negative = int((~target & known).sum())
    if positive == 0:
        raise ValueError("cannot estimate positive class weight without positive event labels")
    raw = negative / positive if negative else min_weight
    weight = min(max_weight, max(min_weight, raw))
    return {"positive_count": positive, "negative_count": negative,
            "unknown_count": int((~known).sum()), "raw_positive_weight": float(raw),
            "positive_weight": float(weight), "clipped": bool(weight != raw),
            "source": "labeled_training_rows_only"}
