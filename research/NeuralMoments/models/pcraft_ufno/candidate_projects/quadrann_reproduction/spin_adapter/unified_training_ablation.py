"""Paired short training diagnostics for fixed and PCGrad unified models.

The caller's model, inputs, and RNG streams are preserved. Both arms start from
the same complete model state and replay the same batches and per-step seeds.
The comparison matches optimizer steps, not FLOPs or elapsed time. The built-in
CLI uses artificial targets and is only an engineering smoke test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .unified_m80_sqno import UnifiedM80SQNO, _source_fingerprints
from .masked_derivative_supervision import derivative_supervision_loss
from .unified_training_state import optimizer_contract, training_source_fingerprints

MAX_STEPS = 100
_REQUIRED = {"geometry", "coefficients", "target_moments"}
_SUPERVISION_FIELDS = ("valid_mask", "jacobian_valid_mask", "restricted_basis",
                       "restricted_jacobian", "restricted_valid", "restricted_dimension")
_OPTIONAL = {"target_jacobian", *_SUPERVISION_FIELDS}


def _fingerprint(value: Any) -> str:
    """Hash values, shapes and dtypes, including both complex coordinates."""
    digest = hashlib.sha256()

    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().resolve_conj().resolve_neg().cpu().contiguous()
            add(b"tensor"); add(str(tensor.dtype).encode()); add(str(tuple(tensor.shape)).encode())
            add(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object arrays are not supported in ablation inputs")
            add(b"ndarray"); add(item.dtype.str.encode()); add(str(item.shape).encode())
            add(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("ablation input mapping keys must be strings")
            add(b"mapping")
            for key in sorted(item):
                add(key.encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            add(type(item).__name__.encode()); add(str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, float) and not math.isfinite(item):
            # Validation already rejects active nonfinite values. Inactive
            # labels/dimension padding may be NaN or infinity in Python lists,
            # just as in tensors/ndarrays; hash them without serializing a report.
            add(b"nonfinite_float"); add(repr(item).encode("ascii"))
        elif item is None or isinstance(item, (str, bool, int, float)):
            add(json.dumps(item, allow_nan=False, separators=(",", ":")).encode())
        else:
            raise TypeError(f"unsupported ablation input type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _clone_inputs(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    return copy.deepcopy(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rng_state() -> dict[str, Any]:
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _validate_batches(batches: Sequence[Mapping[str, Any]], moment_scale: Any,
                      jacobian_scale: Any, restricted_scale: Any) -> list[dict[str, Any]]:
    if not isinstance(batches, Sequence) or not batches:
        raise ValueError("batches must be a nonempty sequence")
    prepared = []
    for batch in batches:
        if not isinstance(batch, Mapping) or not _REQUIRED.issubset(batch):
            raise ValueError("each batch needs geometry, coefficients and target_moments")
        if set(batch) - _REQUIRED - _OPTIONAL:
            raise ValueError("unknown training batch fields")
        if not isinstance(batch["geometry"], Mapping):
            raise TypeError("batch geometry must be a mapping")
        q = torch.as_tensor(batch["coefficients"])
        if q.ndim not in (1, 2) or q.shape[-1] != 10 or q.is_complex():
            raise ValueError("coefficients must be real [10] or [Q,10]")
        qb = q.unsqueeze(0) if q.ndim == 1 else q
        if len(qb) == 0 or not torch.isfinite(qb).all() or (qb.to(torch.float64).norm(dim=-1) <= 0).any():
            raise ValueError("coefficients must contain finite nonzero queries")
        target = torch.as_tensor(batch["target_moments"])
        expected = (10,) if q.ndim == 1 else (len(qb), 10)
        if target.shape != expected or target.is_complex():
            raise ValueError(f"target_moments must be real values of shape {expected}")
        s = torch.as_tensor(moment_scale)
        if s.is_complex() or not torch.isfinite(s).all() or (s <= 0).any():
            raise ValueError("normalization scales must be finite, real and positive")
        try:
            shape = torch.broadcast_shapes((len(qb), 10), s.shape)
        except RuntimeError as exc:
            raise ValueError("normalization scales are not broadcastable") from exc
        if shape != (len(qb), 10):
            raise ValueError("normalization scales must not expand the target shape")
        for key in ("valid_mask", "jacobian_valid_mask", "restricted_valid"):
            if batch.get(key) is not None:
                mask = torch.as_tensor(batch[key])
                if mask.dtype != torch.bool or mask.shape != (len(qb),):
                    raise ValueError(f"{key} must be boolean with shape [Q]")
                if key == "valid_mask" and not mask.any():
                    raise ValueError("valid_mask must select at least one valid moment query")
        selected = target.reshape(-1, 10)
        if batch.get("valid_mask") is not None:
            selected = selected[torch.as_tensor(batch["valid_mask"], device=selected.device)]
        if not torch.isfinite(selected).all():
            raise ValueError("selected target_moments must be finite")
        full = batch.get("target_jacobian")
        if full is not None:
            full = torch.as_tensor(full)
            expected_j = (10, 10) if q.ndim == 1 else (len(qb), 10, 10)
            if full.shape != expected_j or full.is_complex():
                raise ValueError(f"target_jacobian must be real values of shape {expected_j}")
            full = full.reshape(-1, 10, 10)
        full_mask = batch.get("jacobian_valid_mask")
        if full_mask is None and full is not None:
            full_mask = batch.get("valid_mask")
        # Validate only selected full rows and active restricted columns. This
        # deliberately permits NaN placeholders in unused derivative labels.
        derivative_supervision_loss(torch.zeros(len(qb), 10, 10, dtype=torch.float64), full,
            derivative_valid=full_mask, restricted_basis=batch.get("restricted_basis"),
            restricted_target=batch.get("restricted_jacobian"),
            restricted_valid=batch.get("restricted_valid"),
            restricted_dimension=batch.get("restricted_dimension"),
            jacobian_scale=jacobian_scale, restricted_scale=restricted_scale)
        prepared.append(_clone_inputs(batch))
    return prepared


@torch.no_grad()
def _evaluate(model: UnifiedM80SQNO, batches: Sequence[Mapping[str, Any]], *,
              moment_scale: Any, jacobian_scale: Any, jacobian_weight: float,
              branch_aux_weight: float, restricted_scale: Any) -> dict[str, Any]:
    model.eval()
    rows = []
    for batch in batches:
        cache = model.encode_geometry(batch["geometry"], inference=True)
        tasks, _ = model._joint_task_losses_and_metadata(cache, batch["coefficients"],
            batch["target_moments"], batch.get("target_jacobian"),
            moment_scale=moment_scale, jacobian_scale=jacobian_scale,
            restricted_scale=restricted_scale, jacobian_weight=jacobian_weight,
            branch_aux_weight=branch_aux_weight, _create_graph=False,
            **{key: batch.get(key) for key in _SUPERVISION_FIELDS})
        row = {name: float(value.detach().cpu()) for name, value in tasks.items()}
        row["loss"] = row["moment"] + jacobian_weight * row["jacobian"] + branch_aux_weight * row["branch_aux"]
        rows.append(row)
    return {"aggregation": "mean_of_per_batch_masked_losses", "per_batch": rows,
            "mean": {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}}


def _parameter_deltas(model: UnifiedM80SQNO, initial: Mapping[str, Tensor]) -> dict[str, Any]:
    real_sq = imag_sq = 0.0
    groups: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        delta = parameter.detach().cpu() - initial[name]
        real = float(delta.real.to(torch.float64).square().sum())
        imag = float(delta.imag.to(torch.float64).square().sum()) if delta.is_complex() else 0.0
        real_sq += real; imag_sq += imag
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0.0) + real + imag
    return {"total_l2": math.sqrt(real_sq + imag_sq), "real_l2": math.sqrt(real_sq),
            "imaginary_l2": math.sqrt(imag_sq),
            "by_top_level_module": {name: math.sqrt(value) for name, value in groups.items()}}


def run_training_ablation(
    model: UnifiedM80SQNO, batches: Sequence[Mapping[str, Any]], *, steps: int = 3,
    seed: int = 830413, learning_rate: float = 1e-5, weight_decay: float = 1e-4,
    jacobian_weight: float = .1, branch_aux_weight: float = .25,
    gradient_clip: float = 1., moment_scale: Any = 1., jacobian_scale: Any = 1.,
    restricted_scale: Any = 1.,
    scope: str = "short_diagnostic",
) -> dict[str, Any]:
    """Train independent fixed/PCGrad clones for 1--100 identical batch steps.

    Batches contain ``geometry``, ``coefficients`` and ``target_moments``.
    Optional full and restricted derivatives use the same masks, raw-coordinate
    projection and per-query averaging as ``UnifiedM80SQNO.train_step``.
    Batches cycle in supplied order. Normalization and
    AdamW settings match across arms. These are new runs from cloned weights,
    including when the caller model came from a resumed run; neither arm reuses
    a caller optimizer or its resume-mode lock. PCGrad uses a sum reduction so the absence
    of conflicts does not introduce an additional task-count learning-rate
    factor. Timings include fresh encoding and the training step; they exclude
    initial/final evaluation and are not a warmed-up performance benchmark.
    """
    if not isinstance(model, UnifiedM80SQNO):
        raise TypeError("model must be UnifiedM80SQNO")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer in [1,{MAX_STEPS}]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 - MAX_STEPS:
        raise ValueError("seed must be an integer allowing the complete uint32 step seed sequence")
    for name, value, positive in (("learning_rate", learning_rate, True),
                                  ("weight_decay", weight_decay, False),
                                  ("jacobian_weight", jacobian_weight, False),
                                  ("branch_aux_weight", branch_aux_weight, False),
                                  ("gradient_clip", gradient_clip, True)):
        if isinstance(value, bool) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"invalid {name}")
    if scope not in {"synthetic_smoke", "short_diagnostic"}:
        raise ValueError("scope must be synthetic_smoke or short_diagnostic")
    # Normalizers may be caller-owned tensors with requires_grad=True as well.
    # Detach these alongside batches, so fixed backward cannot accumulate into them.
    moment_scale = _clone_inputs(moment_scale)
    jacobian_scale = _clone_inputs(jacobian_scale)
    restricted_scale = _clone_inputs(restricted_scale)
    prepared = _validate_batches(batches, moment_scale, jacobian_scale, restricted_scale)
    input_state = {"batches": prepared, "moment_scale": moment_scale,
                   "jacobian_scale": jacobian_scale, "restricted_scale": restricted_scale}
    input_hash = _fingerprint(input_state)
    batch_hashes = [_fingerprint(batch) for batch in prepared]
    initial_hash = _fingerprint(model.state_dict())
    initial_parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    device = next(model.parameters()).device
    if len({p.device for p in model.parameters()}) != 1:
        raise ValueError("short ablation requires all model parameters on one device")
    schedule = [i % len(prepared) for i in range(steps)]
    seeds = [seed + i for i in range(steps)]
    eval_kwargs = dict(moment_scale=moment_scale, jacobian_scale=jacobian_scale,
                       restricted_scale=restricted_scale, jacobian_weight=jacobian_weight,
                       branch_aux_weight=branch_aux_weight)
    training_kwargs = dict(**eval_kwargs, gradient_clip=gradient_clip, pcgrad_reduction="sum")
    rng = _rng_state()
    reports = {}
    paired_optimizer_contract = None
    try:
        for mode in ("fixed", "pcgrad"):
            _seed(seed)
            arm = copy.deepcopy(model)
            # This is an explicitly new paired experiment with fresh optimizers,
            # not continuation of the caller's possibly locked resumed run.
            arm._resume_training_contract = None
            arm._last_training_contract = None
            arm_batches = _clone_inputs(prepared)
            if _fingerprint(arm.state_dict()) != initial_hash:
                raise RuntimeError("ablation clone changed initial model state")
            before = _evaluate(arm, arm_batches, **eval_kwargs)
            arm.train()
            optimizer = arm.build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
            actual_optimizer_contract = optimizer_contract(arm, optimizer)
            if optimizer.state:
                raise ValueError("training ablation requires a fresh optimizer for each arm")
            if paired_optimizer_contract is None:
                paired_optimizer_contract = actual_optimizer_contract
            elif actual_optimizer_contract != paired_optimizer_contract:
                raise RuntimeError("optimizer contracts differ across ablation arms")
            records = []
            encode_before = arm.geometry_encode_calls
            for index, batch_index in enumerate(schedule):
                _seed(seeds[index])
                batch = arm_batches[batch_index]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                # Optimizer updates invalidate both cached features and their graph.
                cache = arm.encode_geometry(batch["geometry"])
                metrics = arm.train_step(optimizer, cache, batch["coefficients"],
                    batch["target_moments"], batch.get("target_jacobian"),
                    **{key: batch.get(key) for key in _SUPERVISION_FIELDS},
                    gradient_mode=mode, **training_kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - start
                records.append({"step": index + 1, "batch_index": batch_index,
                                "batch_sha256": batch_hashes[batch_index], "seed": seeds[index],
                                "seconds": elapsed, "metrics": _json_value(metrics)})
            training_encode_calls = arm.geometry_encode_calls - encode_before
            if _fingerprint({"batches": arm_batches, "moment_scale": moment_scale,
                             "jacobian_scale": jacobian_scale,
                             "restricted_scale": restricted_scale}) != input_hash:
                raise RuntimeError("training modified ablation inputs")
            after = _evaluate(arm, arm_batches, **eval_kwargs)
            reports[mode] = {
                "initial_state_sha256": initial_hash, "input_sha256": input_hash,
                "final_state_sha256": _fingerprint(arm.state_dict()),
                "training_contract": _json_value(arm.training_contract(gradient_mode=mode, **training_kwargs)),
                "optimizer_steps": len(records), "training_geometry_encode_calls": training_encode_calls,
                "initial_evaluation": before, "final_evaluation": after,
                "steps": records, "training_seconds": sum(row["seconds"] for row in records),
                "parameter_delta_norms": _parameter_deltas(arm, initial_parameters),
                "task_gradient_diagnostics": "collected_by_pcgrad" if mode == "pcgrad" else "not_collected",
            }
    finally:
        _restore_rng(rng)
    report = {
        "schema": "tvcg.unified-training-ablation.v1", "scope": scope,
        "protocol": "paired_initial_state_inputs_seeds_and_optimizer_steps",
        "source_fingerprints": {**_source_fingerprints(), **training_source_fingerprints(),
                                Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "architecture": _json_value(model.architecture_contract()),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": str(torch.__version__), "device": str(device),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "fairness_contract": {
            "initial_state_sha256": initial_hash, "input_sha256": input_hash,
            "batch_sha256": batch_hashes, "batch_schedule": schedule, "step_seeds": seeds,
            "optimizer": "torch.optim.AdamW", "learning_rate": learning_rate,
            "optimizer_contract": paired_optimizer_contract,
            "weight_decay": weight_decay, "optimizer_steps_per_arm": steps,
            "optimizer_state_policy": "fresh_for_each_arm",
            "gradient_reduction": "sum", "fresh_geometry_graph_every_step": True,
            "evaluation_partition": "supplied_training_batches_reused",
            "caller_model_and_rng_preserved": True, "arm_order": ["fixed", "pcgrad"],
            "compute_budget_matched": False,
            "timing_scope": "encoding_and_training_step_without_warmup; diagnostic_only",
        },
        "arms": reports,
        "claims": {"engineering_diagnostic": True, "training_performed": True,
                   "formal_accuracy_evidence": False, "pcgrad_superiority_established": False,
                   "performance_benchmark": False},
    }
    # Do not allow a partial nonfinite report to masquerade as a completed run.
    json.dumps(report, allow_nan=False)
    return report


def synthetic_training_case(seed: int = 830411) -> tuple[UnifiedM80SQNO, list[dict[str, Any]]]:
    """Construct a tiny model/geometry with explicitly artificial zero targets."""
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        NativePCRAFTConfig, PcraftNativeMomentOperator, geometry_from_columns,
    )
    from .spin_quadrann_v12 import SpinQuadrANNV12

    rng = _rng_state()
    try:
        _seed(seed)
        # Non-DC modes exercise the imaginary spectral gradient path as well.
        config = NativePCRAFTConfig(grid_size=8, pino_width=2, pino_modes=2, pino_layers=2,
            interval_width=8, interval_depth=1, chunk_intervals=16, min_native_resolution=1,
            convex_potential_enabled=True, convex_potential_components=2)
        encoder = PcraftNativeMomentOperator(config)
        sqno = SpinQuadrANNV12(pcraft_width=8, hidden_dim=4, final_hidden_dim=16,
            num_layers=2, neighbors=4, temperature=.04, quadrature_order=2,
            chunk_intervals=2, chunk_nodes=64, query_chunk=2)
        model = UnifiedM80SQNO(encoder, sqno)
        geometry = geometry_from_columns({
            "columns": np.array([[1, 1, 1, 5], [1, 2, 2, 7], [2, 1, 0, 6],
                                 [3, 3, 1, 8], [4, 3, 0, 7]], dtype=np.int32),
            "origin": np.array([-1., -2., .5]), "h": np.array(.25),
            "side": np.array(512, dtype=np.int32), "density": np.array(1.7),
        })
        q = torch.randn(4, 10, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        batches = [{"geometry": geometry, "coefficients": q,
                    "target_moments": torch.zeros(4, 10, dtype=torch.float64),
                    "target_jacobian": torch.zeros(4, 10, 10, dtype=torch.float64),
                    "valid_mask": torch.tensor([True, True, False, True]),
                    "jacobian_valid_mask": torch.tensor([True, False, False, True])}]
        return model, batches
    finally:
        _restore_rng(rng)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic fixed/PCGrad short training diagnostic")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=830413, help="per-step RNG seed; fixture initialization is fixed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model, batches = synthetic_training_case()
    model.to(args.device)
    report = run_training_ablation(model, batches, steps=args.steps, seed=args.seed, scope="synthetic_smoke")
    report["synthetic_fixture"] = {"geometry_columns": 5, "query_count": 4,
                                   "initialization_seed": 830411,
                                   "targets": "artificial_zero_targets"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent,
            prefix=args.output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote synthetic training diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
