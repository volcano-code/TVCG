"""Bounded, auditable HENMO training with real M/J updates and exact resume.

The default prediction path is the existing encode_geometry/evaluate_operand
interface. A data snapshot must already pass its original native admission;
this trainer grants no automatic promotion or long-training authorization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from experiments.compact_pino_v21.training.losses import physical_loss
from experiments.compact_pino_v21.training.state import (
    StatefulShuffle, capture_rng, restore_rng, seed_everything,
)
from experiments.compact_pino_v21.training.residual_matching import validate_residual_target
from .dataset import canonical_digest, model_contract


SCHEMA = "henmo.training-state.v1"


@dataclass(frozen=True)
class HENMOTrainingConfig:
    seed: int = 830251
    learning_rate: float = 3e-5
    batch_size: int = 1
    jacobian_loss_weight: float = 1.
    residual_loss_weight: float = 0.
    constraint_loss_weight: float = 0.
    chunk_size: int = 65536
    max_steps: int = 312
    max_attempts: int = 624
    gradient_clip_norm: float | None = None
    deterministic: bool = True

    def validate(self):
        for name in ("seed", "batch_size", "chunk_size", "max_steps", "max_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError("invalid integer training option: " + name)
        if self.seed >= 2**32 or self.max_attempts < self.max_steps:
            raise ValueError("seed or training attempt budget is invalid")
        for name in ("learning_rate", "jacobian_loss_weight", "residual_loss_weight", "constraint_loss_weight"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                    or value < 0 or name == "learning_rate" and value == 0):
                raise ValueError("invalid finite training coefficient: " + name)
        if self.gradient_clip_norm is not None and (
                isinstance(self.gradient_clip_norm, bool) or not math.isfinite(self.gradient_clip_norm)
                or self.gradient_clip_norm <= 0):
            raise ValueError("gradient clip norm must be finite and positive")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a Boolean")
        return self


def default_predict(model, record, *, evaluation, chunk_size, jacobian_loss_weight,
                    geometry_cache=None):
    case, meta = record["case"], record["case"]["meta"]
    key = (id(case["blocks"]), np.asarray(case["fixed_moments"], dtype="<f8").tobytes(), float(meta["density"]))
    features = None if geometry_cache is None else geometry_cache.get(key)
    if features is None:
        features = model.encode_geometry(case["blocks"], case["fixed_moments"],
                                         density=meta["density"], inference=evaluation)
        if geometry_cache is not None:
            geometry_cache[key] = features
    with_j = meta["derivative_valid"] and (
        evaluation or jacobian_loss_weight > 0 and record["jacobian_weight"] > 0)
    kwargs = dict(with_jacobian=with_j, chunk_size=chunk_size)
    if with_j and "parameter_scales" in inspect.signature(model.evaluate_operand).parameters:
        kwargs["parameter_scales"] = case["parameter_scales"]
    if "bandwidth" in record:
        kwargs["bandwidth"] = record["bandwidth"]
    grids = getattr(getattr(model, "config", None), "training_coarse_grids", None)
    if not evaluation and grids:
        # One scale per existing query occurrence, with the Python RNG captured
        # in exact-resume state. This does not add labels or multiply exposure.
        kwargs["coarse_grid"] = random.choice(grids)
    return model.evaluate_operand(features, case["operand"], **kwargs)


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _clone_state(state):
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class HENMOTrainer:
    """A single-process trainer whose step boundaries are safe resume points.

    ``predict`` may be supplied for a solver adapter or a synthetic software
    fixture. Its signature equals default_predict. Training examples are the
    already loaded, registered objects; foreign or held-out rows are rejected.
    """

    def __init__(self, model, config, binding, *, records, predict=None):
        if not isinstance(config, HENMOTrainingConfig):
            raise TypeError("HENMOTrainingConfig required")
        self.config = config.validate()
        self.model = model
        self.binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        if (self.binding.get("schema") != "henmo.training-binding.v1"
                or self.binding.get("architecture") != model_contract(model)):
            raise ValueError("checkpoint binding must match the model's exact architecture semantics")
        manifest_hash = self.binding.get("manifest_sha256")
        if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64
                or any(char not in "0123456789abcdef" for char in manifest_hash)):
            raise ValueError("frozen training manifest SHA256 required")
        self.records = list(records)
        self._by_id = {}
        for row in self.records:
            if row.get("role") != "train" or row.get("loader_eligible") is not True:
                raise ValueError("only admitted training rows may be optimized")
            rid = row.get("record_id")
            if not isinstance(rid, str) or not rid or rid in self._by_id or "case" not in row:
                raise ValueError("distinct loaded training records required")
            if row["case"]["meta"].get("split") not in ("train", "synthetic"):
                raise ValueError("held-out cases cannot enter an optimizer batch")
            self._by_id[rid] = row
        if not self.records:
            raise ValueError("at least one admitted training record required")
        bound_rows = {row["record_id"]: row for row in self.binding.get("record_identities", [])}
        self._bound_rows = bound_rows
        for rid, row in self._by_id.items():
            if rid not in bound_rows or bound_rows[rid].get("role") != "train":
                raise ValueError("training record is missing from the frozen binding: " + rid)
            if row.get("case_sha256") != bound_rows[rid].get("case_sha256"):
                raise ValueError("training record case digest differs from the frozen binding")
        self.predict = predict or default_predict
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("no trainable parameters; deterministic baseline must not pretend to train")
        seed_everything(config.seed, config.deterministic)
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.learning_rate)
        self.sampler = StatefulShuffle(len(self.records), config.seed)
        self.step_count = self.attempts = 0
        self.committed_exposures = 0
        self.history = []
        self._validate_objective_coverage()

    def _validate_objective_coverage(self):
        positive = dict(residual=0, constraint=0)
        for row in self.records:
            target = row.get("residual_target")
            valid = isinstance(target, dict) and target.get("valid") is True
            if valid:
                case = row["case"]
                validate_residual_target(target, case["native_moments"], coefficients=case.get("coefficients"))
                if row.get("residual_weight", row["moment_weight"]) > 0:
                    positive["residual"] += 1
                    positive["constraint"] += bool(target.get("multipliers"))
            effective = (row["moment_weight"] > 0 or
                         row["case"]["meta"]["derivative_valid"] and row["jacobian_weight"] > 0
                         and self.config.jacobian_loss_weight > 0 or
                         valid and row.get("residual_weight", row["moment_weight"]) > 0
                         and (self.config.residual_loss_weight > 0 or
                              self.config.constraint_loss_weight > 0 and target.get("multipliers")))
            if not effective:
                raise ValueError("record has no enabled positive-weight objective: " + row["record_id"])
        for name, count in positive.items():
            if getattr(self.config, name + "_loss_weight") > 0 and count == 0:
                raise ValueError("enabled physical objective has no independently qualified targets: " + name)

    def _invalidate(self):
        invalidate = getattr(self.model, "invalidate_caches", None)
        if callable(invalidate):
            invalidate()

    def _check_contract(self):
        if model_contract(self.model) != self.binding["architecture"]:
            raise ValueError("model architecture changed after freezing the training binding")
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if [id(parameter) for parameter in trainable] != [id(parameter) for parameter in self.parameters]:
            raise ValueError("trainable parameter inventory changed during this optimizer run")

    def _predict(self, row, *, evaluation, geometry_cache=None):
        options = dict(evaluation=evaluation, chunk_size=self.config.chunk_size,
                       jacobian_loss_weight=self.config.jacobian_loss_weight)
        if self.predict is default_predict:
            options["geometry_cache"] = geometry_cache
        return self.predict(self.model, row, **options)

    def step(self, records=None):
        self._check_contract()
        if self.step_count >= self.config.max_steps or self.attempts >= self.config.max_attempts:
            raise RuntimeError("the explicit update/attempt budget is exhausted")
        if records is None:
            records = [self.records[index] for index in self.sampler.next_batch(self.config.batch_size)]
        else:
            records = list(records)
            if not records or any(self._by_id.get(row.get("record_id")) is not row for row in records):
                raise ValueError("optimizer batches must use their registered training record objects")
        self.attempts += 1
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
                  if parameter.requires_grad}
        started = time.perf_counter()
        receipt = dict(schema="henmo.optimizer-step.v1", attempt=self.attempts, step=self.step_count,
                       record_ids=[row["record_id"] for row in records],
                       training_manifest_sha256=self.binding["manifest_sha256"],
                       timing_basis="host_wall_not_an_inference_benchmark", committed=False, error=None)
        try:
            # Reuse one encoder graph within this one combined backward only.
            # Never carry a training graph over a parameter update.
            geometry_cache = {}
            predictions = [self._predict(row, evaluation=False, geometry_cache=geometry_cache) for row in records]
            receipt["coarse_grids"] = [getattr(prediction, "diagnostics", {}).get("coarse_grid")
                                        for prediction in predictions]
            loss, info = physical_loss(predictions, records,
                jacobian_loss_weight=self.config.jacobian_loss_weight,
                residual_loss_weight=self.config.residual_loss_weight,
                constraint_loss_weight=self.config.constraint_loss_weight)
            receipt.update(info)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            receipt["loss"] = float(loss.detach())
            if not loss.requires_grad:
                receipt["reason"] = "NO_DIFFERENTIABLE_CONTRIBUTION"
                receipt["gradient_norm"] = 0.
            else:
                loss.backward()
                gradients = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
                if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise FloatingPointError("nonfinite training gradient; no optimizer update committed")
                norm = math.sqrt(sum(float(gradient.detach().abs().double().square().sum()) for gradient in gradients))
                receipt["gradient_norm"] = norm
                if norm == 0:
                    receipt["reason"] = "ZERO_GRADIENT_NO_UPDATE"
                else:
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.gradient_clip_norm,
                                                       error_if_nonfinite=True)
                    # Keep a recoverable transaction boundary if Adam produces
                    # a nonfinite parameter or a finite-precision no-op.
                    optimizer_before = copy.deepcopy(self.optimizer.state_dict())
                    try:
                        self.optimizer.step()
                    except Exception:
                        current = dict(self.model.named_parameters())
                        with torch.no_grad():
                            for name, value in before.items():
                                current[name].copy_(value)
                        self.optimizer.load_state_dict(optimizer_before)
                        raise
                    current = dict(self.model.named_parameters())
                    nonfinite = any(not bool(torch.isfinite(current[name]).all()) for name in before)
                    changed = [name for name, value in before.items() if not torch.equal(value, current[name].detach())]
                    if nonfinite or not changed:
                        with torch.no_grad():
                            for name, value in before.items():
                                current[name].copy_(value)
                        self.optimizer.load_state_dict(optimizer_before)
                        if nonfinite:
                            raise FloatingPointError("nonfinite optimizer result rolled back")
                        receipt["reason"] = "NO_PARAMETER_CHANGE_ROLLED_BACK"
                    else:
                        self.step_count += 1
                        self.committed_exposures += len(records)
                        receipt.update(committed=True, step=self.step_count,
                                       reason="PARAMETERS_UPDATED", changed_parameters=changed)
                        self._invalidate()
        except Exception as exc:
            receipt["error"] = dict(type=type(exc).__name__, message=str(exc))
            self.optimizer.zero_grad(set_to_none=True)
            self._invalidate()
            raise
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            self.history.append(receipt)
        return receipt

    def evaluate(self, records):
        """Evaluate held-out development rows without hiding invalid predictions."""
        from .metrics import prediction_metrics, summarize_prediction_metrics
        records = list(records)
        if any(row.get("role") != "development" or row.get("loader_eligible") is not True for row in records):
            raise ValueError("evaluation requires admitted development rows")
        for row in records:
            bound = self._bound_rows.get(row.get("record_id"), {})
            if (bound.get("role") != "development" or bound.get("case_sha256") != row.get("case_sha256")
                    or row["case"]["meta"].get("split") not in ("development", "synthetic")):
                raise ValueError("evaluation row is not bound to this development snapshot")
        was_training = self.model.training
        self.model.eval()
        rows = []
        try:
            # The operator may need local first-order derivatives. Its outputs
            # are detached per row; no graph or learned cache crosses this pass.
            with torch.enable_grad():
                for record in records:
                    try:
                        prediction = self._predict(record, evaluation=True)
                        rows.append(prediction_metrics(prediction, record))
                    except Exception as exc:
                        rows.append(dict(record_id=record["record_id"],
                            source_group=record["case"]["meta"]["source_group"], status="FAILED",
                            derivative_label_qualified=record["case"]["meta"]["derivative_valid"],
                            error=dict(type=type(exc).__name__, message=str(exc)),
                            moment=None, jacobian=None))
        finally:
            self.model.train(was_training)
            self._invalidate()
        return summarize_prediction_metrics(rows)

    def checkpoint_state(self):
        self._check_contract()
        return dict(schema=SCHEMA, training_config=asdict(self.config), binding=self.binding,
                    binding_sha256=canonical_digest(self.binding), architecture=model_contract(self.model),
                    record_order=[row["record_id"] for row in self.records],
                    model=_clone_state(self.model.state_dict()), optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    sampler=self.sampler.state_dict(), rng=capture_rng(), step_count=self.step_count,
                    attempts=self.attempts, committed_exposures=self.committed_exposures,
                    history=copy.deepcopy(self.history), model_training=self.model.training)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.checkpoint_state()
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = dict(schema="henmo.checkpoint-receipt.v1", path=str(path.resolve()), sha256=_file_sha256(path),
                       binding_sha256=state["binding_sha256"], model_tensor_sha256=_tensor_digest(state["model"]),
                       step_count=self.step_count, attempts=self.attempts,
                       committed_exposures=self.committed_exposures,
                       physics_status="NOT_GRANTED_BY_TRAINING", long_training_promotion=False)
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        return receipt

    def load_checkpoint(self, path, *, expected_sha256=None):
        path = Path(path)
        receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        actual_hash = _file_sha256(path)
        if receipt.get("schema") != "henmo.checkpoint-receipt.v1" or receipt.get("sha256") != actual_hash:
            raise ValueError("checkpoint receipt SHA256 mismatch")
        if expected_sha256 is not None and actual_hash != expected_sha256:
            raise ValueError("checkpoint is not the explicitly requested state")
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = (state.get("schema") == SCHEMA and state.get("training_config") == asdict(self.config)
                    and state.get("binding") == self.binding
                    and state.get("binding_sha256") == canonical_digest(self.binding)
                    and state.get("architecture") == model_contract(self.model)
                    and state.get("record_order") == [row["record_id"] for row in self.records])
        if not expected:
            raise ValueError("resume architecture/config/dataset/material semantics mismatch")
        if (receipt.get("binding_sha256") != state["binding_sha256"]
                or receipt.get("model_tensor_sha256") != _tensor_digest(state["model"])):
            raise ValueError("checkpoint tensor or binding receipt mismatch")
        for name in ("step_count", "attempts", "committed_exposures"):
            value = state.get(name)
            if type(value) is not int or value < 0 or receipt.get(name) != value:
                raise ValueError("invalid checkpoint counter: " + name)
        if state["step_count"] > state["attempts"] or state["attempts"] > self.config.max_attempts:
            raise ValueError("checkpoint counters exceed the frozen budget")
        reference = self.model.state_dict()
        if set(reference) != set(state["model"]):
            raise ValueError("checkpoint parameter inventory differs")
        for name, value in state["model"].items():
            if (not isinstance(value, torch.Tensor) or value.shape != reference[name].shape
                    or value.dtype != reference[name].dtype
                    or (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())):
                raise ValueError("invalid checkpoint tensor: " + name)
        # Validate sampler state before changing live model tensors.
        sampler = StatefulShuffle(len(self.records), self.config.seed)
        sampler.load_state_dict(state["sampler"])
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler = sampler
        self.step_count, self.attempts = state["step_count"], state["attempts"]
        self.committed_exposures, self.history = state["committed_exposures"], state["history"]
        self.model.train(state["model_training"])
        self._invalidate()
        restore_rng(state["rng"])
        return dict(receipt, exact_continuation=True)
