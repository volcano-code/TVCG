"""Strict current-runner checkpoint loading and bounded frozen inference.

This is evaluation only: no Trainer, optimizer, resume, update or checkpoint
selection is created. Private/legacy weight formats are deliberately refused.
The declared full-cell rule is the only model setting varied by the paired
diagnostic; all rules share the same query-independent encoded geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import time

import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v22_inference_cache import (
    InferenceGeometryCache, geometry_feature_key,
)
from ..p0_diagnostics.baseline_cases import sha256
from ..training.admission import canonical_digest
from ..training.runner import CHECKPOINT_SCHEMA, TrainConfig, model_kwargs


def _tensor_versions(model):
    return tuple((name, id(value), value._version, tuple(value.shape), value.dtype, value.device)
                 for name, value in model.state_dict(keep_vars=True).items())


class FrozenCheckpoint:
    """One fixed-weight inference session; cache budgets come from its config."""

    def __init__(self, model, config, device, receipt):
        self.model, self.config, self.device, self.receipt = model, config, device, receipt
        self.geometry_cache = InferenceGeometryCache(config.evaluation_cache_bytes)
        self.versions = _tensor_versions(model)
        self.encoder_calls = 0

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def encode(self, case):
        self.assert_frozen()
        started = time.perf_counter()
        key = geometry_feature_key(case["blocks"], case["fixed_moments"], case["meta"]["density"])
        encoded = self.geometry_cache.get(key)
        lookup_seconds = time.perf_counter()-started
        cache_hit = encoded is not None
        encoder_seconds = 0.
        if encoded is None:
            self.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                encoded = self.model.encode_geometry(case["blocks"], case["fixed_moments"],
                                                     density=case["meta"]["density"], inference=True)
            self.synchronize()
            encoder_seconds = time.perf_counter()-started
            self.encoder_calls += 1
            self.geometry_cache[key] = encoded
        return encoded, dict(cache_hit=cache_hit, encoder_seconds=encoder_seconds,
                             feature_cache_lookup_seconds=lookup_seconds,
                             shared_across_all_rules=True, device=str(self.device))

    def predict(self, encoded, case, rule, chunk_size):
        self.assert_frozen()
        original_rule = self.model.full_cell_rule
        self.model.full_cell_rule = rule
        try:
            with torch.no_grad():
                return self.model.evaluate_operand(encoded, case["operand"],
                    with_jacobian=case["meta"]["derivative_valid"],
                    bandwidth=case["blocks"].pitch, chunk_size=chunk_size)
        finally:
            self.model.full_cell_rule = original_rule

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError("checkpoint diagnostic requires eval mode and frozen gradient-free parameters")
        if _tensor_versions(self.model) != self.versions:
            raise ValueError("checkpoint parameters or buffers changed during frozen inference")

    def snapshot(self):
        self.assert_frozen()
        return dict(checkpoint=self.receipt, encoder_calls=self.encoder_calls,
                    geometry_features=self.geometry_cache.snapshot(),
                    operator=self.model.efficiency_statistics(), parameter_updates=0,
                    parameter_and_buffer_versions_unchanged=True, training_mode=False,
                    cache_policy="bounded checkpoint-config feature cache; eviction/bypass can require re-encoding across cases; all rules in a case share one encoding")

    def close(self):
        self.geometry_cache.clear()
        self.model.clear_efficiency_caches()


def load_frozen_checkpoint(path, expected_sha256, *, device="cpu"):
    """Load only the explicit current runner schema and strict model state."""
    started = time.perf_counter()
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("--checkpoint-sha256 must be an explicit lowercase SHA256")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("frozen checkpoint diagnostics support cpu or cuda[:index]")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA checkpoint inference but CUDA is unavailable")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("cannot read a safe current-runner checkpoint; private weight formats are unsupported") from exc
    if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("current runner CHECKPOINT_SCHEMA required; private weight formats are unsupported")
    if not isinstance(state.get("config"), dict) or not isinstance(state.get("model"), Mapping) or not state["model"]:
        raise ValueError("runner checkpoint needs its explicit config and nonempty model state")
    if state.get("config_sha256") != canonical_digest(state["config"]):
        raise ValueError("checkpoint config SHA256 mismatch or absent")
    try:
        config = TrainConfig(**state["config"]).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint config is not supported by the current runner") from exc
    for name, value in state["model"].items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("checkpoint model state must contain only named finite tensors")
    model = CompactPINOv21(**model_kwargs(config))
    expected_state = model.state_dict()
    if any(name in expected_state and value.dtype != expected_state[name].dtype for name, value in state["model"].items()):
        raise ValueError("checkpoint tensor dtype differs from the current model; no implicit weight conversion")
    try:
        model.load_state_dict(state["model"], strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint model state does not strictly match the current operator") from exc
    model.to(target_device)
    model.eval()
    model.requires_grad_(False)
    if sha256(path) != expected_sha256:
        raise ValueError("checkpoint changed while being loaded")
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    receipt = dict(path=str(path), sha256=expected_sha256, schema=CHECKPOINT_SCHEMA,
                   config=state["config"], config_sha256=state["config_sha256"],
                   checkpoint_full_cell_rule=config.full_cell_rule,
                   run_id=state.get("run_id"), global_step=state.get("global_step"), attempts=state.get("attempts"),
                   training_manifest_sha256=state.get("training_manifest_sha256"),
                   device=str(target_device), strict_model_state=True, weights_only_load=True,
                   checkpoint_preparation_seconds=time.perf_counter()-started,
                   preparation_scope="two SHA checks, safe checkpoint read, config validation, model construction, strict load, device transfer and eval freeze; excludes input case loading and inference",
                   training_lineage_scope="checkpoint metadata recorded; no training authorization or native-quality claim")
    return FrozenCheckpoint(model, config, target_device, receipt)
