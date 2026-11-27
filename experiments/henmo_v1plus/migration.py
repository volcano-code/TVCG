"""Explicit semantic transfer. Shape matching alone is never a migration rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModuleSemantics:
    input_coordinates: str
    input_features: str
    output_meaning: str
    normalization: str
    cut_semantics: str = "finite_oriented_capped_operand"

    def validate(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("migration requires explicit nonempty module semantics")
        return self


@dataclass(frozen=True)
class TransferRule:
    source_prefix: str
    target_prefix: str
    source_semantics: ModuleSemantics
    target_semantics: ModuleSemantics


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("migration source must be a tensor state_dict")
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        # dtype-independent byte view also handles bfloat16.
        value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def transfer_compatible_state(model, source_state, rules, *, source_checkpoint_sha256,
                               source_label="E16", dry_run=False):
    """Copy only explicitly mapped modules after checking the entire transfer.

    Exact continuation is handled by training checkpoints, not this function.
    No optimizer or RNG state is transferred. Unmapped target parameters retain
    their initialization, allowing either compatible response reuse or a new head.
    """
    if (not isinstance(source_checkpoint_sha256, str) or len(source_checkpoint_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_checkpoint_sha256)):
        raise ValueError("source checkpoint SHA256 required for an audited transfer")
    if type(dry_run) is not bool or not rules:
        raise ValueError("explicit nonempty transfer rules and Boolean dry_run required")
    target = model.state_dict()
    planned, descriptions = {}, []
    for rule in rules:
        if not isinstance(rule, TransferRule):
            raise ValueError("migration entries must be TransferRule objects")
        rule.source_semantics.validate(); rule.target_semantics.validate()
        if rule.source_semantics != rule.target_semantics:
            raise ValueError("module input/output semantics differ; train a new head instead")
        if not isinstance(rule.source_prefix, str) or not isinstance(rule.target_prefix, str):
            raise ValueError("module prefixes must be strings")
        source_names = [name for name in source_state if name == rule.source_prefix or
                        name.startswith(rule.source_prefix + ".") or rule.source_prefix == ""]
        if not source_names:
            raise ValueError("transfer source module has no tensors: " + rule.source_prefix)
        target_names = []
        for name in source_names:
            suffix = name[len(rule.source_prefix):] if rule.source_prefix else "." + name
            new = rule.target_prefix + suffix if rule.target_prefix else suffix.lstrip(".")
            if new not in target:
                raise ValueError("mapped target tensor does not exist: " + new)
            if new in planned:
                raise ValueError("overlapping migration rules: " + new)
            value = source_state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target[new].shape
                    or value.dtype != target[new].dtype):
                raise ValueError("migration shape/dtype mismatch: " + new)
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite source tensor: " + name)
            planned[new] = value.detach().clone()
            target_names.append(new)
        expected_target = {name for name in target if name == rule.target_prefix or
                           name.startswith(rule.target_prefix + ".") or rule.target_prefix == ""}
        if set(target_names) != expected_target:
            raise ValueError("a mapped module must transfer all of its tensors explicitly")
        descriptions.append(dict(source_prefix=rule.source_prefix, target_prefix=rule.target_prefix,
                                 semantics=asdict(rule.source_semantics), tensors=target_names))
    receipt = dict(schema="henmo.semantic-transfer.v1", source_label=source_label,
                   source_checkpoint_sha256=source_checkpoint_sha256,
                   source_tensor_sha256=_state_digest(source_state), rules=descriptions,
                   transferred=sorted(planned), newly_initialized=sorted(set(target) - set(planned)),
                   optimizer_transferred=False, rng_transferred=False, dry_run=dry_run)
    if not dry_run:
        target.update(planned)
        model.load_state_dict(target, strict=True)
        invalidate = getattr(model, "invalidate_caches", None)
        if callable(invalidate):
            invalidate()
    # Make the report independently serializable before returning success.
    return json.loads(json.dumps(receipt, allow_nan=False))


def transfer_from_checkpoint(model, path, rules, *, expected_sha256, state_key="model",
                             source_label="E16", dry_run=False):
    """Verify the actual source file before semantic transfer (safe tensor loader).

    ``state_key=None`` denotes a file containing the bare state_dict. The key
    must be specified when migrating a different legacy checkpoint format; no
    guessed field or optimizer restoration is performed.
    """
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("source checkpoint file SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if state_key is None:
        state = payload
    else:
        if not isinstance(payload, dict) or state_key not in payload:
            raise ValueError("explicit source state key is absent from checkpoint")
        state = payload[state_key]
    if not isinstance(state, dict):
        raise ValueError("source checkpoint field is not a tensor state_dict")
    receipt = transfer_compatible_state(model, state, rules, source_checkpoint_sha256=actual,
                                        source_label=source_label, dry_run=dry_run)
    receipt["source_path"] = str(path)
    receipt["source_state_key"] = state_key
    return receipt
