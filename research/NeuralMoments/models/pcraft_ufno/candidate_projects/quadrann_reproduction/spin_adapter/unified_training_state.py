"""Named training contracts and checkpoint preflight for UnifiedM80SQNO."""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


TRAINING_SCHEMA = "tvcg.unified-m80-sqno.training.v1"
_SETTING_KEYS = ("gradient_mode", "pcgrad_reduction", "jacobian_weight",
                 "branch_aux_weight", "gradient_clip", "moment_scale", "jacobian_scale")


def parameter_partition(model) -> dict[str, list[str]]:
    """Partition by task dependency, not by the branch owning a decoder.

    M80 and SQNO decoders both serve moment, Jacobian and branch auxiliary
    tasks. Optional native event/UQ heads are not used by the unified losses.
    There are currently no active task-private heads in this objective.
    """
    inactive_prefixes = ("shared_encoder.event_validity_head.", "shared_encoder.uq_head.")
    groups = {"shared": [], "private": [], "inactive": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = "inactive" if name.startswith(inactive_prefixes) else "shared"
            groups[group].append(name)
    if not groups["shared"]:
        raise ValueError("at least one shared trainable parameter is required")
    return groups


def _scale(value, name):
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise ValueError(f"{name} must be real")
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if not tensor.numel() or not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return tensor.tolist()


def make_training_contract(model, *, gradient_mode="fixed", pcgrad_reduction="sum",
                           jacobian_weight=.1, branch_aux_weight=.25, gradient_clip=1.,
                           moment_scale=1., jacobian_scale=1.) -> dict:
    if gradient_mode not in {"fixed", "pcgrad"}:
        raise ValueError("gradient_mode must be fixed or pcgrad")
    if pcgrad_reduction not in {"mean", "sum"}:
        raise ValueError("PCGrad reduction must be mean or sum")
    for name, value in (("jacobian_weight", jacobian_weight),
                        ("branch_aux_weight", branch_aux_weight)):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(float(gradient_clip)) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    return {
        "schema": TRAINING_SCHEMA,
        "gradient_mode": gradient_mode,
        "pcgrad_reduction": pcgrad_reduction if gradient_mode == "pcgrad" else "sum",
        "jacobian_weight": float(jacobian_weight),
        "branch_aux_weight": float(branch_aux_weight),
        "gradient_clip": float(gradient_clip),
        "moment_scale": _scale(moment_scale, "moment_scale"),
        "jacobian_scale": _scale(jacobian_scale, "jacobian_scale"),
        "gradient_coordinates": "real-and-imaginary-components-v1",
        "parameter_partition": parameter_partition(model),
        "trainable_parameters": [
            {"name": name, "shape": list(param.shape), "dtype": str(param.dtype)}
            for name, param in model.named_parameters() if param.requires_grad
        ],
    }


def validate_training_contract(model, contract):
    if not isinstance(contract, Mapping) or any(key not in contract for key in _SETTING_KEYS):
        raise ValueError("invalid unified training contract")
    current = make_training_contract(model, **{key: contract[key] for key in _SETTING_KEYS})
    if dict(contract) != current:
        raise ValueError("unified training contract parameter partition or settings mismatch")
    return current


def training_source_fingerprints():
    here = Path(__file__).resolve()
    files = {here.name: here, "multitask.py": here.parent.parents[2] / "spin_pino/multitask.py"}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"unsupported optimizer/scheduler contract value: {type(value).__name__}")


def optimizer_contract(model, optimizer):
    named = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    seen = []
    groups = []
    serialized_groups = optimizer.state_dict()["param_groups"]
    for group, serialized in zip(optimizer.param_groups, serialized_groups):
        ids = [id(param) for param in group["params"]]
        if any(pid not in named for pid in ids):
            raise ValueError("optimizer contains foreign or frozen parameters")
        seen.extend(ids)
        options = {key: _plain(value) for key, value in group.items() if key != "params"}
        # A resumed scheduler restores the current LR from its state. Compare
        # the configured base LR here, so a decayed LR is not mistaken for drift.
        if "initial_lr" in options:
            options["lr"] = options.pop("initial_lr")
        groups.append({"parameter_names": [named[pid] for pid in ids], "options": options,
                       "state_parameter_ids": list(serialized["params"])})
    if len(seen) != len(set(seen)) or set(seen) != set(named):
        raise ValueError("optimizer parameter coverage must be complete and unique")
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "parameter_groups": groups}


def _scheduler_configuration(state):
    # These counters evolve during training; the remaining state describes
    # the schedule and must match the scheduler selected by the caller.
    evolving = {"last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step",
                "best", "num_bad_epochs", "cooldown_counter", "T_cur", "T_i"}
    def configuration(value):
        if isinstance(value, Mapping):
            if "lr_lambdas" in value:
                raise ValueError("callable schedulers are not supported by strict training contracts")
            return {str(key): configuration(item) for key, item in value.items() if key not in evolving}
        if isinstance(value, (tuple, list)):
            return [configuration(item) for item in value]
        return _plain(value)
    return configuration(state)


def scheduler_contract(scheduler):
    if scheduler is None:
        return None
    return {"class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "configuration": _scheduler_configuration(scheduler.state_dict()),
            "children": [scheduler_contract(child) for child in getattr(scheduler, "_schedulers", [])]}


def checkpoint_training_metadata(model, optimizer, scheduler, data_fingerprint, contract):
    if scheduler is not None and (optimizer is None or scheduler.optimizer is not optimizer):
        raise ValueError("scheduler must belong to the supplied optimizer")
    last = getattr(model, "_last_training_contract", None)
    if contract is None and optimizer is not None:
        contract = last
    if contract is not None:
        contract = validate_training_contract(model, contract)
        if optimizer is None:
            raise ValueError("training contract requires optimizer state")
        if not isinstance(data_fingerprint, str) or not data_fingerprint:
            raise ValueError("training checkpoint requires a nonempty data fingerprint")
        if last is not None and last != contract:
            raise ValueError("training contract differs from the last successful train_step")
    elif optimizer is not None and optimizer.state:
        raise ValueError("trained optimizer requires an explicit training contract")
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    return {
        "training_contract": contract,
        "training_source_fingerprints": training_source_fingerprints() if contract else None,
        "optimizer_contract": optimizer_contract(model, optimizer) if optimizer is not None else None,
        "optimizer_group_options": ([_plain({key: value for key, value in group.items() if key != "params"})
            for group in optimizer_state["param_groups"]] if optimizer_state is not None else None),
        "optimizer_state_parameter_ids": list(optimizer_state["state"]) if optimizer_state is not None else None,
        "scheduler_contract": scheduler_contract(scheduler),
    }


def validate_resume(model, payload, optimizer, scheduler, expected_contract,
                    expected_data_fingerprint, restore_rng):
    """Reject contract/state incompatibility before changing any live state."""
    saved = payload.get("training_contract")
    if expected_contract is not None and optimizer is None:
        raise ValueError("training resume requires an optimizer")
    if optimizer is not None:
        saved_optimizer = payload.get("optimizer")
        if saved_optimizer is None:
            raise ValueError("checkpoint has no optimizer state")
        if saved is not None or expected_contract is not None:
            if saved is None or expected_contract is None:
                raise ValueError("training resume requires matching saved and expected training contracts")
            expected_contract = validate_training_contract(model, expected_contract)
            if saved != expected_contract:
                raise ValueError("unified checkpoint training contract mismatch")
            if not isinstance(expected_data_fingerprint, str) or not expected_data_fingerprint:
                raise ValueError("training resume requires an expected data fingerprint")
            if payload.get("training_source_fingerprints") != training_source_fingerprints():
                raise ValueError("unified checkpoint training source fingerprint mismatch")
        elif saved_optimizer.get("state"):
            raise ValueError("legacy optimizer state has no training contract; load weights for inference only")
        current_optimizer_contract = optimizer_contract(model, optimizer)
        if payload.get("optimizer_contract") != current_optimizer_contract:
            # Historical empty optimizers may still be loaded with old weights.
            if payload.get("optimizer_contract") is not None or saved_optimizer.get("state"):
                raise ValueError("unified checkpoint optimizer contract mismatch")
        if payload.get("scheduler") is not None and scheduler is None:
            raise ValueError("training resume requires the saved scheduler")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("scheduler must belong to the supplied optimizer")
        saved_groups = saved_optimizer.get("param_groups", [])
        if len(saved_groups) != len(optimizer.param_groups):
            raise ValueError("optimizer state parameter group mismatch")
        ids = []
        saved_options = payload.get("optimizer_group_options")
        state_ids = payload.get("optimizer_state_parameter_ids")
        if saved is not None and state_ids is None:
            raise ValueError("training checkpoint has no optimizer state identity inventory")
        if state_ids is not None and state_ids != list(saved_optimizer.get("state", {})):
            raise ValueError("optimizer state parameter identity mismatch")
        for index, (old_group, group) in enumerate(zip(saved_groups, optimizer.param_groups)):
            if len(old_group["params"]) != len(group["params"]):
                raise ValueError("optimizer state parameter count mismatch")
            if payload.get("optimizer_contract") is not None:
                expected_ids = current_optimizer_contract["parameter_groups"][index]["state_parameter_ids"]
                if old_group["params"] != expected_ids:
                    raise ValueError("optimizer state parameter order mismatch")
                options = _plain({key: value for key, value in old_group.items() if key != "params"})
                if saved_options is None or len(saved_options) != len(saved_groups) or options != saved_options[index]:
                    raise ValueError("optimizer state options disagree with saved contract")
            for pid, param in zip(old_group["params"], group["params"]):
                ids.append(pid)
                state = saved_optimizer.get("state", {}).get(pid, {})
                if isinstance(optimizer, torch.optim.AdamW) and state:
                    required = {"step", "exp_avg", "exp_avg_sq"}
                    if old_group.get("amsgrad", False):
                        required.add("max_exp_avg_sq")
                    if not required.issubset(state):
                        raise ValueError("incomplete AdamW optimizer state")
                    if any(not isinstance(state[key], torch.Tensor) for key in required - {"step"}):
                        raise ValueError("invalid AdamW optimizer moment state")
                    step = torch.as_tensor(state["step"])
                    if step.is_complex() or step.numel() != 1 or not torch.isfinite(step).all():
                        raise ValueError("invalid AdamW optimizer step state")
                    if step.item() < 0 or step.item() != int(step.item()):
                        raise ValueError("invalid AdamW optimizer step state")
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all():
                            raise ValueError("nonfinite optimizer state")
                        if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}:
                            if value.shape != param.shape or value.dtype != param.dtype:
                                raise ValueError("optimizer state tensor shape/dtype mismatch")
        if len(ids) != len(set(ids)) or not set(saved_optimizer.get("state", {})).issubset(ids):
            raise ValueError("optimizer state parameter identity mismatch")
    if scheduler is not None:
        if optimizer is None or payload.get("scheduler") is None:
            raise ValueError("checkpoint has no scheduler state or resume optimizer")
        if payload.get("scheduler_contract") != scheduler_contract(scheduler):
            raise ValueError("unified checkpoint scheduler contract mismatch")
        if payload["scheduler"].keys() != scheduler.state_dict().keys():
            raise ValueError("scheduler state keys mismatch")
        if _scheduler_configuration(payload["scheduler"]) != payload["scheduler_contract"]["configuration"]:
            raise ValueError("scheduler state disagrees with saved contract")
        last_lrs = payload["scheduler"].get("_last_lr")
        if last_lrs is not None and _plain(last_lrs) != [_plain(group["lr"]) for group in saved_groups]:
            raise ValueError("optimizer learning rates disagree with scheduler state")
    current = model.state_dict()
    saved_model = payload.get("model", {})
    if current.keys() != saved_model.keys():
        raise ValueError("unified checkpoint model state keys mismatch")
    for name, value in current.items():
        other = saved_model[name]
        if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype:
            raise ValueError(f"unified checkpoint model shape/dtype mismatch: {name}")
        if not torch.isfinite(other).all():
            raise ValueError(f"nonfinite checkpoint model state: {name}")
    if restore_rng:
        # Validate against independent RNG objects; a rejected load must not
        # consume or reset the caller's random streams.
        torch.Generator().set_state(payload["torch_rng"].cpu())
        np.random.RandomState().set_state(payload["numpy_rng"])
        random.Random().setstate(payload["python_rng"])
        cuda_states = payload.get("cuda_rng", [])
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise ValueError("saved CUDA RNG states cannot be restored on this runtime")
            for index, state in enumerate(cuda_states):
                torch.Generator(device=f"cuda:{index}").set_state(state.cpu())
        elif optimizer is not None and torch.cuda.is_available():
            raise ValueError("training checkpoint has no CUDA RNG states for this runtime")
    return expected_contract if optimizer is not None else None
