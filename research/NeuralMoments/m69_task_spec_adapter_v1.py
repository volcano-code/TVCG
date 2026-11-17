"""Hash-pinned adapter for extracting the authoritative M69 TaskSpec.

This module deliberately does not reinterpret task fields. It loads the exact
TaskSpec implementation, validates a complete serialized payload using that
implementation, and emits its canonical fields for a reviewed semantic mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

TASK69_SHA256 = "5463dcfe57f3aea21f577fd77d26867ef9e7ec18098b7ff25e7bde5209fc3389"
SCHEMA_VERSION = "tvcg.m69-task-spec-extraction.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any, path: str = "$") -> Any:
    """Convert supported scalar/container values; reject opaque or nonfinite data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite TaskSpec value at {path}")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"non-string mapping key at {path}")
        return {key: _json_safe(item, f"{path}.{key}") for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), path)
    raise TypeError(f"unsupported TaskSpec value at {path}: {type(value).__name__}")


def load_task69_module(source_path: str | Path, *, expected_sha256: str = TASK69_SHA256):
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TaskSpec source not found: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(
            f"TaskSpec source SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )

    # task69.py may import sibling modules from its own source kit.
    source_parent = str(source.parent)
    inserted = source_parent not in sys.path
    if inserted:
        sys.path.insert(0, source_parent)
    try:
        module_name = f"_tvcg_task69_{actual[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load TaskSpec source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        if not hasattr(module, "TaskSpec"):
            raise AttributeError("pinned task69.py does not export TaskSpec")
        return module, actual
    finally:
        if inserted:
            try:
                sys.path.remove(source_parent)
            except ValueError:
                pass


def extract_task_spec(
    source_path: str | Path,
    payload: dict[str, Any] | None = None,
    *,
    expected_sha256: str = TASK69_SHA256,
) -> dict[str, Any]:
    module, source_sha = load_task69_module(
        source_path, expected_sha256=expected_sha256
    )
    task_type = module.TaskSpec
    defaults = task_type()
    if not hasattr(defaults, "to_dict"):
        raise TypeError("TaskSpec instance does not implement to_dict()")
    default_payload = _json_safe(defaults.to_dict(), "$.default")
    if not isinstance(default_payload, dict) or not default_payload:
        raise ValueError("TaskSpec.to_dict() must return a nonempty object")

    selected = default_payload if payload is None else _json_safe(payload, "$.payload")
    if not isinstance(selected, dict):
        raise TypeError("serialized TaskSpec payload must be an object")
    missing = sorted(set(default_payload) - set(selected))
    if missing:
        raise ValueError(f"incomplete TaskSpec payload; missing fields: {missing}")

    if not hasattr(task_type, "from_dict"):
        raise TypeError("TaskSpec class does not implement from_dict()")
    task = task_type.from_dict(selected)
    canonical_payload = _json_safe(task.to_dict(), "$.canonical")
    if not isinstance(canonical_payload, dict):
        raise TypeError("canonical TaskSpec.to_dict() must return an object")
    task_hash = getattr(task, "task_hash", None)
    density = getattr(task, "density", None)
    native_resolution = getattr(task, "native_resolution", None)
    if not isinstance(task_hash, str) or not task_hash:
        raise ValueError("TaskSpec.task_hash must be a nonempty string")
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        raise ValueError("TaskSpec.density must be numeric")
    if not math.isfinite(float(density)) or density <= 0:
        raise ValueError("TaskSpec.density must be finite and positive")
    if isinstance(native_resolution, bool) or not isinstance(native_resolution, int):
        raise ValueError("TaskSpec.native_resolution must be an integer")

    canonical_bytes = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "extracted_source_verified_semantics_pending_review",
        "source": {
            "path": str(Path(source_path).expanduser().resolve()),
            "sha256": source_sha,
        },
        "task": {
            "task_hash": task_hash,
            "density": float(density),
            "native_resolution": native_resolution,
            "field_names": sorted(canonical_payload),
            "payload_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "payload": canonical_payload,
        },
        "semantic_mapping": {
            "status": "pending_source_field_review",
            "axis": None,
            "origin": None,
            "contact_point": None,
            "fixed_shell_policy": None,
            "protected_region_policy": None,
            "stability_constraints": None,
            "tolerances": None,
            "note": (
                "No field meaning is inferred from its name. Review task69.py "
                "and the exact serialized payload before assigning these mappings."
            ),
        },
        "claims": {
            "physical_admission": False,
            "training_label": False,
            "si_units_established": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-source", required=True, help="path to pinned task69.py")
    parser.add_argument("--payload", help="optional JSON file containing a complete TaskSpec")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = None
    if args.payload:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    report = extract_task_spec(args.task_source, payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "source_sha256": report["source"]["sha256"],
        "task_hash": report["task"]["task_hash"],
        "field_names": report["task"]["field_names"],
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
