"""Collect small, read-only Faster107 handoff evidence without running a loader.

Only the five paths below are read. The pinned index is checked before any
output directory is created. Original bytes are copied, and ``handoff.json``
contains those bytes as UTF-8 text (or base64 for non-UTF-8 sources), their
hashes, shallow JSON key summaries, and an AST-only consumer interface.

This is an inspection tool, not a dataset validator or a training entry point.
It does not follow paths found inside JSON, load NPZ/meshes, import the consumer,
construct a corpus/model, call a teacher, or take an optimizer step.
"""
from __future__ import annotations

import argparse
import ast
import base64
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any, Sequence


DEFAULT_DATASET_ROOT = Path("/data/xvxinyao/Spin_test2/datasets/Spin_PINO_Rotation_Benchmark_v2")
EXPECTED_INDEX_SHA256 = "bafed1a784d66fa91f5044131f62abc2513e3cf7b6b594e9e196d960705501af"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 24 * 1024 * 1024
HANDOFF_FILENAME = "handoff.json"
EVIDENCE_FILES = (
    ("training_readiness_chat_review_v1/index.json", True, "json"),
    ("training_readiness_chat_review_v1/audit.json", True, "json"),
    ("ug_pino_107_faster_labels_v1/consumer/loader.py", True, "python"),
    ("ug_pino_107_faster_labels_v1/consumer/smoke.json", False, "json"),
    ("heart_fig7_development_audit_v1/benchmark_coverage.json", False, "json"),
)


class HandoffInspectionError(ValueError):
    """Evidence cannot be safely collected or does not match the pinned index."""


def _paths(dataset_root: str | Path, output: str | Path) -> tuple[Path, Path]:
    source_arg, output_arg = Path(dataset_root).expanduser(), Path(output).expanduser()
    if ".." in source_arg.parts or ".." in output_arg.parts:
        raise HandoffInspectionError("path traversal ('..') is not permitted")
    try:
        source = source_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HandoffInspectionError(f"dataset root does not exist: {source_arg}") from exc
    if not source.is_dir() or source == Path(source.anchor):
        raise HandoffInspectionError("dataset root must be a non-root directory")
    target = Path(os.path.abspath(output_arg))
    # Reject both existing outputs and dangling symlinks; no prior run is replaced.
    if os.path.lexists(target):
        raise HandoffInspectionError(f"output already exists; choose a new dedicated directory: {target}")
    if target.resolve(strict=False) != target:
        raise HandoffInspectionError("output path must not contain symlinks")
    if target == Path(target.anchor) or target.is_relative_to(source) or source.is_relative_to(target):
        raise HandoffInspectionError("dataset root and output must not overlap")
    return source, target


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or str(path) != relative or any(
        part in {".", ".."} for part in path.parts
    ):
        raise HandoffInspectionError(f"unsafe evidence path: {relative}")
    return path.parts


def _read_beneath(root_fd: int, relative: str, *, required: bool, remaining: int) -> bytes | None:
    """Use directory descriptors and O_NOFOLLOW to prevent symlink escapes."""
    parts = _relative_parts(relative)
    parent_fd = os.dup(root_fd)
    source_fd = None
    try:
        for part in parts[:-1]:
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        source_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise HandoffInspectionError(f"evidence must be a regular file: {relative}")
        if before.st_size > MAX_FILE_BYTES:
            raise HandoffInspectionError(f"evidence exceeds 8 MiB file limit: {relative}")
        if before.st_size > remaining:
            raise HandoffInspectionError("evidence exceeds 24 MiB total limit")
        limit = min(MAX_FILE_BYTES, remaining)
        pieces = []
        count = 0
        while True:
            piece = os.read(source_fd, min(65536, limit - count + 1))
            if not piece:
                break
            pieces.append(piece)
            count += len(piece)
            if count > limit:
                raise HandoffInspectionError(f"evidence grew beyond the size limit: {relative}")
        after = os.fstat(source_fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ) or count != after.st_size:
            raise HandoffInspectionError(f"evidence changed while being read: {relative}")
        return b"".join(pieces)
    except FileNotFoundError as exc:
        if required:
            raise HandoffInspectionError(f"required evidence is missing: {relative}") from exc
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise HandoffInspectionError(f"symlink or non-directory evidence path is forbidden: {relative}") from exc
        raise
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def _json_summary(payload: bytes, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise HandoffInspectionError(f"invalid JSON evidence: {relative}: {exc}") from exc
    if isinstance(value, dict):
        return {"root_type": "object", "top_level_keys": sorted(value)}
    if isinstance(value, list):
        return {"root_type": "array", "length": len(value)}
    kind = "null" if value is None else "boolean" if isinstance(value, bool) else (
        "number" if isinstance(value, (int, float)) else "string"
    )
    return {"root_type": kind}


def _consumer_interface(payload: bytes, relative: str) -> dict[str, Any]:
    try:
        tree = ast.parse(payload, filename=relative)
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise HandoffInspectionError(f"cannot parse consumer AST: {relative}: {exc}") from exc
    definitions = []

    def walk(node: ast.AST, scope: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.ClassDef) and node.name == "Faster107Corpus":
            methods = []
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                returns = ast.unparse(item.returns) if item.returns is not None else None
                type_params = getattr(item, "type_params", ())
                generic = "[" + ", ".join(ast.unparse(p) for p in type_params) + "]" if type_params else ""
                signature = ("async " if isinstance(item, ast.AsyncFunctionDef) else "") + item.name + generic
                signature += "(" + ast.unparse(item.args) + ")"
                if returns is not None:
                    signature += " -> " + returns
                methods.append({"name": item.name, "signature": signature,
                    "decorators": [ast.unparse(d) for d in item.decorator_list],
                    "docstring": ast.get_docstring(item, clean=False),
                    "line": item.lineno, "end_line": getattr(item, "end_lineno", None)})
            definitions.append({"name": node.name, "scope": ".".join(scope) or "module",
                "bases": [ast.unparse(base) for base in node.bases],
                "docstring": ast.get_docstring(node, clean=False), "line": node.lineno,
                "end_line": getattr(node, "end_lineno", None), "methods": methods})
        child_scope = scope + (node.name,) if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) else scope
        for child in ast.iter_child_nodes(node):
            walk(child, child_scope)

    walk(tree)
    return {"method": "ast_only_no_import_or_execution", "class_name": "Faster107Corpus",
            "definition_found": bool(definitions), "definitions": definitions,
            "runtime_export_verified": False}


def _encoded_content(payload: bytes) -> dict[str, str]:
    try:
        return {"content_encoding": "utf-8", "content": payload.decode("utf-8")}
    except UnicodeDecodeError:
        return {"content_encoding": "base64", "content": base64.b64encode(payload).decode("ascii")}


def _publish_new_directory(staged: Path, target: Path) -> None:
    """Linux atomic rename with no replacement, including concurrent creators."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise HandoffInspectionError("atomic no-replace directory publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise HandoffInspectionError(f"output already exists; nothing was replaced: {target}")
        raise OSError(code, os.strerror(code), str(target))


def collect_handoff(dataset_root: str | Path, output: str | Path) -> dict[str, Any]:
    """Copy verified small evidence to a new directory and return its bundle.

    The expected index hash is pinned in this module; there is no CLI override.
    Missing optional diagnostics are reported. Invalid optional JSON is kept as
    evidence with a parse error, while required JSON and loader syntax must parse.
    """
    source, target = _paths(dataset_root, output)
    root_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    rows = []
    raw_files: dict[str, bytes] = {}
    total = 0
    interface = None
    actual_index_hash = None
    try:
        for relative, required, kind in EVIDENCE_FILES:
            payload = _read_beneath(root_fd, relative, required=required, remaining=MAX_TOTAL_BYTES - total)
            row: dict[str, Any] = {"path": relative, "required": required, "exists": payload is not None}
            if payload is None:
                rows.append(row)
                continue
            total += len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            if relative == EVIDENCE_FILES[0][0]:
                actual_index_hash = digest
                if digest != EXPECTED_INDEX_SHA256:
                    raise HandoffInspectionError("index SHA-256 mismatch: expected "
                        f"{EXPECTED_INDEX_SHA256}, found {digest}; no output was written")
            row.update({"sha256": digest, "size_bytes": len(payload), **_encoded_content(payload)})
            if kind == "json":
                try:
                    row["json_summary"] = _json_summary(payload, relative)
                except HandoffInspectionError as exc:
                    if required:
                        raise
                    row["json_parse_error"] = str(exc)
            else:
                interface = _consumer_interface(payload, relative)
            rows.append(row)
            raw_files[relative] = payload
    finally:
        os.close(root_fd)
    bundle = {
        "schema": "tvcg.faster107.handoff-inspection.v1",
        "scope": "small_evidence_files_and_static_consumer_interface_only",
        "dataset_root": str(source), "expected_index_sha256": EXPECTED_INDEX_SHA256,
        "actual_index_sha256": actual_index_hash, "index_sha256_verified": True,
        "files": rows, "total_source_bytes": total, "consumer_interface": interface,
        "limits": {"per_file_bytes": MAX_FILE_BYTES, "total_bytes": MAX_TOTAL_BYTES},
        "inspection_complete": True, "integration_ready": False,
        "optimizer_steps": 0, "model_forward_calls": 0, "teacher_calls": 0,
        "consumer_executed": False, "referenced_dataset_payloads_read": False,
    }
    bundle_bytes = (json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # All required evidence and the pinned digest have passed before any writes.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve(strict=False) != target or os.path.lexists(target):
        raise HandoffInspectionError("output path changed or already exists; nothing was published")
    staged = Path(tempfile.mkdtemp(prefix="." + target.name + ".staging-", dir=target.parent))
    try:
        for relative, payload in raw_files.items():
            destination = staged.joinpath(*_relative_parts(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        with (staged / HANDOFF_FILENAME).open("xb") as handle:
            handle.write(bundle_bytes); handle.flush(); os.fsync(handle.fileno())
        _publish_new_directory(staged, target)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect pinned Faster107 handoff files without executing the consumer")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the dataset root")
    args = parser.parse_args(argv)
    try:
        bundle = collect_handoff(args.dataset_root, args.output)
    except (HandoffInspectionError, OSError) as exc:
        print(f"Handoff inspection failed: {exc}", file=sys.stderr)
        return 2
    print(f"Created inspection evidence: {Path(args.output).absolute() / HANDOFF_FILENAME}")
    print(f"Pinned index verified; {sum(row['exists'] for row in bundle['files'])} small files copied; "
          "optimizer_steps=0; integration_ready=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
