"""R3.1.1 native build preflight for package and top-level pybind layouts."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "tvcg.r3.native-build-preflight.v1"
REQUIRED_NATIVE_SYMBOLS = ("solve_spin_it_faster_fullrank_quadric",)
REQUIRED_FACADE_SYMBOLS = (
    "SpinItFasterRequest",
    "SpinSolverConfig",
    "SpinEvaluatorConfig",
    "solve_spin_it_faster_fullrank_quadric",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_cmake_cache(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    result = {}
    for line in p.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line:
            continue
        left, value = line.split("=", 1)
        result[left.split(":", 1)[0]] = value
    return result

def inspect_native_module(module: Any) -> dict:
    missing = [name for name in REQUIRED_NATIVE_SYMBOLS if not hasattr(module, name)]
    module_file = getattr(module, "__file__", None)
    return {
        "module_name": getattr(module, "__name__", type(module).__name__),
        "module_file": str(module_file) if module_file else None,
        "module_file_sha256": sha256(Path(module_file))
            if module_file and Path(module_file).is_file() else None,
        "required_symbols": list(REQUIRED_NATIVE_SYMBOLS),
        "missing_symbols": missing,
        "native_api_available": not missing,
    }

def run_preflight(*, build_dir=None, expected_commit=None, module_name="auto"):
    candidates = (
        ("make_it_stand._make_it_stand", "_make_it_stand")
        if module_name == "auto" else (module_name,)
    )
    module = None
    import_errors = []
    selected_name = None
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
            selected_name = candidate
            break
        except Exception as exc:
            import_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if module is None:
        return {
            "schema": SCHEMA,
            "ok": False,
            "native_api_available": False,
            "module_name": module_name,
            "import_errors": import_errors,
            "build_configuration": {},
            "checks": {"extension_import": False},
        }

    info = inspect_native_module(module)
    try:
        facade = importlib.import_module("make_it_stand.spin")
        facade_missing = [
            name for name in REQUIRED_FACADE_SYMBOLS if not hasattr(facade, name)
        ]
        facade_info = {
            "module_file": str(getattr(facade, "__file__", None)),
            "missing_symbols": facade_missing,
            "available": not facade_missing,
        }
    except Exception as exc:
        facade_info = {
            "module_file": None,
            "missing_symbols": list(REQUIRED_FACADE_SYMBOLS),
            "available": False,
            "import_error": f"{type(exc).__name__}: {exc}",
        }

    config = {}
    cache_path = None
    if build_dir:
        candidate = Path(build_dir) / "CMakeCache.txt"
        cache_path = str(candidate)
        if candidate.is_file():
            cache = parse_cmake_cache(candidate)
            config = {
                k: cache.get(k) for k in (
                    "MAKE_IT_STAND_BUILD_PYTHON",
                    "MAKE_IT_STAND_BUILD_SPIN_IT_FASTER",
                    "CMAKE_BUILD_TYPE",
                    "CMAKE_CXX_COMPILER",
                    "Python_EXECUTABLE",
                    "pybind11_DIR",
                ) if k in cache
            }
        else:
            config = {"cmake_cache_missing": True}

    checks = {
        "extension_import": True,
        "selected_module_name": selected_name,
        "required_native_symbols": info["native_api_available"],
        "python_facade_available": facade_info["available"],
        "python_build_enabled": (
            config.get("MAKE_IT_STAND_BUILD_PYTHON") == "ON" if config else None
        ),
        "spin_it_faster_enabled": (
            config.get("MAKE_IT_STAND_BUILD_SPIN_IT_FASTER") == "ON" if config else None
        ),
        "expected_commit_match": None,
    }
    if expected_commit is not None:
        actual = os.environ.get("TVCG_SOURCE_COMMIT")
        checks["expected_commit_match"] = actual == expected_commit if actual else False

    hard = [
        checks["extension_import"],
        checks["required_native_symbols"],
        checks["python_facade_available"],
    ]
    if config and not config.get("cmake_cache_missing"):
        hard.extend([
            checks["python_build_enabled"],
            checks["spin_it_faster_enabled"],
        ])
    if expected_commit is not None:
        hard.append(checks["expected_commit_match"])

    return {
        "schema": SCHEMA,
        "ok": all(value is True for value in hard),
        "native_api_available": info["native_api_available"],
        "module": info,
        "python_facade": facade_info,
        "build_dir": str(build_dir) if build_dir else None,
        "cmake_cache": cache_path,
        "build_configuration": config,
        "checks": checks,
        "scope": "native extension preflight only; no geometry admission or solver replay",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir")
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--module", default="auto",
        help="module name, or auto for package/top-level layouts",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_preflight(
        build_dir=args.build_dir,
        expected_commit=args.expected_commit,
        module_name=args.module,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 2)

if __name__ == "__main__":
    main()
