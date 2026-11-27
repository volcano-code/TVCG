"""Replay the SHA-bound Work verifier against its cached production extraction.

Only the original ``verify`` function is compiled, never Work's module-level
launcher. The inputs are archived numeric moments. This is not an independent
mesh evaluation or a deployed native performance measurement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import sympy

from . import acceptance


FIELDS = ("accepted", "objective", "constraints", "max_constraint", "raw_residual",
          "physical_relative_residual", "stationarity_residual", "mu", "mass",
          "center_of_mass", "I_xz", "I_yz", "max_transverse_com", "exact_ten_moments")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _ArchivedMoments:
    """Deliberate test double, not a native or learned geometry evaluator."""
    last_integral_evidence = {"archived_numeric_replay_only": True}

    def __init__(self, moments):
        self.moments = np.asarray(moments, dtype=np.float64)

    def __call__(self, q):
        return dict(raw_moments=self.moments.copy(), geometry_hash="archived-replay-only",
                    topology_hash="archived-replay-only")


def replay(legacy_source, replay_file=None):
    legacy_source = Path(legacy_source).resolve()
    raw = legacy_source.read_bytes()
    if _sha(raw) != acceptance.WORK_VERIFY_SHA256:
        raise ValueError("legacy source differs from the frozen Work verify profile")
    text = raw.decode("utf-8")
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == "verify"]
    if len(functions) != 1:
        raise ValueError("expected exactly one frozen Work verify function")
    function = functions[0]
    scope = {"np": np, "clock": time.perf_counter}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(legacy_source), "exec"), scope)
    fixture = Path(replay_file or Path(__file__).with_name("fixtures") / "work_acceptance_replay.json").resolve()
    data = json.loads(fixture.read_text())
    weight = data["beta"]
    rows = {}
    for name, case in data["cases"].items():
        previous = scope["verify"](_ArchivedMoments(case["moments"]), case["q"],
                                   SimpleNamespace(center_of_mass_weight=weight), "CONVERGED")
        current = acceptance.evaluate_moment_formula(case["moments"], case["q"], weight)
        equal = {key: bool(np.array_equal(previous[key], current[key])) for key in FIELDS}
        if not all(equal.values()) or current["accepted"] is not case["expected"]["accepted"]:
            raise AssertionError({"case": name, "field_equality": equal, "accepted": current["accepted"]})
        rows[name] = dict(query=case["q"], moments=case["moments"],
                          old={key: np.asarray(previous[key]).tolist() for key in FIELDS},
                          new={key: current[key] for key in FIELDS}, field_exact_equal=equal,
                          archived_raw_residual=case["expected"]["raw_residual"])
    return dict(schema="compact_pino.work_acceptance_formula_replay.v1", status="PASS",
                scope="archived numeric-moment formula equivalence; no live native mesh or GPU execution",
                legacy_source_path=str(legacy_source), legacy_source_sha256=_sha(raw),
                legacy_function_sha256=_sha(ast.get_source_segment(text, function).encode("utf-8")),
                legacy_function_ast_sha256=_sha(ast.dump(function, include_attributes=False).encode("utf-8")),
                replay_fixture_path=str(fixture), replay_fixture_sha256=_sha(fixture.read_bytes()),
                upstream_archive_member=data["source"], upstream_archive_member_sha256=data["source_sha256"],
                new_source_path=str(Path(acceptance.__file__).resolve()),
                new_source_sha256=_sha(Path(acceptance.__file__).read_bytes()),
                environment=dict(python=sys.version, numpy=np.__version__, sympy=sympy.__version__,
                                 system=platform.system(), machine=platform.machine()),
                center_of_mass_weight=weight, acceptance_profile=acceptance.ACCEPTANCE_PROFILE,
                equation_tolerance=acceptance.EQUATION_TOLERANCE,
                physical_relative_tolerance=acceptance.PHYSICAL_RELATIVE_TOLERANCE,
                compared_fields=list(FIELDS), cases=rows,
                interpretation="same-process old/new fields are exact; different LAPACK runtimes may change tiny residual roundoff")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--replay-fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.legacy_source, args.replay_fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
