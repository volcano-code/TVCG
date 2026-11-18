"""Phase 2.6 R3.0 problem-reconstruction contract."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
from typing import Any

SCHEMA = "tvcg.r3.problem-reconstruction.v1"
QUADRIC_DIM = 10

def _finite_vec(value: Any, n: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{name} must be a length-{n} sequence")
    out = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{name} must contain finite values")
    return out

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass(frozen=True)
class R3ProblemCase:
    case_id: str
    original_mesh: str
    original_mesh_sha256: str
    fixed_shell: str
    fixed_shell_sha256: str
    design_domain: str
    design_domain_sha256: str
    axis: tuple[float, float, float]
    axis_origin: tuple[float, float, float]
    contact: tuple[float, float, float]
    pin_support: str
    initial_material_policy: str
    quadric_coefficients: tuple[float, ...]
    quadric_initialization_kind: str
    quadric_source: str

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "R3ProblemCase":
        if record.get("schema") != SCHEMA:
            raise ValueError("unsupported R3 problem-reconstruction schema")
        required = [
            "case_id", "original_mesh", "original_mesh_sha256",
            "fixed_shell", "fixed_shell_sha256",
            "design_domain", "design_domain_sha256",
            "axis", "axis_origin", "contact", "pin_support",
            "initial_material_policy", "quadric_coefficients",
            "quadric_initialization_kind", "quadric_source",
        ]
        missing = [k for k in required if k not in record]
        if missing:
            raise ValueError("incomplete R3 case: " + ", ".join(missing))
        case_id = str(record["case_id"]).strip()
        if not case_id:
            raise ValueError("case_id must be non-empty")
        for name in ("original_mesh_sha256", "fixed_shell_sha256", "design_domain_sha256"):
            value = str(record[name]).lower()
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        coeff = _finite_vec(record["quadric_coefficients"], QUADRIC_DIM, "quadric_coefficients")
        if math.sqrt(sum(x * x for x in coeff)) <= 1e-15:
            raise ValueError("quadric_coefficients must be nonzero")
        for field in ("original_mesh", "fixed_shell", "design_domain", "pin_support",
                      "initial_material_policy", "quadric_initialization_kind", "quadric_source"):
            if not str(record[field]).strip():
                raise ValueError(f"{field} must be non-empty")
        return cls(
            case_id=case_id,
            original_mesh=str(record["original_mesh"]),
            original_mesh_sha256=str(record["original_mesh_sha256"]).lower(),
            fixed_shell=str(record["fixed_shell"]),
            fixed_shell_sha256=str(record["fixed_shell_sha256"]).lower(),
            design_domain=str(record["design_domain"]),
            design_domain_sha256=str(record["design_domain_sha256"]).lower(),
            axis=_finite_vec(record["axis"], 3, "axis"),
            axis_origin=_finite_vec(record["axis_origin"], 3, "axis_origin"),
            contact=_finite_vec(record["contact"], 3, "contact"),
            pin_support=str(record["pin_support"]),
            initial_material_policy=str(record["initial_material_policy"]),
            quadric_coefficients=coeff,
            quadric_initialization_kind=str(record["quadric_initialization_kind"]),
            quadric_source=str(record["quadric_source"]),
        )

    def validate_files(self, root: str | Path) -> None:
        root = Path(root)
        for path_field, hash_field in (
            ("original_mesh", "original_mesh_sha256"),
            ("fixed_shell", "fixed_shell_sha256"),
            ("design_domain", "design_domain_sha256"),
        ):
            path = root / getattr(self, path_field)
            if not path.is_file():
                raise FileNotFoundError(f"{path_field} not found: {path}")
            actual = _sha256_file(path)
            expected = getattr(self, hash_field)
            if actual != expected:
                raise ValueError(f"{path_field} SHA256 mismatch: expected {expected}, got {actual}")

    def physical_initial_state(self) -> dict[str, Any]:
        return {
            "original_mesh": self.original_mesh,
            "fixed_shell": self.fixed_shell,
            "design_domain": self.design_domain,
            "axis": list(self.axis),
            "axis_origin": list(self.axis_origin),
            "contact": list(self.contact),
            "pin_support": self.pin_support,
            "initial_material_policy": self.initial_material_policy,
        }

    def numerical_initial_state(self) -> dict[str, Any]:
        return {
            "quadric_coefficients": list(self.quadric_coefficients),
            "initialization_kind": self.quadric_initialization_kind,
            "source": self.quadric_source,
        }

def load_case(path: str | Path) -> R3ProblemCase:
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict):
        raise ValueError("R3 manifest must be a JSON object")
    return R3ProblemCase.from_dict(record)

def admission_report(case: R3ProblemCase) -> dict[str, Any]:
    return {
        "schema": "tvcg.r3.admission-report.v1",
        "case_id": case.case_id,
        "physical_initial_state_bound": True,
        "numerical_initial_state_bound": True,
        "geometry_sha256_bound": True,
        "semantic_inference_used": False,
        "physical_admission": False,
        "reason": "file existence/hash and semantic audit must pass before admission",
    }
