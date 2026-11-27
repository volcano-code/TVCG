"""Pre-Boolean finite operands from the deployed triangulator in solver target coordinates.

The geometry-only NPZ format carries the operation and (when available) a
vertex differential with respect to raw coefficients or raw chart coordinates.
There are no Boolean results, training labels or material integrals in an operand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


@dataclass(frozen=True)
class FiniteOperand:
    vertices: np.ndarray
    faces: np.ndarray
    face_is_cap: np.ndarray
    operation: str
    vertex_velocity: np.ndarray | None = None

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces)
        caps = np.asarray(self.face_is_cap)
        if self.operation not in ("intersection", "difference"):
            raise ValueError("invalid Boolean operation")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("operand vertices must be a finite (Nv, 3) array")
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4 or
                faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("operand faces must be valid triangular indices")
        if caps.shape != (len(faces),) or caps.dtype.kind not in 'biu' or not np.isin(caps, [0, 1]).all():
            raise ValueError("operand cap mask must be one Boolean per face")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", np.asarray(faces, dtype=np.int64))
        object.__setattr__(self, "face_is_cap", caps.astype(bool, copy=False))
        if self.vertex_velocity is not None:
            velocity = np.asarray(self.vertex_velocity, dtype=np.float64)
            if (velocity.ndim != 3 or velocity.shape[:2] != vertices.shape or
                    velocity.shape[2] not in (3, 8, 10) or not np.isfinite(velocity).all()):
                raise ValueError("vertex velocity needs 3, 8 or 10 finite derivative columns")
            object.__setattr__(self, "vertex_velocity", velocity)

    def save(self, path: Path | str):
        arrays = dict(vertices=self.vertices, faces=self.faces, face_is_cap=self.face_is_cap,
                      operation=np.asarray(self.operation))
        if self.vertex_velocity is not None:
            arrays["vertex_velocity"] = self.vertex_velocity
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stage:
            temporary = Path(stage.name)
            try:
                np.savez_compressed(stage, **arrays)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> FiniteOperand:
        with np.load(path, allow_pickle=False) as data:
            velocity = data["vertex_velocity"] if "vertex_velocity" in data else None
            return cls(data["vertices"], data["faces"], data["face_is_cap"],
                       str(data["operation"].item()), velocity)


@dataclass(frozen=True)
class RawChart:
    """Tangent directions dq_raw/ds for solver-target q, in the user's column order."""

    surface: str
    basis: np.ndarray

    def __post_init__(self):
        dim = {"PLANE": 3, "PARABOLOID": 8}.get(self.surface)
        basis = np.asarray(self.basis, dtype=np.float64)
        if (dim is None or basis.shape != (10, dim) or not np.isfinite(basis).all()
                or np.linalg.matrix_rank(basis) != dim):
            raise ValueError("expected a full-column-rank PLANE/R3 or PARABOLOID/R8 raw chart")
        object.__setattr__(self, "basis", basis)


def _validated_raw_query(query):
    q = np.asarray(query, dtype=np.float64)
    if q.shape != (10,) or not np.isfinite(q).all():
        raise ValueError("raw query needs ten finite coefficients")
    norm = float(np.linalg.norm(q))
    if norm == 0. or not np.isfinite(norm):
        raise ValueError("raw query cannot be normalized")
    return q, norm


def unit_to_raw_jacobian(query) -> np.ndarray:
    q, norm = _validated_raw_query(query)
    u = q / norm
    return (np.eye(10) - np.outer(u, u)) / norm


def operand_from_value_source(source, query, *, chart: RawChart | None = None) -> FiniteOperand:
    """Decode a finite value without inventing motion or derivative qualification.

    A reduced, versioned value source still needs its explicitly supplied chart.
    A valid geometry at a derivative seam does not grant a Jacobian there.
    """
    _validated_raw_query(query)
    if not source["defined"]:
        raise ValueError("source undefined: " + str(source.get("diagnostic", "")))
    if not source["side_selection_defined"]:
        raise ValueError("side selector undefined: " + str(source.get("side_selection_diagnostic", "")))
    cap = np.asarray(source["face_is_cap"]).reshape(-1)
    if np.count_nonzero(cap) != int(source["cap_face_count"]):
        raise ValueError("native cap count and face identifiers disagree")
    if chart is not None:
        if source["surface_type"] != chart.surface:
            raise ValueError("requested chart differs from native source surface")
    elif "coefficient_basis" in source or source.get("source_map_version") == "ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1":
        raise ValueError("a reduced source needs explicitly typed raw chart coordinates")
    operation = "intersection" if source["interior_is_positive"] else "difference"
    return FiniteOperand(source["vertices"], source["faces"], cap, operation)


def operand_from_source(source, query, *, chart: RawChart | None = None,
                        with_jacobian: bool = True) -> FiniteOperand:
    """Apply coefficient normalization exactly once when motion is requested."""
    if type(with_jacobian) is not bool:
        raise ValueError("with_jacobian must be boolean")
    value = operand_from_value_source(source, query, chart=chart)
    if not with_jacobian:
        return value
    vertices = np.asarray(source["vertices"], dtype=np.float64)
    transform = unit_to_raw_jacobian(query)
    dim = 10
    if chart is not None:
        if source["surface_type"] != chart.surface:
            raise ValueError("requested chart differs from native source surface")
        dim = chart.basis.shape[1]
        native_basis = np.asarray(source["coefficient_basis"], dtype=np.float64)
        if (native_basis.shape != (10, dim) or not np.isfinite(native_basis).all() or
                not np.allclose(native_basis.T @ native_basis, np.eye(dim), atol=1e-10, rtol=0)):
            raise ValueError("invalid native unit tangent basis")
        unit_directions = transform @ chart.basis
        transform = native_basis.T @ unit_directions
        residual = native_basis @ transform - unit_directions
        if (np.linalg.norm(residual) > 1e-9 * max(np.linalg.norm(unit_directions), np.finfo(float).tiny) or
                np.linalg.matrix_rank(transform) != dim):
            raise ValueError("raw chart does not span the native unit tangent space")
    elif "coefficient_basis" in source:
        raise ValueError("a reduced source needs explicitly typed raw chart coordinates")
    velocity = np.asarray(source["vertex_velocity"], dtype=np.float64).reshape(len(vertices), 3, dim)
    return FiniteOperand(vertices, value.faces, value.face_is_cap, value.operation, velocity @ transform)


def physical_operand_vertices(operand, target_axes, target_origin, length, normalization_origin):
    """The binary64 vertex coordinates actually submitted to the production Boolean."""
    return ((operand.vertices @ np.asarray(target_axes).T + target_origin) *
            float(length) + normalization_origin)


def audit_physical_source(native, operand, target_axes, target_origin, length, normalization_origin):
    """Run production's exact physical-source boundary check without performing a Boolean."""
    vertices = physical_operand_vertices(operand,target_axes,target_origin,length,normalization_origin)
    facts = dict(native.audit_exact_triangle_boundary(vertices,np.asarray(operand.faces,dtype=np.int32),'AnyNonzero'))
    if not facts['boundary_valid']:
        raise ValueError(f"physical source boundary invalid: {facts['input_digest']}")
    return facts


def native_operand_capabilities(*, chart: RawChart | None = None, source_backend=None):
    """Describe actual source work; the deployed full-rank API is motion-only.

    ``source_backend`` is an explicitly installed/verified rank-chart extension
    (the existing ``NativeSource`` service), never an automatically substituted
    map. Its build/profile checks remain the caller's responsibility.
    """
    if source_backend is not None and chart is None:
        raise ValueError("a source backend requires an explicitly typed raw chart")
    value_available = source_backend is not None and callable(getattr(source_backend, "source_value", None))
    return dict(value_source_available=value_available,
                moment_only_source_mode="typed_value" if value_available else "motion_fallback",
                source_motion_avoided=value_available)


def native_operand(evaluator, query, *, chart: RawChart | None = None,
                   with_jacobian: bool = True, source_backend=None) -> FiniteOperand:
    """Construct a pre-Boolean operand, optionally omitting unneeded motion.

    Default calls retain the deployed moving-source path. The registered
    R3/R8 extension has a genuine ``source_value`` entry; pass it explicitly to
    use that path. Without it, a value-only request still pays native source
    motion construction, but no motion is attached to the returned operand.
    A source rejection is never caught and retried with a different map.
    """
    if type(with_jacobian) is not bool:
        raise ValueError("with_jacobian must be boolean")
    capabilities = native_operand_capabilities(chart=chart, source_backend=source_backend)
    q, _ = _validated_raw_query(query)
    if source_backend is not None:
        if not evaluator.triangulation.algebraic_finite_map:
            raise ValueError("typed source requires the registered algebraic finite map")
        source_motion = (source_backend.source_value
                         if not with_jacobian and capabilities["value_source_available"]
                         else source_backend.source_motion)
    elif chart is None:
        source_motion = evaluator.native.evaluate_spin_it_faster_source_motion
    else:
        source_motion = evaluator.native.evaluate_spin_it_faster_stratum_source_motion
    source = source_motion(q, evaluator.target_design, evaluator.triangulation)
    operand = operand_from_source(source, q, chart=chart, with_jacobian=with_jacobian)
    audit_physical_source(evaluator.native,operand,evaluator.request.target_axes,evaluator.request.target_origin,
                          evaluator.length,evaluator.normalization_origin)
    return operand
