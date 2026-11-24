"""Independent fixed-rank paths, signed tetrahedra, and Cherchi construction correspondence."""
from dataclasses import dataclass
from hashlib import sha256, blake2b
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parents[1] / "native"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))
from replay import tetra_motion

PROTOCOL_FILE = HERE / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_FILE.read_bytes())
FD = PROTOCOL["central_difference"]
TETRA = PROTOCOL["base"]


def digest(a, dtype):
    return sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def block(q):
    return np.array([[q[4], q[9], q[8]], [q[9], q[5], q[7]], [q[8], q[7], q[6]]])


def rank_signature(q, surface):
    eigen = np.linalg.eigvalsh(block(q))
    tol = FD["numerical_rank_relative_tolerance"] * max(np.linalg.norm(q), np.finfo(float).tiny)
    neg = int(sum(eigen < -tol))
    pos = int(sum(eigen > tol))
    wanted = {"PLANE": 0, "PARABOLOID": 2}[surface]
    if neg + pos != wanted:
        raise ValueError(f"{surface} quadratic block has numerical inertia ({neg},{pos})")
    return (neg, pos)


def retracted(q, direction, step, surface):
    if step == 0:
        return q.copy()
    if surface == "PLANE":
        if np.any(q[4:]) or np.linalg.norm(direction[4:]) > FD["tangent_relative_tolerance"]:
            raise ValueError("plane path has quadratic curvature")
        return np.r_[q[:4] + step * direction[:4], q[4:]]
    q1 = q + step * direction
    eigen0 = np.linalg.eigvalsh(block(q))
    eigen, axes = np.linalg.eigh(block(q1))
    k = np.argmin(abs(eigen))
    change = eigen[k] - eigen0[np.argmin(abs(eigen0))]
    a = block(q1) - change * np.outer(axes[:, k], axes[:, k])
    return np.r_[q1[:4], a.diagonal(), a[1, 2], a[0, 2], a[0, 1]]


def chart_facts(q, chart):
    eigen, axes = np.linalg.eigh(block(q))
    inertia = rank_signature(q, chart.surface)
    null = axes[:, np.argmin(abs(eigen))]
    vals = (np.linalg.norm(chart.basis[4:], axis=0) if chart.surface == "PLANE" else
            np.array([abs(null @ block(direction) @ null) for direction in chart.basis.T]))
    if np.max(vals) > FD["tangent_relative_tolerance"] * max(1., np.linalg.norm(chart.basis)):
        raise ValueError("registered raw column leaves the numerical stratum")
    if not np.array_equal(retracted(q, chart.basis[:, 0], 0, chart.surface), q):
        raise ValueError("retraction moved the supplied center")
    return dict(quadratic_eigenvalues=eigen, quadratic_negative_positive=inertia,
                quadratic_determinant=float(np.linalg.det(block(q))), tangent_residual=vals,
                unit_sphere_tangent_residual=q @ chart.basis / np.linalg.norm(q))


def error_metric(estimate, reference, scope):
    delta = estimate - reference
    error = np.linalg.norm(delta)
    target = np.linalg.norm(reference)
    scale = max(target, np.linalg.norm(estimate))
    atol, rtol = FD[scope + "_fd_atol_norm"], FD[scope + "_fd_rtol_norm"]
    return dict(l2_error=float(error), reference_l2=float(target),
                max_abs=float(abs(delta).max(initial=0.)), relative_l2=float(error / max(scale, 1e-300)),
                tolerance=float(atol + rtol * scale), passed=bool(error <= atol + rtol * scale))


def tetra_pair(vertices, faces, velocity, density, fixed):
    """Sum independent closed signed-tetrahedron mass integrals and their product-rule derivative."""
    m = np.zeros(10)
    j = np.zeros((10, velocity.shape[2])) if velocity is not None else None
    for start in range(0, len(faces), 12000):
        batch = faces[start:start + 12000]
        t = vertices[batch]
        vol = density * np.einsum("ni,ni->n", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
        s = t.sum(axis=1)
        second = (s.T @ (s * vol[:, None]) + np.einsum("n,nti,ntj->ij", vol, t, t)) / 20
        m += np.r_[vol.sum(), vol @ s / 4, second.diagonal(), second[1, 2], second[0, 2], second[0, 1]]
        if j is not None:
            j += tetra_motion(vertices, batch, velocity, density)
    return m + fixed, j


def tetra_error(estimate, reference):
    return dict(max_abs=float(abs(estimate-reference).max()),
                passed=bool(np.allclose(estimate, reference, rtol=TETRA["tetra_elementwise_rtol"],
                                        atol=TETRA["tetra_elementwise_atol"])))


def sorted_faces(faces):
    return np.ascontiguousarray(faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))], dtype="<i8")


@dataclass
class Correspondence:
    keys: np.ndarray
    order: np.ndarray
    unique: bool
    constructions: str
    topology: str
    winding: str

    def summary(self):
        return dict(construction_sha256=self.constructions, unoriented_incidence_sha256=self.topology,
                    oriented_incidence_sha256=self.winding, unique_constructions=self.unique,
                    vertex_count=len(self.order))


def retain_keys(keys, faces):
    """Hash triangles after relabeling by unique construction, independent of output order."""
    keys = np.ascontiguousarray(keys, dtype="<u8")
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    unique = not np.any(np.all(ordered_keys[1:] == ordered_keys[:-1], axis=1))
    old_to_sorted = np.empty_like(order)
    old_to_sorted[order] = np.arange(len(order))
    f = old_to_sorted[np.asarray(faces)]
    unsigned = digest(sorted_faces(np.sort(f, axis=1)), "<i8")
    smallest = np.argmin(f, axis=1)
    oriented = np.take_along_axis(f, (smallest[:, None] + np.arange(3)) % 3, axis=1)
    return Correspondence(keys, order, unique, digest(ordered_keys, "<u8"), unsigned,
                          digest(sorted_faces(oriented), "<i8"))


def _identity(record, operand_key):
    """Canonical support identities for Cherchi's geometric construction (all inputs are indexed)."""
    kind = record["point_type"]
    pts = [operand_key(row) for row in record["operand_coordinates_exact"]]
    if kind == "LPI" and len(pts) == 5:
        ids = (*sorted(pts[:2]), *sorted(pts[2:]))
    elif kind == "TPI" and len(pts) == 9:
        planes = sorted(tuple(sorted(pts[k:k+3])) for k in (0, 3, 6))
        ids = tuple(x for plane in planes for x in plane)
    elif kind == "LNC" and len(pts) == 2:
        # Cherchi's LNC interpolation is oriented, including its parameter.
        parameter = float(record["parameter_t"])
        if pts[0] > pts[1]:
            pts.reverse()
            parameter = 1 - parameter
        ids = (*pts, struct.unpack("<Q", struct.pack("<d", parameter))[0])
    else:
        raise ValueError(f"unsupported retained construction: {kind} ({len(pts)})")
    return (kind, ids)


class PhysicalVertexIDs:
    """The exact binary64 positional lookup deliberately rejects any nonunique input origin."""
    def __init__(self, design, source):
        self.design = design
        self.source = {}
        for i, row in enumerate(source):
            k = tuple(row)
            self.source[k] = -1 if k in self.source or k in design else 2*i + 1

    @staticmethod
    def index_design(vertices):
        result = {}
        for i, row in enumerate(vertices):
            k = tuple(row)
            result[k] = -1 if k in result else 2*i
        return result

    def at(self, coord):
        key = tuple(coord)
        v = self.source[key] if key in self.source else self.design.get(key, -1)
        if v < 0:
            raise ValueError("missing or ambiguous retained input-vertex identity")
        return v

    def evaluate(self, packet, faces):
        from make_it_stand.verification.cherchi_inprocess import exact_fraction
        cache = {}
        def operand_key(tokens):
            key = tuple(tokens)
            if key not in cache:
                exact = tuple(map(exact_fraction, key))
                coord = tuple(map(float, exact))
                if any(t != x for t, x in zip(exact, coord)):
                    raise ValueError("Cherchi input support is not an original binary64 coordinate")
                cache[key] = self.at(coord)
            return cache[key]
        keys = np.empty((len(packet["vertices"]), 2), dtype="<u8")
        collision = {}
        for i, (p, rec) in enumerate(zip(packet["vertices"], packet["construction_records"], strict=True)):
            if rec["vertex_index"] != i:
                raise ValueError("non-indexed Cherchi construction receipt")
            if rec["point_type"] == "EXPLICIT3D":
                keys[i] = (0, self.at(p))
                continue
            value = _identity(rec, operand_key)
            token = blake2b(repr(value).encode("ascii"), digest_size=16).digest()
            a, b = struct.unpack("<QQ", token)
            code = (a | (1 << 63), b)
            if collision.setdefault(code, value) != value:
                raise ValueError("nonunique retained construction digest")
            keys[i] = code
        return retain_keys(keys, faces)


def compare_retained(base, minus, plus, vm, vp, velocity, h):
    sets = (base, minus, plus)
    if not all(x.unique for x in sets):
        return dict(complete=False, n_matched=0, reason="ambiguous construction identities")
    same_set = all(x.constructions == base.constructions for x in sets)
    same_faces = same_set and all(x.topology == base.topology for x in sets)
    same_winding = same_faces and all(x.winding == base.winding for x in sets)
    if same_set:
        ib, im, ip = (x.order for x in sets)
    else:
        def typed(x):
            return x.keys.view(np.dtype([("class", "<u8"), ("id", "<u8")])).reshape(-1)
        common, b0, m0 = np.intersect1d(typed(base), typed(minus), return_indices=True)
        _, bm, ip = np.intersect1d(common, typed(plus), return_indices=True)
        ib, im = b0[bm], m0[bm]
    out = dict(complete=bool(same_faces and same_winding), same_construction_set=same_set,
               same_unoriented_incidence=same_faces, same_oriented_incidence=same_winding,
               n_matched=len(ib), n_center=len(base.order))
    if len(ib) and velocity is not None:
        out["velocity"] = error_metric((vp[ip] - vm[im]) / (2*h), velocity[ib], "retained")
    return out
