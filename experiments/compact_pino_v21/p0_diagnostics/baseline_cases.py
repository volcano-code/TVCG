"""Frozen inputs for the unweighted V2.1 diagnostic and explicit synthetic fixtures.

``load_baseline_case`` accepts a JSON manifest with ``schema``, ``meta`` and
SHA256-addressed ``assets`` named geometry, material, operand and targets.
Material and operand use the repository's existing NPZ formats. Targets is an
NPZ containing JSON ``metadata``, ``coefficients``, ``fixed_moments``,
``native_moments`` (including fixed material), and ``native_full_editable_moments``.
Qualified derivatives additionally require ``native_jacobian``, ``chart_basis``
and positive ``parameter_scales``; an actual chart ``direction`` is optional.

The target metadata repeats LABEL_IDENTITY_FIELDS exactly. These checks bind
the recorded assets and conventions; they do not independently certify the
teacher, a derivative neighbourhood, mesh embedding, or geometric provenance.
Locked test/benchmark inputs are deliberately refused by this diagnostic.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "tvcg.v21.baseline-case.v1"
ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
COEFFICIENT_MULTIPLIERS = (1, 1, 1, 1, 1, 1, 1, 2, 2, 2)
DEGREE = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
PREFIX = "research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter"
LABEL_IDENTITY_FIELDS = (
    "geometry_id", "query_id", "source_group", "split", "teacher_version",
    "coordinate_frame", "units", "moment_order", "coefficient_basis", "coefficient_multipliers",
    "geometry_sha256", "material_identity", "operand_content_sha256",
    "query_content_sha256", "operation", "reference_kind", "moments_include_fixed",
    "derivative_valid", "derivative_dimension", "chart_id",
    "length_scale", "length_scale_definition", "editable_mass", "density",
)


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def moment_scale(mass, length):
    if not np.isfinite([mass, length]).all() or min(mass, length) <= 0:
        raise ValueError("positive editable mass and length required")
    return mass * length ** DEGREE


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite report value; withhold unavailable values explicitly")
    return value


def load_factory(name):
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("adapter must be module:factory")
    factory = getattr(importlib.import_module(module), attribute)
    if not callable(factory):
        raise ValueError("adapter factory is not callable")
    return factory


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {name}: lowercase SHA256 required")
    return value


def _array_digest(tag, *arrays):
    digest = hashlib.sha256(tag.encode("utf8"))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.dtype.str, array.shape], separators=(",", ":")).encode("ascii"))
        digest.update(memoryview(array).cast("B") if array.size else b"")
    return digest.hexdigest()


def operand_fingerprint(operand):
    """Content identity includes caps, operation and any supplied vertex motion."""
    velocity = (np.empty((0,), dtype="<f8") if operand.vertex_velocity is None
                else np.asarray(operand.vertex_velocity, dtype="<f8"))
    return _array_digest("finite-operand-v1:" + operand.operation,
                         np.asarray(operand.vertices, dtype="<f8"),
                         np.asarray(operand.faces, dtype="<i8"),
                         np.asarray(operand.face_is_cap, dtype="u1"), velocity)


def query_fingerprint(coefficients):
    return _array_digest("raw-query-v1", finite_array(coefficients, (10,), "coefficients").astype("<f8"))


def validate_meta(meta):
    for key in ("geometry_id", "query_id", "source_group", "teacher_version", "coordinate_frame", "units", "chart_id", "length_scale_definition"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"missing {key}")
    if meta.get("split") not in ("train", "development", "diagnostic", "synthetic"):
        raise ValueError("only train/development/diagnostic/synthetic inputs allowed; locked test is not inspected")
    if meta.get("reference_kind") not in ("deployed_native", "analytic_fixture"):
        raise ValueError("explicit deployed_native or analytic_fixture reference_kind required")
    if (meta["reference_kind"] == "analytic_fixture") != (meta["split"] == "synthetic"):
        raise ValueError("analytic fixtures must be explicitly labelled synthetic")
    if tuple(meta.get("moment_order", ())) != ORDER or meta.get("coefficient_basis") != "raw_coefficients":
        raise ValueError("moment order or raw coefficient convention mismatch")
    if tuple(meta.get("coefficient_multipliers", ())) != COEFFICIENT_MULTIPLIERS:
        raise ValueError("coefficient cross-term multipliers must be [1,1,1,1,1,1,1,2,2,2]")
    if meta.get("moments_include_fixed") is not True:
        raise ValueError("native_moments must explicitly include fixed material")
    for key in ("geometry_sha256", "material_identity", "operand_content_sha256", "query_content_sha256"):
        _digest(meta.get(key), key)
    if meta.get("operation") not in ("intersection", "difference"):
        raise ValueError("invalid operation")
    if type(meta.get("derivative_valid")) is not bool:
        raise ValueError("derivative_valid must be an explicit Boolean")
    dim = meta.get("derivative_dimension")
    if dim is not None and (type(dim) is not int or dim not in (3, 8, 10)):
        raise ValueError("derivative_dimension must be null, 3, 8 or 10")
    if meta["derivative_valid"] and dim is None:
        raise ValueError("qualified derivatives require their actual dimension")
    if not meta["derivative_valid"] and not str(meta.get("derivative_reason", "")).strip():
        raise ValueError("unqualified derivatives require an explicit reason")
    moment_scale(float(meta["editable_mass"]), float(meta["length_scale"]))
    if not np.isfinite(meta.get("density", np.nan)) or meta["density"] <= 0:
        raise ValueError("positive density required")


def _checked_fixed(value, scale):
    fixed = finite_array(value, (10,), "fixed_moments")
    if fixed[0] < 0 or (fixed[0] == 0 and np.any(fixed != 0)):
        raise ValueError("fixed material must have nonnegative mass and physically consistent zero mass")
    # A positive mass measure has a positive-semidefinite degree-two moment matrix.
    normalized = fixed / scale
    m, x, y, z, xx, yy, zz, yz, xz, xy = normalized
    matrix = np.array([[m, x, y, z], [x, xx, xy, xz], [y, xy, yy, yz], [z, xz, yz, zz]])
    if np.linalg.eigvalsh(matrix).min() < -1e-10 * max(1., np.linalg.norm(matrix)):
        raise ValueError("fixed moments are inconsistent with a nonnegative mass measure")
    return fixed


def _native_resolutions(values):
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(type(value) is not int or value not in (256, 512, 1024) for value in values)):
        raise ValueError("explicit distinct native material resolutions among 256/512/1024 required")
    return values


def validate_case(case, *, allowed_native_resolutions=(256,)):
    """Check loaded identity and qualification before any response evaluation."""
    meta = case["meta"]
    validate_meta(meta)
    for key in LABEL_IDENTITY_FIELDS:
        if key not in case["reference_meta"] or case["reference_meta"][key] != meta[key]:
            raise ValueError(f"reference identity mismatch: {key}")
    blocks, operand = case["blocks"], case["operand"]
    allowed = _native_resolutions(allowed_native_resolutions)
    if blocks.side not in allowed or not np.isfinite(blocks.pitch) or blocks.pitch <= 0:
        raise ValueError("native material resolution outside explicit admission contract or nonpositive pitch")
    # Legacy native256 manifests retain their exact interpretation. New native
    # resolutions additionally bind their dimension in the teacher metadata;
    # material hashes already cover origin, pitch, density and source geometry.
    if blocks.side != 256 or "native_material_resolution" in meta:
        if (type(meta.get("native_material_resolution")) is not int
                or meta["native_material_resolution"] != blocks.side
                or case["reference_meta"].get("native_material_resolution") != blocks.side):
            raise ValueError("native resolution must match material and teacher metadata")
    if blocks.identity != meta["material_identity"]:
        raise ValueError("material identity mismatch")
    if operand.operation != meta["operation"] or operand_fingerprint(operand) != meta["operand_content_sha256"]:
        raise ValueError("operand content identity mismatch")
    if query_fingerprint(case["coefficients"]) != meta["query_content_sha256"]:
        raise ValueError("query content identity mismatch")
    scale = moment_scale(float(meta["editable_mass"]), float(meta["length_scale"]))
    _checked_fixed(case["fixed_moments"], scale)
    finite_array(case["native_moments"], (10,), "native_moments")
    full = finite_array(case["native_full_editable_moments"], (10,), "native_full_editable_moments")
    if full[0] <= 0 or abs(full[0] / meta["editable_mass"] - 1) > 1e-12:
        raise ValueError("editable normalization mass must match full native editable mass")
    dim = meta["derivative_dimension"]
    if meta["derivative_valid"]:
        velocity = operand.vertex_velocity
        if velocity is None:
            raise ValueError("qualified derivatives require matching operand vertex motion")
        finite_array(velocity, (len(operand.vertices), 3, dim), "vertex_velocity")
        finite_array(case["native_jacobian"], (10, dim), "native_jacobian")
        chart = finite_array(case["chart_basis"], (10, dim), "chart_basis")
        if np.linalg.matrix_rank(chart) != dim or (dim == 10 and not np.array_equal(chart, np.eye(10))):
            raise ValueError("chart must be full rank; Full-J uses the raw identity chart")
        parameters = finite_array(case["parameter_scales"], (dim,), "parameter_scales")
        if np.any(parameters <= 0):
            raise ValueError("positive chart parameter scales required")
        if case.get("direction") is not None:
            finite_array(case["direction"], (dim,), "direction")
            if meta.get("direction_source") not in ("native_newton", "synthetic_fixture"):
                raise ValueError("direction source must be explicit")
            if meta["reference_kind"] != "analytic_fixture" and meta["direction_source"] != "native_newton":
                raise ValueError("production direction must come from native Newton")
    return scale


def verify_file_receipt(path, expected, *, cache=None, name="asset"):
    """Hash a frozen asset once per unchanged filesystem identity in this load.

    The optional caller-owned cache lives only for a snapshot loading session.
    A different inode, size, mtime or ctime forces another SHA256 check. No
    learned feature or parameter-dependent value belongs in this cache.
    """
    expected = _digest(expected, name + " sha256")
    path = Path(path).resolve(strict=True)
    def stamp():
        state = path.stat()
        return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
    before = stamp()
    key = (str(path), expected)
    if cache is None or cache.get(key) != before:
        if sha256(path) != expected:
            raise ValueError(f"{name} asset SHA256 mismatch")
        if stamp() != before:
            raise ValueError(f"{name} changed during SHA256 verification")
        if cache is not None:
            cache[key] = before
    return dict(path=str(path), sha256=expected, size_bytes=before[2])


def _checked_assets(spec, parent, verification_cache=None):
    paths, receipts = {}, {}
    for name in ("geometry", "material", "operand", "targets"):
        entry = spec["assets"][name]
        path = Path(entry["path"])
        path = path if path.is_absolute() else parent / path
        receipt = verify_file_receipt(path, entry.get("sha256"), cache=verification_cache, name=name)
        paths[name] = Path(receipt["path"])
        receipts[name] = receipt
    # Newly exported Work A handoffs retain their original qualification and
    # teacher evidence. Verify those optional, content-addressed artifacts too;
    # otherwise a valid case manifest could silently outlive a changed receipt.
    supporting = [(key, spec[key]) for key in ("source_qualification", "source_teacher_targets") if key in spec]
    supporting += [(f"source_evidence_{i}", entry) for i, entry in enumerate(spec.get("source_evidence", []))]
    for name, entry in supporting:
        path = Path(entry["path"])
        path = path if path.is_absolute() else parent / path
        receipts[name] = verify_file_receipt(path, entry.get("sha256"), cache=verification_cache, name=name)
    return paths, receipts


def audit_prepared_material(material, scale, material_cache_moments=None):
    """Audit sparse analytic runs and positive partial atoms without expanding cells.

    The full-cell component is an explicit union of nonoverlapping cuboids.
    Only partial-cell quadrature is stored. This replaces a packed-point audit
    for the opt-in sparse loader; it does not qualify any cut or derivative.
    """
    runs, cells, atoms = material.full_runs, material.partial_cells, material.boundary_nodes
    side = material.side
    if (runs.ndim != 2 or runs.shape[1] != 4 or runs.dtype.kind not in "iu"
            or cells.ndim != 2 or cells.shape[1] != 4 or cells.dtype.kind not in "iu"
            or atoms.shape != (int(cells[:,3].sum(dtype=np.int64)), 4)
            or not np.isfinite(atoms).all() or np.any(atoms[:,3] <= 0)):
        raise ValueError("invalid sparse material runs or positive partial atoms")
    if (np.any(runs[:,:3] < 0) or np.any(runs[:,:2] >= side) or np.any(runs[:,2] >= runs[:,3])
            or np.any(runs[:,3] > side) or np.any(cells[:,:3] < 0) or np.any(cells[:,:3] >= side)
            or np.any(cells[:,3] <= 0)):
        raise ValueError("sparse material indices escape the registered grid")
    run_keys = (runs[:,0].astype(np.int64) * side + runs[:,1]) * side + runs[:,2]
    order = np.argsort(run_keys, kind="stable")
    sorted_runs, sorted_keys = runs[order], run_keys[order]
    if len(runs) > 1:
        same_column = np.all(sorted_runs[1:,:2] == sorted_runs[:-1,:2], axis=1)
        if np.any(same_column & (sorted_runs[1:,2] < sorted_runs[:-1,3])):
            raise ValueError("overlapping analytic material runs")
    cell_keys = (cells[:,0].astype(np.int64) * side + cells[:,1]) * side + cells[:,2]
    if len(np.unique(cell_keys)) != len(cell_keys):
        raise ValueError("duplicate partial material cell")
    if len(runs) and len(cells):
        previous = np.searchsorted(sorted_keys, cell_keys, side="right") - 1
        valid = previous >= 0
        candidate = sorted_runs[previous[valid]]
        if np.any(np.all(candidate[:,:2] == cells[valid,:2], axis=1) & (cells[valid,2] < candidate[:,3])):
            raise ValueError("partial cells overlap full material runs")
    cumulative = np.cumsum(cells[:,3], dtype=np.int64)
    for start in range(0, len(atoms), 65536):
        end = min(start + 65536, len(atoms))
        owners = np.searchsorted(cumulative, np.arange(start, end), side="right")
        low = cells[owners,:3]
        tolerance = 128*np.finfo(float).eps*max(1, side)
        if np.any(atoms[start:end,:3] < low-tolerance) or np.any(atoms[start:end,:3] > low+1+tolerance):
            raise ValueError("partial material atom escaped its registered cell")
    full = np.asarray(material.moments(), dtype=float)
    if full.shape != (10,) or not np.isfinite(full).all() or full[0] <= 0:
        raise ValueError("nonempty finite sparse material required")
    error = None
    if material_cache_moments is not None:
        error = float(np.max(np.abs(full-np.asarray(material_cache_moments))/scale))
        if error > 1e-10:
            raise ValueError("sparse material changed after its static moment receipt")
    return dict(representation="analytic_runs_plus_positive_partial_atoms",
                full_volume_atoms_expanded=False, full_runs=len(runs), partial_cells=len(cells),
                stored_partial_atoms=len(atoms), cubature_moments=full.tolist(),
                cache_scaled_error=error,
                scope="material integrity and full moments only; cut accuracy is not certified")


def load_baseline_case(path, *, cache_directory=None, static_cache=None, asset_verification_cache=None,
                       allowed_native_resolutions=(256,), material_layout="blocks"):
    """Load existing physical assets; no Boolean or new derivative label is computed."""
    from importlib import import_module
    allowed = _native_resolutions(allowed_native_resolutions)
    if material_layout not in ("blocks", "prepared"):
        raise ValueError("material_layout must be blocks or prepared")
    started = time.perf_counter()
    path = Path(path).resolve(strict=True)
    spec = json.loads(path.read_text())
    if spec.get("schema") != SCHEMA:
        raise ValueError("unsupported baseline case schema")
    meta = spec["meta"]
    validate_meta(meta)  # Refuse locked inputs before opening target files.
    paths, receipts = _checked_assets(spec, path.parent, asset_verification_cache)
    if receipts["geometry"]["sha256"] != meta["geometry_sha256"]:
        raise ValueError("geometry content identity mismatch")
    operand = import_module(PREFIX + ".v21_native").FiniteOperand.load(paths["operand"])
    case = dict(meta=meta, operand=operand, asset_receipts=receipts,
                manifest_receipt=dict(path=str(path), sha256=sha256(path)))
    with np.load(paths["targets"], allow_pickle=False) as data:
        case["reference_meta"] = json.loads(str(data["metadata"].item()))
        for name in ("coefficients", "fixed_moments", "native_moments", "native_full_editable_moments"):
            case[name] = data[name].copy()
        if meta["derivative_valid"]:
            for name in ("native_jacobian", "chart_basis", "parameter_scales"):
                case[name] = data[name].copy()
            if "direction" in data:
                case["direction"] = data["direction"].copy()
    width = spec.get("block_voxels", 8)
    if type(width) is not int or width <= 0:
        raise ValueError("block_voxels must be a positive divisor of the grid side")
    directory = Path(cache_directory) if cache_directory is not None else path.parent / "p0-block-cache"
    # PreparedMaterial has no query-local packed-block directory. Including
    # that directory here would decompress the same high-resolution material
    # once per case before the manifest later aliases the objects.
    cache_key = ((receipts["material"]["sha256"], meta["material_identity"], material_layout)
                 if material_layout == "prepared" else
                 (receipts["material"]["sha256"], meta["material_identity"], width, str(directory.resolve()), material_layout))
    prepared = static_cache.get(cache_key) if static_cache is not None else None
    if prepared is None:
        material = import_module(PREFIX + ".v21_material").load_material(paths["material"], meta["material_identity"])
        if material.side not in allowed:
            raise ValueError("native material resolution outside explicit admission contract")
        if material.side % width:
            raise ValueError("block_voxels must divide the actual native material resolution")
        blocks = (material if material_layout == "prepared" else
                  import_module(PREFIX + ".v21_blocks").BlockMeasure.build(
                      material, directory / f"{material.identity}-b{width}", block_voxels=width))
        # One material object per verified static identity, or the legacy
        # memory-mapped packed blocks, is reused across query manifests.
        prepared = dict(blocks=blocks, moments=material.moments(), density=material.density)
        if static_cache is not None:
            static_cache[cache_key] = prepared
    if prepared["density"] != meta["density"]:
        raise ValueError("material density mismatch")
    case["blocks"] = prepared["blocks"]
    case["material_layout"] = material_layout
    case["material_cache_moments"] = prepared["moments"].copy()
    validate_case(case, allowed_native_resolutions=allowed)
    case["load_and_pack_seconds"] = time.perf_counter() - started
    return case


def write_synthetic_finite_case(directory):
    """Export one small cuboid/plane case, labelled synthetic at every boundary.

    Uses the real mesh-to-conservative-material builder (including partial
    cells), a finite closed operand with moving vertices, and an immutable
    surrounding shell. Analytic cuboid/planar boundary integrals are fixture
    references only: no deployed teacher, checkpoint or solver is invoked.
    Destination must be new, so no existing experiment can be overwritten.
    """
    from importlib import import_module
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    def box(lo, hi):
        vertices = np.array([[lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
                             [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
                             [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
                             [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]], dtype=float)
        faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                          [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return vertices, faces
    def integral(lo, hi, density):
        extent, center = hi-lo, (lo+hi)/2
        x,y,z = center
        values = np.array([1.,x,y,z,x*x,y*y,z*z,y*z,x*z,x*y])
        values[4:7] += extent**2/12
        return density*np.prod(extent)*values
    lo, hi = np.array([-.21,-.17,-.13]), np.array([.29,.23,.18])
    density, cut = 1.3, .037
    vertices, faces = box(lo, hi)
    geometry_path, material_path = directory/'geometry.npz', directory/'material.npz'
    operand_path, targets_path = directory/'operand.npz', directory/'targets.npz'
    np.savez(geometry_path, vertices=vertices, faces=faces)
    material = import_module(PREFIX+'.v21_material').prepare_material(
        vertices, faces, [-8.]*3, 1/16, side=256, density=density)
    material.save(material_path)
    vertices, faces = box([-3.]*3, [cut,3.,3.])
    moving = vertices[:,0] == cut
    velocity = np.zeros((8,3,3))
    velocity[moving,0,:] = np.column_stack((np.ones(moving.sum()), vertices[moving,1:3]))
    caps = np.ones(len(faces), dtype=bool); caps[6:8] = False
    operand = import_module(PREFIX+'.v21_native').FiniteOperand(vertices, faces, caps, 'intersection', velocity)
    operand.save(operand_path)
    coefficients = np.r_[-cut, 1., np.zeros(8)]
    full = integral(lo, hi, density)
    fixed = integral(lo-.1, hi+.1, density)-full
    retained_hi = hi.copy(); retained_hi[0] = cut
    native = fixed+integral(lo, retained_hi, density)
    # Section integrals of {1,y,z,yy,zz,yz,yyy,zzz,yyz,yzz}.
    def one(axis, power):
        return (hi[axis]**(power+1)-lo[axis]**(power+1))/(power+1)
    def section(py, pz):
        return density*one(1,py)*one(2,pz)
    powers = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),
              (0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0)]
    native_j = np.array([[cut**px*section(py+dy,pz+dz) for dy,dz in ((0,0),(1,0),(0,1))]
                         for px,py,pz in powers])
    chart = np.zeros((10,3)); chart[0,0] = chart[2,1] = chart[3,2] = -1.
    meta = dict(geometry_id='synthetic_partial_cell_box', query_id='finite_plane_0037',
                source_group='analytic_fixture', split='synthetic',
                teacher_version='independent-box-plane-integral-v1', coordinate_frame='fixture_physical_frame',
                units='m_kg', moment_order=list(ORDER), coefficient_basis='raw_coefficients',
                coefficient_multipliers=list(COEFFICIENT_MULTIPLIERS),
                geometry_sha256=sha256(geometry_path), material_identity=material.identity,
                operand_content_sha256=operand_fingerprint(operand),
                query_content_sha256=query_fingerprint(coefficients), operation='intersection',
                reference_kind='analytic_fixture', moments_include_fixed=True, derivative_valid=True,
                derivative_dimension=3, chart_id='raw-plane-offset-tilts', density=density,
                editable_mass=float(full[0]), length_scale=float(np.max(hi-lo)),
                length_scale_definition='editable_bbox_max_extent',
                direction_source='synthetic_fixture')
    np.savez(targets_path, metadata=np.asarray(json.dumps({key:meta[key] for key in LABEL_IDENTITY_FIELDS})),
             coefficients=coefficients, fixed_moments=fixed, native_moments=native,
             native_full_editable_moments=full, native_jacobian=native_j,
             chart_basis=chart, parameter_scales=np.array([meta['length_scale'],1.,1.]),
             direction=np.array([.02,.03,-.01]))
    spec = dict(schema=SCHEMA, meta=meta, block_voxels=8,
                assets={name:dict(path=path.name,sha256=sha256(path)) for name,path in
                        [('geometry',geometry_path),('material',material_path),
                         ('operand',operand_path),('targets',targets_path)]})
    path = directory/'case.json'
    path.write_text(json.dumps(spec, indent=2, allow_nan=False)+'\n')
    return path
