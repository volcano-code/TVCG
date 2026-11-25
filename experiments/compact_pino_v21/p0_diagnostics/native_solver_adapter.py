"""Executable adapter to the repository's native spherical Eq.14 solver.

This profile is NOT the private 17:56/noodles ``solve.py``. It requires its own
archived native replay, unit geometry coefficients and a hash-bound native
kernel. The shared library compiles the existing C++ source without replacing
its formulas. Full-J one-step diagnostics execute its actual Newton and native
backtracking. R3/R8 steps are rejected until a supported-space adapter exists;
no directional derivative is padded into an ambient Jacobian.

``factory`` is an endpoint_attribution adapter. ``NativeKernel`` is also usable
for explicitly declared numerical fixtures; that API confers no mesh proof,
teacher admission, training promotion or equivalence to a different solver.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re

import numpy as np

from .endpoint_attribution import finite_array, sha256, verify_artifacts, verify_provenance

PROFILE = "TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1"
GEOMETRY_PROFILE = "TVCG_REPO_CGAL_CANONICAL_FULLRANK_EXACT_BINARY64_V1"
SYNTHETIC_GEOMETRY_PROFILE = "MANUFACTURED_NUMERICAL_MAP_V1"
STATE_SCHEMA = "cpino_repo_native_state_v1"
CONVENTION = "dM_d_unnormalized_geometry_at_unit_v1"
MODES = {"STATIC_MAXIMUM_MASS": 0, "STATIC_LOWER_CENTER": 1,
         "STATIC_PARETO": 2, "FIDGET_SPINNER": 3, "YOYO": 4,
         "SPINNING_TOP": 5, "SPINNING_TOP_LOWER_CENTER": 6}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


class NativeAdapterError(ValueError):
    """Identity or native execution failure, never an automatic GO."""


class _Packet(ctypes.Structure):
    _fields_ = [("moments", ctypes.c_double * 10),
                ("jacobian", ctypes.c_double * 100),
                ("derivative_defined", ctypes.c_int),
                ("geometry_hash", ctypes.c_char * 129),
                ("topology_hash", ctypes.c_char * 129),
                ("diagnostic", ctypes.c_char * 1024),
                ("evaluation_map_identity", ctypes.c_char * 129),
                ("boolean_backend_identity", ctypes.c_char * 513),
                ("derivative_identity", ctypes.c_char * 129)]


_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _DOUBLE_PTR,
                           ctypes.POINTER(_Packet), ctypes.c_void_p)


def _unit_coefficients(c):
    c = finite_array(c, (10,), "native unit geometry coefficients")
    if abs(np.linalg.norm(c) - 1.) > 1e-12:
        raise NativeAdapterError("profile requires saved unit coefficients; no silent normalization")
    return np.ascontiguousarray(c, dtype=np.float64)


def _sha(value, label):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise NativeAdapterError(f"{label} requires a lowercase SHA-256")
    return value


def _packet(m, j, *, geometry_hash, topology_hash, defined=True, diagnostic="",
            evaluation_map_identity="UNSEALED_NUMERICAL_INPUT", boolean_backend_identity="UNSEALED_NUMERICAL_INPUT",
            derivative_identity="UNSEALED_NUMERICAL_INPUT"):
    if type(defined) is not bool:
        raise NativeAdapterError("derivative_defined must be an explicit bool")
    packet = _Packet()
    packet.moments[:] = finite_array(m, (10,), "native trial moments")
    if defined:
        packet.jacobian[:] = finite_array(j, (10, 10), "native Full-J").ravel()
    elif j is not None:
        raise NativeAdapterError("undefined derivative must be null, not a zero-filled matrix")
    packet.derivative_defined = int(defined)
    packet.geometry_hash = _sha(geometry_hash, "geometry_hash").encode("ascii")
    packet.topology_hash = _sha(topology_hash, "topology_hash").encode("ascii")
    packet.diagnostic = str(diagnostic).encode("utf-8")[:1023]
    for key, limit in (("evaluation_map_identity", 128), ("boolean_backend_identity", 512), ("derivative_identity", 128)):
        value = locals()[key]
        if not isinstance(value, str) or not value or "\x00" in value or len(value.encode("utf-8")) > limit:
            raise NativeAdapterError("invalid " + key)
        setattr(packet, key, value.encode("utf-8"))
    return packet


def verify_kernel_build(library, *, receipt=None, expected_sha256=None):
    """Bind the binary and all recorded repository headers/source files.

    Third-party header digests remain in the build receipt for reproducibility;
    they are not required to be installed when executing a copied binary.
    """
    library = Path(library).resolve()
    receipt = Path(receipt).resolve() if receipt else library.with_suffix(library.suffix + ".build.json")
    data = json.loads(receipt.read_text())
    if data.get("schema") != "cpino_native_numeric_build_v1" or data.get("profile") != PROFILE:
        raise NativeAdapterError("incompatible native build receipt/profile")
    actual = sha256(library)
    if actual != data.get("kernel_sha256") or (expected_sha256 is not None and actual != expected_sha256):
        raise NativeAdapterError("native kernel binary checksum mismatch")
    source_root = Path(__file__).resolve().parents[3]
    sources = data.get("source_sha256", {})
    if "src/spin/QuadricHollowingSolver.cpp" not in sources:
        raise NativeAdapterError("build receipt does not bind the actual native solver source")
    for relative, expected in sources.items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or sha256(path) != _sha(expected, "source checksum"):
            raise NativeAdapterError(f"native source mismatch: {relative}")
    return data


class NativeKernel:
    """C ABI numerical kernel; native M-only acceptance and one actual step."""

    def __init__(self, library, *, objective, thresholds, max_evaluations=60,
                 build_receipt=None, expected_sha256=None):
        self.build = verify_kernel_build(library, receipt=build_receipt, expected_sha256=expected_sha256)
        objective = copy.deepcopy(objective)
        if objective.get("mode") not in MODES:
            raise NativeAdapterError("unknown native objective mode")
        self.mode = MODES[objective["mode"]]
        self.weight = float(objective.get("weight", 0.))
        self.mass_scale = float(objective.get("mass_scale", 1.))
        if not np.isfinite([self.weight, self.mass_scale]).all() or self.mass_scale <= 0:
            raise NativeAdapterError("invalid native objective settings")
        self.equation = float(thresholds["stationarity"])
        self.balance = float(thresholds["balance"])
        if not np.isfinite([self.equation, self.balance]).all() or min(self.equation, self.balance) <= 0:
            raise NativeAdapterError("invalid native acceptance thresholds")
        if type(max_evaluations) is not int or not 2 <= max_evaluations <= 60:
            raise NativeAdapterError("one-step budget must be 2..60 evaluations including center")
        self.max_evaluations = max_evaluations
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.cpino_native_identity.restype = ctypes.c_char_p
        if self.library.cpino_native_identity().decode() != PROFILE:
            raise NativeAdapterError("loaded native library profile mismatch")
        self.library.cpino_native_accept.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                    _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_double, ctypes.c_double]
        self.library.cpino_native_accept.restype = ctypes.c_char_p
        self.library.cpino_native_step.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                  _DOUBLE_PTR, ctypes.POINTER(_Packet),
                                                  ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                                  _CALLBACK, ctypes.c_void_p]
        self.library.cpino_native_step.restype = ctypes.c_char_p

    @staticmethod
    def _result(raw):
        value = json.loads(raw)
        if "error" in value:
            raise NativeAdapterError(value["error"])
        return value

    def accept(self, c, moments, jacobian=None):
        # J intentionally is never read, validated, transformed or used here.
        c = _unit_coefficients(c)
        m = np.ascontiguousarray(finite_array(moments, (10,), "moments"))
        return self._result(self.library.cpino_native_accept(
            self.mode, self.weight, self.mass_scale, c.ctypes.data_as(_DOUBLE_PTR),
            m.ctypes.data_as(_DOUBLE_PTR), self.equation, self.balance))

    def step(self, c, moments, jacobian, evaluator, *, geometry_hash, topology_hash):
        c = _unit_coefficients(c)
        center_identity=hashlib.sha256(np.asarray(moments,dtype="<f8").tobytes()+
                                      np.asarray(jacobian,dtype="<f8").tobytes()).hexdigest()
        center = _packet(moments, jacobian, geometry_hash=geometry_hash, topology_hash=topology_hash,
                         evaluation_map_identity="ATTRIBUTION_FIXED_CENTER_MJ_V1:"+center_identity,
                         boolean_backend_identity="CENTER_SUBSTITUTION_NO_NEW_BOOLEAN",
                         derivative_identity="ATTRIBUTION_CENTER_FULL_J_V1")
        callbacks = []

        @_CALLBACK
        def callback(pointer, output, _):
            query = np.ctypeslib.as_array(pointer, shape=(10,)).copy()
            record = {"coefficients": query.tolist()}
            try:
                value = evaluator(query)
                if not isinstance(value, dict):
                    raise NativeAdapterError("native trial evaluator must return a mapping")
                packet = _packet(value["raw_moments"], value.get("moment_jacobian"),
                                 geometry_hash=value["geometry_hash"], topology_hash=value["topology_hash"],
                                 defined=value["derivative_defined"], diagnostic=value.get("derivative_diagnostic", ""),
                                 evaluation_map_identity=value["evaluation_map_identity"],
                                 boolean_backend_identity=value["boolean_backend_identity"],
                                 derivative_identity=value["derivative_identity"])
                ctypes.memmove(output, ctypes.byref(packet), ctypes.sizeof(packet))
                record.update(derivative_defined=value["derivative_defined"], geometry_hash=value["geometry_hash"],
                              topology_hash=value["topology_hash"], error=None)
                record.update({key:value[key] for key in ("evaluation_map_identity","boolean_backend_identity","derivative_identity","geometry_map_profile")})
                callbacks.append(record)
                return 0
            except Exception as exc:  # Fail this real trial; native backtracking may try another.
                output.contents.diagnostic = (type(exc).__name__ + ": " + str(exc)).encode("utf-8")[:1023]
                record.update(derivative_defined=None, error=type(exc).__name__ + ": " + str(exc))
                callbacks.append(record)
                return 1

        result = self._result(self.library.cpino_native_step(
            self.mode, self.weight, self.mass_scale, c.ctypes.data_as(_DOUBLE_PTR), ctypes.byref(center),
            self.equation, self.balance, self.max_evaluations, callback, None))
        result["trial_receipts"] = callbacks
        result["profile"] = PROFILE
        result["claim_scope"] = "native_numerical_step_only_no_final_geometry_acceptance"
        return result


def _resolve(root, relative):
    path = Path(relative)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class NativeSolverAdapter:
    """Strict, artifact-bound adapter for ``endpoint_attribution.attribution``.

    Required state additions are documented by ``STATE_SCHEMA`` checks below.
    Trial factories take ``(case, state)`` and return a fresh callable returning
    native M/J packets. Their source AND runtime binary are content-bound.
    """

    def __init__(self, case):
        self.case = copy.deepcopy(case)
        self.meta = self.case["meta"]
        verify_provenance(self.meta)
        if self.meta.get("native_solver_profile") != PROFILE:
            raise NativeAdapterError("incompatible solver profile; private 17:56/noodles is not this native profile")
        contract = self.meta["adapter_contract"]
        if contract["acceptance_reads_jacobian"] or contract["multiplier_policy"] != "native_recompute":
            raise NativeAdapterError("this native profile is M-only with native_recompute multipliers")
        if "asset_root" not in self.case:
            raise NativeAdapterError("artifact-bound case requires asset_root")
        self.root = Path(self.case["asset_root"]).resolve()
        self.receipts = verify_artifacts(self.meta, self.root)
        state_path = _resolve(self.root, self.meta["artifacts"]["solver_state"]["path"])
        self.state = json.loads(state_path.read_text())
        state = self.state
        if state.get("schema") != STATE_SCHEMA or state.get("profile") != PROFILE:
            raise NativeAdapterError("incompatible solver state schema/profile")
        if state.get("equation_profile") != "DUAL_MONOMIAL_RAW_V1":
            raise NativeAdapterError("native equation profile mismatch")
        if state.get("coefficient_convention") != CONVENTION:
            raise NativeAdapterError("unsupported raw coefficient normalization/J convention")
        geometry_profile = (SYNTHETIC_GEOMETRY_PROFILE if self.meta["split"] == "synthetic" else GEOMETRY_PROFILE)
        if (self.meta.get("teacher_map_profile") != geometry_profile
                or state.get("geometry_map_profile") != geometry_profile):
            raise NativeAdapterError("teacher geometry-map profile mismatch; CGAL cannot substitute for deployed Cherchi")
        backend = state.get("boolean_backend_identity", "")
        if backend != self.meta.get("native_geometry_backend_identity") or not backend:
            raise NativeAdapterError("native geometry backend identity mismatch")
        if self.meta["split"] != "synthetic":
            if (not backend.startswith("libigl::copyleft::cgal::mesh_boolean;libigl_source=")
                    or backend.endswith("unversioned")
                    or "cherchi" in self.meta["teacher_version"].lower()):
                raise NativeAdapterError("this profile only admits explicitly versioned repository CGAL labels, never Cherchi")
        if state.get("external_state_kind") != "repository_stateless_spherical_newton_v1":
            raise NativeAdapterError("unsupported mutable solver state; no state may be silently dropped")
        if state.get("multiplier_policy") != "native_recompute" or "multipliers" in state or "multiplier_mu" in state:
            raise NativeAdapterError("native endogenous multipliers must be recomputed, not injected")
        if state.get("thresholds") != self.meta["thresholds"]:
            raise NativeAdapterError("frozen native thresholds differ from case thresholds")
        for key in ("geometry_sha256", "operand_sha256", "teacher_version", "coordinate_frame", "units"):
            if state.get("bindings", {}).get(key) != self.meta[key]:
                raise NativeAdapterError(f"solver state identity mismatch: {key}")
        config = state.get("solver_settings", {})
        required = {"maximum_iterations": 1, "maximum_restarts": 1,
                    "line_search_reduction": .5, "maximum_geodesic_step": .5}
        if any(config.get(k) != v for k, v in required.items()):
            raise NativeAdapterError("one-step settings must preserve native Newton and line search")
        self.coefficients = _unit_coefficients(self.case["coefficients"])
        self.geometry_hash = _sha(state.get("center_geometry_hash"), "center geometry")
        self.topology_hash = _sha(state.get("center_topology_hash"), "center topology")
        self.center_map_identity = _sha(state.get("center_evaluation_map_identity"), "center evaluation map")
        build = state.get("kernel_build_receipt", {})
        receipt_path = _resolve(self.root, build["path"])
        if sha256(receipt_path) != _sha(build.get("sha256"), "kernel build receipt"):
            raise NativeAdapterError("native build receipt checksum mismatch")
        self.kernel = NativeKernel(_resolve(self.root, self.meta["artifacts"]["solver"]["path"]),
                                   objective=state["objective"], thresholds=state["thresholds"],
                                   max_evaluations=config["maximum_function_evaluations_per_restart"],
                                   build_receipt=receipt_path, expected_sha256=self.meta["solver_sha256"])
        self._used = False

    def _start(self, c):
        if self._used:
            raise NativeAdapterError("each attribution arm must use a fresh native adapter")
        self._used = True
        if not np.array_equal(_unit_coefficients(c), self.coefficients):
            raise NativeAdapterError("fixed-query coefficient identity changed")

    def accept(self, c, m, j):
        self._start(c)
        result = self.kernel.accept(c, m, j)
        result["profile"] = PROFILE
        result["claim_scope"] = "repository_native_formula_only_not_private_1756_or_mesh_acceptance"
        return result

    def _trial_factory(self):
        config = self.state.get("trial_evaluator", {})
        entrypoint = config.get("factory", "")
        if ":" not in entrypoint:
            raise NativeAdapterError("one-step requires an explicitly bound native trial evaluator")
        concrete = __name__ + ":repository_fullrank_geometry_factory"
        if self.meta["split"] != "synthetic" and entrypoint != concrete:
            raise NativeAdapterError("this profile requires the concrete repository fullrank geometry factory")
        module, name = entrypoint.split(":", 1)
        constructor = getattr(importlib.import_module(module), name)
        source = inspect.getsourcefile(constructor)
        if not source or sha256(source) != _sha(config.get("source_sha256"), "trial evaluator source"):
            raise NativeAdapterError("native trial evaluator source checksum mismatch")
        runtime = config.get("runtime_artifact", {})
        runtime_path = _resolve(self.root, runtime["path"])
        if sha256(runtime_path) != _sha(runtime.get("sha256"), "trial runtime"):
            raise NativeAdapterError("native trial runtime checksum mismatch")
        evaluator = constructor(copy.deepcopy(self.case), copy.deepcopy(self.state))
        if not callable(evaluator):
            raise NativeAdapterError("native trial factory did not return an evaluator")
        # A declaration alone cannot bind the actually loaded extension.
        if Path(getattr(evaluator, "runtime_path", "")).resolve() != runtime_path:
            raise NativeAdapterError("trial evaluator did not bind its actual loaded runtime")
        def bound_evaluation(c):
            value=evaluator(c)
            if (value.get("geometry_map_profile") != self.state["geometry_map_profile"]
                    or value.get("boolean_backend_identity") != self.state["boolean_backend_identity"]):
                raise NativeAdapterError("trial changed the frozen geometry map or Boolean backend")
            _sha(value.get("evaluation_map_identity"),"native per-query scientific map identity")
            _sha(value.get("source_scientific_hash"),"native source motion identity")
            return value
        return bound_evaluation

    def step(self, c, m, j):
        self._start(c)
        if self.meta.get("derivative_valid") is not True:
            raise NativeAdapterError("one-step requires independently qualified derivatives")
        if self.meta.get("derivative_dimension") != 10 or not np.array_equal(self.case.get("chart_basis"), np.eye(10)):
            raise NativeAdapterError("R3/R8 require supported-space/moving-chart adapter; never zero-pad Full-J")
        evaluator = self._trial_factory()
        measured = evaluator(self.coefficients.copy())
        if measured.get("derivative_defined") is not True:
            raise NativeAdapterError("native center derivative is currently unavailable")
        if measured.get("geometry_hash") != self.geometry_hash or measured.get("topology_hash") != self.topology_hash:
            raise NativeAdapterError("native center geometry/topology changed")
        if measured.get("evaluation_map_identity") != self.center_map_identity:
            raise NativeAdapterError("native center evaluation map identity changed")
        tolerance = self.meta["replay_tolerances"]
        # This independently evaluated native anchor is checked before ANY
        # substituted center M/J reaches the native numerical solver.
        for key, recorded, shape in (("raw_moments", "native_moments", (10,)),
                                     ("moment_jacobian", "native_jacobian", (10, 10))):
            actual = finite_array(measured[key], shape, "native center " + key)
            expected = finite_array(self.case[recorded], shape, recorded)
            if not np.allclose(actual, expected, atol=tolerance["residual_atol"], rtol=tolerance["residual_rtol"]):
                raise NativeAdapterError("native center M/J replay mismatch: " + key)
        result = self.kernel.step(c, m, j, evaluator, geometry_hash=self.geometry_hash, topology_hash=self.topology_hash)
        result["native_center_revalidation_calls"] = 1
        result["total_native_geometry_calls"] = 1 + result["trial_evaluations"]
        result["geometry_map_profile"] = self.state["geometry_map_profile"]
        result["native_center_evaluation_map_identity"] = self.center_map_identity
        return result


def factory(case):
    return NativeSolverAdapter(case)


class _RepositoryFullRankEvaluator:
    def __init__(self, case, state):
        from make_it_stand import spin
        self.spin = spin
        root = Path(case["asset_root"])
        meta = case["meta"]
        geometry = _resolve(root, meta["artifacts"]["geometry"]["path"])
        operand = _resolve(root, meta["artifacts"]["operand"]["path"])
        spec = json.loads(operand.read_text())
        if spec.get("schema") != "cpino_repository_fullrank_geometry_input_v1":
            raise NativeAdapterError("unsupported native geometry request schema")
        if (spec.get("geometry_map_profile") != GEOMETRY_PROFILE
                or spec.get("boolean_backend_identity") != state["boolean_backend_identity"]):
            raise NativeAdapterError("finite-source recipe geometry profile/backend mismatch")
        self.expected_backend=state["boolean_backend_identity"]
        with np.load(geometry, allow_pickle=False) as archive:
            meshes = {name: archive[name].copy() for name in
                      ("outer_vertices", "outer_faces", "design_vertices", "design_faces")}
        options = copy.deepcopy(spec["request"])
        if any(name in options for name in meshes):
            raise NativeAdapterError("duplicate mesh inputs in native request")
        settings = spin.SpinSolverConfig(**options.pop("settings"))
        if settings.equation_profile != "DUAL_MONOMIAL_RAW_V1":
            raise NativeAdapterError("geometry request native profile mismatch")
        self.request = spin.SpinItFasterRequest(**meshes, **options, settings=settings)
        request_mode = self.request.mode
        if request_mode == "STATIC_BALANCE":
            modes = {"MAXIMUM_MASS": "STATIC_MAXIMUM_MASS", "LOWER_CENTER_OF_MASS": "STATIC_LOWER_CENTER",
                     "PARETO_TRADEOFF": "STATIC_PARETO"}
            expected_mode = modes.get(self.request.static_objective)
            expected_weight = self.request.static_pareto_weight if expected_mode == "STATIC_PARETO" else 0.
            expected_scale = (1. if self.request.static_objective_measure == "DENSITY_WEIGHTED_MASS_V1"
                              else 1. / self.request.material_density)
        elif request_mode == "SPINNING_TOP":
            expected_mode = ("SPINNING_TOP_LOWER_CENTER" if self.request.spinning_top_objective == "LOWER_CENTER_OF_MASS"
                             else "SPINNING_TOP")
            expected_weight = self.request.center_of_mass_weight
            expected_scale = 1.
        else:
            expected_mode, expected_weight, expected_scale = request_mode, 0., 1.
        actual_objective = state["objective"]
        if (actual_objective.get("mode") != expected_mode
                or float(actual_objective.get("weight", 0.)) != expected_weight
                or float(actual_objective.get("mass_scale", 1.)) != expected_scale):
            raise NativeAdapterError("native geometry request objective differs from numerical state")
        self.settings = spin.SpinEvaluatorConfig(**spec["evaluator_settings"])
        if self.settings.derivative_method != "DISCRETE_FULL_RANK_QUADRIC":
            raise NativeAdapterError("trial full-J must use qualified native finite-source derivative")
        # The existing binding exposes the same-call topology proof only when
        # exact source AND material checks are enabled; no topology is guessed.
        if not (self.settings.exact_binary64_source_boundary and self.settings.exact_binary64_material_boundary):
            raise NativeAdapterError("full-rank bridge requires exact same-call material topology proof")
        self.native, self.native_request = spin._native_request(self.request)
        self.runtime_path = Path(self.native.__file__).resolve()
        self.native_settings = spin._native_evaluator_settings(self.native, self.settings)

    def __call__(self, c):
        c = _unit_coefficients(c)
        value = self.native.evaluate_spin_it_faster_fullrank_quadric(
            self.native_request, c, self.native_settings, True)
        if value.get("boolean_backend_identity") != self.expected_backend:
            raise NativeAdapterError("loaded native geometry backend differs from frozen CGAL backend")
        proof = value.get("material_proof_attachment", {})
        if proof.get("schema") != "same-call-cgal-material-proof-v1" or proof.get("geometry_hash") != value["geometry_hash"]:
            raise NativeAdapterError("native same-call material proof missing or inconsistent")
        return dict(raw_moments=value["total_raw_moments"], moment_jacobian=value["coefficient_jacobian"],
                    derivative_defined=value["derivative_defined"], derivative_diagnostic=value["derivative_diagnostic"],
                    geometry_hash=value["geometry_hash"], topology_hash=proof["provenance_topology_hash"],
                    geometry_map_profile=GEOMETRY_PROFILE,
                    boolean_backend_identity=value["boolean_backend_identity"],
                    evaluation_map_identity=value["scientific_hash"],
                    source_scientific_hash=value["source_scientific_hash"],
                    derivative_identity="native_fullrank_scientific:"+value["scientific_hash"])


def repository_fullrank_geometry_factory(case, state):
    """Concrete factory calling the repository's real finite-source CGAL API.

    Its native extension and real indexed mesh assets must exist. It never
    substitutes archived moments, a learned response or a synthetic shape.
    """
    return _RepositoryFullRankEvaluator(case, state)
