"""Versioned finite-stratum entry for the physical teacher and solver callbacks.

Full-rank Newton callbacks keep the original native packet in unit coordinates.
A typed caller supplies a raw tangent basis and consumes a 10×D chart derivative.
The native geometry-only entry produces no post-Boolean material information.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import (
    FiniteOperand, RawChart, audit_physical_source, operand_from_source, unit_to_raw_jacobian,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native_physics import (
    NativePhysicalTargets, evaluate_operand_physics, targets_from_full_packet,
)


def _registered_backend(backend):
    root = Path(__file__).resolve().parent
    protocol = json.loads((root/'protocol.json').read_text())
    frozen = json.loads((root.parent/'native/evidence/native_replay.json').read_text())['runtime']
    if (asdict(backend.settings) != frozen['profile'] or frozen['profile_sha256'] != protocol['legacy_profile_sha256']
            or backend.integral_profile != 'CERTIFIED_MPFR256_GMP_FALLBACK_V1'
            or backend.rational_arithmetic_backend != 'NATIVE_GMP_SURFACE_V1'
            or backend.fixed_boundary_proof is None):
        raise ValueError('stratum extension requires the registered source, physical frame and arithmetic settings')
    hashes = {v['module']:v['sha256'] for v in frozen['loaded_modules']}
    for lib in (backend.native,backend.bridge,backend.arithmetic):
        with Path(lib.__file__).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != hashes[lib.__name__]:
                raise ValueError('physical backend differs from frozen runtime: '+lib.__name__)


def value_operand(source, surface):
    if not source['defined'] or not source.get('side_selection_defined'):
        raise ValueError('finite source rejected: '+str(source.get('side_selection_diagnostic') or source.get('diagnostic')))
    if source['surface_type'] != surface:
        raise ValueError('requested chart differs from native source surface')
    caps = np.asarray(source['face_is_cap']).reshape(-1)
    if np.count_nonzero(caps) != source['cap_face_count']:
        raise ValueError('native cap count and mask disagree')
    return FiniteOperand(source['vertices'], source['faces'], caps,
                         'intersection' if source['interior_is_positive'] else 'difference')


class SharedStratumOperator:
    """Use one instance per registered geometry and worker, as in the native evaluator."""

    def __init__(self, backend, extension):
        self.backend = backend
        self.extension = extension
        sources = [Path(__file__),Path(sys.modules[evaluate_operand_physics.__module__].__file__),
                   Path(sys.modules[operand_from_source.__module__].__file__)]
        identity = dict(schema='tvcg.finite-stratum.raw.v1', base=backend.context_identity,
                        source=extension.backend_identity,
                        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        self.identity = hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

    @classmethod
    def from_evaluator(cls, backend, build_dir=None):
        from .native_backend.bridge import load_native
        _registered_backend(backend)
        return cls(backend, load_native(build_dir))

    @classmethod
    def from_request(cls, request, normalization, physical_vertices, *, build_dir=None):
        """Construct the common teacher/solver service for an ordinary registered request."""
        from make_it_stand import spin
        from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
        frozen = Path(__file__).resolve().parents[1]/'native/evidence/native_replay.json'
        config = spin.SpinEvaluatorConfig(**json.loads(frozen.read_text())['runtime']['profile'])
        backend = InProcessCherchiEvaluator(request,config,normalization=normalization,
                        design_vertices_physical=physical_vertices, fixed_boundary_reuse=True,
                        rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1',
                        integral_profile='CERTIFIED_MPFR256_GMP_FALLBACK_V1')
        return cls.from_evaluator(backend,build_dir)

    def condition(self, q, chart, *, with_motion=True):
        """Evaluate the actual finite source in the target frame, including all its caps."""
        q = np.asarray(q,dtype=np.float64)
        unit_to_raw_jacobian(q)
        if not self.backend.triangulation.algebraic_finite_map:
            raise ValueError('algebraic finite source setting changed')
        if with_motion:
            source = self.extension.source_motion(q,self.backend.target_design,self.backend.triangulation)
            operand = operand_from_source(source,q,chart=chart)
        else:
            source = self.extension.source_value(q,self.backend.target_design,self.backend.triangulation)
            operand = value_operand(source,chart.surface)
        b = self.backend
        audit_physical_source(b.native,operand,b.request.target_axes,b.request.target_origin,b.length,b.normalization_origin)
        return operand

    def teacher(self, q, chart=None, *, capture_boundary=False):
        """Emit separate availability for material, ambient derivative and chart derivative."""
        q = np.asarray(q,dtype=np.float64)
        if chart is None:
            key = tuple(q) if q.shape==(10,) else None
            saved = self.backend._materials.get(key)
            last = self.backend.last
            try:
                return targets_from_full_packet(self.backend(q),q)
            except (RuntimeError,ValueError,ArithmeticError) as exc:
                return NativePhysicalTargets(evaluation_map_identity=self.backend.context_identity,moment_diagnostic=str(exc))
            finally:
                self.backend.last = last
                if saved is None:
                    self.backend._materials.pop(key,None)
                else:
                    self.backend._materials[key] = saved
        chart = RawChart(chart.surface,chart.basis)
        motion_diagnostic = None
        try:
            operand = self.condition(q,chart)
        except (RuntimeError,ValueError,ArithmeticError) as exc:
            motion_diagnostic = str(exc)
            try:
                operand = self.condition(q,chart,with_motion=False)
            except (RuntimeError,ValueError,ArithmeticError) as value_exc:
                return NativePhysicalTargets(chart=chart,evaluation_map_identity=self.identity,
                           moment_diagnostic=str(value_exc),derivative_diagnostic=motion_diagnostic)
        targets = evaluate_operand_physics(self.backend,operand,chart=chart,capture_boundary=capture_boundary,
                                          capture_topology=True)
        if targets.evaluation_map_identity != self.backend.context_identity:
            raise ValueError('changed material context while evaluating finite operand')
        targets.evaluation_map_identity = self.identity
        if motion_diagnostic is not None:
            targets.derivative_diagnostic = motion_diagnostic
        return targets

    def solver(self, q, chart=None):
        """An untyped call is the historical ambient Newton entry; typed columns use raw chart coordinates."""
        if chart is None:
            return self.backend(q)
        target = self.teacher(q,chart)
        if not target.moment_valid:
            raise ValueError(target.moment_diagnostic or 'typed material unavailable')
        return dict(raw_moments=target.moments,variable_raw_moments=target.editable_moments,
                    derivative_defined=False, stratum_derivative_defined=target.restricted_valid,
                    stratum_moment_jacobian=target.restricted_jacobian,
                    stratum_coefficient_basis=target.chart.basis.copy(),
                    stratum_derivative_diagnostic=target.derivative_diagnostic,
                    stratum_coordinate_convention='RAW_SUPPLIED_CHART_V1',
                    geometry_hash=target.geometry_hash, topology_hash=target.topology_hash,
                    topology_diagnostic=target.topology_diagnostic,
                    boolean_backend_identity=self.backend.backend_identity,evaluation_map_identity=self.identity)
