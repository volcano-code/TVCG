"""Frozen-checkpoint callback for the registered native solver.

This is a drop-in *callback* factory, not a replacement optimizer or acceptor.
The caller keeps the registered solver, initial guesses and native correction.
Inputs are pre-Boolean native finite operands; candidate queries never request
native retained-material moments. Production acceptance lives in acceptance.py.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import (
    compact_pino_v21, compact_pino_v23, v21_blocks, v21_native,
)
from .metrics import QueryTrace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage(trace, name):
    return nullcontext() if trace is None else trace.stage(name)


def _host(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class RuntimePredictor:
    """One geometry and one checkpoint per solve, explicitly closeable.

    __call__(q) supplies the native solver's full, unit-coordinate Jacobian.
    evaluate(q, with_jacobian=False) is a value request for a caller that really
    needs only M; it must not be substituted for a Newton callback. It is also
    not an independent native acceptance certificate.
    """
    def __init__(self, backend, model, cache, *, checkpoint_sha256, metadata,
                 trace_sink=None, profile_sync=False):
        if type(profile_sync) is not bool or (trace_sink is not None and not callable(trace_sink)):
            raise ValueError("invalid trace sink or synchronization option")
        self.backend, self.model, self.cache = backend, model, cache
        self.digest, self.metadata = checkpoint_sha256, metadata
        self.trace_sink, self.profile_sync = trace_sink, profile_sync
        self.calls = 0
        self._seen = set()
        self._closed = False

    def __call__(self, query):
        return self.evaluate(query, with_jacobian=True)

    def evaluate(self, query, *, with_jacobian=True):
        if self._closed:
            raise RuntimeError("predictor is closed")
        if type(with_jacobian) is not bool:
            raise ValueError("with_jacobian must be an explicit Boolean")
        q = np.asarray(query, dtype=np.float64)
        norm = np.linalg.norm(q)
        if q.shape != (10,) or not np.isfinite(q).all() or not np.isfinite(norm) or norm == 0:
            raise ValueError("ten finite nonzero raw coefficients are required")
        device = next(self.model.parameters()).device
        synchronize = (lambda: torch.cuda.synchronize(device)) if self.profile_sync and device.type == "cuda" else None
        key = q.astype("<f8", copy=False).tobytes()
        self.calls += 1
        trace = None
        if self.trace_sink is not None:
            trace = QueryTrace(f"query-{self.calls:06d}", mode="joint" if with_jacobian else "M",
                query=q, cache_state="repeated_query" if key in self._seen else "new_query",
                synchronize=synchronize, metadata=dict(
                    arm=type(self.model).__name__, device=str(device),
                    callback_returns_completed_host_arrays=True,
                    stage_timing="synchronized_profile" if synchronize else "host_wall_not_cuda_kernel_time",
                    native_source_capabilities=v21_native.native_operand_capabilities(),
                    source_motion_requested=True,
                    material_boolean_requested=False,
                    efficiency_before=self.model.efficiency_statistics()))
            self._seen.add(key)
        started = False
        try:
            with trace if trace is not None else nullcontext():
                started = True
                with _stage(trace, "native_source"):
                    operand = v21_native.native_operand(self.backend, q, with_jacobian=with_jacobian)
                with torch.no_grad():
                    prediction = self.model.evaluate_operand(self.cache, operand,
                        with_jacobian=with_jacobian, trace=trace)
                with _stage(trace, "host_export"):
                    moment = _host(prediction.moments)
                    jac = None if prediction.jacobian is None else _host(prediction.jacobian)
                    reason = prediction.diagnostics.get("solver_status")
                    if moment.shape != (10,) or not np.isfinite(moment).all() or moment[0] <= self.backend.fixed[0]:
                        raise ValueError(reason or "nonfinite or empty editable prediction")
                    if with_jacobian and (jac is None or jac.shape != (10, 10) or not np.isfinite(jac).all()):
                        raise ValueError(reason or "native full Jacobian unavailable")
                    # The operator differentiates raw q; the deployed solver
                    # applies its own normalized-coordinate chain exactly once.
                    if with_jacobian:
                        jac = jac * norm
                    else:
                        jac = None
                    packet = dict(
                        raw_moments=moment, variable_raw_moments=moment-self.backend.fixed,
                        moment_jacobian=jac, derivative_defined=with_jacobian,
                        geometry_hash=self.cache.blocks.identity, topology_hash="surrogate-unverified",
                        evaluation_map_identity=type(self.model).__name__+"-"+self.digest,
                        boolean_backend_identity="surrogate-no-Boolean",
                        derivative_identity="finite-operand-motion-projected-unit" if with_jacobian else "not-requested",
                        request_mode="joint" if with_jacobian else "M",
                        requires_native_acceptance=True)
                    if trace is not None:
                        trace.metadata.update(prediction_defined=True,
                            prediction_solver_status=reason,
                            request_status="PROPOSAL_REQUIRES_NATIVE_ACCEPTANCE" if with_jacobian
                                           else "MOMENT_ONLY_REQUIRES_NATIVE_ACCEPTANCE")
                return packet
        except BaseException as exc:
            if trace is not None:
                trace.metadata.update(prediction_defined=False, refusal_reason=str(exc))
            raise
        finally:
            if trace is not None and started:
                trace.metadata["efficiency_after"] = self.model.efficiency_statistics()
                self.trace_sink(trace.as_dict())

    def close(self):
        if self._closed:
            return
        self.metadata["final_efficiency_statistics"] = self.model.efficiency_statistics()
        if isinstance(self.model, compact_pino_v23.CompactPINOv23):
            self.model.close()
        else:
            self.model.clear_efficiency_caches()
        self._seen.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("predictor is closed")
        return self

    def __exit__(self, kind, error, traceback):
        self.close()
        return False


def make_predictor(backend, pack, checkpoint, metadata, *, expected_geometry,
                   expected_pack_sha256, device="cuda:0", arm="v23",
                   runtime=None, trace_sink=None, profile_sync=False):
    """Load an already registered material pack and frozen trained checkpoint.

    The caller must obtain geometry/manifest identity from its frozen registry,
    not accept an arbitrary freshly read hash as proof of registration.
    No material preprocessing cost is claimed as zero: historical preparation
    is recorded, and cold timing must begin before building the pack/backend.
    """
    if arm not in ("v22", "v23"):
        raise ValueError("arm must be v22 or v23")
    if not isinstance(expected_geometry, str) or not expected_geometry:
        raise ValueError("the registered geometry identity is required")
    if not isinstance(expected_pack_sha256, str) or len(expected_pack_sha256) != 64:
        raise ValueError("the frozen pack manifest SHA256 is required")
    if runtime is not None and arm == "v22":
        raise ValueError("runtime options apply only to the v23 arm")
    pack, checkpoint = Path(pack), Path(checkpoint)
    start = perf_counter()
    manifest = pack / "identity.json"
    if sha256(manifest) != expected_pack_sha256:
        raise ValueError("registered material manifest hash mismatch")
    info = json.loads(manifest.read_text())
    material = info["controls"]["conservative"]
    if info["geometry"] != expected_geometry or info["side"] != 256:
        raise ValueError("registered geometry/native256 mismatch")
    if not np.array_equal(info["fixed_moments"], backend.fixed):
        raise ValueError("fixed shell differs from the registered native evaluator")
    if float(info["density"]) != float(backend.request.material_density):
        raise ValueError("material density differs from the registered native evaluator")
    blocks = v21_blocks.BlockMeasure.open(pack / "conservative",
        identity=material["identity"], block_voxels=8)
    if len(blocks.nodes) != material["nodes"]:
        raise ValueError("registered conservative material node count mismatch")
    pack_open_s = perf_counter()-start
    start = perf_counter()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    digest = sha256(checkpoint)
    cls = compact_pino_v23.CompactPINOv23 if arm == "v23" else compact_pino_v21.CompactPINOv21
    kwargs = dict(efficiency=True, query_cache_bytes=0,
        response_type=weights["response_type"], training_scope="response", full_cell_rule="tetra4")
    if arm == "v23":
        kwargs["runtime"] = runtime
    model = cls(16, **kwargs)
    try:
        model.load_state_dict(weights["model"], strict=True)
        model.to(device).eval()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        weight_load_s = perf_counter()-start
        start = perf_counter()
        with torch.no_grad():
            cache = model.encode_geometry(blocks, backend.fixed, info["density"], inference=True)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        metadata.update(arm=arm, geometry=expected_geometry,
            pack=str(pack.resolve()), pack_manifest_sha256=expected_pack_sha256,
            static_material_identity=material["identity"], source_rle_sha256=info["rle_sha256"],
            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
            checkpoint_epoch=weights.get("epoch"), checkpoint_step=weights.get("step"),
            response_type=weights["response_type"], query_cache_bytes=0,
            model_source_sha256=sha256(compact_pino_v23.__file__ if arm == "v23" else compact_pino_v21.__file__),
            pack_open_s=pack_open_s, checkpoint_load_s=weight_load_s,
            static_feature_encode_s=perf_counter()-start,
            historical_material_generation_s=material.get("material_s"),
            historical_block_index_s=material.get("block_index_s"),
            cold_material_generation_included=False,
            native_source_capabilities=v21_native.native_operand_capabilities(),
            query_condition="native finite operand before Boolean; no retained-material label",
            callback_topology="surrogate-unverified; independent native acceptance required")
        return RuntimePredictor(backend, model, cache, checkpoint_sha256=digest,
            metadata=metadata, trace_sink=trace_sink, profile_sync=profile_sync)
    except BaseException:
        if isinstance(model, compact_pino_v23.CompactPINOv23):
            model.close()
        raise
