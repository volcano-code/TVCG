"""Deployment plumbing tests; native source is stubbed, not a native speed claim."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import deployment
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v23 import CompactPINOv23, RuntimeConfig
from spin_adapter.tests.test_v22_compact_integration import _case


def test_deployment_value_request_raw_to_unit_chain_trace_and_close(monkeypatch):
    blocks, operand, fixed = _case()
    operand = replace(operand, vertex_velocity=np.pad(operand.vertex_velocity, ((0, 0), (0, 0), (0, 7))))
    native_requests = []
    def source(backend, q, *, with_jacobian):
        native_requests.append(with_jacobian)
        return operand if with_jacobian else replace(operand, vertex_velocity=None)
    monkeypatch.setattr(deployment.v21_native, "native_operand", source)
    backend = SimpleNamespace(fixed=fixed)
    rows, metadata = [], {}
    with CompactPINOv23(runtime=RuntimeConfig(surface_workers=1)).eval() as model:
        with torch.no_grad():
            cache = model.encode_geometry(blocks, fixed, inference=True)
            reference = model.evaluate_operand(cache, operand)
        runtime = deployment.RuntimePredictor(backend, model, cache,
            checkpoint_sha256="test-weight-identity", metadata=metadata, trace_sink=rows.append)
        q = np.arange(1., 11.)
        out = runtime(q)
        value = runtime.evaluate(q, with_jacobian=False)
        np.testing.assert_allclose(out["raw_moments"], reference.moments.numpy(), rtol=1e-14)
        np.testing.assert_allclose(out["moment_jacobian"], reference.jacobian.numpy()*np.linalg.norm(q), rtol=1e-14)
        np.testing.assert_array_equal(out["raw_moments"], value["raw_moments"])
        assert value["moment_jacobian"] is None and not value["derivative_defined"]
        assert native_requests == [True, False]
        assert [row["cache_state"] for row in rows] == ["new_query", "repeated_query"]
        assert [row["mode"] for row in rows] == ["joint", "M"]
        assert rows[-1]["metadata"]["prediction_defined"]
        assert not rows[-1]["metadata"]["material_boolean_requested"]
        assert rows[-1]["metadata"]["native_source_capabilities"]["moment_only_source_mode"] == "motion_fallback"
        assert {x["name"] for x in rows[0]["stages"]} >= {"native_source", "host_export", "distance_stencil"}
        assert out["requires_native_acceptance"]
        runtime.close()
        assert metadata["final_efficiency_statistics"]["encoder_calls"] == 1
        with pytest.raises(RuntimeError, match="closed"):
            runtime(q)


def test_deployment_source_refusal_is_not_silently_retried(monkeypatch):
    calls = []
    def refuse(*args, **kwargs):
        calls.append(kwargs["with_jacobian"])
        raise ValueError("ambiguous native side")
    monkeypatch.setattr(deployment.v21_native, "native_operand", refuse)
    rows = []
    with CompactPINOv23(runtime=RuntimeConfig(surface_workers=1)).eval() as model:
        # This branch must reject before touching encoded material or predictor.
        runtime = deployment.RuntimePredictor(SimpleNamespace(fixed=np.zeros(10)), model, None,
            checkpoint_sha256="fixture", metadata={}, trace_sink=rows.append)
        with pytest.raises(ValueError, match="ambiguous native side"):
            runtime(np.ones(10))
        assert calls == [True]
        assert not rows[0]["defined"] and not rows[0]["metadata"]["prediction_defined"]
        assert rows[0]["stages"][0]["failed"]
        runtime.close()


def test_factory_binds_manifest_material_density_fixed_and_strict_checkpoint(tmp_path, monkeypatch):
    blocks, _, fixed = _case()
    backend = SimpleNamespace(fixed=fixed, request=SimpleNamespace(material_density=1.))
    info = dict(geometry="fixture", side=256, fixed_moments=fixed.tolist(), density=1.,
                rle_sha256="f"*64, controls={"conservative":dict(identity=blocks.identity, nodes=len(blocks.nodes))})
    (tmp_path/"identity.json").write_text(json.dumps(info))
    digest = deployment.sha256(tmp_path/"identity.json")
    cp = tmp_path/"weights.pt"
    with CompactPINOv23(runtime=RuntimeConfig(surface_workers=1)) as source_model:
        torch.save(dict(model=source_model.state_dict(), response_type="monotone_warp", epoch=0, step=0), cp)
    monkeypatch.setattr(deployment.v21_blocks.BlockMeasure, "open", lambda *a, **kw: blocks)
    metadata = {}
    kwargs = dict(expected_geometry="fixture", expected_pack_sha256=digest, device="cpu",
                  runtime=RuntimeConfig(surface_workers=1))
    with deployment.make_predictor(backend, tmp_path, cp, metadata, **kwargs) as runtime:
        assert runtime.model.encoder_calls == 1
        assert metadata["checkpoint_sha256"] == deployment.sha256(cp)
        assert metadata["query_cache_bytes"] == 0
        assert not metadata["cold_material_generation_included"]
    with pytest.raises(ValueError, match="manifest hash"):
        deployment.make_predictor(backend, tmp_path, cp, {},
            **{**kwargs, "expected_pack_sha256":"0"*64})
    backend.request.material_density = 2.
    with pytest.raises(ValueError, match="density"):
        deployment.make_predictor(backend, tmp_path, cp, {}, **kwargs)


def test_replay_cli_is_a_distinct_measurement_scope():
    from .replay_cli import parser
    args = parser().parse_args([
        "--alignment", "/alignment", "--corpus", "/corpus", "--physics", "/physics",
        "--pack", "/pack", "--pack-sha256", "0"*64, "--checkpoint", "/weights.pt",
        "--output", "/output", "--arm", "v23"])
    assert args.mode == "joint" and args.passes == 2 and not args.profile_sync


def test_replay_preserves_existing_evidence_before_loading_native_registry(tmp_path):
    from .replay_cli import run
    (tmp_path/"query_traces.jsonl").write_text("preserved\n")
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(output=tmp_path))
    assert (tmp_path/"query_traces.jsonl").read_text() == "preserved\n"
