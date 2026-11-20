"""Boundary accounting and the checkpoint path used by the one-update preflight."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.memory_profile import (
    MemoryProfile, profile_adjoint, profile_count, profile_recompute, profile_stage, profile_tensors,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import build_model
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13 import preflight_unified
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.tests import test_unified_corpus as corpus_fixtures


def test_nested_stage_uses_exclusive_phase_times_and_saves_failure(tmp_path):
    path = tmp_path / "memory_profile.json"
    with pytest.raises(ValueError, match="interrupted"):
        with MemoryProfile("cpu", path):
            with profile_stage("train", phase="backward"):
                with profile_stage("chunk"):
                    profile_count(active_intervals=3)
                    profile_tensors(features=torch.zeros((3, 2)))
                    with profile_recompute("project"):
                        raise ValueError("interrupted")
    data = json.loads(path.read_text())
    assert data["status"] == "failed"
    assert "forward" not in data["exclusive_phase_seconds"]
    assert sum(data["exclusive_phase_seconds"].values()) == pytest.approx(data["elapsed_seconds"])
    child = next(row for row in data["stages"] if row["name"] == "chunk")
    assert child["phase"] == "backward"
    assert child["counts"]["active_intervals"] == 3
    assert child["tensor_observations"]["features"]["total_elements"] == 6
    assert data["recompute_regions"]["project"]["calls"] == 1
    assert "interrupted" in path.with_suffix(".md").read_text()
    # No active session after a failed update.
    profile_count(active_intervals=1)


def test_autograd_worker_inherits_only_the_forward_graph_profile(tmp_path):
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.geometry_physics_attention import GeometryPhysicsCrossAttention
    from torch.utils.checkpoint import checkpoint

    path = tmp_path / "profile.json"
    attention = GeometryPhysicsCrossAttention(4, heads=2, chunk_intervals=2)
    q = torch.randn(2, 10)
    features = torch.randn(3, 4, requires_grad=True)
    mass = torch.ones(3)
    with MemoryProfile("cpu", path):
        with profile_stage("forward"):
            x = checkpoint(lambda f: f.sin(), features, use_reentrant=False,
                           context_fn=lambda: (nullcontext(), profile_recompute("encoder_replay")))
            moment, jac, _ = attention.forward_with_query_jacobian(q, x, mass)
            loss = moment.square().sum() + jac.square().sum()
        with profile_stage("Backward", phase="backward"), ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(loss.backward).result()
    data = json.loads(path.read_text())
    calls = {r["name"]: r["calls"] for r in data["stages"] if r["phase"] == "backward"}
    assert calls["attention_moments"] == 1
    assert data["recompute_regions"]["attention_key_value_projection"]["calls"] == 2
    assert data["recompute_regions"]["encoder_replay"]["calls"] == 1
    assert data["recompute_regions"]["encoder_replay"]["cuda_stream_seconds"] is None


def test_adjoint_span_times_only_training_backward(tmp_path):
    layer = torch.nn.Linear(3, 2)
    value = torch.randn(4, 3, requires_grad=True)
    path = tmp_path / "profile.json"
    with MemoryProfile("cpu", path):
        with profile_stage("forward"):
            prediction = layer(value)
            profile_adjoint("linear", prediction, value, layer.parameters())
            torch.autograd.grad(prediction.sum(), value, retain_graph=True)
            loss = prediction.square().sum()
            profile_adjoint("unused", value.sin(), value)
        with profile_stage("Backward", phase="backward"), ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(loss.backward).result()
    assert not layer.weight._backward_hooks
    assert not value._backward_hooks
    # A later unprofiled update must run with no stale parameter hooks.
    layer(value).sum().backward()
    data = json.loads(path.read_text())["adjoint_spans"]
    assert data["linear"]["completed_spans"] == 1
    assert data["linear"]["output_gradients"] == 1
    assert data["linear"]["input_gradients"] == 1
    assert data["linear"]["parameter_gradients"] == 2
    assert data["linear"]["cpu_wall_seconds"] > 0
    assert data["linear"]["cuda_stream_seconds"] is None
    assert data["unused"]["cpu_wall_seconds"] is None


def test_smoke_restores_checkpoint_and_consumes_partial_labels(tmp_path):
    manifest = corpus_fixtures.UnifiedCorpusTests()._write_corpus(tmp_path, restricted=True)
    geometry_file = tmp_path / "geometry.npz"
    with np.load(geometry_file) as archive:
        geometry = dict(archive)
    geometry["side"] = np.array(512)
    np.savez(geometry_file, **geometry)
    document = json.loads(manifest.read_text())
    for row in document["records"]:
        row["geometry_sha256"] = corpus_fixtures.sha(geometry_file)
    manifest.write_text(json.dumps(document))

    def small_model(name, device):
        return build_model(name, device=device, config_overrides={"pcraft_preset": "smoke", "sqno_neighbors": 2})

    path = tmp_path / "memory_profile.json"
    with patch.object(preflight_unified, "build_model", small_model):
        report = preflight_unified.run_preflight(
            manifest, device="cpu", config_name="reference_v1", grid_size=8, seed=123, geometry_id="train",
            run_update=True, output=tmp_path/"checkpoint.pt", memory_profile=path,
        )
    assert report["status"] == "pass"
    metadata = json.loads(path.read_text())["metadata"]
    assert metadata["update"]["consumed_full_jacobians"] == 1
    assert metadata["update"]["consumed_restricted_directions"] == 3
    assert metadata["update"]["changed_parameter_tensors"] > 0
    assert metadata["cache_invalidated"]
    assert metadata["checkpoint_resume_roundtrip"]["global_step"] == 1
    assert metadata["checkpoint_resume_roundtrip"]["optimizer_state_entries"] > 0
