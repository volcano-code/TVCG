"""Integration checks for the existing V2.1 operator, without Pilot training.

These synthetic positive atoms exercise the production Token16 encoder and
response wiring. They do not establish native teacher accuracy, R3/R8 event
qualification, or the loss masks in a separate local training driver.
"""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import compact_pino_v21 as compact
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_blocks import BlockMeasure
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_native import FiniteOperand
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_response import prepare_surface


def _box(lo, hi):
    a, b, c = lo
    u, v, w = hi
    vertices = np.array([[a,b,c], [u,b,c], [u,v,c], [a,v,c],
                         [a,b,w], [u,b,w], [u,v,w], [a,v,w]], dtype=float)
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7],
                      [0,1,5], [0,5,4], [1,2,6], [1,6,5],
                      [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
    return vertices, faces


def _blocks(tmp_path, nodes=None):
    if nodes is None:
        nodes = np.array([[.501,.213,.337], [.503,.414,.272],
                          [.505,.641,.711], [.507,.782,.425]])
    nodes = np.asarray(nodes, dtype=float)
    weights = np.arange(1, len(nodes)+1, dtype=float)*1e-7
    material = SimpleNamespace(side=256, pitch=1/256., origin=np.zeros(3),
                               identity="synthetic-v21-training-contract")
    material.iter_chunks = lambda **kwargs: iter([(nodes, weights)])
    return BlockMeasure.build(material, tmp_path / "blocks", block_voxels=8)


def _operand(cut=.504, velocity=True):
    vertices, faces = _box([-2.,-2.,-2.], [cut,2.,2.])
    cap = vertices[:, 0] == cut
    motion = np.zeros((len(vertices), 3, 3))
    motion[cap, 0, 0] = 1.
    motion[cap, 0, 1] = vertices[cap, 1]
    motion[cap, 0, 2] = vertices[cap, 2]
    face_is_cap = np.all(cap[faces], axis=1)
    return FiniteOperand(vertices, faces, face_is_cap, "intersection",
                         motion if velocity else None)


def test_compact_import_and_construction_do_not_require_legacy_graph_extensions():
    # Fail even when the optional extensions happen to be installed on CI.
    script = r'''
import importlib.abc
import sys

class RejectLegacyExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch_scatter", "torch_cluster", "torch_geometric"}:
            raise ImportError("unexpected legacy graph extension: " + fullname)

sys.meta_path.insert(0, RejectLegacyExtensions())
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v21 import CompactPINOv21
model = CompactPINOv21()
assert model.token_grid == 16
assert model.shared_encoder.config.min_native_resolution == 256
assert model.shared_encoder.config.multiscale_grids == (32, 64)
assert not any(name.endswith("spin_quadrann_v12") or name.endswith("unified_m80_sqno") for name in sys.modules)
'''
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_production_encoder_j_only_update_and_cache_invalidation(tmp_path):
    torch.manual_seed(830251)
    model = compact.CompactPINOv21().train()
    assert model.shared_encoder.config.multiscale_grids == (32, 64)
    blocks = _blocks(tmp_path)
    fixed = np.zeros(10)
    cache = model.encode_geometry(blocks, fixed)
    assert cache.table.requires_grad and cache.table.dtype == torch.float32
    operand = _operand()
    prediction = model.evaluate_operand(cache, operand, chunk_size=2)
    assert prediction.jacobian is not None
    assert prediction.moments.dtype == prediction.jacobian.dtype == torch.float64
    assert prediction.diagnostics["ambiguous_nodes"] == 0

    # Keep the same learned geometry while rebuilding the finite operand.
    # This tests FP32 learned features combined with FP64 physical reduction.
    fd_errors = []
    direction = operand.vertex_velocity[..., 0]
    with torch.no_grad():
        for step in blocks.pitch*np.array([1e-2, 2e-3, 4e-4]):
            values = []
            for sign in (1., -1.):
                moved = FiniteOperand(operand.vertices + sign*step*direction,
                                      operand.faces, operand.face_is_cap,
                                      operand.operation)
                values.append(model.evaluate_operand(cache, moved, with_jacobian=False,
                                                      chunk_size=2).moments)
            fd = (values[0]-values[1])/(2*step)
            reference = prediction.jacobian[:, 0].detach()
            fd_errors.append(float(torch.linalg.vector_norm(fd-reference)
                                   / torch.linalg.vector_norm(reference)))
    assert fd_errors[-1] < 2e-6, fd_errors
    assert fd_errors[-1] < fd_errors[0], fd_errors

    # A synthetic J-only objective tests parameter transport, not model quality.
    derivative_scale = float(blocks.moments[:, 0].sum()/blocks.pitch)
    loss = (prediction.jacobian/derivative_scale).square().mean()
    loss.backward()
    prefixes = ("shared_encoder.pino_encoder.", "geometry_attention.", "response.")
    watched = {}
    for prefix in prefixes:
        group = [(name, p) for name, p in model.named_parameters()
                 if name.startswith(prefix) and p.grad is not None]
        assert group, prefix
        assert all(torch.isfinite(p.grad).all() for _, p in group), prefix
        name, parameter = max(group, key=lambda item: float(item[1].grad.abs().max()))
        assert float(parameter.grad.abs().max()) > 0., prefix
        watched[name] = parameter.detach().clone()
    # One temporary software test update. No checkpoint or training run is saved.
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    optimizer.step()
    current = dict(model.named_parameters())
    for name, before in watched.items():
        assert not torch.equal(current[name], before), name
    with pytest.raises(ValueError, match="stale"):
        model.evaluate_operand(cache, operand)


def test_invalid_chunk_withholds_whole_packed_jacobian(tmp_path):
    # First atom has a unique closest face; second is on the medial axis.
    blocks = _blocks(tmp_path, [[.99,.2,.3], [.5,.5,.5]])
    surface = prepare_surface(*_box([0.,0.,0.], [1.,1.,1.]))
    motion = np.ones((8, 3, 3))
    result = compact.evaluate_packed(blocks, surface, .6, "intersection",
                                     np.zeros(10), motion, chunk_size=1)
    assert result.diagnostics["chunks"] == 2
    assert result.diagnostics["ambiguous_nodes"] == 1
    assert torch.isfinite(result.moments).all()
    assert result.jacobian is None
    assert result.diagnostics["solver_status"] == "NATIVE_REQUIRED_DERIVATIVE"
