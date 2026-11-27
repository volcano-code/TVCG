"""Reproduce a synthetic CPU surface-kernel scheduling comparison.

Run from the repository root, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
      experiments.compact_pino_v21.runtime_efficiency.surface_benchmark \
      --output experiments/compact_pino_v21/runtime_efficiency/evidence/surface_cpu_benchmark.json

The old-style path creates a fresh pool for each chunk, using the same kernel,
point partitions and result order as the reusable context. This is a 12-face box
fixture, not a native cutting, neural inference, GPU or end-to-end benchmark.
The process temporarily restricts its own CPU affinity; no other process is
reconfigured. Compilation, geometry construction and correctness comparisons
are excluded from the timed call. Raw observations and environment are saved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import (
    v21_response_surface as surface,
    v23_surface_runtime as runtime,
)


def _box():
    vertices = np.asarray([[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.],
                           [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]])
    faces = np.asarray([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return vertices, faces


def _digest(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(repr((array.shape, str(array.dtype))).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _summary(samples):
    return dict(raw_seconds=samples, median_seconds=float(np.median(samples)),
                p95_seconds=float(np.quantile(samples, .95, method='linear')),
                mean_seconds=float(np.mean(samples)), minimum_seconds=min(samples),
                maximum_seconds=max(samples))


def run(*, cpus=2, point_count=16384, warmups=3, repeats=30, seed=830251):
    """Capture a new measurement; this does not reconstruct earlier stdout."""
    for name, number in (('cpus', cpus), ('point_count', point_count),
                         ('warmups', warmups), ('repeats', repeats)):
        if type(number) is not int or number < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        raise RuntimeError('this controlled comparison requires Linux CPU affinity support')
    if not hasattr(surface._query_batch, 'signatures'):
        raise RuntimeError('Numba is required: the Python fallback is not this benchmark')
    before = sorted(os.sched_getaffinity(0))
    chosen = before[:min(cpus, len(before))]
    if not chosen:
        raise RuntimeError('no CPUs available to this process')
    script = Path(__file__).resolve()
    root = script.parents[3]
    started = datetime.now(timezone.utc).isoformat()
    try:
        os.sched_setaffinity(0, chosen)
        vertices, faces = _box()
        prepared = surface.prepare_surface(vertices, faces)
        points = np.random.default_rng(seed).uniform(-1.4, 1.4, (point_count, 3))
        arguments = (*prepared._data, prepared.tolerance)
        # Full serial reference also compiles the unchanged Numba kernel before
        # either timed path, so startup compilation does not favor one scheduler.
        reference = surface._query_batch(points, *arguments)
        workers = min(8, len(chosen), max(1, point_count // 8192))

        def fresh():
            if workers == 1:
                return surface._query_batch(points, *arguments)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(surface._query_batch, part, *arguments)
                        for part in np.array_split(points, workers)]
                outputs = [job.result() for job in jobs]
                return tuple(np.concatenate(parts) for parts in zip(*outputs))

        observations = dict(fresh_pool=[], reusable_pool=[])
        pair_order = []
        with runtime.SurfaceExecutionContext(max_workers=workers) as execution:
            def reused():
                return execution.query(surface._query_batch, points, *arguments)

            for _ in range(warmups):
                for fn in (fresh, reused):
                    actual = fn()
                    for value, expected in zip(actual, reference):
                        np.testing.assert_array_equal(value, expected)
            final = {}
            for trial in range(repeats):
                order = ('fresh_pool', 'reusable_pool') if trial % 2 == 0 else ('reusable_pool', 'fresh_pool')
                pair_order.append(order)
                for name in order:
                    fn = fresh if name == 'fresh_pool' else reused
                    start = time.perf_counter_ns()
                    result = fn()
                    elapsed = (time.perf_counter_ns() - start) / 1e9
                    observations[name].append(elapsed)
                    # Validation remains outside every timing interval.
                    for value, expected in zip(result, reference):
                        np.testing.assert_array_equal(value, expected)
                    final[name] = result
            open_snapshot = execution.snapshot()
        closed_snapshot = execution.snapshot()
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        sources = (script, Path(surface.__file__).resolve(), Path(runtime.__file__).resolve())
        return dict(
            schema='compact-pino-v23-surface-scheduling-benchmark-v1', captured_utc=started,
            measurement_kind='new_capture_from_reproducible_script',
            scope='synthetic_only_cpu_kernel_scheduling',
            exclusions=['native finite-cut construction', 'BVH preparation', 'Numba compilation',
                        'exact ambiguity refinement after the kernel', 'neural response', 'GPU',
                        'native correction', 'independent acceptance', 'end-to-end solve'],
            fixture=dict(name='finite_oriented_box', vertices=len(vertices), triangles=len(faces),
                         points=point_count, seed=seed, point_distribution='uniform[-1.4,1.4]^3',
                         point_dtype=str(points.dtype), point_hash=_digest(points),
                         vertex_hash=_digest(vertices), face_hash=_digest(faces)),
            environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                             machine=platform.machine(), package_versions={name: _version(name) for name in
                                ('numpy', 'numba', 'llvmlite', 'torch')},
                             os_cpu_count=os.cpu_count(), affinity_before=before,
                             affinity_during=sorted(os.sched_getaffinity(0)),
                             requested_affinity_cpus=cpus, selected_workers=workers,
                             torch_intraop_threads=torch.get_num_threads(),
                             torch_interop_threads=torch.get_num_interop_threads(),
                             thread_environment={name: os.environ.get(name) for name in
                                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                             cgroup_cpu_max=_text('/sys/fs/cgroup/cpu.max'),
                             cgroup_cpuset_effective=_text('/sys/fs/cgroup/cpuset.cpus.effective')),
            protocol=dict(warmup_calls_per_path=warmups, timed_calls_per_path=repeats,
                          points_per_worker=8192, fresh_pool_workers=workers,
                          reusable_max_workers=workers, pair_order=pair_order,
                          timer='time.perf_counter_ns', quantile='numpy linear interpolation',
                          timing_boundary='full scheduler call including partition, submit, join and concatenate',
                          warmup_and_correctness_excluded=True,
                          cpu_load_not_isolated=True,
                          note='No speed assertion or deployment promotion threshold is inferred from this fixture.'),
            correctness=dict(all_timed_outputs_equal_to_serial_reference=True,
                             serial_output_hashes=[_digest(value) for value in reference],
                             output_hashes={name: [_digest(value) for value in result]
                                            for name, result in final.items()}),
            results={name: _summary(values) for name, values in observations.items()},
            execution_context=dict(before_close=open_snapshot, after_close=closed_snapshot),
            provenance=dict(git_head=commit, source_sha256={str(path.relative_to(root)):
                              hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}),
        )
    finally:
        os.sched_setaffinity(0, before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--points', type=int, default=16384)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=830251)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(cpus=args.cpus, point_count=args.points, warmups=args.warmups,
                  repeats=args.repeats, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output),
                          results={name: {key: value for key, value in result.items() if key != 'raw_seconds'}
                                   for name, result in receipt['results'].items()},
                          scope=receipt['scope']), indent=2))


if __name__ == '__main__':
    main()
