"""Replay the registered Noodles source queries through a frozen checkpoint.

This measures callback execution, not Newton convergence or end-to-end speedup.
Use the Work host's pinned native launcher and a fresh process for each arm.
The registry validates its original-mesh inputs. Saved operands or teacher
moments are never supplied to the candidate.
"""
from __future__ import annotations
import time
PROCESS_START = time.perf_counter()

import argparse
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

# Import live model code before the registry attaches archived native paths.
from .deployment import make_predictor, sha256
from .metrics import summarize_queries
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v23 import RuntimeConfig


def run(args):
    output = Path(args.output)
    # One immutable experiment directory; never overwrite an earlier receipt.
    output.mkdir(parents=True, exist_ok=False)
    from experiments.compact_pino_v21.native.registry import Registry
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    registry = Registry(args.alignment, args.corpus, args.physics)
    geometry = registry.noodles()
    pack_info = json.loads((Path(args.pack)/"identity.json").read_text())
    if pack_info["rle_sha256"] != sha256(geometry.rle):
        raise ValueError("pack grid differs from the frozen registered geometry")
    backend = registry.panel.direct(geometry.request, geometry.normalization,
                                   np.asarray(geometry.physical).tolist())
    metadata = dict(
        scope="registered_query_replay_not_closed_loop",
        geometry="noodles", device=args.device,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        python_platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
        runner_sha256=sha256(__file__), registered_geometry_files=geometry.hashes,
        mode=args.mode, passes=args.passes, profile_sync=args.profile_sync,
        formal_cuda_profile=bool(args.profile_sync),
        cold_material_generation_included=False,
        shared_execution_updates_in_v22_arm=True)
    traces, predictions = [], []
    with (output/"query_traces.jsonl").open("w", buffering=1) as journal:
        def record(row):
            traces.append(row)
            journal.write(json.dumps(row, allow_nan=False)+"\n")
        runtime = RuntimeConfig(surface_workers=args.surface_workers) if args.arm == "v23" else None
        with make_predictor(backend, args.pack, args.checkpoint, metadata,
                expected_geometry="noodles", expected_pack_sha256=args.pack_sha256,
                device=args.device, arm=args.arm, runtime=runtime,
                trace_sink=record, profile_sync=args.profile_sync) as predictor:
            metadata["setup_before_first_query_s"] = time.perf_counter()-PROCESS_START
            for pass_index in range(args.passes):
                for probe in geometry.probes:
                    row = dict(pass_index=pass_index, query_id=probe.id,
                               query=np.asarray(probe.query).tolist(), source=probe.origin)
                    try:
                        result = predictor.evaluate(probe.query, with_jacobian=args.mode == "joint")
                        row.update(defined=True, moments=result["raw_moments"].tolist(),
                                   jacobian=None if result["moment_jacobian"] is None else result["moment_jacobian"].tolist(),
                                   jacobian_coordinate="deployed_unit_geometric" if args.mode == "joint" else None,
                                   requires_native_acceptance=True)
                    except (ValueError, RuntimeError, ArithmeticError) as error:
                        row.update(defined=False, error_type=type(error).__name__, error=str(error))
                    predictions.append(row)
    metadata.update(replay_measurement_wall_s=time.perf_counter()-PROCESS_START,
        result_serialization_excluded_from_measurement=True,
        full_job_wall_s=None,
        no_native_retained_material_queries=not bool(backend._materials) and not bool(backend.last),
        query_summary=summarize_queries(traces))
    (output/"predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False)+"\n")
    (output/"summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    if not metadata["no_native_retained_material_queries"]:
        raise RuntimeError("replay entered a native material evaluation; inspect recorded evidence")
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment", "corpus", "physics", "pack", "checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pack-sha256", required=True)
    p.add_argument("--arm", choices=("v22", "v23"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=("M", "joint"), default="joint")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--surface-workers", type=int)
    p.add_argument("--profile-sync", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.passes < 1 or args.threads < 1:
        p.error("passes and threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
