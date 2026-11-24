"""Time the frozen production full-rank control on the large registered combB mesh."""
import argparse
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[3]), str(HERE.parents[1] / "native")]
from registry import Registry, save_json
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS
from v21_native import native_operand
from experiments.compact_pino_v21.stratum.primal import retained_value
from v21_native_physics import evaluate_operand_physics


def run(out):
    from dataclasses import replace
    t0 = perf_counter()
    reg = Registry(DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS)
    geom = reg.example("combB")
    backend = reg.panel.direct(geom.request, geom.normalization, geom.physical.tolist())
    probe = next(p for p in geom.probes if p.id == 'combB_hyperbolic_plus')
    src = native_operand(backend,probe.query)
    t1 = perf_counter()
    target = evaluate_operand_physics(backend, replace(src, vertex_velocity=None), capture_boundary=True)
    t2 = perf_counter()
    if not target.moment_valid:
        raise ValueError(target.moment_diagnostic)
    w = target.boundary_witness
    fast=retained_value(backend,replace(src,vertex_velocity=None))
    t3=perf_counter()
    if (not np.array_equal(fast.moments, target.moments) or not np.array_equal(fast.vertices,w.vertices)
            or not np.array_equal(fast.faces,w.faces)):
        raise ValueError("optimized value and reference differ on registered combB")
    result = dict(fast_value_seconds=t3-t2, fast_path=fast.path, arithmetic=fast.arithmetic,
                  same_oriented_faces=True,same_target_vertices=True,same_certified_moments=True,
                  scope="Archived full-rank production timing and reconstruction control",
                  row_id=probe.id,identity=backend.context_identity, n_design=len(geom.physical),
                  n_retained=len(w.vertices), n_faces=len(w.faces), moments=target.moments,
                  reference_max_abs=float(np.max(abs(target.moments-probe.reference["M"]))),
                  prepare_seconds=t1-t0, frozen_value_seconds=t2-t1, native_stage_seconds=backend.stage_seconds,
                  peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save_json(out, result)
    print(result, flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)
