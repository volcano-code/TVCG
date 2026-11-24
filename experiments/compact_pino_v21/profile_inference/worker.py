"""Single-process two-core native source -> deterministic V2.1 response profile."""
import time
PROCESS_START = time.perf_counter()

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / 'experiments/compact_pino_v21/native'
sys.path.insert(0, str(ROOT))
# Pin the live-worktree package before the archived registry adds its own
# historical checkout to the module search path for the deployed runtime.
import numba
import numpy as np
import torch
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter import (
    compact_pino_v21 as compact, v21_blocks, v21_native, v21_response, v21_response_surface,
)
sys.path.insert(0, str(NATIVE))
from registry import Registry, sha


class RSS:
    """Sample current Linux process RSS, also record the kernel lifetime high water."""
    def __init__(self):
        self.page_bytes = os.sysconf('SC_PAGE_SIZE')
        self.peak = 0
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def current(self):
        with open('/proc/self/statm', encoding='ascii') as stat:
            return int(stat.read().split()[1])*self.page_bytes

    def _sample(self):
        while not self.done.wait(.01):
            with self.lock:
                self.peak = max(self.peak, self.current())

    def start_call(self):
        with self.lock:
            self.peak = self.current()
            return self.peak

    def close(self):
        self.done.set()
        self.thread.join()


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()


def wall_cpu():
    return time.perf_counter(), time.process_time()


def finish_stage(row, key, start):
    wall, cpu = wall_cpu()
    row[key+'_wall_s'] = wall - start[0]
    row[key+'_cpu_s'] = cpu - start[1]


def cpu_ticks():
    """Linux scheduler ticks for the target CPUs and their hardware siblings."""
    selected = {}
    with open('/proc/stat', encoding='ascii') as stat:
        for line in stat:
            name, *values = line.split()
            if name in ('cpu6','cpu7','cpu22','cpu23'):
                user, nice, system, idle, wait, irq, softirq, steal = map(int, values[:8])
                active = user + nice + system + irq + softirq + steal
                selected[name] = (active, active+idle+wait)
            if name == 'intr':
                break
    return selected


def benchmark_query(probe, geometry, backend, blocks, mode, cold_jit, rss):
    q = np.asarray(probe.query, dtype=np.float64)
    row = dict(call=int(probe.id.split('_')[-1]), query=q.tolist(), defined=False,
               rss_at_start_bytes=rss.start_call())
    ticks = cpu_ticks()
    clock = wall_cpu()
    source_start = clock
    try:
        # Exact deployed source, side selector, physical-space boundary audit,
        # and unit-to-raw motion chain. No material Boolean or saved operand.
        operand = v21_native.native_operand(backend, q)
    except (RuntimeError, ValueError, ArithmeticError) as e:
        row.update(error=str(e), exception=type(e).__name__)
        finish_stage(row, 'source', source_start)
    else:
        finish_stage(row, 'source', source_start)
        t = wall_cpu()
        surface = v21_response.prepare_surface(operand.vertices, operand.faces)
        finish_stage(row, 'surface_index', t)
        if cold_jit:
            before = len(v21_response_surface._query_batch.signatures)
            t = wall_cpu()
            surface.signed_distance(blocks.centers[:1])
            finish_stage(row, 'first_point_jit', t)
            row['jit_signatures'] = [before, len(v21_response_surface._query_batch.signatures)]
            if row['jit_signatures'] != [0, 1]:
                raise RuntimeError('profile requires a fresh JIT with first-call compilation')
        t = wall_cpu()
        y = compact.evaluate_packed(blocks, surface, blocks.pitch, operand.operation,
                                    geometry['fixed_moments'],
                                    operand.vertex_velocity if mode == 'joint' else None)
        finish_stage(row, 'solve_response', t)
        if not torch.isfinite(y.moments).all() or (mode == 'joint' and
                (y.jacobian is None or not torch.isfinite(y.jacobian).all())):
            raise ValueError(f'nonfinite or undefined numerical response for {probe.id}')
        row.update(defined=True, operation=operand.operation, nv=len(operand.vertices), nf=len(operand.faces),
                   cap_faces=int(operand.face_is_cap.sum()), M0=float(y.moments[0]), M_sha256=sha_array(y.moments),
                   J_sha256=sha_array(y.jacobian) if mode=='joint' else None, **y.diagnostics)
    # Numerical validation, array digest and metadata are instrument overhead;
    # retain the inclusive callback time as the primary measured total.
    finish_stage(row, 'callback', clock)
    now = cpu_ticks()
    row['host_cpu_busy_total_ticks'] = {c: [now[c][i]-v[i] for i in (0,1)] for c,v in ticks.items()}
    row['rss_at_end_bytes'] = rss.current()
    row['rss_sampled_max_bytes'] = max(rss.peak, row['rss_at_end_bytes'])
    row['process_peak_rss_through_call_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return row


def run(args):
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [6, 7] or os.cpu_count() != 2 or 'libstdc++.so.6' not in os.environ.get('LD_PRELOAD',''):
        raise RuntimeError('run via the pinned launcher for CPUs 6,7 and the official ABI')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    setup = dict(import_wall_s=time.perf_counter()-PROCESS_START)
    t = wall_cpu()
    registry = Registry(args.alignment,args.corpus,args.physics)
    noodles = registry.noodles()
    if [p.id for p in noodles.probes] != [f'noodles_{k:02d}' for k in range(1, 26)]:
        raise ValueError('historical seed-0 callback order mismatch')
    finish_stage(setup, 'pinned_registry_and_trajectory', t)
    t = wall_cpu()
    backend = registry.panel.direct(noodles.request, noodles.normalization, np.asarray(noodles.physical).tolist())
    finish_stage(setup, 'deployed_static_backend', t)
    t = wall_cpu()
    geometry = json.loads((args.pack/'identity.json').read_bytes())
    if geometry['geometry'] != 'noodles' or geometry['side'] != 256:
        raise ValueError('expected the frozen Noodles native256 pack')
    material = geometry['controls']['conservative']
    blocks = v21_blocks.BlockMeasure.open(args.pack/'conservative', identity=material['identity'], block_voxels=8)
    if len(blocks.nodes) != material['nodes'] or not np.array_equal(backend.fixed,geometry['fixed_moments']):
        raise ValueError('frozen static material or fixed shell mismatch')
    finish_stage(setup, 'open_static_pack', t)
    rss = RSS()
    setup['before_first_call_wall_s'] = time.perf_counter()-PROCESS_START
    setup['rss_at_first_call_bytes'] = rss.current()
    output = dict(mode=args.mode, setup=setup, passes={}, pid=os.getpid(), physical_host=platform.node(),
                  runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                               torch=torch.__version__, numba=numba.__version__, platform=platform.platform(),
                               affinity=affinity, visible_logical_cpus=os.cpu_count(), nice=os.getpriority(os.PRIO_PROCESS,0),
                               torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                               numba_threads=numba.get_num_threads(), gpu_visible=os.getenv('CUDA_VISIBLE_DEVICES'),
                               response_file=compact.__file__, native_extension=backend.native.__file__,
                               allocator_arenas=os.environ.get('MALLOC_ARENA_MAX'), ld_preload=os.environ.get('LD_PRELOAD')),
                  backend=dict(context=backend.context_identity, boolean_binding=backend.backend_identity,
                               production_profile_sha256=sha(registry.panel.PROFILE_PATH),
                               arithmetic=backend.arithmetic_identity),
                  pack=dict(identity=blocks.identity, nodes=len(blocks.nodes), blocks=len(blocks.ids), pitch=blocks.pitch,
                            precomputed_material_s=material['material_s'], precomputed_block_index_s=material['block_index_s']))
    with torch.inference_mode():
        for phase in ('cold', 'warm'):
            rows=[]
            for i, probe in enumerate(noodles.probes):
                row = benchmark_query(probe, geometry, backend, blocks, args.mode, phase=='cold' and i==0, rss)
                rows.append(row)
                print(args.mode,phase,row['call'],round(row['callback_wall_s'],3),row.get('error',''),flush=True)
            output['passes'][phase]=rows
    rss.close()
    output['worker_measured_wall_s'] = time.perf_counter()-PROCESS_START
    output['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    output['no_material_evaluations'] = not backend._materials and not backend.last
    if not output['no_material_evaluations']:
        raise RuntimeError('profiling unexpectedly entered the native material evaluation path')
    args.output.write_text(json.dumps(output,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--physics',type=Path,required=True)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--mode',choices=('M','joint'),required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
