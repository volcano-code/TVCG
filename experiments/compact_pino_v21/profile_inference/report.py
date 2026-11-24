"""Pair measured query timing to the original archived seed-zero callback journal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def summarize(rows):
    good = [r for r in rows if r['defined']]
    out = dict(attempts=len(rows), accepted=len(good), rejected_calls=[r['call'] for r in rows if not r['defined']],
                all_callback_wall_sum_s=sum(r['callback_wall_s'] for r in rows),
                all_callback_median_s=statistics.median(r['callback_wall_s'] for r in rows),
                median_excluding_first_s=statistics.median(r['callback_wall_s'] for r in rows[1:]) if len(rows)>1 else None,
                source_wall_sum_s=sum(r['source_wall_s'] for r in rows),
                surface_index_wall_sum_s=sum(r['surface_index_wall_s'] for r in good),
                accepted_response_wall_sum_s=sum(r['solve_response_wall_s'] for r in good),
                accepted_response_median_s=statistics.median(r['solve_response_wall_s'] for r in good))
    if 'host_cpu_busy_total_ticks' in rows[0]:
        out['host_cpu_busy_fraction'] = {}
        for cpu in rows[0]['host_cpu_busy_total_ticks']:
            busy, total = zip(*(r['host_cpu_busy_total_ticks'][cpu] for r in rows))
            out['host_cpu_busy_fraction'][cpu] = sum(busy) / max(1, sum(total))
    return out


def pair_archive(rows, archive):
    if len(rows) != len(archive):
        raise ValueError('callback count mismatch with solver archive')
    for row, native in zip(rows,archive):
        if row['query'] != native['q'] or row['call'] != native['index']:
            raise ValueError('raw query or chronological index mismatch')
        if row['defined'] == ('error' in native) or row.get('error') != native.get('error'):
            raise ValueError('source acceptance or rejection diagnostic mismatch')
        row['archived_native_joint_s'] = native['seconds']


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1<<20):
            digest.update(block)
    return digest.hexdigest()


def render(j):
    lines = ['# Noodles deterministic inference profile', '',
             'The two workers each evaluated the 25 recorded seed-0 queries twice on CPUs 6 and 7. Each request rebuilt the native moving operand, ran the production physical-boundary audit, and built its surface index. Query 3 reproduced the archived source rejection in every pass. The M-only entry pays the full source-motion construction cost used by the joint entry.', '',
             '| Entry | Process setup (s) | First-query JIT (s) | Cold 25 callbacks (s) | Warm 25 callbacks (s) | Warm median (s) | Peak process RSS (GiB) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for k,name in [('M','M only'),('joint','M and raw J')]:
        mode=j['modes'][k]
        lines.append(f"| {name} | {mode['setup']['before_first_call_wall_s']:.2f} | {mode['passes']['cold'][0]['first_point_jit_wall_s']:.2f} | {mode['summary']['cold']['all_callback_wall_sum_s']:.2f} | {mode['summary']['warm']['all_callback_wall_sum_s']:.2f} | {mode['summary']['warm']['all_callback_median_s']:.3f} | {mode['process_peak_rss_bytes']/2**30:.3f} |")
    lines += ['', '| Warm pass, 25 requests | Source and physical audit (s) | Surface indexes (s) | Local readouts (s) |',
              '|---|---:|---:|---:|']
    for k,name in [('M','M only'),('joint','M and raw J')]:
        part=j['modes'][k]['summary']['warm']
        lines.append(f"| {name} | {part['source_wall_sum_s']:.2f} | {part['surface_index_wall_sum_s']:.2f} | {part['accepted_response_wall_sum_s']:.2f} |")
    a=j['archive']
    pack=j['modes']['joint']['pack']
    lines.extend(['', f"The original native solver journal records **{a['callback_seconds_sum']:.2f} s** across these 25 joint callbacks on the same CPU IDs, including one rejection. Its warm median excluding the first callback was {a['warm_median_joint_call_s']:.3f} s. Applying that exclusion to V2.1's warm joint replay gives {j['modes']['joint']['summary']['warm']['median_excluding_first_s']:.3f} s. The archived native process peaked at {a['peak_process_rss_gib']:.3f} GiB. Its verified solve, including independent verification, took {a['warm_solve_plus_verification_s']:.2f} s. These historical values were collected in a separate run.", '',
             f"The frozen material pack contains {pack['nodes']:,} positive nodes and {pack['blocks']:,} occupied blocks. Its preparation record separately lists {pack['precomputed_material_s']:.3f} s for material preparation and {pack['precomputed_block_index_s']:.3f} s for indexing. The frozen-pack profile starts from those assets.", '',
             'Each worker started a fresh process with an empty Numba compiler cache. Both passes used the same frozen positive material index. The operating-system page cache was available. Cold callback totals include first-point JIT and exclude static setup. The JSON includes process startup, static setup, source, index, reductions, callback wall and CPU times, current RSS, 10 ms sampled per-query peaks, and cumulative kernel RSS.', '',
             'CPU affinity was pinned on a shared host. The measured runs replayed fixed queries. They performed zero solver steps, line searches, fallbacks, and independent final verifications. Qualified solve time and learned-model speed remain unmeasured.', '',
             'Reproduce from the worktree root with `bash experiments/compact_pino_v21/profile_inference/reproduce.sh /path/to/output`. To rebuild the static pack from a registered geometry export, run `python -m experiments.compact_pino_v21.pack_geometry --native-root /path/to/native --indexes /path/to/indexes noodles`, then set `V21_NOODLES_PACK=/path/to/indexes/noodles` for the launcher. The JSON binds the inputs and executable sources by SHA-256. The archive export is documented in [the native reproduction instructions](../native/README.md).'])
    if 'host_cpu_busy_fraction' in j['modes']['joint']['summary']['warm']:
        util=j['modes']['joint']['summary']['warm']['host_cpu_busy_fraction']
        text=(f"Across the joint warm callbacks, Linux reported {100*util['cpu22']:.1f}% busy time on sibling CPU 22 "
              f"and {100*util['cpu23']:.1f}% on sibling CPU 23. The JSON includes raw scheduler ticks for every callback.")
        if 'earlier_shared_host_profile' in j:
            text += " A first complete run overlapped a chart diagnostic on CPU 23. The earlier output is retained in [the shared-host run](shared_host_first/summary.md)."
        lines[-1:-1]=[text,'']
    return '\n'.join(lines)+'\n'


def run(args):
    root=Path(__file__).resolve().parents[3]
    here=Path(__file__).resolve().parent
    archivepath=args.alignment/'reference/seed-00.json'
    journalpath=args.alignment/'reference/seed-00.progress.jsonl'
    archive=json.loads(archivepath.read_bytes())
    journal=[json.loads(line) for line in journalpath.read_text().splitlines()]
    if [r['index'] for r in journal] != list(range(1,26)) or archive['seed']!=0 or archive['allowed_cpu_ids']!=[6,7]:
        raise ValueError('native timing archive is not the frozen seed-0 CPU 6,7 trajectory')
    if abs(sum(r['seconds'] for r in journal)-archive['callback_seconds_sum'])>1e-6:
        raise ValueError('archived per-query timings fail to sum to the solver record')
    modes={}
    seen={}
    for k in ('M','joint'):
        worker=json.loads((args.work/(k+'.json')).read_bytes())
        worker['external_process_wall_s']=float((args.work/(k+'.elapsed')).read_text())
        worker['summary']={}
        for phase,rows in worker['passes'].items():
            pair_archive(rows,journal)
            worker['summary'][phase]=summarize(rows)
            for row in rows:
                if not row['defined']:
                    continue
                for field in ('M_sha256','J_sha256'):
                    value=row[field]
                    if value is None:
                        continue
                    key=(row['call'],field)
                    if seen.setdefault(key,value)!=value:
                        raise ValueError('cold/warm or M-only/joint numerical result differs')
        modes[k]=worker
    adapter=root/'research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter'
    files=[archivepath,journalpath,args.alignment/'reference/trajectory_M_J.json',args.alignment/'run.sh',
           args.alignment.parent/'phase3_1b/solver/batch.sh',args.alignment.parent/'phase3_1b/solver/launch.sh',
           here/'worker.py',here/'report.py',here/'reproduce.sh',root/'experiments/compact_pino_v21/native/registry.py',
           root/'experiments/compact_pino_v21/native/host_inputs.json',args.pack/'identity.json',
           *(args.pack/'conservative'/n for n in ('blocks.npz','metadata.json','nodes.npy')),
           *(adapter/n for n in ('compact_pino_v21.py','v21_response.py','v21_response_surface.py','v21_native.py','v21_blocks.py')),
           Path(modes['M']['runtime']['native_extension']),Path(modes['M']['runtime']['executable']),
           Path(modes['M']['runtime']['ld_preload'])]
    j=dict(schema='tvcg.compact-pino-v2.1.inference-profile.1', seed=0, mode='fixed-query deterministic M or M+raw-J, no training',
           native_source_construction='live deployed source-motion entry, unit-to-raw pullback and exact physical source boundary audit, no material Boolean',
           cold='fresh worker, empty JIT cache, frozen prepared material; OS file cache not evicted',
           warm='second ordered trajectory in same worker; only static assets and loaded code are reused',
           wall_scope='callback includes source, spatial index, optional first-point JIT, response and instrument validation; setup separate',
           rss_scope='Linux /proc/self/statm sampled every 10 ms; ru_maxrss is a cumulative lifetime peak, including imports and pinned registry',
           workers='M-only then joint, sequential; each uses taskset 6,7, nice 10, two library threads, one torch interop thread, CPU only',
           limitations='fixed query sequence; timings exclude optimization and independent final verification. Additional archived static material preparation is reported separately. No learned weights or qualified solve performance evaluated.',
           inputs_sha256={str(p):sha(p) for p in files},
           archive={k:archive[k] for k in ('allowed_cpu_ids','threads','calls','cold_setup_s','cold_first_joint_call_s',
           'warm_median_joint_call_s','callback_failure_count','solve_s','callback_seconds_sum','verification_s',
           'warm_solve_plus_verification_s','cold_through_acceptance_s','peak_process_rss_gib')},modes=modes)
    j['archive']['source']='historical native seed-0 verified solver and per-call callback journal; CPU RSS covers its whole solve'
    j['archive']['warm_median_excluding_first_s_recomputed']=statistics.median(r['seconds'] for r in journal[1:])
    j['cpu_environment']={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                          'NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHON_CPU_COUNT','MALLOC_ARENA_MAX','LD_PRELOAD',
                          'CUDA_VISIBLE_DEVICES','PYTHONPATH')}
    j['cpu_model']=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name'))
    cpuroot=Path('/sys/devices/system/cpu')
    j['cpu_topology']={str(i):{k:(cpuroot/f'cpu{i}'/n).read_text().strip() for k,n in
                       [('core_id','topology/core_id'),('logical_siblings','topology/thread_siblings_list'),
                        ('governor','cpufreq/scaling_governor')] if (cpuroot/f'cpu{i}'/n).is_file()} for i in (6,7)}
    first = here/'shared_host_first/profile.json'
    if first.is_file():
        prev=json.loads(first.read_bytes())
        j['earlier_shared_host_profile']=dict(sha256=sha(first),file='shared_host_first/profile.json',
            observed_interference='Native chart diagnostic PID 830090 on CPU 23 (sibling of CPU 7) from 22:18:14 on November 24 2026, during the end of the joint warm pass. Both workers rerun after moving that job to CPU 26.',
            earlier_warm_callback_seconds={k:v['summary']['warm']['all_callback_wall_sum_s'] for k,v in prev['modes'].items()})
    for name,obj in [('profile.json',json.dumps(j,indent=1,allow_nan=False)+'\n'),('summary.md',render(j))]:
        tmp=args.output/(name+'.tmp');tmp.write_text(obj);tmp.replace(args.output/name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','alignment','output','pack'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
