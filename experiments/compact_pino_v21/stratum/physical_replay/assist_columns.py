"""Run selected pending pairs on an existing authenticated center, optionally timing their hot functions."""
import argparse
import cProfile
import fcntl
from functools import lru_cache
import multiprocessing
import os
from pathlib import Path
import pstats
import sys
from time import perf_counter,process_time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import replay_registered as replay


def trace_stages(backend,operator):
    def watch(owner,name):
        old=getattr(owner,name)
        def observed(*args,**kwargs):
            start=perf_counter()
            cpu=process_time()
            print("begin",name,flush=True)
            try:
                return old(*args,**kwargs)
            finally:
                print("finish",name,round(perf_counter()-start,3),round(process_time()-cpu,3),flush=True)
        setattr(owner,name,observed)
    watch(operator.extension,"source_value")
    watch(backend.native,"audit_exact_triangle_boundary")
    watch(backend.bridge,"boolean")
    watch(replay,"retained_value")
    watch(replay.checks.PhysicalVertexIDs,"evaluate")


@lru_cache(maxsize=1)
def loaded_center(row_id):
    args,out,geom,backend,operator,design,probes=WORK
    return replay.center_checkpoint(out,geom,probes[row_id],backend,backend,operator,design,args.asset,"fast")


def pin_worker(cpus):
    number=multiprocessing.current_process()._identity[-1]-1
    os.sched_setaffinity(0,{cpus[number%len(cpus)]})


def schedule_pairs(probes,out):
    prepared=[]
    for row_id,probe in probes.items():
        path=out/"centers"/(row_id+".json")
        center=replay.read(path)
        if center.get("source") or center.get("primal"):
            prepared.append((center["center_seconds"],row_id,probe.chart.basis.shape[1]))
    jobs=[]
    for _,row_id,dim in sorted(prepared,reverse=True):
        for step in range(len(replay.checks.FD["raw_steps"])):
            jobs.extend(f"{row_id}/{step}/{col}" for col in range(dim))
    return jobs


def run_claimed_pair(spec):
    args,out,geom,backend,operator,design,probes=WORK
    row_id,step,col=spec.split("/")
    si,k=int(step),int(col)
    probe=probes[row_id]
    if not 0<=si<len(replay.checks.FD["raw_steps"]) or not 0<=k<probe.chart.basis.shape[1]:
        raise ValueError("requested step or raw column lies outside this named chart")
    h=replay.checks.FD["raw_steps"][si]
    path=out/"parts"/row_id/(f"s{si}_c{k:02d}.json")
    if path.exists():
        if replay.read(path)["run_fingerprint"]!=replay.fingerprint_run(operator,probe,"fast"):
            raise ValueError("existing pair has a different scientific fingerprint")
        return
    row,base,target,ids=loaded_center(row_id)
    if base is None:
        return
    work=lambda:replay.step_column(probe,backend,operator,base,target,ids,design,h,k,"fast")
    if args.profile:
        profiler=cProfile.Profile()
        part=profiler.runcall(work)
        profile_file=out/"logs"/(f"profile_{row_id}_s{si}c{k}.txt")
        with profile_file.open("w") as handle:
            pstats.Stats(profiler,stream=handle).strip_dirs().sort_stats("cumulative").print_stats(40)
    else:
        part=work()
    part["run_fingerprint"]=row["run_fingerprint"]
    if path.exists():
        old=replay.read(path)
        if (old["run_fingerprint"]!=part["run_fingerprint"] or
                not np.array_equal(old.get("material_FD"),part.get("material_FD"))):
            raise ValueError("concurrent scientific result differs")
    else:
        replay.save_json(path,part)
    print(row_id,si,k,part["passed_sampled_column"],flush=True)

def run_pair(spec):
    args,out,_,_,_,_,probes=WORK
    row_id,step,col=spec.split("/")
    if row_id not in probes or int(step) not in range(len(replay.checks.FD["raw_steps"])) or int(col) not in range(probes[row_id].chart.basis.shape[1]):
        raise ValueError("pair does not belong to the registered asset and chart")
    lockfile=out/"pair_locks"/row_id/f"s{int(step)}_c{int(col):02d}.lock"
    lockfile.parent.mkdir(parents=True,exist_ok=True)
    with lockfile.open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            # Another pair worker is already constructing exactly these two physical cuts.
            return
        run_claimed_pair(spec)


def run(args):
    registry=replay.Registry(replay.DEFAULT_ALIGNMENT,replay.DEFAULT_CORPUS,replay.DEFAULT_PHYSICS)
    geom=registry.example(args.asset)
    backend=registry.panel.direct(geom.request,geom.normalization,np.asarray(geom.physical).tolist())
    operator=replay.SharedStratumOperator.from_evaluator(backend)
    design=replay.checks.PhysicalVertexIDs.index_design(backend.design)
    probes={p.id:p for p in geom.probes if p.chart}
    out=args.out.resolve()
    if args.trace: trace_stages(backend,operator)
    global WORK
    WORK=(args,out,geom,backend,operator,design,probes)
    jobs=schedule_pairs(probes,out) if args.pending else args.pairs
    if args.workers==1:
        for spec in jobs:
            run_pair(spec)
        return
    cpus=[int(cpu) for cpu in args.cpus.split(",")] if args.cpus else sorted(os.sched_getaffinity(0))
    if len(cpus)<args.workers or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):
        raise ValueError("provide one available CPU per pair worker")
    with multiprocessing.get_context("fork").Pool(args.workers,initializer=pin_worker,initargs=(cpus,)) as pool:
        for _ in pool.imap_unordered(run_pair,jobs,chunksize=1):
            pass


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True,choices=replay.ASSETS)
    p.add_argument("--out",required=True,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs",nargs="+",help="row_id/step_index/column_index")
    group.add_argument("--pending",action="store_true",help="complete every registered center and column for this asset")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--cpus",help="comma-separated distinct CPU numbers for forked workers")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--trace",action="store_true")
    run(p.parse_args())
