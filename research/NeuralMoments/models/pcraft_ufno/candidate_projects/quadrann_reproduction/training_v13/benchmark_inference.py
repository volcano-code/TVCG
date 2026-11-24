"""Synchronized cached-vs-uncached inference benchmark (no training)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..spin_adapter.unified_factory import build_model
from .native256_readiness import admit_native_training
from .unified_corpus import UnifiedOperatorCorpus, file_sha256


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, queries, device, warmup, repeats):
    with torch.no_grad():
        for i in range(warmup):
            fn(queries[i % len(queries)])
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed=[]
        for i in range(repeats):
            q=queries[i % len(queries)]
            _sync(device)
            start=time.perf_counter()
            fn(q)
            _sync(device)
            elapsed.append(time.perf_counter()-start)
    return {"seconds":elapsed,"p50_seconds":float(np.percentile(elapsed,50)),
            "p95_seconds":float(np.percentile(elapsed,95)),"p99_seconds":float(np.percentile(elapsed,99)),"mean_seconds":float(np.mean(elapsed)),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--native-cut-contract",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--config",default="compact_v1_candidate_256")
    p.add_argument("--geometry-id",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=30);p.add_argument("--cache-mib",type=int,default=256)
    p.add_argument("--allow-absolute-assets",action="store_true")
    args=p.parse_args()
    if args.warmup<0 or args.repeats<1 or args.cache_mib<0:
        p.error("invalid measurement budget")
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    device=torch.device(args.device)
    corpus=UnifiedOperatorCorpus(args.manifest,grid_size=64,allow_absolute_paths=args.allow_absolute_assets)
    admission=admit_native_training(corpus,args.config,args.native_cut_contract)
    if admission is None:
        raise ValueError("benchmark requires an admitted native256 configuration")
    row=next((r for r in corpus.splits["train"]+corpus.splits["val"] if r["id"]==args.geometry_id),None)
    if row is None:
        raise ValueError("geometry must belong to train/development, not locked benchmark")
    geom,labels=corpus.load(row)
    q=torch.as_tensor(labels["queries"][labels["valid_mask"]].copy(),device=device,dtype=torch.float64)
    model=build_model(args.config,device=device)
    payload=model.load_checkpoint(args.checkpoint,expected_data_fingerprint=corpus.manifest_sha256)
    extra=payload.get("extra",{})
    config=extra.get("config",{})
    receipt=config.get("native_cut_admission") or extra.get("native_cut_admission")
    if not isinstance(receipt,dict) or receipt.get("contract_sha256")!=admission["contract_sha256"]:
        raise ValueError("checkpoint native cut receipt mismatch; use a new native256 training/preflight checkpoint")
    model.eval()
    _sync(device);start=time.perf_counter()
    cache=model.encode_geometry(geom,inference=True)
    _sync(device);encoding=time.perf_counter()-start
    base_m=_measure(lambda x:model.predict_moments(cache,x),q,device,args.warmup,args.repeats)
    base_j=_measure(lambda x:model.evaluate_queries(cache,x),q,device,args.warmup,args.repeats)
    _sync(device);start=time.perf_counter()
    intrinsic=getattr(model,"operator_kind",None)=="compact_pino_v2"
    if intrinsic:
        # V2 stores token K/V and physical moments as part of its one encoding;
        # it has no uncached full-domain SQNO path or optional V1 acceleration.
        fast=cache
        cache_report={"mode":"intrinsic_v2_token_memory","physical_tokens":len(cache.token_cache.moments),
                      "native_intervals":cache.native_intervals,"split_segments":cache.split_segments,
                      "optional_acceleration_applicable":False}
    else:
        fast,cache_report=model.prepare_inference_acceleration(cache,max_extra_bytes=args.cache_mib*1024**2)
    _sync(device);setup=time.perf_counter()-start
    errors={"moment_abs_max":0.,"jacobian_abs_max":0.,"moment_only_abs_max":0.}
    # All valid queries get a parity check, independent of the timed subset.
    with torch.no_grad():
        for x in q:
            direct=model.evaluate_queries(cache,x)
            cached=model.evaluate_queries(fast,x)
            m=model.predict_moments(fast,x)
            for key,name in (("moments","moment_abs_max"),("jacobian_coefficients","jacobian_abs_max")):
                torch.testing.assert_close(cached[key],direct[key],rtol=2e-6,atol=2e-8)
                errors[name]=max(errors[name],float((cached[key]-direct[key]).abs().max()))
            torch.testing.assert_close(m,direct["moments"],rtol=2e-6,atol=2e-8)
            errors["moment_only_abs_max"]=max(errors["moment_only_abs_max"],float((m-direct["moments"]).abs().max()))
    fast_m=None if intrinsic else _measure(lambda x:model.predict_moments(fast,x),q,device,args.warmup,args.repeats)
    fast_j=None if intrinsic else _measure(lambda x:model.evaluate_queries(fast,x),q,device,args.warmup,args.repeats)
    result={"schema":"tvcg.native256.inference-benchmark.v1","device":str(device),
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "torch_version":str(torch.__version__),"cuda_version":torch.version.cuda,
            "config":args.config,"geometry_id":row["id"],"geometry_sha256":row["geometry_sha256"],
            "checkpoint_sha256":file_sha256(args.checkpoint),"manifest_sha256":corpus.manifest_sha256,
            "contract_sha256":admission["contract_sha256"],"native_resolution":int(geom["native_resolution"]),
            "valid_queries":len(q),"warmup_count":args.warmup,"repeat_count":args.repeats,
            "synchronization":"torch.cuda.synchronize before and after each timed query" if device.type=="cuda" else "synchronous CPU",
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "checkpoint_bytes":Path(args.checkpoint).stat().st_size,"encoding_seconds":encoding,"acceleration_setup_seconds":setup,
            "cache":cache_report,"parity":errors,"uncached_moment":base_m,"uncached_moment_jacobian":base_j,
            "cached_moment":fast_m,"cached_moment_jacobian":fast_j,
            "scope":"single_query_inference_not_native_solver_end_to_end"}
    if intrinsic:
        result["intrinsic_token_moment"]=result.pop("uncached_moment")
        result["intrinsic_token_moment_jacobian"]=result.pop("uncached_moment_jacobian")
        result["acceleration_setup_seconds"]=None
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
