"""Reproducible, fail-closed Phase 2.8.3 three-head ablation runner."""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, subprocess, sys
from pathlib import Path
from .native_pcraft import make_config
from .native_train import M72Corpus, source_fingerprints

SCHEMA="phase2.8.3.head-ablation.v1"
HEADS=("xl_ms_gno_transolver_bounded_odd","xl_ms_gno_transolver_convex","xl_ms_gno_transolver_convex_atan")
SEEDS=(830301,830302,830303)
HEAD_FIELDS={"convex_potential_enabled","convex_potential_components","convex_amplitude_mode"}

def _hash(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def assert_matched_backbones():
    cfg={n:make_config(n) for n in HEADS}
    ref=vars(cfg[HEADS[0]])
    for name,obj in cfg.items():
        cur=vars(obj)
        if set(cur)!=set(ref): raise ValueError(f"{name}: config fields differ")
        diff={k for k in ref if ref[k]!=cur[k]}
        if diff-HEAD_FIELDS: raise ValueError(f"{name}: unmatched trunk fields {sorted(diff-HEAD_FIELDS)}")
    return {n:vars(c) for n,c in cfg.items()}

def build_plan(manifest,audit,output,seeds=SEEDS,min_queries=256,min_train_groups=50,
               min_val_groups=7,min_test_groups=10,max_steps=3000,max_seconds=86400,
               device="cuda:0",amp="bf16",batch_queries=64,learning_rate=3e-4,
               weight_decay=1e-4,jacobian_weight=.05,gradient_clip=1.0,
               chunk=2048,min_native_resolution=512,threads=1,cache_mib=512,
               val_every=20,save_every=25,val_max_records=8,val_max_queries=256):
    seeds=tuple(int(s) for s in seeds)
    if len(seeds)<3 or len(set(seeds))!=len(seeds) or min(seeds)<0: raise ValueError("need >=3 unique nonnegative seeds")
    if (min_queries<1 or max_steps<1 or max_seconds<=0 or batch_queries<1 or chunk<1
            or min_native_resolution<64 or threads<1 or cache_mib<0
            or val_every<1 or save_every<1 or val_max_records<1 or val_max_queries<1
            or gradient_clip<=0 or learning_rate<=0 or weight_decay<0 or jacobian_weight<0):
        raise ValueError("invalid data/training budget or runtime configuration")
    if device.startswith("cuda") and amp not in ("bf16","none"): raise ValueError("unsupported CUDA AMP")
    trunk=assert_matched_backbones()
    corpus=M72Corpus(manifest,audit,grid_size=64,min_queries=min_queries)
    groups={s:len(corpus._group(corpus.splits[s])) for s in ("train","val","test")}
    mins={"train":min_train_groups,"val":min_val_groups,"test":min_test_groups}
    bad=[f"{s}={groups[s]} < {mins[s]}" for s in groups if groups[s]<mins[s]]
    coverage=corpus.verify_query_coverage(min_queries)
    if bad: raise ValueError("source-group admission failed: "+", ".join(bad))
    geometry_resolutions=[]; interval_counts=[]
    for row in corpus.rows:
        geometry,_=corpus.get(row)
        resolution=int(geometry["native_resolution"])
        if resolution<min_native_resolution:
            raise ValueError(f"{row['id']}: native resolution {resolution} < required {min_native_resolution}")
        geometry_resolutions.append(resolution)
        interval_counts.append(int(len(geometry["xyz"])))
    geometry_audit={"records":len(geometry_resolutions),"min_native_resolution":min(geometry_resolutions),
        "max_native_resolution":max(geometry_resolutions),"min_intervals":min(interval_counts),
        "max_intervals":max(interval_counts),"geometry_sha256_verified":True}
    p={"schema":SCHEMA,"dataset":corpus.summary(),"split_source_groups":groups,
       "minimum_source_groups":mins,"query_coverage":{"records":len(coverage),"min":min(coverage.values()),
       "max":max(coverage.values()),"required":min_queries},"heads":list(HEADS),
       "matched_configs":trunk,"source_fingerprints":source_fingerprints(),"geometry_audit":geometry_audit,"seeds":list(seeds),
       "training":{"optimizer":"AdamW","max_steps":max_steps,"max_seconds":max_seconds,
       "device":device,"amp":amp,"batch_queries":batch_queries,"learning_rate":learning_rate,
       "weight_decay":weight_decay,"jacobian_weight":jacobian_weight,"gradient_clip":gradient_clip,
       "chunk":chunk,"min_native_resolution":min_native_resolution,"threads":threads,"cache_mib":cache_mib,
       "val_every":val_every,"save_every":save_every,"val_max_records":val_max_records,
       "val_max_queries":val_max_queries,"selection":"best validation checkpoint; test split not used"}}
    p["fingerprint"]=_hash(p)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    (out/"plan.json").write_text(json.dumps(p,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return p

def run_plan(plan_path,manifest,audit,output,min_queries=256,min_native_resolution=512,
             min_train_groups=50,min_val_groups=7,min_test_groups=10,threads=1,cache_mib=512):
    p=json.loads(Path(plan_path).read_text())
    if p.get("schema")!=SCHEMA or p.get("fingerprint")!=_hash({k:v for k,v in p.items() if k!="fingerprint"}):
        raise ValueError("plan fingerprint/schema mismatch")
    corpus=M72Corpus(manifest,audit,grid_size=64,min_queries=min_queries)
    if corpus.doc["fingerprint"]!=p["dataset"]["fingerprint"]: raise ValueError("corpus differs from admitted plan")
    expected_mins=p["minimum_source_groups"]
    if (min_queries!=p["query_coverage"]["required"] or min_train_groups!=expected_mins["train"]
            or min_val_groups!=expected_mins["val"] or min_test_groups!=expected_mins["test"]
            or min_native_resolution!=p["training"]["min_native_resolution"]):
        raise ValueError("run arguments differ from the admitted experiment plan")
    if source_fingerprints()!=p["source_fingerprints"]:
        raise ValueError("source code changed after plan admission; create a new plan")
    out=Path(output); out.mkdir(parents=True,exist_ok=True); runs=[]
    t=p["training"]
    for seed in p["seeds"]:
        for preset in p["heads"]:
            d=out/preset/f"seed-{seed}"; report=d/"report.json"
            if report.is_file():
                old=json.loads(report.read_text())
                if old.get("contract",{}).get("corpus_fingerprint")!=corpus.doc["fingerprint"] or old.get("contract",{}).get("seed")!=seed:
                    raise ValueError(f"stale report in {d}")
                runs.append({"preset":preset,"seed":seed,"report":str(report),"status":"reused"}); continue
            d.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,"-m","research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train","train",
             "--manifest",str(Path(manifest).resolve()),"--audit",str(Path(audit).resolve()),"--out",str(d),
             "--preset",preset,"--device",t["device"],"--amp",t["amp"],"--seed",str(seed),
             "--threads",str(t["threads"]),"--cache-mib",str(t["cache_mib"]),"--chunk",str(t["chunk"]),
             "--min-native-resolution",str(t["min_native_resolution"]),"--gradient-clip",str(t["gradient_clip"]),
             "--val-every",str(t["val_every"]),"--save-every",str(t["save_every"]),
             "--val-max-records",str(t["val_max_records"]),"--val-max-queries",str(t["val_max_queries"]),
             "--batch-queries",str(t["batch_queries"]),"--learning-rate",str(t["learning_rate"]),
             "--weight-decay",str(t["weight_decay"]),"--jacobian-weight",str(t["jacobian_weight"]),
             "--max-steps",str(t["max_steps"]),"--max-seconds",str(t["max_seconds"]),
             "--min-queries-per-record",str(min_queries),"--min-train-groups",str(min_train_groups),
             "--min-val-groups",str(min_val_groups),"--min-test-groups",str(min_test_groups)]
            env=os.environ.copy(); env.setdefault("OMP_NUM_THREADS",str(t["threads"])); env.setdefault("MKL_NUM_THREADS",str(t["threads"]))
            subprocess.run(cmd,check=True,env=env)
            if not report.is_file(): raise RuntimeError(f"missing training report {report}")
            runs.append({"preset":preset,"seed":seed,"report":str(report),"status":"completed"})
    index={"schema":SCHEMA,"plan_fingerprint":p["fingerprint"],"runs":runs}
    (out/"run-index.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
    return index

def _stats(values):
    v=[float(x) for x in values if x is not None]
    if not v:return None
    if not all(math.isfinite(x) for x in v): raise FloatingPointError("nonfinite metric")
    return {"n":len(v),"mean":statistics.mean(v),"sd":statistics.stdev(v) if len(v)>1 else 0.,
            "min":min(v),"max":max(v)}

def aggregate(plan_path,index_path,output):
    p=json.loads(Path(plan_path).read_text()); ix=json.loads(Path(index_path).read_text())
    if ix.get("plan_fingerprint")!=p.get("fingerprint"): raise ValueError("run index/plan mismatch")
    expected={(h,s) for h in p["heads"] for s in p["seeds"]}; seen=set(); data={h:[] for h in p["heads"]}
    common_contract_fingerprint=None
    for item in ix["runs"]:
        key=(item["preset"],int(item["seed"]))
        if key not in expected or key in seen: raise ValueError(f"unexpected/duplicate run {key}")
        seen.add(key); r=json.loads(Path(item["report"]).read_text())
        if r.get("schema")!="m80.pcraft-native-training.v1": raise ValueError(f"{key}: report schema mismatch")
        c=r.get("contract",{})
        if c.get("corpus_fingerprint")!=p["dataset"]["fingerprint"] or c.get("seed")!=key[1]:
            raise ValueError(f"{key}: corpus/seed mismatch")
        normalized=dict(c); normalized.pop("seed",None)
        cfg=dict(normalized.get("config",{}))
        for field in HEAD_FIELDS: cfg.pop(field,None)
        normalized["config"]=cfg
        contract_fp=_hash(normalized)
        if common_contract_fingerprint is None: common_contract_fingerprint=contract_fp
        elif contract_fp!=common_contract_fingerprint:
            raise ValueError("head/seed runs do not share an identical training/runtime/source contract")
        v=r.get("best_checkpoint_validation") or {}; amp=v.get("convex_amplitude") or {}
        data[key[0]].append({"seed":key[1],"steps_completed":r.get("steps_completed"),
          "stop_reason":r.get("stop_reason"),"moment_scaled_mse":v.get("moment_scaled_mse"),
          "jacobian_scaled_mse":v.get("jacobian_scaled_mse"),
          "moment_p95":(v.get("moment_query_rmse") or {}).get("p95"),
          "jacobian_p95":(v.get("jacobian_query_rmse") or {}).get("p95"),
          "gate_fraction_p95":(amp.get("fraction") or {}).get("p95"),
          "effective_amplitude_p95":(amp.get("effective") or {}).get("p95"),
          "gradient_norm_p95":(r.get("gradient_diagnostics") or {}).get("global_norm",{}).get("p95"),
          "elapsed_seconds":r.get("elapsed_seconds"),"parameter_count":r.get("parameter_count")})
    missing=expected-seen
    if missing: raise ValueError(f"missing head/seed runs: {sorted(missing)}")
    summaries={}
    metrics=("moment_scaled_mse","jacobian_scaled_mse","moment_p95","jacobian_p95",
             "gate_fraction_p95","effective_amplitude_p95","gradient_norm_p95","elapsed_seconds")
    for h,rows in data.items():
        summaries[h]={m:_stats([x[m] for x in rows]) for m in metrics}; summaries[h]["runs"]=rows
    result={"schema":SCHEMA+".aggregate","plan_fingerprint":p["fingerprint"],
      "dataset_fingerprint":p["dataset"]["fingerprint"],"source_fingerprints":p["source_fingerprints"],
      "training_contract_fingerprint":common_contract_fingerprint,"seeds":p["seeds"],"heads":summaries,
      "freeze_decision":{"status":"pending","test_split_used_for_selection":False,
       "reason":"requires expert review of matched validation tails, gradient stability, checkpoint parity and exact-solver replay"},
      "limitations":["validation metrics do not prove exact-solver feasibility",
       "wall-time-stopped runs are not full-step matched experiments","no automatic head selection"]}
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("plan")
    q.add_argument("--manifest",required=True); q.add_argument("--audit",required=True); q.add_argument("--output",required=True)
    q.add_argument("--seeds",default=",".join(map(str,SEEDS))); q.add_argument("--min-queries",type=int,default=256)
    q.add_argument("--min-train-groups",type=int,default=50); q.add_argument("--min-val-groups",type=int,default=7)
    q.add_argument("--min-test-groups",type=int,default=10); q.add_argument("--max-steps",type=int,default=3000)
    q.add_argument("--max-seconds",type=float,default=86400); q.add_argument("--device",default="cuda:0")
    q.add_argument("--amp",choices=("bf16","none"),default="bf16"); q.add_argument("--batch-queries",type=int,default=64)
    q.add_argument("--gradient-clip",type=float,default=1.0); q.add_argument("--chunk",type=int,default=2048)
    q.add_argument("--min-native-resolution",type=int,default=512); q.add_argument("--threads",type=int,default=1)
    q.add_argument("--cache-mib",type=float,default=512); q.add_argument("--val-every",type=int,default=20)
    q.add_argument("--save-every",type=int,default=25); q.add_argument("--val-max-records",type=int,default=8)
    q.add_argument("--val-max-queries",type=int,default=256)
    q=sub.add_parser("run")
    for n in ("plan","manifest","audit","output"): q.add_argument("--"+n,required=True)
    q.add_argument("--min-queries",type=int,default=256); q.add_argument("--min-native-resolution",type=int,default=512)
    q.add_argument("--min-train-groups",type=int,default=50); q.add_argument("--min-val-groups",type=int,default=7)
    q.add_argument("--min-test-groups",type=int,default=10)
    q=sub.add_parser("aggregate")
    for n in ("plan","run-index","output"): q.add_argument("--"+n,required=True)
    a=ap.parse_args()
    if a.cmd=="plan":
        build_plan(a.manifest,a.audit,a.output,[int(x) for x in a.seeds.split(",") if x],
          a.min_queries,a.min_train_groups,a.min_val_groups,a.min_test_groups,a.max_steps,a.max_seconds,
          a.device,a.amp,a.batch_queries,gradient_clip=a.gradient_clip,chunk=a.chunk,
          min_native_resolution=a.min_native_resolution,threads=a.threads,cache_mib=a.cache_mib,
          val_every=a.val_every,save_every=a.save_every,val_max_records=a.val_max_records,
          val_max_queries=a.val_max_queries)
    elif a.cmd=="run":
        run_plan(a.plan,a.manifest,a.audit,a.output,a.min_queries,a.min_native_resolution,
          a.min_train_groups,a.min_val_groups,a.min_test_groups)
    else: aggregate(a.plan,a.run_index,a.output)

if __name__=="__main__":main()
