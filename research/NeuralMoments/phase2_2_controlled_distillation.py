"""Controlled synthetic distillation for Phase 2.2; not a physical oracle."""
from __future__ import annotations
from dataclasses import replace
import json, math
import numpy as np
import torch
from research.NeuralMoments.m69_task_conditioned_m80_v1 import (
    build_parameter_matched_variants, encode_m69_task_features
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig, PcraftNativeMomentOperator
)

BASE_TASK={"schema":"m69.fixed-axis-task.v1","axis":[0.,0.,1.],"axis_origin":[0.,0.,0.],
"contact":[0.,0.,0.],"density":1.,"native_resolution":512,"shell_fraction":.03,
"protect_radius_fraction":.04,"protect_height_fraction":0.,"polar_retention":.9,
"min_removal_fraction":.1,"max_centroid_rise":1e-7,"com_tolerance":1e-4,
"alignment_tolerance":1e-4,"principal_gap_min":-1e-8,"require_connected":True,
"mass_tolerance":1e-10,"inertia_tolerance":1e-10,"moment_psd_tolerance":1e-9,
"solver_tolerance":1e-9,"solver_polar_margin":1e-5,"training_polar_margin":5e-4,
"objective":"minimum_retained_mass","inertia_reference":"original_fixed_axis",
"material_policy":"single_density_removal_only","purpose":"research"}

def synthetic_geometry(index):
    rng=np.random.default_rng(8200+index)
    xyz=rng.uniform(-.65,.65,(12,3)); mass=np.full(12,.012); x,y,z=xyz.T
    basis=np.stack([np.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y],-1)
    grid=np.zeros((4,8,8,8),np.float32); grid[0]=1.; grid[1]=1.
    grid[2,1+index:6+index,1:7,1:7]=1.
    grid[3]=np.linspace(.1,.9,8,dtype=np.float32)[None,None,:]
    return {"xyz":xyz,"mass":mass,"cell_extent":np.full((12,3),.08),
      "moment_basis":basis,"fixed_moments":np.zeros(10),"original_moments":mass@basis,
      "native_resolution":8,"origin":np.array([-1.,-1.,-1.]),"h":.25,
      "pcraft_grid":grid,"geometry_sha256":f"phase22-synthetic-{index}"}

def task_features():
    rows=[]
    for shell,protect,polar,removal in [
      (.020,.025,.86,.08),(.025,.030,.88,.09),(.030,.035,.90,.10),
      (.035,.040,.92,.11),(.040,.045,.94,.12),(.045,.050,.91,.13),
      (.028,.055,.87,.14),(.038,.022,.93,.07),(.022,.048,.89,.15),
      (.042,.032,.95,.06),(.026,.042,.85,.12),(.036,.028,.925,.085),
      (.024,.052,.915,.105),(.044,.038,.875,.135)]:
        t=dict(BASE_TASK);t.update(shell_fraction=shell,protect_radius_fraction=protect,
                                   polar_retention=polar,min_removal_fraction=removal)
        rows.append(encode_m69_task_features(t,length_scale=2.0))
    return torch.as_tensor(np.stack(rows),dtype=torch.float32)

def base_model(seed):
    torch.manual_seed(seed)
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
      interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=8,
      convex_potential_enabled=True,convex_potential_components=2)
    return PcraftNativeMomentOperator(cfg)

def teacher_cache(cache,features,geometry_id):
    # Fixed analytic nonlinear interaction of task features and geometry statistics.
    gen=torch.Generator(device="cpu");gen.manual_seed(99173+geometry_id)
    w_task=torch.randn((31,cache.latent.shape[-1]),generator=gen)*.17
    w_stats=torch.randn((36,cache.latent.shape[-1]),generator=gen)*.11
    t=torch.tanh(features.cpu()@w_task)
    s=torch.tanh(cache.statistics.detach().cpu().float().reshape(1,36)@w_stats)
    delta=.045*torch.tanh(torch.sin(t+s)+.35*t*s).to(cache.latent.device,cache.latent.dtype)
    return replace(cache,latent=cache.latent+delta)

def queries(seed,count=8):
    rng=np.random.default_rng(seed);q=rng.normal(size=(count,10))
    q[:,0]+=.25;q[:,1]-=.8;q/=np.linalg.norm(q,axis=1,keepdims=True)
    return torch.as_tensor(q,dtype=torch.float64)

def run_controlled_distillation(seeds=(22021,22022,22023),steps=60,learning_rate=2e-3):
    if steps<1 or learning_rate<=0 or not math.isfinite(learning_rate):
        raise ValueError("steps and learning_rate must be positive")
    results=[]; last_budget=None
    for seed in seeds:
        base=base_model(int(seed))
        geoms=[synthetic_geometry(0),synthetic_geometry(1)]
        caches=[base.encode(g) for g in geoms]
        feats=task_features(); q=queries(int(seed)+7)
        variants,budget=build_parameter_matched_variants(base,tolerance=.20,freeze_backbone=True)
        last_budget=budget
        train_pairs=[(ti,gi) for ti in range(10) for gi in range(2)]
        valid_pairs=[(ti,gi) for ti in range(10,14) for gi in range(2)]
        train_targets=[]
        for ti,gi in train_pairs:
            with torch.no_grad(): train_targets.append(base.predict(teacher_cache(caches[gi],feats[ti:ti+1],gi),q))
        valid_targets=[]
        for ti,gi in valid_pairs:
            with torch.no_grad(): valid_targets.append(base.predict(teacher_cache(caches[gi],feats[ti:ti+1],gi),q))
        train_targets=torch.stack(train_targets);valid_targets=torch.stack(valid_targets)
        for name,model in variants.items():
            student_caches=[model.encode_geometry(g) for g in geoms]
            torch.manual_seed(int(seed)+{"film":1,"conditional_norm":2,"cross_attention":3}[name])
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
            best=float("inf")
            for step in range(steps):
                idx=step%len(train_pairs);ti,gi=train_pairs[idx]
                opt.zero_grad(set_to_none=True)
                pred=model.predict(student_caches[gi],q,feats[ti:ti+1])
                loss=(pred-train_targets[idx]).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                opt.step()
                if (step+1)%10==0 or step==steps-1:
                    model.eval()
                    with torch.no_grad():
                        vp=torch.stack([model.predict(student_caches[gi],q,feats[ti:ti+1]) for ti,gi in valid_pairs])
                        score=float(torch.sqrt((vp-valid_targets).square().mean()))
                    best=min(best,score);model.train()
            model.eval()
            with torch.no_grad():
                vp=torch.stack([model.predict(student_caches[gi],q,feats[ti:ti+1]) for ti,gi in valid_pairs])
                final=float(torch.sqrt((vp-valid_targets).square().mean()))
            jac=[]
            for ti,gi in valid_pairs:
                sc=model.condition_cache(student_caches[gi],feats[ti:ti+1])
                tc=teacher_cache(caches[gi],feats[ti:ti+1],gi)
                _,sj=model.base_model.moments_and_jacobian(sc,q)
                _,tj=base.moments_and_jacobian(tc,q)
                jac.append((sj-tj).square().mean())
            jrmse=float(torch.sqrt(torch.stack(jac).mean()))
            results.append({"seed":int(seed),"architecture":name,"best_validation_moment_rmse":best,
              "final_validation_moment_rmse":final,"validation_jacobian_rmse":jrmse,
              "trainable_parameters":model.trainable_parameter_count(),"steps":int(steps)})
    summary={}
    for name in ("film","conditional_norm","cross_attention"):
        rows=[x for x in results if x["architecture"]==name];summary[name]={}
        for key in ("best_validation_moment_rmse","final_validation_moment_rmse","validation_jacobian_rmse"):
            vals=np.asarray([x[key] for x in rows]);summary[name][key+"_mean"]=float(vals.mean())
            summary[name][key+"_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        summary[name]["trainable_parameters"]=int(round(np.mean([x["trainable_parameters"] for x in rows])))
    return {"protocol":{"kind":"synthetic_teacher_distillation_not_physical_oracle",
      "seeds":list(map(int,seeds)),"steps_per_variant":int(steps),"learning_rate":learning_rate,
      "optimizer":"Adam","train_task_ids":list(range(10)),"heldout_task_ids":list(range(10,14)),
      "synthetic_geometry_count":2,"query_count":8,"frozen_backbone":True,
      "teacher":"fixed deterministic nonlinear task-geometry latent interaction"},
      "parameter_budget":last_budget,"results":results,"summary":summary}

if __name__=="__main__":
    print(json.dumps(run_controlled_distillation(),indent=2,sort_keys=True))
