from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
import torch
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import NativePCRAFTConfig,PcraftNativeMomentOperator
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_m80_sqno import UnifiedM80SQNO

def make_geometry():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns
    columns=np.array([[1,1,1,5],[1,2,2,7],[2,1,0,6],[3,3,1,8],[4,3,0,7]],dtype=np.int32)
    return geometry_from_columns({"columns":columns,"origin":np.array([-1.,-2.,.5]),
        "h":np.array(.25),"side":np.array(512,dtype=np.int32),"density":np.array(1.7)})

def make_unified(*,cross_attention_enabled=False,cross_attention_heads=2,cross_attention_chunk=1024,
                 cross_scale_attention_enabled=False,cross_scale_heads=2,cross_scale_disabled_ids=(),
                 gno_enabled=False,interaction_variant="independent"):
    torch.manual_seed(830411)
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1,
        convex_potential_enabled=True,convex_potential_components=2,gno_enabled=gno_enabled,gno_neighbors=4)
    shared=PcraftNativeMomentOperator(cfg)
    sqno=SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=16,
        num_layers=2,neighbors=4,temperature=.04,quadrature_order=2,
        chunk_intervals=2,chunk_nodes=64,query_chunk=2)
    return UnifiedM80SQNO(shared,sqno,cross_attention_enabled=cross_attention_enabled,
        cross_attention_heads=cross_attention_heads,cross_attention_chunk=cross_attention_chunk,
        cross_scale_attention_enabled=cross_scale_attention_enabled,cross_scale_heads=cross_scale_heads,
        cross_scale_disabled_ids=cross_scale_disabled_ids,interaction_variant=interaction_variant)

def queries():
    torch.manual_seed(830412)
    q=torch.randn(4,10,dtype=torch.float64)
    return q/q.norm(dim=-1,keepdim=True)

def test_external_pcraft_mode_has_no_duplicate_adapter_and_keeps_shared_encoder_trainable():
    geometry=make_geometry()
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1)
    encoder=PcraftNativeMomentOperator(cfg).train()
    cache=encoder.encode(geometry,retain_interval_features=True)
    payload=encoder.export_shared_geometry_features(cache)
    sqno=SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=16,
        num_layers=2,neighbors=4,temperature=.04,quadrature_order=2,
        chunk_intervals=2,chunk_nodes=64,query_chunk=2).train()
    assert sqno.pcraft is None
    assert not any(name.startswith("pcraft.") for name,_ in sqno.named_parameters())
    sqcache=sqno.encode_geometry(geometry,pcraft_features=payload)
    out=sqno.evaluate_queries(sqcache,queries()[:2],with_jacobian=True)
    (out["moments"].square().mean()+out["jacobian_q"].square().mean()).backward()
    grads=[p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert encoder.counters["encode_calls"]==1

def test_external_pcraft_requires_features_and_matching_geometry_identity():
    geometry=make_geometry()
    sqno=SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=16,
        num_layers=2,neighbors=4,temperature=.04,quadrature_order=2,
        chunk_intervals=2,chunk_nodes=64,query_chunk=2)
    with pytest.raises(ValueError,match="requires externally encoded"):
        sqno.encode_geometry(geometry)
    bad={"local_features":torch.zeros(len(geometry["xyz"]),8),"global_context":torch.zeros(8),
         "global_scale_weights":torch.ones(1),"interval_mass":torch.as_tensor(geometry["mass"]),
         "geometry_id":"wrong"}
    with pytest.raises(ValueError,match="identity"):
        sqno.encode_geometry(geometry,pcraft_features=bad)

def test_external_pcraft_width_must_be_positive():
    with pytest.raises(ValueError,match="positive"):
        SpinQuadrANNV12(pcraft_width=0,hidden_dim=4,final_hidden_dim=16,num_layers=2,neighbors=4)

def test_shared_feature_export_uses_physical_not_normalized_interval_mass():
    model=make_unified()
    geometry=make_geometry()
    cache=model.shared_encoder.encode(geometry,retain_interval_features=True)
    payload=model.shared_encoder.export_shared_geometry_features(cache)
    assert torch.allclose(payload["interval_mass"].sum(),
        torch.as_tensor(geometry["mass"].sum(),dtype=payload["interval_mass"].dtype),
        rtol=1e-7,atol=1e-9)

def test_unified_query_many_outputs_and_shared_encoder_called_once():
    model=make_unified().eval()
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry())
        out=model.evaluate_queries(cache,queries())
    assert model.shared_encoder.counters["encode_calls"]==1
    assert model.geometry_encode_calls==1
    assert out["moments"].shape==(4,10)
    assert out["jacobian_q"].shape==(4,10,10)
    assert torch.isfinite(out["moments"]).all() and torch.isfinite(out["jacobian_q"]).all()
    assert torch.allclose(out["moments"],out["fusion_weight"]*out["m80_moments"]+
        (1-out["fusion_weight"])*out["sqno_moments"],rtol=1e-9,atol=1e-10)
    assert torch.allclose(out["jacobian_q"],out["fusion_weight"]*out["m80_jacobian_q"]+
        (1-out["fusion_weight"])*out["sqno_jacobian_q"],rtol=1e-9,atol=1e-10)

def test_unified_joint_backward_reaches_both_branches_and_fusion():
    model=make_unified().train()
    cache=model.encode_geometry(make_geometry())
    out=model.evaluate_queries(cache,queries(),create_graph=True)
    (out["moments"].square().mean()+.1*out["jacobian_q"].square().mean()).backward()
    for params in (model.shared_encoder.parameters(),model.sqno.parameters()):
        grads=[p.grad for p in params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.fusion_logit.grad is not None and torch.isfinite(model.fusion_logit.grad)

def test_unified_optimizer_has_no_duplicate_parameters():
    model=make_unified()
    opt=model.build_optimizer(learning_rate=1e-4)
    actual=[id(p) for group in opt.param_groups for p in group["params"]]
    assert len(actual)==len(set(actual))
    assert set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
    assert not any(name.startswith("sqno.pcraft.") for name,_ in model.named_parameters())

def test_unified_checkpoint_roundtrip_and_data_contract(tmp_path):
    model=make_unified().eval()
    optimizer=model.build_optimizer()
    q=queries(); geometry=make_geometry()
    with torch.no_grad(): before=model.evaluate_queries(model.encode_geometry(geometry),q)
    path=tmp_path/"unified.pt"
    model.save_checkpoint(path,optimizer=optimizer,data_fingerprint="synthetic-fixture-v1")
    restored=make_unified().eval()
    restored_opt=restored.build_optimizer()
    payload=restored.load_checkpoint(path,optimizer=restored_opt,expected_data_fingerprint="synthetic-fixture-v1")
    with torch.no_grad(): after=restored.evaluate_queries(restored.encode_geometry(geometry),q)
    assert payload["schema"]=="tvcg.unified-m80-sqno.v1"
    assert torch.allclose(before["moments"],after["moments"],rtol=1e-8,atol=1e-9)
    assert torch.allclose(before["jacobian_q"],after["jacobian_q"],rtol=1e-8,atol=1e-9)
    with pytest.raises(ValueError,match="data fingerprint"):
        restored.load_checkpoint(path,expected_data_fingerprint="wrong-data")

def test_unified_cache_rejects_stale_parameter_versions():
    model=make_unified()
    cache=model.encode_geometry(make_geometry())
    with torch.no_grad(): next(model.parameters()).add_(1e-4)
    with pytest.raises(ValueError,match="stale"):
        model.evaluate_queries(cache,queries())


def test_inference_cache_is_graph_free_reusable_and_rejected_for_training():
    model=make_unified().eval()
    geometry=make_geometry()
    with torch.no_grad():
        cache=model.encode_geometry(geometry,inference=True)
        assert cache.inference_only
        assert not cache.shared_cache.latent.requires_grad
        first=model.evaluate_queries(cache,queries())
        assert not first["moments"].requires_grad
        assert not first["jacobian_q"].requires_grad
        second=model.evaluate_queries(cache,queries().flip(0))
    assert torch.allclose(first["moments"],second["moments"].flip(0),rtol=1e-8,atol=1e-9)
    model.train()
    with pytest.raises(ValueError,match="inference-only"):
        model.evaluate_queries(cache,queries())


def test_joint_training_loss_reports_fused_and_branch_terms_with_masks():
    model=make_unified().train()
    cache=model.encode_geometry(make_geometry())
    q=queries()
    target_m=torch.zeros(4,10,dtype=torch.float64)
    target_j=torch.zeros(4,10,10,dtype=torch.float64)
    mask=torch.tensor([True,True,False,True])
    result=model.joint_training_loss(cache,q,target_m,target_j,
        moment_scale=torch.ones(10),jacobian_scale=torch.ones(10,10),
        valid_mask=mask,jacobian_valid_mask=mask,
        jacobian_weight=.2,branch_aux_weight=.3)
    required={"loss","moment_loss","jacobian_loss","m80_moment_loss",
        "sqno_moment_loss","m80_jacobian_loss","sqno_jacobian_loss","branch_aux_loss"}
    assert required.issubset(result)
    assert all(torch.isfinite(value).all() for value in result.values())
    result["loss"].backward()
    assert model.fusion_logit.grad is not None


def test_joint_training_loss_rejects_empty_or_invalid_masks_and_scales():
    model=make_unified().train()
    cache=model.encode_geometry(make_geometry())
    q=queries()
    with pytest.raises(ValueError,match="no valid queries"):
        model.joint_training_loss(cache,q,torch.zeros(4,10),torch.zeros(4,10,10),
            valid_mask=torch.zeros(4,dtype=torch.bool))
    with pytest.raises(ValueError,match="positive"):
        model.joint_training_loss(cache,q,torch.zeros(4,10),torch.zeros(4,10,10),
            moment_scale=0.0)


def test_train_step_updates_parameters_and_invalidates_preupdate_cache():
    model=make_unified().train()
    optimizer=model.build_optimizer(learning_rate=1e-5)
    cache=model.encode_geometry(make_geometry())
    before=model.fusion_logit.detach().clone()
    metrics=model.train_step(optimizer,cache,queries(),
        torch.zeros(4,10,dtype=torch.float64),torch.zeros(4,10,10,dtype=torch.float64),
        gradient_clip=1.0)
    assert metrics["loss"]>=0 and metrics["gradient_norm"]>=0
    assert metrics["cache_invalidated"]==1.0
    assert not torch.equal(before,model.fusion_logit.detach())
    with pytest.raises(ValueError,match="stale"):
        model.evaluate_queries(cache,queries())


def test_unified_jacobian_uses_raw_coefficient_coordinates_and_matches_finite_difference():
    model=make_unified().eval()
    geometry=make_geometry()
    q=queries()[:1]
    with torch.no_grad():
        cache=model.encode_geometry(geometry,inference=True)
        base=model.evaluate_queries(cache,q)
        eps=2e-6
        for column in (0,4,8):
            delta=torch.zeros_like(q)
            delta[0,column]=eps
            plus=model.evaluate_queries(cache,q+delta)
            minus=model.evaluate_queries(cache,q-delta)
            fd=(plus["moments"]-minus["moments"])/(2*eps)
            predicted=base["jacobian_coefficients"][:,:,column]
            assert torch.allclose(predicted,fd,rtol=4e-3,atol=3e-5), (
                f"fused raw-coefficient Jacobian mismatch at column {column}: "
                f"max_abs={(predicted-fd).abs().max().item()}"
            )
            assert torch.allclose(
                base["jacobian_coefficients"][:,:,column],
                base["jacobian_q"][:,:,column],rtol=0,atol=0,
            )


def test_unified_moments_are_positive_scale_invariant_and_jacobian_scales_reciprocally():
    model=make_unified().eval()
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        q=queries()[:2]
        scale=3.7
        a=model.evaluate_queries(cache,q)
        b=model.evaluate_queries(cache,q*scale)
    assert torch.allclose(a["moments"],b["moments"],rtol=2e-7,atol=2e-9)
    assert torch.allclose(a["jacobian_coefficients"]/scale,
                          b["jacobian_coefficients"],rtol=2e-5,atol=2e-7)


def test_unified_single_query_shapes_and_mass_quadrature_contract():
    model=make_unified().eval()
    geometry=make_geometry()
    with torch.no_grad():
        cache=model.encode_geometry(geometry,inference=True)
        out=model.evaluate_queries(cache,queries()[0])
    assert out["moments"].shape==(10,)
    assert out["jacobian_coefficients"].shape==(10,10)
    assert out["m80_jacobian_coefficients"].shape==(10,10)
    assert out["sqno_jacobian_coefficients"].shape==(10,10)
    assert torch.isfinite(out["mass_error"])
    assert float(out["mass_error"])<1e-7
    assert out["geometry_id"]==cache.geometry_id


def test_unified_architecture_contract_rejects_different_fusion_initialization(tmp_path):
    model=make_unified().eval()
    optimizer=model.build_optimizer()
    path=tmp_path/"contract.pt"
    model.save_checkpoint(path,optimizer=optimizer,data_fingerprint="pilot-mini7")
    other=UnifiedM80SQNO(
        model.shared_encoder.__class__(model.shared_encoder.config),
        SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=16,
            num_layers=2,neighbors=4,temperature=.04,quadrature_order=2,
            chunk_intervals=2,chunk_nodes=64,query_chunk=2),
        fusion_initial_weight=.6,
    )
    with pytest.raises(ValueError,match="architecture contract"):
        other.load_checkpoint(path)


def test_geometry_physics_attention_is_query_order_and_interval_permutation_invariant():
    model=make_unified(cross_attention_enabled=True,cross_attention_chunk=2).eval()
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=5e-3)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].bias,std=1e-3)
    torch.manual_seed(830497)
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        first=model.evaluate_queries(cache,queries())
        qperm=torch.tensor([2,0,3,1])
        reordered=model.evaluate_queries(cache,queries()[qperm])
        n=len(cache.shared_cache.interval_features)
        perm=torch.randperm(n)
        permuted_shared_cache=replace(
            cache.shared_cache,
            interval_features=cache.shared_cache.interval_features[perm],
            interval_mass=cache.shared_cache.interval_mass[perm],
        )
        permuted_cache=replace(cache,shared_cache=permuted_shared_cache)
        interval_reordered=model.evaluate_queries(permuted_cache,queries())
    assert torch.allclose(first["moments"][qperm],reordered["moments"],rtol=1e-6,atol=1e-8)
    assert torch.allclose(first["moments"],interval_reordered["moments"],rtol=1e-6,atol=1e-8)
    assert torch.allclose(first["attention_entropy"],interval_reordered["attention_entropy"],rtol=1e-6,atol=1e-7)
    assert torch.isfinite(first["attention_effective_support"]).all()
    assert (first["attention_effective_support"]>=1).all()


def test_geometry_physics_attention_gradients_and_chunked_execution_are_finite():
    model=make_unified(cross_attention_enabled=True,cross_attention_chunk=2).train()
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=5e-3)
    cache=model.encode_geometry(make_geometry())
    out=model.evaluate_queries(cache,queries(),create_graph=True)
    loss=out["moments"].square().mean()+.05*out["jacobian_coefficients"].square().mean()
    loss.backward()
    params=[p.grad for p in model.geometry_physics_attention.parameters() if p.grad is not None]
    assert params and all(torch.isfinite(g).all() for g in params)
    assert torch.isfinite(out["attention_entropy"]).all()
    assert torch.isfinite(out["attention_effective_support"]).all()


def test_geometry_physics_attention_raw_coefficient_jacobian_matches_finite_difference():
    model=make_unified(cross_attention_enabled=True,cross_attention_chunk=2).eval()
    torch.manual_seed(830499)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=2e-2)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].bias,std=1e-3)
    q=queries()[:1]
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        base=model.evaluate_queries(cache,q)
        eps=1e-4
        for column in (0,4,8):
            delta=torch.zeros_like(q); delta[0,column]=eps
            plus=model.evaluate_queries(cache,q+delta)
            minus=model.evaluate_queries(cache,q-delta)
            fd=(plus["moments"]-minus["moments"])/(2*eps)
            predicted=base["jacobian_coefficients"][:,:,column]
            assert torch.allclose(predicted,fd,rtol=1e-2,atol=2e-6), (
                f"attention Jacobian mismatch column={column}, "
                f"max_abs={(predicted-fd).abs().max().item()}"
            )


def test_geometry_physics_attention_preserves_positive_quadric_scale_invariance():
    model=make_unified(cross_attention_enabled=True,cross_attention_chunk=2).eval()
    torch.manual_seed(830498)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=5e-3)
    q=queries()[:2]
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        a=model.evaluate_queries(cache,q)
        b=model.evaluate_queries(cache,7.3*q)
    assert torch.allclose(a["moments"],b["moments"],rtol=2e-6,atol=2e-8)
    assert torch.allclose(a["jacobian_coefficients"]/7.3,b["jacobian_coefficients"],rtol=3e-5,atol=3e-7)


def test_geometry_physics_attention_disabled_path_is_legacy_global_fusion():
    model=make_unified(cross_attention_enabled=False).eval()
    assert model.geometry_physics_attention is None
    assert model.architecture_contract()["geometry_physics_attention"]["enabled"] is False
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        out=model.evaluate_queries(cache,queries())
    assert "attention_entropy" not in out
    assert torch.isfinite(out["moments"]).all()


def test_legacy_global_fusion_checkpoint_migrates_without_attention(tmp_path):
    model=make_unified().eval()
    path=tmp_path/"legacy-global.pt"
    model.save_checkpoint(path,data_fingerprint="legacy-fixture")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    payload["architecture"].pop("geometry_physics_attention")
    payload["source_fingerprints"].pop("geometry_physics_attention.py",None)
    payload["source_fingerprints"]["unified_m80_sqno.py"]="pre-2.8.5-wrapper"
    legacy_path=tmp_path/"legacy-global-v0.pt"
    torch.save(payload,legacy_path)
    restored=make_unified().eval()
    loaded=restored.load_checkpoint(legacy_path,expected_data_fingerprint="legacy-fixture")
    assert loaded["architecture"].get("geometry_physics_attention") is None


def test_cross_scale_attention_query_and_scale_permutation_invariance():
    model=make_unified(cross_scale_attention_enabled=True).eval()
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight,std=1e-3)
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        a=model.evaluate_queries(cache,queries())
        qp=torch.tensor([2,0,3,1])
        b=model.evaluate_queries(cache,queries()[qp])
        perm=torch.arange(cache.shared_cache.scale_features.shape[0]-1,-1,-1)
        sc=replace(cache.shared_cache,scale_features=cache.shared_cache.scale_features[perm],
                   scale_ids=cache.shared_cache.scale_ids[perm])
        c=model.evaluate_queries(replace(cache,shared_cache=sc),queries())
    assert torch.allclose(a["moments"][qp],b["moments"],rtol=1e-6,atol=1e-8)
    assert torch.allclose(a["moments"],c["moments"],rtol=1e-6,atol=1e-8)
    assert torch.allclose(a["cross_scale_weights"],c["cross_scale_weights"][:,perm],rtol=1e-6,atol=1e-7)


def test_cross_scale_attention_missing_and_disabled_scales_are_masked():
    model=make_unified(cross_scale_attention_enabled=True).eval()
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        out=model.evaluate_queries(cache,queries())
        disabled=make_unified(cross_scale_attention_enabled=True,
            cross_scale_disabled_ids=tuple(int(v) for v in cache.shared_cache.scale_ids[:-1])).eval()
        disabled.load_state_dict(model.state_dict(),strict=False)
        dc=disabled.encode_geometry(make_geometry(),inference=True)
        dout=disabled.evaluate_queries(dc,queries())
    assert torch.isfinite(out["cross_scale_weights"]).all()
    assert torch.allclose(out["cross_scale_weights"].sum(-1),torch.ones(4),atol=1e-6)
    assert (dout["cross_scale_valid_count"]==1).all()


def test_cross_scale_attention_raw_jacobian_finite_difference_and_scale_invariance():
    model=make_unified(cross_scale_attention_enabled=True).eval()
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight,std=2e-3)
    q=queries()[:1]
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        a=model.evaluate_queries(cache,q)
        b=model.evaluate_queries(cache,5.2*q)
        delta=torch.zeros_like(q); delta[0,4]=1e-4
        plus=model.evaluate_queries(cache,q+delta)
        minus=model.evaluate_queries(cache,q-delta)
        fd=(plus["moments"]-minus["moments"])/2e-4
    assert torch.allclose(a["moments"],b["moments"],rtol=2e-6,atol=2e-8)
    assert torch.allclose(a["jacobian_coefficients"][:,:,4],fd,rtol=1e-2,atol=2e-6)


def test_cross_scale_attention_gradients_are_finite_and_disabled_path_is_legacy():
    model=make_unified(cross_scale_attention_enabled=True).train()
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight,std=1e-3)
    cache=model.encode_geometry(make_geometry())
    out=model.evaluate_queries(cache,queries(),create_graph=True)
    (out["moments"].square().mean()+.01*out["jacobian_coefficients"].square().mean()).backward()
    grads=[p.grad for p in model.cross_scale_attention.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    baseline=make_unified().eval()
    assert baseline.cross_scale_attention is None


def test_gno_attention_interaction_variants_validate_execution_contracts():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import NativePCRAFTConfig,PcraftNativeMomentOperator
    from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1,
        convex_potential_enabled=True,convex_potential_components=2,gno_enabled=True,gno_neighbors=4)
    def build(variant,geo=False,scale=False):
        shared=PcraftNativeMomentOperator(cfg)
        sq=SpinQuadrANNV12(pcraft_width=8,hidden_dim=4,final_hidden_dim=16,
            num_layers=2,neighbors=4,temperature=.04,quadrature_order=2,
            chunk_intervals=2,chunk_nodes=64,query_chunk=2)
        return UnifiedM80SQNO(shared,sq,cross_attention_enabled=geo,
            cross_scale_attention_enabled=scale,interaction_variant=variant)
    gno=build("gno_only")
    assert gno.geometry_physics_attention is None and gno.cross_scale_attention is None
    geo=build("geometry_attention_only",geo=True)
    assert geo.geometry_physics_attention is not None and geo.cross_scale_attention is None
    scale=build("cross_scale_only",scale=True)
    assert scale.geometry_physics_attention is None and scale.cross_scale_attention is not None
    seq=build("gno_then_attention",geo=True,scale=True)
    assert seq.shared_encoder.config.gno_enabled
    par=build("parallel_gno_attention",geo=True,scale=True)
    assert par.parallel_interaction_logit is not None
    with pytest.raises(ValueError,match="requires shared_encoder"):
        UnifiedM80SQNO(make_unified().shared_encoder,make_unified().sqno,interaction_variant="gno_only")
    with pytest.raises(ValueError,match="unsupported"):
        UnifiedM80SQNO(make_unified().shared_encoder,make_unified().sqno,interaction_variant="attention_then_gno")


def test_parallel_gno_attention_variant_has_finite_gate_gradients_and_jacobian():
    model=make_unified(cross_attention_enabled=True,cross_scale_attention_enabled=True,
        gno_enabled=True,interaction_variant="parallel_gno_attention").eval()
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=1e-3)
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight,std=1e-3)
    q=queries()[:1]
    with torch.no_grad():
        cache=model.encode_geometry(make_geometry(),inference=True)
        base=model.evaluate_queries(cache,q)
        eps=1e-4
        delta=torch.zeros_like(q); delta[0,4]=eps
        plus=model.evaluate_queries(cache,q+delta)
        minus=model.evaluate_queries(cache,q-delta)
        fd=(plus["moments"]-minus["moments"])/(2*eps)
    assert torch.allclose(base["jacobian_coefficients"][:,:,4],fd,rtol=2e-2,atol=5e-6)
    model.train()
    train_cache=model.encode_geometry(make_geometry())
    out=model.evaluate_queries(train_cache,q,create_graph=True)
    (out["moments"].square().mean()+out["jacobian_coefficients"].square().mean()*.01).backward()
    assert model.parallel_interaction_logit.grad is not None
    assert torch.isfinite(model.parallel_interaction_logit.grad)


def test_non_parallel_variants_register_no_unused_interaction_gate():
    model=make_unified()
    assert model.parallel_interaction_logit is None
    names=[name for name,_ in model.named_parameters()]
    assert not any("parallel_interaction_logit" in name for name in names)
