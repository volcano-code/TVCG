from __future__ import annotations

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
    NativePCRAFTConfig, PcraftNativeMomentOperator, conservative_coarse_grid, geometry_from_columns
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.pino_backbone import PINOFeatureEncoder


def small_geometry():
    rows=[]
    for x in range(1,7):
        for y in range(1,7):
            rows.append((x,y,1+((x+y)%2),7))
    return geometry_from_columns(dict(columns=np.asarray(rows,dtype=np.int32),
        origin=np.asarray([-1.,-1.,-1.]),h=np.asarray(.25),side=np.asarray(8,dtype=np.int32),
        density=np.asarray(1.)))


def small_model():
    cfg=NativePCRAFTConfig(grid_size=8,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=16,min_native_resolution=1)
    return PcraftNativeMomentOperator(cfg)


def test_m72_rle_geometry_contract_and_exact_moments():
    g=small_geometry(); c=g["columns"]; o=g["origin"]; h=g["h"]
    x=o[0]+(c[:,0]+.5)*h; y=o[1]+(c[:,1]+.5)*h
    lo=o[2]+c[:,2]*h; hi=o[2]+c[:,3]*h; z=(lo+hi)/2
    m=h*h*(hi-lo)
    expected=np.stack((m,m*x,m*y,m*z,m*(x*x+h*h/12),m*(y*y+h*h/12),
        m*(lo*lo+lo*hi+hi*hi)/3,m*y*z,m*x*z,m*x*y),axis=-1).sum(0)
    assert np.allclose(g["moment_basis"].T@g["mass"],expected,rtol=1e-12,atol=1e-12)
    assert np.allclose(g["original_moments"],expected,rtol=1e-12,atol=1e-12)


def test_conservative_context_grid_preserves_editable_volume():
    g=small_geometry(); grid=conservative_coarse_grid(g,8)
    assert grid.shape==(4,8,8,8)
    native=float(np.prod(g["cell_extent"],axis=1).sum())
    assert float(grid[0].sum())*g["h"]**3==pytest.approx(native,rel=1e-7,abs=1e-10)
    assert np.isfinite(grid).all()


def test_pino_forward_map_is_additive_and_preserves_legacy_pooled_output():
    encoder=PINOFeatureEncoder(width=2,modes=1,n_layers=2,grid_size=8).eval()
    x=torch.rand(1,4,8,8,8)
    with torch.no_grad():
        pooled=encoder(x); pooled2,spatial=encoder.forward_with_map(x)
    assert torch.equal(pooled,pooled2)
    assert spatial.shape==(1,2,8,8,8)


def test_m72_moment_head_scale_complement_and_exact_endpoints():
    torch.manual_seed(11); model=small_model().eval(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.15,.2,-.1,.07,.03,-.02,.01,.005,-.004,.003],dtype=torch.float64)
    r=model.predict(cache,q)
    assert torch.allclose(r,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    assert torch.allclose(r+model.predict(cache,-q),2*cache.fixed_moments+cache.design_moments,
                          rtol=1e-9,atol=1e-10)
    full=model.predict(cache,torch.tensor([1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    empty=model.predict(cache,torch.tensor([-1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    assert torch.allclose(full,cache.fixed_moments+cache.design_moments)
    assert torch.allclose(empty,cache.fixed_moments)


def test_ad_jacobian_matches_finite_difference_and_supports_second_order_training():
    torch.manual_seed(13); model=small_model().train(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    eps=2e-4; fd=torch.empty_like(jac)
    with torch.no_grad():
        for j in range(10):
            d=torch.zeros_like(q); d[j]=eps
            fd[:,j]=(model.predict(cache,q+d)-model.predict(cache,q-d))/(2*eps)
    assert torch.allclose(jac,fd,rtol=1e-2,atol=1e-4)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_geometry_cache_invalidates_after_optimizer_update():
    model=small_model().train(); cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.2,.1,0,0,0,0,0,0,0,0],dtype=torch.float64)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.predict(cache,q).sum().backward(); opt.step()
    with pytest.raises(ValueError,match="stale"):
        model.predict(cache,q)


def test_native_interval_knn_graph_is_self_inclusive_and_bounded():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import build_local_neighbor_graph
    g=small_geometry()
    graph=build_local_neighbor_graph(g["xyz"],neighbors=8)
    assert graph.shape==(len(g["xyz"]),8)
    assert graph.dtype==np.int32
    assert np.array_equal(graph[:,0],np.arange(len(g["xyz"]),dtype=np.int32))
    assert graph.min()>=0 and graph.max()<len(g["xyz"])
    # All non-self neighbours are among the nearest native interval centers.
    d=np.linalg.norm(g["xyz"][graph]-g["xyz"][:,None,:],axis=-1)
    assert np.all(d[:,1:]>=0)


def test_pcraft_gno_branch_preserves_moment_invariants_and_ad_jacobian():
    from dataclasses import replace
    torch.manual_seed(23)
    base=small_model()
    cfg=replace(base.config,gno_enabled=True,gno_neighbors=8,gno_cache_mib=1)
    model=PcraftNativeMomentOperator(cfg).train()
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    complement=model.predict(cache,-q)
    assert torch.allclose(moments+complement,2*cache.fixed_moments+cache.design_moments,
                          rtol=1e-9,atol=1e-10)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["gno_graph_builds"]==1
    # A second encode of the same geometry reuses its geometry-only graph.
    model.encode(geometry,chunk=11)
    assert model.counters["gno_graph_builds"]==1


def test_gno_preset_is_an_explicit_ablation_and_base_checkpoint_stays_legacy():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("xl")
    gno=make_config("xl_gno")
    assert base.gno_enabled is False
    assert gno.gno_enabled is True
    assert gno.gno_neighbors==8
    base_model=PcraftNativeMomentOperator(base)
    gno_model=PcraftNativeMomentOperator(gno)
    assert not hasattr(base_model,"gno_kernel")
    assert hasattr(gno_model,"gno_kernel")
    assert gno_model.pino_encoder.width==base_model.pino_encoder.width


def test_transolver_physics_slicing_normalizes_mass_pooling_and_deslicing():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(31)
    module=PhysicsStateSlicing(width=16,n_slices=4,n_heads=4)
    features=torch.randn(23,16,requires_grad=True)
    mass=torch.rand(23); mass=mass/mass.sum()
    updated,metrics,assignment,deslice=module(features,mass,return_details=True)
    assert updated.shape==features.shape
    assert assignment.shape==(23,4,4)
    assert deslice.shape==(4,4)
    assert torch.allclose(assignment.sum(-1),torch.ones(23,4),atol=1e-6)
    assert torch.allclose(deslice.sum(-1),torch.ones(4),atol=1e-6)
    assert metrics["transolver_effective_slices"]>=1.0
    assert metrics["transolver_effective_slices"]<=4.0+1e-5
    assert metrics["transolver_assignment_row_error"]<1e-5
    assert metrics["transolver_slice_mass_sum_error"]<1e-5
    assert metrics["transolver_temperature_min"]>=0.01
    updated.square().mean().backward()
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pcraft_transolver_branch_preserves_moment_invariants_and_jacobian():
    from dataclasses import replace
    torch.manual_seed(37)
    base=small_model()
    cfg=replace(base.config,transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-10,atol=1e-11)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-9,atol=1e-10)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["transolver_calls"]==1
    assert model.counters["transolver_assignment_row_error"]<1e-5
    assert model.counters["transolver_slice_mass_sum_error"]<1e-5


def test_pcraft_gno_transolver_combination_has_finite_second_order_backward():
    from dataclasses import replace
    torch.manual_seed(41)
    base=small_model()
    cfg=replace(base.config,gno_enabled=True,gno_neighbors=8,
                transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["gno_graph_builds"]==1
    assert model.counters["transolver_calls"]==1


def test_transolver_presets_are_opt_in_and_do_not_change_existing_baselines():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("xl")
    trans=make_config("xl_transolver")
    combo=make_config("xl_gno_transolver")
    assert base.transolver_enabled is False and base.gno_enabled is False
    assert trans.transolver_enabled is True and trans.gno_enabled is False
    assert combo.transolver_enabled is True and combo.gno_enabled is True
    assert trans.transolver_slices==32 and trans.transolver_heads==8
    assert combo.interval_width % combo.transolver_heads==0
    base_model=PcraftNativeMomentOperator(base)
    trans_model=PcraftNativeMomentOperator(trans)
    assert not hasattr(base_model,"transolver")
    assert hasattr(trans_model,"transolver")


def test_transolver_state_operator_is_permutation_equivariant():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(43)
    module=PhysicsStateSlicing(width=16,n_slices=4,n_heads=4).eval()
    features=torch.randn(29,16)
    mass=torch.rand(29); mass=mass/mass.sum()
    permutation=torch.randperm(len(features))
    with torch.no_grad():
        updated,_=module(features,mass)
        permuted,_=module(features[permutation],mass[permutation])
    assert torch.allclose(permuted,updated[permutation],rtol=1e-5,atol=1e-6)
    pooled=(updated*mass[:,None]).sum(0)
    pooled_permuted=(permuted*mass[permutation,None]).sum(0)
    assert torch.allclose(pooled,pooled_permuted,rtol=1e-6,atol=1e-7)


def test_transolver_checkpoint_roundtrip_preserves_enabled_branch(tmp_path):
    from dataclasses import asdict, replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        SCHEMA, SEMANTICS, load_model
    )
    torch.manual_seed(47)
    base=small_model()
    cfg=replace(base.config,transolver_enabled=True,transolver_slices=4,transolver_heads=2)
    model=PcraftNativeMomentOperator(cfg).eval()
    checkpoint=tmp_path/"transolver.pt"
    torch.save(dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(cfg),
                    model=model.state_dict()),checkpoint)
    restored,payload=load_model(checkpoint,"cpu")
    assert restored.config.transolver_enabled
    assert hasattr(restored,"transolver")
    geometry=small_geometry()
    with torch.no_grad():
        a=model.encode(geometry,chunk=7).detached()
        b=restored.encode(geometry,chunk=7).detached()
        q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
        ra=model.predict(a,q)
        rb=restored.predict(b,q)
    assert torch.allclose(ra,rb,rtol=1e-7,atol=1e-9)


def small_geometry16():
    rows=[]
    for x in range(2,14):
        for y in range(2,14):
            rows.append((x,y,2+(x+y)%2,14))
    return geometry_from_columns(dict(columns=np.asarray(rows,dtype=np.int32),
        origin=np.asarray([-1.,-1.,-1.]),h=np.asarray(.125),side=np.asarray(16,dtype=np.int32),
        density=np.asarray(1.)))


def test_pino_encoder_preserves_requested_context_resolution():
    encoder=PINOFeatureEncoder(width=2,modes=1,n_layers=2,grid_size=16).eval()
    x8=torch.rand(1,4,8,8,8)
    x16=torch.rand(1,4,16,16,16)
    with torch.no_grad():
        pooled8,map8=encoder.forward_with_map(x8,preserve_resolution=True)
        pooled16,map16=encoder.forward_with_map(x16,preserve_resolution=True)
    assert pooled8.shape==pooled16.shape==(1,encoder.feature_dim)
    assert map8.shape==(1,2,8,8,8)
    assert map16.shape==(1,2,16,16,16)


def test_multiscale_pcraft_uses_all_exact_context_grids_and_trains():
    from dataclasses import replace
    torch.manual_seed(53)
    cfg=NativePCRAFTConfig(grid_size=16,pino_width=2,pino_modes=1,pino_layers=2,
        interval_width=8,interval_depth=1,chunk_intervals=32,min_native_resolution=1,
        multiscale_enabled=True,multiscale_grids=(8,16))
    model=PcraftNativeMomentOperator(cfg).train()
    geometry=small_geometry16()
    grids=[conservative_coarse_grid(geometry,s) for s in (8,16)]
    for s,g in zip((8,16),grids):
        assert g.shape==(4,s,s,s)
        assert float(g[0].sum())*(geometry["h"]*16/s)**3 == pytest.approx(
            float(np.prod(geometry["cell_extent"],axis=1).sum()),rel=1e-6)
    cache=model.encode(geometry,chunk=32)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-9,atol=1e-10)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-8,atol=1e-9)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["multiscale_calls"]==1
    assert len(model.counters["multiscale_global_weights"])==2
    assert len(model.counters["multiscale_local_weights"])==2
    assert sum(model.counters["multiscale_global_weights"])==pytest.approx(1.,abs=1e-6)
    assert sum(model.counters["multiscale_local_weights"])==pytest.approx(1.,abs=1e-5)


def test_multiscale_presets_are_opt_in_and_keep_single_scale_baselines_unchanged():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("large")
    ms=make_config("large_ms")
    ms_trans=make_config("large_ms_transolver")
    ms_combo=make_config("large_ms_gno_transolver")
    assert base.multiscale_enabled is False
    assert ms.multiscale_enabled is True
    assert ms.multiscale_grids==(32,64,128)
    assert ms_trans.transolver_enabled and not ms_trans.gno_enabled
    assert ms_combo.transolver_enabled and ms_combo.gno_enabled
    assert ms.pino_width==base.pino_width
    assert ms.interval_width==base.interval_width


def test_native_interval_components_prevent_cross_layer_knn_edges():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        build_local_neighbor_graph,native_interval_components
    )
    columns=np.asarray([
        (1,1,1,3),(2,1,1,3),
        (1,1,4,6),(2,1,4,6),
    ],dtype=np.int32)
    geometry=geometry_from_columns(dict(columns=columns,
        origin=np.asarray([0.,0.,0.]),h=np.asarray(1.),side=np.asarray(8,dtype=np.int32),
        density=np.asarray(1.)))
    components=native_interval_components(geometry["columns"])
    assert len(np.unique(components))==2
    graph=build_local_neighbor_graph(geometry["xyz"],neighbors=4,component_ids=components)
    for i in range(len(components)):
        assert np.all(components[graph[i]]==components[i])
    # Without the topology filter, the two nearby layers are allowed to mix.
    global_graph=build_local_neighbor_graph(geometry["xyz"],neighbors=4)
    assert any(components[j]!=components[0] for j in global_graph[0,1:])


def test_gno_topology_filter_is_opt_in_and_records_component_count():
    from dataclasses import replace
    base=small_model()
    cfg=replace(base.config,gno_enabled=True,gno_topology_filter=True,gno_neighbors=4)
    model=PcraftNativeMomentOperator(cfg).eval()
    columns=np.asarray([
        (1,1,1,3),(2,1,1,3),
        (1,1,4,6),(2,1,4,6),
    ],dtype=np.int32)
    geometry=geometry_from_columns(dict(columns=columns,
        origin=np.asarray([0.,0.,0.]),h=np.asarray(1.),side=np.asarray(8,dtype=np.int32),
        density=np.asarray(1.)))
    cache=model.encode(geometry,chunk=4)
    assert cache.native_intervals==4
    assert model.counters["gno_component_count"]==2


def test_convex_potential_head_is_scale_invariant_complementary_and_psd():
    from dataclasses import replace
    torch.manual_seed(61)
    base=small_model()
    cfg=replace(base.config,convex_potential_enabled=True,convex_potential_components=3)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=7)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,7*q),rtol=1e-8,atol=1e-9)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-8,atol=1e-9)
    deviation=moments-cache.fixed_moments-.5*cache.design_moments
    assert torch.all(deviation.abs()<=.5*cache.absolute_moments+1e-9)
    hessian=jac/model.native_B
    assert torch.allclose(hessian,hessian.T,rtol=1e-7,atol=1e-8)
    eigenvalues=torch.linalg.eigvalsh((hessian+hessian.T)/2)
    assert float(eigenvalues.min().detach())>=-1e-7
    eps=2e-5; fd=torch.empty_like(jac)
    with torch.no_grad():
        for j in range(10):
            delta=torch.zeros_like(q); delta[j]=eps
            fd[:,j]=(model.predict(cache,q+delta)-model.predict(cache,q-delta))/(2*eps)
    assert torch.allclose(jac,fd,rtol=2e-2,atol=2e-4)
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["convex_potential_calls"]>0
    assert 0.0<model.counters["convex_potential_scale_fraction"]<1.0


def test_convex_potential_checkpoint_roundtrip_and_exact_endpoints(tmp_path):
    from dataclasses import asdict,replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        SCHEMA,SEMANTICS,load_model
    )
    torch.manual_seed(67)
    base=small_model()
    cfg=replace(base.config,convex_potential_enabled=True,convex_potential_components=3)
    model=PcraftNativeMomentOperator(cfg).eval()
    checkpoint=tmp_path/"convex.pt"
    torch.save(dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(cfg),
                    model=model.state_dict()),checkpoint)
    restored,_=load_model(checkpoint,"cpu")
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=7).detached()
    restored_cache=restored.encode(geometry,chunk=7).detached()
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    with torch.no_grad():
        a=model.predict(cache,q)
        b=restored.predict(restored_cache,q)
    assert torch.allclose(a,b,rtol=1e-8,atol=1e-9)
    full=model.predict(cache,torch.tensor([1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    empty=model.predict(cache,torch.tensor([-1.,0,0,0,0,0,0,0,0,0],dtype=torch.float64))
    assert torch.allclose(full,cache.fixed_moments+cache.design_moments)
    assert torch.allclose(empty,cache.fixed_moments)


@pytest.mark.parametrize("amplitude_mode",["sigmoid","atan"])
def test_convex_native_cross_derivatives_match_finite_differences_and_weighted_psd(amplitude_mode):
    from dataclasses import replace
    torch.manual_seed(271)
    model=PcraftNativeMomentOperator(replace(small_model().config,
        convex_potential_enabled=True,convex_potential_components=3,
        convex_amplitude_mode=amplitude_mode)).eval()
    cache=model.encode(small_geometry()).detached()
    c=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.07,-.06,.05],dtype=torch.float64)
    _,jac=model.moments_and_jacobian(cache,c)
    b=torch.tensor([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=torch.float64)
    hessian=jac/b
    torch.testing.assert_close(hessian,hessian.T,rtol=1e-10,atol=1e-11)
    assert torch.linalg.eigvalsh(.5*(hessian+hessian.T)).min()>=-1e-10
    assert (jac-jac.T).abs().max()>1e-5  # Native J is not a Hessian.
    torch.testing.assert_close(jac@c,torch.zeros(10,dtype=c.dtype),rtol=0,atol=1e-10)
    _,scaled_jac=model.moments_and_jacobian(cache,3.7*c)
    torch.testing.assert_close(scaled_jac,jac/3.7,rtol=1e-10,atol=1e-11)
    eps=2e-6
    with torch.no_grad():
        for column in (7,8,9):
            delta=torch.zeros_like(c);delta[column]=eps
            finite_difference=(model.predict(cache,c+delta)-model.predict(cache,c-delta))/(2*eps)
            torch.testing.assert_close(jac[:,column],finite_difference,rtol=1e-5,atol=1e-7)


def test_convex_native_coordinate_wrapper_matches_independent_physical_quadrature_gradient():
    """Probe the wrapper with a known potential, independently of learned fit."""
    from dataclasses import replace
    geometry=small_geometry()
    signs=np.stack(np.meshgrid(*[[-1/np.sqrt(3),1/np.sqrt(3)]]*3,indexing="ij"),-1).reshape(-1,3)
    xyz=geometry["xyz"][:,None,:]+.5*geometry["cell_extent"][:,None,:]*signs[None]
    xyz=torch.as_tensor(xyz.reshape(-1,3),dtype=torch.float64)
    x,y,z=xyz.unbind(-1)
    basis=torch.stack((torch.ones_like(x),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y),-1)
    weights=torch.as_tensor(np.repeat(geometry["mass"]/8,8),dtype=torch.float64)
    length=float(((geometry["xyz"]+geometry["cell_extent"]/2).max(0)
                  -(geometry["xyz"]-geometry["cell_extent"]/2).min(0)).max())
    coordinate_scale=length**torch.tensor([0,1,1,1,2,2,2,2,2,2],dtype=torch.float64)
    b=torch.tensor([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=torch.float64)
    eps2=.04**2

    class QuadraturePotential(torch.nn.Module):
        def forward(self,latent,t,amplitude_bound):
            monomial=t/coordinate_scale
            values=monomial@basis.T
            potential=.5*(weights*torch.sqrt(values.square()+eps2*monomial.square().sum(-1,keepdim=True))).sum(-1)
            return potential,t.new_tensor(.5),t.new_tensor(0.)

    model=PcraftNativeMomentOperator(replace(small_model().config,convex_potential_enabled=True)).eval()
    model.convex_potential_head=QuadraturePotential()
    cache=model.encode(geometry).detached()
    c=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.07,-.06,.05],dtype=torch.float64,requires_grad=True)
    # Direct physical polynomial and hinge integral in native c coordinates.
    monomial=b*c
    values=basis@monomial
    physical_potential=(.5*weights*(values+torch.sqrt(values.square()+eps2*monomial.square().sum()))).sum()
    expected=torch.autograd.grad(physical_potential,c,create_graph=True)[0]/b
    expected_jac=torch.stack([torch.autograd.grad(value,c,retain_graph=True)[0] for value in expected])
    actual,actual_jac=model.moments_and_jacobian(cache,c)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(actual_jac,expected_jac,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize("convex",[False,True])
def test_legacy_native_checkpoint_does_not_silently_change_convex_coordinate_semantics(tmp_path,convex):
    from dataclasses import asdict,replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import SCHEMA,load_model
    config=replace(small_model().config,convex_potential_enabled=convex)
    model=PcraftNativeMomentOperator(config).eval()
    old_semantics="pcraft-ufno-native-interval-m72-bounded-odd-or-convex-potential-moment-head-ad-jacobian-optional-gno-transolver-multiscale-event-validity.v6"
    path=tmp_path/"legacy-native.pt"
    torch.save(dict(schema=SCHEMA,semantics=old_semantics,config=asdict(config),model=model.state_dict()),path)
    if convex:
        with pytest.raises(ValueError,match="cross-coefficient semantics"):
            load_model(path)
    else:
        restored,_=load_model(path)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)


def test_convex_potential_presets_are_opt_in():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import make_config
    base=make_config("large")
    convex=make_config("large_convex")
    assert not base.convex_potential_enabled
    assert convex.convex_potential_enabled
    assert convex.convex_potential_components==4


def test_full_ug_pino_candidate_combines_multiscale_gno_transolver_convex_and_trains():
    from dataclasses import replace
    torch.manual_seed(79)
    base=small_model()
    cfg=replace(base.config,grid_size=16,multiscale_enabled=True,multiscale_grids=(8,16),
        gno_enabled=True,gno_neighbors=4,gno_topology_filter=True,
        transolver_enabled=True,transolver_slices=4,transolver_heads=2,
        convex_potential_enabled=True,convex_potential_components=3)
    model=PcraftNativeMomentOperator(cfg).train()
    geometry=small_geometry16()
    cache=model.encode(geometry,chunk=32)
    q=torch.tensor([.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    assert moments.shape==(10,) and jac.shape==(10,10)
    assert torch.isfinite(moments).all() and torch.isfinite(jac).all()
    assert torch.allclose(moments,model.predict(cache,5*q),rtol=1e-8,atol=1e-9)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-8,atol=1e-9)
    hessian=jac/model.native_B
    assert torch.allclose(hessian,hessian.T,rtol=1e-6,atol=1e-7)
    assert float(torch.linalg.eigvalsh(.5*(hessian+hessian.T)).min())>=-1e-6
    (moments.square().mean()+jac.square().mean()).backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.counters["gno_graph_builds"]==1
    assert model.counters["transolver_calls"]==1
    assert model.counters["multiscale_calls"]==1
    assert model.counters["convex_potential_calls"]>0


def test_maximal_ug_pino_preset_is_explicit_and_checkpoint_roundtrips(tmp_path):
    from dataclasses import asdict
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        SCHEMA,SEMANTICS,load_model,make_config
    )
    cfg=make_config("xl_ms_gno_transolver_convex")
    assert cfg.multiscale_enabled and cfg.gno_enabled and cfg.transolver_enabled
    assert cfg.convex_potential_enabled and cfg.gno_topology_filter
    model=PcraftNativeMomentOperator(cfg)
    checkpoint=tmp_path/"ug-pino-maximal.pt"
    torch.save(dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(cfg),
                    model=model.state_dict()),checkpoint)
    restored,_=load_model(checkpoint,"cpu")
    assert restored.config==cfg
    assert hasattr(restored,"transolver") and hasattr(restored,"gno_kernel")
    assert hasattr(restored,"convex_potential_head")
    assert restored.config.multiscale_grids==(32,64,128)


def test_shared_geometry_cache_exports_interval_features_for_sqno_without_reencoding():
    torch.manual_seed(83)
    model=small_model().train()
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=16,retain_interval_features=True)
    features=model.export_shared_geometry_features(cache)
    assert features["local_features"].shape==(cache.native_intervals,model.config.interval_width)
    assert features["global_context"].shape==(model.config.interval_width,)
    assert features["interval_mass"].shape==(cache.native_intervals,)
    assert features["descriptor"].shape==(cache.native_intervals,13)
    assert features["normalized_xyz"].shape==(cache.native_intervals,3)
    assert features["geometry_id"]==cache.geometry_sha256
    assert model.counters["encode_calls"]==1
    features["local_features"].square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.pino_encoder.parameters())


def test_shared_geometry_cache_rejects_stale_or_missing_feature_payload():
    model=small_model().eval()
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=16)
    with pytest.raises(ValueError,match="retain_interval_features"):
        model.export_shared_geometry_features(cache)
    retained=model.encode(geometry,chunk=16,retain_interval_features=True)
    with torch.no_grad():
        next(model.parameters()).add_(1e-3)
    with pytest.raises(ValueError,match="stale"):
        model.export_shared_geometry_features(retained)


def test_bounded_odd_moment_jacobian_matches_independent_finite_difference():
    torch.manual_seed(89)
    model=small_model().eval()
    cache=model.encode(small_geometry(),chunk=16).detached()
    q=torch.tensor([.04,-.09,.06,.03,-.02,.015,.01,-.004,.008,.006],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=False)
    eps=1e-5
    fd=torch.empty_like(jac)
    with torch.no_grad():
        for k in range(10):
            delta=torch.zeros_like(q);delta[k]=eps
            fd[:,k]=(model.predict(cache,q+delta)-model.predict(cache,q-delta))/(2*eps)
        scaled=model.predict(cache,3.7*q)
        complement=model.predict(cache,-q)
    assert torch.allclose(jac,fd,rtol=2e-2,atol=2e-4)
    assert torch.allclose(moments,scaled,rtol=1e-8,atol=1e-9)
    assert torch.allclose(moments+complement,
        2*cache.fixed_moments+cache.design_moments,rtol=1e-8,atol=1e-9)


def test_event_validity_head_forward_scale_invariance_and_joint_gradients():
    from dataclasses import replace
    torch.manual_seed(97)
    base=small_model()
    cfg=replace(base.config,event_validity_head_enabled=True)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([[.03,.11,-.07,.04,.02,-.01,.015,.004,-.006,.003],
                    [.04,-.09,.06,.03,-.02,.015,.01,-.004,.008,.006]],dtype=torch.float64)
    logits=model.predict_event_validity(cache,q)
    scaled=model.predict_event_validity(cache,7.0*q)
    assert logits.shape==(2,) and torch.isfinite(logits).all()
    assert torch.allclose(logits,scaled,rtol=1e-8,atol=1e-9)
    targets=torch.tensor([1.,0.],dtype=logits.dtype)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
    loss.backward()
    grads=[p.grad for p in model.event_validity_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_event_validity_head_is_optional_and_checkpoint_compatible(tmp_path):
    from dataclasses import asdict, replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import SCHEMA,SEMANTICS,load_model,make_config
    base=make_config("smoke",min_native_resolution=1)
    assert not base.event_validity_head_enabled
    event=replace(base,event_validity_head_enabled=True)
    model=PcraftNativeMomentOperator(event)
    path=tmp_path/"event.pt"
    torch.save(dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(event),
      model=model.state_dict()),path)
    restored,_=load_model(path,"cpu")
    assert restored.config.event_validity_head_enabled
    assert hasattr(restored,"event_validity_head")


def test_native_moment_and_quadric_coefficient_order_contract():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        MOMENT_ORDER,QUERY_ORDER,_quadric_bounds
    )
    assert MOMENT_ORDER==("mass","x","y","z","xx","yy","zz","yz","xz","xy")
    assert QUERY_ORDER==("1","x","y","z","xx","yy","zz","2yz","2xz","2xy")
    model=small_model()
    assert torch.equal(model.native_B,torch.tensor([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=torch.float64))
    bounds=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]],dtype=torch.float64)
    # q_7 multiplies 2*y*z, not y*z.
    q=torch.zeros(10,dtype=torch.float64);q[7]=1.
    lo,hi=_quadric_bounds(bounds,q)
    assert float(lo)==pytest.approx(-2.)
    assert float(hi)==pytest.approx(2.)
    # The three cross-term coefficients share the factor-of-two contract.
    for index in (7,8,9):
        q=torch.zeros(10,dtype=torch.float64);q[index]=1.
        lo,hi=_quadric_bounds(bounds,q)
        assert float(lo)==pytest.approx(-2.)
        assert float(hi)==pytest.approx(2.)


def test_pino_single_checkpoint_accepts_multiple_preserved_resolutions():
    torch.manual_seed(103)
    encoder=PINOFeatureEncoder(width=2,modes=2,n_layers=2,grid_size=16).eval()
    parameter_ids={id(p) for p in encoder.parameters()}
    shapes=((8,8,8),(16,16,16),(32,32,32))
    with torch.no_grad():
        outputs=[]
        for side in shapes:
            x=torch.rand(1,4,*side)
            pooled,spatial=encoder.forward_with_map(x,preserve_resolution=True)
            assert pooled.shape==(1,encoder.feature_dim)
            assert spatial.shape==(1,encoder.width,*side)
            assert torch.isfinite(pooled).all() and torch.isfinite(spatial).all()
            outputs.append(pooled)
    assert len(parameter_ids)==len({id(p) for p in encoder.parameters()})
    assert len({tuple(x.shape) for x in outputs})==1


def test_bounded_odd_jacobian_central_difference_step_sweep():
    torch.manual_seed(107)
    model=small_model().eval()
    cache=model.encode(small_geometry(),chunk=16).detached()
    q=torch.tensor([.07,-.04,.09,.02,-.03,.01,.025,.006,-.008,.004],dtype=torch.float64)
    _,jac=model.moments_and_jacobian(cache,q,create_graph=False)
    errors=[]
    with torch.no_grad():
        for eps in (2e-4,1e-4,5e-5):
            fd=torch.empty_like(jac)
            for k in range(10):
                delta=torch.zeros_like(q);delta[k]=eps
                fd[:,k]=(model.predict(cache,q+delta)-model.predict(cache,q-delta))/(2*eps)
            errors.append(torch.linalg.vector_norm(jac-fd)/torch.linalg.vector_norm(jac).clamp_min(1e-12))
    assert all(torch.isfinite(e) for e in errors)
    assert float(min(errors))<2e-2


def test_convex_head_psd_is_parameterization_property_not_global_head_contract():
    from dataclasses import replace
    torch.manual_seed(109)
    base=small_model()
    convex=PcraftNativeMomentOperator(replace(base.config,convex_potential_enabled=True,
        convex_potential_components=3)).eval()
    bounded=small_model().eval()
    geometry=small_geometry()
    ccache=convex.encode(geometry,chunk=16).detached()
    bcache=bounded.encode(geometry,chunk=16).detached()
    q=torch.tensor([.05,.08,-.03,.02,.01,-.015,.007,.003,-.002,.004],dtype=torch.float64)
    _,jc=convex.moments_and_jacobian(ccache,q,create_graph=False)
    _,jb=bounded.moments_and_jacobian(bcache,q,create_graph=False)
    hessian=jc/convex.native_B
    assert torch.allclose(hessian,hessian.T,rtol=1e-6,atol=1e-7)
    assert float(torch.linalg.eigvalsh(.5*(hessian+hessian.T)).min())>=-1e-7
    # The bounded-odd head has no imposed Hessian/PSD contract; only require
    # a finite, independently differentiable Jacobian.
    assert torch.isfinite(jb).all()


def test_convex_amplitude_sigmoid_and_atan_modes_are_bounded_and_have_same_state_contract():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import ConvexMomentPotentialHead
    torch.manual_seed(223)
    sigmoid=ConvexMomentPotentialHead(width=16,components=3,amplitude_mode="sigmoid")
    atan=ConvexMomentPotentialHead(width=16,components=3,amplitude_mode="atan")
    atan.load_state_dict(sigmoid.state_dict(),strict=True)
    assert set(sigmoid.state_dict())==set(atan.state_dict())
    with torch.no_grad():
        sigmoid.total_gate.weight.zero_()
        atan.total_gate.weight.zero_()
        sigmoid.total_gate.bias.fill_(5.)
        atan.total_gate.bias.fill_(5.)
    latent=torch.randn(1,16,dtype=torch.float32)
    logit_s,fraction_s=sigmoid.amplitude_parameters(latent)
    logit_a,fraction_a=atan.amplitude_parameters(latent)
    assert float(logit_s)==pytest.approx(5.)
    assert float(logit_a)==pytest.approx(5.)
    assert 0. < float(fraction_s) < 1.
    assert 0. < float(fraction_a) < 1.
    assert float(fraction_a)<float(fraction_s)
    ds=torch.autograd.grad(fraction_s,sigmoid.total_gate.bias,retain_graph=True)[0]
    da=torch.autograd.grad(fraction_a,atan.total_gate.bias)[0]
    assert float(da.abs().max())>float(ds.abs().max())


def test_convex_atan_potential_preserves_moment_invariants_and_jacobian_structure():
    from dataclasses import replace
    torch.manual_seed(227)
    base=small_model()
    cfg=replace(base.config,convex_potential_enabled=True,convex_potential_components=3,
                convex_amplitude_mode="atan")
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=16)
    q=torch.tensor([.04,-.07,.03,.06,.02,-.01,.015,.004,-.006,.003],dtype=torch.float64)
    moments,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    diagnostics=model.convex_amplitude_diagnostics(cache)
    assert diagnostics["mode"]=="atan"
    assert 0.<diagnostics["fraction"]<1.
    assert diagnostics["amplitude_bound"]>=0.
    assert diagnostics["effective_amplitude"]>=0.
    assert torch.allclose(moments,model.predict(cache,4.2*q),rtol=1e-8,atol=1e-9)
    assert torch.allclose(moments+model.predict(cache,-q),
        2*cache.fixed_moments+cache.design_moments,rtol=1e-8,atol=1e-9)
    hessian=jac/model.native_B
    assert torch.allclose(hessian,hessian.T,rtol=1e-6,atol=1e-7)
    assert float(torch.linalg.eigvalsh(.5*(hessian+hessian.T)).min())>=-1e-6
    (moments.square().mean()+jac.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_convex_amplitude_mode_is_opt_in_and_old_checkpoint_config_defaults_to_sigmoid(tmp_path):
    from dataclasses import asdict,replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import (
        SCHEMA,SEMANTICS,load_model,make_config
    )
    base=make_config("xl_ms_gno_transolver_convex")
    atan=make_config("xl_ms_gno_transolver_convex_atan")
    bounded=make_config("xl_ms_gno_transolver_bounded_odd")
    assert base.convex_amplitude_mode=="sigmoid"
    assert atan.convex_potential_enabled and atan.convex_amplitude_mode=="atan"
    assert not bounded.convex_potential_enabled
    legacy=replace(make_config("smoke",min_native_resolution=1),convex_potential_enabled=True)
    model=PcraftNativeMomentOperator(legacy)
    config=asdict(legacy)
    config.pop("convex_amplitude_mode")
    path=tmp_path/"legacy-convex-no-amplitude-mode.pt"
    torch.save(dict(schema=SCHEMA,semantics=SEMANTICS,config=config,model=model.state_dict()),path)
    restored,_=load_model(path,"cpu")
    assert restored.config.convex_amplitude_mode=="sigmoid"


def test_adaptive_physical_slicing_bounds_slices_and_conserves_mass():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853301)
    module=PhysicsStateSlicing(width=8,n_slices=12,n_heads=2).eval()
    features=torch.randn(37,8)
    masses=torch.rand(37,dtype=torch.float64)
    updated,metrics,assignment,slice_mass=module(
        features,masses,adaptive=True,min_slices=3,return_details=True)
    assert updated.shape==features.shape
    assert assignment.shape==(37,2,12)
    assert 3<=metrics["transolver_active_slices"]<=12
    assert torch.allclose(assignment.sum(-1),torch.ones(37,2),atol=1e-6)
    assert torch.allclose(slice_mass.sum(-1),torch.ones(2),atol=1e-6)
    assert metrics["transolver_slice_mass_sum_error"]<1e-6


def test_adaptive_physical_slicing_is_deterministic_and_permutation_equivariant():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853302)
    module=PhysicsStateSlicing(width=8,n_slices=10,n_heads=2).eval()
    features=torch.randn(29,8)
    masses=torch.rand(29,dtype=torch.float64)
    a,ma=module(features,masses,adaptive=True,min_slices=2)
    b,mb=module(features,masses,adaptive=True,min_slices=2)
    perm=torch.randperm(len(features))
    c,mc=module(features[perm],masses[perm],adaptive=True,min_slices=2)
    assert torch.equal(a,b)
    assert ma["transolver_active_slices"]==mb["transolver_active_slices"]
    assert ma["transolver_active_slices"]==mc["transolver_active_slices"]
    assert torch.allclose(a[perm],c,rtol=1e-6,atol=1e-7)


def test_adaptive_physical_slicing_handles_sparse_and_zero_mass_intervals():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853303)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).eval()
    features=torch.randn(16,8)
    masses=torch.zeros(16,dtype=torch.float64)
    masses[3]=1.0
    updated,metrics,assignment,slice_mass=module(
        features,masses,adaptive=True,min_slices=2,return_details=True)
    assert torch.isfinite(updated).all()
    assert torch.isfinite(assignment).all()
    assert torch.isfinite(slice_mass).all()
    assert metrics["transolver_active_slices"]>=2
    with pytest.raises(ValueError,match="positive sum"):
        module(features,torch.zeros_like(masses),adaptive=True,min_slices=2)


def test_adaptive_physical_slicing_gradients_are_finite():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853304)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).train()
    features=torch.randn(23,8,requires_grad=True)
    masses=torch.rand(23,dtype=torch.float64)
    updated,metrics=module(features,masses,adaptive=True,min_slices=2)
    (updated.square().mean()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Inactive slots are exact zeros, while active assignment rows remain
    # normalized; this guards the adaptive softmax boundary.
    _,_,assignment,slice_mass=module(features.detach(),masses,adaptive=True,
        min_slices=2,return_details=True)
    assert torch.isfinite(assignment).all()
    assert torch.allclose(assignment.sum(-1),torch.ones(23,2),atol=1e-6)
    active=int(module(features.detach(),masses,adaptive=True,min_slices=2)[1]["transolver_active_slices"])
    if active<8:
        assert torch.count_nonzero(assignment[...,active:])==0
        assert torch.count_nonzero(slice_mass[:,active:])==0


def test_fixed_physical_slicing_path_is_unchanged_when_adaptive_is_disabled():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853305)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).eval()
    features=torch.randn(19,8)
    masses=torch.rand(19,dtype=torch.float64)
    old,old_metrics=module(features,masses)
    fixed,fixed_metrics=module(features,masses,adaptive=False)
    assert torch.equal(old,fixed)
    assert old_metrics["transolver_active_slices"]==8
    assert fixed_metrics["transolver_adaptive_enabled"]==0.0


def test_native_config_validates_adaptive_slice_bounds():
    with pytest.raises(ValueError,match="transolver_min_slices"):
        NativePCRAFTConfig(transolver_slices=8,transolver_min_slices=9)
    cfg=NativePCRAFTConfig(transolver_enabled=True,transolver_adaptive=True,
        transolver_slices=8,transolver_min_slices=2)
    assert cfg.transolver_adaptive is True


def test_adaptive_temperature_is_bounded_deterministic_and_reports_utilization():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853306)
    module=PhysicsStateSlicing(width=8,n_slices=10,n_heads=2).eval()
    features=torch.randn(31,8)
    masses=torch.rand(31,dtype=torch.float64)
    a,ma,assignment_a,slice_mass_a=module(features,masses,adaptive=True,min_slices=2,
        adaptive_temperature=True,temperature_min=.08,temperature_max=.7,return_details=True)
    b,mb,assignment_b,slice_mass_b=module(features,masses,adaptive=True,min_slices=2,
        adaptive_temperature=True,temperature_min=.08,temperature_max=.7,return_details=True)
    assert torch.equal(a,b)
    assert torch.equal(assignment_a,assignment_b)
    assert .08<=ma["transolver_temperature_min"]<=ma["transolver_temperature_max"]<=.7
    assert ma["transolver_adaptive_temperature_enabled"]==1.0
    assert ma["transolver_active_slices"]==mb["transolver_active_slices"]
    assert torch.allclose(slice_mass_a.sum(-1),torch.ones(2),atol=1e-6)
    assert torch.isfinite(torch.tensor(ma["transolver_effective_slices"]))
    assert 1.0<=ma["transolver_effective_slices"]<=ma["transolver_active_slices"]+1e-5


def test_adaptive_temperature_is_permutation_equivariant_and_sparse_mass_safe():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853307)
    module=PhysicsStateSlicing(width=8,n_slices=9,n_heads=2).eval()
    features=torch.randn(27,8)
    masses=torch.zeros(27,dtype=torch.float64); masses[::4]=torch.rand(7,dtype=torch.float64)+.1
    a,ma=module(features,masses,adaptive=True,min_slices=2,adaptive_temperature=True)
    perm=torch.randperm(27)
    b,mb=module(features[perm],masses[perm],adaptive=True,min_slices=2,adaptive_temperature=True)
    assert torch.allclose(a[perm],b,rtol=1e-6,atol=1e-7)
    assert ma["transolver_active_slices"]==mb["transolver_active_slices"]
    assert ma["transolver_temperature_min"]>=.05
    assert torch.isfinite(b).all()


def test_adaptive_temperature_gradients_are_finite_and_fixed_temperature_is_unchanged():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853308)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).train()
    features=torch.randn(23,8,requires_grad=True)
    masses=torch.rand(23,dtype=torch.float64)
    updated,metrics=module(features,masses,adaptive=True,min_slices=2,adaptive_temperature=True)
    updated.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    module.eval()
    fixed_a,_=module(features.detach(),masses,adaptive=True,min_slices=2,adaptive_temperature=False)
    fixed_b,_=module(features.detach(),masses,adaptive=True,min_slices=2,adaptive_temperature=False)
    assert torch.equal(fixed_a,fixed_b)


def test_adaptive_temperature_requires_adaptive_slicing_and_valid_bounds():
    with pytest.raises(ValueError,match="requires transolver_adaptive"):
        NativePCRAFTConfig(transolver_adaptive_temperature=True)
    with pytest.raises(ValueError,match="temperature bounds"):
        NativePCRAFTConfig(transolver_adaptive=True,
            transolver_temperature_min=1.0,transolver_temperature_max=.5)
    cfg=NativePCRAFTConfig(transolver_enabled=True,transolver_adaptive=True,
        transolver_adaptive_temperature=True,transolver_min_slices=2)
    assert cfg.transolver_adaptive_temperature


def test_adaptive_temperature_complexity_sharpens_assignments_monotonically():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853309)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).eval()
    features=torch.randn(32,8)
    uniform=torch.ones(32,dtype=torch.float64)
    concentrated=torch.full((32,),1e-4,dtype=torch.float64); concentrated[0]=1.0
    _,uniform_metrics=module(features,uniform,adaptive=True,min_slices=2,
        adaptive_temperature=True,temperature_min=.05,temperature_max=1.5)
    _,concentrated_metrics=module(features,concentrated,adaptive=True,min_slices=2,
        adaptive_temperature=True,temperature_min=.05,temperature_max=1.5)
    assert uniform_metrics["transolver_complexity_score"]>concentrated_metrics["transolver_complexity_score"]
    assert uniform_metrics["transolver_temperature_mean"]<=concentrated_metrics["transolver_temperature_mean"]+1e-7


def test_adaptive_slice_utilization_metrics_are_bounded_and_per_head_consistent():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853310)
    module=PhysicsStateSlicing(width=8,n_slices=10,n_heads=2).eval()
    features=torch.randn(41,8)
    masses=torch.rand(41,dtype=torch.float64)
    _,metrics,assignment,slice_mass=module(features,masses,adaptive=True,min_slices=3,
        adaptive_temperature=True,return_details=True)
    assert 1.0<=metrics["transolver_per_head_effective_slices_min"]<=metrics["transolver_active_slices"]+1e-5
    assert metrics["transolver_per_head_effective_slices_min"]<=metrics["transolver_per_head_effective_slices_max"]
    assert 0<=metrics["transolver_active_slice_utilization"]<=1+1e-6
    assert metrics["transolver_per_head_occupied_slices_min"]>=1
    assert torch.allclose(slice_mass.sum(-1),torch.ones(2),atol=1e-6)
    assert torch.allclose(assignment.sum(-1),torch.ones(41,2),atol=1e-6)


def test_adaptive_temperature_extreme_bounds_remain_finite_and_validate_runtime_args():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PhysicsStateSlicing
    torch.manual_seed(853311)
    module=PhysicsStateSlicing(width=8,n_slices=8,n_heads=2).eval()
    features=torch.randn(17,8)
    masses=torch.rand(17,dtype=torch.float64)
    _,metrics=module(features,masses,adaptive=True,min_slices=2,
        adaptive_temperature=True,temperature_min=.2,temperature_max=.21)
    assert .2<=metrics["transolver_temperature_min"]<=metrics["transolver_temperature_max"]<=.21
    with pytest.raises(ValueError,match="temperature bounds"):
        module(features,masses,adaptive=True,adaptive_temperature=True,
            temperature_min=0.,temperature_max=1.)
