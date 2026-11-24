"""Actual native-side-256 smoke with reduced neural/context width, plus exact cache parity."""
import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import build_model
from spin_adapter.tests.test_unified_m80_sqno import make_unified, make_geometry, queries


def test_native256_factory_joint_update_roundtrip(tmp_path):
    # The actual factory still enforces native_resolution==256 and min==256.
    # Only context-grid / channel sizes are reduced for CPU testing.
    overrides={
        "pcraft_preset":"smoke","pcraft_grid_size":8,"multiscale_grids":(8,16),
        "sqno_hidden_dim":4,"sqno_final_hidden_dim":16,"sqno_num_layers":2,
        "sqno_neighbors":4,"sqno_chunk_intervals":2,"sqno_chunk_nodes":64,
        "query_chunk":2,"query_fusion_hidden":8,"geometry_attention_chunk":2,
    }
    model=build_model("compact_v1_candidate_256",device="cpu",config_overrides=overrides)
    assert model.shared_encoder.config.min_native_resolution==256
    geom=geometry_from_columns({"columns":np.array([[1,1,1,5],[1,2,2,7],[2,1,0,6],[3,3,1,8],[4,3,0,7]]),
        "origin":np.array([-1.,-2.,.5]),"h":np.array(.25),"side":np.array(256),"density":np.array(1.7)})
    model.train()
    cache=model.encode_geometry(geom)
    assert cache.shared_cache.scale_ids.tolist()==[8,16]
    assert cache.native_intervals==5 and cache.sqno_cache.n_nodes==40
    q=queries()[:2]
    out=model.evaluate_queries(cache,q,create_graph=True)
    optimizer=model.build_optimizer(learning_rate=1e-4)
    report=model.train_step(optimizer,cache,q,torch.zeros_like(out["moments"]),
                            torch.zeros_like(out["jacobian_coefficients"]))
    assert np.isfinite(report["loss"]) and report["gradient_norm"]>0
    with pytest.raises(ValueError,match="stale"):
        model.evaluate_queries(cache,q)
    path=tmp_path/"compact256.pt"
    contract=model.training_contract()
    model.save_checkpoint(path,optimizer=optimizer,training_contract=contract,data_fingerprint="synthetic-native256")
    restored=build_model("compact_v1_candidate_256",device="cpu",config_overrides=overrides)
    restored_optimizer=restored.build_optimizer(learning_rate=1e-4)
    restored.load_checkpoint(path,optimizer=restored_optimizer,expected_training_contract=contract,
                             expected_data_fingerprint="synthetic-native256")
    torch.testing.assert_close(model.state_dict(),restored.state_dict(),rtol=0,atol=0)
    model.eval();restored.eval()
    with torch.no_grad():
        expected=model.evaluate_queries(model.encode_geometry(geom,inference=True),q)
        actual=restored.evaluate_queries(restored.encode_geometry(geom,inference=True),q)
    torch.testing.assert_close(actual["moments"],expected["moments"])
    torch.testing.assert_close(actual["jacobian_coefficients"],expected["jacobian_coefficients"])


@pytest.mark.parametrize("attention,quadrature",[(True,False),(False,True),(True,True)])
def test_inference_cache_matches_uncached_and_respects_budget(attention,quadrature):
    model=make_unified(cross_attention_enabled=True,cross_attention_chunk=2,
                       cross_scale_attention_enabled=True,fusion_mode="query_conditioned",
                       query_fusion_hidden=8).eval()
    torch.manual_seed(91)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=.02)
    torch.nn.init.normal_(model.query_fusion_gate[-1].weight,std=.2)
    cache=model.encode_geometry(make_geometry(),inference=True)
    q=queries()[:3]*.7
    with torch.no_grad():
        expected=model.evaluate_queries(cache,q)
        base_m=model.predict_moments(cache,q)
    empty,zero=model.prepare_inference_acceleration(cache,max_extra_bytes=0)
    assert zero["extra_bytes"]==0 and not zero["attention_cached"] and not zero["quadrature_cached"]
    accelerated,receipt=model.prepare_inference_acceleration(cache,max_extra_bytes=1024*1024,
                                                             attention=attention,quadrature=quadrature)
    assert receipt["attention_cached"]==attention and receipt["quadrature_cached"]==quadrature
    assert 0<receipt["extra_bytes"]<=receipt["max_extra_bytes"]
    with torch.no_grad():
        actual=model.evaluate_queries(accelerated,q)
        actual_m=model.predict_moments(accelerated,q)
    for k in ("moments","jacobian_coefficients"):
        torch.testing.assert_close(actual[k],expected[k],rtol=2e-6,atol=2e-8)
    torch.testing.assert_close(actual_m,base_m,rtol=2e-6,atol=2e-8)
    assert model.geometry_encode_calls==1
    with torch.no_grad():
        cache.sqno_cache.node_weights.add_(0.)
    with pytest.raises(ValueError,match="mutated"):
        model.predict_moments(accelerated,q)
