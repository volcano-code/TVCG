"""Actual P-CRAFT -> V2 token response -> native256 training integration."""
import numpy as np
import pytest
import torch

from spin_adapter.unified_factory import build_model,architecture_manifest
from spin_adapter.ablation_initialization import initialize_from_reference
from spin_adapter.unified_prediction import predict_full
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns


def test_v2_native256_single_response_joint_update(tmp_path):
    overrides={"pcraft_preset":"smoke","pcraft_grid_size":8,"multiscale_grids":(8,16),
               "v2_width":16,"v2_heads":4,"v2_token_grid":8,"query_chunk":2}
    model=build_model("compact_pino_v2_256",device="cpu",config_overrides=overrides)
    assert model.shared_encoder.config.min_native_resolution==256
    assert not hasattr(model.shared_encoder,"query_head")
    assert not hasattr(model,"sqno") and not hasattr(model,"query_fusion_gate")
    assert not any("m80" in name or "sqno" in name for name,_ in model.named_parameters())
    columns=np.array([[50,50,0,210],[70,120,20,200],[120,140,10,240],[150,60,40,230],[210,200,0,255]],dtype=np.int32)
    geom=geometry_from_columns({"columns":columns,"origin":np.array([-.5,-.5,-.5]),
        "h":np.array(1/256),"side":np.array(256),"density":np.array(1.7)})
    wrong=geometry_from_columns({"columns":columns,"origin":np.array([-.5,-.5,-.5]),
        "h":np.array(1/512),"side":np.array(512),"density":np.array(1.7)})
    with pytest.raises(ValueError,match="exact native resolution 256"):
        model.encode_geometry(wrong)
    model.train()
    cache=model.encode_geometry(geom)
    assert 1<len(cache.token_cache.moments)<=8**3
    torch.testing.assert_close(cache.token_cache.moments.sum(0),cache.shared_cache.design_moments,rtol=1e-8,atol=1e-10)
    q=torch.tensor([[.01,.15,.12,1.,.05,.04,.02,.03,.04,.02],
                    [-.08,.1,-.2,1.,.04,.03,-.03,.04,.01,.03],
                    [.05,.2,.1,1.,.03,.05,.02,.04,.03,-.02]],dtype=torch.float64)
    result=model.evaluate_queries(cache,q,create_graph=True)
    m=result["moments"].detach()*.8
    j=result["jacobian_coefficients"].detach()*.9
    basis=torch.full((3,10,8),float("nan"),dtype=torch.float64)
    rj=torch.full_like(basis,float("nan"))
    for i,d in ((1,3),(2,8)):
        basis[i,:,:d]=torch.eye(10,dtype=torch.float64)[:,:d]
        rj[i,:,:d]=j[i]@basis[i,:,:d]
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    optimizer=model.build_optimizer(learning_rate=1e-4)
    masks=dict(valid_mask=torch.ones(3,dtype=torch.bool),
               jacobian_valid_mask=torch.tensor([True,False,False]),
               restricted_basis=basis,restricted_jacobian=rj,
               restricted_valid=torch.tensor([False,True,True]),restricted_dimension=torch.tensor([0,3,8]))
    metrics=model.train_step(optimizer,cache,q,m,j,**masks)
    assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"]>0
    assert (metrics["valid_jacobian_queries"],metrics["valid_restricted_queries"],metrics["restricted_dimensions"])==(1,2,11)
    for prefix in ("shared_encoder.","response."):
        assert any(not torch.equal(p,before[name]) for name,p in model.named_parameters() if name.startswith(prefix))
    with pytest.raises(ValueError,match="stale"):
        model.evaluate_queries(cache,q)
    path=tmp_path/"v2.pt"
    contract=model.training_contract()
    model.save_checkpoint(path,optimizer=optimizer,training_contract=contract,data_fingerprint="synthetic-v2")
    restored=build_model("compact_pino_v2_256",device="cpu",config_overrides=overrides)
    new_optimizer=restored.build_optimizer(learning_rate=1e-4)
    restored.load_checkpoint(path,optimizer=new_optimizer,expected_training_contract=contract,
                             expected_data_fingerprint="synthetic-v2")
    torch.testing.assert_close(model.state_dict(),restored.state_dict(),rtol=0,atol=0)
    model.eval();restored.eval()
    with torch.no_grad():
        c=model.encode_geometry(geom,inference=True)
        d=restored.encode_geometry(geom,inference=True)
        expected=predict_full(model,c,q,query_chunk_size=1)
        actual=predict_full(restored,d,q)
        torch.testing.assert_close(actual.moments,expected.moments,rtol=2e-6,atol=1e-9)
        torch.testing.assert_close(actual.jacobian_coefficients,expected.jacobian_coefficients,rtol=2e-6,atol=1e-9)
        torch.testing.assert_close(model.predict_moments(c,q),expected.moments,rtol=2e-6,atol=1e-9)
    assert expected.uq_prediction_available is False and expected.event_prediction_available is False


@pytest.mark.parametrize("name,grid",[("compact_pino_v2_256",16),("compact_pino_v2_tokens8_256",8)])
def test_production_v2_factory_and_shared_initialization(name,grid):
    manifest=architecture_manifest(name)
    assert manifest["effective_geometry"]["max_physical_tokens"]==grid**3
    assert not manifest["modules"]["m80"] and not manifest["modules"]["sqno"]
    model=build_model(name,device="cpu")
    report=initialize_from_reference(model,reference_config_name="full_v1_candidate",seed=830251)
    assert report["v2_shared_encoder_complete"] and report["new_v2_response_keys"]
    assert all(key.startswith("response.") for key in report["new_v2_response_keys"])
    assert not report["mismatched_keys"]
    assert all(key in report["copied_keys"] for key in model.state_dict() if key.startswith("shared_encoder."))
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in build_model("compact_lite_256").parameters())
    with pytest.raises(ValueError,match="branch_aux_weight"):
        model.training_contract(branch_aux_weight=.25)
