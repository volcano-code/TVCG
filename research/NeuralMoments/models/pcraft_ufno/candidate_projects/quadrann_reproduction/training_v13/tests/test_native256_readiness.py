"""Phase 3.2.3 contract regressions; no native teacher claim."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.unified_factory import (
    architecture_manifest, build_model, get_config,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.native256_readiness import (
    SCHEMA, MOMENT_ORDER, PHYSICAL_FIELDS, admit_native_training, derivative_scales,
)
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.unified_corpus import (
    UnifiedOperatorCorpus, file_sha256,
)
from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import geometry_from_columns
from spin_adapter.tests.test_unified_m80_sqno import make_unified, make_geometry, queries


@pytest.mark.parametrize("name,grids", [
    ("ablation_g1_t1_256",[32,64,128]),
    ("ablation_no_cross_scale_256",[32,64,128]),
    ("compact_v1_candidate_256",[64,128]),
    ("compact_lite_256",[32,64]),
    ("compact_single_256",[64]),
])
def test_exact_native_configs(name, grids):
    m = architecture_manifest(name)
    assert m["pcraft_config"]["min_native_resolution"] == 256
    assert m["effective_geometry"]["required_native_resolution"] == 256
    assert m["effective_geometry"]["grids"] == grids
    assert get_config("full_v1_candidate").native_resolution is None
    assert architecture_manifest("full_v1_candidate")["pcraft_config"]["min_native_resolution"] == 512


def test_exact_resolution_model_rejects_wrong_side_before_encoding():
    model = build_model("compact_v1_candidate_256", device="cpu").eval()
    with pytest.raises(ValueError, match="exact native resolution 256"):
        model.encode_geometry(make_geometry(), inference=True)


@pytest.mark.parametrize("mode", ["global_scalar", "query_conditioned"])
@pytest.mark.parametrize("variant", ["independent","parallel_gno_attention"])
def test_moment_only_matches_full_without_calling_full_jacobian(monkeypatch,mode,variant):
    model=make_unified(gno_enabled=variant=="parallel_gno_attention",
        cross_attention_enabled=True,cross_attention_chunk=2,
        cross_scale_attention_enabled=True,interaction_variant=variant,
        fusion_mode=mode,query_fusion_hidden=8).eval()
    torch.manual_seed(88)
    torch.nn.init.normal_(model.geometry_physics_attention.output[-1].weight,std=.02)
    torch.nn.init.normal_(model.cross_scale_attention.out[-1].weight,std=.02)
    if model.query_fusion_gate is not None:
        torch.nn.init.normal_(model.query_fusion_gate[-1].weight,std=.2)
    cache=model.encode_geometry(make_geometry(),inference=True)
    q=queries()[:2]*.7
    expected=model.evaluate_queries(cache,q)["moments"]
    def forbidden(*args,**kwargs):
        raise AssertionError("full Jacobian path was called")
    monkeypatch.setattr(model.shared_encoder,"moments_and_jacobian",forbidden)
    monkeypatch.setattr(model.geometry_physics_attention,"forward_with_query_jacobian",forbidden)
    torch.testing.assert_close(model.predict_moments(cache,q),expected,rtol=2e-6,atol=2e-8)
    torch.testing.assert_close(model.predict_moments(cache,q[0]),expected[0],rtol=2e-6,atol=2e-8)
    model.train()
    with pytest.raises(ValueError,match="inference-only"):
        model.predict_moments(cache,q)


@pytest.fixture
def admitted(tmp_path):
    rows=[]
    receipts={}
    for split in ("train","val","test"):
        g=tmp_path/f"{split}_g.npz"
        l=tmp_path/f"{split}_l.npz"
        np.savez(g,columns=np.array([[1,1,0,4]]),origin=np.zeros(3),h=np.array(.01),
                 side=np.array(256),fixed_moments=np.zeros(10))
        q=np.ones((3,10))
        j=np.full((3,10,10),np.nan)
        j[0]=2 if split=="train" else 200
        basis=np.full((3,10,8),np.nan)
        restricted=np.full_like(basis,np.nan)
        basis[1,:,:3]=np.eye(10)[:,:3];basis[2]=np.eye(10)[:,:8]
        restricted[1,:,:3]=3;restricted[2]=8
        np.savez(l,queries=q,moments=np.ones((3,10)),valid_mask=np.ones(3,dtype=bool),
                 jacobian=j,derivative_valid=np.array([True,False,False]),
                 restricted_basis=basis,restricted_jacobian=restricted,
                 restricted_valid=np.array([False,True,True]),restricted_dimension=np.array([0,3,8]))
        row={"id":split,"split":split,"group_id":split,"source_type":"TRELLIS",
             "geometry_file":g.name,"labels_file":l.name,"geometry_sha256":file_sha256(g),
             "labels_sha256":file_sha256(l),"native_resolution":256}
        rows.append(row)
        if split!="test":
            receipts[split]={"geometry_sha256":row["geometry_sha256"],"labels_sha256":row["labels_sha256"],
                              "status":"approved_native_discrete"}
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"schema":UnifiedOperatorCorpus.SCHEMA,"records":rows}))
    contract=tmp_path/"contract.json"
    data={k:"synthetic-only" for k in PHYSICAL_FIELDS}
    data.update(schema=SCHEMA,manifest_sha256=file_sha256(manifest),native_resolution=256,
                retained_side="positive",coefficient_multipliers=[1,1,1,1,1,1,1,2,2,2],
                status="approved_native_discrete",moment_order=MOMENT_ORDER,jacobian_basis="raw_coefficients",
                records=receipts)
    contract.write_text(json.dumps(data))
    return manifest,contract


def test_admission_and_train_only_scales(admitted,monkeypatch):
    manifest,contract=admitted
    c=UnifiedOperatorCorpus(manifest,grid_size=64)
    original=c.load_labels
    def guarded(row):
        assert row["split"]!="test"
        return original(row)
    monkeypatch.setattr(c,"load_labels",guarded)
    report=admit_native_training(c,"compact_lite_256",contract)
    assert report["reviewed_train_dev_records"]==2
    assert report["locked_test_labels_opened"] is False
    def train_only(row):
        assert row["split"]=="train"
        return original(row)
    monkeypatch.setattr(c,"load_labels",train_only)
    scales=derivative_scales(c)
    np.testing.assert_allclose(scales["jacobian_scale"],2)
    np.testing.assert_allclose(scales["restricted_scale"],np.array([[3.]*10,[8.]*10]))
    assert scales["restricted_scale_order"]==["r3","r8"]
    assert scales["query_counts"]=={"full":1,"r3":1,"r8":1}


def test_admission_rejects_missing_or_changed_receipt(admitted):
    manifest,contract=admitted
    c=UnifiedOperatorCorpus(manifest,grid_size=64)
    with pytest.raises(ValueError,match="native-cut-contract"):
        admit_native_training(c,"compact_lite_256")
    doc=json.loads(contract.read_text())
    doc["records"]["val"]["labels_sha256"]="0"*64
    contract.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="receipt/asset"):
        admit_native_training(c,"compact_lite_256",contract)
