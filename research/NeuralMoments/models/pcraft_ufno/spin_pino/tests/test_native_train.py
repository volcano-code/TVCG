from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import M72Corpus


def fingerprint(doc):
    body={k:v for k,v in doc.items() if k!="fingerprint"}
    raw=json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_corpus(root):
    records=[]
    for split in ("train","val","test"):
        gp=root/f"{split}-geometry.npz"
        lp=root/f"{split}-labels.npz"
        cols=np.asarray([(x,y,1,7) for x in range(1,7) for y in range(1,7)],dtype=np.int32)
        np.savez(gp,columns=cols,origin=np.asarray([-1.,-1.,-1.]),h=np.asarray(.25),
            side=np.asarray(8,dtype=np.int32),fixed_moments=np.zeros(10),density=np.asarray(1.))
        q=np.asarray([[0.,1.,0,0,0,0,0,0,0,0],[0.,0,1.,0,0,0,0,0,0,0],
            [0.,0,0,1.,0,0,0,0,0,0],[0.,-1.,0,0,0,0,0,0,0,0]],dtype=np.float64)
        np.savez(lp,queries=q,moments=np.zeros((4,10)),jacobian=np.zeros((4,10,10)),
                 derivative_valid=np.asarray([True,True,False,True]))
        records.append(dict(id=f"{split}-0",status="qualified",split=split,group_id=f"group-{split}",
            geometry_file=gp.name,geometry_sha256=sha(gp),labels_file=lp.name,labels_sha256=sha(lp)))
    doc=dict(schema="m69.corpus.v1",scope="formal",frozen=True,records=records)
    doc["fingerprint"]=fingerprint(doc)
    mp=root/"operator-manifest.json"
    mp.write_text(json.dumps(doc,sort_keys=True))
    audit=dict(schema="m69.corpus-audit.v1",valid=True,ready_for_training=True,
        fingerprint=doc["fingerprint"],scope="formal",blockers=[],counts=dict(qualified_tasks=3))
    ap=root/"audit.json"
    ap.write_text(json.dumps(audit))
    return mp,ap


def test_m72_manifest_audit_hashes_and_native_geometry(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    corpus=M72Corpus(manifest,audit,grid_size=8,min_queries=4)
    assert len(corpus.rows)==3 and len(corpus.train_groups)==1
    geometry,labels=corpus.get(corpus.splits["train"][0])
    assert int(geometry["native_resolution"])==8
    assert geometry["pcraft_grid"].shape==(4,8,8,8)
    assert labels["queries"].shape==(4,10)
    assert int(labels["derivative_valid"].sum())==3
    assert len(corpus.verify_query_coverage(4))==3


def test_m72_audit_not_ready_is_rejected(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(audit.read_text())
    doc["ready_for_training"]=False
    doc["blockers"]=["fixture"]
    audit.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="audit"):
        M72Corpus(manifest,audit,grid_size=8,min_queries=1)


def test_manifest_path_traversal_is_rejected(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(manifest.read_text())
    doc["records"][0]["geometry_file"]="../escape.npz"
    doc["fingerprint"]=fingerprint(doc)
    manifest.write_text(json.dumps(doc,sort_keys=True))
    a=json.loads(audit.read_text())
    a["fingerprint"]=doc["fingerprint"]
    audit.write_text(json.dumps(a))
    corpus=M72Corpus(manifest,audit,grid_size=8,min_queries=1)
    with pytest.raises(ValueError,match="escapes"):
        corpus.get(corpus.splits["train"][0])


def test_cpu_preflight_runs_real_pcraft_m72_backward_and_checkpoint_roundtrip(tmp_path):
    from argparse import Namespace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import preflight
    manifest,audit=make_corpus(tmp_path)
    args=Namespace(device="cpu",amp="none",threads=1,seed=800071,preset="smoke",
        min_native_resolution=1,manifest=manifest,audit=audit,cache_mib=16.,
        batch_queries=2,chunk=16,learning_rate=3e-4,weight_decay=1e-4,
        jacobian_weight=.05,event_weight=.05,gradient_clip=1.,min_event_margin=0.,require_fd_audit=False,max_jacobian_error_bound=None,out=tmp_path/"preflight.json")
    report=preflight(args)
    assert report["status"]=="passed"
    assert report["native_intervals"]>0
    assert report["checkpoint_roundtrip_max_abs"]<=1e-10
    assert report["parameter_count"]>0
    assert report["gradient_norm"]>0


def test_validation_distribution_reports_tail_quantiles_and_rejects_nonfinite():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import _distribution_summary
    summary=_distribution_summary([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
    assert summary["count"]==10
    assert summary["mean"]==pytest.approx(4.5)
    assert summary["p50"]==pytest.approx(4.5)
    assert summary["p95"]==pytest.approx(8.55)
    assert summary["p99"]==pytest.approx(8.91)
    assert summary["max"]==9.
    assert _distribution_summary([]) is None
    with pytest.raises(FloatingPointError,match="nonfinite"):
        _distribution_summary([0.,float("nan")])


def test_m72_optional_event_and_derivative_audit_fields_are_loaded_and_validated(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(manifest.read_text())
    row=doc["records"][0]
    label_path=tmp_path/row["labels_file"]
    with np.load(label_path,allow_pickle=False) as z:
        arrays={k:np.array(z[k],copy=True) for k in z.files}
    arrays.update(
        event_type=np.asarray(["regular","regular","tangent","regular"],dtype="U16"),
        event_valid=np.asarray([True,True,False,True],dtype=bool),
        derivative_valid_reason=np.asarray(["smooth","smooth","event","smooth"],dtype="U16"),
        event_margin=np.asarray([.2,.1,0.,.3],dtype=np.float64),
        finite_difference_audit=np.asarray([True,True,False,True],dtype=bool),
        moment_error_bound=np.full(4,1e-8,dtype=np.float64),
        jacobian_error_bound=np.asarray([1e-6,2e-6,0.,1e-6],dtype=np.float64),
    )
    np.savez(label_path,**arrays)
    row["labels_sha256"]=sha(label_path)
    doc["fingerprint"]=fingerprint(doc)
    manifest.write_text(json.dumps(doc,sort_keys=True))
    audit_doc=json.loads(audit.read_text())
    audit_doc["fingerprint"]=doc["fingerprint"]
    audit.write_text(json.dumps(audit_doc))
    corpus=M72Corpus(manifest,audit,grid_size=8,min_queries=4)
    geometry,labels=corpus.get(corpus.splits["train"][0])
    assert labels["event_type"].tolist()==["regular","regular","tangent","regular"]
    assert labels["event_valid"].tolist()==[True,True,False,True]
    assert labels["derivative_valid_reason"][2]=="event"
    assert labels["finite_difference_audit"].dtype==bool
    assert labels["jacobian_error_bound"][1]==pytest.approx(2e-6)


def test_event_aware_derivative_supervision_mask():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import derivative_supervision_mask
    labels={
      "derivative_valid":np.asarray([True,True,True,False,True]),
      "event_margin":np.asarray([.2,.01,0.0,.4,.3]),
      "finite_difference_audit":np.asarray([True,False,True,True,True]),
      "jacobian_error_bound":np.asarray([1e-6,1e-6,1e-6,1e-6,1e-2]),
    }
    idx=np.arange(5)
    base=derivative_supervision_mask(labels,idx)
    assert base.tolist()==[True,True,False,False,True]
    gated=derivative_supervision_mask(labels,idx,min_event_margin=.05,
        require_fd_audit=True,max_jacobian_error_bound=1e-4)
    assert gated.tolist()==[True,False,False,False,False]


def test_event_aware_derivative_mask_fails_closed_when_requested_labels_missing():
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import derivative_supervision_mask
    labels={"derivative_valid":np.asarray([True,True])}
    with pytest.raises(ValueError,match="event_margin labels are absent"):
        derivative_supervision_mask(labels,np.arange(2),min_event_margin=.1)
    with pytest.raises(ValueError,match="finite-difference audit required"):
        derivative_supervision_mask(labels,np.arange(2),require_fd_audit=True)
    with pytest.raises(ValueError,match="error-bound gate requested"):
        derivative_supervision_mask(labels,np.arange(2),max_jacobian_error_bound=1e-4)


def test_event_margin_and_fd_audit_are_applied_to_jacobian_training_loss(tmp_path):
    import torch
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import loss_for_batch
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PcraftNativeMomentOperator,make_config
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_native_pcraft import small_geometry
    model=PcraftNativeMomentOperator(make_config("smoke",min_native_resolution=1)).train()
    geometry=small_geometry()
    cache=model.encode(geometry,chunk=16)
    q=np.asarray([[0.,1.,0,0,0,0,0,0,0,0],
                  [0.,0,1.,0,0,0,0,0,0,0]],dtype=np.float64)
    labels={"queries":q,"moments":np.zeros((2,10)),"jacobian":np.zeros((2,10,10)),
      "derivative_valid":np.asarray([True,True]),"event_valid":np.asarray([True,False],dtype=bool),"event_margin":np.asarray([.2,0.]),
      "finite_difference_audit":np.asarray([True,True])}
    _,_,derivative,event_loss=loss_for_batch(model,cache,labels,np.arange(2),.05,"cpu",
      min_event_margin=.01,require_fd_audit=True)
    # Only the first, event-regular query contributes to the Jacobian loss.
    pred,jac=model.moments_and_jacobian(cache,torch.tensor(q,dtype=torch.float64),create_graph=True)
    expected=(jac[0]/(cache.moment_scales[:,None]*cache.coordinate_scales[None,:])).square().mean()
    assert torch.allclose(derivative,expected)
    assert float(event_loss)==0.0


def test_optional_event_head_is_jointly_supervised_by_derivative_valid_labels():
    import torch
    from dataclasses import replace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import loss_for_batch
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_pcraft import PcraftNativeMomentOperator,make_config
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.tests.test_native_pcraft import small_geometry
    cfg=replace(make_config("smoke",min_native_resolution=1),event_validity_head_enabled=True)
    model=PcraftNativeMomentOperator(cfg).train()
    cache=model.encode(small_geometry(),chunk=16)
    q=np.asarray([[0.,1.,0,0,0,0,0,0,0,0],
                  [0.,0,1.,0,0,0,0,0,0,0]],dtype=np.float64)
    labels={"queries":q,"moments":np.zeros((2,10)),"jacobian":np.zeros((2,10,10)),
      "derivative_valid":np.asarray([True,False]),"event_valid":np.asarray([True,False],dtype=bool)}
    loss,_,_,event_loss=loss_for_batch(model,cache,labels,np.arange(2),.05,"cpu",event_weight=.1)
    assert torch.isfinite(loss) and torch.isfinite(event_loss) and float(event_loss)>0
    loss.backward()
    grads=[p.grad for p in model.event_validity_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_cpu_preflight_accepts_moment_only_labels(tmp_path):
    from argparse import Namespace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import preflight
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(manifest.read_text())
    for row in doc["records"]:
        lp=tmp_path/row["labels_file"]
        with np.load(lp,allow_pickle=False) as z:
            arrays={k:np.array(z[k],copy=True) for k in z.files}
        arrays["derivative_valid"]=np.zeros_like(arrays["derivative_valid"],dtype=bool)
        np.savez(lp,**arrays)
        row["labels_sha256"]=sha(lp)
    doc["fingerprint"]=fingerprint(doc)
    manifest.write_text(json.dumps(doc,sort_keys=True))
    audit_doc=json.loads(audit.read_text())
    audit_doc["fingerprint"]=doc["fingerprint"]
    audit.write_text(json.dumps(audit_doc))
    args=Namespace(device="cpu",amp="none",threads=1,seed=800072,preset="smoke",
        min_native_resolution=1,manifest=manifest,audit=audit,cache_mib=16.,
        batch_queries=2,chunk=16,learning_rate=3e-4,weight_decay=1e-4,
        jacobian_weight=.05,event_weight=.05,gradient_clip=1.,min_event_margin=0.,
        require_fd_audit=False,max_jacobian_error_bound=None,out=tmp_path/"moment-only-preflight.json")
    report=preflight(args)
    assert report["status"]=="passed"
    assert report["moment_only_preflight"] is True
    assert report["derivative_supervision_queries"]==0
    assert np.isfinite(report["moment_loss"])
