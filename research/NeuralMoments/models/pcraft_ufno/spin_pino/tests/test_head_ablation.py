from __future__ import annotations
import json
import pytest
from research.NeuralMoments.models.pcraft_ufno.spin_pino.head_ablation import (
    HEADS, _stats, aggregate, assert_matched_backbones,
)

def test_head_presets_share_same_xl_multiscale_gno_transolver_trunk():
    c=assert_matched_backbones()
    assert tuple(c)==HEADS
    a,b,d=(c[x] for x in HEADS)
    for k in a:
        if k not in {"convex_potential_enabled","convex_potential_components","convex_amplitude_mode"}:
            assert a[k]==b[k]==d[k], k
    assert a["multiscale_grids"]==(32,64,128)
    assert a["gno_enabled"] and a["transolver_enabled"]
    assert not a["convex_potential_enabled"] and b["convex_potential_enabled"] and d["convex_potential_enabled"]
    assert b["convex_amplitude_mode"]=="sigmoid" and d["convex_amplitude_mode"]=="atan"

def test_stats_finite_and_missing():
    assert _stats([1.,3.])["mean"]==2.
    assert _stats([None]) is None
    with pytest.raises(FloatingPointError): _stats([float("nan")])

def test_aggregate_requires_full_seed_head_matrix_and_never_freezes(tmp_path):
    seeds=[11,13,17]; plan={"fingerprint":"p","dataset":{"fingerprint":"d"},"seeds":seeds,"heads":list(HEADS)}
    pp=tmp_path/"plan.json"; pp.write_text(json.dumps(plan)); rows=[]
    for h in HEADS:
        for s in seeds:
            r={"schema":"m80.pcraft-native-training.v1","contract":{"corpus_fingerprint":"d","seed":s},
              "best_checkpoint_validation":{"moment_scaled_mse":.1,"jacobian_scaled_mse":.2,
              "moment_query_rmse":{"p95":.3},"jacobian_query_rmse":{"p95":.4},
              "convex_amplitude":{"fraction":{"p95":.99},"effective":{"p95":.02}}},
              "gradient_diagnostics":{"global_norm":{"p95":1.2}},"steps_completed":20,
              "stop_reason":"step_budget","elapsed_seconds":2.}
            rp=tmp_path/f"{h}-{s}.json"; rp.write_text(json.dumps(r))
            rows.append({"preset":h,"seed":s,"report":str(rp)})
    ip=tmp_path/"index.json"; ip.write_text(json.dumps({"plan_fingerprint":"p","runs":rows}))
    out=tmp_path/"aggregate.json"; result=aggregate(pp,ip,out)
    assert result["freeze_decision"]["status"]=="pending"
    assert result["freeze_decision"]["test_split_used_for_selection"] is False
    assert result["heads"][HEADS[0]]["moment_scaled_mse"]["n"]==3

def test_aggregate_rejects_missing_run(tmp_path):
    pp=tmp_path/"p.json"; pp.write_text(json.dumps({"fingerprint":"p","dataset":{"fingerprint":"d"},"seeds":[1,2,3],"heads":list(HEADS)}))
    ip=tmp_path/"i.json"; ip.write_text(json.dumps({"plan_fingerprint":"p","runs":[]}))
    with pytest.raises(ValueError,match="missing"): aggregate(pp,ip,tmp_path/"o.json")
