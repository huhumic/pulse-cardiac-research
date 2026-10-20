"""Synthetic reports verify strict validation; no Kaggle, GPU or HF downloads."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulse_repro.experiment_protocol import (ProtocolError, load_fold,
        pooled_summary, paired_compare)
from pulse_repro.experiment_report import main as report_main


def make_report(root: Path, fold: int, *, kind: str="official", uplift=0.0,
                tta=True, partial=False, control="correct", hash_string="a"*64,
                skip_surface=False):
    root.mkdir(parents=True,exist_ok=True)
    ids = [f"patient{i:03d}" for i in range(fold*20+1, fold*20+21)]
    rows=[]
    for j, patient in enumerate(ids):
        for phase in ("ED","ES"):
            # Phase-dependent values; ensure patient-then-phase weighting is tested.
            d = .79 + j*.001 + (.02 if phase == "ED" else 0) + uplift
            row={"patient":patient, "phase":phase, "seconds":.2}
            for cls in ("RV","Myo","LV"):
                row[f"{cls}_dice"]=d + {"RV":.03, "Myo":0, "LV":.06}[cls]
                row[f"{cls}_hd95_mm"]=2.5 - uplift*2 + j*.01
                row[f"{cls}_assd_mm"]=.35 - uplift*.5 + j*.002
                row[f"{cls}_status"]="valid"
            rows.append(row)
    df=pd.DataFrame(rows)
    if skip_surface:
        df.loc[0,"RV_status"]="one_empty"
        df.loc[0,"RV_hd95_mm"]=np.nan
        df.loc[0,"RV_assd_mm"]=np.nan
        df.loc[0,"RV_dice"]=0.0
    df.to_csv(root/"per_volume.csv",index=False)
    metric_cols=[col for col in df if col.endswith(("_dice","_hd95_mm","_assd_mm"))]
    means=df.groupby("patient")[metric_cols].mean().mean(axis=0)
    summary={"fold":fold,"seed":42,"tta":tta,"full_fold":not partial,
             "model_variant":kind,"val_patients":ids,"n_valid_patients":20,"n_volumes":40,
             "weights_sha256":hash_string,"per_patient_means":means.to_dict(),
             "undefined_surface_case_counts":{c:int((df[f"{c}_status"]!="valid").sum())
                 for c in ("RV","Myo","LV")}}
    if kind=="cine-reliability-v2":
        summary["variant"]=summary.pop("model_variant")
        summary["validation_patients"]=summary.pop("val_patients")
        summary["n_patients"]=summary.pop("n_valid_patients")
        summary["checkpoint_sha256"]=summary.pop("weights_sha256")
        summary["temporal_control"]=control
    (root/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
    return root


def patch_summary(folder, **kwargs):
    path=folder/"summary.json"
    record=json.loads(path.read_text())
    record.update(kwargs)
    path.write_text(json.dumps(record))


def test_accepts_full_original_and_cine_reports(tmp_path):
    a=load_fold(make_report(tmp_path/"a",0))
    b=load_fold(make_report(tmp_path/"b",0,kind="cine-reliability-v2",uplift=.01))
    assert len(a.patients)==len(b.patients)==20
    assert a.patients.Mean_Dice.mean()<b.patients.Mean_Dice.mean()
    assert a.fold == b.fold == 0


def test_bootstrap_patient_paired_dice_and_distance_direction(tmp_path):
    ref=load_fold(make_report(tmp_path/"ref",0))
    cand=load_fold(make_report(tmp_path/"cand",0,kind="selective-v1",uplift=.01))
    per, result=paired_compare([ref],[cand],bootstrap_reps=1000,seed=0)
    assert len(per)==20
    assert result["n_volumes"]==40
    d=result["metrics"]["Mean_Dice"]
    assert abs(d["delta_candidate_minus_reference"]-.01)<1e-12
    assert d["delta_95ci"][0]>0
    h=result["metrics"]["RV_hd95_mm"]
    assert h["delta_candidate_minus_reference"]<0
    assert h["benefit_positive"]>0
    assert h["benefit_95ci"][0]>0


def test_five_fold_pool_and_disjoint_100(tmp_path):
    folds=[load_fold(make_report(tmp_path/f"f{i}",i)) for i in range(5)]
    frame,summary=pooled_summary(folds)
    assert len(frame)==100 and summary["scope"]=="five_fold_internal_cv"
    assert summary["metrics"]["RV_dice"]["n_finite"]==100
    paired,comparison=paired_compare(folds,[load_fold(make_report(tmp_path/f"c{i}",i,
                                           kind="selective-v1",uplift=.005)) for i in range(5)],
                                     bootstrap_reps=1000)
    assert comparison["n_patients"]==100
    assert abs(comparison["metrics"]["Mean_Dice"]["delta_candidate_minus_reference"]-.005)<1e-12


def test_refuses_incomplete_fold(tmp_path):
    folder=make_report(tmp_path/"f",0,partial=True)
    with pytest.raises(ProtocolError,match="Incomplete fold"):
        load_fold(folder)


def test_refuses_non_tta_fold(tmp_path):
    with pytest.raises(ProtocolError,match="4-flip TTA"):
        load_fold(make_report(tmp_path/"f",0,tta=False))


def test_refuses_misaligned_patient_list(tmp_path):
    folder=make_report(tmp_path/"f",0)
    patch_summary(folder,val_patients=["patient999"]*20)
    with pytest.raises(ProtocolError,match="CSV patient list"):
        load_fold(folder)


def test_refuses_missing_one_phase(tmp_path):
    folder=make_report(tmp_path/"f",0)
    df=pd.read_csv(folder/"per_volume.csv")
    df.loc[0,"phase"]="ES"
    df.to_csv(folder/"per_volume.csv",index=False)
    with pytest.raises(ProtocolError,match="Duplicate patient/phase"):
        load_fold(folder)


def test_refuses_duplicate_patient_between_folds(tmp_path):
    folder=make_report(tmp_path/"f",0)
    a=load_fold(folder)
    # A report cannot be presented as a different fold without assigning different patients.
    folder2=make_report(tmp_path/"g",1)
    df=pd.read_csv(folder2/"per_volume.csv")
    df["patient"]=[f"patient{i:03d}" for i in range(1,21) for _ in range(2)]
    df.to_csv(folder2/"per_volume.csv",index=False)
    patch_summary(folder2,val_patients=[f"patient{i:03d}" for i in range(1,21)])
    b=load_fold(folder2)
    with pytest.raises(ProtocolError,match="Patient leakage"):
        pooled_summary([a,b])


def test_refuses_bad_summary_mean(tmp_path):
    folder=make_report(tmp_path/"f",0)
    info=json.loads((folder/"summary.json").read_text())
    info["per_patient_means"]["RV_dice"]+=.02
    patch_summary(folder,per_patient_means=info["per_patient_means"])
    with pytest.raises(ProtocolError,match="Summary mismatch"):
        load_fold(folder)


def test_refuses_undefined_surface_by_default_and_allows_explicitly(tmp_path):
    a=load_fold(make_report(tmp_path/"f",0,skip_surface=True))
    b=load_fold(make_report(tmp_path/"g",0,kind="selective-v1"))
    with pytest.raises(ProtocolError,match="Missing paired observations"):
        paired_compare([a],[b],bootstrap_reps=1000)
    _,info=paired_compare([a],[b],bootstrap_reps=1000,allow_undefined_surface=True)
    assert info["metrics"]["RV_hd95_mm"]["n_complete_pairs"]==19
    assert info["metrics"]["RV_hd95_mm"]["n_missing_pairs"]==1


def test_refuses_invented_zero_distance_for_empty_surface(tmp_path):
    folder=make_report(tmp_path/"f",0,skip_surface=True)
    df=pd.read_csv(folder/"per_volume.csv")
    df.loc[0,"RV_assd_mm"]=0.0
    df.to_csv(folder/"per_volume.csv",index=False)
    with pytest.raises(ProtocolError,match="Undefined RV surface distances"):
        load_fold(folder)


def test_refuses_counterfactual_mixed_checkpoint(tmp_path):
    ref=load_fold(make_report(tmp_path/"f",0,kind="cine-reliability-v2"))
    cf=load_fold(make_report(tmp_path/"g",0,kind="cine-reliability-v2",
                  control="swap",hash_string="b"*64),allow_counterfactual=True)
    with pytest.raises(ProtocolError,match="identical checkpoint"):
        paired_compare([ref],[cf],bootstrap_reps=1000)


def test_allows_fixed_checkpoint_counterfactual(tmp_path):
    a=load_fold(make_report(tmp_path/"f",0,kind="cine-reliability-v2"))
    b=load_fold(make_report(tmp_path/"g",0,kind="cine-reliability-v2",
                     control="repeat-target",uplift=-.01),allow_counterfactual=True)
    _,info=paired_compare([a],[b],bootstrap_reps=1000)
    assert info["candidate_control"]=="repeat-target"


def test_refuses_wrong_split_manifest(tmp_path):
    folder=make_report(tmp_path/"f",0)
    manifest={"seed":42,"nfolds":5,"folds":{"0":{"val_patients":["patient900"]*20}}}
    with pytest.raises(ProtocolError,match="frozen patient split"):
        load_fold(folder,split_manifest=manifest)


def test_cli_audit_aggregate_compare_smoke(tmp_path, capsys):
    a=make_report(tmp_path/"ref",0)
    b=make_report(tmp_path/"cand",0,kind="selective-v1",uplift=.01)
    assert report_main(["audit","--eval-dir",str(a)]) == 0
    assert report_main(["aggregate","--eval-dirs",str(a),"--out-dir",str(tmp_path/"agg")]) == 0
    assert report_main(["compare","--reference",str(a),"--candidate",str(b),
                        "--out-dir",str(tmp_path/"paired"),"--bootstrap","1000"]) == 0
    assert (tmp_path/"paired"/"paired_per_patient_delta.csv").is_file()
    assert json.loads((tmp_path/"paired"/"paired_summary.json").read_text())["n_patients"]==20


def test_manifest_freeze_exact_fivefold_contract(tmp_path,monkeypatch):
    from pulse_repro import experiment_protocol as proto
    class FakePulse:
        @staticmethod
        def list_acdc(root):
            return [{"pid":f"patient{i:03d}","phase":phase,"group":f"G{i%5}"}
                    for i in range(1,101) for phase in ("ED","ES")]
        @staticmethod
        def stratified_kfold_pids(recs,nfolds,fold,seed):
            assert nfolds==5 and seed==42
            ids={r["pid"] for r in recs}
            selected={f"patient{i:03d}" for i in range(1,101)
                      if fold*20 < i <= (fold+1)*20}
            return ids-selected,selected
    script=tmp_path/"PULSE"/"pulse"/"pulse_seg.py"
    script.parent.mkdir(parents=True)
    script.write_text("upstream test fixture")
    import sys
    import types
    monkeypatch.setitem(sys.modules,"pulse_repro.runtime",types.SimpleNamespace(
        upstream=lambda root:(None,FakePulse),
        file_sha256=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest=proto.build_split_manifest(tmp_path/"ACDC",tmp_path/"PULSE")
    assert manifest["seed"]==42 and len(manifest["folds"])==5
    assert all(len(x["val_patients"])==20 for x in manifest["folds"].values())
    folder=make_report(tmp_path/"eval",0)
    assert load_fold(folder,split_manifest=manifest).fold==0


def test_refuses_duplicate_phase_when_status_claims_valid(tmp_path):
    folder=make_report(tmp_path/"f",0)
    df=pd.read_csv(folder/"per_volume.csv")
    df.loc[0,"Myo_status"]="one_empty"
    # A surface value may not remain finite if metric is undefined.
    df.to_csv(folder/"per_volume.csv",index=False)
    with pytest.raises(ProtocolError,match="Undefined Myo surface distances"):
        load_fold(folder)


def test_refuses_different_model_arms_in_same_aggregate(tmp_path):
    a=load_fold(make_report(tmp_path/"f",0))
    b=load_fold(make_report(tmp_path/"g",1,kind="selective-v1"))
    with pytest.raises(ProtocolError,match="Do not mix method variants"):
        pooled_summary([a,b])
