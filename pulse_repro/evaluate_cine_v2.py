"""ED/ES out-of-fold full-volume 4-flip TTA for genuine cardiac-time V2."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cine_data import build_temporal_input, find_cine, neighboring_indices, raw_root, verify_patient_cine
from .cine_v2 import load_cine_v2
from .metrics import measure_binary
from .runtime import read_volume, file_sha256, upstream


def infer_volume(model, volume, cine, frame, ed_frame, pulse_root, device,
                 tta: bool=True, temporal_control: str="correct",
                 return_diagnostics: bool=False):
    if temporal_control not in ("correct","swap","repeat-target"):
        raise ValueError("temporal_control must be correct/swap/repeat-target")
    """Match official PULSE spatial input, flip-TTA, resizing and postprocess."""
    infer, _ = upstream(pulse_root)
    h,w,s=volume.shape
    out=np.zeros((h,w,s),dtype=np.uint8)
    diagnostics=[]
    model.eval()
    flip_pairs=((False,False),(True,False),(False,True),(True,True)) if tta else ((False,False),)
    with torch.inference_mode():
        for z in range(s):
            spatial,temporal,phase=build_temporal_input(volume,cine,z,frame,ed_frame,size=256)
            if temporal_control == "swap":
                temporal=temporal[::-1].copy()  # keep semantic direction codes unchanged
            elif temporal_control == "repeat-target":
                temporal=np.stack((spatial[1],spatial[1]),axis=0)
            x=torch.from_numpy(spatial[None]).to(device)
            temporal_t=torch.from_numpy(temporal[None]).to(device)
            phase_t=torch.from_numpy(phase[None]).to(device)
            probabilities=[]
            for flip_h,flip_v in flip_pairs:
                xi=x.flip(-1) if flip_h else x
                ti=temporal_t.flip(-1) if flip_h else temporal_t
                if flip_v:xi=xi.flip(-2);ti=ti.flip(-2)
                with torch.amp.autocast(device.type,enabled=device.type=="cuda"):
                    if return_diagnostics and not flip_h and not flip_v:
                        logits,_,_,aux = model(xi,ti,phase_t,return_aux=True)
                        if aux is not None:
                            weights=aux["weights"].float()
                            diagnostics.append([
                                *weights.mean((0,2,3)).tolist(),
                                float(aux["valid_minus"].float().mean()),
                                float(aux["valid_plus"].float().mean()),
                            ])
                    else:
                        logits,*_=model(xi,ti,phase_t)
                if flip_h:logits=logits.flip(-1)
                if flip_v:logits=logits.flip(-2)
                probabilities.append(F.softmax(logits.float(),dim=1))
            prediction=torch.stack(probabilities).mean(0).argmax(1)[0].cpu().numpy().astype(np.uint8)
            out[...,z]=cv2.resize(prediction,(w,h),interpolation=cv2.INTER_NEAREST)
    mask=infer.postprocess(out)
    if not return_diagnostics:
        return mask
    summary={}
    if diagnostics:
        mean=np.mean(diagnostics,axis=0)
        summary=dict(zip(("skip_weight","past_weight","future_weight",
                          "past_valid_fraction","future_valid_fraction"),
                         [float(x) for x in mean]))
    return mask,summary


def evaluate(acdc_root: Path,pulse_root: Path,weights: Path,out_dir: Path,
             fold: int=0,tta: bool=True,max_patients: int|None=None,
             temporal_control: str="correct"):
    if temporal_control not in ("correct","swap","repeat-target"):
        raise ValueError("Invalid temporal control")
    if fold not in range(5):raise ValueError("fold outside 0..4")
    _,pulse=upstream(pulse_root)
    import nibabel as nib
    rows=pulse.list_acdc(acdc_root/"training")
    if len(rows)!=200 or len({r["pid"] for r in rows})!=100:
        raise RuntimeError("Expected exactly 100 ACDC training patients / 200 ED+ES records")
    train,val=pulse.stratified_kfold_pids(rows,5,fold,seed=42)
    if len(val)!=20 or train & val:raise ValueError("Patient split differs from PULSE")
    ck=torch.load(str(weights),map_location="cpu",weights_only=True)
    if ck.get("variant")!="cine-reliability-v2" or int(ck.get("fold",-1))!=fold:
        raise ValueError("V2 model variant/fold metadata does not match evaluation")
    original=raw_root(acdc_root)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=load_cine_v2(pulse_root,weights,device)
    ids=sorted(val)
    if max_patients is not None:
        if max_patients<1:raise ValueError("max-patients must be >=1")
        ids=ids[:max_patients]
    out_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    selected=sorted((r for r in rows if r["pid"] in ids),key=lambda x:(x["pid"],x["phase"]))
    for i,r in enumerate(selected,1):
        root=original/"training"/r["pid"]
        verify_patient_cine(root,acdc_root)
        info=pulse.parse_info_cfg(root/"Info.cfg")
        cine_nii=nib.load(str(find_cine(root)))
        if len(cine_nii.shape)!=4:raise ValueError("Expected 4D cine")
        frame=int(info[r["phase"]]);neighboring_indices(frame,cine_nii.shape[-1])
        volume,img=read_volume(Path(r["img"]))
        # verify_patient_cine() already validates indexed voxel alignment
        # when a 4D NIfTI has no declared qform/sform geometry.
        if cine_nii.shape[:3] != volume.shape:
            raise ValueError(f"Cine/phase shape differs: {r['pid']}")
        cine=np.asarray(cine_nii.dataobj,dtype=np.float32)
        gt_nii=nib.load(r["gt"])
        if gt_nii.shape!=volume.shape or not np.allclose(img.affine,gt_nii.affine,atol=1e-3):
            raise ValueError(f"Image/GT misalignment: {r['pid']}")
        label=np.asarray(gt_nii.dataobj,dtype=np.uint8)
        spacing=tuple(float(v) for v in nib.affines.voxel_sizes(img.affine))
        if device.type=="cuda":torch.cuda.synchronize()
        start=time.perf_counter()
        segmentation,trust_stats=infer_volume(model,volume,cine,frame,int(info["ED"]),pulse_root,device,tta=tta,
                                  temporal_control=temporal_control,return_diagnostics=True)
        if device.type=="cuda":torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        row={"patient":r["pid"],"phase":r["phase"],"seconds":elapsed,
             **trust_stats}
        for c,name in ((1,"RV"),(2,"Myo"),(3,"LV")):
            m=measure_binary(label==c,segmentation==c,spacing)
            for metric,value in m.items():row[f"{name}_{metric}"]=value
        print(f"[{i}/{len(selected)}] {r['pid']} {r['phase']} " +
              " ".join(f"{name}={row[name+'_dice']:.4f}" for name in ("RV","Myo","LV")),flush=True)
        results.append(row)
    df=pd.DataFrame(results)
    df.to_csv(out_dir/"per_volume.csv",index=False)
    columns=[c for c in df if c.endswith(("_dice","_hd95_mm","_assd_mm"))]
    patients=df.groupby("patient",sort=True)[columns+["seconds"]].mean()
    patients.to_csv(out_dir/"per_patient.csv")
    mean_trust={k:float(df[k].mean()) for k in (
        "skip_weight","past_weight","future_weight",
        "past_valid_fraction","future_valid_fraction") if k in df}
    summary={"variant":"cine-reliability-v2","fold":fold,"seed":42,
             "mean_reliability_diagnostics":mean_trust,
             "temporal_control":temporal_control,
             "use_structure":ck.get("use_structure"),
             "use_temporal":ck.get("use_temporal",True),
             "tta":tta,"validation_patients":ids,"n_patients":len(ids),
             "n_volumes":len(df),"full_fold":len(ids)==20,
             "warning":"Patient-level fold validation, not untouched independent test",
             "checkpoint":str(weights),
             "checkpoint_sha256":file_sha256(weights),
             "metric_protocol":"full 3D volumes patient-first mean; symmetric physical-space HD95/ASSD",
             "per_patient_means":{c:(float(patients[c].mean())
                if np.isfinite(patients[c].mean()) else None) for c in columns},
             "undefined_surface_case_counts":{name:int((df[f"{name}_status"]!="valid").sum())
                for name in ("RV","Myo","LV")}}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("RESULT",summary["per_patient_means"])
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--acdc-root",type=Path,required=True)
    p.add_argument("--pulse-root",type=Path,default=Path("/kaggle/working/PULSE"))
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--fold",type=int,default=0)
    p.add_argument("--tta",action="store_true")
    p.add_argument("--max-patients",type=int,default=None)
    p.add_argument("--temporal-control",choices=("correct","swap","repeat-target"),
                   default="correct",help="Counterfactual ablation, not main result")
    args=p.parse_args();evaluate(**vars(args))

if __name__=="__main__":main()
