"""Patient-disjoint PULSE Cine V2 120-epoch 2-stage segmentation experiment.

Training starts from facebook/dinov2-base, not a fitted fold checkpoint. The
PULSE supervised main/aux losses, optimizer, LR curriculum and augmented ED/ES
split are retained. Cardiac-time inputs are genuinely time-neighboring 4D MRI
slices. Consistency probes predict the *target frame* only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cine_data import ACDCCineDataset, inspect_cine, raw_root
from .cine_v2 import PULSECineReliabilityV2, reliable_agreement_loss
from .runtime import upstream


def run(acdc_root: Path, pulse_root: Path, out_dir: Path,
        fold: int=0, epochs: int=120, batch_size: int=4,
        lr: float=3e-4, seed: int=42, use_structure: bool=True,
        use_consistency: bool=True, consistency_weight: float=0.05,
        probe_weight: float=0.1, num_workers: int=2,
        max_steps: int | None=None):
    if fold not in range(5) or epochs<1 or batch_size<1 or lr<=0:
        raise ValueError("Invalid fold, epoch, batch size or learning rate")
    if num_workers<0 or consistency_weight<0 or probe_weight<0:
        raise ValueError("Invalid worker/loss configuration")
    if max_steps is not None and max_steps<1:
        raise ValueError("max_steps must be positive")
    if not (acdc_root/"training").is_dir():
        raise FileNotFoundError("Normalized ACDC directory is missing")
    preflight=inspect_cine(acdc_root)
    if preflight["total_patients"] != 100 or preflight["cine_missing"]:
        raise RuntimeError(
            f"Cine preflight: {preflight['cine_available']}/100 found. "
            "Download a complete 4D ACDC cine dataset before V2 training; "
            "ED/ES-only data are NOT sufficient. Run `python -m pulse_repro.cine_data`."
        )
    original=raw_root(acdc_root)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Existing V2 run directory cannot be overwritten: {out_dir}")
    out_dir.mkdir(parents=True,exist_ok=False if not out_dir.exists() else True)

    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    _,pulse=upstream(pulse_root)
    recs=pulse.list_acdc(acdc_root/"training")
    if len(recs)!=200 or len({r["pid"] for r in recs})!=100:
        raise RuntimeError("Expected 100 original ACDC patients with 200 ED/ES ground truths")
    train_ids,val_ids=pulse.stratified_kfold_pids(recs,5,fold,seed=42)
    if len(val_ids)!=20 or train_ids & val_ids:
        raise RuntimeError("Invalid patient split")
    trainset=ACDCCineDataset([r for r in recs if r["pid"] in train_ids],
                             original,pulse,train=True)
    valset=ACDCCineDataset([r for r in recs if r["pid"] in val_ids],
                           original,pulse,train=False)
    trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,
                            pin_memory=True,num_workers=num_workers,drop_last=True)
    valloader=DataLoader(valset,batch_size=batch_size,shuffle=False,
                         pin_memory=True,num_workers=num_workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PULSECineReliabilityV2(pulse.PULSESeg(256),
                                 use_structure=use_structure).to(device)
    blocks=model.backbone_blocks
    frozen_count=pulse.FREEZE_BLK
    bb_params=[p for blk in blocks[frozen_count:] for p in blk.parameters() if p.requires_grad]
    backbone_ids={id(p) for p in bb_params}
    others=[p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([
        {"params":bb_params,"lr":lr*.1},
        {"params":others,"lr":lr}],weight_decay=pulse.WD)
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=="cuda")
    dice=pulse.WeightedDice(pulse.DICE_CLASS_W.to(device)).to(device)
    cross_entropy=nn.CrossEntropyLoss(weight=pulse.CE_CLASS_W.to(device))
    meta={"variant":"cine-reliability-v2","fold":fold,"epochs":epochs,
          "batch_size":batch_size,"seed":seed,"lr":lr,
          "use_structure":use_structure,"use_consistency":use_consistency,
          "consistency_weight":consistency_weight,"probe_weight":probe_weight,
          "max_steps":max_steps,"train_patients":sorted(train_ids),
          "val_patients":sorted(val_ids),"cine_root":str(original),
          "initialization":"PULSE DINOv2 backbone; decoder and temporal branch randomly initialised",
          "supervision":"only original ACDC ED/ES ground-truth masks",
          "selection_metric":"original PULSE average per-slice Dice (TTA after epoch 70)"}
    (out_dir/"run_config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("TRAIN",len(trainset),"slices VAL",len(valset),"device",device,
          "parameters",sum(p.numel() for p in model.parameters()),flush=True)
    best=-1;history=[];unfroze=False;clock=time.perf_counter()
    try:
        for epoch in range(1,epochs+1):
            if epoch==pulse.STAGE1+1 and not unfroze:
                new=[]
                for block in blocks[:frozen_count]:
                    for p in block.parameters():
                        p.requires_grad_(True);new.append(p)
                optimizer.add_param_group({"params":new,"lr":pulse.STAGE2_LR*.05})
                unfroze=True
            pulse.set_lr(optimizer,epoch-1,lr,pulse.STAGE2_LR)
            model.train();running=[];tic=time.perf_counter()
            for i,b in enumerate(tqdm(trainloader,desc=f"CineV2 ep{epoch}",leave=False)):
                if max_steps is not None and i>=max_steps:break
                images=b["img"].to(device,non_blocking=True)
                times=b["temporal"].to(device,non_blocking=True)
                labels=b["mask"].to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type,enabled=device.type=="cuda"):
                    main,a1,a2,aux=model(images,times,return_aux=True)
                    y1=F.interpolate(labels[:,None].float(),size=a1.shape[-2:],
                                     mode="nearest")[:,0].long()
                    y2=F.interpolate(labels[:,None].float(),size=a2.shape[-2:],
                                     mode="nearest")[:,0].long()
                    loss=(pulse.DS_WEIGHTS[0]*pulse.seg_loss(main,labels,dice,cross_entropy)
                          +pulse.DS_WEIGHTS[1]*pulse.seg_loss(a2,y2,dice,cross_entropy)
                          +pulse.DS_WEIGHTS[2]*pulse.seg_loss(a1,y1,dice,cross_entropy))
                    if use_consistency:
                        loss=loss+consistency_weight*reliable_agreement_loss(aux,main)
                        # Labeled ED/ES target probes anchored to GT, so probe
                        # agreement alone cannot minimise by trivial uniform maps.
                        low_gt=F.interpolate(labels[:,None].float(),size=aux["probe_minus"].shape[-2:],
                                             mode="nearest")[:,0].long()
                        loss=loss + probe_weight*0.5*(
                            F.cross_entropy(aux["probe_minus"],low_gt) +
                            F.cross_entropy(aux["probe_plus"],low_gt))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();running.append(float(loss.detach()))
            if not running:raise RuntimeError("Zero training steps")
            model.eval();dices=[]
            with torch.no_grad():
                for batch in valloader:
                    images=batch["img"].to(device)
                    times=batch["temporal"].to(device)
                    gt=batch["mask"].to(device)
                    if epoch>pulse.STAGE1:
                        probes=[]
                        for flip_h,flip_v in ((False,False),(True,False),(False,True),(True,True)):
                            xi=images.flip(-1) if flip_h else images
                            ti=times.flip(-1) if flip_h else times
                            if flip_v:xi=xi.flip(-2);ti=ti.flip(-2)
                            with torch.amp.autocast(device.type,enabled=device.type=="cuda"):
                                logits,*_=model(xi,ti)
                            if flip_h:logits=logits.flip(-1)
                            if flip_v:logits=logits.flip(-2)
                            probes.append(logits.float().softmax(1))
                        prob=torch.stack(probes).mean(0)
                    else:
                        with torch.amp.autocast(device.type,enabled=device.type=="cuda"):
                            logits,*_=model(images,times)
                        prob=logits.float().softmax(1)
                    dices.append(pulse.dice_per_class(prob,gt))
            mean_classes=np.asarray(dices).mean(0)
            val=float(mean_classes.mean())
            record={"epoch":epoch,"loss":float(np.mean(running)),
                    "val_RV":float(mean_classes[0]),"val_Myo":float(mean_classes[1]),
                    "val_LV":float(mean_classes[2]),"val_mean":val,
                    "elapsed_sec":round(time.perf_counter()-tic,2)}
            history.append(record)
            (out_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            if val>best:
                best=val
                checkpoint={"variant":"cine-reliability-v2","epoch":epoch,
                            "fold":fold,"best_dice":val,"model":model.state_dict(),
                            "use_structure":use_structure,"use_consistency":use_consistency}
                tmp=out_dir/"best_model.part"
                torch.save(checkpoint,tmp)
                os.replace(tmp,out_dir/"best_model.pth")
            print(f"Ep{epoch:03d} loss={record['loss']:.4f} | RV={mean_classes[0]:.4f} "
                  f"Myo={mean_classes[1]:.4f} LV={mean_classes[2]:.4f} "
                  f"mean={val:.4f} best={best:.4f}",flush=True)
        (out_dir/"run_status.json").write_text(json.dumps({
            "exit_code":0,"epochs_completed":len(history),
            "elapsed_seconds":time.perf_counter()-clock,"best_dice":best,
        },indent=2),encoding="utf-8")
    except BaseException as e:
        (out_dir/"run_status.json").write_text(json.dumps({
            "exit_code":1,"epochs_completed":len(history),"error":repr(e),
        },indent=2),encoding="utf-8")
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acdc-root",type=Path,required=True)
    p.add_argument("--pulse-root",type=Path,default=Path("/kaggle/working/PULSE"))
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--fold",type=int,default=0)
    p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-structure",action="store_false",dest="use_structure")
    p.add_argument("--no-consistency",action="store_false",dest="use_consistency")
    p.add_argument("--consistency-weight",type=float,default=.05)
    p.add_argument("--probe-weight",type=float,default=.1)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--max-steps",type=int,default=None,
                   help="SMOKE TEST ONLY, not a full training result")
    a=p.parse_args();run(**vars(a))

if __name__=="__main__":main()
