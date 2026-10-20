"""Fine-tune only new V1 skip-fusion adapters from a frozen PULSE fold.

ACDC ED and ES use exactly the original PULSE 2.5D spatial-slice dataloader.
No temporal claim, no pseudo-label generation and no patient overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .runtime import file_sha256, upstream
from .selective_v1 import PULSESelectiveV1


def run(acdc_root: Path, pulse_root: Path, init_weights: Path,
        out_dir: Path, fold: int = 0, epochs: int = 15,
        batch_size: int = 4, lr: float = 5e-4,
        steps_per_epoch: int | None = None, seed: int = 42):
    if fold not in range(5) or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid training arguments")
    if steps_per_epoch is not None and steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError("Missing normalized ACDC database/training")
    if not init_weights.is_file():
        raise FileNotFoundError(f"Missing PULSE fold checkpoint: {init_weights}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f).exists() for f in ("best_model.pth", "history.csv", "run_config.json")):
        raise FileExistsError(f"Output directory already contains a run: {out_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _, source = upstream(pulse_root)
    records = source.list_acdc(acdc_root / "training")
    if len(records) != 200 or len({r["pid"] for r in records}) != 100:
        raise ValueError("Expected official ACDC training: 100 patients / 200 ED+ES records")

    train_ids, val_ids = source.stratified_kfold_pids(records, 5, fold, seed=42)
    if len(val_ids) != 20 or train_ids & val_ids:
        raise ValueError("Invalid fold split")
    ck = torch.load(str(init_weights), map_location="cpu", weights_only=True)
    if int(ck.get("fold", -1)) != fold:
        raise ValueError(f"Initialization checkpoint belongs to fold {ck.get('fold')}, not {fold}")

    train_data = source.ACDCDataset(
        [r for r in records if r["pid"] in train_ids], img_size=256, train=True
    )
    val_data = source.ACDCDataset(
        [r for r in records if r["pid"] in val_ids], img_size=256, train=False
    )
    tr_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=False)
    va_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = source.PULSESeg(img_size=256)
    base.load_state_dict(ck["model"], strict=True)
    model = PULSESelectiveV1(base).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and len(trainable) < len(list(model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dice_fn = source.WeightedDice(source.DICE_CLASS_W.to(device)).to(device)
    ce_fn = torch.nn.CrossEntropyLoss(weight=source.CE_CLASS_W.to(device))

    meta = {
        "variant": "selective-v1",
        "fold": fold, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "lr": lr,
        "steps_per_epoch": steps_per_epoch,
        "source_checkpoint": str(init_weights),
        "source_checkpoint_sha256": file_sha256(init_weights),
        "pulse_root": str(pulse_root), "acdc_root": str(acdc_root),
        "train_patients": sorted(train_ids), "val_patients": sorted(val_ids),
        "note": "frozen official PULSE; train only zero-initialized enhancement adapters",
        "val_protocol": "original PULSE sampled-slice Dice, no TTA for selection",
        "independent_full_volume_eval": "run pulse_repro.evaluate --model-variant selective-v1"
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Device:", device, "train slices:", len(train_data),
          "val slices:", len(val_data), "trainable params:",
          sum(p.numel() for p in trainable), flush=True)

    columns = ["epoch", "loss", "val_RV", "val_Myo", "val_LV",
               "val_mean", "lr", "seconds", "steps"]
    best = -1.0
    start_all = time.perf_counter()

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            start = time.perf_counter()
            for index, batch in enumerate(tqdm(tr_loader, desc=f"V1 ep{epoch}", leave=False)):
                if steps_per_epoch is not None and index >= steps_per_epoch:
                    break
                image = batch["img"].to(device, non_blocking=True)
                labels = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits, _, _ = model(image)
                    # The baseline auxiliary heads are frozen. Their losses do
                    # not train V1 adapters, so train only the main prediction.
                    loss = source.seg_loss(logits, labels, dice_fn, ce_fn)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            if not losses:
                raise RuntimeError("No training steps performed")
            model.eval()
            with torch.no_grad():
                score = source.eval_epoch(model, va_loader, device, tta=False)
            val = float(np.mean(score))
            row = {
                "epoch": epoch, "loss": float(np.mean(losses)),
                "val_RV": float(score[0]), "val_Myo": float(score[1]),
                "val_LV": float(score[2]), "val_mean": val,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.perf_counter() - start, 2),
                "steps": len(losses)
            }
            writer.writerow(row)
            log.flush()
            if val > best:
                best = val
                torch.save({
                    "variant": "selective-v1",
                    "epoch": epoch,
                    "fold": fold, "seed": seed,
                    "best_slice_val_dice": val,
                    "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
                    "model": model.state_dict(),
                }, out_dir / "best_model.pth")
            scheduler.step()
            print(
                f"V1 Ep{epoch:03d} loss={row['loss']:.4f} "
                f"RV={score[0]:.4f} Myo={score[1]:.4f} "
                f"LV={score[2]:.4f} mean={val:.4f} "
                f"best={best:.4f} time={row['seconds']:.1f}s",
                flush=True
            )
    (out_dir / "run_status.json").write_text(json.dumps({
        "exit_code": 0, "elapsed_seconds": time.perf_counter() - start_all,
        "best_slice_val_dice": best, "epochs_completed": epochs,
    }, indent=2), encoding="utf-8")
    print("Finished:", out_dir, "best slice Dice:", best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--pulse-root", type=Path,
                        default=Path("/kaggle/working/PULSE"))
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
