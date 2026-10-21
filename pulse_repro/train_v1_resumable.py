"""AutoDL full-state-resumable PULSE + selective V1, from original DINOv2 initialization.

The segmentation model, loss, augmentation, backbone schedule, and validation
are imported from the pinned upstream PULSE implementation. Checkpoints add
optimizer, GradScaler and RNG states; no upstream file is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .joint_v1 import PULSESelectiveJointV1

VARIANT = "selective-v1"
STAGE1 = 70


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_torch_save(obj: dict, destination: Path) -> None:
    temp = destination.with_name(destination.name + ".part")
    torch.save(obj, temp)
    os.replace(temp, destination)


def _atomic_json(obj, destination: Path) -> None:
    temp = destination.with_name(destination.name + ".part")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, destination)


def _rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": np_state[0], "keys": np_state[1].tolist(),
                  "position": int(np_state[2]), "has_gauss": int(np_state[3]),
                  "cached_gaussian": float(np_state[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(record: dict) -> None:
    random.setstate(record["python"])
    s = record["numpy"]
    np.random.set_state((s["name"], np.asarray(s["keys"], dtype=np.uint32),
                         int(s["position"]), int(s["has_gauss"]),
                         float(s["cached_gaussian"])))
    torch.set_rng_state(record["torch_cpu"].cpu())
    if torch.cuda.is_available():
        states = record["torch_cuda"]
        if len(states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from saved RNG state")
        torch.cuda.set_rng_state_all(states)


def _optimizer(model: PULSESelectiveJointV1, pulse, lr: float):
    blocks = model.backbone_blocks
    if len(blocks) <= pulse.FREEZE_BLK:
        raise ValueError("PULSE backbone block count is unexpectedly small")
    group0 = [p for block in blocks[pulse.FREEZE_BLK:]
              for p in block.parameters() if p.requires_grad]
    ids = {id(p) for p in group0}
    group1 = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    if not group0 or not group1:
        raise ValueError("Empty initial PULSE optimizer parameter group")
    return torch.optim.AdamW([{"params": group0, "lr": lr * 0.1},
                              {"params": group1, "lr": lr}], weight_decay=pulse.WD)


def _unfreeze_stage2(model, optimizer, pulse) -> None:
    if len(optimizer.param_groups) != 2:
        raise ValueError("Cannot enter stage2: optimizer groups not 2")
    early = []
    for blk in model.backbone_blocks[:pulse.FREEZE_BLK]:
        for p in blk.parameters():
            p.requires_grad_(True)
            early.append(p)
    optimizer.add_param_group({"params": early, "lr": pulse.STAGE2_LR * 0.05})


def _run_metadata(fold: int, epochs: int, batch_size: int, lr: float,
                  seed: int, source_hash: str) -> dict:
    return {"variant": VARIANT, "fold": fold, "nfolds": 5,
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
            "upstream_source_sha256": source_hash,
            "optimizer": "AdamW", "stage1_epochs": STAGE1,
            "baseline_protocol": "PULSE 2.5D, 70+50 curriculum, deep-supervision, sampled-slice selection",
            "validation_type": "internal fold; no external test",
            "initialization": "facebook/dinov2-base and randomly initialized PULSE decoder/V1 module"}


def _check_checkpoint(ck: dict, wanted: dict, existing_best: Path) -> int:
    if ck.get("variant") != VARIANT:
        raise ValueError("Not a selective-v1 full-state checkpoint")
    if ck.get("checkpoint_kind") != "full-epoch-state-v1":
        raise ValueError("Checkpoint lacks complete optimizer/scaler/RNG state")
    for key in ("fold", "batch_size", "lr", "seed", "upstream_source_sha256",
                "selective_v1_sha256", "joint_v1_sha256", "acdc_root", "pulse_root"):
        if key not in wanted:
            continue
        if ck.get("run_config", {}).get(key) != wanted[key]:
            raise ValueError(f"Resume configuration mismatch: {key}")
    epoch = ck.get("epoch")
    if not isinstance(epoch, int) or not 1 <= epoch < wanted["epochs"]:
        raise ValueError("Saved epoch is invalid for requested final epoch")
    for key in ("model", "optimizer", "scaler", "rng", "history", "best_dice"):
        if key not in ck:
            raise ValueError(f"Missing resume state field: {key}")
    if not existing_best.is_file():
        raise FileNotFoundError("Missing best_model.pth; refusing incomplete run directory")
    if len(ck["history"]) != epoch:
        raise ValueError("Checkpoint history and last epoch disagree")
    return epoch


def run(acdc_root: Path, pulse_root: Path, out_dir: Path, fold: int = 0,
        epochs: int = 120, batch_size: int = 4, lr: float = 3e-4,
        seed: int = 42, workers: int = 4, save_every: int = 2,
        resume: bool = False, dry_run: bool = False) -> None:
    if fold not in range(5) or epochs < 2 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid fold, epochs, batch size or learning rate")
    if workers < 0 or save_every < 1:
        raise ValueError("workers must be >=0, save_every >=1")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError(f"ACDC database/training not found at {acdc_root}")
    script = pulse_root / "pulse" / "pulse_seg.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pinned PULSE script missing: {script}")
    source_hash = _sha256(script)
    config = _run_metadata(fold, epochs, batch_size, lr, seed, source_hash)
    v1_source = Path(__file__).resolve().parent
    config["selective_v1_sha256"] = _sha256(v1_source / "selective_v1.py")
    config["joint_v1_sha256"] = _sha256(v1_source / "joint_v1.py")
    config["acdc_root"] = str(acdc_root.resolve())
    config["pulse_root"] = str(pulse_root.resolve())
    cfg_path, state_path, best_path = [out_dir / x for x in
        ("run_config.json", "last_state.pth", "best_model.pth")]

    if resume:
        if not cfg_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("--resume needs run_config.json and last_state.pth")
        stored = json.loads(cfg_path.read_text(encoding="utf-8"))
        if stored != config:
            raise ValueError("The original run configuration differs; do not resume")
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Nonempty output directory must not be overwritten: {out_dir}")

    # These checks are cheap and intentionally occur before HuggingFace downloads.
    if dry_run:
        if resume:
            ck = torch.load(state_path, map_location="cpu", weights_only=True)
            last = _check_checkpoint(ck, config, best_path)
            print(f"DRY RUN: FULL-state resume epoch {last+1}..{epochs} ; fold {fold}")
        else:
            print(f"DRY RUN: FRESH training epoch 1..{epochs} ; fold {fold}")
        print(json.dumps({**config, "acdc_root": str(acdc_root),
                          "pulse_root": str(pulse_root), "out_dir": str(out_dir),
                          "save_every": save_every}, ensure_ascii=False, indent=2))
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from .runtime import upstream
    # Prevent silent upstream drift when reproducing the pinned PULSE protocol.
    expected_commit = "781e0712ea3b2dd246f86346d73008ecdb73434f"
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(pulse_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("PULSE checkout must be a Git repository at the pinned commit") from exc
    if actual_commit != expected_commit:
        raise RuntimeError(f"PULSE checkout commit {actual_commit} != expected {expected_commit}")
    _, pulse = upstream(pulse_root)
    if (pulse.STAGE1, pulse.FREEZE_BLK, pulse.IMG_SIZE) != (70, 2, 256):
        raise RuntimeError("Pinned PULSE stage/block/image protocol changed")
    recs = pulse.list_acdc(acdc_root / "training")
    if len(recs) != 200 or len({r["pid"] for r in recs}) != 100:
        raise ValueError("Expected 100 ACDC training patients with 200 ED/ES records")
    train_ids, val_ids = pulse.stratified_kfold_pids(recs, 5, fold, seed=42)
    if len(val_ids) != 20 or train_ids & val_ids:
        raise ValueError("Unexpected fivefold patient split")
    tr = [r for r in recs if r["pid"] in train_ids]
    va = [r for r in recs if r["pid"] in val_ids]
    tr_ds = pulse.ACDCDataset(tr, pulse.IMG_SIZE, train=True)
    va_ds = pulse.ACDCDataset(va, pulse.IMG_SIZE, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                       pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PULSESelectiveJointV1(pulse.PULSESeg(pulse.IMG_SIZE)).to(device)
    optimizer = _optimizer(model, pulse, lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    dice = pulse.WeightedDice(pulse.DICE_CLASS_W.to(device)).to(device)
    ce = nn.CrossEntropyLoss(weight=pulse.CE_CLASS_W.to(device))

    # Write a fresh run directory only after the upstream version, ACDC split,
    # and initial model have all passed preflight, minimizing unusable dirs.
    if not resume:
        out_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(config, cfg_path)

    best, history, unfroze, start = 0.0, [], False, 1
    if resume:
        ck = torch.load(state_path, map_location="cpu", weights_only=True)
        last = _check_checkpoint(ck, config, best_path)
        # Recreate the original stage-2 three-group parameter topology *before*
        # optimizer.load_state_dict, when resumed from epoch >= 71.
        if last > STAGE1:
            _unfreeze_stage2(model, optimizer, pulse)
            unfroze = True
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        best = float(ck["best_dice"])
        # The best checkpoint may have been updated in an epoch after the last
        # periodic full-state snapshot. Preserve that previously achieved best
        # even though optimizer/model must roll back to the snapshot epoch.
        best_ck = torch.load(best_path, map_location="cpu", weights_only=True)
        if best_ck.get("variant") != VARIANT or best_ck.get("fold") != fold:
            raise ValueError("Existing best_model.pth variant or fold mismatch")
        if int(best_ck.get("epoch", -1)) > last:
            print(f"[RECOVERY] Preserving best validation checkpoint from epoch "
                  f"{best_ck['epoch']} after the last full-state epoch {last}", flush=True)
        best = max(best, float(best_ck["best_dice"]))
        history = list(ck["history"])
        _restore_rng(ck["rng"])
        start = last + 1
        print(f"[RESUME] exact available full state: epoch {last} -> {start}; "
              f"optimizer/scaler/RNG restored; best={best:.6f}", flush=True)
    else:
        print("[FRESH] V1 epoch 1..120; DINOv2 initialized without segmentation checkpoint", flush=True)
    print(f"[DATA] train={len(train_ids)} val={len(val_ids)} patients; "
          f"training slices={len(tr_ds)} validation slices={len(va_ds)}; device={device}", flush=True)

    for epoch in range(start, epochs + 1):
        if epoch == STAGE1 + 1 and not unfroze:
            _unfreeze_stage2(model, optimizer, pulse)
            unfroze = True
            print("[STAGE2] epoch 71: early DINOv2 blocks unfrozen", flush=True)
        pulse.set_lr(optimizer, epoch - 1, lr, pulse.STAGE2_LR)
        model.train()
        running = 0.0
        started = time.perf_counter()
        for batch in tqdm(tr_dl, desc=f"V1 epoch {epoch}", leave=False):
            x, y = batch["img"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                main, a1, a2 = model(x)
                y1 = torch.nn.functional.interpolate(y.unsqueeze(1).float(),
                    a1.shape[-2:], mode="nearest").squeeze(1).long()
                y2 = torch.nn.functional.interpolate(y.unsqueeze(1).float(),
                    a2.shape[-2:], mode="nearest").squeeze(1).long()
                loss = (pulse.DS_WEIGHTS[0] * pulse.seg_loss(main, y, dice, ce)
                        + pulse.DS_WEIGHTS[1] * pulse.seg_loss(a2, y2, dice, ce)
                        + pulse.DS_WEIGHTS[2] * pulse.seg_loss(a1, y1, dice, ce))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        # Identical source validation procedure (TTA for epochs 71..120).
        scores = pulse.eval_epoch(model, va_dl, device, tta=(epoch > STAGE1))
        val = float(scores.mean())
        record = {"epoch": epoch, "loss": running / len(tr_dl),
                  "val_RV": float(scores[0]), "val_Myo": float(scores[1]),
                  "val_LV": float(scores[2]), "val_mean": val,
                  "seconds": round(time.perf_counter() - started, 2)}
        history.append(record)
        if val > best:
            best = val
            _atomic_torch_save({"epoch": epoch, "model": model.state_dict(),
                                "best_dice": best, "fold": fold,
                                "variant": VARIANT, "training_mode": "joint"}, best_path)
        _atomic_json(history, out_dir / "history.json")

        # Capture RNG **after** validation, like uninterrupted next-epoch entry.
        # All tensors/primitive containers remain torch.load(weights_only=True)-safe.
        if epoch % save_every == 0 or epoch == epochs:
            state = {"checkpoint_kind": "full-epoch-state-v1",
                     "variant": VARIANT, "epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "rng": _rng_state(), "history": history,
                     "best_dice": best, "run_config": config}
            _atomic_torch_save(state, state_path)
        print(f"Ep{epoch:03d} loss={record['loss']:.4f} | val RV={scores[0]:.3f} "
              f"Myo={scores[1]:.3f} LV={scores[2]:.3f} mean={val:.4f} "
              f"best={best:.4f}", flush=True)

    _atomic_json({"exit_code": 0, "epochs_completed": epochs,
                  "best_dice": best, "last_checkpoint": str(state_path)},
                 out_dir / "run_status.json")
    print(f"[COMPLETE] epoch={epochs} best={best:.6f}; {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acdc-root", type=Path, required=True)
    p.add_argument("--pulse-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--resume", action="store_true",
                   help="Restore complete last_state.pth from the same output directory")
    p.add_argument("--dry-run", action="store_true")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
