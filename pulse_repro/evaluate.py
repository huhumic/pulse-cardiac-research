"""Strict out-of-fold single-checkpoint evaluation (20 patients, 40 volumes)."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .metrics import measure_binary
from .runtime import file_sha256, load_one, predict_volume, read_volume, upstream

CLASSES = ((1, "RV"), (2, "Myo"), (3, "LV"))


def evaluate(acdc_root: Path, pulse_root: Path, weights: Path, out_dir: Path,
             fold: int = 0, tta: bool = True, batch_size: int = 8,
             max_patients: int | None = None, save_masks: bool = False,
             model_variant: str = "official") -> dict:
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError("Expected normalized ACDC database/training; run prepare_acdc")
    out_dir.mkdir(parents=True, exist_ok=True)
    _, source = upstream(pulse_root)
    recs = source.list_acdc(acdc_root / "training")
    pids = {r["pid"] for r in recs}
    if len(pids) != 100 or len(recs) != 200:
        raise ValueError(f"Expected 100 patients / 200 ED,ES records, got {len(pids)}/{len(recs)}; check adapter")
    train_ids, val_ids = source.stratified_kfold_pids(recs, 5, fold, seed=42)
    if train_ids & val_ids:
        raise AssertionError("Train/val patient overlap")
    if len(val_ids) != 20:
        raise AssertionError(f"Expected 20 validation patients, got {len(val_ids)}")
    selected = sorted(val_ids)
    if max_patients is not None:
        if max_patients < 1:
            raise ValueError("max-patients must be positive")
        selected = selected[:max_patients]
    sample_recs = sorted((r for r in recs if r["pid"] in selected),
                         key=lambda r: (r["pid"], r["phase"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(weights), map_location="cpu", weights_only=True)
    if "fold" not in ck or int(ck["fold"]) != fold:
        raise ValueError(f"Checkpoint fold metadata {ck.get('fold')} conflicts with --fold {fold}")
    if model_variant == "official":
        if ck.get("variant") == "selective-v1":
            raise ValueError("selective-v1 weights require --model-variant selective-v1")
        model = load_one(pulse_root, weights, device)
    elif model_variant == "selective-v1":
        from .selective_v1 import load_variant
        model = load_variant(pulse_root, weights, device)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    results = []
    for k, r in enumerate(sample_recs, 1):
        img_path, gt_path = Path(r["img"]), Path(r["gt"])
        vol, image = read_volume(img_path)
        label_nii = nib.load(str(gt_path))
        gt = np.asanyarray(label_nii.dataobj).astype(np.uint8)
        if gt.shape != vol.shape or not np.allclose(image.affine, label_nii.affine, atol=1e-3):
            raise ValueError(f"Image/GT not aligned for {r['pid']} {r['phase']}")
        if not np.isin(np.unique(gt), [0, 1, 2, 3]).all():
            raise ValueError(f"Unexpected labels in {gt_path}")
        spacing = nib.affines.voxel_sizes(image.affine)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predict_volume(model, vol, device, pulse_root,
                              tta=tta, slice_batch_size=batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
        row = {"patient": r["pid"], "phase": r["phase"],
               "shape": str(vol.shape), "spacing_mm": str(tuple(np.round(spacing, 4))),
               "seconds": round(t, 4)}
        for cid, name in CLASSES:
            m = measure_binary(gt == cid, pred == cid, tuple(float(x) for x in spacing))
            for metric, v in m.items():
                row[f"{name}_{metric}"] = v
        results.append(row)
        if save_masks:
            folder = out_dir / "predictions" / r["pid"]
            folder.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(pred, image.affine, image.header),
                     str(folder / f"{r['phase']}_pred.nii.gz"))
        print(f"[{k:02d}/{len(sample_recs):02d}] {r['pid']} {r['phase']} "
              + " ".join(f"{name}={row[f'{name}_dice']:.4f}" for _, name in CLASSES)
              + f" {t:.2f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "per_volume.csv", index=False)
    numeric_cols = [x for x in df if x.endswith(("_dice", "_hd95_mm", "_assd_mm"))]
    by_patient = df.groupby("patient", sort=True)[numeric_cols + ["seconds"]].mean()
    by_patient.to_csv(out_dir / "per_patient.csv")
    info = {
        "protocol": "single-fold out-of-fold full-volume ED/ES: 20 patients/40 vols",
        "warning": ("Full-volume metrics differ from upstream sampled-slice validation, "
                    "and are NOT the paper's 5-fold ensemble metrics."),
        "fold": fold, "seed": 42, "model_variant": model_variant,
        "n_valid_patients": len(selected), "n_volumes": len(df),
        "full_fold": max_patients is None or len(selected) == 20,
        "tta": bool(tta), "weights": str(weights), "weights_sha256": file_sha256(weights),
        "acdc_root": str(acdc_root), "device": str(device),
        "val_patients": selected,
        "per_patient_means": {key: (float(by_patient[key].mean())
                                     if np.isfinite(by_patient[key].mean()) else None)
                              for key in numeric_cols},
        "mean_inference_seconds": float(df["seconds"].mean()),
        "undefined_surface_case_counts": {
            name: int((df[f"{name}_status"] != "valid").sum())
            for _, name in CLASSES
        },
    }
    try:
        info["upstream_commit"] = subprocess.check_output(
            ["git", "-C", str(pulse_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        info["upstream_commit"] = "unknown"
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nResults:", out_dir)
    print("Per-patient means:", info["per_patient_means"])
    print("Undefined surface metrics:", info["undefined_surface_case_counts"])
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acdc-root", type=Path, required=True)
    p.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--tta", action="store_true", help="Official 4-flip TTA; use for comparable baseline")
    p.add_argument("--batch-size", type=int, default=8, help="For the explicitly non-TTA mode only")
    p.add_argument("--max-patients", type=int, default=None, help="Smoke test; not a full-fold result")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--model-variant", choices=("official", "selective-v1"), default="official")
    args = p.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
