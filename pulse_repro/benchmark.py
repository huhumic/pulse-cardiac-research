"""CUDA-synchronized PULSE inference benchmark with model loading excluded."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import nibabel as nib
import torch

from .runtime import load_one, predict_volume, read_volume


def run(acdc_root: Path, pulse_root: Path, weights: Path,
        patient: str, phase: str, mode: str, repeats: int,
        out: Path, slice_batch_size: int = 8) -> dict:
    if mode not in {"no-tta", "tta"}:
        raise ValueError("--mode must be tta or no-tta")
    if repeats < 1:
        raise ValueError("--repeats must be >=1")
    # Resolve phase without reading true labels. Same data parsing as upstream.
    import sys
    sys.path.insert(0, str(pulse_root / "pulse"))
    from pulse_seg import parse_info_cfg
    pdir = acdc_root / "training" / patient
    if not pdir.is_dir():
        raise FileNotFoundError(pdir)
    cfg = parse_info_cfg(pdir / "Info.cfg")
    n = int(cfg[phase])
    image_path = pdir / f"{patient}_frame{n:02d}.nii"
    if not image_path.exists():
        image_path = image_path.with_suffix(".nii.gz")
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    volume, image = read_volume(image_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_one(pulse_root, weights, device)
    timings = []
    allocated = []
    reserved = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(repeats + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pred = predict_volume(model, volume, device, pulse_root,
                              tta=(mode == "tta"),
                              slice_batch_size=slice_batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
        phase_name = "warmup" if i == 0 else "steady"
        print(f"{phase_name}: {t:.4f}s, shape={pred.shape}")
        if i > 0:
            timings.append(t)
            allocated.append(torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None)
            reserved.append(torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else None)
    result = {
        "patient": patient, "phase": phase, "image": str(image_path),
        "image_shape": list(volume.shape),
        "voxel_spacing_mm": [float(x) for x in nib.affines.voxel_sizes(image.affine)],
        "device": str(device), "mode": mode, "repeats": repeats,
        "time_seconds": timings, "median_s": statistics.median(timings),
        "mean_s": statistics.mean(timings),
        "peak_allocated_gib": max(v for v in allocated if v is not None) if device.type == "cuda" else None,
        "peak_reserved_gib": max(v for v in reserved if v is not None) if device.type == "cuda" else None,
        "scope": "single checkpoint loaded once; excludes NIfTI reading and checkpoint loading",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", out)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acdc-root", type=Path, required=True)
    p.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--patient", default="patient014")
    p.add_argument("--phase", choices=["ED", "ES"], default="ED")
    p.add_argument("--mode", choices=["tta", "no-tta"], default="tta")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--slice-batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/pulse_results/benchmark.json"))
    a = p.parse_args()
    run(a.acdc_root, a.pulse_root, a.weights, a.patient, a.phase,
        a.mode, a.repeats, a.out, a.slice_batch_size)


if __name__ == "__main__":
    main()
