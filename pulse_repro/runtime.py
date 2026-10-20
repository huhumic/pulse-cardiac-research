"""Import PULSE and run the original 4-TTA mode or a labeled fast mode."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def upstream(pulse_root: Path):
    pulse_dir = pulse_root.resolve() / "pulse"
    if not (pulse_dir / "pulse_seg.py").is_file():
        raise FileNotFoundError(f"Missing upstream PULSE source: {pulse_dir}")
    path = str(pulse_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    import infer
    import pulse_seg
    return infer, pulse_seg


def load_one(pulse_root: Path, weights: Path, device: torch.device):
    infer, _ = upstream(pulse_root)
    models = infer.load_models(str(weights.resolve()), None, 256, device)
    if len(models) != 1:
        raise RuntimeError("Single-fold reproduction requires exactly one checkpoint")
    return models[0]


@torch.inference_mode()
def predict_volume(model, vol: np.ndarray, device: torch.device,
                   pulse_root: Path, tta: bool = True,
                   slice_batch_size: int = 8) -> np.ndarray:
    infer, _ = upstream(pulse_root)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D (H,W,S); got {vol.shape}")
    if tta:
        # EXACT upstream inference path: four flips, postprocess, slice-by-slice.
        return infer.predict([model], vol, device, 256)
    if slice_batch_size < 1:
        raise ValueError("slice_batch_size must be positive")

    # Fast non-TTA variant for iteration/latency ablation (NOT official paper result).
    h, w, s = vol.shape
    images = []
    for z in range(s):
        indices = (max(z - 1, 0), z, min(z + 1, s - 1))
        slices = [np.ascontiguousarray(infer.zscore(vol[..., k])) for k in indices]
        resized = [cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
                   for x in slices]
        images.append(np.stack(resized).astype(np.float32))
    out = np.zeros((h, w, s), dtype=np.uint8)
    for first in range(0, s, slice_batch_size):
        chunk = torch.from_numpy(np.stack(images[first:first + slice_batch_size])).to(device)
        logits = model(chunk)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        cls = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for i, label_slice in enumerate(cls):
            out[..., first + i] = cv2.resize(label_slice, (w, h),
                                               interpolation=cv2.INTER_NEAREST)
    return infer.postprocess(out)


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_volume(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    vol = nii.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Unexpected image shape {vol.shape}: {path}")
    return vol, nii
