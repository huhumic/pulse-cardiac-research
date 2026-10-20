"""Volumetric segmentation metrics; physical dimensions in mm.

Surface distances are computed on binary voxel boundary sets and their
physical nearest-neighbor distances. Undefined metrics remain NaN and are
explicitly flagged, never silently replaced by a success score.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def measure_binary(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, ...]) -> dict:
    a, b = np.asarray(gt, dtype=bool), np.asarray(pred, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    if len(spacing) != a.ndim or any(x <= 0 for x in spacing):
        raise ValueError("Spacing must match image dimensionality and be positive")

    ga, pb = int(a.sum()), int(b.sum())
    if ga == pb == 0:
        return {"dice": float("nan"), "hd95_mm": float("nan"),
                "assd_mm": float("nan"), "status": "both_empty",
                "gt_voxels": ga, "pred_voxels": pb}
    dice = 2 * np.count_nonzero(a & b) / (ga + pb)
    if ga == 0 or pb == 0:
        return {"dice": float(dice), "hd95_mm": float("nan"),
                "assd_mm": float("nan"), "status": "one_empty",
                "gt_voxels": ga, "pred_voxels": pb}

    conn = generate_binary_structure(a.ndim, 1)
    ba = a & ~binary_erosion(a, structure=conn, border_value=0)
    bb = b & ~binary_erosion(b, structure=conn, border_value=0)
    # Values at surface A from the nearest surface B, and vice versa.
    d_a_to_b = distance_transform_edt(~bb, sampling=spacing)[ba]
    d_b_to_a = distance_transform_edt(~ba, sampling=spacing)[bb]
    d = np.concatenate((d_a_to_b, d_b_to_a))
    return {"dice": float(dice), "hd95_mm": float(np.percentile(d, 95)),
            "assd_mm": float(d.mean()), "status": "valid",
            "gt_voxels": ga, "pred_voxels": pb}
