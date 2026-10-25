"""Read-only ACDC -> frozen DCR-MI teacher-grid adapter.

Two explicit modes, never selected automatically:
- raw_uvi: reproduces public UVI-Net ACDCHeartDataset spatial transform and
  *per-volume* [0,1] min/max intensity scaling on original NIfTI arrays.
- already_uvi: validates tensors already on the UVI 128x128x32 grid and
  preserves intensities byte-for-byte at float32 precision (no preprocessing).

The user's historical "old/new preprocessed" UVI data version has not yet been
identified. In the official UVI-Net README, the published revised preprocessed
data versions refer to 4D-Lung, not to ACDC. No training is performed here.

CRITICAL: The three model spatial axes follow original nibabel data axes
[native_axis0,native_axis1,native_axis2], not patient anatomical xyz. FR-BME's
"flow_zyx" channels are indices in this tensor grid, not millimetre DVFs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

TEACHER_SHAPE = (128, 128, 32)
Mode = Literal["raw_uvi", "already_uvi"]


@dataclass(frozen=True)
class UVIGrid:
    """Invertible-on-kept-voxels index map between native and UVI tensor grids."""
    native_shape: tuple[int, int, int]
    start0: int
    start1: int
    start2: int
    pad2_before: int
    pad2_after: int

    @classmethod
    def for_native_shape(cls, shape: tuple[int, int, int]) -> "UVIGrid":
        if len(shape) != 3:
            raise ValueError("Expected three spatial native axes")
        h, w, s = (int(v) for v in shape)
        if h < 128 or w < 128 or s < 1:
            raise ValueError(f"Unsupported native shape {shape}: cannot apply original UVI center crop")
        pad_before = max(0, (32 - s) // 2)
        pad_after = max(0, 32 - s - pad_before)
        return cls((h, w, s), (h - 128) // 2, (w - 128) // 2,
                   (s - 32) // 2 if s >= 32 else 0, pad_before, pad_after)

    def forward(self, native: np.ndarray) -> np.ndarray:
        """Apply UVI spatial crop/pad only, preserving trailing non-spatial axes."""
        arr = np.asarray(native)
        if arr.ndim < 3 or tuple(arr.shape[:3]) != self.native_shape:
            raise ValueError(f"Native spatial shape must be {self.native_shape}, got {arr.shape}")
        x = arr[self.start0:self.start0+128, self.start1:self.start1+128, ...]
        if self.native_shape[2] >= 32:
            x = x[:, :, self.start2:self.start2+32, ...]
        else:
            pads = [(0, 0), (0, 0), (self.pad2_before, self.pad2_after)]
            pads += [(0, 0)] * (arr.ndim - 3)
            x = np.pad(x, pads, mode="constant", constant_values=0)
        if tuple(x.shape[:3]) != TEACHER_SHAPE:
            raise AssertionError(f"Wrong transformed shape {x.shape}")
        return np.ascontiguousarray(x)

    def inverse(self, teacher: np.ndarray, *, fill: int | float = 0) -> np.ndarray:
        """Restore native array shape; cropped-off native voxels are unrecoverable."""
        x = np.asarray(teacher)
        if x.ndim < 3 or tuple(x.shape[:3]) != TEACHER_SHAPE:
            raise ValueError(f"Teacher grid must be {TEACHER_SHAPE}, got {x.shape}")
        native = np.full(self.native_shape + x.shape[3:], fill, dtype=x.dtype)
        if self.native_shape[2] >= 32:
            native[self.start0:self.start0+128,
                   self.start1:self.start1+128,
                   self.start2:self.start2+32, ...] = x
        else:
            native[self.start0:self.start0+128,
                   self.start1:self.start1+128, :, ...] = (
                       x[:, :, self.pad2_before:32-self.pad2_after, ...]
                   )
        return native

    def teacher_valid(self) -> np.ndarray:
        """True only where UVI grid refers to a measured native voxel."""
        return self.forward(np.ones(self.native_shape, dtype=np.uint8)).astype(bool)

    def native_valid(self) -> np.ndarray:
        """True only for original voxels retained by the center crop."""
        return self.inverse(np.ones(TEACHER_SHAPE, dtype=np.uint8)).astype(bool)

    def as_dict(self) -> dict:
        return {
            "native_shape": list(self.native_shape),
            "teacher_shape": list(TEACHER_SHAPE),
            "axis0_crop_start": self.start0, "axis1_crop_start": self.start1,
            "axis2_crop_start": self.start2,
            "axis2_pad_before": self.pad2_before,
            "axis2_pad_after": self.pad2_after,
            "tensor_axis_mapping": ["native_axis0", "native_axis1", "native_axis2"],
        }


def _normalise_raw_uvi(image: np.ndarray) -> np.ndarray:
    """Match official ACDCHeartDataset: min/max after crop+pad, then float32."""
    x = np.asarray(image, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite image values")
    low, high = float(x.min()), float(x.max())
    if not high > low:
        raise ValueError("Constant image: official UVI min/max would be undefined")
    return np.ascontiguousarray(((x - low) / (high - low)).astype(np.float32))


def validate_already_uvi(image: np.ndarray) -> np.ndarray:
    """Reject incompatible 'processed' input instead of guessing or reprocessing."""
    x = np.asarray(image)
    if x.shape != TEACHER_SHAPE:
        raise ValueError(f"Preprocessed image must be {TEACHER_SHAPE}, got {x.shape}")
    if not np.isfinite(x).all() or x.min() < -1e-5 or x.max() > 1.00001:
        raise ValueError("Already-UVI image must be finite and scaled to [0,1]")
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def validate_label(label: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(label)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"Label shape/values invalid, expected {shape}, got {x.shape}")
    if not np.isin(np.unique(x), [0, 1, 2, 3]).all():
        raise ValueError("ACDC labels must contain integers from {0,1,2,3}")
    return np.ascontiguousarray(x.astype(np.uint8, copy=False))


@dataclass
class TeacherEndpoints:
    """Arrays are [native0,native1,native2]; add B,C before FR-BME forward."""
    ed_image: np.ndarray
    es_image: np.ndarray
    ed_label: np.ndarray
    es_label: np.ndarray
    valid_spatial: np.ndarray
    grid: UVIGrid | None
    mode: Mode


def prepare_endpoints(ed_image: np.ndarray, es_image: np.ndarray,
                      ed_label: np.ndarray, es_label: np.ndarray,
                      *, mode: Mode, native_grid: UVIGrid | None = None
                      ) -> TeacherEndpoints:
    """Prepare a pair without GPU and without silent double preprocessing.

    For 'already_uvi', both images AND labels must already occupy the UVI grid.
    To transform raw GT to match already-processed images, explicitly call
    UVIGrid.forward(raw_gt) beforehand; do not infer an unknown processed affine.
    """
    if mode == "raw_uvi":
        if ed_image.shape != es_image.shape or ed_image.ndim != 3:
            raise ValueError("Raw ED/ES volumes must have the same 3D shape")
        grid = UVIGrid.for_native_shape(tuple(ed_image.shape))
        if native_grid is not None and native_grid != grid:
            raise ValueError("Supplied native_grid mismatches current raw images")
        ed = _normalise_raw_uvi(grid.forward(ed_image))
        es = _normalise_raw_uvi(grid.forward(es_image))
        y0 = grid.forward(validate_label(ed_label, shape=grid.native_shape))
        y1 = grid.forward(validate_label(es_label, shape=grid.native_shape))
        valid = grid.teacher_valid()
    elif mode == "already_uvi":
        ed = validate_already_uvi(ed_image)
        es = validate_already_uvi(es_image)
        y0 = validate_label(ed_label, shape=TEACHER_SHAPE)
        y1 = validate_label(es_label, shape=TEACHER_SHAPE)
        grid = native_grid
        if grid is not None and grid.native_shape != TEACHER_SHAPE:
            # Can be used for native-space inverse mapping only; ensure correct grid.
            valid = grid.teacher_valid()
        else:
            valid = np.ones(TEACHER_SHAPE, dtype=bool)
    else:
        raise ValueError(f"Mode must be explicitly raw_uvi or already_uvi: {mode!r}")
    return TeacherEndpoints(ed, es, y0, y1, valid, grid, mode)


def read_info_cfg(path: Path) -> dict[str, int]:
    text = Path(path).read_text(encoding="utf-8")
    vals: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in ("ED", "ES", "NbFrame"):
            vals[key] = int(value)
    if set(vals) != {"ED", "ES", "NbFrame"}:
        raise ValueError("Info.cfg must specify ED, ES and NbFrame")
    if not (1 <= vals["ED"] <= vals["NbFrame"]
            and 1 <= vals["ES"] <= vals["NbFrame"]
            and vals["ED"] != vals["ES"] and vals["NbFrame"] >= 3):
        raise ValueError(f"Invalid ED/ES/number of frames: {vals}")
    return vals


def ed_to_es_tau(ed_1based: int, es_1based: int,
                 target_1based: int, frame_count: int) -> float:
    """Candidate t only for the forward ED->ES arc, never ES->ED->next ED.

    Runtime use requires confirmation that DCR was trained with this exact
    time parameterisation. This function is NOT such confirmation.
    """
    if not (1 <= min(ed_1based, es_1based, target_1based)
            and max(ed_1based, es_1based, target_1based) <= frame_count):
        raise ValueError("Frame index outside 1..NbFrame")
    arc = (es_1based - ed_1based) % frame_count
    delta = (target_1based - ed_1based) % frame_count
    if arc == 0 or delta > arc:
        raise ValueError("Target is not on forward ED->ES arc")
    return delta / arc


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    if x.size != y.size or x.size == 0:
        raise ValueError("Cannot compare different phase volumes")
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        raise ValueError("Constant phase image")
    return float(np.dot(x, y) / denom)


def inspect_raw_patient(patient_dir: Path) -> dict:
    """One-patient read-only check; does NOT invoke the pretrained teacher.

    The original 4D NIfTI may carry inconsistent qform/sform vs its ED/ES files.
    Compare indexed voxel arrays; do NOT flip based only on aff2axcodes.
    """
    import nibabel as nib
    patient_dir = Path(patient_dir)
    pid = patient_dir.name
    if not (pid.startswith("patient") and pid[7:].isdigit()):
        raise ValueError("Expected ACDC patientNNN folder")
    info = read_info_cfg(patient_dir / "Info.cfg")
    cine = nib.load(str(patient_dir / f"{pid}_4d.nii.gz"))
    if len(cine.shape) != 4 or cine.shape[3] != info["NbFrame"]:
        raise ValueError("Cine frame count mismatch")
    imgs, masks, correlations = [], [], {}
    for phase in ("ED", "ES"):
        frame = info[phase]
        stem = f"{pid}_frame{frame:02d}"
        image = nib.load(str(patient_dir / f"{stem}.nii.gz"))
        mask = nib.load(str(patient_dir / f"{stem}_gt.nii.gz"))
        if image.shape != mask.shape or image.shape != cine.shape[:3]:
            raise ValueError(f"{phase}: image/label/Cine spatial shape mismatch")
        if not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-3):
            raise ValueError(f"{phase}: image and GT affine mismatch")
        arr = np.asarray(image.dataobj, dtype=np.float32)
        cine_frame = np.asarray(cine.dataobj[..., frame-1], dtype=np.float32)
        nc = _corr(arr, cine_frame)
        if nc < 0.995:
            raise ValueError(f"{phase}: Cine frame mismatch, corr={nc:.6f}")
        imgs.append(arr)
        masks.append(np.asarray(mask.dataobj))
        correlations[phase] = round(nc, 6)
    pair = prepare_endpoints(*imgs, *masks, mode="raw_uvi")
    return {
        "patient": pid, "mode": pair.mode, "ed_1based": info["ED"],
        "es_1based": info["ES"], "frame_count": info["NbFrame"],
        "grid": pair.grid.as_dict() if pair.grid else None,
        "cine_endpoint_correlation": correlations,
        "teacher_image_range": [
            [float(pair.ed_image.min()), float(pair.ed_image.max())],
            [float(pair.es_image.min()), float(pair.es_image.max())],
        ],
        "labels": sorted(set(np.unique(pair.ed_label).tolist() +
                             np.unique(pair.es_label).tolist())),
        "teacher_measured_voxel_fraction": round(float(pair.valid_spatial.mean()), 6),
        "note": "Read-only input check; no DCR teacher inference and no patient-level split proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-patient", type=Path, required=True,
                        help="Read-only single patient inspection of original ACDC NIfTI")
    args = parser.parse_args()
    print(json.dumps(inspect_raw_patient(args.raw_patient), indent=2))


if __name__ == "__main__":
    main()
