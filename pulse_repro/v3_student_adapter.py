"""G2: bridge frozen 3D teacher pseudo-label cache to PULSE 2.5D student.

Data and axis contract:
  teacher: [native_axis0_cropped, native_axis1_cropped, native_axis2_padded]
  PULSE:   [3 spatial-neighbour slices, 256, 256] target native_axis2=z
  time:    PULSE receives two real adjacent *Cine time* frames separately.

The teacher labels do NOT cover the full original field of view. Inverse
mapping fills crop-excluded regions with ignore=255; never invent background.
No train/validation mixing. Only consumes TRAIN candidate caches, without
running the teacher. No network, no GPU or checkpoint loading needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cine_data import build_temporal_input
from .v3_uvi_acdc_adapter import UVIGrid, TEACHER_SHAPE, split_role, read_info_cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified_grid(grid_meta: dict) -> UVIGrid:
    if not isinstance(grid_meta, dict):
        raise ValueError("Missing native-grid metadata")
    native_shape = tuple(grid_meta["native_shape"])
    grid = UVIGrid.for_native_shape(native_shape)
    if grid_meta != grid.as_dict():
        raise ValueError("UVI grid mapping differs from expected native-axis crop/pad convention")
    return grid


@dataclass(frozen=True)
class NativePseudo:
    patient: str
    frame1: int
    ed1: int
    es1: int
    grid: UVIGrid
    pseudo: np.ndarray     # native [X,Y,Z], uint8 (ignore=255 outside teacher crop)
    weights: np.ndarray    # native [X,Y,Z], float32 zero outside teacher crop
    binary: np.ndarray     # native [X,Y,Z], {0,1}
    image_weights: np.ndarray   # native [X,Y,Z], float32
    manifest: dict


def load_native_pseudo(npz_path: Path, *, verify_hash: bool = True) -> NativePseudo:
    """Strict read-only cache loader; metadata required and hash validated.

    B5 binary_consensus_uvi: baseline valid+double-agreement confidence.
    B6 weight_image_uvi: experimental classwise residual confidence.
    Both share exactly the same pseudo label and spatial support.
    """
    npz_path = Path(npz_path)
    meta_path = npz_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "v3_acdc_candidate_pseudolabel_cache_v1":
        raise ValueError("Unexpected cache schema")
    if meta.get("split") != "train_001_090_only":
        raise ValueError("Cache did not declare training-only split")
    patient = meta.get("patient")
    if not isinstance(patient, str) or split_role(patient) != "train":
        raise ValueError("Refusing validation or malformed patient cache")
    frame1 = int(meta["target_1based"])
    ed1, es1 = int(meta["ed_1based"]), int(meta["es_1based"])
    if not ed1 < frame1 < es1:
        raise ValueError("Cache frame not strictly inside teacher's ED->ES arc")
    if npz_path.name != f"{patient}_t{frame1:02d}.npz" or npz_path.parent.name != patient:
        raise ValueError("Cache filename and patient metadata mismatch")
    if verify_hash and meta.get("npz_sha256") != _sha256(npz_path):
        raise ValueError("Candidate cache hash mismatch; refusing corrupted data")
    grid = _verified_grid(meta["native_grid_mapping"])
    if meta.get("mode") != "raw_uvi":
        raise ValueError("Cache must be explicit raw_uvi geometry")
    if meta.get("flow_semantics") != "target_grid_pullback_voxel_offsets_native_nifti_axis012":
        raise ValueError("Unexpected teacher flow geometry")
    if (meta.get("pseudo_policy") != "strict_dual_endpoint_nn_agreement_ignore255"
        or meta.get("weight_policy", {}).get("policy") !=
            "classwise_median_normalized_registration_residual_softweight_v1"):
        raise ValueError("Unexpected pseudo or weight definition")

    with np.load(npz_path, allow_pickle=False) as arrays:
        required = ("pseudo_uvi", "weight_image_uvi", "binary_consensus_uvi",
                    "valid_both_uvi", "measured_uvi", "frame_uvi")
        if not all(name in arrays for name in required):
            raise ValueError("Candidate cache missing required arrays")
        labels = arrays["pseudo_uvi"]
        image_weights = arrays["weight_image_uvi"]
        binary = arrays["binary_consensus_uvi"]
        both = arrays["valid_both_uvi"]
        measured = arrays["measured_uvi"]
        target = arrays["frame_uvi"]
        if any(x.shape != TEACHER_SHAPE for x in
               (labels, image_weights, binary, both, measured, target)):
            raise ValueError("Cache array geometry mismatch")
        if not np.isin(np.unique(labels), [0,1,2,3,255]).all():
            raise ValueError("Candidate labels are not in 0..3/255")
        if not all(np.isin(np.unique(x), [0,1]).all() for x in (binary,both,measured)):
            raise ValueError("Validity and consensus flags must be binary")
        if not np.array_equal(measured.astype(bool), grid.teacher_valid()):
            raise ValueError("Teacher native-measured mask does not match geometry metadata")
        valid = binary.astype(bool)
        if not np.array_equal(valid, labels != 255):
            raise ValueError("Consensus mask and ignore=255 do not match")
        if np.any(valid & ~both.astype(bool)) or np.any(both.astype(bool) & ~measured.astype(bool)):
            raise ValueError("Candidate mask uses padded/unobserved voxels")
        if not np.isfinite(image_weights).all() or (
            image_weights.min() < 0 or image_weights.max() > 1
        ) or not np.isfinite(target).all():
            raise ValueError("Nonfinite or out-of-range target/weights")
        if np.any(image_weights[~valid] != 0):
            raise ValueError("Ignored candidates have nonzero image reliability")
        if not (0 <= float(target.min()) <= float(target.max()) <= 1.001):
            raise ValueError("Cached target image is not minmax scaled")
        native_label = grid.inverse(labels, fill=255).astype(np.uint8)
        native_binary = grid.inverse(binary, fill=0).astype(np.uint8)
        native_image_weight = grid.inverse(image_weights, fill=0).astype(np.float32)
        assert np.all(native_label[native_binary == 0] == 255)
        assert np.all(native_image_weight[native_binary == 0] == 0)
        return NativePseudo(
            patient, frame1, ed1, es1, grid, native_label,
            native_image_weight, native_binary,
            native_image_weight, meta
        )


def pulse_slice_pseudo(cache: NativePseudo, z: int, *, size: int = 256,
                       policy: str = "binary") -> tuple[np.ndarray, np.ndarray]:
    """Spatial slice of candidate segmentation target in original PULSE pixel grid.

    OpenCV width,height is (size,size). Never use bilinear label interpolation;
    nearest-neighbour is applied equally to validity and image weights so
    soft weights cannot leak into 255 pixels at crop boundary.
    """
    if policy not in ("binary", "image"):
        raise ValueError("Choose 'binary' (B5) or 'image' (B6)")
    if size < 8 or not 0 <= z < cache.grid.native_shape[2]:
        raise ValueError("Invalid PULSE target size or native slice index")
    labels = cv2.resize(cache.pseudo[..., z], (size, size), interpolation=cv2.INTER_NEAREST)
    confidence_source = cache.binary if policy == "binary" else cache.image_weights
    weights = cv2.resize(confidence_source[..., z], (size, size), interpolation=cv2.INTER_NEAREST)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if not np.all(weights[labels == 255] == 0):
        raise AssertionError("Resizing changed the validity/weight correspondence")
    return labels, weights


def prepare_training_slice(cache: NativePseudo, raw_cine: np.ndarray, *,
                           z: int, policy: str = "binary", size: int = 256) -> dict:
    """Build one real intermediate-target PULSE 2.5D training example.

    Uses EXACT already-existing build_temporal_input() for spatial Z neighbor
    images, real time neighbors and ED-anchored phase context. Target is an
    UNLABELED real Cine frame; supervision is teacher pseudo only.
    """
    if raw_cine.ndim != 4 or tuple(raw_cine.shape[:3]) != cache.grid.native_shape:
        raise ValueError("Native Cine geometry doesn't match metadata")
    if not cache.es1 <= raw_cine.shape[3]:
        raise ValueError("Cached endpoint index exceeds Cine temporal length")
    frame_volume = np.asarray(raw_cine[..., cache.frame1-1], dtype=np.float32)
    spatial, temporal, phase = build_temporal_input(
        frame_volume, raw_cine, z, cache.frame1, cache.ed1, size,
    )
    label, confidence = pulse_slice_pseudo(cache, z, size=size, policy=policy)
    return {
        "img": torch.from_numpy(np.ascontiguousarray(spatial)).float(),
        "temporal": torch.from_numpy(np.ascontiguousarray(temporal)).float(),
        "phase_context": torch.from_numpy(np.ascontiguousarray(phase)).float(),
        "pseudo": torch.from_numpy(label.astype(np.int64)),
        "pseudo_weight": torch.from_numpy(confidence),
        "valid_count": int((label != 255).sum()),
        "foreground_count": int(((label > 0) & (label < 4)).sum()),
        "patient": cache.patient, "frame1": cache.frame1, "z": z,
        "policy": policy,
    }


@lru_cache(maxsize=5)
def _load_cine_raw(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


class CandidateStudentDataset(Dataset):
    """Three-case G2 only. Do not use on validation patients.

    Dataset emits all *original* native_axis2 slices having nonzero agreement
    candidate coverage (includes potential all-background slices), preserving
    PULSE's original Z-axis definition, rather than padded UVI slices.
    """
    def __init__(self, cache_root: Path, raw_root: Path, *,
                 policy: str = "binary", size: int = 256,
                 max_patients: int = 3):
        if policy not in ("binary", "image"):
            raise ValueError("policy must be binary or image")
        paths = sorted(Path(cache_root).glob("patient*/patient*_t*.npz"))
        if not paths or len(paths) > max_patients:
            raise ValueError("G2 dataset only accepts 1..3 candidate caches")
        self.policy = policy
        self.size = size
        self.entries: list[tuple[NativePseudo, int, str]] = []
        self.report = []
        for path in paths:
            cache = load_native_pseudo(path)
            cine_file = Path(raw_root) / cache.patient / f"{cache.patient}_4d.nii.gz"
            if not cine_file.is_file():
                raise FileNotFoundError(cine_file)
            if cache.manifest.get("raw_cine_sha256") != _sha256(cine_file):
                raise ValueError("Real Cine input differs from the teacher's cached raw file")
            info = read_info_cfg(cine_file.parent / "Info.cfg")
            if (info["ED"], info["ES"]) != (cache.ed1, cache.es1):
                raise ValueError("Target Cine ED/ES differ from cached endpoints")
            if info["NbFrame"] < cache.frame1:
                raise ValueError("Target time outside Cine length")
            arr = _load_cine_raw(str(cine_file))
            if tuple(arr.shape[:3]) != cache.grid.native_shape or arr.shape[3] != info["NbFrame"]:
                raise ValueError("Original Cine dimensions differ from cache metadata")
            accepted = []
            for z in range(cache.grid.native_shape[2]):
                label, confidence = pulse_slice_pseudo(cache,z,size=size,policy=policy)
                if int((confidence > 0).sum()) > 0:
                    self.entries.append((cache, z, str(cine_file)))
                    accepted.append(z)
            self.report.append({
                "patient": cache.patient, "frame1": cache.frame1,
                "usable_native_slices": accepted,
                "slice_count": len(accepted),
                "policy": policy,
            })
        if not self.entries:
            raise ValueError("No usable candidate pseudo-label slices")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        cache, z, cine_file = self.entries[index]
        return prepare_training_slice(cache, _load_cine_raw(cine_file),
                                      z=z, policy=self.policy, size=self.size)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",type=Path,
                   default=Path("/root/autodl-tmp/v3_candidate_cache_g2_20261025"))
    p.add_argument("--raw-root",type=Path,
                   default=Path("/root/autodl-tmp/datasets/ACDC/database/training"))
    p.add_argument("--policy",choices=("binary","image"),default="binary")
    a=p.parse_args()
    dataset=CandidateStudentDataset(a.cache_root,a.raw_root,policy=a.policy)
    sample=dataset[0]
    print(json.dumps({
        "status":"G2_STUDENT_SPATIAL_BRIDGE_PASS",
        "policy":a.policy, "num_training_slices":len(dataset),
        "sample_img_shape":list(sample["img"].shape),
        "sample_temporal_shape":list(sample["temporal"].shape),
        "sample_phase_shape":list(sample["phase_context"].shape),
        "pseudo_shape":list(sample["pseudo"].shape),
        "confidence_shape":list(sample["pseudo_weight"].shape),
        "sample_valid_pixels":sample["valid_count"],
        "sample_foreground_pixels":sample["foreground_count"],
        "per_patient":dataset.report,
        "note":"No segmentation model forward/training, only CPU 3D->2.5D mapping"
    }, indent=2))


if __name__=="__main__":
    main()
