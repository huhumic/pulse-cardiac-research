"""Patient-disjoint, ED/ES-labeled cardiac-time data for PULSE.

Spatial neighboring slices and neighboring cardiac *time* frames are kept as
separate inputs. 4D cine must be explicitly present. Refuse ambiguous datasets
rather than silently substitute spatial slices for time frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np


def _nib():
    import nibabel as nib
    return nib


def find_cine(patient_dir: Path) -> Path:
    """Find exactly one real .nii[.gz] in patientNNN_4d.nii[.gz] directory/file."""
    stem = patient_dir.name + "_4d"
    matches = []
    for ending in (".nii", ".nii.gz"):
        candidate = patient_dir / (stem + ending)
        if candidate.is_file():
            matches.append(candidate)
        elif candidate.is_dir():
            files = sorted(p for p in candidate.rglob("*")
                           if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz")))
            if len(files) != 1:
                raise ValueError(f"Ambiguous nested 4D NIfTI under {candidate}: {len(files)} files")
            matches.append(files[0])
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Exactly one {stem}.nii[.gz] required in {patient_dir}; found {len(matches)}. "
            "The available ACDC mirror may contain only ED/ES frames."
        )
    return matches[0].resolve()


def raw_root(normalized_root: Path) -> Path:
    manifest_path = normalized_root / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run pulse_repro.prepare_acdc first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["input_root"])
    if not (path / "training").is_dir():
        raise FileNotFoundError(f"Original training dir no longer available: {path}")
    return path


def _phase_similarity(phase_volume: np.ndarray, cine_frame: np.ndarray) -> float:
    """Intensity-affine invariant Pearson correlation. Fail closed on flat data."""
    x = np.asarray(phase_volume, dtype=np.float32).ravel()[::max(1, phase_volume.size // 80000)]
    y = np.asarray(cine_frame, dtype=np.float32).ravel()[::max(1, cine_frame.size // 80000)]
    if x.shape != y.shape or x.size < 100 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Invalid/non-finite voxel samples when checking cine phase")
    x = x.astype(np.float64); y = y.astype(np.float64)
    x -= x.mean(); y -= y.mean()
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx < 1e-8 or ny < 1e-8:
        raise ValueError("Constant image prevents ED/ES-to-cine identity check")
    return float(np.dot(x, y) / (nx * ny))


def verify_patient_cine(original: Path, normalized_root: Path,
                        min_ncc: float = .995) -> dict:
    """Verify ED/ES files really match their Info.cfg time indices in 4D.

    Check normalized images (the actual inputs to PULSE), matching GT masks,
    geometry, NbFrame and correlations against the declared and adjacent time
    indices. We avoid assuming 3D files are *bitwise* identical to 4D frames:
    mild affine intensity scaling is allowed, wrong phase is not.
    """
    nib = _nib()
    raw_pdir = original
    pdir = normalized_root / "training" / original.name
    info = {}
    for line in (original / "Info.cfg").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    for key in ("ED", "ES", "NbFrame"):
        if key not in info:
            raise ValueError(f"Missing {key} in {original/'Info.cfg'}")
    path = find_cine(raw_pdir)
    cine_nii = nib.load(str(path))
    shape = cine_nii.shape
    if len(shape) != 4 or min(shape[:3]) < 2 or shape[3] < 3:
        raise ValueError(f"Invalid 4D cine dimensions: {shape}")
    if int(info["NbFrame"]) != shape[3]:
        raise ValueError(f"NbFrame={info['NbFrame']} does not match 4D T={shape[3]}")
    diagnostics = {}
    for phase in ("ED", "ES"):
        frame = int(info[phase])
        current, previous, following = neighboring_indices(frame, shape[3])
        img_path = pdir / f"{pdir.name}_frame{frame:02d}.nii"
        if not img_path.is_file():
            img_path = img_path.with_suffix(".nii.gz")
        gt_path = pdir / f"{pdir.name}_frame{frame:02d}_gt.nii"
        if not gt_path.is_file():
            gt_path = gt_path.with_suffix(".nii.gz")
        if not img_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f"Missing normalized {phase} image/GT in {pdir}")
        img = nib.load(str(img_path)); gt = nib.load(str(gt_path))
        if img.shape != shape[:3] or gt.shape != shape[:3]:
            raise ValueError(f"{phase}: phase/GT shape differs from 4D cine")
        # When 4D qform/sform are unset, nibabel supplies a fallback affine.
        # Accept voxel-index alignment only after exact indexed-frame evidence.
        affine_ok = bool(np.allclose(
            cine_nii.affine, img.affine, rtol=0, atol=1e-3
        ))
        if not np.allclose(img.affine, gt.affine, rtol=0, atol=1e-3):
            raise ValueError(f"{phase}: label geometry differs from image")
        phase_arr = np.asarray(img.dataobj, dtype=np.float32)
        scores = {}
        for idx in (previous, current, following):
            scores[idx] = _phase_similarity(
                phase_arr, np.asarray(cine_nii.dataobj[..., idx], dtype=np.float32)
            )
        target_score = scores[current]
        max_neighbor = max(scores[previous], scores[following])
        if target_score < min_ncc:
            raise ValueError(f"{phase}: frame index {frame} has NCC={target_score:.5f} < {min_ncc}")
        if max_neighbor > target_score + 0.0001:
            raise ValueError(f"{phase}: declared frame {frame} NCC={target_score:.5f}, "
                             f"adjacent frame matches BETTER ({max_neighbor:.5f}): possible time off-by-one")
        if target_score - max_neighbor < 1e-6:
            raise ValueError(f"{phase}: ambiguous 4D phase mapping; adjacent frame "
                             "is indistinguishable from declared ED/ES. Manual review required")
        alignment = "affine"
        if not affine_ok:
            if (int(cine_nii.header["qform_code"]) != 0 or
                    int(cine_nii.header["sform_code"]) != 0):
                raise ValueError(
                    f"{phase}: affine mismatch with valid 4D geometry"
                )
            actual_frame = np.asarray(
                cine_nii.dataobj[..., current], dtype=np.float32
            )
            if not np.allclose(
                phase_arr, actual_frame, rtol=1e-5, atol=1e-5
            ):
                raise ValueError(
                    f"{phase}: missing 4D qform/sform AND "
                    "target-frame voxel arrays do not match"
                )
            alignment = "voxel_identity_missing_4d_geometry"

        diagnostics[phase] = {"frame_1based": frame,
                              "alignment": alignment,
                              "indexed_frame_ncc": target_score,
                              "best_adjacent_ncc": max_neighbor}
    if int(info["ED"]) == int(info["ES"]):
        raise ValueError("ED and ES are the same frame")
    return {"path": str(path), "shape": list(shape), "phase_checks": diagnostics}


def inspect_cine(normalized_root: Path) -> dict:
    original = raw_root(normalized_root)
    results = []
    for folder in sorted((original / "training").glob("patient*")):
        if not folder.is_dir():
            continue
        rec = {"patient": folder.name}
        try:
            checks = verify_patient_cine(folder, normalized_root)
            rec.update({"found": True, "validated": True, **checks})
        except (FileNotFoundError, ValueError, OSError, KeyError) as ex:
            rec.update({"found": False, "validated": False, "reason": str(ex)})
        results.append(rec)
    good = sum(v["validated"] for v in results)
    return {"protocol": "strict-cine-ed-es-v2", "original_root": str(original),
            "total_patients": len(results), "cine_available": good,
            "cine_missing": len(results) - good,
            "ready": len(results) == 100 and good == 100,
            "patients": results}


@lru_cache(maxsize=8)
def _load_cine(path_str: str):
    # Cache eight patient cine volumes per DataLoader worker to avoid decoding
    # the same NIfTI repeatedly. Memory cost depends on actual image dimensions.
    nib = _nib()
    nii = nib.load(path_str)
    return np.asarray(nii.dataobj, dtype=np.float32), nii.affine


def validate_cine_pair(record: dict, cine: np.ndarray, affine: np.ndarray) -> None:
    nib = _nib()
    ed_es = nib.load(record["img"])
    if cine.ndim != 4 or cine.shape[:3] != ed_es.shape:
        raise ValueError(f"4D cine/ED-ES spatial shape mismatch for {record['pid']}")
    if not np.allclose(affine, ed_es.affine, atol=1e-3):
        raise ValueError(f"4D cine/ED-ES affine mismatch for {record['pid']}")


def neighboring_indices(frame: int, num_frames: int) -> tuple[int, int, int]:
    """ACDC Info.cfg frame IDs are 1-based; full cine time axis is 0-based."""
    if num_frames < 3 or not (1 <= frame <= num_frames):
        raise ValueError(f"Invalid ED/ES frame {frame} for {num_frames} cine frames")
    t = frame - 1
    return t, (t - 1) % num_frames, (t + 1) % num_frames


def _zscore(img: np.ndarray) -> np.ndarray:
    return ((img - float(img.mean())) / (float(img.std()) + 1e-6)).astype(np.float32)


def phase_context(frame: int, num_frames: int, ed_frame: int) -> np.ndarray:
    """ED-anchored target phase sin/cos and 1/T for directional coding.

    The start of a NIfTI cine series need not mean the same physiological
    state for all patients; use the verified ED frame as cycle origin.
    """
    t, _, _ = neighboring_indices(frame, num_frames)
    reference, _, _ = neighboring_indices(ed_frame, num_frames)
    angle = 2 * np.pi * ((t - reference) % num_frames) / num_frames
    return np.array([np.sin(angle), np.cos(angle), 1. / num_frames], dtype=np.float32)


def build_temporal_input(volume: np.ndarray, cine: np.ndarray,
                         z: int, frame: int, ed_frame: int, size: int = 256):
    """Return spatial [3,H,W], real time neighbors [2,H,W], phase [3]."""
    if volume.ndim != 3 or cine.shape[:3] != volume.shape:
        raise ValueError("Volume and cine shapes must agree")
    s = volume.shape[2]
    if not 0 <= z < s:
        raise IndexError(z)
    _, tm, tp = neighboring_indices(frame, cine.shape[3])
    spatial = [cv2.resize(_zscore(volume[..., max(0,min(s-1,zz))]),
                          (size, size), interpolation=cv2.INTER_LINEAR)
               for zz in (z-1, z, z+1)]
    minus = cv2.resize(_zscore(cine[..., z, tm]), (size, size),
                       interpolation=cv2.INTER_LINEAR)
    plus = cv2.resize(_zscore(cine[..., z, tp]), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    return np.stack(spatial, axis=0), np.stack((minus, plus), axis=0), phase_context(frame, cine.shape[3], ed_frame)


class ACDCCineDataset:
    """Wrap original PULSE slice selection with time-synchronised augmentations.

    `records` must include `pid,phase,img,gt`. Only ED/ES have ground truth.
    Geometry and photometric augmentation is synchronised across the 5 input
    channels through one Albumentations Compose call, including labels.
    """

    def __init__(self, records, original_root: Path, pulse_source,
                 img_size: int = 256, train: bool = False):
        import torch  # imported late for lightweight data inspection
        self.torch = torch
        self.size = img_size
        self.train = train
        self.base = pulse_source.ACDCDataset(records, img_size=img_size, train=train)
        self.rec_paths = {}
        nib = _nib()
        for r in records:
            patient = original_root / "training" / r["pid"]
            path = find_cine(patient)
            nii = nib.load(str(path))
            if len(nii.shape) != 4 or nii.shape[3] < 3:
                raise ValueError(f"Cine is not valid 4D: {path}, shape={nii.shape}")
            if nii.shape[:3] != nib.load(r["img"]).shape:
                raise ValueError(f"Shape mismatch: {path} vs {r['img']}")
            phase_img = nib.load(r["img"])
            if not np.allclose(
                nii.affine, phase_img.affine, rtol=0, atol=1e-3
            ):
                if (int(nii.header["qform_code"]) != 0 or
                        int(nii.header["sform_code"]) != 0):
                    raise ValueError(f"Valid 4D affine mismatch: {path}")
                phase_cfg = pulse_source.parse_info_cfg(
                    patient / "Info.cfg"
                )
                phase_frame = int(phase_cfg[r["phase"]])
                neighboring_indices(phase_frame, nii.shape[3])
                target_3d = np.asarray(phase_img.dataobj, dtype=np.float32)
                target_4d = np.asarray(
                    nii.dataobj[..., phase_frame - 1], dtype=np.float32
                )
                if not np.allclose(
                    target_3d, target_4d, rtol=1e-5, atol=1e-5
                ):
                    raise ValueError(f"Unverified 3D/4D voxel mapping: {path}")
            cfg = pulse_source.parse_info_cfg(patient / "Info.cfg")
            frame = int(cfg[r["phase"]])
            neighboring_indices(frame, nii.shape[3])
            self.rec_paths[(r["pid"], r["phase"])] = (
                str(path), frame, int(cfg["ED"]))
        self.aug = None
        if train:
            source_aug = pulse_source._build_aug()
            if source_aug is not None:
                import albumentations as A
                self.aug = A.Compose(
                    source_aug.transforms,
                    additional_targets={"temporal_minus": "image", "temporal_plus": "image"},
                )

    def __len__(self):
        return len(self.base.slices)

    def __getitem__(self, index):
        rec, z, s = self.base.slices[index]
        path, frame, ed_frame = self.rec_paths[(rec["pid"], rec["phase"])]
        cine, affine = _load_cine(path)
        nib = _nib()
        image = np.asarray(nib.load(rec["img"]).dataobj, dtype=np.float32)
        label = np.asarray(nib.load(rec["gt"]).dataobj[..., z], dtype=np.uint8)
        spatial, temporal, phase = build_temporal_input(image, cine, z, frame, ed_frame, self.size)
        gt = cv2.resize(label, (self.size,self.size),interpolation=cv2.INTER_NEAREST)
        x = np.ascontiguousarray(spatial.transpose(1,2,0))
        minus, plus = temporal
        if self.aug is not None:
            aug = self.aug(image=x, mask=gt, temporal_minus=minus,
                           temporal_plus=plus)
            x, gt = aug["image"], aug["mask"]
            minus, plus = aug["temporal_minus"], aug["temporal_plus"]
        return {
            "img": self.torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float(),
            "temporal": self.torch.from_numpy(
                np.ascontiguousarray(np.stack((minus,plus),axis=0))).float(),
            "phase_context": self.torch.from_numpy(phase.copy()).float(),
            "mask": self.torch.from_numpy(np.ascontiguousarray(gt)).long(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-root", type=Path, required=True,
                        help="Normalized ACDC database with adapter_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-all",action="store_true")
    args = parser.parse_args()
    info = inspect_cine(args.acdc_root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"ACDC 4D cine: {info['cine_available']}/{info['total_patients']} patients")
    for r in info["patients"]:
        if not r["validated"]:
            print(f"  Failed {r['patient']}: {r['reason']}")
    if args.require_all and (info["cine_missing"] or info["total_patients"] != 100):
        raise SystemExit("4D cine/ED-ES correspondence preflight failed; V2 training REFUSED. "
                         "Do not fabricate or misindex neighboring time frames.")


if __name__ == "__main__":
    main()
