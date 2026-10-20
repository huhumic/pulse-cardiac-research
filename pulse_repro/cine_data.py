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


def inspect_cine(normalized_root: Path) -> dict:
    original = raw_root(normalized_root)
    results = []
    nib = _nib()
    for folder in sorted((original / "training").glob("patient*")):
        if not folder.is_dir():
            continue
        rec = {"patient": folder.name}
        try:
            path = find_cine(folder)
            nii = nib.load(str(path))
            if len(nii.shape) != 4 or min(nii.shape[:3]) < 2 or nii.shape[3] < 3:
                raise ValueError(f"Expected (H,W,Z,T>=3) 4D cine, got {nii.shape}")
            rec.update({"found": True, "path": str(path),
                        "shape": list(nii.shape)})
        except (FileNotFoundError, ValueError, OSError) as ex:
            rec.update({"found": False, "reason": str(ex)})
        results.append(rec)
    summary = {
        "original_root": str(original),
        "total_patients": len(results),
        "cine_available": sum(v["found"] for v in results),
        "cine_missing": sum(not v["found"] for v in results),
        "patients": results,
    }
    return summary


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


def build_temporal_input(volume: np.ndarray, cine: np.ndarray,
                         z: int, frame: int, size: int = 256):
    """Return spatial [3,H,W] and true cardiac neighbors [2,H,W]."""
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
    return np.stack(spatial, axis=0), np.stack((minus, plus), axis=0)


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
            if not np.allclose(nii.affine, nib.load(r["img"]).affine, atol=1e-3):
                raise ValueError(f"Affine mismatch: {path} vs {r['img']}")
            cfg = pulse_source.parse_info_cfg(patient / "Info.cfg")
            frame = int(cfg[r["phase"]])
            neighboring_indices(frame, nii.shape[3])
            self.rec_paths[(r["pid"], r["phase"])] = (str(path), frame)
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
        path, frame = self.rec_paths[(rec["pid"], rec["phase"])]
        cine, affine = _load_cine(path)
        nib = _nib()
        image = np.asarray(nib.load(rec["img"]).dataobj, dtype=np.float32)
        label = np.asarray(nib.load(rec["gt"]).dataobj[..., z], dtype=np.uint8)
        spatial, temporal = build_temporal_input(image, cine, z, frame, self.size)
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
        if not r["found"]:
            print(f"  Missing {r['patient']}: {r['reason']}")
    if args.require_all and (info["cine_missing"] or info["total_patients"] != 100):
        raise SystemExit("Missing complete 4D cine data; V2 training REFUSED. "
                         "Do not fabricate neighboring time frames.")


if __name__ == "__main__":
    main()
