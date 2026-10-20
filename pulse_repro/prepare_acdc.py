"""Adapt nested ACDC NIfTI folders on Kaggle into PULSE's expected tree.

No image conversion: symlink the exact source files. See README for provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np

PHASE_PATTERN = re.compile(r"^(patient\d+_frame\d{2}(?:_gt)?)\.nii(?:\.gz)?$")


def discover_acdc_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "training").is_dir():
        return root
    candidates = sorted(
        p for p in root.rglob("database")
        if p.is_dir() and (p / "training").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"ACDC database/training not found under {root}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple ACDC roots found; set --input-root directly: {candidates}")
    return candidates[0]


def get_frames(patient: Path) -> dict[str, Path]:
    """Locate original phase/GT NIfTI under ordinary or nested .nii folders."""
    found: dict[str, Path] = {}
    for p in patient.iterdir():
        m = PHASE_PATTERN.fullmatch(p.name)
        if not m:
            continue
        key = m.group(1)
        if p.is_file():
            source = p
        elif p.is_dir():
            choices = sorted(
                f for f in p.rglob("*")
                if f.is_file() and f.name.endswith((".nii", ".nii.gz"))
            )
            if len(choices) != 1:
                raise ValueError(f"Expected exactly one file under {p}: {choices}")
            source = choices[0]
        else:
            continue
        if key in found:
            raise ValueError(f"Duplicate NIfTI frame for {key} in {patient}")
        found[key] = source.resolve()
    return found


def safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"Cannot overwrite existing file: {target}")
    os.symlink(source, target)


def parse_cfg(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def build(input_root: Path, output_root: Path, check_labels: bool = True,
          include_testing: bool = False) -> dict:
    original = discover_acdc_root(input_root)
    dest = output_root.resolve()
    if original == dest:
        raise ValueError("Input and output must be different")
    if str(dest).startswith(str(original) + os.sep):
        raise ValueError("Output cannot be inside the input directory")
    dest.mkdir(parents=True, exist_ok=True)

    details = []
    for subset in (("training", "testing") if include_testing else ("training",)):
        src_dir = original / subset
        if not src_dir.exists():
            continue
        for p in sorted(src_dir.glob("patient*")):
            if not p.is_dir():
                continue
            cfg = p / "Info.cfg"
            if not cfg.is_file():
                raise FileNotFoundError(f"Missing Info.cfg in {p}")
            folder = dest / subset / p.name
            safe_symlink(cfg.resolve(), folder / "Info.cfg")
            frames = get_frames(p)
            info = parse_cfg(cfg)
            phase_info = {}
            for phase in ("ED", "ES"):
                if phase not in info:
                    raise ValueError(f"Missing {phase} phase in {cfg}")
                frame = int(info[phase])
                stem = f"{p.name}_frame{frame:02d}"
                if stem not in frames:
                    raise FileNotFoundError(f"Missing {stem} in {p}")
                if subset == "training" and f"{stem}_gt" not in frames:
                    raise FileNotFoundError(f"Missing {stem}_gt in {p}")
                phase_info[phase] = frame
            for stem, source in frames.items():
                ext = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
                # Prefix uses original stem; follow the actual file extension.
                safe_symlink(source, folder / (stem + ext))
                if check_labels and subset == "training" and stem.endswith("_gt"):
                    arr = np.asanyarray(nib.load(str(source)).dataobj)
                    unique = np.unique(arr)
                    if not np.isin(unique, [0, 1, 2, 3]).all():
                        raise ValueError(f"Unexpected labels {unique} in {source}")
            details.append({"subset": subset, "patient": p.name,
                            "phases": phase_info, "files": len(frames)})

    manifest = {
        "input_root": str(original),
        "output_root": str(dest),
        "strategy": "read-only symlinks, no modification to voxels or orientation",
        "training_patients": sum(d["subset"] == "training" for d in details),
        "testing_patients": sum(d["subset"] == "testing" for d in details),
        "patients": details,
    }
    (dest / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--output-root", type=Path,
                   default=Path("/kaggle/working/ACDC_pulse/database"))
    p.add_argument("--skip-label-check", action="store_true")
    p.add_argument("--include-testing", action="store_true",
                   help="Optional: test data are not needed for out-of-fold validation")
    args = p.parse_args()
    m = build(args.input_root, args.output_root,
              check_labels=not args.skip_label_check,
              include_testing=args.include_testing)
    print(f"Original ACDC: {m['input_root']}")
    print(f"Symlink tree: {m['output_root']}")
    print(f"Training patients: {m['training_patients']}; testing patients: {m['testing_patients']}")
    print("Done; file bytes are unchanged.")


if __name__ == "__main__":
    main()
