from pathlib import Path

import pytest
nib = pytest.importorskip("nibabel")
import numpy as np

from pulse_repro.metrics import measure_binary
from pulse_repro.prepare_acdc import build, discover_acdc_root, get_frames


def test_perfect_and_offset_masks():
    x = np.zeros((8, 9, 7), dtype=bool)
    x[2:5, 2:6, 2:5] = True
    m = measure_binary(x, x, (0.8, 1.0, 2.5))
    assert m["dice"] == 1.0
    assert m["hd95_mm"] == 0.0
    assert m["assd_mm"] == 0.0
    y = np.roll(x, 1, axis=2)
    m2 = measure_binary(x, y, (0.8, 1.0, 2.5))
    assert 0 < m2["dice"] < 1
    assert m2["hd95_mm"] > 0


def test_empty_masks_flagged():
    empty = np.zeros((5, 5, 5), bool)
    x = empty.copy()
    x[2, 2, 2] = True
    assert measure_binary(empty, x, (1, 1, 1))["status"] == "one_empty"
    assert measure_binary(empty, x, (1, 1, 1))["dice"] == 0.0
    assert np.isnan(measure_binary(empty, empty, (1, 1, 1))["hd95_mm"])


def test_nested_dataset_adapter(tmp_path: Path):
    base = tmp_path / "input" / "ACDC" / "database" / "training" / "patient001"
    base.mkdir(parents=True)
    (base / "Info.cfg").write_text("ED: 1\nES: 12\nGroup: NOR\nNbFrame: 14\n")
    vol = np.zeros((5, 6, 3), np.uint8)
    for frame in (1, 12):
        for gt in (False, True):
            label = "_gt" if gt else ""
            f = base / f"patient001_frame{frame:02d}{label}.nii"
            f.mkdir()
            fpath = f / ("strange_gt.nii" if gt else "original.nii")
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(fpath))
    out = tmp_path / "working" / "database"
    m = build(tmp_path / "input", out)
    assert m["training_patients"] == 1
    assert (out / "training/patient001/patient001_frame01.nii").is_symlink()
    assert (out / "training/patient001/patient001_frame01_gt.nii").is_symlink()
    # Repeat is safe and idempotent for symlinks from the same files.
    build(tmp_path / "input", out)
    assert len(get_frames(out / "training/patient001")) == 4


def test_bad_label_rejected(tmp_path: Path):
    base = tmp_path / "in" / "training" / "patient001"
    base.mkdir(parents=True)
    (base / "Info.cfg").write_text("ED: 1\nES: 12\nGroup: NOR\n")
    for frame in (1, 12):
        for gt in (False, True):
            suffix = "_gt" if gt else ""
            p = base / f"patient001_frame{frame:02d}{suffix}.nii"
            vol = np.full((3, 3, 3), 7 if gt else 0, dtype=np.uint8)
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
    with pytest.raises(ValueError, match="Unexpected labels"):
        build(tmp_path / "in", tmp_path / "out")
