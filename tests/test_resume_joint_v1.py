"""CPU-only regression checks for a model-only V1 warm restart."""
from pathlib import Path
import json
import re

import pytest
import torch

from pulse_repro.resume_joint_v1 import (
    build_resume_entry, parse_prior_history, resume, _INIT_ANCHOR,
    _ARG_ANCHOR, _EPOCH_SAVE_ANCHOR, _LOOP_ANCHOR,
)
from pulse_repro.train_joint_v1 import (
    _MODEL_ANCHOR, _IMPORT_ANCHOR, _TEST_END, _SAVE_ANCHOR, file_sha256,
)


def fixture_upstream(path):
    """Exact code anchors; real scripts are separately pinned by hash."""
    original = "\n".join([
        _IMPORT_ANCHOR,
        "import json, numpy as np, random, os, torch",
        "from pathlib import Path",
        "def main():",
        "    ap = object() # temporary fixture placeholder",
        _ARG_ANCHOR,
        _MODEL_ANCHOR,
        _INIT_ANCHOR,
        _LOOP_ANCHOR,
        '        print(f"epoch={epoch}")',
        "        if vm > best:",
        _SAVE_ANCHOR,
        _EPOCH_SAVE_ANCHOR,
        _TEST_END,
        "",
    ])
    path.write_text(original, encoding="utf-8")
    return original


def test_patched_resume_restores_exact_epoch_schedule_and_state(tmp_path):
    upstream = tmp_path / "pulse_seg.py"
    old = fixture_upstream(upstream)
    entry = tmp_path / "resume_entry.py"
    build_resume_entry(upstream, entry)
    modified = entry.read_text(encoding="utf-8")
    assert "for epoch in range(start_epoch + 1, args.epochs + 1):" in modified
    assert "start_epoch = int(raw[\"epoch\"])" in modified
    assert 'opt.add_param_group({"params": early_params, "lr": STAGE2_LR * .05})' in modified
    assert "opt.load_state_dict(raw[\"optimizer\"])" in modified
    assert "scaler.load_state_dict(raw[\"scaler\"])" in modified
    assert "last_training_state.pth" in modified
    assert "weights_only=True" in modified
    assert "prior_history.json" not in modified  # path provided via CLI
    assert upstream.read_text(encoding="utf-8") == old
    compile(modified, str(entry), "exec")


def test_logs_trim_to_best_checkpoint_and_validate_contiguity(tmp_path):
    logfile = tmp_path / "training.log"
    logfile.write_text("\n".join(
        f"Ep{epoch:03d} loss=0.1100 | val RV=0.900 Myo=0.850 LV=0.940 mean=0.897"
        for epoch in range(1, 76)), encoding="utf-8")
    rows = parse_prior_history(logfile, 72)
    assert len(rows) == 72 and rows[-1]["epoch"] == 72
    assert rows[0]["val_mean"] == .897
    logfile.write_text("Ep071 loss=0.1 | val RV=0.9 Myo=0.9 LV=0.9 mean=0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        parse_prior_history(logfile,72)
    assert parse_prior_history(tmp_path / "missing.log",72) == []


def make_run_fixture(root, checkpoint_fold=0,checkpoint_epoch=72):
    original=root/"old"
    upstream=root/"PULSE"/"pulse"/"pulse_seg.py"
    upstream.parent.mkdir(parents=True)
    fixture_upstream(upstream)
    acdc=root/"ACDC"/"training"
    acdc.mkdir(parents=True)
    original.mkdir()
    torch.save({"fold":checkpoint_fold,"epoch":checkpoint_epoch,"best_dice":.90221300494,
                "variant":"selective-v1","training_mode":"joint",
                "model":{"w":torch.tensor([1.0])}}, original/"best_model.pth")
    (original/"run_config.json").write_text(json.dumps({
        "variant":"selective-v1", "training_mode":"joint", "epochs":120,
        "fold":0, "batch_size":4, "seed":42,"lr":3e-4,
        "pulse_root":str(upstream.parent.parent),"acdc_root":str(acdc.parent),
        "source_sha256":file_sha256(upstream)
    }),encoding="utf-8")
    return original, upstream.parent.parent, acdc.parent


def test_dry_run_reports_warm_restart_with_no_mutation(tmp_path):
    previous, pulse, acdc=make_run_fixture(tmp_path)
    target=tmp_path/"new"
    report=resume(previous,pulse,acdc,target,dry_run=True)
    assert report["resumed_from_epoch"]==72
    assert report["next_epoch"]==73 and report["final_epoch"]==120
    assert report["best_dice_at_start"] == pytest.approx(.90221300494)
    assert report["optimizer_restored"] is False
    assert report["exact_resume"] is False
    assert not target.exists()
    assert (previous/"best_model.pth").exists()


def test_recovery_refuses_same_directory_bad_fold_and_wrong_source(tmp_path):
    previous, pulse, acdc = make_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="MUST differ"):
        resume(previous,pulse,acdc,previous,dry_run=True)
    new=tmp_path/"new"
    new.mkdir();(new/"do-not-overwrite.txt").write_text("safe")
    with pytest.raises(FileExistsError):
        resume(previous,pulse,acdc,new,dry_run=True)
    (new/"do-not-overwrite.txt").unlink()
    (pulse/"pulse"/"pulse_seg.py").write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        resume(previous,pulse,acdc,tmp_path/"new2",dry_run=True)


def test_recovery_rejects_wrong_checkpoint_epoch(tmp_path):
    previous,pulse,acdc=make_run_fixture(tmp_path,checkpoint_epoch=69)
    with pytest.raises(ValueError,match="epoch must"):
        resume(previous,pulse,acdc,tmp_path/"new",dry_run=True)


def test_recovery_rejects_wrong_checkpoint_fold(tmp_path):
    previous,pulse,acdc=make_run_fixture(tmp_path,checkpoint_fold=1)
    with pytest.raises(ValueError,match="fold mismatch"):
        resume(previous,pulse,acdc,tmp_path/"new",dry_run=True)
