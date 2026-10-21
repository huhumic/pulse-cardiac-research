from pathlib import Path
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from pulse_repro.train_v1_resumable import (
    _atomic_torch_save, _atomic_json, _check_checkpoint, _optimizer,
    _restore_rng, _rng_state, _run_metadata, _unfreeze_stage2, run,
)


class TinyV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.decoder = nn.Linear(4, 2)
        for layer in self.backbone_blocks[:2]:
            for p in layer.parameters():
                p.requires_grad_(False)


def test_optimizer_grouping_and_stage2_learning_rates():
    mod = TinyV1()
    pulse = SimpleNamespace(FREEZE_BLK=2, WD=.05, STAGE2_LR=2e-5)
    optimizer = _optimizer(mod, pulse, 3e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(3e-4)
    initial_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    _unfreeze_stage2(mod, optimizer, pulse)
    assert len(optimizer.param_groups) == 3
    final_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(final_ids) == len(set(final_ids))
    assert len(final_ids) > len(initial_ids)
    assert all(p.requires_grad for p in mod.parameters())
    with pytest.raises(ValueError, match="optimizer groups"):
        _unfreeze_stage2(mod, optimizer, pulse)


def test_safe_rng_roundtrip_with_torch_weights_only(tmp_path):
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    saved = _rng_state()
    _atomic_torch_save(saved, tmp_path / "rng.pt")
    loaded = torch.load(tmp_path / "rng.pt", map_location="cpu", weights_only=True)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    _restore_rng(loaded)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert actual == pytest.approx(expected)


def test_atomic_output_no_temporary_remnants(tmp_path):
    _atomic_json({"state": 2}, tmp_path / "status.json")
    _atomic_torch_save({"epoch": 5}, tmp_path / "last.pth")
    assert json.loads((tmp_path / "status.json").read_text()) == {"state": 2}
    assert torch.load(tmp_path / "last.pth", weights_only=True)["epoch"] == 5
    assert not list(tmp_path.glob("*.part"))


def test_checkpoint_rejects_missing_and_mismatch(tmp_path):
    wanted = _run_metadata(0, 120, 4, 3e-4, 42, "hash")
    ck = {"checkpoint_kind": "full-epoch-state-v1", "variant": "selective-v1",
          "epoch": 72, "run_config": wanted, "model": {},
          "optimizer": {}, "scaler": {}, "rng": {}, "history": [{}]*72,
          "best_dice": .9}
    best = tmp_path / "best_model.pth"
    best.write_bytes(b"not parsed here")
    assert _check_checkpoint(ck, wanted, best) == 72
    with pytest.raises(ValueError, match="fold"):
        _check_checkpoint(ck, {**wanted, "fold": 1}, best)
    with pytest.raises(ValueError, match="optimizer/scaler/RNG"):
        _check_checkpoint({**ck, "checkpoint_kind": "model-only"}, wanted, best)
    with pytest.raises(ValueError, match="last epoch"):
        _check_checkpoint({**ck, "epoch": 71}, wanted, best)


def test_fresh_dry_run_and_protect_nonempty(tmp_path, capsys):
    pulse = tmp_path / "PULSE" / "pulse"
    pulse.mkdir(parents=True)
    (pulse / "pulse_seg.py").write_text("# pinned stand-in\n")
    acdc = tmp_path / "database" / "training"
    acdc.mkdir(parents=True)
    out = tmp_path / "run"
    run(acdc.parent, pulse.parent, out, dry_run=True)
    assert "FRESH training epoch 1..120" in capsys.readouterr().out
    assert not out.exists()
    out.mkdir()
    (out / "best_model.pth").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        run(acdc.parent, pulse.parent, out, dry_run=True)


def test_resume_dry_run_refuses_missing_file(tmp_path):
    pulse = tmp_path / "PULSE" / "pulse"
    pulse.mkdir(parents=True)
    (pulse / "pulse_seg.py").write_text("# dummy\n")
    acdc = tmp_path / "database" / "training"
    acdc.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run_config.json"):
        run(acdc.parent, pulse.parent, tmp_path / "run", resume=True, dry_run=True)


def test_tiny_training_crash_then_full_state_resume(tmp_path, monkeypatch):
    """End-to-end orchestration without downloading DINOv2 or ACDC files."""
    import sys
    from torch.nn import functional as F
    from pulse_repro import train_v1_resumable as trainer

    class ToyData(torch.utils.data.Dataset):
        def __init__(self, records, img_size=256, train=False):
            self.records = records
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {"img": torch.ones(3, 8, 8) * (index + 1),
                    "mask": torch.ones(8, 8, dtype=torch.long)}

    class ToySeg(nn.Module):
        def __init__(self, *_):
            super().__init__()
            self.backbone_blocks = nn.ModuleList([nn.Conv2d(3, 3, 1) for _ in range(4)])
            self.head = nn.Conv2d(3, 4, 1)
            for block in self.backbone_blocks[:2]:
                for p in block.parameters(): p.requires_grad_(False)
        def forward(self, x):
            for block in self.backbone_blocks:
                x = F.gelu(block(x))
            out = self.head(x)
            return out, F.avg_pool2d(out, 2), F.avg_pool2d(out, 4)

    records = [{"pid": f"patient{i:03d}", "phase": phase}
               for i in range(1, 101) for phase in ("ED", "ES")]
    n_calls = [0]
    def sample_eval(*args, **kwargs):
        n_calls[0] += 1
        if n_calls[0] == 2:
            raise RuntimeError("simulated instance crash")
        return np.asarray([.81, .82, .83])

    stub = SimpleNamespace(
        STAGE1=70, FREEZE_BLK=2, IMG_SIZE=256, STAGE2_LR=2e-5,
        WD=.05, DICE_CLASS_W=torch.tensor([0.,2.,3.,1.]),
        CE_CLASS_W=torch.tensor([.1,2.,3.,1.]), DS_WEIGHTS=[1.,.5,.25],
        PULSESeg=ToySeg, ACDCDataset=ToyData,
        list_acdc=lambda root: records,
        stratified_kfold_pids=lambda recs,nfolds,fold,seed:
            (set(f"patient{i:03d}" for i in range(21,101)),
             set(f"patient{i:03d}" for i in range(1,21))),
        WeightedDice=lambda classweights: nn.CrossEntropyLoss(),
        seg_loss=lambda logits,y,dice,ce: ce(logits,y),
        eval_epoch=sample_eval,
        set_lr=lambda opt,epoch,base,s2: None,
    )
    monkeypatch.setitem(sys.modules, "pulse_repro.runtime", SimpleNamespace(upstream=lambda path:(None,stub)))
    monkeypatch.setattr(trainer, "PULSESelectiveJointV1", lambda p:p)
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda *a, **k:
                        "781e0712ea3b2dd246f86346d73008ecdb73434f\n")
    pulse_dir = tmp_path / "PULSE" / "pulse"
    pulse_dir.mkdir(parents=True)
    (pulse_dir / "pulse_seg.py").write_text("# toy pinned source\n")
    acdc_root = tmp_path / "database"
    (acdc_root / "training").mkdir(parents=True)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError,match="simulated instance crash"):
        trainer.run(acdc_root, pulse_dir.parent, out, epochs=2,
                    batch_size=2, workers=0, save_every=1)
    ck=torch.load(out / "last_state.pth",map_location="cpu",weights_only=True)
    assert ck["epoch"] == 1
    assert "optimizer" in ck and "rng" in ck
    assert torch.load(out / "best_model.pth",weights_only=True)["variant"] == "selective-v1"
    trainer.run(acdc_root, pulse_dir.parent, out, epochs=2,
                batch_size=2,workers=0,save_every=1,resume=True)
    final=torch.load(out / "last_state.pth",map_location="cpu",weights_only=True)
    assert final["epoch"] == 2
    assert len(final["history"]) == 2
    assert json.loads((out/"run_status.json").read_text())["exit_code"] == 0
