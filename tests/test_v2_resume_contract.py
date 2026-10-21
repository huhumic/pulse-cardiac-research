"""CPU-level tests for V2 full-state checkpoint and recovery helpers.

Full end-to-end GPU interruption/resume remains a separate acceptance test.
"""
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from pulse_repro.train_cine_v2 import (
    CHECKPOINT_KIND, VARIANT, _atomic_torch_save, _check_resume,
    _restore_rng, _rng_state, _unfreeze_stage2,
)


def test_rng_checkpoint_roundtrip(tmp_path):
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    checkpoint = tmp_path / "rng_state.pth"
    _atomic_torch_save({"rng": _rng_state()}, checkpoint)

    target = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.random(); np.random.rand(); torch.rand(10)

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _restore_rng(ck["rng"])
    after = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert target == after


def test_full_state_checkpoint_contract(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step()
    config = {"epochs": 5, "fold": 0, "checkpoint_kind": CHECKPOINT_KIND}
    best = tmp_path / "best_model.pth"
    _atomic_torch_save({"variant": VARIANT, "fold": 0, "best_dice": .4,
                        "epoch": 2, "model": model.state_dict()}, best)
    state = {"checkpoint_kind": CHECKPOINT_KIND,
             "variant": VARIANT, "run_config": config,
             "epoch": 2, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scaler": {},
             "rng": _rng_state(), "history": [{"epoch": 1}, {"epoch": 2}],
             "best_dice": .4}
    dest = tmp_path / "last_state.pth"
    _atomic_torch_save(state, dest)
    ck = torch.load(dest, map_location="cpu", weights_only=True)
    assert _check_resume(ck, config, best) == 2
    new_model = nn.Linear(2, 2)
    new_model.load_state_dict(ck["model"], strict=True)
    torch.testing.assert_close(model.weight, new_model.weight, atol=0, rtol=0)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_opt.load_state_dict(ck["optimizer"])
    assert new_opt.state_dict()["state"]
    with pytest.raises(ValueError, match="config"):
        _check_resume(ck, {**config, "fold": 1}, best)
    with pytest.raises(ValueError, match="history"):
        _check_resume({**ck, "history": []}, config, best)


def test_stage2_add_optimizer_group_for_resume():
    blocks = [nn.Linear(2, 2) for _ in range(4)]
    for block in blocks[:2]:
        block.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": list(blocks[2].parameters()), "lr": 1e-5},
        {"params": list(blocks[3].parameters()), "lr": 1e-4},
    ])
    assert len(optimizer.param_groups) == 2
    pulse = SimpleNamespace(STAGE2_LR=1e-5)
    _unfreeze_stage2(blocks, 2, optimizer, pulse)
    assert len(optimizer.param_groups) == 3
    assert all(p.requires_grad for b in blocks[:2] for p in b.parameters())
    assert optimizer.param_groups[2]["lr"] == pulse.STAGE2_LR * .05
