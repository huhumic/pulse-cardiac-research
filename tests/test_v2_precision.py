"""Regression tests for the FP16 underflow / zero Adam state incident."""
import pytest
import torch
from torch import nn

from pulse_repro.train_cine_v2 import _clip_checked_gradients, _resolve_precision


def test_precision_defaults_to_fp32_when_bf16_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _resolve_precision("auto", torch.device("cpu")) == "fp32"
    assert _resolve_precision("auto", torch.device("cuda")) == "fp32"
    assert _resolve_precision("fp32", torch.device("cuda")) == "fp32"
    with pytest.raises(RuntimeError, match="BF16"):
        _resolve_precision("bf16", torch.device("cuda"))


def test_bf16_selected_for_supported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _resolve_precision("auto", torch.device("cuda")) == "bf16"
    assert _resolve_precision("bf16", torch.device("cuda")) == "bf16"
    assert _resolve_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError, match="Precision"):
        _resolve_precision("fp16", torch.device("cuda"))


def test_nonfinite_gradients_abort_with_param_names():
    model = nn.Linear(3, 2)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    model.bias.grad = torch.ones_like(model.bias)
    with pytest.raises(FloatingPointError, match="weight"):
        _clip_checked_gradients(model)


def test_valid_gradient_updates_adam_state_without_scaler():
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(4, 3)).square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = _clip_checked_gradients(model)
    scaler.step(optimizer)
    scaler.update()
    assert norm > 0
    assert len(optimizer.state) > 0
    assert scaler.state_dict() == {}


def test_zero_gradients_are_rejected():
    model = nn.Linear(3, 2)
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)
    with pytest.raises(FloatingPointError, match="zero"):
        _clip_checked_gradients(model)
