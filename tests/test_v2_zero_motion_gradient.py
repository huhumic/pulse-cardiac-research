"""Regression: V2 zero-initialized motion offsets must not poison gradients.

The original sqrt(sum(offset**2)) had an infinite derivative at offsets=0.
Both FP32 and BF16 training could silently skip every AdamW update.
"""
import pytest
import torch
import torch.nn.functional as F

from pulse_repro.cine_v2 import BidirectionalEvidenceFusion


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_zero_motion_magnitude_has_finite_zero_subgradient(dtype):
    offsets = torch.zeros(1, 2, 6, 7, dtype=dtype, requires_grad=True)
    magnitude = torch.linalg.vector_norm(
        offsets, ord=2, dim=1, keepdim=True
    ) / 3.0
    assert torch.count_nonzero(magnitude).item() == 0
    magnitude.sum().backward()
    assert offsets.grad is not None
    assert bool(torch.isfinite(offsets.grad).all())
    assert torch.count_nonzero(offsets.grad).item() == 0


def test_initial_zero_motion_fusion_backward_and_adam_step_are_finite():
    torch.manual_seed(23)
    model = BidirectionalEvidenceFusion(
        decoder_channels=8, temporal_channels=8
    ).train()
    assert torch.count_nonzero(model.align[-1].weight).item() == 0
    assert torch.count_nonzero(model.align[-1].bias).item() == 0

    source = torch.rand(2, 1, 24, 24)
    neighbors = torch.rand(2, 2, 24, 24)
    phase = torch.tensor([[0., 1., 1./30.]], dtype=torch.float32).repeat(2, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    residual, aux = model(source, neighbors, (8, 8), phase)
    assert torch.count_nonzero(residual).item() == 0
    target = torch.randint(0, 4, (2, 8, 8))
    loss = (
        F.cross_entropy(aux["probe_minus"], target)
        + F.cross_entropy(aux["probe_plus"], target)
        + aux["weights"][:, 1:].square().mean()
    )
    loss.backward()

    checked = ["align.2.weight", "align.2.bias",
               "encoder.net.0.weight", "direction_embed"]
    named = dict(model.named_parameters())
    for name in checked:
        grad = named[name].grad
        assert grad is not None, name
        assert bool(torch.isfinite(grad).all()), name

    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1., error_if_nonfinite=True
    )
    assert bool(torch.isfinite(norm))
    assert float(norm) > 0.
    optimizer.step()
    assert len(optimizer.state) > 0
