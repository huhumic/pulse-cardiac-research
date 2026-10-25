"""CPU-only regression tests for full-volume V1.5/V2 cine inference.

These tests use fake logits; they do not claim pretrained checkpoint accuracy.
Run: python -m pytest tests/test_evaluate_cine_v2.py -q
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pulse_repro import evaluate_cine_v2 as cine_eval
from pulse_repro.metrics import measure_binary


class AlignmentAssertingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, spatial, temporal, phase):
        assert spatial.shape == (1, 3, 256, 256)
        assert temporal.shape == (1, 2, 256, 256)
        assert phase.shape == (1, 3)
        # All cine timepoints match the target in this synthetic test.
        # This assertion fails if TTA flips spatial and temporal independently.
        assert torch.allclose(temporal[:, 0], spatial[:, 1], atol=1e-5)
        assert torch.allclose(temporal[:, 1], spatial[:, 1], atol=1e-5)
        self.calls += 1
        logits = torch.full((1, 4, 256, 256), -12.0)
        logits[:, 1] = torch.where(spatial[:, 1] > 0, 12.0, -12.0)
        logits[:, 2] = torch.where(spatial[:, 1] <= 0, 12.0, -12.0)
        return logits, None, None


class DirectionRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, spatial, temporal, phase):
        self.received.append(temporal[:, :, 0, 0].cpu().tolist()[0])
        logits = torch.zeros((1, 4, 256, 256))
        logits[:, 1] = 2.0
        return logits, None, None


def test_fp32_context_never_uses_implicit_fp16():
    device = torch.device("cpu")
    with cine_eval._amp_context(device, "fp32"):
        value = torch.tensor([2.0]).square()
    assert value.dtype == torch.float32
    with pytest.raises(RuntimeError, match="BF16"):
        cine_eval._amp_context(device, "bf16")
    with pytest.raises(ValueError, match="precision"):
        cine_eval._amp_context(device, "fp16")


def test_real_cine_3d_reassembly_and_synchronized_four_flip_tta(monkeypatch):
    monkeypatch.setattr(
        cine_eval, "upstream",
        lambda _: (SimpleNamespace(postprocess=lambda mask: mask), None),
    )
    # Different 3D slices with spatial intensity variation, but the same
    # raw frame repeated in time; 4-flip TTA must preserve temporal alignment.
    h, w, z, t = 9, 13, 3, 5
    yy, xx = np.mgrid[:h, :w]
    volume = np.stack([xx.astype(np.float32) - 6 + 0.2 * k
                       + 0.15 * yy for k in range(z)], axis=-1)
    cine = np.repeat(volume[..., None], t, axis=-1)
    model = AlignmentAssertingModel()
    pred = cine_eval.infer_volume(
        model, volume, cine, frame=2, ed_frame=2, pulse_root=Path("."),
        device=torch.device("cpu"), tta=True, precision="fp32"
    )
    assert model.calls == z * 4
    assert pred.shape == (h, w, z)
    assert pred.dtype == np.uint8
    assert set(np.unique(pred)) == {1, 2}


@pytest.mark.parametrize(
    ("control", "expected"),
    [
        ("correct", [1.0, 2.0]),
        ("swap", [2.0, 1.0]),
        ("repeat-target", [7.0, 7.0]),
    ],
)
def test_temporal_control_changes_only_neighbor_inputs(monkeypatch, control, expected):
    monkeypatch.setattr(
        cine_eval, "upstream",
        lambda _: (SimpleNamespace(postprocess=lambda mask: mask), None),
    )

    def artificial_inputs(*args, **kwargs):
        spatial = np.zeros((3, 256, 256), dtype=np.float32)
        spatial[1] = 7.0
        temporal = np.stack([
            np.full((256, 256), 1.0, np.float32),
            np.full((256, 256), 2.0, np.float32),
        ])
        phase = np.array([0, 1, 0.2], dtype=np.float32)
        return spatial, temporal, phase

    monkeypatch.setattr(cine_eval, "build_temporal_input", artificial_inputs)
    model = DirectionRecordingModel()
    volume = np.zeros((6, 7, 2), dtype=np.float32)
    cine = np.zeros((6, 7, 2, 5), dtype=np.float32)
    result = cine_eval.infer_volume(
        model, volume, cine, frame=1, ed_frame=1, pulse_root=Path("."),
        device=torch.device("cpu"), tta=False, temporal_control=control,
        precision="fp32",
    )
    assert result.shape == volume.shape
    assert model.received == [expected, expected]


def test_invalid_3d_cine_shapes_or_frame_ids_fail_before_inference():
    model = DirectionRecordingModel()
    volume = np.zeros((5, 6, 2), dtype=np.float32)
    cine = np.zeros((5, 6, 2, 4), dtype=np.float32)
    kw = dict(model=model, volume=volume, cine=cine,
              frame=1, ed_frame=1, pulse_root=Path("."),
              device=torch.device("cpu"))
    with pytest.raises(ValueError, match="aligned"):
        cine_eval.infer_volume(**{**kw, "cine": cine[:4]})
    with pytest.raises(ValueError, match="1-based"):
        cine_eval.infer_volume(**{**kw, "frame": 0})
    with pytest.raises(ValueError, match="temporal_control"):
        cine_eval.infer_volume(**{**kw, "temporal_control": "future-gt"})


def test_volume_dice_matches_3d_voxel_formula_and_empty_policy():
    gt = np.zeros((5, 5, 3), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[1:3, 1:3, 0:2] = 1   # 8 voxels
    pred[1:3, 1:3, 0] = 1    # 4 intersecting voxels
    result = measure_binary(gt == 1, pred == 1, (1.0, 1.0, 1.0))
    assert result["dice"] == pytest.approx(2 * 4 / (8 + 4))
    absent = measure_binary(gt == 3, pred == 3, (1.0, 1.0, 1.0))
    assert np.isnan(absent["dice"])
