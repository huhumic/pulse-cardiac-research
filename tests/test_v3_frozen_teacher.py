"""CPU-only tests for frozen-teacher pseudolabel flow geometry, no checkpoints/GPU."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse_repro.v3_frozen_teacher import (
    TEACHER_SHAPE, TeacherAssets, propagate_endpoint_labels,
)


def arrays():
    label0 = torch.zeros((1, 1, *TEACHER_SHAPE), dtype=torch.long)
    label1 = label0.clone()
    valid = torch.zeros((1, 1, *TEACHER_SHAPE), dtype=torch.bool)
    valid[:, :, :, :, 11:21] = True
    flow0 = torch.zeros((1, 3, *TEACHER_SHAPE), dtype=torch.float32)
    flow1 = flow0.clone()
    return label0, label1, valid, flow0, flow1


def test_identity_propagation_is_native_padding_aware():
    a, b, valid, f0, f1 = arrays()
    a[0, 0, 32, 40, 14] = 2
    b[0, 0, 32, 40, 14] = 2
    out = propagate_endpoint_labels(a, b, f0, f1, valid)
    assert out.agreement.sum().item() == 128 * 128 * 10
    assert out.hard_pseudo[0, 0, 32, 40, 14].item() == 2
    assert out.hard_pseudo[0, 0, 32, 40, 0].item() == 255
    assert out.summary()["agreement_fraction"] == pytest.approx(10/32)
    assert out.summary()["agreement_among_both_valid"] == pytest.approx(1.0)


def test_true_voxel_disagreement_excluded():
    a, b, valid, f0, f1 = arrays()
    a[0, 0, 32, 40, 14] = 1
    b[0, 0, 32, 40, 14] = 3
    out = propagate_endpoint_labels(a, b, f0, f1, valid)
    assert out.ed[0, 0, 32, 40, 14] == 1
    assert out.es[0, 0, 32, 40, 14] == 3
    assert not out.agreement[0, 0, 32, 40, 14]
    assert out.hard_pseudo[0, 0, 32, 40, 14] == 255


def test_source_sampling_direction_and_padded_region_excluded():
    a, b, valid, f0, f1 = arrays()
    a[0, 0, 20, 20, 15] = 3
    b[0, 0, 20, 20, 15] = 3
    f0[:, 0] = 1.0  # output axis-0 index 19 samples source index 20
    f1[:, 0] = 1.0
    out = propagate_endpoint_labels(a, b, f0, f1, valid)
    assert out.hard_pseudo[0, 0, 19, 20, 15] == 3
    # Continuous sample inside tensor boundaries but in synthetic depth padding
    f0[:, 2] = -10.0
    f1[:, 2] = -10.0
    out = propagate_endpoint_labels(a, b, f0, f1, valid)
    assert not out.ed_valid[0, 0, 19, 20, 15]


def test_fractional_out_of_bounds_rejected_even_if_nearest_samples_border():
    a, b, valid, f0, f1 = arrays()
    f0[:, 0] = -0.2
    f1[:, 0] = -0.2
    out = propagate_endpoint_labels(a, b, f0, f1, valid)
    assert not out.ed_valid[0, 0, 0, 0, 15]
    assert out.ed_valid[0, 0, 1, 0, 15]
    assert out.hard_pseudo[0, 0, 0, 0, 15] == 255


def test_nan_flow_and_label_class_fail_closed():
    a, b, valid, f0, f1 = arrays()
    f0[0, 0, 10, 10, 15] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        propagate_endpoint_labels(a, b, f0, f1, valid)
    f0.zero_()
    a[0, 0, 10, 10, 15] = 4
    with pytest.raises(ValueError, match="0..3"):
        propagate_endpoint_labels(a, b, f0, f1, valid)


def test_file_audit_refuses_wrong_manifest_reference(tmp_path):
    assets = TeacherAssets(tmp_path)
    with pytest.raises(FileNotFoundError):
        assets.audit_file_provenance()
    for f in (assets.front, assets.refiner):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"placeholder")
    assets.manifest.parent.mkdir(parents=True, exist_ok=True)
    assets.manifest.write_text(
        '{"dataset":"acdc","structural_variant":"full",'
        '"fr_checkpoint":"checkpoints/wrong_front.pth",'
        '"variant_config":{"use_relation":true,"use_warp":true,"state_conditioned":true}}'
    )
    with pytest.raises(ValueError, match="does not reference"):
        assets.audit_file_provenance()
