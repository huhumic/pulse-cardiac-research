"""V3 G2 CPU contracts for 3D teacher -> native -> 2D PULSE segmentation."""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from pulse_repro.v3_uvi_acdc_adapter import UVIGrid, TEACHER_SHAPE
from pulse_repro.v3_student_adapter import (
    NativePseudo, load_native_pseudo, prepare_training_slice, pulse_slice_pseudo,
    _verified_grid,
)
from pulse_repro.v3_pseudo_cache import class_balanced_pseudolabel_ce


def fake_native(shape=(192, 208, 3)):
    grid = UVIGrid.for_native_shape(shape)
    raw_pseudo = np.full(TEACHER_SHAPE, 255, dtype=np.uint8)
    binary = np.zeros(TEACHER_SHAPE, dtype=np.uint8)
    image_weights = np.zeros(TEACHER_SHAPE, dtype=np.float16)
    # Source native z=1 -> pad UVI z=15 (pad_before = 14 for native depth 3)
    z = grid.pad2_before + 1
    raw_pseudo[40:80, 40:80, z] = 1
    binary[40:80, 40:80, z] = 1
    image_weights[40:80, 40:80, z] = 0.5
    native_pseudo = grid.inverse(raw_pseudo, fill=255)
    native_binary = grid.inverse(binary, fill=0)
    native_weight = grid.inverse(image_weights, fill=0).astype(np.float32)
    manifest = {"patient": "patient001"}
    cache = NativePseudo("patient001", 6, 1, 12, grid,
                         native_pseudo, native_weight,
                         native_binary, native_weight, manifest)
    return cache


def test_teacher_inverse_restores_native_slice_axis_without_flip_or_fake_background():
    cache = fake_native()
    grid = cache.grid
    assert grid.native_shape == (192, 208, 3)
    assert (grid.start0, grid.start1, grid.pad2_before, grid.pad2_after) == (32, 40, 14, 15)
    assert cache.pseudo[grid.start0+40, grid.start1+40, 1] == 1
    assert cache.pseudo[0, 0, 1] == 255
    assert cache.pseudo[grid.start0+40, grid.start1+40, 0] == 255
    assert cache.weights[0,0,1] == 0
    assert cache.weights[grid.start0+40,grid.start1+40,1] == .5


def test_pulse_256_resize_preserves_ignore_and_original_native_xy():
    cache = fake_native()
    label, weight = pulse_slice_pseudo(cache, 1, size=256, policy="image")
    assert label.shape == (256,256)
    assert label.dtype == np.uint8 and weight.dtype == np.float32
    assert np.any(label == 1)
    assert np.all(weight[label == 255] == 0)
    assert np.any(label == 255)
    assert np.all(weight[label == 1] == 0.5)
    base_label, base_confidence = pulse_slice_pseudo(cache, 1, size=256, policy="binary")
    assert np.array_equal(base_label, label)
    assert np.array_equal(base_confidence > 0, weight > 0)
    _, empty = pulse_slice_pseudo(cache, 0, size=256, policy="binary")
    assert not np.any(empty)


def test_true_cine_time_and_spatial_slice_are_distinct_model_inputs():
    cache = fake_native()
    rng = np.random.default_rng(42)
    cine = rng.normal(size=(*cache.grid.native_shape,30)).astype(np.float32)
    sample = prepare_training_slice(cache, cine, z=1, size=256, policy="binary")
    assert sample["img"].shape == (3,256,256)
    assert sample["temporal"].shape == (2,256,256)
    assert sample["phase_context"].shape == (3,)
    assert sample["pseudo"].shape == sample["pseudo_weight"].shape == (256,256)
    assert sample["valid_count"] > 0 and sample["foreground_count"] > 0
    assert not torch.equal(sample["temporal"][0],sample["temporal"][1])
    assert torch.all(sample["pseudo_weight"][sample["pseudo"]==255] == 0)
    logits = torch.randn(1,4,256,256,requires_grad=True)
    loss = class_balanced_pseudolabel_ce(
        logits,sample["pseudo"][None,None],sample["pseudo_weight"][None,None])
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(logits.grad).all().item()


def test_90_10_validation_metadata_blocked():
    cache = fake_native()
    bad = cache.grid.as_dict()
    bad["axis0_crop_start"] += 1
    with pytest.raises(ValueError, match="grid mapping"):
        _verified_grid(bad)


def test_loader_rejects_corrupt_hash_and_metadata(tmp_path):
    grid = UVIGrid.for_native_shape((128,128,3))
    measured = grid.teacher_valid()
    label = np.full(TEACHER_SHAPE,255,dtype=np.uint8)
    good = (np.indices(TEACHER_SHAPE)[0] < 8) & measured
    label[good] = 2
    binary = good.astype(np.uint8)
    weighted = np.where(good,.4,0).astype(np.float16)
    out_dir = tmp_path/"patient001"
    out_dir.mkdir()
    npz = out_dir/"patient001_t06.npz"
    np.savez_compressed(
        npz,
        pseudo_uvi=label,
        weight_image_uvi=weighted,
        binary_consensus_uvi=binary,
        valid_both_uvi=measured.astype(np.uint8),
        measured_uvi=measured.astype(np.uint8),
        frame_uvi=np.zeros(TEACHER_SHAPE,dtype=np.float16),
    )
    digest=hashlib.sha256(npz.read_bytes()).hexdigest()
    meta={
        "schema":"v3_acdc_candidate_pseudolabel_cache_v1",
        "split":"train_001_090_only", "patient":"patient001",
        "ed_1based":1,"es_1based":12,"target_1based":6,
        "native_grid_mapping":grid.as_dict(),
        "mode":"raw_uvi",
        "flow_semantics":"target_grid_pullback_voxel_offsets_native_nifti_axis012",
        "pseudo_policy":"strict_dual_endpoint_nn_agreement_ignore255",
        "weight_policy":{"policy":"classwise_median_normalized_registration_residual_softweight_v1"},
        "npz_sha256":digest,
    }
    jsonpath=npz.with_suffix(".json")
    jsonpath.write_text(json.dumps(meta))
    out=load_native_pseudo(npz)
    assert out.pseudo.shape == grid.native_shape
    assert out.pseudo[0,0,0]==2
    assert out.pseudo[20,20,0]==255
    assert out.weights[0,0,0]==pytest.approx(.4,abs=1e-3)
    meta["split"]="validation"
    jsonpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="training-only"):
        load_native_pseudo(npz)
    meta["split"]="train_001_090_only"
    meta["npz_sha256"]="bad"
    jsonpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="hash"):
        load_native_pseudo(npz)
