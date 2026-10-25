"""CPU-only tests: legacy UVI ACDC crop/pad, label geometry and phase guards."""
from __future__ import annotations

import numpy as np
import pytest

from pulse_repro.v3_uvi_acdc_adapter import (
    TEACHER_SHAPE, UVIGrid, ed_to_es_tau,
    prepare_endpoints, prepare_target_frame, model_endpoints, split_role,
    read_info_cfg, validate_already_uvi,
)


def test_patient001_exact_legacy_center_crop_and_depth_padding():
    grid = UVIGrid.for_native_shape((216, 256, 10))
    assert (grid.start0, grid.start1, grid.start2) == (44, 64, 0)
    assert (grid.pad2_before, grid.pad2_after) == (11, 11)
    volume = np.zeros((216, 256, 10), dtype=np.uint8)
    volume[100, 101, 5] = 3
    mapped = grid.forward(volume)
    assert mapped.shape == TEACHER_SHAPE
    assert mapped[56, 37, 16] == 3
    assert mapped.sum() == 3
    inv = grid.inverse(mapped)
    assert inv.shape == volume.shape
    np.testing.assert_array_equal(inv, grid.native_valid() * volume)
    assert grid.teacher_valid().sum() == 128 * 128 * 10
    assert grid.native_valid().sum() == 128 * 128 * 10
    assert not grid.native_valid()[0, 0, 5]
    assert not grid.teacher_valid()[56, 37, 0]


def test_odd_padding_and_depth_crop_inverse():
    odd = UVIGrid.for_native_shape((129, 129, 9))
    assert (odd.pad2_before, odd.pad2_after) == (11, 12)
    vol = np.ones((129, 129, 9), dtype=np.float32)
    mapped = odd.forward(vol)
    assert mapped[:, :, :11].sum() == 0
    assert mapped[:, :, 19:20].sum() == 128*128
    assert mapped[:, :, 21:].sum() == 0
    large = UVIGrid.for_native_shape((140, 144, 40))
    assert large.start2 == 4
    a = np.random.default_rng(7).integers(0, 4, size=(140, 144, 40), dtype=np.uint8)
    b = large.forward(a)
    np.testing.assert_array_equal(
        b, a[6:134, 8:136, 4:36]
    )
    np.testing.assert_array_equal(
        large.inverse(b)[6:134, 8:136, 4:36], b
    )


def test_uvi_original_scales_after_crop_and_padding_preserving_labels():
    shape = (132, 134, 10)
    img0 = np.zeros(shape, dtype=np.float64)
    img0[2:130, 3:131, :] = 10.0
    img0[50, 60, 4] = 20.0
    img1 = img0 * 4 + 8
    y0 = np.zeros(shape, dtype=np.uint8)
    y1 = np.zeros(shape, dtype=np.uint8)
    y0[50, 60, 4] = 3
    y1[90, 120, 8] = 2
    pair = prepare_endpoints(img0, img1, y0, y1, mode="raw_uvi")
    assert pair.ed_image.dtype == pair.es_image.dtype == np.float32
    assert pair.ed_image.min() == 0 and pair.ed_image.max() == 1
    assert pair.ed_image.shape == pair.ed_label.shape == TEACHER_SHAPE
    assert set(np.unique(pair.ed_label)) == {0, 3}
    assert set(np.unique(pair.es_label)) == {0, 2}
    assert pair.ed_label[48, 57, 15] == 3
    assert pair.es_label[88, 117, 19] == 2
    assert np.all(pair.ed_label[:, :, :11] == 0)
    assert not pair.valid_spatial[:, :, :11].any()


def test_already_uvi_never_double_scales_or_crops():
    image = np.linspace(.2, .8, num=np.prod(TEACHER_SHAPE),
                        dtype=np.float32).reshape(TEACHER_SHAPE)
    label = np.zeros(TEACHER_SHAPE, dtype=np.uint8)
    label[20, 23, 5] = 3
    pair = prepare_endpoints(image, image, label, label, mode="already_uvi")
    np.testing.assert_array_equal(pair.ed_image, image)
    np.testing.assert_array_equal(pair.ed_label, label)
    assert pair.grid is None
    assert pair.valid_spatial.all()
    with pytest.raises(ValueError, match="Preprocessed image"):
        validate_already_uvi(image[:127])
    with pytest.raises(ValueError, match="shape"):
        prepare_endpoints(image, image, label[:127], label, mode="already_uvi")
    with pytest.raises(ValueError, match="Mode"):
        prepare_endpoints(image, image, label, label, mode="auto")


def test_correct_one_based_time_arc_and_refuse_other_arc(tmp_path):
    cfg = tmp_path / "Info.cfg"
    cfg.write_text("ED: 1\nES: 12\nNbFrame: 30\nGroup: DCM\n")
    assert read_info_cfg(cfg) == {"ED": 1, "ES": 12, "NbFrame": 30}
    assert ed_to_es_tau(1, 12, 1, 30) == 0
    assert ed_to_es_tau(1, 12, 12, 30) == 1
    assert ed_to_es_tau(1, 12, 6, 30) == pytest.approx(5/11)
    with pytest.raises(ValueError, match="not on forward"):
        ed_to_es_tau(1, 12, 22, 30)
    assert ed_to_es_tau(27, 4, 1, 30) == pytest.approx(4/7)


def test_fail_closed_on_corrupted_labels_and_dimensions():
    grid = UVIGrid.for_native_shape((128, 128, 32))
    bad = np.zeros(TEACHER_SHAPE, dtype=np.float32)
    bad[2, 3, 1] = 4
    with pytest.raises(ValueError, match="labels"):
        prepare_endpoints(np.indices(TEACHER_SHAPE, dtype=np.float32)[0],
                          np.indices(TEACHER_SHAPE, dtype=np.float32)[0]+1, bad, bad,
                          mode="raw_uvi")
    assert grid.start0 == 0 and grid.start1 == 0 and grid.start2 == 0
    with pytest.raises(ValueError, match="Unsupported native shape"):
        UVIGrid.for_native_shape((100, 256, 10))



def test_target_frame_matches_endpoint_grid_and_split_guard():
    source_shape = (128, 128, 10)
    grid = UVIGrid.for_native_shape(source_shape)
    raw = np.zeros(source_shape, dtype=np.float32)
    raw[10, 20, 3] = 5.
    labels = np.zeros(source_shape, dtype=np.uint8)
    pair = prepare_endpoints(raw, raw, labels, labels, mode="raw_uvi")
    x = model_endpoints(pair)
    assert x.shape == (1, 2, *TEACHER_SHAPE)
    target = prepare_target_frame(raw, mode="raw_uvi", grid=grid)
    np.testing.assert_array_equal(target, pair.ed_image)
    already = np.full(TEACHER_SHAPE, .37, dtype=np.float32)
    np.testing.assert_array_equal(
        prepare_target_frame(already, mode="already_uvi"), already
    )
    with pytest.raises(ValueError, match="requires"):
        prepare_target_frame(raw, mode="raw_uvi")
    assert split_role("patient001") == "train"
    assert split_role("patient090") == "train"
    assert split_role("patient091") == "validation"
    assert split_role("patient100") == "validation"
    with pytest.raises(ValueError):
        split_role("patient101")
