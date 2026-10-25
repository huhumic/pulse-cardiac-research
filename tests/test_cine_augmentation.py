"""Regression tests: real cardiac-time images share PULSE spatial augmentations.

Albumentations GaussNoise uses a channel-shaped noise tensor derived from its
primary HWC3 image. Supplying temporal images as HW previously failed inside
albucore with a broadcast ValueError.
"""
import numpy as np
import pytest

from pulse_repro.cine_data import augment_cine_sample


def _inputs():
    yy, xx = np.mgrid[:32, :40]
    x = np.stack((xx, yy, xx + yy), axis=-1).astype(np.float32) / 80.
    minus = (xx + 2 * yy).astype(np.float32) / 120.
    plus = (2 * xx - yy).astype(np.float32) / 120.
    gt = ((xx + yy) % 4).astype(np.uint8)
    return x, gt, minus, plus


def test_channel_compatible_targets_and_aligned_geometric_change():
    x, gt, minus, plus = _inputs()

    class ChannelSensitiveGeometricNoise:
        def __call__(self, image, mask, temporal_minus, temporal_plus):
            assert image.shape == temporal_minus.shape == temporal_plus.shape
            assert image.shape[-1] == 3
            # Simulate the exact channel-shaped addition that failed for HW.
            noise = np.arange(image.size, dtype=np.float32).reshape(image.shape) * 1e-6
            return {
                "image": image[:, ::-1, :] + noise,
                "mask": mask[:, ::-1],
                "temporal_minus": temporal_minus[:, ::-1, :] + noise,
                "temporal_plus": temporal_plus[:, ::-1, :] + noise,
            }

    xo, go, mo, po = augment_cine_sample(
        ChannelSensitiveGeometricNoise(), x, gt, minus, plus
    )
    assert xo.shape == x.shape
    assert go.shape == gt.shape
    assert mo.shape == po.shape == minus.shape
    np.testing.assert_array_equal(go, gt[:, ::-1])
    noise = np.arange(x.size, dtype=np.float32).reshape(x.shape) * 1e-6
    np.testing.assert_allclose(mo, minus[:, ::-1] + noise[..., 1], atol=1e-7)
    np.testing.assert_allclose(po, plus[:, ::-1] + noise[..., 1], atol=1e-7)


def test_incompatible_sample_shape_fails_early():
    x, gt, minus, plus = _inputs()
    with pytest.raises(ValueError, match="must match"):
        augment_cine_sample(None, x, gt, minus[:, 1:], plus)


def test_actual_albumentations_gaussnoise_and_flip():
    A = pytest.importorskip("albumentations")
    x, gt, minus, plus = _inputs()
    aug = A.Compose(
        [A.HorizontalFlip(p=1.0), A.GaussNoise(p=1.0)],
        additional_targets={"temporal_minus": "image",
                            "temporal_plus": "image"},
    )
    xo, go, mo, po = augment_cine_sample(aug, x, gt, minus, plus)
    assert xo.shape == x.shape
    assert mo.shape == po.shape == minus.shape
    np.testing.assert_array_equal(go, gt[:, ::-1])
    assert np.isfinite(xo).all()
    assert np.isfinite(mo).all() and np.isfinite(po).all()
