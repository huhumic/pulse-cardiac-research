"""Unit tests avoid Hugging Face downloads and original PULSE training data."""
import copy
import math

import pytest
import torch

# Zero-initialized residuals are mathematically identity, but forward kernels
# can differ by several FP32 ULPs across CPU/GPU. Require numerical equality.
import torch.nn as nn
import torch.nn.functional as F

from pulse_repro.selective_v1 import (
    MultiscaleSelectiveEnhancement, SelectiveDPTDecoder, PULSESelectiveV1
)


class MiniDecoder(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.out_size = 64
        self.projs = nn.ModuleList(nn.Linear(channels, channels) for _ in range(4))
        self.res = nn.ModuleList(nn.Identity() for _ in range(4))
        self.ups = nn.ModuleList(nn.Upsample(scale_factor=2, mode="nearest") for _ in range(4))
        self.head = nn.Conv2d(channels, 4, 1)
        self.aux1 = nn.Conv2d(channels, 4, 1)
        self.aux2 = nn.Conv2d(channels, 4, 1)

    def _reassemble(self, hidden, proj):
        t = hidden[:, 1:]
        b, n, _ = t.shape
        hw = math.isqrt(n)
        assert hw * hw == n
        return proj(t).transpose(1, 2).reshape(b, -1, hw, hw)

    def forward(self, hs):
        f = [self._reassemble(h, self.projs[i]) for i, h in enumerate(hs)]
        x = self.ups[3](self.res[3](f[3]))
        x = self.res[2](x + F.interpolate(f[2], size=x.shape[-2:],
                                          mode="bilinear", align_corners=False))
        x = self.ups[2](x)
        a1 = self.aux1(x)
        x = self.res[1](x + F.interpolate(f[1], size=x.shape[-2:],
                                          mode="bilinear", align_corners=False))
        x = self.ups[1](x)
        a2 = self.aux2(x)
        x = self.res[0](x + F.interpolate(f[0], size=x.shape[-2:],
                                          mode="bilinear", align_corners=False))
        x = self.ups[0](x)
        return F.interpolate(self.head(x), (self.out_size,)*2, mode="bilinear",
                             align_corners=False), a1, a2


class MiniPULSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 8)
        self.decoder = MiniDecoder(8)

    def forward(self, x):
        # spatial encoder stub with PULSE-style 4 hidden states
        flat = F.adaptive_avg_pool2d(x, (4, 4)).flatten(2).transpose(1, 2)
        features = self.proj(flat)
        cls = torch.zeros(x.shape[0], 1, 8, device=x.device)
        hs = [torch.cat((cls, features + float(i)*0.01), dim=1)
              for i in range(4)]
        return self.decoder(hs)


def test_zero_init_preserves_exact_decoder_outputs():
    torch.manual_seed(5)
    core = MiniDecoder()
    frozen = copy.deepcopy(core)
    adapter = SelectiveDPTDecoder(core, channels=8)
    states = [torch.randn(2, 17, 8) for _ in range(4)]
    with torch.no_grad():
        expected = frozen(states)
        actual = adapter(states)
    for ref, got in zip(expected, actual):
        torch.testing.assert_close(ref, got, atol=1e-6, rtol=1e-5)


def test_zero_update_and_shape_check():
    mod = MultiscaleSelectiveEnhancement(8, 4)
    x = torch.randn(2, 8, 32, 32)
    s = torch.randn_like(x)
    assert torch.count_nonzero(mod(x, s)) == 0
    with pytest.raises(ValueError, match="must match"):
        mod(x, s[..., :-1])


def test_frozen_base_adapter_receives_gradients_and_changes_prediction():
    torch.manual_seed(4)
    original = MiniPULSE()
    base = copy.deepcopy(original)
    variant = PULSESelectiveV1(base)
    x = torch.randn(2, 3, 16, 16)
    variant.train()
    # frozen original BN/statistics stay in eval; adapters remain trainable
    assert not variant.base.training
    assert variant.base.decoder.enhancers.training
    with torch.no_grad():
        reference = original.eval()(x)[0]
        first = variant.eval()(x)[0]
    torch.testing.assert_close(reference, first, atol=1e-6, rtol=1e-5)
    variant.train()
    opt = torch.optim.SGD(variant.adapter_parameters(), lr=0.05)
    target = torch.randint(0, 4, (2, 64, 64))
    opt.zero_grad()
    logits, _, _ = variant(x)
    F.cross_entropy(logits, target).backward()
    assert all(p.grad is None for p in variant.base.decoder.core.parameters())
    grads = [m.update.weight.grad for m in variant.base.decoder.enhancers]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    opt.step()
    with torch.no_grad():
        updated = variant.eval()(x)[0]
    assert not torch.equal(first, updated)


def test_original_model_does_not_train_when_v1_train():
    model = PULSESelectiveV1(MiniPULSE()).train()
    assert all(not p.requires_grad for p in model.base.decoder.core.parameters())
    assert all(p.requires_grad for p in model.base.decoder.enhancers.parameters())


def test_256_channel_enhancers_remain_lightweight():
    decoder = SelectiveDPTDecoder(MiniDecoder(8), channels=8)
    assert sum(p.numel() for p in decoder.enhancers.parameters()) > 0
    big = MultiscaleSelectiveEnhancement(256, 64)
    assert sum(p.numel() for p in big.parameters()) < 60000
