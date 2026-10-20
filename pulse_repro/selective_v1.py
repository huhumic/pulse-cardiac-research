"""Selective structure enhancement for the *existing* PULSE 2.5D decoder.

Research variant V1, inspired by multi-scale context and feature-edge design in
CENet. Does NOT copy CENet code. Preserves PULSE encoder, decoder heads,
spatial-slice input and checkpoint semantics. It adds zero-initialized residual
updates at the two higher-resolution feature-fusion levels.

This is *spatial* anatomy enhancement. It is NOT temporal phase modeling.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class MultiscaleSelectiveEnhancement(nn.Module):
    """Use decoder state to gate multi-scale skip-edge and dilated context cues."""

    def __init__(self, channels: int = 256, hidden: int = 64):
        super().__init__()
        if channels <= 0 or hidden <= 0:
            raise ValueError("channels and hidden must be positive")
        groups = math.gcd(channels, 16)
        self.edge_depthwise = nn.Conv2d(
            2 * channels, 2 * channels, 3, padding=1, groups=2 * channels,
            bias=False
        )
        self.edge_project = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels), nn.GELU()
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1),
            nn.GELU()
        )
        self.state_gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )
        self.update = nn.Conv2d(channels, channels, 1)
        # Identity at initialization: official PULSE prediction is unchanged.
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def forward(self, decoder_state: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        if decoder_state.shape != skip.shape or decoder_state.ndim != 4:
            raise ValueError(
                f"decoder state and skip must match BCHW, got "
                f"{decoder_state.shape} and {skip.shape}"
            )
        # Detail across two image-space scales, from *feature maps*, not GT.
        edge3 = skip - F.avg_pool2d(skip, 3, stride=1, padding=1,
                                    count_include_pad=False)
        edge5 = skip - F.avg_pool2d(skip, 5, stride=1, padding=2,
                                    count_include_pad=False)
        edge = self.edge_project(
            self.edge_depthwise(torch.cat((edge3, edge5), dim=1))
        )
        context = self.context(decoder_state)
        gate = self.state_gate(torch.cat((decoder_state, skip), dim=1))
        return self.update(gate * (edge + context))


class SelectiveDPTDecoder(nn.Module):
    """Wrap PULSE DPT decoder with two gated residual skip corrections.

    Explicitly matches the original PULSE decoding order; the frozen upstream
    code is never overwritten. Auxiliary logits retain the original locations.
    """

    def __init__(self, core: nn.Module, channels: int | None = None):
        super().__init__()
        self.core = core
        if channels is None:
            proj = core.projs[0]
            linear = next((m for m in proj.modules() if isinstance(m, nn.Linear)), None)
            if linear is None:
                raise TypeError("Cannot infer PULSE DPT projection channel count")
            channels = linear.out_features
        self.enhancers = nn.ModuleList([
            MultiscaleSelectiveEnhancement(channels),
            MultiscaleSelectiveEnhancement(channels),
        ])

    def forward(self, hs):
        core = self.core
        f = [core._reassemble(h, core.projs[i]) for i, h in enumerate(hs)]
        x = core.res[3](f[3])
        x = core.ups[3](x)

        skip2 = F.interpolate(f[2], size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = x + skip2
        x = core.res[2](x)
        x = core.ups[2](x)
        a1 = core.aux1(x)

        skip1 = F.interpolate(f[1], size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = x + skip1 + self.enhancers[0](x, skip1)
        x = core.res[1](x)
        x = core.ups[1](x)
        a2 = core.aux2(x)

        skip0 = F.interpolate(f[0], size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = x + skip0 + self.enhancers[1](x, skip0)
        x = core.res[0](x)
        x = core.ups[0](x)
        x = core.head(x)
        main = F.interpolate(
            x, (core.out_size, core.out_size), mode="bilinear",
            align_corners=False
        )
        return main, a1, a2


class PULSESelectiveV1(nn.Module):
    """Adapters only are trainable; pretrained PULSE remains frozen."""

    def __init__(self, pretrained_pulse: nn.Module):
        super().__init__()
        if not hasattr(pretrained_pulse.decoder, "projs"):
            raise TypeError("Expected original PULSE DPTDecoderDS")
        self.base = pretrained_pulse
        self.base.decoder = SelectiveDPTDecoder(self.base.decoder)
        self.freeze_official_model()

    def freeze_official_model(self):
        for param in self.base.parameters():
            param.requires_grad_(False)
        for param in self.base.decoder.enhancers.parameters():
            param.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Prevent BN running-stat drift in the frozen PULSE decoder.
        self.base.eval()
        self.base.decoder.enhancers.train(mode)
        return self

    def forward(self, x):
        return self.base(x)

    def adapter_parameters(self):
        return self.base.decoder.enhancers.parameters()


def load_variant(pulse_root: Path, checkpoint: Path, device: torch.device):
    """Build the architecture and load one trained V1 checkpoint."""
    from .runtime import upstream
    _, source = upstream(pulse_root)
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if raw.get("variant") != "selective-v1":
        raise ValueError("Expected a selective-v1 checkpoint, not official PULSE")
    base = source.PULSESeg(img_size=256)
    model = PULSESelectiveV1(base)
    model.load_state_dict(raw["model"], strict=True)
    model.to(device).eval()
    return model
