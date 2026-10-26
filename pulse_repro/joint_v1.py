"""End-to-end trainable PULSE + V1 selective skip fusion.

Use this for parity with PULSE's *full* 120-epoch two-stage curriculum.
The zero-initialized residual updates have identical initial forward outputs
as PULSE, but the PULSE backbone and decoder train jointly thereafter.

The state_dict key names intentionally match PULSESelectiveV1, which is only
used as an eval-time loader in pulse_repro.selective_v1.load_variant().
"""
from __future__ import annotations

from torch import nn
from .selective_v1 import SelectiveDPTDecoder


class PULSESelectiveJointV1(nn.Module):
    def __init__(self, pulse: nn.Module):
        super().__init__()
        if not hasattr(pulse.decoder, "projs"):
            raise TypeError("Expected unmodified upstream PULSE DPTDecoderDS")
        self.base = pulse
        self.base.decoder = SelectiveDPTDecoder(self.base.decoder)

    @property
    def backbone_blocks(self):
        """Required by upstream optimizer and epoch-71 unfreeze code."""
        return self.base.backbone_blocks

    def forward(self, x):
        return self.base(x)
