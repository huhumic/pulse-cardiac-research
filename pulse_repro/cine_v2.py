"""V2: target-time selective motion evidence + anatomical skip enhancement.

Spatiotemporal fusion is explicit and independent of spatial-slice stacking:
`spatial` = neighboring Z slices at ED/ES; `temporal` = same Z slice from
preceding and following cine time frames. No fake temporal GT is assumed.

Components:
- M1: optional existing V1 anatomy skip enhancers.
- M2: local feature offset warping of both cardiac-time neighbors; a third
  'no evidence' softmax branch and feature-mismatch reliability weighting.
- M3: reliability-weighted agreement of *two target-aligned* evidence probes;
  NOT supervised segmentation of unlabelled intermediate cine frames.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .selective_v1 import SelectiveDPTDecoder


class TinyTemporalEncoder(nn.Module):
    """Shared 2D encoder, far cheaper than re-running DINOv2 for each time."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16),
            nn.Conv2d(16, channels, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GELU(),
        )

    def forward(self, image):
        return self.net(image)


def warp_2d(source: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Pull-back resampling in feature-pixel coordinates (x,y), align_corners=True."""
    if source.ndim != 4 or offsets.shape != (source.shape[0],2,*source.shape[-2:]):
        raise ValueError("warp expects BCHW features and B2HW offsets")
    b, _, h, w = source.shape
    dtype = source.dtype
    ys = torch.linspace(-1, 1, h, device=source.device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=source.device, dtype=dtype)
    y,x = torch.meshgrid(ys,xs,indexing="ij")
    grid = torch.stack((x,y),dim=-1).unsqueeze(0).expand(b,-1,-1,-1)
    scale = torch.tensor([2/max(1,w-1), 2/max(1,h-1)],device=source.device,dtype=dtype)
    shifted = grid + offsets.permute(0,2,3,1) * scale
    return F.grid_sample(source,shifted,mode="bilinear",padding_mode="border",align_corners=True)


class BidirectionalEvidenceFusion(nn.Module):
    def __init__(self, decoder_channels: int, temporal_channels: int = 32,
                 max_offset: float = 3.0):
        super().__init__()
        self.encoder = TinyTemporalEncoder(temporal_channels)
        in_ch = temporal_channels * 2
        self.align = nn.Sequential(
            nn.Conv2d(in_ch, temporal_channels, 3,padding=1),nn.GELU(),
            nn.Conv2d(temporal_channels, 2, 3,padding=1),
        )
        nn.init.zeros_(self.align[-1].weight)
        nn.init.zeros_(self.align[-1].bias)
        self.max_offset = float(max_offset)
        self.quality = nn.Sequential(
            nn.Conv2d(temporal_channels * 2 + 2, temporal_channels, 3,padding=1),
            nn.GELU(),nn.Conv2d(temporal_channels,1,1),
        )
        self.to_decoder = nn.Conv2d(temporal_channels, decoder_channels, 1)
        nn.init.zeros_(self.to_decoder.weight)
        nn.init.zeros_(self.to_decoder.bias)
        # Both probes segment the same *target frame*, with separately warped
        # temporal evidence. They are not artificial GT for unlabeled frames.
        self.probe = nn.Conv2d(temporal_channels * 2, 4, 1)

    def forward(self, spatial_middle: torch.Tensor,
                temporal: torch.Tensor, feature_shape: tuple[int,int]):
        if spatial_middle.ndim!=4 or spatial_middle.shape[1]!=1:
            raise ValueError("Expected spatial-middle intensity B1HW")
        if temporal.shape != (spatial_middle.shape[0],2,*spatial_middle.shape[-2:]):
            raise ValueError("Expected true temporal neighbors B2HW")
        h,w = feature_shape
        image0 = self.encoder(spatial_middle)
        imgm = self.encoder(temporal[:,0:1])
        imgp = self.encoder(temporal[:,1:2])
        features = [F.interpolate(x,size=(h,w),mode="bilinear",align_corners=False)
                    for x in (image0,imgm,imgp)]
        current, before, after = features
        aligned=[]
        quality=[]
        for neighboring in (before,after):
            offsets = self.max_offset * torch.tanh(self.align(torch.cat((current,neighboring),1)))
            moved = warp_2d(neighboring,offsets)
            delta = (current-moved).abs().mean(dim=1,keepdim=True)
            magnitude = offsets.square().sum(dim=1,keepdim=True).sqrt() / self.max_offset
            score = self.quality(torch.cat((current,moved,delta,magnitude),dim=1))
            # Independent agreement and motion penalties discourage blindly
            # trusting poorly matched temporal evidence.
            score = score - delta - 0.25*magnitude
            aligned.append(moved)
            quality.append(score)
        logits = torch.cat((torch.zeros_like(quality[0]),quality[0],quality[1]),1)
        weights = torch.softmax(logits.float(),dim=1).to(current.dtype)
        residual = (
            weights[:,1:2]*(aligned[0]-current) +
            weights[:,2:3]*(aligned[1]-current)
        )
        update = self.to_decoder(residual)
        probe_minus = self.probe(torch.cat((current,aligned[0]),1))
        probe_plus = self.probe(torch.cat((current,aligned[1]),1))
        return update, {"weights": weights, "probe_minus": probe_minus,
                        "probe_plus": probe_plus}


def reliable_agreement_loss(aux: dict, target_logits: torch.Tensor,
                            threshold: float = 0.55) -> torch.Tensor:
    """Confidence/quality-weighted Jensen-Shannon of two target-aligned probes.

    Both branches see the same target state after motion alignment. Teacher
    weights are detached. No pseudo-mask used as ground-truth at other frames.
    """
    a = aux["probe_minus"].float()
    b = aux["probe_plus"].float()
    p = torch.softmax(a,1).clamp_min(1e-6)
    q = torch.softmax(b,1).clamp_min(1e-6)
    mid = (p+q).mul(0.5).detach().clamp_min(1e-6)
    jsd = 0.5 * (
        (p * (p.log()-mid.log())).sum(1,keepdim=True) +
        (q * (q.log()-mid.log())).sum(1,keepdim=True)
    )
    conf = torch.softmax(F.interpolate(target_logits.detach().float(),size=a.shape[-2:],
                       mode="bilinear",align_corners=False),dim=1).max(1,keepdim=True)[0]
    trust = aux["weights"][:,1:2].detach().minimum(aux["weights"][:,2:3].detach())
    mask = (conf >= threshold).float()
    weighted = trust * mask
    # No selected pixels => exactly zero with valid gradients.
    return (weighted * jsd).sum() / weighted.sum().clamp_min(1.)


class CineDPTDecoder(nn.Module):
    def __init__(self, core: nn.Module, structure: bool = True):
        super().__init__()
        self.spatial = SelectiveDPTDecoder(core)
        projection = next(m for m in core.projs[0].modules() if isinstance(m,nn.Linear))
        self.temporal = BidirectionalEvidenceFusion(projection.out_features)
        self.enable_structure = structure

    def forward(self, hs, spatial_middle, neighbors):
        dec = self.spatial
        core = dec.core
        f = [core._reassemble(h,core.projs[i]) for i,h in enumerate(hs)]
        x = core.ups[3](core.res[3](f[3]))
        x = core.ups[2](core.res[2](x+F.interpolate(
            f[2],size=x.shape[-2:],mode="bilinear",align_corners=False)))
        a1 = core.aux1(x)
        skip1=F.interpolate(f[1],size=x.shape[-2:],mode="bilinear",align_corners=False)
        delta, aux = self.temporal(spatial_middle,neighbors,x.shape[-2:])
        pre = x
        x = pre + skip1 + delta
        if self.enable_structure:
            x = x + dec.enhancers[0](pre,skip1)
        x = core.ups[1](core.res[1](x))
        a2 = core.aux2(x)
        skip0=F.interpolate(f[0],size=x.shape[-2:],mode="bilinear",align_corners=False)
        old=x
        x = x + skip0
        if self.enable_structure:
            x = x + dec.enhancers[1](old,skip0)
        x = core.ups[0](core.res[0](x))
        main = F.interpolate(core.head(x), (core.out_size,core.out_size),
                             mode="bilinear",align_corners=False)
        return (main,a1,a2), aux


class PULSECineReliabilityV2(nn.Module):
    EXTRACT = (3,6,9,12)

    def __init__(self, original_pulse: nn.Module, use_structure: bool = True):
        super().__init__()
        if not hasattr(original_pulse.decoder,"projs"):
            raise TypeError("Expected an untouched PULSESeg model")
        self.base=original_pulse
        self.decoder=CineDPTDecoder(original_pulse.decoder,structure=use_structure)
        # Avoid separately registering the original decoder twice.
        self.base.decoder=nn.Identity()

    @property
    def backbone_blocks(self):
        return self.base.backbone_blocks

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor,
                return_aux: bool=False):
        if spatial.ndim!=4 or spatial.shape[1]!=3:
            raise ValueError("Expected PULSE spatial input B3HW")
        if temporal.shape != (spatial.shape[0],2,*spatial.shape[-2:]):
            raise ValueError("Expected cardiac-time neighbors B2HW")
        x=self.base.stem(spatial)
        out=self.base.vit(pixel_values=x,output_hidden_states=True)
        hs=[out.hidden_states[i] for i in self.EXTRACT]
        outputs,aux=self.decoder(hs,spatial[:,1:2],temporal)
        return (*outputs,aux) if return_aux else outputs


def load_cine_v2(pulse_root, checkpoint, device):
    from .runtime import upstream
    _, source = upstream(pulse_root)
    raw=torch.load(str(checkpoint),map_location="cpu",weights_only=True)
    if raw.get("variant")!="cine-reliability-v2":
        raise ValueError("Expected cine-reliability-v2 checkpoint")
    model=PULSECineReliabilityV2(source.PULSESeg(256),
            use_structure=bool(raw.get("use_structure",True)))
    model.load_state_dict(raw["model"],strict=True)
    return model.to(device).eval()
