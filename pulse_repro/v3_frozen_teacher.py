"""Frozen ACDC FR-BME motion teacher and conservative endpoint-label propagation.

This module does NOT train the teacher/student or assume teacher temporal phase
mapping matches the raw frame-index ratio. Actual tau must be externally audited.

Tensor spatial axes: [native NIfTI axis0, axis1, axis2], NOT physical z/y/x.
Flow channels follow these tensor axes in native-index voxel units and encode
target-grid pull-back offsets. The verified FR-BME warp uses align_corners=True.

Requires torch. Only a deliberate loader call imports migrated historical code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .v3_uvi_acdc_adapter import TEACHER_SHAPE, TeacherEndpoints


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


@dataclass(frozen=True)
class TeacherAssets:
    imported_root: Path

    @property
    def front(self) -> Path:
        return (self.imported_root / "checkpoints" /
                "frbme_acdc_4090_formal/epoch_050.pth")

    @property
    def refiner(self) -> Path:
        return (self.imported_root / "checkpoints" /
                "relation_reliability_ablation_acdc_full_acdc_frE050_seed12345" /
                "epoch_100.pth")

    @property
    def manifest(self) -> Path:
        return (self.imported_root / "reports" /
                "relation_reliability_ablation_acdc_full_acdc_frE050_seed12345" /
                "manifest.json")

    @property
    def source(self) -> Path:
        return self.imported_root / "source"

    def audit_file_provenance(self) -> dict:
        """No weight unpickling and no GPU. Refuse mismatched E100->E50 pair."""
        for path in (self.front, self.refiner, self.manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        meta = json.loads(self.manifest.read_text(encoding="utf-8"))
        recorded = str(meta.get("fr_checkpoint", "")).replace("\\", "/")
        if not recorded.endswith("checkpoints/frbme_acdc_4090_formal/epoch_050.pth"):
            raise ValueError("E100 manifest does not reference the imported E50 front")
        if meta.get("dataset") != "acdc" or meta.get("structural_variant") != "full":
            raise ValueError("Unexpected teacher dataset or variant")
        if meta.get("variant_config") != {
            "use_relation": True, "use_warp": True, "state_conditioned": True
        }:
            raise ValueError("Unexpected ACDC refiner configuration")
        return {
            "front": str(self.front),
            "front_sha256": sha256_file(self.front),
            "refiner": str(self.refiner),
            "refiner_sha256": sha256_file(self.refiner),
            "manifest": str(self.manifest),
            "manifest_sha256": sha256_file(self.manifest),
            "refiner_depends_on_front_e50": True,
            "refiner_variant": meta["variant_config"],
            "note": "Pair matches manifest; does not independently prove final paper main-table identity",
        }


def check_flow(flow: torch.Tensor, *, batch: int, shape: tuple[int, int, int]) -> None:
    if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (batch, 3, *shape):
        raise ValueError(f"Expected voxel flow [{batch},3,{','.join(map(str,shape))}]")
    if not torch.isfinite(flow).all().item():
        raise ValueError("Nonfinite teacher flow")


def _locations(flow: torch.Tensor) -> torch.Tensor:
    d, h, w = flow.shape[2:]
    if min(d, h, w) < 2:
        raise ValueError("Flow shape must have all spatial dimensions >=2")
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=flow.device, dtype=flow.dtype),
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    identity = torch.stack((zz, yy, xx), dim=0).unsqueeze(0)
    return identity + flow


def _inside(loc: torch.Tensor) -> torch.Tensor:
    dims = tuple(int(v) for v in loc.shape[2:])
    assert len(dims) == 3
    inside = torch.ones((loc.shape[0], 1, *dims),
                        dtype=torch.bool, device=loc.device)
    for axis, limit in enumerate(dims):
        inside &= ((loc[:, axis:axis+1] >= 0)
                   & (loc[:, axis:axis+1] <= limit - 1))
    return inside


def _grid_xyz(loc: torch.Tensor) -> torch.Tensor:
    d, h, w = loc.shape[2:]
    factors = loc.new_tensor([d - 1, h - 1, w - 1]).view(1, 3, 1, 1, 1)
    return (2.0 * (loc / factors - 0.5)).permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]


@dataclass
class PropagatedLabels:
    """All masks are [B,1,D,H,W]. Uncertain pseudolabel pixels equal 255."""
    ed: torch.Tensor
    es: torch.Tensor
    ed_valid: torch.Tensor
    es_valid: torch.Tensor
    agreement: torch.Tensor
    hard_pseudo: torch.Tensor

    def summary(self) -> dict[str, float]:
        return {
            "ed_valid_fraction": float(self.ed_valid.float().mean().item()),
            "es_valid_fraction": float(self.es_valid.float().mean().item()),
            "agreement_fraction": float(self.agreement.float().mean().item()),
            "agreement_among_both_valid": float(
                self.agreement.sum().item() /
                max(1, int((self.ed_valid & self.es_valid).sum().item()))
            ),
        }


@torch.no_grad()
def propagate_endpoint_labels(
    ed_label: torch.Tensor,
    es_label: torch.Tensor,
    flow0: torch.Tensor,
    flow1: torch.Tensor,
    measured_valid: torch.Tensor,
) -> PropagatedLabels:
    """Conservative NN pull-back; zero-padding cannot create false background GT.

    measured_valid is the teacher UVI-grid native-measurement mask. We exclude
    1) padded target locations, 2) out-of-bounds continuous sampling locations,
    3) source locations that sample the native padding area.
    Both propagated labels remain available for future reliability analysis;
    hard_pseudo is 255 unless both endpoints agree at a valid target voxel.
    """
    if ed_label.shape != es_label.shape or ed_label.ndim != 5 or ed_label.shape[1] != 1:
        raise ValueError("Endpoint labels must have shape [B,1,D,H,W]")
    shape = tuple(ed_label.shape[2:])
    if measured_valid.shape != ed_label.shape:
        raise ValueError("Measured voxel mask must match labels [B,1,D,H,W]")
    if (not ed_label.device == es_label.device == flow0.device ==
            flow1.device == measured_valid.device):
        raise ValueError("All tensors must be on the same device")
    if shape != TEACHER_SHAPE:
        raise ValueError(f"Teacher label shape must be {TEACHER_SHAPE}, received {shape}")
    n = ed_label.shape[0]
    for lab in (ed_label, es_label):
        if not torch.isfinite(lab).all().item():
            raise ValueError("Nonfinite labels")
        if not (((lab >= 0) & (lab <= 3) & (lab == lab.round())).all().item()):
            raise ValueError("Labels must be integers in 0..3")
    check_flow(flow0, batch=n, shape=shape)
    check_flow(flow1, batch=n, shape=shape)

    target_valid = measured_valid.bool()
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for source, flow in ((ed_label, flow0), (es_label, flow1)):
        loc = _locations(flow.float())
        xyz = _grid_xyz(loc)
        warped = F.grid_sample(source.float(), xyz, mode="nearest",
                               padding_mode="zeros", align_corners=True).round().long()
        source_valid = F.grid_sample(
            target_valid.float(), xyz, mode="nearest",
            padding_mode="zeros", align_corners=True) > .5
        valid = target_valid & _inside(loc) & source_valid
        results.append((warped, valid))
    y0, v0 = results[0]
    y1, v1 = results[1]
    agreement = v0 & v1 & (y0 == y1)
    pseudo = torch.where(agreement, y0, torch.full_like(y0, 255))
    return PropagatedLabels(y0, y1, v0, v1, agreement, pseudo)


class FrozenFRBMETeacher:
    """Load E50 front with exact migrated source. Not trainable.

    E100 refiner is not used for flow/label propagation, but its recorded
    provenance is checked. Only load checkpoints from your own trusted server.
    """

    def __init__(self, assets: TeacherAssets, *, device: str = "cpu"):
        assets.audit_file_provenance()
        if not assets.source.is_dir():
            raise FileNotFoundError(assets.source)
        resolved = str(assets.source.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        module = importlib.import_module("research_modules.frbme.frbme")
        self.device = torch.device(device)
        self.assets = assets

        # The user's own trusted training checkpoints may contain non-tensor
        # metadata. DO NOT enable weights_only=False for untrusted inputs.
        state = torch.load(assets.front, map_location="cpu", weights_only=False)
        # Strict state_dict loading validates the exact instantiated architecture.
        # Do not restrict the training-script-specific model_kind spelling here.
        cfg = module.FRBMEConfig(**state["model_config"])
        if tuple(cfg.inshape) != TEACHER_SHAPE:
            raise ValueError(f"Teacher checkpoint requires unexpected shape: {cfg.inshape}")
        model = module.FRBME(cfg)
        model.load_state_dict(state["model"], strict=True)
        model.requires_grad_(False).eval()
        self.model = model.to(self.device)

    @torch.inference_mode()
    def infer(self, pair: TeacherEndpoints, *, tau: float):
        """Real teacher forward; tau must match audited DCR time policy.

        No implicit frame->tau conversion: call only after provenance of
        historical training phase policy is verified. Returns propagated
        labels plus both raw flow tensors and warped images on self.device.
        """
        if not 0 <= float(tau) <= 1:
            raise ValueError("tau must be in [0,1]")
        ed = pair.ed_image
        es = pair.es_image
        for x in (ed, es):
            if x.shape != TEACHER_SHAPE or not np.isfinite(x).all():
                raise ValueError("Unverified teacher endpoint shape or values")
        images = torch.as_tensor(
            np.stack([ed, es], axis=0)[None].copy(), device=self.device,
            dtype=torch.float32,
        )
        warped0, warped1, flow0, flow1 = self.model(images, float(tau))
        check_flow(flow0, batch=1, shape=TEACHER_SHAPE)
        check_flow(flow1, batch=1, shape=TEACHER_SHAPE)
        lab0 = torch.as_tensor(pair.ed_label.copy(), device=self.device).long()[None, None]
        lab1 = torch.as_tensor(pair.es_label.copy(), device=self.device).long()[None, None]
        valid = torch.as_tensor(pair.valid_spatial.copy(), device=self.device).bool()[None, None]
        propagation = propagate_endpoint_labels(lab0, lab1, flow0, flow1, valid)
        return {
            "warped_ed": warped0, "warped_es": warped1,
            "flow0": flow0, "flow1": flow1, "labels": propagation,
            "tau": float(tau),
        }
