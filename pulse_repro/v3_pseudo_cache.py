"""V3 G2 candidate pseudolabel cache for REAL ACDC training patients only.

Teacher: frozen E50 FR-BME, reused for multiple cases. This is NOT training.
Pseudo targets: dual NN pull-back label intersection, padded/OOB invalid=255.
Per-structure *relative image agreement heuristic* is an ablation candidate,
NOT a probability of correctness, NOT a learned or calibrated confidence.

Each target is a strictly interior ED->ES Cine time frame. No validation
patients may be written. UVI-index coordinates preserved; map to PULSE
2.5D input space explicitly in a subsequent adapter, not by axis swapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .v3_uvi_acdc_adapter import (
    TEACHER_SHAPE, prepare_endpoints, prepare_target_frame,
    read_info_cfg, ed_to_es_tau, split_role,
)
from .v3_frozen_teacher import FrozenFRBMETeacher, TeacherAssets, PropagatedLabels
from .v3_teacher_pilot import _find_patient, _load_phase, structural_agreement_diagnostic


@torch.no_grad()
def reliability_from_image_residuals(
    propagation: PropagatedLabels,
    warped_ed: torch.Tensor,
    warped_es: torch.Tensor,
    target: torch.Tensor,
    *,
    residual_floor: float = 0.005,
) -> tuple[torch.Tensor, dict]:
    """Two explicit B5/B6 candidates:
    B5 binary geometry + class consensus in propagation.agreement;
    B6 image-weighted reliability of those *same* candidates.

    For each class c, scale = max(median of candidate registration image
    residuals within class c, residual_floor). Weight = exp(-err/scale).
    This avoids one class's intensity distribution dominating all others,
    but it has not been validated as a calibrated pseudolabel correctness
    probability. Do not tune floor using validation ED/ES annotations.
    """
    if residual_floor <= 0:
        raise ValueError("residual_floor must be positive")
    if target.shape != warped_ed.shape or target.shape != warped_es.shape:
        raise ValueError("Image tensor geometry mismatch: images must share [B,1,D,H,W]")
    if target.ndim != 5 or target.shape[1] != 1 or tuple(target.shape[2:]) != TEACHER_SHAPE:
        raise ValueError("Unexpected 3D teacher image tensor geometry")
    valid = propagation.agreement
    if valid.shape != target.shape or propagation.hard_pseudo.shape != target.shape:
        raise ValueError("Pseudo label and image tensor geometry mismatch")
    if not all(torch.isfinite(x).all().item() for x in (warped_ed, warped_es, target)):
        raise ValueError("Nonfinite source warp or target image")
    residual = 0.5 * ((warped_ed.float() - target.float()).abs() +
                       (warped_es.float() - target.float()).abs())
    weights = torch.zeros_like(residual)
    per_class = {}
    for class_id, name in enumerate(("background", "RV", "Myo", "LV")):
        mask = valid & (propagation.hard_pseudo == class_id)
        count = int(mask.sum().item())
        if count:
            class_error = residual[mask]
            scale = max(float(class_error.median().item()), residual_floor)
            confidence = torch.exp(-class_error / scale)
            weights[mask] = confidence
            per_class[name] = {
                "candidate_voxels": count,
                "mean_image_weight": float(confidence.mean().item()),
                "median_warp_to_target_abs_error": float(class_error.median().item()),
                "relative_residual_scale": float(scale),
            }
        else:
            per_class[name] = {
                "candidate_voxels": 0,
                "mean_image_weight": None,
                "median_warp_to_target_abs_error": None,
                "relative_residual_scale": None,
            }
    assert torch.all(weights[~valid] == 0).item()
    return weights, {
        "policy": "classwise_median_normalized_registration_residual_softweight_v1",
        "interpretation": "Relative image-residual weighting, not correctness probability",
        "per_class": per_class,
    }


def class_balanced_pseudolabel_ce(
    logits: torch.Tensor, pseudo: torch.Tensor, reliability: torch.Tensor,
    *, background_factor: float = 0.25,
) -> torch.Tensor:
    """Average present classes; no foreground drowned by padded background.

    Accepts [B,4,H,W] or [B,4,D,H,W] logits and aligned B1* labels/weights.
    Pseudo 255 denotes ignore. Zero valid pixels give finite zero gradients.
    """
    import torch.nn.functional as F
    if not 0 <= background_factor <= 1:
        raise ValueError("background_factor must be between 0 and 1")
    if logits.ndim not in (4, 5) or logits.shape[1] != 4:
        raise ValueError("Expected segmentation logits B4HW or B4DHW")
    wanted = (logits.shape[0], 1, *logits.shape[2:])
    if pseudo.shape != wanted or reliability.shape != wanted:
        raise ValueError("Pseudo and reliability shapes must match logits")
    if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(reliability).all().item()):
        raise ValueError("Nonfinite loss inputs")
    if torch.any(reliability < 0).item() or torch.any(reliability > 1).item():
        raise ValueError("Reliability must be within [0,1]")
    if not torch.all((pseudo == 255) | ((pseudo >= 0) & (pseudo < 4))).item():
        raise ValueError("Pseudo labels must be in 0..3 or ignore=255")
    if torch.any(reliability[pseudo == 255] != 0).item():
        raise ValueError("Ignored voxels must have reliability=0")

    loss_map = F.cross_entropy(
        logits.float(), pseudo[:, 0].long(),
        ignore_index=255, reduction="none",
    )
    components = []
    coefficients = []
    for class_id in range(4):
        selected = (pseudo[:, 0] == class_id)
        weights = reliability[:, 0] * selected.to(reliability.dtype)
        total = weights.sum()
        if total.item() > 0:
            class_loss = (loss_map * weights.float()).sum() / total.float()
            factor = background_factor if class_id == 0 else 1.0
            if factor > 0:
                components.append(class_loss * factor)
                coefficients.append(factor)
    if not components:
        return logits.sum() * 0.0
    return torch.stack(components).sum() / sum(coefficients)


def _sha256(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            hash_obj.update(block)
    return hash_obj.hexdigest()


@torch.inference_mode()
def cache_one(
    patient_dir: Path,
    frame1: int,
    teacher: FrozenFRBMETeacher,
    assets_provenance: dict,
    output_dir: Path,
) -> dict:
    """One patient+frame, immutable output, only training set."""
    import nibabel as nib

    patient_dir = Path(patient_dir)
    pid = patient_dir.name
    if split_role(pid) != "train":
        raise ValueError(f"Refuse validation patient cache: {pid}")
    info = read_info_cfg(patient_dir / "Info.cfg")
    ed1, es1, frame_count = (info["ED"], info["ES"], info["NbFrame"])
    if not ed1 < frame1 < es1:
        raise ValueError("Teacher was trained for nonwrapping strictly interior ED->ES frame")
    tau = ed_to_es_tau(ed1, es1, frame1, frame_count)
    name = f"{pid}_t{frame1:02d}"
    out_dir = Path(output_dir).resolve() / pid
    out_file = out_dir / f"{name}.npz"
    meta_file = out_dir / f"{name}.json"
    if out_file.exists() or meta_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing training cache for {name}")

    ed, lab_ed = _load_phase(patient_dir, f"{pid}_frame{ed1:02d}")
    es, lab_es = _load_phase(patient_dir, f"{pid}_frame{es1:02d}")
    cine_path = patient_dir / f"{pid}_4d.nii.gz"
    cine = nib.load(str(cine_path))
    if tuple(cine.shape) != (*ed.shape, frame_count):
        raise ValueError("Cine NIfTI spatial dimensions or length inconsistent with Info.cfg")
    for source, index in ((ed, ed1), (es, es1)):
        indexed = np.asarray(cine.dataobj[..., index - 1], dtype=np.float32)
        if not np.allclose(source, indexed, atol=1e-5, rtol=1e-5):
            raise ValueError("ED or ES original volume doesn't match Cine indexed frame")

    pair = prepare_endpoints(ed, es, lab_ed, lab_es, mode="raw_uvi")
    if pair.grid is None:
        raise AssertionError("No raw UVI crop mapping")
    target = prepare_target_frame(
        np.asarray(cine.dataobj[..., frame1 - 1], dtype=np.float32),
        mode="raw_uvi", grid=pair.grid,
    )
    out = teacher.infer(pair, tau=tau)
    device = out["flow0"].device
    target_t = torch.as_tensor(target.copy(), device=device)[None, None]
    pseudo = out["labels"]
    weights, weight_summary = reliability_from_image_residuals(
        pseudo, out["warped_ed"], out["warped_es"], target_t,
    )
    structure = structural_agreement_diagnostic(pseudo)
    agreement_np = pseudo.agreement[0,0].detach().cpu().numpy()
    image_weight_np = weights[0,0].detach().cpu().numpy().astype(np.float16)
    pseudo_np = pseudo.hard_pseudo[0,0].detach().cpu().numpy().astype(np.uint8)
    both_valid_np = (pseudo.ed_valid & pseudo.es_valid)[0,0].detach().cpu().numpy()

    if not agreement_np.any() or not (pseudo_np[~agreement_np] == 255).all():
        raise RuntimeError("No valid consensus label, or invalid label not 255")
    if np.any(image_weight_np[~agreement_np] != 0):
        raise AssertionError("Invalid pseudo voxels have positive confidence")

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmp = out_dir / f"{name}.part.npz"
        # target is an actual unlabeled Cine volume on DCR/UVI spatial grid.
        np.savez_compressed(
            tmp,
            frame_uvi=target.astype(np.float16),
            pseudo_uvi=pseudo_np,
            weight_image_uvi=image_weight_np,
            binary_consensus_uvi=agreement_np.astype(np.uint8),
            valid_both_uvi=both_valid_np.astype(np.uint8),
            measured_uvi=pair.valid_spatial.astype(np.uint8),
            ed_source_label_uvi=pair.ed_label,
            es_source_label_uvi=pair.es_label,
        )
        os.replace(tmp, out_file)
        manifest = {
            "schema": "v3_acdc_candidate_pseudolabel_cache_v1",
            "split": "train_001_090_only",
            "patient": pid, "ed_1based": ed1, "es_1based": es1,
            "target_1based": frame1, "tau": tau,
            "native_grid_mapping": pair.grid.as_dict(),
            "mode": "raw_uvi",
            "flow_semantics": "target_grid_pullback_voxel_offsets_native_nifti_axis012",
            "teacher_front_sha256": assets_provenance["front_sha256"],
            "teacher_refiner_sha256": assets_provenance["refiner_sha256"],
            "raw_cine_sha256": _sha256(cine_path),
            "pseudo_policy": "strict_dual_endpoint_nn_agreement_ignore255",
            "weight_policy": weight_summary,
            "agreement_structure": structure,
            "measured_fraction": float(pair.valid_spatial.mean()),
            "consensus_fraction_of_measured": float(
                agreement_np.sum() / max(1, int(pair.valid_spatial.sum()))
            ),
            "npz_sha256": _sha256(out_file),
            "scientific_limits": "No intermediate GT; classwise image weights are heuristic, not teacher accuracy",
        }
        temp_meta = meta_file.with_suffix(".json.part")
        temp_meta.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        os.replace(temp_meta, meta_file)
        return manifest
    except Exception:
        # Never leave orphan partial pseudo-label cache if metadata failed.
        if out_file.exists() and not meta_file.exists():
            out_file.unlink()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcr-root", type=Path,
                        default=Path("/root/autodl-tmp/pulse_results/dcrmi_acdc_import"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/root/autodl-tmp/datasets/ACDC/database/training"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/autodl-tmp/v3_candidate_cache"))
    parser.add_argument("--patients", nargs="+",
                        default=["patient001", "patient002", "patient003"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.patients) > 3:
        raise ValueError("G2 candidate mode currently capped at 3 patients; no bulk cache")
    if len(set(args.patients)) != len(args.patients):
        raise ValueError("Duplicate patients")
    if not all(split_role(pid) == "train" for pid in args.patients):
        raise ValueError("Cannot create pseudolabel cache for validation patients")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device required but no GPU is available")
    assets = TeacherAssets(args.dcr_root)
    audit = assets.audit_file_provenance()
    teacher = FrozenFRBMETeacher(assets, device=args.device)
    summaries = []
    for pid in args.patients:
        patient_dir = args.raw_root / pid
        info = read_info_cfg(patient_dir / "Info.cfg")
        ed, es = info["ED"], info["ES"]
        if es - ed <= 1:
            raise ValueError(f"{pid}: no strictly internal ED->ES frame")
        frame = (ed + es) // 2
        result = cache_one(patient_dir, frame, teacher, audit, args.output_dir)
        classwise = result["agreement_structure"]["classes"]
        print(json.dumps({
            "patient": pid, "frame1": frame, "tau": result["tau"],
            "consensus_fraction_of_measured": result["consensus_fraction_of_measured"],
            "classwise_pair_dice_not_gt": {
                key: cls["pairwise_dice_not_gt"] for key, cls in classwise.items()
            },
            "classwise_image_weight": {
                key: cls["mean_image_weight"] for key, cls in
                result["weight_policy"]["per_class"].items()
            },
            "npz_sha256": result["npz_sha256"],
        }), flush=True)
        summaries.append(result)
    print("G2_CANDIDATE_CACHE_PASS", len(summaries),
          "training-only samples", flush=True)


if __name__ == "__main__":
    main()
