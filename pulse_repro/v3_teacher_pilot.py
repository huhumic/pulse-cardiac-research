"""One real ACDC *training-patient* frozen DCR-MI E50 teacher forward.

Standalone G1 diagnostic, NOT segmentation training or full G1 acceptance.
Requires a CUDA GPU, uses original UVI ACDC image/label spatial mapping.
Only verifies native Cine's indexed ED/ES correspondence; does not claim
full physical-world affine congruence when original NIfTI headers disagree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .v3_uvi_acdc_adapter import (
    TEACHER_SHAPE, prepare_endpoints, prepare_target_frame, ed_to_es_tau,
    read_info_cfg, split_role,
)
from .v3_frozen_teacher import FrozenFRBMETeacher, TeacherAssets


def _find_default_root() -> Path:
    choices = [
        Path("/root/autodl-tmp/project"),
        Path("/root/autodl-tmp/pulse_results/dcrmi_acdc_import"),
    ]
    for candidate in choices:
        if (candidate / "checkpoints/frbme_acdc_4090_formal/epoch_050.pth").is_file():
            return candidate
    raise FileNotFoundError("DCR teacher root not found. Supply --dcr-root explicitly.")


def _find_patient(patient: str) -> Path:
    candidates = [
        Path("/root/autodl-tmp/datasets/ACDC/database/training") / patient,
        Path("/root/autodl-tmp/project/datasets/ACDC/database/training") / patient,
        Path("/root/autodl-tmp/ACDC_pulse/database/training") / patient,
    ]
    for p in candidates:
        if (p / f"{patient}_4d.nii.gz").is_file():
            return p
    raise FileNotFoundError("Raw 4D Cine not found. Supply --raw-patient explicitly.")


def _load_phase(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    image = nib.load(str(path / f"{prefix}.nii.gz"))
    seg = nib.load(str(path / f"{prefix}_gt.nii.gz"))
    if image.shape != seg.shape:
        raise ValueError(f"Image/label mismatch {prefix}")
    if not np.allclose(image.affine, seg.affine, atol=1e-3, rtol=0):
        raise ValueError(f"Image/label affine mismatch {prefix}")
    return (np.asarray(image.dataobj, dtype=np.float32),
            np.asarray(seg.dataobj))


def _compare_training_cache(cache_root: Path, patient: str, pair) -> dict:
    folder = cache_root / patient
    ed_path, es_path = folder / "ed.npy", folder / "es.npy"
    if not (ed_path.is_file() and es_path.is_file()):
        return {"checked": False, "reason": "No historical processed ED/ES npy in this directory"}
    differences = {}
    for name, path, expected in (
        ("ed", ed_path, pair.ed_image),
        ("es", es_path, pair.es_image),
    ):
        cached = np.load(path, mmap_mode="r")
        if cached.shape != TEACHER_SHAPE:
            raise ValueError(f"Historical {name} cache shape {cached.shape} != {TEACHER_SHAPE}")
        max_err = float(np.max(np.abs(np.asarray(cached, np.float32) - expected)))
        differences[name + "_max_abs"] = max_err
        if not max_err <= 1e-4:
            raise ValueError(f"Historical UVI cache {name} differs from raw-data adapter: maxabs={max_err:.6g}")
    return {"checked": True, **differences}



def structural_agreement_diagnostic(propagation) -> dict:
    """Measure agreement for RV/Myo/LV separately; never call it accuracy.

    Class statistics are restricted to pixels where BOTH endpoint pull-backs
    sample an originally measured (not padded) voxel. Background prevalence
    therefore cannot inflate class-specific metrics.
    """
    import torch
    both = propagation.ed_valid & propagation.es_valid
    ed = propagation.ed[both].long()
    es = propagation.es[both].long()
    confusion = torch.bincount(ed * 4 + es, minlength=16).reshape(4, 4)
    total_both = int(both.sum().item())
    diagonals = int(confusion.diag().sum().item())
    fg_any = total_both - int(confusion[0, 0].item())
    fg_agree = int(confusion[1, 1].item() + confusion[2, 2].item()
                   + confusion[3, 3].item())

    classes = {}
    for name, idx in (("RV", 1), ("Myo", 2), ("LV", 3)):
        n_ed = int(confusion[idx, :].sum().item())
        n_es = int(confusion[:, idx].sum().item())
        inter = int(confusion[idx, idx].item())
        denom = n_ed + n_es
        union = n_ed + n_es - inter
        classes[name] = {
            "ed_propagated_voxels_both_valid": n_ed,
            "es_propagated_voxels_both_valid": n_es,
            "both_agree_voxels": inter,
            "pairwise_dice_not_gt": float(2 * inter / denom) if denom else None,
            "pairwise_iou_not_gt": float(inter / union) if union else None,
        }

    return {
        "both_endpoint_valid_voxels": total_both,
        "agreement_all_classes_in_both_valid": (
            float(diagonals / total_both) if total_both else None
        ),
        "foreground_union_voxels": fg_any,
        "foreground_agree_voxels": fg_agree,
        "foreground_agreement_given_either_foreground": (
            float(fg_agree / fg_any) if fg_any else None
        ),
        "classes": classes,
        "meaning": "Agreement of two warped endpoint labels; no intermediate GT or Dice accuracy",
    }

def run(raw_patient: Path, dcr_root: Path, frame1: int,
        report: Path | None = None) -> dict:
    import nibabel as nib
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU in this instance. Do not run expensive teacher forward on CPU.")
    gpu = "cuda:0"

    raw_patient = raw_patient.resolve()
    patient = raw_patient.name
    if split_role(patient) != "train":
        raise ValueError(f"Teacher pilot is restricted to training patients 001..090: {patient}")
    if not (raw_patient / "Info.cfg").is_file():
        raise FileNotFoundError(raw_patient / "Info.cfg")
    info = read_info_cfg(raw_patient / "Info.cfg")
    ed1, es1, nframes = info["ED"], info["ES"], info["NbFrame"]
    if ed1 >= es1:
        raise ValueError("Historical ACDC hidden frames use nonwrapping ED<ES; refusing ambiguous wrap")
    if not ed1 < frame1 < es1:
        raise ValueError(f"Choose strictly hidden ED->ES frame: ED={ed1}, frame={frame1}, ES={es1}")
    tau = ed_to_es_tau(ed1, es1, frame1, nframes)

    # Explicitly check the checkpoint's actual training protocol before real inference.
    assets = TeacherAssets(dcr_root.resolve())
    hashes = assets.audit_file_provenance()
    ck = torch.load(assets.front, map_location="cpu", weights_only=False)
    policy = ck.get("phase_policy")
    contract = ck.get("dataset_contract") or {}
    if policy != "ACDC_uniform_real_hidden_hash" or contract.get("endpoints") != "ED->ES":
        raise ValueError(f"Unexpected training time protocol {policy!r}/{contract.get('endpoints')!r}")
    if contract.get("train") != "patient001-patient090":
        raise ValueError("Historical teacher's training-patient provenance does not match V3")
    if tuple(ck["model_config"]["inshape"]) != TEACHER_SHAPE:
        raise ValueError("Teacher checkpoint geometry mismatch")

    ed_raw, y0 = _load_phase(raw_patient, f"{patient}_frame{ed1:02d}")
    es_raw, y1 = _load_phase(raw_patient, f"{patient}_frame{es1:02d}")
    cine = nib.load(str(raw_patient / f"{patient}_4d.nii.gz"))
    if len(cine.shape) != 4 or cine.shape[:3] != ed_raw.shape or cine.shape[3] != nframes:
        raise ValueError("Raw Cine geometry/count mismatch")
    for name, raw, frame in (("ED", ed_raw, ed1), ("ES", es_raw, es1)):
        arr = np.asarray(cine.dataobj[..., frame-1], dtype=np.float32)
        if not np.allclose(arr, raw, atol=1e-5, rtol=1e-5):
            raise ValueError(f"{name} NIfTI voxel array differs from indexed Cine frame")
    pair = prepare_endpoints(ed_raw, es_raw, y0, y1, mode="raw_uvi")
    if pair.grid is None:
        raise AssertionError("Raw UVI needs spatial inverse mapping")
    target = prepare_target_frame(
        np.asarray(cine.dataobj[..., frame1-1], dtype=np.float32),
        mode="raw_uvi", grid=pair.grid,
    )

    cache_root = Path("/root/autodl-tmp/project/data_cache/ACDC_UVI128_CACHE")
    cache_check = _compare_training_cache(cache_root, patient, pair)
    teacher = FrozenFRBMETeacher(assets, device=gpu)

    with torch.inference_mode():
        forward = teacher.infer(pair, tau=tau)
        w0 = forward["warped_ed"].float()
        w1 = forward["warped_es"].float()
        t = torch.as_tensor(target, device=w0.device, dtype=torch.float32)[None, None]
        measured = torch.as_tensor(pair.valid_spatial, device=w0.device)[None, None]
        if w0.shape != t.shape or w1.shape != t.shape:
            raise ValueError("Warped image shape mismatch with real Cine target")
        if not torch.isfinite(w0).all().item() or not torch.isfinite(w1).all().item():
            raise ValueError("Nonfinite warped teacher image")
        # Diagnostic only; not full DCR-MI reconstruction/PSNR because E100 refiner
        # is deliberately not used by this motion-teacher pilot.
        ed_mae = float((w0-t).abs()[measured].mean().item())
        es_mae = float((w1-t).abs()[measured].mean().item())
        pseudo = forward["labels"]
        structure = structural_agreement_diagnostic(pseudo)
        valid_pixels = int(pseudo.agreement.sum().item())
        valid_fraction_of_measured = float(valid_pixels / max(1, int(measured.sum().item())))
        label_hist = {
            str(i): int((pseudo.hard_pseudo == i).sum().item()) for i in range(4)
        }
    result = {
        "gate": "G1_B_SINGLE_PATIENT_FROZEN_TEACHER_PILOT",
        "patient": patient, "role": "training_only",
        "ed_frame_1based": ed1, "es_frame_1based": es1,
        "target_frame_1based": frame1, "tau": tau, "frame_count": nframes,
        "checkpoint_phase_policy": policy,
        "checkpoint_front_sha256": hashes["front_sha256"],
        "checkpoint_refiner_sha256": hashes["refiner_sha256"],
        "preprocessing": "raw_uvi_recreated_from_official_ACDC",
        "historical_cached_endpoints": cache_check,
        "native_shape": list(pair.grid.native_shape),
        "teacher_shape": list(TEACHER_SHAPE),
        "measured_fraction": float(pair.valid_spatial.mean()),
        "warped_endpoint_vs_real_target_MAE_measured_only": {
            "ed": ed_mae, "es": es_mae,
        },
        "propagation": pseudo.summary(),
        "structure": structure,
        "hard_agreement_fraction_of_measured": valid_fraction_of_measured,
        "hard_pseudo_class_voxels": label_hist,
        "device": gpu, "cuda_peak_MB": round(torch.cuda.max_memory_allocated()/2**20, 1),
        "limitations": "One train-patient diagnostic only; not real intermediate label GT or formal quality result",
    }
    if report is not None:
        report = report.resolve()
        if report.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic report: {report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient", default="patient001")
    p.add_argument("--raw-patient", type=Path)
    p.add_argument("--dcr-root", type=Path)
    p.add_argument("--frame", type=int, default=6)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    patient_dir = args.raw_patient or _find_patient(args.patient)
    root = args.dcr_root or _find_default_root()
    result = run(patient_dir, root, args.frame, args.report)
    print(json.dumps({
        "status": "PILOT_PASS",
        "patient": result["patient"],
        "frame": result["target_frame_1based"],
        "tau": result["tau"],
        "cache_comparison": result["historical_cached_endpoints"],
        "endpoint_mae": result["warped_endpoint_vs_real_target_MAE_measured_only"],
        "propagation": result["propagation"],
        "foreground_agreement": result["structure"]["foreground_agreement_given_either_foreground"],
        "classwise_pair_dice_not_gt": {
            k: v["pairwise_dice_not_gt"]
            for k, v in result["structure"]["classes"].items()
        },
        "hard_agreement_fraction_of_measured": result["hard_agreement_fraction_of_measured"],
        "cuda_peak_MB": result["cuda_peak_MB"],
    }, indent=2))


if __name__ == "__main__":
    main()
