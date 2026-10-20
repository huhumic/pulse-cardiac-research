"""Full two-stage 120-epoch training of PULSE + selective V1 decoder.

No checkpoint of the *segmentation-trained* PULSE fold is used for initialization.
Exactly the upstream PULSE training loop and two-stage LR/unfreeze schedule are
retained through an audited temporary entrypoint with three minimal changes:
  1. swap PULSESeg for joint-trainable PULSESelectiveJointV1(PULSESeg),
  2. add variant metadata to model checkpoint,
  3. skip end-of-run test report if the ACDC test labels are unavailable.

The original upstream files and the existing PULSE run are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path



def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_TEST_END = r'''    t = eval_epoch(model, te_dl, device, tta=True)
    print(f"\n[FOLD {args.fold} TEST+TTA] RV={t[0]:.3f} Myo={t[1]:.3f} LV={t[2]:.3f} mean={t.mean():.3f}")'''
_TEST_GUARDED = r'''    if len(test_recs) > 0:
        t = eval_epoch(model, te_dl, device, tta=True)
        print(f"\n[FOLD {args.fold} TEST+TTA] RV={t[0]:.3f} Myo={t[1]:.3f} LV={t[2]:.3f} mean={t.mean():.3f}")
    else:
        print("[TEST] No publicly labeled testing records: evaluation skipped.")'''

_MODEL_ANCHOR = "    model = PULSESeg(IMG_SIZE).to(device)"
_MODEL_REPLACEMENT = "    model = PULSESelectiveJointV1(PULSESeg(IMG_SIZE)).to(device)"
_IMPORT_ANCHOR = "from transformers import AutoModel"
_IMPORT_REPLACEMENT = (
    "from transformers import AutoModel\n"
    "from pulse_repro.joint_v1 import PULSESelectiveJointV1"
)
_SAVE_ANCHOR = '''            torch.save({"epoch": epoch, "model": model.state_dict(), "best_dice": best,
                        "fold": args.fold}, ckpt_dir / "best_model.pth")'''
_SAVE_REPLACEMENT = '''            torch.save({"epoch": epoch, "model": model.state_dict(), "best_dice": best,
                        "fold": args.fold, "variant": "selective-v1", "training_mode": "joint"},
                       ckpt_dir / "best_model.pth")'''


def _replace_once(source: str, anchor: str, replacement: str) -> str:
    occurrences = source.count(anchor)
    if occurrences != 1:
        raise RuntimeError(
            f"PULSE pinned source mismatch; expected exactly one anchor "
            f"but found {occurrences}: {anchor[:80]!r}"
        )
    return source.replace(anchor, replacement, 1)


def build_entry(upstream_script: Path, output: Path) -> Path:
    """Produce inspected source copy, without changing PULSE's installed code."""
    source = upstream_script.read_text(encoding="utf-8")
    source = _replace_once(source, _TEST_END, _TEST_GUARDED)
    source = _replace_once(source, _IMPORT_ANCHOR, _IMPORT_REPLACEMENT)
    source = _replace_once(source, _MODEL_ANCHOR, _MODEL_REPLACEMENT)
    source = _replace_once(source, _SAVE_ANCHOR, _SAVE_REPLACEMENT)
    output.write_text(source, encoding="utf-8")
    compile(source, str(output), "exec")
    return output


def run(acdc_root: Path, pulse_root: Path, out_dir: Path,
        fold: int = 0, epochs: int = 120, batch_size: int = 4,
        lr: float = 3e-4, seed: int = 42, dry_run: bool = False) -> None:
    if fold not in range(5) or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid fold / epochs / batch-size / learning rate")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError(f"ACDC data not found: {acdc_root}/training")
    script = pulse_root / "pulse" / "pulse_seg.py"
    if not script.is_file():
        raise FileNotFoundError(f"PULSE source missing: {script}")
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = ("best_model.pth", "history.json", "run_config.json", "run_status.json")
    if any((out_dir / name).exists() for name in protected):
        raise FileExistsError(f"Refusing to overwrite existing training run: {out_dir}")

    entry = out_dir / "joint_v1_training_entry.py"
    cmd = [sys.executable, str(entry), "--fold", str(fold),
           "--nfolds", "5", "--epochs", str(epochs), "--bs", str(batch_size),
           "--lr", str(lr), "--seed", str(seed),
           "--ckpt_dir", str(out_dir), "--data_root", str(acdc_root)]
    # Build/validate the temporary script *before* writing run metadata.
    build_entry(script, entry)
    repo_root = Path(__file__).resolve().parent.parent
    meta = {
        "variant": "selective-v1", "training_mode": "joint",
        "fold": fold, "nfolds": 5, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "seed": seed, "acdc_root": str(acdc_root),
        "pulse_root": str(pulse_root), "source_sha256": file_sha256(script),
        "training_script_sha256": file_sha256(entry),
        "initialization": "facebook/dinov2-base backbone; PULSE decoder from scratch; no fold checkpoint",
        "training_protocol": "official PULSE 2-stage 70+50 schedule, optimizer, loss, data and augmentation",
        "model_change": "selective-v1 high-resolution skip-fusion blocks",
        "validation_selection": "official PULSE per-slice validation (4-flip TTA after epoch 70)",
        "command": cmd,
    }
    if dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return
    (out_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["ACDC_ROOT"] = str(acdc_root)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH", "")]))
    start = time.perf_counter()
    print("Executing:", " ".join(cmd), flush=True)
    with (out_dir / "training.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            cmd, cwd=str(pulse_root), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    elapsed = time.perf_counter() - start
    (out_dir / "run_status.json").write_text(json.dumps({
        "exit_code": code, "elapsed_seconds": elapsed,
        "history_exists": (out_dir / "history.json").exists(),
        "best_model_exists": (out_dir / "best_model.pth").exists(),
    }, indent=2), encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"Full joint V1 training exited with code {code}; see training.log")
    print(f"V1 full training complete in {elapsed:.1f} sec: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
