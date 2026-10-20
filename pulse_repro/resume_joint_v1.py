"""Resume a V1 joint-training run from its latest available saved model/state.

Model-only V1 best_model.pth (e.g., epoch 72) lacks optimizer/scaler/RNG;
continuation is a WARM RESTART, not exact interruption-free resumption.
The historical run directory is read-only. Every output goes to a new directory.
The pinned PULSE training entry is patched only in the new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tempfile

import torch

from .train_joint_v1 import build_entry, _replace_once, file_sha256

_INIT_ANCHOR = "    best = 0.; unfroze = False; hist = []"
_LOOP_ANCHOR = "    for epoch in range(1, args.epochs + 1):"
_ARG_ANCHOR = "    args = ap.parse_args()"
_EPOCH_SAVE_ANCHOR = '            print(f"  >> saved best mean={best:.4f}")'

# Do not alter the upstream training schedule, optimizer parameter grouping,
# evaluation, training inputs, losses or augmentations.
_INIT_RESUME = '''    raw = torch.load(str(args.resume_from), map_location="cpu", weights_only=True)
    if raw.get("variant") != "selective-v1" or raw.get("training_mode") != "joint":
        raise ValueError("Resume checkpoint is not selective-v1/joint")
    if int(raw.get("fold", -1)) != args.fold:
        raise ValueError("Checkpoint fold mismatch")
    start_epoch = int(raw["epoch"])
    if not (STAGE1 <= start_epoch < args.epochs):
        raise ValueError("Resume requires stage-2 checkpoint and remaining epochs")
    model.load_state_dict(raw["model"], strict=True)
    early_params = []
    for block in bb[:FREEZE_BLK]:
        for param in block.parameters():
            param.requires_grad_(True)
            early_params.append(param)
    opt.add_param_group({"params": early_params, "lr": STAGE2_LR * .05})
    if "optimizer" in raw:
        opt.load_state_dict(raw["optimizer"])
        scaler.load_state_dict(raw["scaler"])
        if "python_rng" in raw:
            random.setstate(raw["python_rng"])
        if "numpy_rng" in raw:
            nr = raw["numpy_rng"]
            np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        if "torch_rng" in raw:
            torch.set_rng_state(raw["torch_rng"])
        if torch.cuda.is_available() and raw.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(raw["cuda_rng"])
    best = float(raw["best_dice"])
    unfroze = True
    hist = json.load(open(args.history_prefix, encoding="utf-8"))
    print(f"[V1-RESUME] from best/state epoch={start_epoch}; next={start_epoch+1}; "
          f"through={args.epochs}; best={best:.8f}; "
          f"optimizer_restored={'optimizer' in raw}", flush=True)
'''

_EPOCH_SAVE = '''            print(f"  >> saved best mean={best:.4f}")
        # Keep recoverable progress even if this run is interrupted again.
        history_temp = ckpt_dir / "history.part"
        history_temp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
        os.replace(history_temp, ckpt_dir / "history.json")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            numpy_rng = np.random.get_state()
            snapshot = {
                "epoch": epoch, "fold": args.fold,
                "variant": "selective-v1", "training_mode": "joint",
                "best_dice": best, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(),
                              numpy_rng[2], numpy_rng[3], numpy_rng[4]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            state_temp = ckpt_dir / "last_training_state.part"
            torch.save(snapshot, state_temp)
            os.replace(state_temp, ckpt_dir / "last_training_state.pth")
            print(f"[V1-RESUME] saved full training state at epoch {epoch}", flush=True)
'''

_LOG_RE = re.compile(r"Ep(\d{3}) loss=([\d.]+) \| val RV=([\d.]+) Myo=([\d.]+) LV=([\d.]+) mean=([\d.]+)")


def parse_prior_history(logfile: Path, through_epoch: int) -> list[dict]:
    """Only retain completed history up to the model checkpoint epoch."""
    if not logfile.is_file():
        return []
    text = logfile.read_text(encoding="utf-8", errors="replace")
    found = {}
    for m in _LOG_RE.finditer(text):
        ep = int(m.group(1))
        if 1 <= ep <= through_epoch:
            loss, rv, myo, lv, avg = map(float, m.groups()[1:])
            found[ep] = {"epoch": ep, "loss": loss, "val_RV": rv,
                         "val_Myo": myo, "val_LV": lv, "val_mean": avg}
    if found and sorted(found) != list(range(1, through_epoch + 1)):
        raise ValueError("Original training.log has incomplete epoch results up to checkpoint")
    return [found[k] for k in sorted(found)]


def build_resume_entry(upstream_script: Path, destination: Path) -> Path:
    build_entry(upstream_script, destination)
    content = destination.read_text(encoding="utf-8")
    content = _replace_once(content, _ARG_ANCHOR, '''    ap.add_argument("--resume_from", required=True)
    ap.add_argument("--history_prefix", required=True)
    ap.add_argument("--save_every", type=int, default=2)
    args = ap.parse_args()
    if args.save_every < 1:
        raise ValueError("save_every must be >=1")''')
    content = _replace_once(content, _INIT_ANCHOR, _INIT_RESUME.rstrip("\n"))
    content = _replace_once(content, _LOOP_ANCHOR,
                            "    for epoch in range(start_epoch + 1, args.epochs + 1):")
    content = _replace_once(content, _EPOCH_SAVE_ANCHOR, _EPOCH_SAVE.rstrip("\n"))
    compile(content, str(destination), "exec")
    destination.write_text(content, encoding="utf-8")
    return destination


def _valid_checkpoint(path: Path, fold: int, epochs: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("variant") != "selective-v1" or ckpt.get("training_mode") != "joint":
        raise ValueError("Not a V1 joint-training checkpoint")
    if int(ckpt.get("fold", -1)) != fold:
        raise ValueError("Checkpoint fold mismatch")
    epoch = int(ckpt["epoch"])
    if epoch < 71 or epoch >= epochs:
        raise ValueError(f"Checkpoint epoch must be in [71, {epochs-1}], got {epoch}")
    if not isinstance(ckpt.get("model"), dict):
        raise ValueError("Missing model weights")
    if "optimizer" in ckpt and "scaler" not in ckpt:
        raise ValueError("Optimizer state present but GradScaler state is missing")
    result = {"epoch": epoch, "best_dice": float(ckpt["best_dice"]),
              "has_optimizer": "optimizer" in ckpt}
    del ckpt
    return result


def resume(previous_run: Path, pulse_root: Path, acdc_root: Path, out_dir: Path,
           save_every: int = 2, dry_run: bool = False) -> dict:
    previous_run = previous_run.resolve()
    out_dir = out_dir.resolve()
    if previous_run == out_dir:
        raise ValueError("New output directory MUST differ from interrupted run")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite occupied output dir: {out_dir}")
    if save_every < 1:
        raise ValueError("save_every must be >=1")
    config_file = previous_run / "run_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Original run_config.json is missing: {config_file}")
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    if cfg.get("variant") != "selective-v1" or cfg.get("training_mode") != "joint":
        raise ValueError("Original run must be V1 joint training")
    if Path(cfg["acdc_root"]).resolve() != acdc_root.resolve():
        raise ValueError("ACDC root differs from original V1 run")
    if Path(cfg["pulse_root"]).resolve() != pulse_root.resolve():
        raise ValueError("PULSE root differs from original V1 run")
    if int(cfg["epochs"]) != 120 or int(cfg["fold"]) != 0:
        raise ValueError("Expected original V1 fold0/120 config; inspect before adapting")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError("ACDC training dataset missing")
    upstream_file = pulse_root / "pulse" / "pulse_seg.py"
    if not upstream_file.is_file():
        raise FileNotFoundError(f"PULSE source missing: {upstream_file}")
    upstream_hash = file_sha256(upstream_file)
    if upstream_hash != cfg.get("source_sha256"):
        raise ValueError("PULSE upstream source does not match interrupted run")
    # The best model is the only durable save in the original 120-epoch runner.
    # To resume a later state use a 'last_training_state.pth' explicitly in a
    # previous recovery directory, where best_model.pth is also available.
    snapshot = previous_run / "last_training_state.pth"
    resume_from = snapshot if snapshot.exists() else previous_run / "best_model.pth"
    info = _valid_checkpoint(resume_from, int(cfg["fold"]), int(cfg["epochs"]))
    original_best = previous_run / "best_model.pth"
    if not original_best.is_file():
        raise FileNotFoundError(f"Missing best-model copy for continuity: {original_best}")
    prior_history_file = previous_run / "history.json"
    if prior_history_file.exists():
        raw_history = json.loads(prior_history_file.read_text(encoding="utf-8"))
        previous_hist = [row for row in raw_history if int(row["epoch"]) <= info["epoch"]]
    else:
        previous_hist = parse_prior_history(previous_run / "training.log", info["epoch"])
    prefix_complete = len(previous_hist) == info["epoch"]
    report = {"previous_dir": str(previous_run), "new_dir": str(out_dir),
              "resume_checkpoint": str(resume_from),
              "resumed_from_epoch": info["epoch"],
              "next_epoch": info["epoch"]+1,
              "final_epoch": int(cfg["epochs"]),
              "best_dice_at_start": info["best_dice"],
              "optimizer_restored": info["has_optimizer"],
              "exact_resume": False,
              "note": ("Optimizer moments/RNG absent: model-only stage-2 warm restart" if not info["has_optimizer"]
                       else "State checkpoint restored; dataset rebuild may affect exact batch order"),
              "original_history_complete_through_checkpoint": prefix_complete,
              "history_rows_recovered": len(previous_hist),
              "batch_size": int(cfg["batch_size"]), "lr": float(cfg["lr"]),
              "seed": int(cfg["seed"]), "save_every": save_every}
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="v1_resume_preflight_") as temporary:
            build_resume_entry(upstream_file, Path(temporary) / "resume_entry.py")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    out_dir.mkdir(parents=True, exist_ok=False)
    entry = out_dir / "joint_v1_resume_entry.py"
    build_resume_entry(upstream_file, entry)
    (out_dir / "prior_history.json").write_text(json.dumps(previous_hist, indent=2), encoding="utf-8")
    (out_dir / "recovery_config.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Seed the new run's best_model from previous best; never modify old files.
    staged = out_dir / "best_model.part"
    shutil.copy2(original_best, staged)
    os.replace(staged, out_dir / "best_model.pth")
    cmd = [sys.executable, str(entry), "--fold", str(cfg["fold"]),
           "--nfolds", "5", "--epochs", str(cfg["epochs"]),
           "--bs", str(cfg["batch_size"]), "--lr", str(cfg["lr"]),
           "--seed", str(cfg["seed"]), "--ckpt_dir", str(out_dir),
           "--data_root", str(acdc_root), "--resume_from", str(resume_from),
           "--history_prefix", str(out_dir / "prior_history.json"),
           "--save_every", str(save_every)]
    env = os.environ.copy()
    env["ACDC_ROOT"] = str(acdc_root)
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH", "")]))
    new_cfg = dict(cfg)
    new_cfg.update({"resume_epoch": info["epoch"],
                    "resume_checkpoint": str(resume_from),
                    "run_type": "V1 weight/state continuation (non-identical to uninterrupted training)",
                    "command": cmd})
    (out_dir / "run_config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[V1-RESUME] executing", " ".join(cmd), flush=True)
    start = time.perf_counter()
    with (out_dir / "training.log").open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(pulse_root), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        code = proc.wait()
    (out_dir / "run_status.json").write_text(json.dumps({"exit_code": code,
        "elapsed_seconds": time.perf_counter()-start,
        "model_only_warm_restart": not info["has_optimizer"],
        "checkpoint_epoch": info["epoch"]}, indent=2), encoding="utf-8")
    if code != 0:
        raise RuntimeError("Recovered V1 training failed; inspect new training.log")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resume(**vars(args))


if __name__ == "__main__":
    main()
