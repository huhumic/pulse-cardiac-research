"""Launch unmodified upstream 120-epoch PULSE single-fold training.

'epochs=1' is a smoke test only; the original unfreezing schedule has 70+50 epochs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(acdc_root: Path, pulse_root: Path, fold: int, epochs: int,
        batch_size: int, out_dir: Path, dry_run: bool = False) -> None:
    if fold not in range(5) or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid fold/epochs/batch-size")
    if not (acdc_root / "training").is_dir():
        raise FileNotFoundError(f"No normalized ACDC training directory: {acdc_root}")
    if not (pulse_root / "pulse" / "train.py").is_file():
        raise FileNotFoundError(f"No upstream train.py under {pulse_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_target = out_dir / "best_model.pth"
    if weights_target.exists():
        raise FileExistsError(
            f"Would overwrite an existing run: {weights_target}; choose a new --out-dir"
        )
    cmd = [sys.executable, "pulse/train.py", "--fold", str(fold),
           "--nfolds", "5", "--epochs", str(epochs),
           "--bs", str(batch_size), "--ckpt_dir", str(out_dir),
           "--data_root", str(acdc_root)]
    meta = {"command": cmd, "fold": fold, "epochs": epochs,
            "batch_size": batch_size, "acdc_root": str(acdc_root),
            "upstream": str(pulse_root),
            "note": "Official 120-epoch 2-stage training only when --epochs 120"}
    print("Executing:", " ".join(cmd), flush=True)
    if dry_run:
        print(json.dumps(meta, indent=2))
        return
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["ACDC_ROOT"] = str(acdc_root)
    start = time.perf_counter()
    with (out_dir / "training.log").open("w", encoding="utf-8") as f:
        # Live console and saved log (checkpoint and metrics remain at out_dir).
        p = subprocess.Popen(cmd, cwd=pulse_root, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        code = p.wait()
    elapsed = time.perf_counter() - start
    (out_dir / "run_status.json").write_text(
        json.dumps({"exit_code": code, "elapsed_seconds": elapsed}, indent=2),
        encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"Upstream training failed with exit code {code}")
    print(f"Complete after {elapsed:.1f}s. Results in {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acdc-root", type=Path, required=True)
    p.add_argument("--pulse-root", type=Path, default=Path("/kaggle/working/PULSE"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run(**vars(a))


if __name__ == "__main__":
    main()
