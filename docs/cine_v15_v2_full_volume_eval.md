# Cine V1.5 / V2: full 3D ED/ES Fold-0 evaluation

This evaluates an already trained **best_model.pth**, not the latest checkpoint.
No training, scientific model modification, or extra labels are needed.

## Contract

- 100 ACDC labeled training patients; patient-disjoint stratified Fold 0 with seed 42.
- Exactly 20 validation patients and 40 labeled complete ED/ES volumes.
- Actual 4D Cine temporal neighbours from each patient's NIfTI; verify 4D-to-ED/ES indexed phase identity (not spatial neighbour or generated label).
- Single model; four H/V flip TTA synchronized for 3-channel spatial and 2-channel temporal input.
- Average softmax probabilities, invert flips, argmax, resize labels to original 3D volume, PULSE original largest-component postprocessing.
- Report RV/Myo/LV full-3D Dice, patient-first mean; optionally surface metrics, which require an independent spatial-mm geometry audit.
- V1.5 and V2 checkpoints share a checkpoint format; use_structure=false identifies V1.5.
- The Fold-0 validation subset also selected the best checkpoint: results are validation metrics, NOT untouched test/5-fold ensemble or independent significance evidence.

## On AutoDL

From the project environment:

~~~bash
source /root/autodl-tmp/pulse_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/autodl-tmp/pulse-cardiac-v2-synced

# Check for uncommitted local edits before updating your existing branch.
git status --short
git fetch origin feature/cine-temporal-reliability-v2
git pull --ff-only origin feature/cine-temporal-reliability-v2

# CPU-only synthetic correctness tests. These do not load checkpoints.
python -m pytest tests/test_evaluate_cine_v2.py -q
~~~

Run a 1-patient smoke test of the V2 E91 best checkpoint (2 whole volumes):

~~~bash
python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --weights /root/autodl-tmp/pulse_results/v2_cine_bf16_fold0_e120/best_model.pth \
  --out-dir /root/autodl-tmp/pulse_results/v2_fullvolume_smoke \
  --fold 0 --tta --precision fp32 --max-patients 1
~~~

If this succeeds, remove --max-patients and use a fresh output directory:

~~~bash
python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --weights /root/autodl-tmp/pulse_results/v2_cine_bf16_fold0_e120/best_model.pth \
  --out-dir /root/autodl-tmp/pulse_results/v2_e91_fold0_fullvolume \
  --fold 0 --tta --precision fp32
~~~

For V1.5, after training is finished:

~~~bash
python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --weights /root/autodl-tmp/pulse_results/pulse_temporal_only_fold0_e120/best_model.pth \
  --out-dir /root/autodl-tmp/pulse_results/v15_best_fold0_fullvolume \
  --fold 0 --tta --precision fp32
~~~

### Output

- per_volume.csv: 40 individual full-3D RV/Myo/LV Dice results, plus QA diagnostics.
- per_patient.csv: the ED+ES patient means.
- summary.json: variant (V1.5/V2), selected epoch, mean_dice, per_patient_means, checkpoint SHA-256, full_fold, and protocol metadata.
- With --save-masks: per-patient ED/ES predicted 3D NIfTI volumes.

The default is FP32 evaluation, avoiding implicit CUDA FP16. Optional --precision bf16 is supported on compatible CUDA devices; use **the same** precision for comparisons. The CLI requires --tta explicitly to reproduce the four-flip evaluation protocol.

### Read the final summary

~~~bash
python - <<'PY'
import json
from pathlib import Path
base=Path("/root/autodl-tmp/pulse_results")
for d in ["v2_e91_fold0_fullvolume","v15_best_fold0_fullvolume"]:
    p=base/d/"summary.json"
    if not p.exists():
        print(d, "NOT YET EVALUATED")
        continue
    x=json.loads(p.read_text())
    print(d, "variant=",x["variant"],"epoch=",x["checkpoint_epoch"],
          "3D mean=",f'{x["mean_dice"]:.6f}',
          "classes=",{k:x["per_patient_means"][f"{k}_dice"] for k in ("RV","Myo","LV")})
PY
~~~

For ablations only: --temporal-control swap or --temporal-control repeat-target; use separate output directories and do not substitute these scores for the normal correct-time result.
