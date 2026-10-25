# Cine V1.5/V2 Fold-0 full-volume validation (2026-10-25)

This runner uses existing trained checkpoints and the unchanged `evaluate_cine_v2.py`. No training.

**Protocol:** ACDC 100 labeled subjects, patient-stratified Fold 0 (80 training, 20 validation, seed 42), 40 complete ED/ES volumes, four synchronized flip TTA, FP32, original PULSE largest-component postprocessing. Report RV/Myo/LV complete 3D Dice; ED/ES mean within patient, then patient mean, then class macro mean. Fold 0 was used to select checkpoints: **internal validation, not independent test**.

From your existing AutoDL environment:

```bash
set -euo pipefail
source /root/autodl-tmp/pulse_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/autodl-tmp/pulse-cardiac-v2-synced
git status --short
git fetch origin feature/cine-fullvolume-validation-20261025
git switch --track origin/feature/cine-fullvolume-validation-20261025
python -m pytest tests/test_evaluate_cine_v2.py tests/test_validate_cine_fold0.py -q

ACDC=/root/autodl-tmp/ACDC_pulse/database
PULSE=/root/autodl-tmp/PULSE
V15=/root/autodl-tmp/pulse_results/pulse_temporal_only_fold0_e120/best_model.pth
V2=/root/autodl-tmp/pulse_results/v2_cine_bf16_fold0_e120/best_model.pth
OUT=/root/autodl-tmp/pulse_results/cine_fullvolume_validation_20261025
ls -lh "$V15" "$V2" "$ACDC/adapter_manifest.json"

# one patient (two ED/ES volumes) per model; not a formal result:
python -m pulse_repro.validate_cine_fold0 --acdc-root "$ACDC" --pulse-root "$PULSE" --v15-weights "$V15" --v2-weights "$V2" --out-dir "$OUT" --smoke

# full 20 patients / 40 volumes per model:
python -m pulse_repro.validate_cine_fold0 --acdc-root "$ACDC" --pulse-root "$PULSE" --v15-weights "$V15" --v2-weights "$V2" --out-dir "$OUT"
```

The preflight checks checkpoint `variant`, `fold`, `protocol_version`, `use_structure`, `use_temporal`, and best epochs E84 (V1.5) / E91 (V2). A mismatch is an error, not an automatic fallback.

**Outputs:** `$OUT/full/V15_correct/` and `$OUT/full/V2_correct/` each contain `summary.json`, `per_volume.csv`, `per_patient.csv`; `$OUT/validation_manifest.json` has SHA-256 provenance and paired patient bootstrap confidence intervals. Upload these when finished.

**Optional V2 temporal input controls** after the main run (reuse verified results):

```bash
python -m pulse_repro.validate_cine_fold0 --acdc-root "$ACDC" --pulse-root "$PULSE" --v15-weights "$V15" --v2-weights "$V2" --out-dir "$OUT" --resume --controls v2
```

`swap` swaps past and future channels with phase code unchanged; `repeat-target` repeats current image as both temporal channels. This is input sensitivity, not motion registration accuracy. `--controls both` adds the controls for V1.5.

To compare **paired** with existing V1 or PULSE, add `--baseline-v1 /path/to/per_volume.csv` and/or `--baseline-pulse /path/to/per_volume.csv` with the **same** patient IDs and ED/ES records.

Do not use sampled-slice `val_mean` as 3D Dice. Do not claim statistical independence from the checkpoint-selected fold.
