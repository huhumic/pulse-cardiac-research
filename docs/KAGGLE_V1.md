# Kaggle V1 runbook

**Do not run the V1 training concurrently with the ongoing PULSE 120-epoch
training on the same T4.** The development branch never modifies upstream PULSE
or the existing experiment outputs.

## Step 1 — pull development branch

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin feature/selective-structure-v1
!git switch feature/selective-structure-v1
!python -m pytest -q tests/test_selective_v1.py
```

If the research project was not cloned:

```python
%cd /kaggle/working
!git clone --branch feature/selective-structure-v1 \\
    https://github.com/huhumic/pulse-cardiac-research.git pulse-cardiac-v1
%cd /kaggle/working/pulse-cardiac-v1
!bash scripts/setup_kaggle.sh
!python -m pulse_repro.prepare_acdc \\
    --input-root /kaggle/input \\
    --output-root /kaggle/working/ACDC_pulse/database
```

## Step 2 — initialise from an already completed fold_0 checkpoint

First smoke-test the *downloaded official* checkpoint; it is safe and does not
alter the original. It must already exist in your Kaggle session.

```python
!python -m pulse_repro.train_selective_v1 \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE \\
  --init-weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --fold 0 --epochs 1 --steps-per-epoch 3 --batch-size 2 \\
  --out-dir /kaggle/working/pulse_results/v1_smoke_fold0
```

If you prefer your own trained PULSE checkpoint, change only `--init-weights`;
verify its `fold` metadata is 0 and its training completed.

## Step 3 — evaluate the smoke checkpoint (two volumes only)

```python
!python -m pulse_repro.evaluate \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE --fold 0 \\
  --model-variant selective-v1 \\
  --weights /kaggle/working/pulse_results/v1_smoke_fold0/best_model.pth \\
  --out-dir /kaggle/working/pulse_results/v1_smoke_eval \\
  --max-patients 1 --tta
```

This is a **functional check**, not an independent effectiveness result.

## Step 4 — research training: only after all tests pass

```python
!python -m pulse_repro.train_selective_v1 \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE \\
  --init-weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --fold 0 --epochs 15 --batch-size 4 \\
  --out-dir /kaggle/working/pulse_results/v1_fold0_e15
```

Then a full comparable 20-patient validation:

```python
!python -m pulse_repro.evaluate \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE --fold 0 --tta \\
  --model-variant selective-v1 \\
  --weights /kaggle/working/pulse_results/v1_fold0_e15/best_model.pth \\
  --out-dir /kaggle/working/pulse_results/v1_fold0_e15_eval
```

Never reuse an existing `--out-dir`; the trainer refuses to overwrite earlier
checkpoints.

## Expected files

- `best_model.pth`: complete PULSE weights plus V1 adapters.
- `history.csv`: sampled-slice training validation curve.
- `run_config.json`: fold, seed, source checkpoint SHA256, patient split.
- `run_status.json`: training completion code and elapsed time.
- Evaluation: `per_volume.csv`, `per_patient.csv`, `summary.json`.

Current status: code unit tests passed offline. **No Kaggle V1 training or
end-to-end inference result is claimed yet.**
