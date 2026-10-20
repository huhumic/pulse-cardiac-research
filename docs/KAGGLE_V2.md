# Kaggle V2 — audit revision R2 (direction-aware cardiac-time reliability)

**Status:** source/unit contracts verified offline; **no original ACDC 4D run yet**. Do **not** run alongside ongoing V1; the feature branch does not touch V1 or main.

## Protocol and limitations

- Current-frame input is the official PULSE 2.5D spatial stack `[z−1,z,z+1]` at labeled ED/ES. The distinct temporal input is the **same z slice** of 4D cine frames `t−1` and `t+1`, with cyclic indexing.
- Strict preflight verifies **each of the 100 patients**, 4D dimensions, affine, `NbFrame`, 1-based ED/ES indexing, label geometry and ED/ES image-to-4D indexed-frame normalized correlation. A missing or ambiguous phase is a **hard stop**; check `cine_preflight.json` for reasons.
- ED-anchored periodic target phase `[sin(2π(t−ED)/T),cos(2π(t−ED)/T),1/T]` plus opposite signed time offsets conditions the shared before/after feature processing. This creates direction-sensitive **capacity**, not proof that the trained model uses direction.
- Local feature-pixel pull-back warping returns valid sampling masks. Invalid samples receive negligible temporal weight; the third softmax class rejects both neighbors.
- M3 is reliability-weighted **target ED/ES probe agreement** with optional target-mask probe supervision. It does **not** provide segmentation of unlabeled intermediate frames or use intermediate-frame pseudo-GT.
- No checkpoint for V1 or baseline is modified. ACDC fold_0 is an **internal 20-patient validation fold**, not untouched external test data.

## 0. Pull branch only after V1 finishes

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin feature/cine-temporal-reliability-v2
!git switch feature/cine-temporal-reliability-v2
!git pull --ff-only origin feature/cine-temporal-reliability-v2
!python -m pytest -q tests/test_cine_v2.py tests/test_cine_protocol.py
```

## 1. Mandatory real-data preflight (NO GPU training)

```python
!python -m pulse_repro.cine_data \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --output /kaggle/working/pulse_results/cine_preflight.json \
  --require-all
```

**Required:** `ACDC 4D cine: 100/100 patients` and every patient has verified `ED`, `ES` matching indexed 4D frames. If it reports any failures, STOP, send the `cine_preflight.json` summary. Do not silently switch to spatial-slice neighbors or assume wrong time indices. This validation can take several minutes because it reads actual image voxels.

## 2. Single-epoch 3-step GPU and checkpoint/inference smoke checks

```python
!python -m pulse_repro.train_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --fold 0 --epochs 1 --max-steps 3 --batch-size 2 \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_smoke

!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_R2_smoke/best_model.pth \
  --fold 0 --max-patients 1 --tta \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_smoke_eval
```

`--max-steps` is allowed **only with `--epochs 1`** and must never be interpreted as a full training result.

## 3. Train the fully supervised ED/ES V2 model (120 epochs)

```python
!python -m pulse_repro.train_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_fold0_e120
```

Batch size 4 is the original user's fold_0 baseline setting. If OOM, change both the batch size and output directory; record the resulting protocol difference. Only one T4 is used; GPU VRAM is not pooled.

## 4. Same 20-patient/40-volume full-3D validation protocol

```python
!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_R2_fold0_e120/best_model.pth \
  --fold 0 --tta \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_fold0_eval
```

Reports: volume/patient CSV, `summary.json` including Dice, HD95, ASSD, and mean reliability skip/past/future weights plus per-neighbor valid fractions. **Do not compare training per-slice Dice with full-volume patient-first Dice.** Original user's self-trained PULSE benchmark: fold_0 mean Dice ≈ 0.88817; official released fold_0 ≈ 0.88946, both with same 40-volume 4-TTA evaluation. These are internally selected fold results, not five-fold or external tests.

## 5. Fixed-checkpoint temporal counterfactuals

```python
!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_R2_fold0_e120/best_model.pth \
  --fold 0 --tta --temporal-control swap \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_swap_eval

!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_R2_fold0_e120/best_model.pth \
  --fold 0 --tta --temporal-control repeat-target \
  --out-dir /kaggle/working/pulse_results/cine_v2_R2_repeat_eval
```

`swap` exchanges the two **images** while preserving their before/after direction encodings; `repeat-target` replaces both images with target slice intensities. These are diagnostics only, not alternative independently trained model benchmarks. A reduction in Dice under a temporal control alone does not establish mechanism; inspect gate weights and other controls.

## 6. Independently controlled ablations (separate, independent training runs)

| M1 structure | M2 temporal | M3 JSD | Probe CE | CLI overrides |
|---|---|---|---|---|
| Yes | Yes | Yes | Yes | default Full |
| No | Yes | Yes | Yes | `--no-structure` |
| Yes | No | No | No | `--no-temporal --no-consistency --no-probe-supervision` |
| Yes | Yes | No | Yes | `--no-consistency` |
| Yes | Yes | Yes | No | `--no-probe-supervision` |

Use a **new output directory** per ablation. `--no-consistency` no longer silently turns off the probe supervision term. `--no-temporal` may only run when the two temporal-specific loss terms are independently disabled. Same split, epochs and augmentation are maintained.

## Thesis linkage

The direction- and target-phase-conditioned evidence selection follows a *research idea* related to DCR-MI's target-state and bidirectional evidence formulation. DCR-MI weights and interpolated images **are not used** in this V2 implementation. Demonstrating the connection scientifically requires temporal-controls/ablations and, later, a leakage-safe study of DCR-MI-generated frames. No accuracy or generalization improvement is claimed before Kaggle results.
