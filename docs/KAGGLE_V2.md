# V2 / V3: real cardiac-time reliability — Kaggle procedure

This branch is a **research implementation, not a validated model result**.
No currently running V1 run is modified or interrupted. First determine whether
this Kaggle mirror contains full, unlabeled, patient-level **4D cine** scans.

## 0. Pull the code (safe; do this after the V1 run ends)

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin feature/cine-temporal-reliability-v2
!git switch feature/cine-temporal-reliability-v2
!python -m pytest -q tests/test_cine_v2.py
```

## 1. **Mandatory** 4D preflight before any GPU training

```python
!python -m pulse_repro.cine_data \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --output /kaggle/working/pulse_results/cine_preflight.json \
  --require-all
```

Expected `ACDC 4D cine: 100/100 patients`. If it reports 0/100 or an
incomplete count, **STOP**. A 3D ED/ES image or adjacent Z-slice is NOT a
cardiac time neighbor. Obtain the corresponding complete cine series for the
same 100 labeled training patients. Do not pretend fake temporal inputs are a
valid V2 experiment.

## 2. Smoke test (1 epoch, 3 steps)

Only after preflight succeeds:

```python
!python -m pulse_repro.train_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --fold 0 --epochs 1 --max-steps 3 --batch-size 2 \
  --out-dir /kaggle/working/pulse_results/cine_v2_smoke_fold0
```

Evaluation smoke test on two ED/ES volumes:

```python
!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_smoke_fold0/best_model.pth \
  --fold 0 --max-patients 1 --tta \
  --out-dir /kaggle/working/pulse_results/cine_v2_smoke_eval
```

## 3. Full 120 epoch experiment

```python
!python -m pulse_repro.train_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 \
  --out-dir /kaggle/working/pulse_results/cine_v2_fold0_e120
```

If CUDA OOM, change `--batch-size 2` AND a **new output directory** and record
this change, because the baseline batch size of 4 then differs. One T4 is
used; two T4s do not pool GPU memory.

Finally evaluate all 20 validation patients (40 ED+ES 3D volumes):

```python
!python -m pulse_repro.evaluate_cine_v2 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/cine_v2_fold0_e120/best_model.pth \
  --fold 0 --tta \
  --out-dir /kaggle/working/pulse_results/cine_v2_fold0_eval
```

## Research contract

- Spatial branch: PULSE standard `[z-1,z,z+1]` for the target ED/ES frame.
- Time branch: true 4D cine `[t-1,t+1]` at the **same slice z**.
- ACDC frame numbering from `Info.cfg` is **1-based**, 4D index **0-based**;
  time wraps cyclically at beginning/end.
- Both temporal neighbors are **unlabeled**; no fake masks are assigned.
- M2: per-neighbor feature alignment and 3-way reliability softmax (`skip` / before / after).
- M3: Jensen-Shannon agreement **between two target-aligned evidence probes**;
  probe CE uses only labeled target ED/ES, not unlabeled neighbors.
- Main / aux supervised loss and PULSE two-stage schedule are preserved.
- All metrics are computed on **unseen fold_0 patients**, using same 3D
  Dice/HD95/ASSD and four-flip TTA protocol as the original PULSE comparison.
- Runs with `--max-steps` are smoke tests, never publishable results.
- No claims of novelty/effectiveness before controlled comparisons.

## Thesis connection

DCR-MI's target-state/bidirectional-evidence/reliability concepts motivate the
segmentation analysis, but its interpolation weights are **not** implicitly
used or needed by V2. A later data-leakage-safe experiment can test DCR-MI
interpolated frames as additional temporal context. Never make a false claim
that interpolated frames carry manually annotated segmentation ground truth.
