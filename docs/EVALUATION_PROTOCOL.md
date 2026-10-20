# PULSE / V1 / V2: frozen ACDC evaluation protocol (v1)

**Purpose.** Compare matched patient-level 3D segmentation results without changing any training script or checkpoint. This branch adds **only** two Python utilities, tests, and documentation. No Git checkout or writes are required while V1 is training.

## 1. Scientific scope and denominator

- Dataset: ACDC original **100 labeled training patients**, each with labeled **ED + ES**; the *unlabeled official 50-patient test* is not inferred from the labels here.
- Split: upstream PULSE `stratified_kfold_pids(records,5,fold,seed=42)`; **80 train, 20 validation per fold**, repeated across `fold=0..4` with each patient in validation once.
- Checkpoint selection: upstream PULSE and V1 select best epoch by **sampled 2D slice validation Dice**; V2 follows the same recipe. This selection uses that fold's 20 validation patients. Therefore **the final 20-patient full-volume score is internal validation, not independent test**. Concatenating all five folds produces *cross-validation aggregate*, still not untouched external performance.
- Input: PULSE 2.5D uses **spatial adjacent slices**, NOT temporal cine frames. V2 additionally uses truly temporal neighboring cine images and must pass its own strict 4D preflight. External data inputs differ; disclose this in the paper.
- Comparison: same fold, same patient IDs, same ED/ES phases, full **4-flip TTA** and post-processing, same binary volume metric implementation. Do not transplant unmatched published ACDC tables into the quantitative results.

## 2. Metrics and aggregation

The versioned source implementation lives in `pulse_repro/metrics.py` (`measure_binary`) and `pulse_repro/evaluate.py` / `evaluate_cine_v2.py`.

- Evaluate each **whole 3D phase volume** at its original voxel resolution, with spacing in millimeters read from NIfTI affine.
- RV(1), Myo(2), LV(3); binary Dice: `2 |P∩G| / (|P| + |G|)`.
- Symmetric physical-space **HD95**: percentile 95 of the *concatenated* surface-to-surface nearest distances in both directions. **ASSD**: arithmetic mean of that same concatenated distance array. These are implementation-specific surface definitions; other publications may use differing conventions.
- Exclude background from macro Dice.
- **Patient-first**: each patient's ED and ES scores are averaged, then each class is averaged across patients; **Mean Dice** = mean of the three patient-first class Dice means (equivalently patient macro mean if all labels defined). No slice weighting.
- `both_empty`: undefined Dice / HD95 / ASSD. `one_empty`: Dice=0, undefined surface distances. Never impute a finite surface metric for these cases.
- For formal paired **surface metrics**, a patient is complete only when **both ED and ES** yield finite distances. Missing patients cause failure by default. `--allow-undefined-surface` explicitly reports and excludes incomplete pairs for that metric (reduced denominator). Dice requires complete pairs and cannot silently drop patients.
- Inference seconds and memory usage should be measured separately on the same GPU and inference batch/TTA setting. Exclude first-call setup, warm up consistently, report sample count and device; results should not be conflated with different backbones, full vs light input and batch size.

## 3. Pairing, confidence intervals, and interpretation

- Pair by `(fold,patient,phase)`, then average ED/ES within each patient.
- For each metric define `delta = candidate_mean - reference_mean`. For Dice **positive delta is beneficial**; for HD95/ASSD negative delta is beneficial.
- Nonparametric paired bootstrap: **patient is resampling unit**, 10,000 resamples with replacement by default, seed `20261021`. Report percentile 95% CI for *mean paired delta*. Bootstrap re-samples patients, never individual slices or ED/ES volumes separately.
- Five-fold reports require exactly 100 unique validation patients and all fold IDs 0–4; one-fold comparison uses 20 patients. Fivefold pooling is not 5 separate independent draws of training; report patients and fold dispersion if relevant.
- A CI excluding zero on **internally selected validation checkpoints** does not establish prospective / unbiased performance; external or nested test data are needed for stronger inference. No automatic p-values or acceptance-probability claims.

## 4. Kaggle commands — safe after V1 training finishes

### Pull isolated tools branch (will not rewrite V1 results)

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin feature/evaluation-protocol-v1
!git switch feature/evaluation-protocol-v1
!git pull --ff-only origin feature/evaluation-protocol-v1
!python -m pytest -q tests/test_experiment_protocol.py
```

### Freeze actual patient split in JSON

```bash
python -m pulse_repro.experiment_report freeze-split \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --out-dir /kaggle/working/pulse_results/frozen_eval_protocol
```

Produces `acdc_fivefold_manifest.json`: all train/val patient IDs, upstream script SHA256 and canonical patient/phase/group digest. Do this **before** running more folds. Never edit the manifest to match a desired number.

### Existing self-trained PULSE fold0

```bash
python -m pulse_repro.experiment_report audit \
  --eval-dir /kaggle/working/pulse_results/selftrained_fold0_eval \
  --split-manifest /kaggle/working/pulse_results/frozen_eval_protocol/acdc_fivefold_manifest.json
```

### Compare independent PULSE and V1 120-epoch runs (once V1 full-volume evaluation finishes)

```bash
python -m pulse_repro.experiment_report compare \
  --reference /kaggle/working/pulse_results/selftrained_fold0_eval \
  --candidate /kaggle/working/pulse_results/v1_joint_fold0_eval \
  --split-manifest /kaggle/working/pulse_results/frozen_eval_protocol/acdc_fivefold_manifest.json \
  --bootstrap 10000 \
  --out-dir /kaggle/working/pulse_results/paired_pulse_vs_v1_fold0
```

Output `paired_summary.json`, `paired_per_patient_delta.csv`, and publication-friendly `table_paired_metrics.csv` (all three structures; Dice / HD95 / ASSD; macro Mean Dice). A run without a complete 20-patient `per_volume.csv` and `summary.json` is automatically rejected.

### For five-fold datasets only

```bash
python -m pulse_repro.experiment_report aggregate \
  --eval-dirs /kaggle/working/pulse_results/pulse_fold0_eval \
              /kaggle/working/pulse_results/pulse_fold1_eval \
              /kaggle/working/pulse_results/pulse_fold2_eval \
              /kaggle/working/pulse_results/pulse_fold3_eval \
              /kaggle/working/pulse_results/pulse_fold4_eval \
  --out-dir /kaggle/working/pulse_results/pulse_fivefold_aggregate
```

This command **does not** train folds or fabricate fold paths. Substitute the exact folders you really generated. Add `--split-manifest <path>` when freezing the patient split.

### V2 temporal counterfactuals

Compare normal vs `swap` or `repeat-target` evaluations only when checkpoint hash matches, and explicitly append `--allow-counterfactual`. Counterfactual differences are **diagnostic**, not independently trained method rankings. Add `--allow-undefined-surface` only if the output explicitly declares reduced complete-pair denominators.

## 5. Required method metadata and experimental honesty

Record each run's `git commit`, exact configuration, fold, seed, checkpoint SHA256, best epoch, batch size, original ACDC root, DINOv2 initialization, 70/50 stages, optimizer, losses, effective TTA, dataset preparation, GPU model, and software versions. For comparison with nnU-Net/CENet/other methods, write adapters to this CSV schema and prove identical patient IDs / time frames / spatial units; do not import a score from their paper and call it matched.

**Research gates:** M1 measurable vs PULSE; M2 vs same-backbone naïve temporal fusion; M2 reliability vs no-reject gate; M3 vs identical model without consistency; correct-time vs counterfactual at the same checkpoint; external multi-site cine if available, otherwise state data limitations. Target numbers (90%, 90.5%, etc.) are planning goals only—not fabricated results or journal requirements.
