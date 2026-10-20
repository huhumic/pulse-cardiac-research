# PULSE Cardiac Research — Kaggle Reproduction Harness

Independent reproduction harness for the public [PULSE](https://github.com/BRAIN-Lab-AI/PULSE) cardiac cine-MRI segmentation model. **Does not alter the upstream PULSE model or its checkpoints.**

This project is for a controlled baseline before any research-model modifications. All evaluation outputs are saved as CSV/JSON; weights and patient data are excluded from Git.

## What has been validated by the user (Kaggle T4)

- `fold_0` model checkpoint downloaded: ~356 MiB.
- ACDC Kaggle dataset located at `/kaggle/input/datasets/ceenen/acdc-original-dataset/ACDC/database`.
- Its `.nii` *folders* contain actual NIfTI files, unlike the layout expected by upstream PULSE.
- One training-set patient `patient001` generated labels successfully; three-class mean Dice 0.9181 on this **training** patient (NOT an independent result).
- One T4, one fold, four-flip TTA: 1.97 s stable inference for shape `(216, 256, 10)`; PyTorch allocated memory peak 0.58 GiB.
- Official stratified `fold_0` (seed 42) comprises 80 training / 20 validation patients; `patient001` belongs to training and must not contribute to validation results.
- **Independent fold_0 validation completed by user, single checkpoint, 4-flip TTA:** 20 patients / 40 ED–ES volumes; RV Dice **0.9072**, Myo Dice **0.8438**, LV Dice **0.9173**, patient-averaged mean Dice **0.8895**. RV/Myo/LV HD95 **1.09/1.15/1.20 mm**, mean inference time **2.28 s/volume**. These results came from the user's earlier Kaggle evaluation Cell 10, not from the new harness; the new harness must reproduce them within metric/method tolerance before experiments.

## Start in Kaggle (GPU enabled, Internet on for first setup)

In an existing Kaggle notebook, with ACDC Original Dataset attached as Input:

Open `notebooks/kaggle_run.ipynb` and run the cells from top to bottom. The clone cell supports both public and private repositories. For a **private** repository, configure a Kaggle notebook Secret named `GITHUB_READ_TOKEN` holding a fine-grained, **read-only Contents** GitHub token restricted to this repo. The notebook passes it to Git only as a subprocess environment value; it never prints the token. No token is needed for a public repo.

The current GitHub repository is **public**; no GitHub token is needed. The equivalent manual commands are:

```python
%cd /kaggle/working
!git clone https://github.com/huhumic/pulse-cardiac-research.git
%cd /kaggle/working/pulse-cardiac-research
!bash scripts/setup_kaggle.sh
```

If you already downloaded PULSE and its weight in `/kaggle/working/PULSE`, the setup script will **reuse** them (no repeat 356 MiB download). Upstream code is pinned to commit `781e0712ea3b2dd246f86346d73008ecdb73434f` for clean clones; an existing upstream clone is not automatically changed.

Prepare symlink-based ACDC training files (no full dataset copy). Test patients are not needed for 5-fold validation:

```python
!python -m pulse_repro.prepare_acdc --input-root /kaggle/input --output-root /kaggle/working/ACDC_pulse/database
```

Full independent validation of the single **fold_0 checkpoint**, **20 patients × ED/ES = 40 volumes**, evaluated in physical units with *per-volume* and *per-patient* CSVs:

```python
!python -m pulse_repro.evaluate --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE --fold 0 --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \
  --out-dir /kaggle/working/pulse_results/fold0 --tta
```

Fast single-fold/no-TTA speed + correctness experiment:

```python
!python -m pulse_repro.benchmark --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \
  --patient patient014 --phase ED --mode no-tta --repeats 5 --out /kaggle/working/pulse_results/benchmark.json
```

First do a **single-fold smoke train** to check the current GPU runtime. By default this would run a full epoch, not a fixed number of optimization steps:

```python
!python -m pulse_repro.train_official --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE --fold 0 --epochs 1 --batch-size 4 \
  --out-dir /kaggle/working/pulse_results/smoke_fold0
```

The Kaggle challenge testing partition normally lacks public reference labels. The training wrapper writes an audit-visible copy of the upstream entrypoint in the run directory and adds a guard only around the **final test reporting** block, skipping evaluation when no labeled test cases exist. The original PULSE files and all training/validation/checkpoint logic remain unchanged. This avoids a false failure after the last epoch. For official method training, change to `--epochs 120` and independently reproduce each requested fold. `--epochs 1` is only a smoke test; it does not reproduce the original two-stage 120-epoch curriculum. The upstream model uses one GPU; the second Kaggle T4 remains idle unless explicit distributed training is implemented.

## Important protocol distinctions

1. **No leakage:** `fold_0` can be independently evaluated only on its 20 validation patients. A training patient is not a validation result. A 5-fold ensemble is not the same as 5 out-of-fold predictions.
2. **Data provenance:** Inputs are original patient DICOM-converted NIfTI images+GT with a third-party Kaggle nesting convention. The adapter only creates symlinks; it does not interpolate, resample or change labels.
3. **Metrics:** This harness calculates full-volume, physical-spacing Dice, HD95 and ASSD with a patient-mean protocol. It is **not** identical to upstream's sampled-slice validation metric or the published five-fold ensemble score.
4. **Missing structure:** If a class is absent in either prediction or GT, its HD95/ASSD is **undefined** and flagged. Such cases must be reported before drawing statistical conclusions.
5. **Runtime:** The benchmark excludes model load and disk input; it reports separately timed steady-state inference. `--tta` uses the upstream 4-flip predictor. `--mode no-tta` uses a clearly labeled non-paper fast inference variant.
6. **License/data:** Keep downloaded ACDC patient data and official checkpoints outside Git and honor upstream model/dataset licenses. This code does not include PULSE upstream code or copyrighted patient data.
7. **External evaluation:** Do not use the independent held-out test partition for architecture tuning. The scope of this initial harness is fold_0 validation of the known released checkpoint.

## Repository layout

```
scripts/setup_kaggle.sh          # fetch upstream code & one published fold checkpoint
pulse_repro/prepare_acdc.py      # nested NIfTI => PULSE-compatible symlink tree
pulse_repro/metrics.py           # full-volume Dice, HD95, ASSD, empty-case flags
pulse_repro/runtime.py           # official TTA and a fast, explicitly separated no-TTA mode
pulse_repro/evaluate.py          # strict fold 0 validation, patient-level statistics, CSV/JSON
pulse_repro/benchmark.py         # warm/cold speed and CUDA memory audit
pulse_repro/train_official.py    # upstream training with isolated empty-test reporting guard
notebooks/kaggle_run.ipynb       # cells for all steps
requirements-kaggle.txt         # lightweight evaluation dependencies
```

## Versioning

- Baseline files and model weights should be frozen before writing a novel architecture.
- Store seed, fold, upstream commit, model SHA256, dataset mapping manifest, and evaluation settings with each run.
- Never overwrite checkpoints from an earlier experiment.
