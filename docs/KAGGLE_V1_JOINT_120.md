# V1 complete 120-epoch training: Kaggle T4

## Design and comparison

- This is **end-to-end** PULSE + V1 selective skip-fusion training.
- It starts from the **DINOv2 pretraining** used by the original PULSE, **not** from its already segmentation-finetuned fold checkpoint. No `--init-weights` argument is accepted.
- Original `pulse_seg.py` is copied to a temporary file. The upstream original is never modified.
- The temporary training source changes only the variant model, checkpoint metadata, and empty-test reporting. The upstream dataloader, augmentation, optimizer setup, deep supervision loss, 70+50 epoch schedule and sampled-slice validation stay intact.
- The Kaggle folder `/kaggle/working/pulse_results/` stores all checkpoints and results; **do not commit images or weights to Git**.
- This project has *not* yet run joint 120-epoch training on Kaggle; unit tests only.

## Kaggle: after finishing the original PULSE training

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin feature/selective-structure-v1
!git switch feature/selective-structure-v1
!git pull --ff-only origin feature/selective-structure-v1
!python -m pytest -q tests/test_joint_v1.py tests/test_selective_v1.py
```

Keep the original normalized ACDC symlink dataset at:

`/kaggle/working/ACDC_pulse/database`

and original PULSE checkout at:

`/kaggle/working/PULSE`

## Full 120 epoch training (one fold)

```python
!python -m pulse_repro.train_joint_v1 \
    --acdc-root /kaggle/working/ACDC_pulse/database \
    --pulse-root /kaggle/working/PULSE \
    --fold 0 --epochs 120 --batch-size 4 --seed 42 \
    --out-dir /kaggle/working/pulse_results/v1_joint_fold0_e120
```

**GPU note:** Kaggle 2 × T4 provide separate memory pools. The upstream PULSE loop uses just one card. The V1 skip modules add activation overhead, so 4 samples per batch may cause OOM even if the original PULSE batch 4 succeeded. In that case start a **new** output directory with `--batch-size 2`. Do not overwrite the failed directory. Batch size changes optimizer behavior and should be documented when comparing methods; ideally rerun both baselines with matched batch settings.

Training follows the original 70 epoch stage 1 + 50 epoch stage 2. If 120 epoch completes successfully, `best_model.pth` and `history.json` will be produced; `run_status.json` contains exit code and elapsed time.

## Model validation on all 20 fold_0 patients / 40 3D volumes

```python
!python -m pulse_repro.evaluate \
    --acdc-root /kaggle/working/ACDC_pulse/database \
    --pulse-root /kaggle/working/PULSE \
    --fold 0 --tta --model-variant selective-v1 \
    --weights /kaggle/working/pulse_results/v1_joint_fold0_e120/best_model.pth \
    --out-dir /kaggle/working/pulse_results/v1_joint_fold0_eval
```

Compare `summary.json` / per-patient CSV with the original fold_0 full-volume results: RV 0.9072, Myo 0.8438, LV 0.9173, mean 0.8895. Those values are from the **published** PULSE fold_0 checkpoint. Comparisons against the user's **own** PULSE 120-epoch run must additionally be made to control for reproduction variability.

Do not interpret this as a **temporal** model: V1 uses adjacent spatial slices. Explicit cardiac-time modeling, needed to establish a stronger connection to DCR-MI, is reserved for V2.
