# AutoDL RTX 3090 / PULSE V1: fresh 120-epoch train with full-state recovery

This is a **new run** starting at epoch 1 after the previous Kaggle 72-epoch
best checkpoint was lost. Do not treat it as a continuation of that run.
Use a single 24-GB RTX 3090 GPU. Model, loss, patient split, augmentation,
optimizer grouping, and 70+50 schedule follow the pinned PULSE training source.

## 0. Environment and source

Choose an AutoDL PyTorch image with a working CUDA-enabled PyTorch installation
(prefer Python 3.10/3.11 and CUDA 12.x), and verify with `nvidia-smi`. Keep data,
checkpoints and code on the **data disk**, not a disposable system directory.

```bash
mkdir -p /root/autodl-tmp && cd /root/autodl-tmp

git clone https://github.com/huhumic/pulse-cardiac-research.git
cd pulse-cardiac-research
git fetch origin feature/autodl-v1-resumable
git switch feature/autodl-v1-resumable

cd /root/autodl-tmp
git clone https://github.com/BRAIN-Lab-AI/PULSE.git
cd PULSE
git checkout 781e0712ea3b2dd246f86346d73008ecdb73434f

python -m pip install 'transformers>=4.38,<5' 'nibabel>=5,<6' \
  'numpy>=1.24,<2' 'scipy>=1.10' 'pandas>=2' \
  'opencv-python-headless>=4.8' 'albumentations>=1.3,<3' \
  'tqdm>=4.65' 'scikit-learn>=1.3' 'pytest>=7'

python - <<'PY'
import torch, transformers
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')
print('transformers=', transformers.__version__)
assert torch.cuda.is_available(), 'CUDA is mandatory for the planned GPU train'
PY
```

Do NOT run `scripts/setup_kaggle.sh`: it is specific to Kaggle and downloads
an official checkpoint that is not used for training V1 from scratch.

## 1. Provide original ACDC data

Upload the original ACDC database to, e.g.,
`/root/autodl-tmp/datasets/ACDC/database`, with `training/patient001/...`
through `training/patient100/...` and `Info.cfg` + ED/ES/GT NIfTI files.
Input may have nested `.nii` folders as in the Kaggle mirror.
**This guide does not supply or invent a dataset download URL.**

```bash
cd /root/autodl-tmp/pulse-cardiac-research
python -m pulse_repro.prepare_acdc \
  --input-root /root/autodl-tmp/datasets/ACDC/database \
  --output-root /root/autodl-tmp/ACDC_pulse/database
```

Inspect 100 training patients and 200 ED/ES image + GT records. The trainer
checks this strictly. No full 4D cine data are necessary for **V1**; V2 requires
an additional independent 4D preflight.

## 2. Tests + dry run (no GPU training)

```bash
cd /root/autodl-tmp/pulse-cardiac-research
python -m pytest -q tests/test_train_v1_resumable.py tests/test_joint_v1.py tests/test_selective_v1.py

python -m pulse_repro.train_v1_resumable \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 \
  --out-dir /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_e120 \
  --save-every 2 --dry-run
```

## 3. Launch robustly inside tmux

```bash
tmux new -s v1
cd /root/autodl-tmp/pulse-cardiac-research
mkdir -p /root/autodl-tmp/pulse_results/logs
set -o pipefail
python -m pulse_repro.train_v1_resumable \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 --workers 4 \
  --out-dir /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_e120 \
  --save-every 2 2>&1 | tee -a /root/autodl-tmp/pulse_results/logs/v1_3090.log
```

Detach while the command keeps running with `Ctrl+B`, then `D`.
Reattach: `tmux attach -t v1`. Do not delete/release the paid instance
without saving checkpoints elsewhere. **tmux protects against SSH disconnection
only; it cannot protect from instance release, disk deletion or machine failure.**

Each 2-epoch boundary atomically updates `last_state.pth` with model weights,
AdamW state, GradScaler, RNG and training history. `best_model.pth` tracks the
best selected fold checkpoint separately; `history.json` is written each epoch.
The most recent one epoch after a periodic snapshot can be lost on interruption.
Copy `last_state.pth`, `best_model.pth`, `run_config.json`, and `history.json`
regularly to a safe external location. Storage may require multiple gigabytes.

## 4. Resume after interrupted process

**Keep exactly the same run output directory and training settings**.

```bash
cd /root/autodl-tmp/pulse-cardiac-research
python -m pulse_repro.train_v1_resumable \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 \
  --out-dir /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_e120 \
  --save-every 2 --resume --dry-run

# If the dry run reports the correct next epoch, run:
python -m pulse_repro.train_v1_resumable \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --fold 0 --epochs 120 --batch-size 4 --seed 42 --workers 4 \
  --out-dir /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_e120 \
  --save-every 2 --resume 2>&1 | tee -a /root/autodl-tmp/pulse_results/logs/v1_3090.log
```

The resumed model retains optimizer, scaler and PRNG states as saved at the last
completed periodic boundary. This is **full-state epoch-boundary restoration**,
not a guarantee of bitwise-identical CUDA results after a machine interruption.

## 5. Complete fold_0 full-volume evaluation

```bash
cd /root/autodl-tmp/pulse-cardiac-research
python -m pulse_repro.evaluate \
  --acdc-root /root/autodl-tmp/ACDC_pulse/database \
  --pulse-root /root/autodl-tmp/PULSE \
  --fold 0 --model-variant selective-v1 \
  --weights /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_e120/best_model.pth \
  --out-dir /root/autodl-tmp/pulse_results/v1_autodl_3090_fold0_eval \
  --tta
```

Report patient-first mean Dice, physical-space HD95/ASSD and the complete
40-volume fold validation. Fold_0 is **internally selected validation**, not
an independent 50-patient paper test. The original user's PULSE self-trained
fold_0 comparison is 0.888166 for patient-first 3D Dice, evaluated with TTA.

## 6. Auditable conditions

- V1 starts from `facebook/dinov2-base`, **not** the previously trained PULSE
  fold checkpoint.
- Upstream PULSE commit is pinned and verified before GPU training.
- The fold split is patient-disjoint, seed 42, batch size 4; Stage 1 = 70
  epochs, Stage 2 = 50 epochs.
- Original PULSE source and earlier experiment folders are not modified.
- Output is in a new directory; accidental overwrite is explicitly blocked.
- No automatically retrieved weight, old Kaggle cache or pseudo-GT is assumed.
