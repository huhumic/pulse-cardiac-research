# V1 训练意外中断：Epoch 72 权重恢复方案

- 当前记录：fold=0, V1 selective joint, `best_model.pth` = 356.4 MiB, checkpoint epoch=72, best slice-level validation Dice=0.9022130049443691。
- **重要**：原始权重没有 AdamW 的优化器动量、GradScaler 或随机状态。因此这不是精确断点续训，而是从第 72 轮最佳模型进行第二阶段的 **warm restart**；后续最优指标可能与未中断训练不同。论文记录中必须如实说明。
- 不需要重新训练 Epoch 1–72。第二阶段 Stage-2 从 Epoch 71 开始。重新构造三个优化器参数组（未冻结骨干的后续块、其他参数、前两个解冻骨干块），恢复 V1 完整模型权重，从 Epoch 73 继续至 120，遵守 PULSE 第二阶段 LR 设定。
- **绝不复用原输出目录**。原 `best_model.pth`, `training.log`, `run_config.json` 完全保留；输出到新的 `v1_resume_from72_to120`。
- 新训练每 `--save-every`（默认 2）轮写出 `last_training_state.pth`（模型、优化器、scaler、RNG）和 `history.json`，采用临时文件原子替换；若再次中断可用同一脚本指定恢复目录为 `--previous-run`，在新的第三个目录续训。
- 原 `training.log` 的前 72 轮会重新解析并写入历史。如果原日志不完整，会标识历史不完整，不伪造数据。若日志记录了超过 72 轮，由于当时未保存这些轮次的网络权重，恢复报告只包括至 72 轮的历史。

## 1. Kaggle 先确认文件仍在

```python
from pathlib import Path
old = Path('/kaggle/working/pulse_results/v1_joint_fold0_e120')
for name in ('run_config.json','best_model.pth','training.log'):
    p = old / name
    print(name, p.is_file(), p.stat().st_size/2**20 if p.is_file() else None)
```

## 2. 切换安全恢复分支（仅在 V1 已停止后）

```python
%cd /kaggle/working/pulse-cardiac-research
!git fetch origin fix/v1-recovery-training
!git switch fix/v1-recovery-training
!git pull --ff-only origin fix/v1-recovery-training
!python -m pytest -q tests/test_resume_joint_v1.py
```

## 3. 预检：只读，不启动 GPU

```python
!python -m pulse_repro.resume_joint_v1 \
  --previous-run /kaggle/working/pulse_results/v1_joint_fold0_e120 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --out-dir /kaggle/working/pulse_results/v1_resume_from72_to120 \
  --dry-run
```

必须确认报告 `resumed_from_epoch=72`，`next_epoch=73`，`final_epoch=120`，`optimizer_restored=false`。预检会核实原 PULSE 源码 SHA，并编译临时恢复脚本。

## 4. 从 73 轮开始实际续训

```python
!python -m pulse_repro.resume_joint_v1 \
  --previous-run /kaggle/working/pulse_results/v1_joint_fold0_e120 \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --out-dir /kaggle/working/pulse_results/v1_resume_from72_to120 \
  --save-every 2
```

**第一个实际训练打印应为 `[V1-RESUME] ... epoch=72; next=73; through=120 ...`，随后训练进度为 `ep73 train`。** 不应出现 `ep1 train`。

## 5. 续训完后评价

```python
!python -m pulse_repro.evaluate \
  --acdc-root /kaggle/working/ACDC_pulse/database \
  --pulse-root /kaggle/working/PULSE \
  --weights /kaggle/working/pulse_results/v1_resume_from72_to120/best_model.pth \
  --out-dir /kaggle/working/pulse_results/v1_resume_from72_to120_eval \
  --fold 0 --tta --model-variant selective-v1
```

这是一种从第 72 轮训练模型权重继续的实际恢复实验。若将其用于正式论文，与原 baseline 的训练条件差异（优化器重置）需在实验记录中保留；有条件时从头重复完整训练以做独立种子验证。
