# Kaggle 运行指南（PULSE 官方基线）

本工程是 PULSE 2026 单折/五折复现实验的**独立实验脚手架**，不覆盖 PULSE 原始代码、模型权重和患者影像。现阶段只评估 fold_0 已公开权重的 20 名验证患者，共 40 个 ED/ES MRI 体积。

## 0. 先完成 GitHub 设置

在 https://github.com/new 以 `huhumic` 账号创建 `pulse-cardiac-research`，然后在 https://github.com/settings/installations 找到 ChatGPT/OpenAI 的 GitHub App，Configure 后把新仓库加到授权范围。

**私有仓库需要另外解决 Kaggle 拉取权限。** Kaggle 的 Git 环境不会自动继承 ChatGPT 的 GitHub 授权。推荐在 GitHub 创建仅有该仓库 **Contents: Read-only** 权限的 fine-grained personal access token，在 Kaggle Notebook 的 Add-ons / Secrets（具体名称以界面为准）中添加 `GITHUB_READ_TOKEN`。不要在 Notebook 代码里直接写 Token，也不要发给聊天窗口。若将代码仓库设为 Public，则不需要这个步骤；数据和权重不会提交。

## 1. 创建 Kaggle Notebook

开启 GPU T4 x2，Notebook Internet 首次打开，Add Input 中加入 `ceenen/acdc-original-dataset`。

## 2. 执行 Notebook

在 GitHub 上打开 `notebooks/kaggle_run.ipynb`，把 Notebook 导入 Kaggle（新建 Notebook → File → Import Notebook，界面入口可能有所不同）。依次运行全部预期执行的单元格：

- GPU 状态；
- 安全 clone 新仓库、PULSE 上游代码、下载或复用 fold_0 checkpoint；
- 将嵌套的 NIfTI 目录映射为不修改影像数据的软链接目录；
- 一名验证患者的 smoke eval；
- 20 名验证患者完整评价，输出 `per_volume.csv`、`per_patient.csv`、`summary.json`；
- 分别运行官方四路 TTA 和快速无 TTA 测速；
- 如需训练，手动取消注释一个 epoch 的训练 smoke test；
- 输出文件压缩保存。

## 3. 重要边界

- patient001 属于 fold_0 训练集，其 Dice=0.9181 仅是功能测试，正式验证**不可纳入**。
- 首次模型冷启动 38.84s 不代表稳态推理；你实测 10 切片单折 4-TTA 稳态为 1.97s，其他患者可能不同。
- `--tta` 与 `--mode no-tta` 是不同设置，不混合填同一张主表。
- 40 个体积的 Dice、HD95 和 ASSD 均在 3D 空间计算，和论文原始 2D slice-wise mean Dice 及五折 ensemble 数字**不完全同口径**。
- 不允许使用官方测试集调结构；先固定 fold_0 验证集，完成论文比较时另建独立实验划分。

## 4. 若只想快速运行命令

在 Kaggle Notebook 当前工作目录是 `/kaggle/working/pulse-cardiac-research` 时：

```bash
bash scripts/setup_kaggle.sh
python -m pulse_repro.prepare_acdc --input-root /kaggle/input --output-root /kaggle/working/ACDC_pulse/database
python -m pulse_repro.evaluate --acdc-root /kaggle/working/ACDC_pulse/database --pulse-root /kaggle/working/PULSE --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth --fold 0 --tta --out-dir /kaggle/working/pulse_results/fold0
```

**注意**：这些是 shell 命令；放入 Kaggle Notebook 单元格时需在前面加 `!`，或使用 Notebook 提供的 `%%bash`。

## 5. 当前静态验证状态

- Python 代码、Notebook JSON 可解析；
- 表面距离、Dice 的基础单元测试已通过；
- 本次离线环境缺少 `nibabel`，因此 NIfTI 文件夹适配的单元测试处于 skip，尚不能声称整套流程经过端到端运行。Kaggle 端成功运行 smoke eval 后才通过集成验收。
