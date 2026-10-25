# V3 新方案：冻结 DCR-MI 既有 ACDC 权重，重新训练分割（UVI-aligned 90/10）

状态：**协议决定已确认，DCR checkpoint/export interface 仍需 G0 核查；尚未启动训练**  
日期：2026-10-25  
分支：`feature/cine-dcr-motion-v3-spec-20261025`  
**此文是 V3 当前优先执行的协议，替代此前规格中“在 PULSE 80/20 Fold0 上必须重训 DCR-MI teacher”的默认路径；此前 80/20 对照结果保留作为历史研究证据。**

## 1. 用户明确决定
- DCR-MI **不重新训练**，直接读取第一篇已有的 ACDC best checkpoint；全程冻结，记录 SHA。
- 心脏分割网络可从头重新训练；严禁把旧 V1 E95、V1.5 E84、V2 E91、PULSE 在 80/20 Fold0 训练的 checkpoint 直接热启动进新的 90/10 实验。理由：那些分割 checkpoint 的训练集包含新的 90/10 验证患者 `patient091,092,094,096,097,098,100`（7 例），属于已知直接监督交集。
- 新的 V3 主协议按 UVI-Net ACDC 目录默认 split=90 对齐：`patient001..090` 训练；`patient091..100` 验证。已确认 AutoDL 目录有 `MANDATORY_CITATION.md` 排在 patient001 之前，且 UVI 默认 `[1:91]` / `[91:]` 为 90/10。
- 若有模型在 90/10 的末 10 人上**用于选取 DCR-MI checkpoint（以 NCC 验证）**，这十人仍是模型选择验证集，不是完全 untouched independent test。研究报告需披露；最终泛化结论需要另一个独立有标签 cohort/外部数据，不能把该 10 人称为 independent test。

## 2. 患者来源与污染边界

| 模块 | 训练时可用 | 验证时可用 | 限制 |
| --- | --- | --- | --- |
| DCR-MI 原权重 | UVI 原 90 人 ED/ES，已训练；不用重训 | 91–100 可推理 | 模型选择是否使用这些人需要记录 |
| DCR teacher 伪标签 cache | **仅 001–090** | 严禁混入分割训练 | 教师 frozen + provenance |
| V3 segmentation train | 001–090 的 ED/ES 真值和真实中间 Cine；训练内 generated image/flow/pseudo | 不训练 91–100 | 可全新训练或从无 ACDC 监督的通用 backbone 预训练初始化 |
| V3 segmentation val | 禁止 | 091–100 的 ED/ES 真值 | 全 20 个 3D ED/ES volumes，patient-first mean |
| 同协议 V1/V2/PULSE 基线 | **必须重新训练 90/10** | 091–100 | 模型选择、训练预算与标注量一致；旧 80/20 result 只能历史呈现 |
| 第三方 benchmark test | 禁止在 test 患者上训练/调参 | 仅报告允许的指标 | 没有可用 GT 时不得声称 Dice |

必要不变量：`DCR_training_patients ∩ segmentation_validation_patients = ∅`（**weight-optimization only**）；`V3_segmentation_training_patients ∩ segmentation_validation_patients=∅`。若 DCR 权重也曾在 91–100 继续训练，该方案失效，必须另查来源。

## 3. 真实输入与运动接口的硬门槛

当前只通过 UVI 官方代码证明：ACDC 是 128×128×32 centered crop/pad；`SpatialTransformer` 使用基于 native voxel-index 的 grid + displacement、`align_corners=True`、grid last-dimension xyz 反序传给 PyTorch 3D `grid_sample`。

尚未独立核实 DCR-MI 的训练 checkpoint 实际路径、`flow` 返回语义和有无真实 target grid 到 ED/ES 的运动场，以及 teacher 对 ED→ES 的插值状态 `tau` 是否与 `Info.cfg` 中间真实 Cine time frame 一一对应。

**只有插值图像、没有可证明方向的 warp**：先只做 image-only 增强可行性对照，不允许用不可靠的流场传播真实 ED/ES 标签。

运动场 label propagation 以 source labels NN-nearest warp 和 valid mask、物理空间/裁剪 padding 逆变换约定为硬条件；禁止直接把 forward flow 当 pull-back、禁止不同 resolution 的 grid 混用；所有 warp 自测 identity / translation / OOB。

## 4. V3 任务、训练和实验矩阵

- 主目标：利用真实中间 Cine 帧与 DCR 对应证据的**可靠伪标签/跨时相一致性分割**。只使用训练患者，teacher 冻结；不加入诊断和规则报告头。
- 先用 1–3 人 / 若干 target frames 做 teacher image/flow pilot；验证 source、target、DCR intermediate frame 与 geometry 对齐。
- 训练 `V2-90/10 baseline`（无 DCR）与 `V3-90/10`（有 DCR）从同一基础预训练初始化且同训练 budget。固定 data augmentation、TTA 和 checkpoint selection。
- 必要对照：仅 image augmentation；单端传播；双端无可靠性；双端可靠性；真实 temporal input vs 训练期 repeat-target input。避免为 V3 独享更长训练/更多 ground truth。
- ACDC 正式分割验证：10 patients × ED+ES=20 volumes；RV/Myo/LV 3D Dice patient-first mean，逐病例分数、配对改进例数、Bootstrap CI。HD95/ASSD 需要按 geometry 单独核验。
- 若 90/10 数据偏少，在不碰原 DCR 权重前提下，可独立设置其他**不与 DCR teacher 训练患者重叠**的外部 cohort；不得把已见过的病例包装成 independent test。

## 5. 不允许的操作
1. **不重训/不微调 DCR-MI**，保持权重只读；
2. 不能在 V3 新 90/10 验证中复用旧 V1/V2/PULSE Fold0 的 segmentation checkpoint（已包含 7 名新验证患者的真值训练）；
3. 不修改/覆盖现有 80/20 的 `summary.json`、`best_model.pth`、`per_volume.csv`；
4. 禁止为了留在旧 20 人分割验证却直接使用旧 DCR teacher，仍声称 fully patient-independent training；
5. 禁止根据验证集 label 调 DCR reliability 阈值、传播方向、伪标签规则。

## 6. G0 接下来需要的 DCR-MI 实际文件
- Kaggle / AutoDL 中**已训练的 DCR-MI ACDC checkpoint path** 及其 epoch、SHA；
- 它当时使用的训练患者 list，是否只有 UVI 官方前 90；
- DCR-MI forward/model output 定义：中间生成影像、变形场、方向、像素/体素单位、valid mask；
- 输入输出空间是 centered 128×128×32 还是有其后额外变换；
- 若 DCR checkpoint 只放 Kaggle，需把 checkpoint 和必须的代码部署到 AutoDL；不需要新训练，但不能凭论文方法描述凭空声称具体参数已经导出。

## 7. 结论

**可行：冻结 DCR-MI + 重新训练 90/10 分割。** 现有代码 `train_cine_v2.py` 和 `evaluate_cine_v2.py` 目前硬编码 Fold0 的 stratified 80/20，故需要建立**独立新的 V3/split adapter 与 evaluation runner**，不能只往旧 CLI 传 `--fold 0` 就实现 90/10。

旧 80/20 E91 权重保留为诊断基线，不能在新协议作为无泄漏初始化。

## 8. 2026-10-25 冻结投稿证据：正式 DCR-MI 是两份权重

通过用户私有仓库 `huhumic/4d-lung-motion` 的 `dcrmi-submission-frozen-20260830` 分支：`reports/submission_experiments/acdc_heldout101_150_frozen_v2/CHECKPOINTS.json` 和 `run_manifest.json`，直接确认：

| role | canonical path (project/checkpoints 下) | SHA256 |
| --- | --- | --- |
| `dcr_motion` | `frbme_acdc_4090_formal/best_psnr.pth` | `ea1889917dd4bcf3180dbaaded02345686e63b24c9863349649200910de5bdda` |
| `dcr_refiner` | `mgrr_protocol_confirmation100/acdc/best/best_psnr.pth` | `0152096dd6452f94ef75b43a2c9e4bcdc7c7bfdee5276a9d04d8953cb7a5da3c` |

投稿冻结记录证明正式 DCR-MI 是 FR-BME 前端 + frozen MGRR E77 后端的**两件套**。 `dcrmi_front_ablation250/acdc/...` 不属于正式完整组合；`relation_reliability_ablation_...` 不得替代已冻结的正式身份。上游`research_modules/frbme/refinement.py` 明确 MGRR 只修正合成强度，不预测/修改位移场：V3 标签传播首先应核验 FR-BME 的目标状态运动场，不能从 MGRR 图像残差虚构 flow。

源服务器用户列出目录 `frbme_acdc_4090_formal`，但尚未独立运行磁盘 `sha256sum` 验证。 `mgrr_protocol_confirmation100` 未被用户先前关键词为 ACDC/DCR 的顶层目录枚举覆盖，**不代表不存在**。必须两个都在源磁盘存在、SHA 匹配；传目标服务器时以角色分别命名/分目录，避免 `best_psnr.pth` 同名冲突。

**状态仍为需实机检查**：源文件 SHA、目标传输完成与双源 checkpoint load/flow contract。不能声称已经完成拷贝。
