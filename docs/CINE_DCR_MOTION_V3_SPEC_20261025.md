# V3 设计规格（设计冻结候选）：双向运动可靠性约束的 Cine MRI 时序分割

> **2026-10-25 协议更新：用户选择复用现有 DCR-MI ACDC checkpoint、不重训练 DCR；分割端重新训练。当前优先协议为 [V3_FROZEN_DCR_TEACHER_9010_PROTOCOL_20261025.md](./V3_FROZEN_DCR_TEACHER_9010_PROTOCOL_20261025.md)。下文 80/20 Fold0 + fold-specific teacher 方案保留为历史设计备选，不能再当作当前默认执行协议。**

状态：**DESIGN ONLY / NOT APPROVED FOR TRAINING**  
日期：2026-10-25  
基础分支：`feature/cine-fullvolume-validation-20261025`  
本分支：`feature/cine-dcr-motion-v3-spec-20261025`

## 0. 目标与边界

核心研究问题：在 ACDC 只有 ED/ES 人工分割标注、其余真实 Cine 时相未标注的前提下，能否利用**患者独立的 DCR-MI 时相对应/双向运动证据**，提供可靠中间状态结构监督，并改善 ED/ES 完整体积分割及跨时相一致性？

V3 不是“给 PULSE 再加一个注意力/卷积块”，也不把普通的双向配准+伪标签本身宣称为独创。差异必须由**可校验的双端点传播可靠性、真实中间时相影像一致性，以及可靠性控制的监督机制**和实验支撑。

限制：
- V1 E95、V1.5 E84、V2 E91 所有权重与既有结果只读。不要修改或覆盖原训练/评估源码、checkpoint、manifest。
- 此轮只设计、不训练；**在第 1 节 G0 核查通过前，禁止把旧 DCR-MI ACDC checkpoint 接入正式 Fold 0**。
- 保留 PULSE 2.5D 相邻**空间切片**输入，另行引入真实 4D Cine **时间轴**；两种“邻近”绝不能混用。
- ED/ES 为人工真值（RV=1、Myo=2、LV=3），中间时相**无人工分割 GT**；所有中间伪标签与推断误差必须明确标识为 pseudo，不报告为人工标注 Dice。
- 以分割/动态结构一致性作为第二工作的研究主线；心肌病诊断及规则式报告是后续下游，不放入 V3 最小实验闭环。

## 1. G0：资料和泄漏预检（硬阻断）

### 1.1 ACDC Fold 0
复用当前 PULSE `stratified_kfold_pids(records,5,fold=0,seed=42)`，100 名有标签训练库病例，80 train、20 validation；每人 2 个 ED/ES 真值体积。真实 4D Cine 由 `adapter_manifest.json` 指向原始数据，逐病例用 `verify_patient_cine` 校验 `Info.cfg`、shape、NIfTI geometry/体素对应和时间帧索引。

每个衍生文件和 cache manifest 都必须带 `patient_id`、`fold`、`source_phase`、`target_phase_index_1based`、`frame_count`、空间 spacing/affine、input SHA、teacher checkpoint SHA、生成方法、flow convention、数据 split、code commit。

### 1.2 DCR-MI 模型的训练来源（优先级最高）
必须取得用于 ACDC 的 DCR-MI：
1. 确切代码 revision / checkpoint / config / 数据 manifest；
2. 训练、验证患者 ID 列表（包括数据预训练、继续训练和 teacher 蒸馏来源）；
3. 中间图像输出定义与运动输出是否实际导出；
4. 位移场几何约定：`target-grid pull-back` 还是 forward push，位移单位 mm/voxel、轴序、spacing、`align_corners`、有效采样定义；
5. ED↔ES 之间目标相位 t 的对应规则、是否覆盖两段生理周期、是否可匹配真实 Cine frame。

**阻断规则**：如果 DCR-MI ACDC checkpoint 曾训练/适配于当前 Fold 0 验证患者（即便无分割标签），不能在严格患者独立的 V3 验证中用它生成教师信息；应重训练仅用 80 train 患者的 fold-specific teacher。现有可能重叠的权重仅供开发诊断，结果标记 exploratory/transductive，不能与 Fold 0 正式结果并列主张。

如果 DCR-MI 没有可验证的像素级可逆几何映射，只能做 **image-only pilot**，不能假装它导出了可用的标签传播流场。需要补独立的配准器时应清楚区分 DCR-MI 与配准来源。

### 1.3 时间坐标
ACDC `Info.cfg` 为 1-based frame，4D NIfTI 时间轴为 0-based；周期索引可回绕，但 ED→ES 与 ES→下一次 ED 是两个不同时间弧段，不能用单纯的最短距离或线性帧差取代生理路径。必须为每个 `target_t` 明确来源弧段和插值比例 `tau`；若 DCR-MI teacher 不支持某段则不生成相应伪标签。

## 2. 两层设计：先可验证的标签证据，再讨论网络结构

### 2.1 固定监督主干
沿用当前可运行的 V2 segmentation 主干/数据管线，保持 ED/ES 监督损失和一致的 TTA、后处理、fold。V3 的第一个版本不增加诊断头，不重写 DINOv2，不改 PULSE 原始代码。

避免直接从 V2 E91 断言“公平提升”：从该权重继续训练可作为 low-cost feasibility pilot；正式论文须报告相同初始化、训练预算及模型选择协议下的对照，尤其包含**额外训练步数/未标注数据使用量相匹配**的 control。

### 2.2 教师生成候选中间标签（必须先通过 G0）
对 training 患者真实中间 Cine 帧 `I_t`，利用标签来源 ED 和 ES 的已验证运动对应，分别生成候选 `Y_{ED->t}` 与 `Y_{ES->t}`。

若运动为 target-grid pull-back `u_{t->ED}`，则：
`Y^ED_t(x) = NN(Y_ED, x + u_{t->ED}(x))`；
ES 同理；标签采用 nearest-neighbor，不能用图像双线性插值直接软化类别。若只有 forward `u_{ED->t}`，必须经过经测试的反演或变换合成；严禁直接把 forward flow 当作 pull-back 用。

每张伪标签附带有效采样 mask、形变 Jacobian/折叠检查、映射方向、图像几何对应、两端传播的一致性记录。没有可靠双端点传播时跳过样本/区域，不回退到自信的随机伪标签。

### 2.3 可靠性图：核心待验证机制
从**几何有效性**、**双端点标签一致性**、**双向流/循环一致性**、**局部图像配准误差**构造训练监督权重 `r_t(x)`。

初版以可解释的 hard-invalid + soft confidence 规则为主；权重的阈值/尺度仅在 train 患者上校准（可用 ED↔ES 交叉传播与真实 ED/ES GT 校验传播误差），不能查看验证 GT 调阈值。

DCR-MI 原有残差细化门控 **不能直接声称是经过校准的解剖标签正确率**；可作为候选证据，必须证明它与传播误差相关。对小结构、心肌边界和 RV 区域需报告保留比例，防止“只监督容易区域”的表面提升。

### 2.4 学生训练目标（先保持简单）
`L_total = L_EDES_supervised + lambda_u * L_reliable_pseudo + lambda_c * L_temporal_consistency`。

- `L_EDES_supervised` 与 V2 原始训练定义保持一致；
- `L_reliable_pseudo` 是 `r_t` 加权的 cross-entropy / soft Dice，teacher 伪标签严格 stop-gradient，空可靠区域时返回有限值 0 且保持正确梯度；
- `L_temporal_consistency` 优先使用已校验 warp 后的预测概率一致性，不强制把两个不重合时相逐像素直接比较；
- 初期 `lambda_c=0`，先判断标签传播单独是否有收益；再逐步增加一致性（同预算 ablation）；`lambda_u` 采用预先约定 warm-up，不对验证集反复搜索。

**重要**：为建立真实时间贡献，需有 `repeat-target` 的**训练时匹配对照**。已有仅在推理阶段替换邻帧的实验不能替代这一比较。

## 3. 最小闭环和阶段门槛

### G0 — 只读数据/教师契约审计（无 GPU 训练）
输出 `reports/v3_g0_audit.json`：Fold0 patient list SHA、ACDC/Cine 检验、DCR-MI checkpoint provenance、交集数量、真实 image/flow 输出、空间方向及单位、阻断项。

G0 未通过：禁止正式 V3 训练。可以独立做图像插值输出质量分析，但不能算 Fold 0 segmentation 主实验。

### G1 — 双端点传播小样本质量诊断（无新分割训练）
仅 training 患者；选择若干患者，多组 t。检查变形可视化、mask 所在切片、有效区域、RV/Myo/LV 保留比例，ED↔ES 传播 Dice/HD95（只作为训练数据教师标定，不当作中间 t GT）；对生成图像计算 PSNR/SSIM 等仅在与真实 frame 对齐且有正确对应时使用。

G1 失败则停在教师契约/运动估计阶段，不启动 120 epoch。

### G2 — 低成本 V3 pilot
使用固定 fold 和小范围训练患者，独立的全新输出目录与训练日志。可冻结重骨干，仅更新有限参数，先跑功能 Smoke Test，再做短训练；检查伪监督非零且不会大幅破坏 ED/ES Dice、可靠区域覆盖合理、梯度有限、缓存 provenance 完整。单次 pilot 不宣称正式性能。

### G3 — 四组必要对照（训练预算/初始化尽可能匹配）
| 组别 | DCR 运动传播 | 可靠性筛选 | 额外时相数据 | 用途 |
| --- | --- | --- | --- | --- |
| B0 `V2-budget-match` | 无 | 无 | 按同预算训练 | 排除训练额外步数 |
| B1 `single-end-propagation` | 单端点 | 有效采样基本 mask | 有 | 普通伪标签对照 |
| B2 `bidirectional-unweighted` | 双端点 | 仅无效区域剔除 | 有 | 隔离双端点本身 |
| B3 `reliability-weighted` | 双端点 | 双向一致性/几何/图像可靠性 | 有 | V3 核心机制 |

附加必需对照：主实验的真实时间邻帧训练 vs 重复目标帧训练（共享除时间输入外的配置和预算），及 image-only augmentation vs motion+mask teacher（如果能生成图像）。不必第一阶段同时跑完整五折。

### G4 — 严格验证和论文评估
首先使用与 V1/V2 相同的 Fold 0：20 patients/40 ED+ES full volumes，4-flip TTA、FP32、PULSE 同后处理，逐病例三类 Dice；报告 paired `V3−V2`、`V3−V1` 的患者级 Bootstrap 95% CI、improved patients、HD95/ASSD（待物理空间度量单独审计），以及伪标签可靠区域占比/跨时相预测一致性。

阶段性充分条件：相同训练预算下 V3 相对 B0 在 full-volume patient-mean Dice 有一致的正向增益，且相对无可靠性 B2 的收益能定位；真正的动态信息收益必须由真实时间 vs repeat-target 的 **matched training** 对照支持。若效果不成立，就不把运动信息作为已验证主贡献。

最终论文需要患者级多 Fold/外部测试、明确 validation checkpoint 选择规则。Fold 0 曾被用于选择 checkpoint，不是 untouched test set，不能将 Fold0 bootstrap CI 夸大为独立外部泛化证据。外部数据如没有匹配时相真值，只做可评价任务，不捏造中间帧真值指标。

## 4. 建议软件接口（批准实现前再定稿）

- `pulse_repro/v3_provenance.py`：患者交集检查、来源 SHA 与坐标契约；
- `pulse_repro/v3_teacher_adapter.py`：DCR-MI image/flow output 适配器，所有输入输出显式 geometry；
- `pulse_repro/v3_pseudolabel.py`：ED/ES 双端点标签传播与可靠性图；
- `pulse_repro/v3_data.py`：真实中间 Cine 训练样本与统一 transform；
- `pulse_repro/train_cine_v3.py`：在 V2 之上受控扩展训练、只写新 run directory；
- `pulse_repro/evaluate_cine_v3.py`：尽量复用 V2 统一 full-volume evaluator 与患者级聚合；
- `tests/test_v3_provenance.py`, `tests/test_v3_geometry.py`, `tests/test_v3_pseudolabel.py`, `tests/test_v3_losses.py`。

必要的单元测试：patient-overlap 必失败；1-based/0-based/frame cycle 正确；identity warp 必须原样返回 ED/ES 标签；常量平移/轴序/spacing 与 inverse warp 正确；out-of-bounds 无效；nearest-label 不产生非整数类；双端标签矛盾降权；完全无可信像素时 loss 有限且不破坏梯度；同步 augmentation 不破坏 flow/label 几何；不能在 train cache 混入 validation patient。

## 5. 现有已核实的基线及解释约束

V1 E95 `0.889185`，V1.5 E84 `0.885517`，V2 E91 `0.890406`（Fold0，20 patients, 40 volumes）。

V2 correct vs swap: `+0.000086`（95% CI 跨 0）；correct vs repeat-target `−0.00000046`（CI 跨 0）；correct vs temporal bypass `+0.016465`（20/20 patients 增益）；correct vs bias-only `+0.016256`。这些结果支持 V2 残差输入相关更新的作用，但不支持现有 V2 充分利用真实邻帧的主张；直接关闭模型组件会改变已训练网络分布，不能作为动态运动贡献的全部量化。

已有相关工作：[PULSE (IEEE JBHI, 2026)](https://doi.org/10.1109/JBHI.2026.3725919)；[Qin et al., Semi-supervised cine cardiac MRI segmentation via joint registration and temporal attention perceiver (Medical Physics, 2025)](https://doi.org/10.1002/mp.70094)。V3 必须与后者进行机制定位比较，不能把“配准+伪标签”当作新颖性来源。

## 6. 开发顺序与不得越界的事项

1. 先提交/执行 G0 审计，不训练。
2. G0 PASS 才写 teacher adapter + geometry tests（独立接口，不改 PULSE 源文件）。
3. G1 PASS 才做可信伪标签 cache。
4. G2 先短训练，再 G3 matched controls。
5. 冻结最终方法后做 G4 并考虑论文实验矩阵；诊断和结构化规则报告推迟到主任务成立之后。

**此文件是设计规格，不表示 DCR-MI 的真实代码、checkpoint、患者来源或 flow 约定已核查通过。**
