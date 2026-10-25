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

## 8. 2026-10-25 纠错：采用论文固定 Epoch，不采用旧 GitHub best checkpoint

**已撤回前版第 8 节将旧 GitHub `CHECKPOINTS.json` 中 `best_psnr.pth` 定义为“正式 DCR-MI 权重”的断言，也撤回以旧哈希直接传输的命令。不要执行旧命令。**

用户明确指出 GitHub `dcrmi-submission-frozen-20260830` 分支已明显滞后。重新核对 2026-10-20 的《DCRMI_EN_Final_v12_Submission_Clean(1).docx》 Section 4.2：TSC-BME motion 前端训练 50 epochs 后冻结，refinement 随后训练 100 epochs；用户确认论文使用固定 E50 前端与 E100 后端，而非对 validation 选择的 `best_psnr`。

因此 V3 要传的是 **同一实验配置/种子的固定 E50 + 固定 E100** 两份权重，是否能在源服务器上组合加载仍须实机验证。旧仓库中 FR-BME best E229 / MGRR best E77 的 checkpoints 和其 SHA 仅能描述旧历史实验，**不得冒充当前论文最终权重或 V3 teacher**。

在源服务器上要先只读审计：
1. E50 运动前端的目录、checkpoint 内部 `epoch`/`step`、前端结构变体、seed、数据集 ACDC 90/10；
2. E100 后端的目录、checkpoint 内部 `epoch`、seed、记录的前端来源路径/hash；
3. `run_config.json`/`manifest.json`/`progress.json`/训练日志与正式论文 Section 4.2 是否一致；
4. 模型导出流场的语义/坐标空间，以便未来 V3 训练阶段的标签传播。

尤其注意目录 `relation_reliability_ablation_acdc_full_acdc_frE050_seed12345` 等虽然是**潜在候选**，但名称 `full` 与 `frE050` 不能替代内部来源证据；不经过审计不能直接判为正式主实验。

**G0 checkpoint identity: PENDING。训练数据划分 90/10 已核验；权重传输：未执行。**
## 9. 2026-10-25 第二台服务器实测迁移状态

用户在源服务器完成 ACDC 固定 E50 + E100 candidate 两文件打包并用 SCP 上传到目标服务器；源端压缩包 SHA256 与目标端验证 `sha256sum -c` 一致，目标端输出 `TRANSFER VERIFIED`。目标机 CPU-only 解压验收输出：

- `checkpoints/frbme_acdc_4090_formal/epoch_050.pth`: 10,840,050 bytes, present。
- `checkpoints/relation_reliability_ablation_acdc_full_acdc_frE050_seed12345/epoch_100.pth`: 451,786 bytes, SHA256 `6e198ceff0e587d9d7c5e7aa08d6c551caa3fdbe1ad4de3c3c9895fc4e8dc5e5` PASS。
- `reports/relation_reliability_ablation_acdc_full_acdc_frE050_seed12345/manifest.json`: 3,253 bytes, `fr_checkpoint` 指向上述 FR E50；`structural_variant: full`, `dataset: acdc`, E100 training, `seed: 0`。
- 文件落地目录：`/root/autodl-tmp/pulse_results/dcrmi_acdc_import`，保留打包的原相对路径。

**特别注意**：该源端 manifest 仍包含 `checkpoint_selection: best monitored PSNR among E1..E100`，意味着该训练过程原始记录存在 validation-based best 模型选择；在 V3 我们明示使用 **固定 E100 文件**，不能将该 manifest 的 best-PSNR 结果冒充 fixed-E100 的论文指标。论文终稿 E50+E100 规则与具体单份权重产生的主表仍需代码/评估结果双向核实。

G0 传输和依赖清单：PASS；GPU 是否存在与文件完整性无关。FR-E50 独立 SHA256、两权重安全反序列化、完整模型严格 state_dict 加载、实际位移场返回形状/方向/单位、真实 ACDC replay：**尚未执行**。代码仍应从源服务器当前文件复制，因为现有 GitHub 分支已过时。不得声称模型部署成功。

### Current source migration (user confirmed)

User executed source-to-target remote transfer on target: tar of `research_modules`, `tools`, `third_party` matching `*.py` from source `/root/autodl-tmp/project` was saved as `/root/autodl-tmp/pulse_results/dcrmi_acdc_import/current_source_py.tar.gz` and extracted to `/root/autodl-tmp/pulse_results/dcrmi_acdc_import/source/`. Shell output: `SOURCE TRANSFER COMPLETE`; `find ... -name '*.py' | wc -l` returned **817**. This is file-transfer/untar evidence only, NOT import/strict model load/flow replay evidence. Non-Python configs and package resources were not included by the transfer filter. Target machine currently has no GPU, so next actions must be CPU-only source / checkpoint structural audit and should print bounded output.

### CPU strict checkpoint load (user confirmed, 2026-10-25)

On target CPU-only environment, user ran trusted checkpoint `torch.load(..., map_location="cpu", weights_only=False)` and constructed models from the **migrated current source**, not the outdated GitHub snapshot:

- Motion checkpoint keys include `model_kind`, `model`, `model_config`, `cursor`, `seed`, `dataset_contract`.
- Refiner checkpoint keys include `fr_checkpoint`, `fr_cursor`, `fr_model_config`, `fr_frozen`, `refiner_model`, `epoch`.
- Motion: `FRBME(FRBMEConfig(**front['model_config']))`, strict `front['model']` loading **PASS**, 663,262 parameters. Inherited `forward(endpoints, t, return_trace=False)`.
- Refiner: `StateConditionedRelationReliabilityRefiner(**variant_config)`, strict `back['refiner_model']` loading **PASS**, 25,075 parameters; `variant_config = {'use_relation': True, 'use_warp': True, 'state_conditioned': True}`.
- Joint parameter total **688,337 = 0.688337 M**. This must be reconciled against reported manuscript size (possible revision/variant difference; do not invent cause). Does not imply the manuscript's fixed E50/E100 checkpoint identities or performance are reproduced.
- **Next gate:** use transferred source to prove returned flow direction/units, shape and downstream pull-back sampling; then run a real ACDC forward/replay when GPU is available. No training or GPU was used in the strict-load gate.

### Verified frozen FR-BME flow contract (2026-10-25; migrated source)

On the target server the user inspected the **current** `research_modules/rbme/adapters.py` and `research_modules/rbme/rbme.py`:

- `forward(endpoints, t)` expects `endpoints[B,2,D,H,W]` matching `self.config.inshape`; returns `warped_0, warped_1, phi_0t, phi_1t`, where each displacement is `[B,3,D,H,W]`.
- `VoxelPullbackWarp3D.forward(source, flow_zyx)` computes `locations = identity_zyx_grid + flow_zyx`; reorders normalized coordinates `[..., [2,1,0]]` to PyTorch `xyz` and calls `F.grid_sample(..., align_corners=True)` with default zero padding. Displacement channels mean `dz,dy,dx` **in voxel units** and are **target-grid-to-source sampling offsets**, despite the `phi_0t` naming. Do not invert or swap when propagating endpoint labels.
- `upsample_voxel_flow` uses trilinear `align_corners=True` followed by per-axis factor `(new_dim-1)/max(old_dim-1,1)` in **zyx order**; applies correct voxel-unit rescaling for normal dimensions greater than one.
- To propagate ED/ES label volumes already resampled to the teacher's identical input grid, use `VoxelPullbackWarp3D(size, mode='nearest')` separately for `(label_ED,phi_0t)` and `(label_ES,phi_1t)`. Use warped all-ones masks to flag out-of-bounds nearest samples instead of treating padding zeros as actual background. Keep pseudolabels and valid masks distinct. For segmentation CE use ignore_index on invalid pixels. No GPU required for a small synthetic geometry test.
- Future real-case gate: verify ACDC ED/ES frame order and phase normalization, teacher fixed 128x128x32 geometry (read from checkpoint config rather than assume), segmentation preprocessing/orientation/label alignment, and train/held-out patient split; do not report real-case performance or teacher label correctness based on code-level contract alone.
