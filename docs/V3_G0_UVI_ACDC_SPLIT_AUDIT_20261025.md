# V3 G0: UVI-Net ACDC 数据划分与 PULSE Fold 0 交集核查（2026-10-25）

**状态：G0 发现显著患者重叠，旧 DCR-MI ACDC 教师不得用于严格独立的 PULSE Fold 0 V3 主实验。**

## 1. 已核实：UVI-Net 官方代码

上游： https://github.com/jungeun122333/UVI-Net （CVPR 2024）。

- `train.py` 对 cardiac 使用 `data_dir=dataset/ACDC/database/training`，`split=90 if args.split is None else args.split`，同一个 `ACDCHeartDataset` 分别以 `phase="train"` 和 `phase="test"` 加载，影像尺寸为 `(128,128,32)`，默认 `max_epoch=200`、`batch_size=1`。
- `utils/datasets.py` **准确原始行为**：`sorted(os.listdir(self.path))[1:split+1]` 是 train；`sorted(os.listdir(self.path))[split+1:]` 是 test。它是**原始目录项切片**，不是筛选患者 ID 后的 90/10。
- 若 training 文件夹下恰有一个排在 patient001 之前的额外条目（例如根级文档），产生 train `patient001..090` 和 test `patient091..100`。
- 若仅有 `patient001..100` 共 100 个患者目录，则 train 为 **`patient002..091`**，test 为 **`patient092..100`（9 例）**，且 `patient001` 被跳过。务必核查实际训练时的 `os.listdir` 目录快照，不能仅凭宣称的 90/10 推断精确 patient IDs。
- 官方训练循环虽然从 dataset 读取完整 `video`，只把 batch[0],batch[1] ED/ES 两端点用于训练损失；完整 4D 视频中间帧不作为 training supervision。测试时使用中间帧量化 PSNR/NCC/SSIM/NMSE/LPIPS。
- **ACDC/database/testing 的另外 50 个病例不是官方默认 `train.py --dataset cardiac` 的验证来源。**

来源：
- https://github.com/jungeun122333/UVI-Net/blob/main/train.py
- https://github.com/jungeun122333/UVI-Net/blob/main/utils/datasets.py
- https://github.com/jungeun122333/UVI-Net/blob/main/evaluation.py

## 2. 与 PULSE Fold0 验证集交集

当前 PULSE 已完成 Fold0 的 20 名患者（从 40 体积日志及 `per_volume.csv` 可核对）：
`patient014,patient017,patient018,patient020,patient023,patient028,patient035,patient037,patient042,patient046,patient047,patient050,patient065,patient071,patient072,patient080,patient089,patient093,patient095,patient099`。

在 UVI 的上述两种典型目录布局中，训练集合与 PULSE 验证集重叠 **17/20**：
`014,017,018,020,023,028,035,037,042,046,047,050,065,071,072,080,089`（带 patient 前缀）。
不重叠 **3/20**：`patient093,patient095,patient099`。

这份交集是**以 DCR-MI 使用 UVI 官方默认 split/目录顺序为前提**。用户明确称自己的训练与 UVI 一致，结合此前 DCR-MI 文稿中所述 ACDC 90/10，可将此列为强风险。尚未读取 DCR-MI 实际训练 manifest 或 checkpoint provenance，因此不得把“DCR-MI 17 名交集”描述成已由真实训练日志独立证实的事实。

**决策**：即使 DCR-MI 在训练中只看到这 17 人的无标签 ED/ES，也不符合 V3 患者独立 teacher 实验预期。不得将该旧 teacher 的伪标签/流场混入训练并据此声称 Fold0 严格独立。

## 3. 已核实：UVI warp 的几何语义，但不能直接当 DCR-MI 定义

上游 `utils/utils.py:SpatialTransformer` 与 `models/VoxelMorph/model.py` 使用 `new_locs = grid + flow`、normalized grid [-1,1]、`align_corners=True`、3D 分量末尾 `[..., [2,1,0]]` 后传入 `grid_sample`。因此 UVI 的 flow 在其 native `128×128×32` 张量索引网格中以**voxel/grid-index displacement** 使用，具备 pull-back sampling 形式，并**非直接毫米 DVF**；轴顺序必须根据 native image layout 做显式转换。实现需要确认 `flow_0_1` 对应从 I0 重采样到 I1 的输出目标网格，不能仅凭名字认为是 push-forward ED->ES。

- `utils/datasets.py` 以 center crop 和 z 方向 pad/crop 将体积变为 128×128×32，每帧单独 min-max normalize；此空间变换与 PULSE 本地 full-volume 原始 `H×W×S` 网格不相同。
- 即便 DCR-MI 沿用 UVI 的几何接口，流场传播标签也需要保留裁剪/pad、分辨率、voxel spacing、affine 与分量顺序，做 identity / known-translation / out-of-bound 测试后才能用于 full-volume 分割。
- DCR-MI 具体 flow 的变量名、最终 pull-back 方向、M-Mod 修正及中间目标 `t` 的流输出**尚未从 DCR-MI 实现核实**。

来源：
- https://github.com/jungeun122333/UVI-Net/blob/main/utils/utils.py
- https://github.com/jungeun122333/UVI-Net/blob/main/models/VoxelMorph/model.py

## 4. G0 结论与下一步

| 子项 | 结果 |
| --- | --- |
| UVI 官方 split 及代码 | PASS（90 的索引切片，附目录 sentinel 风险） |
| PULSE Fold0 验证患者名单 | PASS（依现有正式 40-volume 结果） |
| UVI 默认 train vs PULSE Fold0 overlap | **17/20**（两种典型目录布局一致） |
| DCR-MI 真实 checkpoint provenance | **UNVERIFIED** |
| DCR-MI image/flow 输出与时相接口 | **UNVERIFIED** |
| DCR-MI 到 PULSE 的物理空间映射 | **UNVERIFIED** |
| 使用旧 DCR-MI ACDC teacher 做严格 V3 Fold0 | **BLOCKED** |

严格方案：为 V3 Fold0 **仅用 PULSE 的 80 名训练患者重新训练 DCR-MI teacher**，从该版本在这 80 名训练患者内产生中间时相伪标签；Fold0 验证患者一律只用于评价及 checkpoint selection。重训后检查 fold-specific teacher 的训练来源及 label warping geometry。

备选 exploratory 方案：旧 DCR teacher 可用于纯内部可行性/teacher-only geometry diagnostics，但不得与严格 V3 结果混为同一证据；更不能在旧 teacher 之上仅对 80 人微调后称验证无泄漏。

**G0 保持 BLOCKED，直到用户真实训练日志、代码和导出接口完成来源审计并落实 fold-specific teacher。**
