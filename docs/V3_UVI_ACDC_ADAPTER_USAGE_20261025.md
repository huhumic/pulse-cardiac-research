# V3：UVI-Net ACDC 数据适配层（CPU-only）

2026-10-25。状态：源码与单元测试已提交；用户服务器上的测试和真实患者核验尚未执行；训练禁止启动。

## 依据与版本边界

官方代码：https://github.com/jungeun122333/UVI-Net/blob/main/utils/datasets.py

官方说明：https://github.com/jungeun122333/UVI-Net/blob/main/README.md

UVI 官方 README 中的“旧版／新版预处理数据”主要针对 4D-Lung 数据发布，不构成两个不同 ACDC 预处理协议的证据。历史 DCR 训练使用的旧版处理结果仍应依赖实际 loader、实际保存数据和 checkpoint 来源进行核对。

官方 ACDCHeartDataset 直接读取 NIfTI 数据，中心裁剪前两空间轴到 128×128，第三空间轴居中裁剪或补零到 32，再对 ED、ES 和每个视频时相分别进行 0–1 Min-Max 归一化。不会使用 affine 自动重定向或进行 spacing 重采样。网络输入张量的三个空间轴对应 NIfTI 数组存储轴，不能误称患者物理 zyx。

## 独立实现

- pulse_repro/v3_uvi_acdc_adapter.py：空间映射/逆映射、来源有效 mask、ED/ES/Cine 适配、时相边界、标签验证、90/10 划分保护。
- tests/test_v3_uvi_acdc_adapter.py：原始空间裁剪补零、已预处理模式的数值保持、整数标签、时间弧和分组检查。

两种显式模式：
1. raw_uvi：从原始 NIfTI 张量空间按官方方法裁剪/填充，图像归一化，GT 仅空间变换。
2. already_uvi：已处理好的 128×128×32 图像/标签，图像在 [0,1]；不再裁剪、不再归一化、不交换轴。GT 必须提前处于同一网格。拒绝不兼容数据，绝不自动猜测。

patient001 原始 216×256×10 → 裁剪轴0 [44:172]、轴1 [64:192]、轴2 两侧各补 11。补零体素及原始图像被裁掉的区域都不能当成有效解剖标注。

## AutoDL（无 GPU）

独立 clone，不覆盖已有 80/20 工程：

    cd /root/autodl-tmp
    git clone --depth 1 --single-branch --branch feature/cine-dcr-motion-v3-spec-20261025 https://github.com/huhumic/pulse-cardiac-research.git pulse-v3-adapter
    cd /root/autodl-tmp/pulse-v3-adapter
    python -m pytest -q tests/test_v3_uvi_acdc_adapter.py
    python -m pulse_repro.v3_uvi_acdc_adapter --raw-patient /root/autodl-tmp/datasets/ACDC/database/training/patient001

后一条只核对原始 ACDC 经官方流程后的单病例数据，不能证明历史 checkpoint 使用的是这个文件来源。若实际存在已保存的“旧版已处理 ACDC”，须进一步读取那组数组，并核对其尺寸、强度范围及与 raw_uvi 数值/空间对应关系。

## 未通过的 Gate

- 历史 DCR checkpoint 的真实 ACDC 保存数组与当前读取路径的身份仍待核实。
- 真实训练患者名录需与冻结教师 manifest 双向验证。
- 真实中间帧的 phase tau 定义需与历史模型训练一致，不能用一个候选计算公式当作已验证协议。
- GPU 恢复后仍需冻结教师的真实前向/双端点标签传播小样本试验。
- V3 segmentation 不使用旧 80/20 checkpoint，不在旧 Fold0 上宣称 DCR teacher 患者独立，也不在核验结束前启动训练。
