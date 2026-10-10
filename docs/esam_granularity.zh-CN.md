# ESAM 解码器是否在做「粒度对齐」？

**假设**：ESAM 的解码器（Query Refinement）本质上在做 mask 合并。SAM/FastSAM 会把椅子
切成座面、靠背、腿几块，解码器把这些块合成一个 mask，和 ScanNet 的单个 GT 对齐。

**结论：基本成立。** 在 16 个 ScanNet 验证场景、809 帧上：

- ESAM 每个 mask 平均由 **7.0 个** FastSAM 组构成，**82%** 的 mask 合并了 ≥2 个组。
- 对被 FastSAM 切碎的 GT 实例（53% 的 thing 实例），IoU≥0.5 的召回率从 **0.39 → 0.88**。
- 只允许「整组合并」的 ESAM-snap 已拿到解码器提升的 **77%**（平均最佳 IoU 0.532 → 0.702，
  完整 ESAM 为 0.754）。剩下的 23% 来自逐点（`mask_pred_mode` 最后一层为 `P`）的边界修整。
- ESAM 与「只合并」GT oracle 的差距只有 0.025 IoU/GT。

## 实验设置

- 场景：本机拥有完整原始数据的 16 个 ScanNet 验证场景（`configs/esam_eval.yaml`）。
- 数据：用 ESAM **原版**预处理（`load_scannet_mv_data_fast.process_cur_scan`，FastSAM-x，
  40 帧间隔，每帧 20,000 点）重新生成，写入 `work_dirs/esam_eval/`。scene0690 与 ESAM 仓库中
  已有的数据逐字节一致。唯一改动：导出 `.sens` 时保留位姿无效的帧，让 ESAM 自己的
  NaN 检查跳过它们。原 `local/extract_sens.py` 在导出时就丢掉这些帧，会触发 ESAM 的帧号断言。
- 模型：官方 ESAM-E（`ESAM-E_CA_online_epoch_128.pth`），配置与代码都没有修改。包装
  `OnlineMerge.merge` 记录每帧解码器的输出，即 ESAM 自己做完 top-k、matrix NMS、分数阈值
  之后、跨帧合并之前的 mask。scene0690 的场景级 AP 与之前的复现完全一致
  （39.52 / 50.46 / 63.54）。
- 16 个场景的官方场景级 AP：**34.7 / AP50 53.6 / AP25 72.4**。
- GT：ESAM 训练用的逐帧 GT（`instance_mask/<scene>/<frame>.bin`）。只统计 ≥100 点的实例；
  thing = 去掉 wall/floor/ceiling。ESAM-E CA 不预测 stuff，所以以 thing 为准。
- 对比对象（同一帧、同一批点）：
  - **FastSAM**：互斥分组，即 ESAM 的输入 query，含 KMeans 补组。
  - **ESAM**：解码器逐帧输出。
  - **ESAM-snap**：把每个 ESAM mask 取整为「≥50% 在 mask 内的 FastSAM 组」的并集。
  - **Oracle-adj**：GT 指导、只沿 3D 相邻（<5 cm）组合并。
  - **Oracle**：GT 指导的任意合并（purity≥0.8，不拆分）。

## 结果（thing，帧平均）

| 方法 | mask 数/帧 | 平均最佳 IoU | 召回@0.5 | 召回@0.25 | 精度@0.5 | AP50 | 碎片数/GT |
|---|---|---|---|---|---|---|---|
| FastSAM | 72.7 | 0.532 | 0.515 | 0.876 | 0.047 | 0.232 | 2.22 |
| ESAM | 8.6 | **0.754** | **0.837** | 0.916 | **0.583** | **0.751** | 1.07 |
| ESAM-snap | 8.6 | 0.702 | 0.792 | 0.897 | 0.538 | 0.709 | 1.01 |
| Oracle-adj | 33.4 | 0.769 | 0.847 | 0.964 | 0.159 | 0.581 | 0.97 |
| Oracle | 31.8 | 0.784 | 0.871 | 0.966 | 0.172 | 0.607 | 0.90 |

- 「碎片数/GT」：purity≥0.8 且贡献 ≥max(50, 5%) 点的 mask 个数。FastSAM 平均把一个物体
  切成 2.2 块，ESAM 约 1 块，**粒度已经对齐到 ScanNet**。
- Oracle 的 AP 低于 ESAM，因为 oracle 不删除也不打分残余小组（按点数作分数），精度低。
  比较「能否覆盖 GT」时应看最佳 IoU 和召回率，这两项上 ESAM ≈ Oracle-adj < Oracle。
- Oracle-adj 只比 Oracle 低 0.015 IoU：**同一物体的碎片几乎都在 3D 上相邻**，
  合并模型只需看相邻组对。

逐 GT（4,314 个 thing 实例）：

| | FastSAM 最佳单组 | ESAM | ESAM-snap | Oracle-adj | Oracle |
|---|---|---|---|---|---|
| 平均 IoU | 0.530 | 0.728 | 0.672 | 0.738 | 0.753 |
| 被切碎 GT 的平均 IoU | 0.455 | 0.787 | 0.753 | 0.822 | 0.846 |
| 被切碎 GT 的召回@0.5 | 0.385 | 0.883 | 0.863 | 0.927 | 0.971 |

71% 的 GT 上 ESAM 比最佳单组好 0.05 以上，15% 更差（漏检或把相邻物体合在一起）。

## ESAM mask 的组成

- 「组被包含的比例」直方图在 1.0 处有尖峰：多数情况是**整组并入**。
- 0.2–0.8 之间的部分包含占 31%：解码器也会修整组的边界，相当于少量拆分。这部分正是
  ESAM 比 ESAM-snap 多出的 0.05 IoU。

## 对 training-free 路线的含义

1. 「把 FastSAM 的 mask 合并到 ScanNet 粒度」是 ESAM 解码器收益的主要来源，可以作为单独
   的学习目标。
2. 只做合并的上限（Oracle-adj 0.769）已经高于 ESAM（0.754）。一个只学「相邻两组是否属于
   同一物体」的小模型，理论上能追上 ESAM 的逐帧质量。
3. 剩余短板是逐点边界修整（约 0.05 IoU），可以作为后续的 split/refine 头。

## 复现

```bash
source tools/env.sh
python tools/esam_prepare.py      # 16 个场景，约 5 分钟
python tools/esam_export.py       # 官方 ESAM-E 推理并导出逐帧 mask，约 2 分钟
python tools/esam_granularity.py  # 本文的表格和图，约 2 分钟
```

输出在 `work_dirs/esam_eval/granularity/`：
- `summary.md` / `summary.json`
- `per_frame.csv` / `per_gt.csv` / `esam_masks.csv`
- `01_summary_*.png`（指标对比）、`02_esam_composition.png`（组成直方图）、`03_per_gt_scatter.png`（逐 GT 散点）
- `frames/*.png`（逐帧投影到图像：FastSAM / ESAM / Oracle / GT）
