# 提议打分：把「合并」换成「从 FastSAM 的 mask 层级里选」

验证集：16 个场景、805 帧，ESAM 3D GT，thing 实例 ≥100 点，与 ESAM 粒度报告同口径。
缓存 `work_dirs/merge_cache/proposals/`，模型 `work_dirs/mask_scorer/`，评估 `tools/scorer_eval.py`。

## 0. 结论先行

| 方法 | mask/帧 | 平均最佳 IoU | 召回@0.5 | AP50 | AP25 |
|---|---|---|---|---|---|
| FastSAM 互斥组（不合并） | 69.2 | 0.532 | 0.513 | 0.237 | 0.459 |
| GroupMergeNet 合并 + objectness 打分 | 11.8 | 0.627 | 0.686 | 0.506 | 0.665 |
| **提议打分 + 嵌套 NMS（contain 0.8）** | 11.8 | 0.637 | 0.701 | **0.606** | **0.745** |
| **提议打分 + NMS，不抑制嵌套，top-20** | 17.6 | **0.688** | **0.793** | **0.626** | 0.741 |
| 同一候选集的 oracle 打分（上限） | 12.2 | 0.717 | 0.816 | 0.815 | 0.907 |
| ESAM-snap（ESAM 只整组合并的部分） | 8.6 | 0.702 | 0.792 | 0.709 | 0.800 |
| ESAM 完整解码器 | 8.6 | 0.754 | 0.837 | 0.751 | 0.822 |

- AP50 从 0.506 提到 0.626（+0.12），召回@0.5 从 0.686 到 0.793，**追平 ESAM-snap 的召回**。
- 三个随机种子 AP50 在 0.603–0.606（contain 0.8 设定）。
- 打分器 28k 参数，在 1,201 个训练场景的 40 万个候选上训练 55 秒。

## 1. 为什么换思路：FastSAM 的原始 mask 本身就是一个层级

ESAM 把 FastSAM 的重叠 mask 压成互斥组，再用 47M 参数的解码器把组合并回物体。而 FastSAM 的
原始 mask 是嵌套的（部件 ⊂ 整体）：86% 的 mask 至少有一个父 mask，32% 有 3 个以上。直接量它们：

| 候选集 | 每个 GT 的最佳 IoU | 召回@0.5 |
|---|---|---|
| 互斥组 | 0.508 | 0.466 |
| **原始重叠 mask** | **0.672** | **0.758** |
| GroupMergeNet 合并后 | 0.627 | 0.686 |

49% 的 GT 实例，某个原始 mask 比最佳互斥组好 0.1 以上。最佳原始 mask 的 purity 为 0.82。
也就是说，**整体 mask 常常已经在候选里，问题是选对层级，而不是把碎片拼回去。**

用 oracle 分数（真实 IoU）看各候选集的选择上限：

| 候选集 + oracle 分数 + NMS | AP50 |
|---|---|
| 互斥组 | 0.467 |
| 原始 mask | 0.783 |
| 原始 mask ∪ 合并 mask | **0.815** |

所以合并模块不是白做：它贡献的候选把上限从 0.783 抬到 0.815。两条路线合成一条：
**候选 = 原始 mask ∪ 合并 mask，学一个与 GT 无关的「mask 质量」打分器，再做 NMS。**

## 2. AP50 的损失在哪里（旧流程）

对合并流程做 oracle 替换：

| 替换哪一项为 oracle | AP50 |
|---|---|
| 无（objectness 打分） | 0.510 |
| 打分 = 真实 IoU | 0.697 |
| 合并决策 = Oracle-adj，打分不变 | 0.633 |
| 两者都 oracle | 0.829 |

59% 的输出 mask 与任何 GT 的 IoU <0.25，而 objectness 打分（组平均）把它们排不下去。
打分是最大的单项损失，这与 ESAM 一致：ESAM 的分数把 IoU≥0.5 的 mask 分开的 AUC 为 0.82，
按大小排序只能到 AP50 0.66（它自己的分数 0.70）。

## 3. 打分器

输入（每个候选，不用 GT）：
- 几何 9 维：log 点数、包围盒、PCA 三特征值、扁平度、高度
- 组成 4 维：跨几个互斥组、集中度、FastSAM 置信度均值、proto 特征一致性
- 层级 5 维：是否原始 mask、原始置信度、父 mask 数、子 mask 数、**包含多少个原始 mask**
- objectness 2 维（来自 GroupMergeNet）
- 640 维 backbone 组特征的点数加权平均 → 32 维投影

模型：LayerNorm + 两层 MLP，两个输出头（IoU 回归、IoU≥0.5 分类），28k 参数。
训练集 IoU≥0.5 的候选占 24%。

打分器质量（val）：AUC(IoU≥0.5) **0.857**，Spearman 0.61；校准良好（预测 0.8 的候选真实 IoU 均值 0.83）。
对比：FastSAM 自带置信度 AUC 0.58，objectness 0.61。

消融（contain 0.8）：

| 去掉哪组输入 | AP50 |
|---|---|
| 全部保留 | 0.606 |
| 去掉几何 9 维 | 0.578 |
| 去掉层级 5 维 | 0.591 |
| 去掉组成 4 维 | 0.602 |
| 去掉 objectness | 0.610 |
| 去掉 640 维特征（scorer_scalars） | 0.585 |

几何和层级最重要；GroupMergeNet 的 objectness 对打分器是冗余的。

## 4. 剩余差距分解（contain 0.8）

| | AP50 |
|---|---|
| 打分器选 + 打分器排 | 0.606 |
| 打分器选 + oracle 排 | 0.701 |
| oracle 选 + oracle 排 | 0.815 |

一半在排序质量，一半在选择（NMS 把对的候选压掉了）。最差案例是大物体偏置：墙面大小的
原始 mask 排在小物体（包、床头柜）之前。

## 5. 操作点：要不要抑制嵌套

ESAM 协议的 AP 对每个 GT 贪心匹配，允许输出重叠。不抑制嵌套（部件与整体同时输出）时
召回大涨而精度只小跌：

| 设定 | mask/帧 | IoU | 召回@0.5 | AP50 |
|---|---|---|---|---|
| contain 0.8 | 11.8 | 0.637 | 0.701 | 0.606 |
| 不抑制嵌套 | 24.3 | 0.699 | 0.807 | 0.630 |
| **不抑制嵌套 + top-20** | 17.6 | 0.688 | 0.793 | 0.626 |
| 不抑制 + 分数 ≥0.3 | 7.5 | 0.588 | 0.668 | 0.571 |

分数硬阈值只会伤害，与 GroupMergeNet 的 objectness 过滤结论一致。默认取「不抑制 + top-20」，
与 ESAM 的 `topk_insts=20` 对齐；需要互斥输出时用 contain 0.8。

## 6. 推理开销

`GroupMerger(..., scorer_checkpoint=...)` 在合并之后加一级。候选特征全部是 (P,N)@(N,·) 的矩阵运算
（`proposals_gpu.py`，与 CPU 参考实现 `proposals.py` 的差异 <2e-3，有单元测试对照）：

| 阶段 | ms/帧 |
|---|---|
| FastSAM 前向 | 72 |
| 提升 + 特征 + 建图 + 合并模型 + 聚类 | 27 |
| 候选构造 + 打分器 + 嵌套 NMS | **4.6** |
| **合计** | **103** |

同一条 GPU 路径端到端（top-20，不抑制嵌套）：平均最佳 IoU 0.689，与缓存评估一致。
numpy 版候选构造是 42 ms，其中 `np.where` 算包围盒就占 18 ms；GPU 版 0.4 ms。

## 7. 下一步

1. 排序：训练时按帧做 listwise/pairwise 损失，直接优化排序而不是逐候选回归。
2. 大物体偏置：对 log 点数做分桶归一化，或在损失里按物体大小重加权。
3. 候选集：加入「原始 mask 的两两并集」（相邻且 group_cos 高的对），把上限从 0.815 再抬。

## 复现

```bash
source tools/env.sh
python tools/proposal_build_cache.py --split val          # 809 帧，5 min
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((i+1)) python tools/proposal_build_cache.py --split train --shard $i/3 & done   # 17 min
python tools/scorer_train.py --tag scorer                  # 55 s
python tools/scorer_eval.py --scorers scorer scorer_scalars
python tools/scorer_eval.py --scorers scorer --contain-thr 1.01 --tag eval_no_contain
python tools/merge_benchmark.py --frames 60 --scorer scorer
```
