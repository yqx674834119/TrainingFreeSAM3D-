# TrainingFreeSAM3D

ESAM（训练版）的 training-free 平行实现：只用冻结的 **FastSAM-x** 作图像模型；不训练 3D 骨干网络，只训练 <1M 参数的小头（2D 实例头 / 3D 几何头 / 图合并模型）。
项目文件组织规则见 [AGENTS.md](AGENTS.md)。

## 进度

| ESAM 模块 | 本项目 | 状态 |
|---|---|---|
| Query Lifting | `freesam3d/lifting/` | ✅ baseline 完成 |
| Query Refinement / Decoder | 验证：ESAM 解码器 ≈ 把 FastSAM 组合并到 ScanNet 粒度 | ✅ 已验证，见 [docs/esam_granularity.zh-CN.md](docs/esam_granularity.zh-CN.md) |
| 轻量合并模型 GroupMergeNet | `freesam3d/merging/` | ✅ 已训练，见 [docs/merge_net.zh-CN.md](docs/merge_net.zh-CN.md)（设计：[merge_net_design](docs/merge_net_design.zh-CN.md)） |
| 统一 MergeNet（帧内 + 跨帧 + 物体合并 + 节点排序，一个模型） | `freesam3d/streaming/unified_flow.py`, `unified_train.py`, `merging/model.py` | ✅ 已完成：16 场景 AP50 0.422（旧管线 0.311、ESAM 0.589）；第 3 阶段（排序诊断 + 包含度去重）同水平——masks 74→60、dup_rate 0.24→0.17、AP 中性，见下与 [docs/findings.zh-CN.md](docs/findings.zh-CN.md) |
| 2D 实例特征头（第 4 阶段：冻结 FastSAM 多层特征 + PartField 式对比学习） | `freesam3d/models/instance_head.py` | ❌ 探针未过门槛（按规则未接入）：最难配对 0.672（手工 0.792）；第 4b 阶段修配方后仍只有 0.653，而外观信息上限本身只有 0.721——原 0.85 门槛属标定错误，见下 |
| 排序瓶颈（第 5 阶段：排序上限 / 掩码集合上界 / 成对几何边） | `freesam3d/analysis/`, `tools/phase5_*.py` | ✅ 已完成（结论均为负）：P1 判定缺口是信息、不训排序头；P3 三个机制不达建造线；P4 探针过门槛但端到端未过、不接入，见下 |
| 提议打分（从 FastSAM mask 层级里选） | `freesam3d/merging/proposals.py`, `scorer.py` | ✅ 见 [docs/mask_scorer.zh-CN.md](docs/mask_scorer.zh-CN.md) |
| Query Merging（跨帧，流式 Query Memory） | `freesam3d/streaming/` | ✅ 见 [docs/streaming.zh-CN.md](docs/streaming.zh-CN.md) |
| CVPR 投稿规划 | [docs/cvpr_plan.zh-CN.md](docs/cvpr_plan.zh-CN.md) | 📝 方法与实验设计初稿 |

环境：`source tools/env.sh`（venv + CUDA 动态链接，见 AGENTS.md §4）。

## ESAM 粒度对齐验证（摘要）

在 16 个 ScanNet 验证场景、809 帧上运行官方 ESAM-E，逐帧比较 FastSAM 分组、ESAM 解码器
输出和 GT oracle（thing 实例，平均最佳 IoU / 召回@0.5）：

| FastSAM | ESAM-snap（只整组合并） | ESAM | Oracle-adj（只合并相邻组） | Oracle（任意合并） |
|---|---|---|---|---|
| 0.532 / 0.515 | 0.702 / 0.792 | 0.754 / 0.837 | 0.769 / 0.847 | 0.784 / 0.871 |

ESAM 每个 mask 平均合并 7 个 FastSAM 组，解码器 77% 的提升来自整组合并。详见
[docs/esam_granularity.zh-CN.md](docs/esam_granularity.zh-CN.md)；复现命令为
`tools/esam_prepare.py`、`tools/esam_export.py`、`tools/esam_granularity.py`。

## GroupMergeNet（摘要）

40k 参数的边分类器 + objectness 头，在 ESAM 官方 1,201 个训练场景（8.8k 帧）上训练 2 分钟，
判断相邻 FastSAM 组是否同一物体，再做平均连接聚类。同一 809 帧验证集、同一 3D GT：

| | FastSAM | **GroupMergeNet** | Oracle-adj（上限） | ESAM-snap | ESAM |
|---|---|---|---|---|---|
| 平均最佳 IoU | 0.532 | **0.627** | 0.757 | 0.702 | 0.754 |
| 召回@0.5 | 0.513 | **0.686** | 0.827 | 0.792 | 0.837 |
| AP50 | 0.237 | **0.506** | 0.567 | 0.709 | 0.751 |

推理 100 ms/帧（FastSAM 72 + 合并模块 28，全部矩阵运算，无 KD-tree；ESAM-E 3D 部分 110 ms）。
单次调用入口 `freesam3d/merging/infer.py::GroupMerger`。训练前的探针、消融和失败分析见
[docs/merge_net.zh-CN.md](docs/merge_net.zh-CN.md)。

## 提议打分（摘要）

FastSAM 的原始重叠 mask 本身是「部件 ⊂ 整体」的层级，整体 mask 常常已经在里面（每个 GT 的最佳
原始 mask IoU 0.672，互斥组只有 0.508）。所以把问题从「合并碎片」改成「从候选里选」：候选 =
原始 mask ∪ GroupMergeNet 合并 mask，一个 28k 参数的打分器预测每个候选的 IoU，再做 NMS：

| | GroupMergeNet 合并 | **提议打分 + NMS，top-20** | ESAM-snap | ESAM |
|---|---|---|---|---|
| 平均最佳 IoU / 召回@0.5 | 0.627 / 0.686 | **0.688 / 0.793** | 0.702 / 0.792 | 0.754 / 0.837 |
| AP50 / AP25 | 0.506 / 0.665 | **0.626 / 0.741** | 0.709 / 0.800 | 0.751 / 0.822 |

细节、消融与剩余差距分解见 [docs/mask_scorer.zh-CN.md](docs/mask_scorer.zh-CN.md)。

## 流式 Query Memory（摘要）

逐帧的最终 mask 进入一张按 5 cm 体素去重的稀疏 (体素, query) 记忆（对标 ESAM `OnlineMerge`，但大小随
场景体积而非帧数增长）：跨帧匹配分 = max 包含度 × 投影特征余弦（投影头 395k 参数，InfoNCE，对标 ESAM
的 MergeHead），匈牙利指派，入库门槛，重叠合并。16 个验证场景、ESAM 的网格顶点与 GT，两边都做 ESAM 的
mesh-segment 平滑：

| | 逐帧堆叠（无记忆） | 记忆，纯几何 | **记忆 + 投影特征** | GT 提议上限 | ESAM-E |
|---|---|---|---|---|---|
| AP50 / AP25 | 0.080 / 0.265 | 0.272 / 0.450 | **0.365 / 0.511** | 0.513 / 0.645 | 0.589 / 0.722 |
| 平均最佳 IoU / 召回@0.5 | 0.402 / 0.357 | 0.436 / 0.436 | **0.458 / 0.498** | 0.567 / 0.647 | 0.691 / 0.726 |

记忆每场景约 37 个 query、2.6 万条记录（ESAM 的 `cur_masks` 为 K × 20000 × 帧数 位），update 7.6 ms/帧，
整条流水线 98.8 ms/帧（`tools/merge_benchmark.py --memory proj`）。差距主要来自逐帧提议质量（GT 提议喂
同一记忆 0.513）；设计、消融与差距分析见 [docs/streaming.zh-CN.md](docs/streaming.zh-CN.md)。

## 统一 MergeNet（第 2 阶段，摘要）

一个 41 548 参数的 `UnifiedMergeNet` 取代 GroupMergeNet：帧内边、block↔object 边、object↔object 边共用同一套
权重与消息传递，另加一个节点质量头（预测节点与它 GT 实例的 IoU）用于输出排序。分组三步：帧内边平均连接成
block → 每个 block 并入分数最高且过阈值的接触物体 → 接触的物体对按阈值合并；输出按节点质量排序、只做 NMS、
不丢任何单帧物体。训练数据由第 1 阶段的规则在线跑一遍采集（A 轮），再用 A 轮模型跑一遍（B 轮），两轮合并
训练；三类边的阈值只在 60 个留出场景上标定。最终配置：A∪B seed0，intra 0.80 / block-object 0.60 /
object-object 1.01（第 3 步关掉）。16 个验证场景、ESAM 网格协议、segment_smooth 后：

| | 逐帧堆叠（无记忆，旧管线） | 第 1 阶段 | 旧管线 P1 | **第 2 阶段统一模型** | ESAM-E |
|---|---|---|---|---|---|
| AP50 | 0.080 | 0.287 | 0.311 | **0.422** | 0.589 |
| AP25 | 0.265 | — | — | **0.620** | 0.722 |
| 平均最佳 IoU | 0.402 | — | — | **0.585** | 0.691 |
| 召回@0.5 | 0.357 | — | — | **0.635** | 0.726 |

**主要缺口在排序，不在合并**：按观测帧数排 AP50 只有 0.271，节点质量头把它抬到 0.422，而完美排序上界 0.634
已经高于 ESAM 的实际 0.589 —— 掩码集合够用，是排序把分数压下来。退化检查（关掉历史、只用单帧）0.6308 vs
GroupMergeNet 0.6340（门槛 ≥0.62），即没有丢掉帧内能力。逐帧耗时 49.5 ms（阶段 2 模块），加 FastSAM 约 72 ms
≈ 120 ms/帧。四个问题的完整回答见 `work_dirs/stream_v2/phase2/summary.md`，逐条发现见
[docs/findings.zh-CN.md](docs/findings.zh-CN.md) 的 F2.1–F2.14。

## 统一 MergeNet · 第 3 阶段（摘要）

不换模型，回答"排序头为什么只学到 41%"并做免费清理。**诊断**：质量头对真实 IoU 的相关在训练图上
0.687、留出上 0.608（只差 0.08）→ 不是过拟合，是节点侧信息不足；排在最前的错误答案里重复为 0。
**去重**（NMS 后丢"被更高分 mask 包含 ≥0.7"的 mask，阈值在留出上选）：16 场景 masks 74.4→60.2、
dup_rate 0.240→0.168、frags/GT 1.21→0.97，AP50 0.422→0.413（留出 +0.003，方向不一致、都小）→
清 mask 免费，AP 中性。**关系线索**（8 列物体间关系 + 场景内成对排序损失）：三种接线在留出上都不
高于 baseline（最好的 relq 0.2899 vs 0.3031；拼进共享主干会经边决策把 mask 弄坏），不纳入。
**扩候选**（接触膨胀 2、质心+外观规则）无正贡献，不纳入。**物体级自注意力**：条件（留出 ≥+0.03）
未满足，未加（已实现并冒烟验证）。最终配置 A 轮 seed0 @0.75/0.60/1.01 + 去重 0.7；排序头吃到约
38% 的完美排序空间，oracle 上界 0.636 仍高于 ESAM 0.589——差距结构与第 2 阶段一致。耗时 53.6 ms/帧
（阶段 3 模块）。完整回答见 `work_dirs/stream_v2/phase3/summary.md`，逐条发现见
[docs/findings.zh-CN.md](docs/findings.zh-CN.md) 的「阶段 3」（F3.1–F3.6）。

## 实例特征头 · 第 4 阶段（探针，未接入）

冻结 FastSAM 六层特征（backbone / neck_p3–p5 / proto_hidden / proto）训一个 258 592 参数的 2D 实例头
（PartField 式多正样本 InfoNCE + 3D 与特征困难负样本，3 个种子，6.4 分钟/种子），碎片特征 = 成员点
特征均值（点级先减帧均值再 L2）。在 60 个留出场景、与 F1.7 完全相同的配对上探针（14.2 万条跨帧边、
11.3 万条帧内边）：

| 最难配对（再观测 vs 相邻不同物体） | 帧内相邻碎片 | 全部跨帧边 |
|---|---|---|
| 学到 0.672±0.003 | 0.719±0.003 | 0.673±0.004 |
| 手工 15 列 0.792 | GroupMergeNet 0.869 | 手工 15 列 0.820 |

门槛（最难 ≥0.85 且帧内 ≥GroupMergeNet；**该 0.85 在第 4b 阶段被判定为标定错误——它高于外观信息上限本身**）3/3 种子未过，按任务书「过门槛才接入」停止：第 5 步三臂
对比与验证集重跑未做，统一流程最终配置与第 3 阶段相同。逐层消融（去掉任何一层都不变好，neck_p4
贡献最大、neck_p3≈0）与瓶颈判断（训练配方为主因：头弱于它自己的输入；纯外观路线有真实上限：补上
接触量也只有 0.736 < 0.792）见 `work_dirs/stream_v2/phase4/summary.md` 与
[docs/findings.zh-CN.md](docs/findings.zh-CN.md) 的「阶段 4」（F4.1–F4.7）。

## 实例特征头 · 第 4b 阶段（配方修复 + 几何判定，仍未接入）

第 4 阶段门槛未过后，第 4b 阶段先做两个**不训练**的判定探针，再按判定决定范围（协议与第 4 阶段
逐位一致：60 个留出场景，跨帧 141 745 / 帧内 112 767 / 最难 134 844 条边）：

- **P2D.0（外观信息上限）**：六层冻结特征各建一份碎片级缓存跑同一个探针，最难配对最好是
  **backbone 0.7208**（其余 0.660–0.706）；同一批层在**逐点**上最好是 proto 0.8396（backbone 0.7707）
  ——碎片级与点级的最优层不是同一层。判定：落在 0.70–0.78 → 轨道 A 只做低成本修复。
  **关键旁证：外观上限 0.7208 本身就低于 §四 门槛 0.7723（= 手工 15 列 − 0.02），所以该门槛与 0.85
  同属标定错误（定在信息上限之上）。**
- **P3D.0（几何信息上限）**：逐碎片 16 维几何描述子（bbox/尺寸/PCA 线状-面状-散射/离地高度/法线
  一致性/点密度）最难配对只有 **0.6112**（≤0.70）。结构性原因：单碎片描述子只能表达「形状像不像」，
  表达不了「重叠/包含多少」，而后者才是手工几何列里唯一有信号的部分（成对量）→ **不做碎片级 3D 头**。
- **轨道 A 五点配方阶梯**（各 1 个种子）：逐点层面三条修复都有效（0.6836 → 0.7150，+0.031），但碎片级
  探针**没有兑现**（0.6717 → 0.6531），回退来自修复 1 与 3 的**组合**（各自单独接近中性）→
  **A 门槛未过，且按现状不可达**。另测：推理端中心化必需（去掉掉 0.09–0.10）。
- **B/B2 未做、C 未做 → §五 条件性接入未触发，没有跑任何一臂**；特批条件（手工 15 列 + 学到余弦 +
  接触量 ≥ 手工 15 列 + 0.01）实测最大 +0.0072，不满足。
- **16 个验证场景未重跑**（最终配置未变，仍为 AP50 0.422 / oracle 0.635 / ESAM 0.589）；
  `merge_cache/val` 未被改动。每帧 12.77 ms（读图 1.28 + FastSAM 前向 10.14 + 2D 头 0.51 + 采样池化 0.85）。

完整回答见 `work_dirs/stream_v2/phase4b/summary.md`，逐条发现见
[docs/findings.zh-CN.md](docs/findings.zh-CN.md) 的「阶段 4b」（F4b.1–F4b.7）。

## 排序瓶颈 · 第 5 阶段（排序上限 / 掩码集合上界 / 成对几何边，未接入）

任务书把留出 AP50 的 0.213 缺口（quality 0.3069 → 完美排序 0.4519）定位在排序上。
第 5 阶段先量信息上限再决定是否训排序头（全部只在 60 个留出场景上做，16 个验证场景未重跑）：

- **P1（排序信息上限，不训练）**：口径逐位复现第 3 阶段留出扫描。每族特征（self / 外观 / 阶段 4 头 /
  关系 / 时序）的留一逻辑回归 AP50 天花板都低于现有头（0.19–0.23 vs 0.3069）；全特征 0.2642
  （GBM 0.2928；再加现有头输出 0.3437）→ **C 档：缺口是信息，不训排序头**（P2 按任务书跳过）。
  外观族每空间只有 1 个自由度（相干度）；规模类是仅有的正相关族。
- **P3（掩码集合上界）**：原始 FastSAM mask 全池单独更差（0.3199）；∪现有候选 + 去重 0.5214
  （需要给原始提议配打分器，未实现）；GT 切割上界 0.6964，但可实现机制只到 +0.02（连通裁剪）/
  +0.005（外观一致性裁剪），均 < +0.05 建造线 → 不实现拆分机制；「22% 混合」前提修正（混合候选
  只占漏检候选约 10%）；去重不改变 oracle 上界。
- **P4（成对几何边，只做一次）**：44 列 [e_a,e_b,e_a⊙e_b,|e_a−e_b|] 追加在 29 列后（离线扩列 A 轮图 +
  `phase5.p4.geom_pair` 开关，默认关）。探针过门槛（最难配对 hand15 0.7923 → 0.8025 ≥ 0.7623），
  但端到端 seed0 未过：quality 0.2832 vs 基线 0.3069、oracle 0.4358 vs
  0.4519，且低于基线三个种子的最差（0.3069/0.3020/0.2932，非种子波动）→ **不接入**（训练图 AUC 变好没有迁移到留出，见 F5.4）。
- **16 个验证场景未重跑**（最终配置未变：AP50 0.422 / oracle 0.635 / ESAM 0.589）；`merge_cache/val` 未改动。

完整回答见 `work_dirs/stream_v2/phase5/summary.md`，逐条发现见
[docs/findings.zh-CN.md](docs/findings.zh-CN.md) 的「阶段 5」（F5.1–F5.4）。

## 模块 1：Query Lifting

| 步骤 | ESAM | TrainingFreeSAM3D |
|---|---|---|
| 1a 分组 | SAM/FastSAM 重叠 mask → 互斥分组 | 同 ESAM：按面积降序写入，小 mask 覆盖大 mask |
| 1b 点云 | 深度+位姿 → 世界坐标，采样 20,000 点 | 同 ESAM |
| 1c 补组 | 未覆盖点 KMeans(20) | 同 ESAM（聚类特征 `[xyz, rgb]`，<20 点并为一组） |
| 2 逐点特征 | 3D U-Net + Memory Adapter | FastSAM 特征图在每个点的像素位置双线性采样 |
| 3 区域特征 | Geometric-aware Pooling | 组内平均池化 + L2 归一化（baseline） |

FastSAM 没有 SAM 的 ViT 和 neck。各特征层与 SAM 的对应关系，以及 Personalize-SAM
如何取特征，见 [docs/fastsam_features.zh-CN.md](docs/fastsam_features.zh-CN.md)。简要对应：

- `backbone`（L9 SPPF，640 维，stride 32）≈ SAM「neck 之前」
- `neck_p3`（L15，320 维，stride 8）≈ SAM「neck 之后」（mask 解码器的输入）
- `neck_fpn`、`proto_hidden`、`proto`：多尺度 neck、解码器中间层、32 维 mask 原型

### 运行

```bash
cd /home/yqx/SegmentAnything3D/TrainingFreeSAM3D
source tools/env.sh          # venv + CUDA 动态链接（见 AGENTS.md §4）
python tools/run_query_lifting.py --output work_dirs/query_lifting/baseline_mean_l2
# 变体：先减帧均值再池化
python tools/run_query_lifting.py --output work_dirs/query_lifting/center_mean_l2 \
    --opts pooling.method=center_mean_l2
# 只跑一帧 / 只要部分特征
python tools/run_query_lifting.py --inputs <frame.npz> --opts "features=[backbone,neck_p3]"
pytest -q tests
```

每帧耗时（含 GT 和画图）：GPU 约 5 秒，CPU 约 11 秒。

### 输出（每帧一个目录）

| 文件 | 内容 |
|---|---|
| `01_overview.png` | RGB / 深度 / FastSAM 原始重叠 mask / 互斥分组 / ScanNet GT（类别+ID）/ 20k 采样点分组 |
| `02_feature_pca.png` | 各特征图前 3 个 PCA 分量的伪彩色图，用来看特征编码了什么、分辨率多粗 |
| `03_gt_similarity_<feat>.png` | **GT 实例 × GT 实例** cosine 热力图，标签是 `类别 #实例ID`，按类别排序，粗线分隔类别 |
| `04_mask_similarity_<feat>.png` | **FastSAM mask × mask** cosine 热力图。标签 `类别 #GT-ID \| m<FastSAM编号> (purity)`，按多数票 GT 排序；细线分隔 GT 实例，粗线分隔类别 |
| `05_mask_pair_similarity.png` | 同一 GT 与不同 GT 的 mask 对的相似度分布，附 AUC / gap / NN-hit |
| `points_{rgb,groups,gt}.ply` | 20k 点云，分别按 RGB、分组、GT 上色（MeshLab/CloudCompare） |
| `lifted.npz` | xyz、像素、group_ids、group→FastSAM mask 映射、各特征的 group 特征、GT |
| `metrics.json` | 本帧的全部指标 |

运行目录下还有 `summary.md`、`summary.json`、`summary_features.png`（各特征对比）。

GT 的取法和 ESAM 一致：每个点取最近 mesh 顶点的实例，再用实例 bbox（+5 cm）去噪。
类别名由 `aggregation.json` 的原始 label 经 `scannetv2-labels.combined.tsv` 映射为
`category`。mask 的 GT = 点数多数票，purity = 多数票占比。

### 指标

- **mask-pair AUC**：mask 两两配对，判断「同一 GT 实例」与「不同 GT 实例」的 ROC-AUC。
  它直接衡量「靠特征相似度能否把过分割的 mask 聚回同一物体」。`pure` 版本只统计
  purity ≥ 0.8 的 mask，排除 GT 漏标和欠分割带来的噪声。
- **NN-hit**：对于 GT 被切成 ≥2 个 mask 的情况，每个 mask 最相似的另一个 mask 是否属于同一
  GT。也就是贪心合并第一步的正确率。
- **gap**：同 GT 对的平均相似度减去异 GT 对的平均相似度。它决定能否用固定阈值合并。

### 结果（9 帧：scene0334 / 0414 / 0690 各 3 帧，均值）

| 特征 | 池化 | AUC 全部 | AUC pure | NN-hit pure | gap | GT：同类别实例 / 异类别 |
|---|---|---|---|---|---|---|
| **backbone** | mean_l2 | **0.786** | **0.799** | **0.867** | 0.229 | 0.659 / 0.525 |
| neck_p3 | mean_l2 | 0.602 | 0.606 | 0.792 | 0.010 | 0.951 / 0.958 |
| neck_fpn | mean_l2 | 0.703 | 0.716 | 0.788 | 0.107 | 0.576 / 0.625 |
| proto_hidden | mean_l2 | 0.741 | 0.785 | 0.771 | 0.058 | 0.833 / 0.838 |
| proto | mean_l2 | 0.763 | 0.786 | 0.798 | 0.097 | 0.773 / 0.743 |
| backbone | center_mean_l2 | 0.781 | 0.794 | **0.883** | **0.413** | 0.227 / −0.184 |
| neck_p3 | center_mean_l2 | 0.722 | 0.737 | 0.812 | 0.217 | 0.212 / −0.187 |
| neck_fpn | center_mean_l2 | 0.758 | 0.774 | 0.858 | 0.235 | 0.082 / −0.185 |

分割统计：每帧平均 53 个 FastSAM mask（帧间差异很大，std 48），未覆盖点约 2.4%，mask 的
点加权 purity 为 0.925；每帧约 6 个 GT 实例，其中约 4 个被切成多个 mask。点到 GT mesh 的
距离中位数为 1.4 cm，说明位姿与 GT 对齐正确。

结论（样本只有 9 帧，作为方向参考）：

1. **neck 之前的 `backbone` 最好**：AUC ≈ 0.79–0.80，NN-hit ≈ 0.87。neck 之后的 `neck_p3`
   在 baseline 下几乎没有区分度：所有 cosine 都挤在 0.85–1.0，gap 只有 0.01。
2. 原因是 FastSAM 特征经过 SiLU，带有很大的公共分量（SAM 的 neck 以 LayerNorm 结尾，没有
   这个问题）。减去帧均值（`center_mean_l2`）后，`neck_p3` 的 AUC 从 0.60 升到 0.72，所有
   特征的 gap 明显变大，backbone 从 0.23 升到 0.41，更适合用固定阈值合并。
3. GT 热力图显示 `backbone` 下同类别不同实例（例如两把椅子）更相似（0.66 对 0.53）。特征
   带有语义信息，但这意味着**只靠特征相似度会把相邻的同类物体错误合并**，后续合并需要
   加入几何/邻接约束（对应 ESAM 的 geometric-aware pooling 与合并中的几何项）。
4. 原始 ScanNet GT 有漏标（如 scene0414 的部分椅子、scene0690 的书架），所以热力图里有
   `unlabeled` 块，AUC 的绝对值偏保守。

### 与 ESAM 的差异

- 内参：使用每帧 npz 里的 `intrinsic_depth`。ESAM 使用统一内参
  `fx=fy=577.87, cx=319.5, cy=239.5`。
- 随机性：采样和 KMeans 都固定 `seed`（ESAM 未固定）。
- GT KNN：用 `scipy.cKDTree`，ESAM 用 `pointops.knn_query`，结果等价。
- 和 ESAM 一样，把 640×480 彩色图与深度视为逐像素对齐，没有补偿彩色/深度相机外参。
- 跨帧记忆（`freesam3d/streaming/`）：ESAM 存所有帧的点级 mask（(K, 20000·F)）并在最后 KNN 到网格；
  本项目存 5 cm 体素占用的稀疏 (体素, query) 表，渲染时把网格顶点量化查表。匹配分用 max 包含度 ×
  投影特征余弦（ESAM：MergeHead 余弦 × bbox IoU），新 query 有入库门槛，维护阶段合并体素 IoU > 0.5
  的 query（ESAM 只截 top-K）。进记忆前的逐帧 NMS 包含阈值 0.8（逐帧评估用 1.01）。评估时同样做
  ESAM 的 mesh-segment 平滑（`streaming/evaluate.py::segment_smooth`）。
- 阶段 2 的统一模型（`streaming/unified_flow.py`）：用一个 `UnifiedMergeNet` 取代 GroupMergeNet，
  帧内边、block<->object 边、object<->object 边共用同一套权重与消息传递，另加一个节点质量头
  （预测节点与它 GT 实例的 IoU，输出端用它排序）。分组三步：帧内边平均连接成 block → 每个 block
  并入分数最高且过阈值的接触物体 → 接触的物体对按阈值合并。三类边各自一个阈值，只在 60 个留出场景上标定。
- 阶段 4b 的 3D 几何头（`models/instance_head3d.py`）：对标 ESAM 的 3D U-Net（ESAM 用它把逐点
  backbone 特征编码成 query 特征）。本项目不训练 3D 骨干，只训练 <1M 参数的小头，输入是缓存 npz 里
  的逐点几何（相对帧质心的 xyz、相对地面分位的高度、局部 PCA 法线与线状/面状/散射度）；缓存里没有
  位姿，法线因此用不依赖符号的形式。该模块同时提供 P3D.0 的逐碎片几何描述子（16 维）。
- 与 ESAM 的差别：ESAM 的 MergeHead 只回答"两个 query 是否同物"，没有节点质量回归，也没有"先把本帧
  碎片粘成 block 再跨帧比较"这一步（block 步骤来自第 1 阶段诊断出的失败模式）；ESAM 输出端会丢弃
  单帧物体，阶段 2 只做 NMS、不丢任何物体。训练数据由第 1 阶段的规则在线跑一遍采集（`window_data.py`），
  再用同一个模型跑第二遍（B 轮），两轮数据合并训练。
