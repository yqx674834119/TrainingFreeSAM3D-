# TrainingFreeSAM3D

ESAM（训练版）的 training-free 平行实现：只用冻结的 **FastSAM-x**，不训练任何 3D 网络。
项目文件组织规则见 [AGENTS.md](AGENTS.md)。

## 进度

| ESAM 模块 | 本项目 | 状态 |
|---|---|---|
| Query Lifting | `freesam3d/lifting/` | ✅ baseline 完成 |
| Query Refinement / Decoder | `freesam3d/refinement/`（待建） | — |
| Query Merging（跨帧） | `freesam3d/merging/`（待建） | — |

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
