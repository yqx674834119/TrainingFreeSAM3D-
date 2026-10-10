# GroupMergeNet：把 FastSAM 分组合并到 ScanNet 粒度的轻量模型（设计）

目标：替代 ESAM 的 3D U-Net + 解码器，只学一件事：**相邻两个 FastSAM 组是否属于同一个
ScanNet 实例**。FastSAM 保持冻结，逐点特征 Fp 和组特征 Fs 都来自 Query Lifting。

## 1. 为什么这样设计（证据见 `docs/esam_granularity.zh-CN.md`）

| 观察 | 设计决定 |
|---|---|
| ESAM 解码器 77% 的收益来自整组合并（ESAM-snap 0.702 vs ESAM 0.754，FastSAM 0.532） | 只学合并，不学逐点 mask |
| 只沿 3D 相邻（<5 cm）合并的 oracle 仍有 0.769，高于 ESAM | 候选边 = 相邻组对，图很稀疏 |
| 组特征 cosine 的 AUC 只有约 0.79，同类不同实例也很像 | 不能只靠 Fs 阈值，需要边界和几何特征 |
| FastSAM `backbone` 特征最好；减帧均值后 gap 变大 | Fs 用 backbone，并做帧内中心化 |

## 2. 输入：每帧一张组图

- 节点 = Query Lifting 得到的互斥组（20k 点上，平均约 70 个）。
- 边 = 3D 相邻组对：两组之间存在点距 < ε=5 cm（KD-tree）。实测每帧约 73 个节点、410 条边，
  平均度 11（p10–p90：7–16）。
- **节点特征**（不训练、预先缓存）：
  - Fs：组内 backbone 特征（640 维），先减帧均值再平均池化 + L2；可选拼接 proto（32 维）。
  - 几何：点数（log）、质心高度、PCA 三个特征值（形状）、主法向、包围盒尺寸、平均深度。
  - 外观：平均 Lab 颜色；FastSAM 置信度、2D 面积。
- **边特征**（不训练、预先缓存，约 20 维）：
  - 外观：cos(Fs_i, Fs_j)（backbone / neck / proto 各一个）。
  - **边界 Fp 对比**：边界点对（距离<ε）两侧逐点特征的平均 cosine。这比整组平均更局部，
    能区分「椅子贴着桌子」这类相邻但不同的物体。
  - 边界长度（点对数 / 较小组点数）、边界两侧深度差、法向夹角、颜色差。
  - 尺寸比、包围盒 IoU、合并后的凸性（并集 PCA 第三特征值的变化）。
  - **原始 mask 嵌套关系**：互斥化之前，一个组的原始 FastSAM mask 是否被另一个包含。
    SAM 的部件 mask 往往嵌在整体 mask 里，这个线索不需要任何学习。

## 3. 模型（约 6 万参数）

```
node:  h_i = MLP([P·Fs_i, geo_i])            P: 640→32 线性投影, MLP: 48→64→64
edge:  e_ij = MLP([h_i + h_j, |h_i − h_j|, edge_feat_ij])   →64→64
(可选) 1 轮消息传递: h_i ← h_i + mean_j MLP(e_ij)，然后重算 e_ij
out:   p_ij = sigmoid(Linear(e_ij))            对 i,j 对称
```

- 训练：边级 BCE（正样本 pos_weight 或 focal），AdamW，约 20 个 epoch。特征已缓存，
  GPU 上几分钟训练完。
- 推理：每帧约 400 条边，<1 ms。然后在组图上做**平均连接的凝聚聚类**（阈值 τ 在验证集上调），
  不用直接连通分量，以免 A–B–C 链式误合并。合并后 mask 的分数 = 内部边概率均值 × FastSAM
  置信度均值（按点数加权）。

## 4. 标签

对每条相邻边 (i, j)，只在 GT 标注点上计算 purity（与 oracle 定义一致）：

- 两组 purity 都 ≥0.8（且已标注点 ≥30%），多数 GT 相同 → y=1
- 两组都纯但多数 GT 不同 → y=0
- 有一组不纯或基本未标注 → 忽略，不参与 loss。这类组需要拆分，合并解决不了

这样学到的正是 Oracle-adj 的决策，所以 Oracle-adj（0.769）就是这个模型的上限。

## 5. 数据与防泄漏

- **训练**：`/media/yqx/Win10Pro X64/3D_Catch_Seg/downloads/scannet_r2_official_train_20260924_0410/scans`
  里的 120 个**官方训练集**场景。有 `.sens`、mesh、aggregation、segs，与验证集无交集；
  缺 `scene.txt`（axisAlignment），GT 的 bbox 去噪用单位矩阵（在 README 记为偏差）。
  按 40 帧间隔约 6,000 帧、约 40 万个组、约 250 万条边，量足够。
- **验证 / 测试**：本次的 16 个验证场景、809 帧，复用 `tools/esam_granularity.py` 的同一套
  评估，可以直接和 FastSAM / ESAM / ESAM-snap / Oracle 同表比较。

## 6. 成功标准（thing，逐帧平均）

| 指标 | FastSAM | ESAM-snap | ESAM | Oracle-adj（上限） | **目标** |
|---|---|---|---|---|---|
| 平均最佳 IoU | 0.532 | 0.702 | 0.754 | 0.769 | ≥ 0.70 |
| 召回@0.5 | 0.515 | 0.792 | 0.837 | 0.847 | ≥ 0.78 |
| AP50 | 0.232 | 0.709 | 0.751 | — | ≥ 0.65 |

## 7. 消融顺序

1. 只用 cos(Fs) + 阈值（不训练）→ 下限。
2. 加几何和边界特征的 MLP。
3. 加原始 mask 嵌套特征。
4. 加 1 轮消息传递。
5. 聚类方式：连通分量 vs 平均连接。

## 8. 后续扩展

- **split/refine 头**：处理不纯的组（ESAM 中 31% 的组被部分包含，值约 0.05 IoU）。
- **跨帧合并**：用同一个边分类器给「当前帧实例 – 记忆中的实例」打分，对应 ESAM 的
  merge head；边特征换成 3D 重叠、包围盒距离、Fs 相似度。
- **stuff 处理**：wall/floor 单独一个「背景」节点头，对应 ESAM CA 不输出 stuff 的设定。

## 9. 代码位置（遵循 AGENTS.md）

- `freesam3d/merging/graph.py`：建相邻图、计算节点和边特征（纯 numpy/torch）
- `freesam3d/merging/model.py`：GroupMergeNet
- `freesam3d/merging/cluster.py`：平均连接聚类
- `freesam3d/merging/labels.py`：由 GT 生成边标签（复用 `evaluation.instance.oracle_merge` 的 purity 定义）
- `tools/merge_build_cache.py`、`tools/merge_train.py`、`tools/merge_eval.py`
- `configs/merge_net.yaml`
