# GroupMergeNet：训练前验证、设计决定与结果

所有数字来自 16 个验证场景、809 帧（与 `docs/esam_granularity.zh-CN.md` 相同协议），
验证 GT 为 ESAM 3D 传播的逐帧标注。训练数据为 ESAM 官方 ScanNet 2D 导出的 1,201 个训练场景。
缓存：`work_dirs/merge_cache/`，探针：`work_dirs/merge_probe/`，模型与评估：`work_dirs/merge_net/`。

## 0. 结论先行

| 方法 | 参数量 | mask/帧 | 平均最佳 IoU | 召回@0.5 | AP50 |
|---|---|---|---|---|---|
| FastSAM 原始组（不合并） | — | 69.2 | 0.532 | 0.513 | 0.237 |
| cosine 阈值合并（不训练） | — | 9.3 | 0.476 | 0.444 | 0.281 |
| edge_only（14 个边标量） | 1.6k | 15.4 | 0.586 | 0.611 | 0.316 |
| geom_only（+ 节点几何） | 29k | 13.2 | 0.615 | 0.661 | 0.425 |
| **full（+ 640 维 backbone 投影 + objectness 头）** | **40k** | **11.8** | **0.627** | **0.686** | **0.506** |
| Oracle-adj（GT 指导、只合并相邻组：上限） | — | 32.5 | 0.757 | 0.827 | 0.567 |
| ESAM-snap（ESAM 中只整组合并的部分） | 47M | 8.6 | 0.702 | 0.792 | 0.709 |
| ESAM 完整解码器 | 47M | 8.6 | 0.754 | 0.837 | 0.751 |

- 相对 FastSAM：IoU +0.095，召回@0.5 +0.173，AP50 +0.269。三个随机种子的 IoU 在 0.627–0.630、AP50 在 0.487–0.506。
- 距 ESAM-snap 仍差 0.075 IoU、0.106 召回、0.203 AP50；本模块的理论上限（Oracle-adj）为 0.757。
- 推理：FastSAM 72 ms + 合并模块 **28 ms**（提升 12 + 特征 1 + 建图 8 + 模型 2 + 聚类 3）= 100 ms/帧，
  全部矩阵运算、无 KD-tree；checkpoint 167 KB。ESAM-E 的 3D 部分实测 110 ms/帧。

## 1. FastSAM 特征值不值得学？

**探针（训练前）**：按场景分组的 5 折交叉验证，边分类「相邻两组是否同一实例」：

| 输入 | AUC（逻辑回归） | AUC（MLP） |
|---|---|---|
| 10 个几何/颜色/置信度标量 | 0.781 | — |
| + 组特征 cosine（backbone） | 0.784 | — |
| + 交界局部对比（proto） | 0.784 | — |
| + 原始 mask 嵌套 → 14 个标量 | 0.792 | edge_only 5k 参数：0.823 |
| 14 标量 + 几何节点嵌入 | — | geom_only：0.827 |
| 14 标量 + 几何 + 640 维 backbone 投影 | — | full：0.833 |

单列 AUC：centroid_dist 0.756、**group_cos 0.740**、height_diff 0.710、contact_norm 0.650、
normal_angle 0.642、nested 0.606、boundary_cos 0.582。

探针的结论是「特征只值 +0.01 AUC」。**但训练后的下游指标不是这样**：full 比 edge_only 高 0.041 IoU、
0.075 召回、0.190 AP50。原因有两个：
1. 全体边的 AUC 不能代表聚类质量。决定合并结果的是少数难边（相邻的不同物体），特征正是在这些边上起作用。
2. AP50 的差距主要来自 **objectness 头**（见 §5），它需要节点特征：有 backbone 特征时组级 objectness
   AUC 0.794，纯几何只有 0.724。

所以：**FastSAM 特征值得学，但不是作为「合并的主要线索」，而是作为「区分不同类别相邻物体」和
「判断一个组是不是物体」的线索。** 探针若只看全体边 AUC，会得出错误结论。

关于两个具体特征：
- 交界局部对比必须用 stride-4 的 `proto` 层。用 stride-32 的 backbone 算交界对比时，5 cm 内的两点落在
  同一个特征格里，正负边都是 0.99，完全无信号。即便换成 proto，增益也很小（+0.000–0.003）。
- 原始 mask 嵌套（互斥化之前一个 FastSAM mask 是否包含另一个）是最便宜、最稳定的额外线索（+0.008 AUC）。

## 2. 推理开销——全 GPU 矩阵路径，不建 KD-tree

ESAM 的 merge 全是矩阵运算：G×G 的分数矩阵 + 匈牙利匹配。这里照同样思路改写：G≈70 个组，
G×G 矩阵只有几千个元素，所有边特征都能写成 `one_hot(groups).T @ X` 和一次 `cdist` + `scatter`。
`freesam3d/merging/infer.py::GroupMerger` 是单次调用的推理入口，RGB-D 进、实例 mask 出：

| 阶段 | GPU（RTX 5090） | 旧 CPU 路径 | 做法 |
|---|---|---|---|
| FastSAM 前向 | 72 ms | 68 ms | 不变；特征图留在显存，不再拷到 CPU |
| 互斥分组 + 反投影 + 20k 采样 + KMeans 补组 | **12 ms** | 170 ms | 互斥图 = 每像素取覆盖它的最小面积 mask（逐位一致，0.2 ms）；KMeans 用 GPU Lloyd（1 次 k-means++ 初始化，inertia 与 sklearn n_init=10 相差 ±5%，8 ms vs 140 ms） |
| 逐点特征采样 + 组池化 | **1 ms** | 83 ms | `grid_sample` 在特征图所在设备上做（640 维图 65 ms → 0.7 ms） |
| 邻接图 + 14 个边标量 | **8 ms** | 56 ms | 2 cm 体素子采样后 `cdist`（约 5k×5k）；min/count/mean/min 用 `scatter_reduce` 到 G×G |
| GroupMergeNet 前向 | 2 ms | 2 ms | |
| 平均连接聚类 | **3 ms** | 9 ms | 稠密 (G,G) 和/计数矩阵，每次合并是两行两列相加；torch 版反而 13 ms（kernel 启动开销），留在 CPU |
| **合计** | **100 ms** | 365 ms | ESAM-E 的 3D 部分（不含 FastSAM）实测 110 ms/帧 |

GPU 路径的边特征与 CPU 缓存逐列一致（均值/方差差异 <0.03），**训练好的模型不用重训**，
val 平均最佳 IoU 0.632 vs 0.628。纯 CPU（8 线程）跑同一路径为 600 ms/帧，FastSAM 占 366 ms。

CPU 参考实现（`graph.py`，KD-tree）保留为单元测试的对照；`configs/merge_net.yaml: graph.backend`
和 `configs/query_lifting.yaml: lifting.fill_backend` 切换。
优化记录：原始 mask 嵌套矩阵在 4× 下采样网格上算（误差 0.0014），从 130 ms 降到 10 ms。

## 3. 标签：purity 0.8 → 0.6

| purity | 有标签边比例 | 正例率 | 与 purity-0.9 标签一致率 |
|---|---|---|---|
| 0.9 | 20% | 0.73 | 1.000 |
| 0.8 | 29% | 0.66 | 1.000 |
| **0.6** | **51%** | 0.58 | **1.000** |

放宽到 0.6 多拿 75% 的标注边，且凡是两个阈值都给标签的边，标签完全一致。消融：purity 0.8 训练的模型
IoU 0.620 / AP50 0.472，比 0.6 低（见 §6）。缓存里保存了每组的多数 GT 与 purity，阈值在训练时选。

## 4. 上限分解：合并能修什么

对 1,388 个 thing GT 实例（3D GT，≥100 点）：

| 类别 | 占比 | FastSAM 最佳 IoU | Oracle-adj | **full** |
|---|---|---|---|---|
| 单个组已 ≥0.5 | 50% | 0.724 | 0.825 | 0.715 |
| 被切碎、纯组可合并 | **29%** | 0.340 | 0.823 | **0.611** |
| 主要落在不纯组，需要拆分 / 漏检 | 22% | 0.301 | 0.323 | 0.302 |

- 模型在「可合并」类上拿到 oracle 收益的 56%（0.340→0.611，oracle 0.823），66% 的实例达到 IoU≥0.5。
- 在「已经好」的类上有 15% 的实例被过度合并、IoU 下降 >0.1。查了 172 个这样的案例：114 个是**不同类别**
  的相邻物体（背包放在脚凳上、钟挂在墙上），31 个同类别，27 个并入了墙/地面。这是语义问题，
  证明节点需要特征，也说明还有空间。
- 22% 的实例需要拆分或根本没被检出，合并模块无能为力，与 ESAM 粒度报告一致。

## 5. objectness 头：AP50 从 0.353 到 0.506

合并后 AP50 一度只有 0.353：输出 19 个 mask/帧，其中 50% 不到 200 点、32% 落在墙/地面上，只有 18% 对应
一个 thing 物体。按「内部边概率 × FastSAM 置信度」打分排序比按点数排序还差（0.358 vs 0.439）。

对策是在同一个网络上加一个节点头，预测「这个组属于 thing 物体」（标签：纯组且多数语义不是
wall/floor/ceiling；未标注组不作为负样本）。组级 AUC 0.934。合并后每个 mask 的分数 = 组 objectness 的
点数加权平均。效果：

| 打分 / 过滤 | AP50（留出训练帧） | AP50（val） |
|---|---|---|
| 置信度 × 边概率，不过滤 | — | 0.353 |
| objectness，不过滤 | 0.470 | 0.484 |
| objectness + 丢弃 <100 点（ESAM npoint_thr） | **0.506** | **0.506** |
| objectness + <100 点 + 分数 <0.3（ESAM inst_score_thr） | 0.483 | — |

objectness 只用于排序，不做硬过滤：在留出训练帧上扫 min_score ∈ {0.3…0.7}，AP50 只降不升，IoU/召回明显下降。
这个头对合并质量没有影响（有/无该头的 IoU 为 0.627 / 0.630）。

## 6. 消融（val，3D GT）

| 变体 | IoU | 召回@0.5 | AP50 | 说明 |
|---|---|---|---|---|
| full | 0.627 | 0.686 | 0.506 | 默认 |
| 无消息传递 | 0.626 | 0.685 | 0.449 | 合并不变，但 objectness 变差 |
| purity 0.8 标签 | 0.620 | 0.674 | 0.472 | 少 42% 标注边 |
| 无 objectness 头 | 0.630 | 0.690 | 0.405 | 打分退回置信度 × 边概率 |
| seed 1 / seed 2 | 0.630 / 0.627 | 0.689 / 0.691 | 0.487 / 0.489 | 阈值标定在 0.65–0.70 之间 |
| 单连接（连通分量）聚类 | 0.570@0.7 | 0.598 | 0.398 | 链式过合并，如预期 |

阈值在 60 个留出训练场景上按平均最佳 IoU 标定，没有用验证集调参。

## 7. GT 来源

数据集自带的 `instance/*.png` 与 ESAM 的 3D 传播 GT 在共同标注点上一致率 93–98%，但 PNG 在物体边缘有
串色，使所有方法的 IoU 系统性低 0.03–0.07（full 在 2D GT 下为 0.555 / AP50 0.442）。所以
**训练用 2D PNG**（1,201 个训练场景只有 PNG），**验证用 3D 传播 GT**，与 ESAM 报告同口径。

## 8. 与 ESAM 的差距在哪里，下一步

1. **AP50 差 0.20**：ESAM 的分数来自分类头 + mask 质量（`obj_normalization`）；我们的 objectness 只看组
   自身。给合并后的 mask 再算一次 objectness（而不是组平均）是最直接的改进。
2. **过度合并不同类别的相邻物体**：加入「合并后 mask 的凸性/尺寸变化」作为边特征，或在聚类时对
   objectness 高的两个组设更高的合并门槛。
3. **22% 的实例需要拆分**：ESAM 的逐点 mask 预测能做到，这里需要一个 split 头，超出本模块范围。

## 复现

```bash
source tools/env.sh
python tools/merge_build_cache.py --split val                 # 16 场景 809 帧，约 10 min
python tools/merge_build_cache.py --split val --attach-3d-gt  # 附加 ESAM 3D GT（需先跑 tools/esam_prepare.py）
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$((i+1)) python tools/merge_build_cache.py --split train --shard $i/3 & done
python tools/merge_probe.py --split val --gt 3d                # §1 的探针
python tools/merge_train.py --kind full --tag full             # 约 2 min
python tools/merge_eval.py --checkpoints edge_only geom_only full --tag eval_3d --gt 3d
python tools/merge_visualize.py --checkpoint full --eval-tag eval_3d
python tools/merge_benchmark.py --frames 60                   # 全 GPU 推理路径计时 + 与缓存路径对比
```
