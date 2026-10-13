# 执行计划：跨帧合并 v2（窗口记忆 + 统一 MergeNet）

给 AI / 协作者执行的计划。设计动机和人话解释见 [stream_merge_v2.zh-CN.md](stream_merge_v2.zh-CN.md)，
执行前先读它和 `AGENTS.md`。本计划只在标注 **【停】** 的地方停下来向用户汇报；其余步骤按顺序做完。

---

## 0. 执行规则（摘自 AGENTS.md，必须遵守）

- 库代码只放 `freesam3d/`；`tools/` 只做参数解析 + 调用；画图只在 `freesam3d/vis/`；实例级指标在 `freesam3d/evaluation/`。
- 路径与超参写在 `configs/merge_net.yaml` 的新段 `stream_v2:` 里，不在代码里硬编码。
- 可训练模块：先写 `tools/*_probe.py`，再训练；阈值只在**留出的训练场景**上标定；16 个验证场景只用于最终汇报，不调参。
- 推理路径用矩阵运算（`one_hot.T @ X`、`cdist`、`scatter_reduce`、`voxel.overlap_matrix`），不用 KD-tree / sklearn / 逐边 Python 循环。
- 新功能附最小单元测试，`pytest -q tests` 必须保持通过；CPU 可跑的测试不得依赖 GPU 或权重。
- 不修改现有 `QueryMemory`、`GroupMergeNet` 的默认行为；新方法用新模块 + 配置开关接入。
- 任何偏离 ESAM 的地方写进 README「与 ESAM 的差异」。
- 结果写到 `work_dirs/stream_v2/<phase>/<tag>/`，含 `summary.md`、`summary.json`、`per_scene.json`。

---

## 1. 术语与数据结构

| 术语 | 定义 |
|---|---|
| 碎片 fragment | 某帧的一个 FastSAM 互斥组（`groups` 中的一个 id）。属性：帧号、点集（20k 点的子集）、5 cm 体素集合、`node_geom`(11)、`node_feat`(C)、点数 |
| 块 block | 窗口内某个**过去帧**里，属于同一物体的碎片之并（`past_unit=block` 时的节点单位） |
| 物体 object | 记忆中的一个实例 = 定稿部分 ∪ 窗口内属于它的碎片 |
| 定稿部分 frozen | 滑出窗口的碎片压成的部分：稀疏 (体素, 物体) 表 + 可加统计量 |
| 窗口 window | 最近 `W` 帧（默认 5，含本帧），其碎片保留并记录当前物体编号，可重新分配 |
| 节点 | 本帧碎片（总是逐碎片）+ 过去帧的块（或碎片）+ 与窗口相接触的定稿物体 |

**可加统计量**（碎片、块、定稿部分统一用这套，合并 = 相加，全部 `index_add_`）：
点数 `n`、`Σxyz`、`Σ xyzᵀxyz`(3×3)、`Σrgb`、包围盒 min/max、特征和 `Σfeat` 与计数、首/末次观测帧、
观测帧数；训练/oracle 模式另存 GT 投票 `gt_votes`（GT id → 点数，稀疏）。
由统计量算 `node_geom`（定义必须与 `merging/graph_gpu.py` 的 `NODE_GEOM` 一致：log size、height、extents(3)、sqrt-eigvals(3)、mean rgb(3)）。

---

## 2. 数据来源

| 用途 | 路径 | 说明 |
|---|---|---|
| 验证（第 0、3 阶段） | `work_dirs/merge_cache/val/<scene>_<fid>.npz` | 809 帧，40 帧间隔，16 场景全部帧；含 `xyz rgb groups node_geom node_feat edges edge_feat gt_instance gt_instance_3d gt_sem_3d group_mask_index fastsam_scores` |
| 场景级 GT 与网格 | `work_dirs/esam_eval/data/scannet200/{points,instance_mask,semantic_mask,super_points}/<scene>.bin` | 与 `tools/stream_eval.py` 相同 |
| ESAM-E 场景输出 | `work_dirs/esam_eval/predictions/<scene>.npz` | 对照 |
| 提议缓存（可选特征） | `work_dirs/merge_cache/proposals/val/` | 原始 mask 与打分器输入，用于 §5.3 的 `raw_cover` 特征 |
| 训练（第 1、2 阶段） | **新建** `work_dirs/merge_cache/seq/train/<scene>_<fid>.npz` | 见 §4.1；现有 `merge_cache/train` 每场景 8 帧、相隔约 200 帧，不能学跨帧 |

验证集 `.sens` 已删除，`merge_cache/val` 不可再生，**只读，不得覆盖**。

---

## 3. 第 0 阶段：GT 上限（不训练）

目标：回答"同样的 GT 输入下，窗口记忆框架的上限是否高于 ESAM 的跨帧合并规则"。

### 3.1 新文件与函数

1. `freesam3d/merging/cluster.py`
   - 新增 `average_linkage_sparse(num_nodes, edges, prob, threshold)`：与 `average_linkage` 语义完全相同
     （两簇之间的分数 = 两簇之间**已有边**概率的均值，取最大者合并直到低于阈值），但只维护有边的簇对（堆 + 邻接字典），复杂度约 O(E log E)。
   - `LINKAGE` 加 `'average_sparse'`。不改 `average_linkage`。
2. `freesam3d/streaming/pair_features.py`
   - `cross_feature_names()`：
     `inter0_frac_min, inter1_frac_min, iou0, contain_a_in_b, contain_b_in_a, centroid_dist, height_diff, log_size_ratio, extent_ratio, bbox_iou, feat_cos, frame_gap, same_frame, has_intra, frozen_count`
     （`inter{r}_frac_min` = 膨胀 r 格后的共享体素数 / min(两者体素数)；`frozen_count` ∈ {0,1,2} 为该边端点中定稿物体的个数；`has_intra` = 是否为本帧缓存边）。
   - `pair_feature_names()` = `edge_feature_names()`（14 个本帧特征，非本帧边填 0）+ `cross_feature_names()`。
   - `pair_features(stats_a, stats_b, inter0, inter1, ...) -> (E, D)`：全部张量运算。
3. `freesam3d/streaming/window_memory.py`
   - `@dataclass WindowConfig`：`window=5, past_unit='block'|'fragment', voxel=0.05, touch_dilation=1, threshold=0.5, linkage='average_sparse', admit_min_points=100, object_merge=True`。
   - `class WindowMemory`：
     - `update(frame, score_fn) -> frame_labels`：一帧的完整流程（§3.2），返回本帧每个点的物体 id（-1 = 未分配）。
     - `render(xyz) -> (masks (K,V) bool, scores (K,))`：物体体素 = 定稿表 ∪ 窗口内成员碎片体素，顶点量化查表（复用 `QueryMemory.render` 的查表逻辑，抽成 `streaming/voxel.py::render_voxel_table`，`QueryMemory` 改为调用它，行为不变）。
     - `stats()`：物体数、记录数、窗口节点数、本帧边数、各阶段耗时。
   - `score_fn(graph) -> prob (E,)`：可插拔；本阶段提供 `oracle_scores`（§3.3）。
4. `freesam3d/streaming/esam_merge.py`
   - `class EsamOnlineMerge`：按 ESAM `oneformer3d/instance_merge.py::OnlineMerge` 重写：
     `S = clamp(cos(feat_mem, feat_new), 0) × bbox_iou_3d(mem, new)`，`S ≤ 0` 置 0，匈牙利（`scipy.optimize.linear_sum_assignment` 在 −S 上），只接受 `S > 0`；
     匹配上：体素并集、特征按计数运行平均、分数运行平均；未匹配：追加；超过 `topk=200` 按分数截断。无入库门槛、无维护合并。
     渲染用体素查表（偏离 ESAM 的点拼接 + KNN，写进 README 差异）。
5. `freesam3d/evaluation/instance.py`
   - `association_stats(masks, gts, thr=0.25)`：`dup_rate`（被 ≥2 个预测各自 IoU ≥ thr 覆盖的 GT 比例）、`overmerge_rate`（覆盖 ≥2 个 GT 各自 ≥ 50% 的预测比例）、`frags_per_gt`（复用 `fragments_per_gt`）。
6. `freesam3d/streaming/evaluate.py`
   - `frames_from_merge_cache(files, gt='3d')`：按场景、帧号排序读入 `merge_cache/val`。
   - `oracle_frame_instances(fr, purity=0.8, eps=0.05)`：用 `evaluation.instance.oracle_merge_adjacent` 得到本帧 oracle 实例（给对照方法当输入），特征 = 成员 `node_feat` 的点数加权均值。
7. `tools/window_eval.py`：入口。参数 `--memory {window,esam_rule,query_memory,gt_assoc}`、`--score {oracle,model}`、`--checkpoint`、`--window`、`--past-unit`、`--no-object-merge`、`--segment-smooth`、`--scenes`、`--tag`。评估复用 `streaming/evaluate.py` 的 `mesh_gtset / scene_metrics / segment_smooth`，并加 `association_stats`。

### 3.2 `WindowMemory.update` 流程（每帧）

1. 读本帧碎片：体素化（`voxel.voxelize` + `mask_voxel_sets`），算统计量。
2. 组节点：本帧碎片（逐个）＋ 窗口内过去帧（`block`：同帧同物体的碎片合成一块；`fragment`：逐碎片）＋ 定稿物体中与上述节点膨胀 `touch_dilation` 格后有接触的那些（一次 `overlap_matrix`）。
3. 组边：
   - 本帧碎片之间：直接用缓存的 `edges`、`edge_feat`（`has_intra=1`）；
   - 其余所有节点对：膨胀 1 格后有共享体素才连边（`overlap_matrix` 一次求出 inter0 与 inter1）；`object_merge=False` 时去掉定稿–定稿边。
4. `prob = score_fn(graph)`，`average_linkage_sparse(..., threshold)` 分组。
5. 分配物体编号：
   - 组内有定稿物体 → 用最小编号；多个定稿物体 → 合并（定稿表并集、统计量相加、窗口内指向它们的碎片重映射）；
   - 组内无定稿物体、有过去帧节点 → 用成员上一帧编号的点数加权多数；
   - 组内全是本帧碎片 → 总点数 ≥ `admit_min_points` 时新开物体，否则 −1；
   - 失去全部成员且无定稿部分的物体删除。
6. 窗口滑动：窗口超过 `W` 帧时，最老一帧的碎片按其物体编号并入定稿部分（体素表并集 + 票数累加 + 统计量相加），−1 的碎片丢弃。
7. 返回本帧点的物体 id；计时（每阶段 `torch.cuda.synchronize`）。

`W=1` 等价于"不改历史"：过去帧立即定稿，只剩本帧碎片 + 定稿物体。

### 3.3 oracle 打分

节点标签 = `gt_votes` 的多数 GT 与纯度（本帧碎片用 `gt_instance_3d`，经 `labels.group_purity`）。
`prob = 1`：两端纯度均 ≥ 0.8 且多数 GT 相同且 GT ≠ 0；其余 `prob = 0`。阈值 0.5。

### 3.4 要跑的组合（全部 16 场景）

| tag | memory | 输入 | 说明 |
|---|---|---|---|
| `gt_assoc` | gt_assoc | 碎片 | 碎片直接按多数 GT（纯度 ≥0.8）归类：碎片 + 体素渲染的天花板 |
| `win_w1` / `win_w3` / `win_w5` | window, oracle | 碎片 | 本框架上限，窗口 1/3/5 |
| `win_w5_frag` | window, oracle, `past_unit=fragment` | 碎片 | 过去帧逐碎片 |
| `esam_rule` | esam_rule | 每帧 oracle 实例 | ESAM 跨帧规则 + GT 帧内合并 |
| `qmem_rule` | query_memory（现有默认几何规则） | 每帧 oracle 实例 | 现有记忆 + GT 帧内合并 |

每个 tag 报：AP50 / AP25 / 平均最佳 IoU / 召回@0.5（未平滑与段平滑后）、物体数/场景、`dup_rate`、`overmerge_rate`、`frags_per_gt`、记忆更新 ms/帧。
汇总写 `work_dirs/stream_v2/phase0/summary.md`。

### 3.5 前置检查（失败即修，不得跳过）

- `merge_cache/val` 的 `xyz` 与 ESAM 网格同坐标系：抽 1 个场景，帧点到最近网格顶点距离中位数 < 3 cm（只在检查脚本里用 cKDTree，推理路径不用）。
- `gt_assoc` 的 AP50 应显著高于 `win_w*`，否则 oracle 或渲染有 bug。

### 3.6 测试 `tests/test_window_memory.py`（CPU）

1. `average_linkage_sparse` 与 `average_linkage` 在 50 个随机图上分组完全一致（同阈值）。
2. `pair_features`：两个手造立方体点集的 IoU / 包含度 / 质心距正确；对称特征交换端点不变。
3. 统计量算出的 `node_geom` 与 `graph_gpu` 对同一点集算出的一致（容差 1e-4）。
4. 合成序列：3 个物体 × 8 帧，每帧每物体被切成 2 块，oracle 打分 → 最终 3 个物体；
   在窗口内人为把一块挂错物体，下一帧被纠正；挂错发生在窗口之外则保持（验证"只改最近几帧"）。
5. 窗口滑动后 `(owner, voxel)` 记录无重复，票数守恒。
6. `EsamOnlineMerge`：两帧同一立方体 → 1 个物体；两帧不相交 → 2 个物体。

### 3.7 【停】汇报 G0

把 §3.4 的表发给用户，并回答：`win_w5` 段平滑后 AP50 是否比 `esam_rule` 高；`win_w5` 相对 `win_w1` 提升多少（改错的价值）。
**等用户确认后再进入第 1 阶段。**

## 3b. 第 0b 阶段：输出端处理与 AP 上限（不训练，已完成）

背景：第 0 阶段发现 oracle 完美合并的覆盖率高于 ESAM（IoU 0.768 vs 0.691，召回 0.877 vs 0.726），
但 AP50 只有 0.362（ESAM 0.589），每场景 112 个 mask（ESAM 40 个）。本阶段验证"损失在输出端"并测出上限。

内容：
1. 只用 win_w1（窗口 1/3/5 在 oracle 下逐位相同），每场景跑一次 oracle WindowMemory，把渲染到 ESAM 网格的物体 mask
   （平滑前/后各一份）与逐物体属性（顶点数、观测帧数 n_obs、FastSAM 置信度均值、不纯点数占比、多数语义是否 stuff）
   缓存到 `work_dirs/stream_v2/phase0b/cache/<scene>.npz`；后续所有组合只对缓存做后处理，不重跑记忆。
2. 去向分解：按 AP50 贪心匹配把段平滑后每个 mask 归为 正确/重复/部分重叠/垃圾，统计来源占比（混合碎片 / stuff / 太小 / 只看一帧）。
3. 32 组后处理：排序 {S_conf, S_size, S_nobs, S_oracle} × 清理 {none, nms, nms+impure(GT), nms+single} × topK {all, 40}，
   段平滑后为主、未平滑同表；不用 GT 的组合阈值固定，不挑最好当结论。
4. 参考行：ESAM-E 原始（复现 0.589）、ESAM-E+S_oracle（ESAM 自己的排序上限）、GT 实例 5 cm 体素渲染到网格 + S_oracle。
5. 代码：`freesam3d/streaming/postprocess.py`（纯函数，矩阵运算；CPU 测试 `tests/test_postprocess.py`）、
   入口 `tools/window_postprocess_eval.py`；`WindowMemory.object_attrs()` 增加逐物体统计（只加字段，不改默认行为，49 项测试全绿）。

结果（16 场景，段平滑后；未平滑值见 phase0b summary）：
- 去向分解：112 个 mask 里垃圾 75.4 个/场景（67%），其中 90% 只被看到 1 帧、88% 由不纯碎片组成、72% <500 顶点、stuff 占 0%。
- 上限：S_oracle + NMS = **0.877**（未平滑 0.772），> ESAM 0.589，> ESAM+S_oracle 0.724；不清理、仅换排序也有 0.832。
- 无 GT 最好：S_conf + NMS + 丢 n_obs==1 = **0.721**；离上限 0.156，其中排序 0.092、清理只差 0.003 → 差距主要在排序。
- 免费信号：只换排序 S_nobs 不清理 0.362→0.647；只做清理不换排序 0.362→0.721。
- 第 0 阶段说法修正："合并层不是瓶颈" → "合并决定覆盖率（IoU 0.458→0.768），AP 卡在输出端的清理和排序"。
- 参考行口径：网格源 GT 走 5 cm 体素再渲染回网格几乎无损（平滑后 1.000）；观测源（逐帧 20k 点、全帧累计）0.965。
  旧文档 0.788 复现不出，以本次重测为准。
- 对后续阶段：除第 2 阶段的跨帧打分器外，输出端还需要"排序 + 清理"环节；n_obs 排序 + NMS + 丢单帧是强 baseline（0.721），
  第 3 阶段用真实（非 oracle）合并复核这些结论。

【停】等用户确认后进入第 1 阶段。

---

## 3c. 第 0c 阶段：真实管线（不用 GT）+ 0b 的免费收尾（不训练，已完成）

背景：0b 的 0.721 建立在 GT oracle 合并之上——oracle 把"混了两个物体的碎片"判成孤立的单帧物体，
"丢掉只出现一帧的物体"于是干净地把它们删掉。真实合并可能把混合碎片粘进真物体，0.721 未必成立。
本阶段把 0b 的免费收尾搬到不用 GT 的真实管线上，量真实 AP50。

内容：
1. P1 = 现有默认管线（QueryMemory + 投影头 + MaskScorer + 逐帧提案），按 `docs/streaming.zh-CN.md` §3 跑。
2. P2 = WindowMemory(window=1)，输入只读 `work_dirs/merge_cache/val`。新打分函数
   `freesam3d/streaming/window_scores.py`：帧内碎片之间用现有 GroupMergeNet；其余边用
   P2a = max(两向包含度)、P2b = max(两向包含度) × clamp(feat_cos,0,1)，直接取 pair_features 的列。
   **不用投影头**：merge_cache 的特征已做过帧内去均值，与投影头的训练分布不符。
3. 阈值取 GroupMergeNet checkpoint 标定的 0.65，另跑固定 0.5 作对照；**不在验证集上调**。
4. 后处理复用 `postprocess.py`，阈值同 0b：排序 {conf,size,nobs} × 清理 {none, nms, nms+丢单帧}，
   不做 top-K 截断；每组另加"S_oracle 排序 + nms"上限行。段平滑为主、未平滑附表。
5. 代码：`freesam3d/streaming/window_scores.py`、`freesam3d/streaming/fragment_fate.py`、
   `postprocess.overmerge_flags()`、入口 `tools/window_real_eval.py`（cache/eval 分阶段），
   结果在 `work_dirs/stream_v2/phase0c/`；测试 `tests/test_window_scores.py`、`tests/test_iou_torch.py`。
   评测用的 IoU 原语加了可选 `device`（CUDA 路径与 numpy 参考逐位对照），默认行为不变。

结果（16 场景，段平滑后）：
- **P1 复现 0.365**（与文档一致）；免费收尾后 0.369，
  mask 35.7 → 33.8，几乎零收益。
- **P2 四个变体最好 0.239 / 0.164 / 0.202 /
  0.172**（p2a_calib / p2b_calib / p2a_0.5 / p2b_0.5），全部低于 P1 的
  0.369，也低于 ESAM 0.589。
- 合并不完美的代价：同排序清理下 0b oracle 0.411 对 P2 真实
  0.084–0.137；
  完美排序下 P2 上限 0.610 对 0b oracle
  0.877 → **合并本身 −0.27**。
  主因是**漏粘**（dup_rate 0.13–0.46、frags/GT 1.1–3.6、overmerge ≤0.061）。
- **「丢单帧」失效**：p2b_calib 0.164 → 0.068、
  p2b_0.5 0.172 → 0.110、P1 基本不变。
  真实欠合并下几乎每个物体 n_obs == 1，删掉的是真物体。
- 混合碎片去向：阈值过严时 88.2% 原地成为孤立单帧物体；阈值可用时 56.6% 被干净粘入本属物体。
- 计时：P2 记忆 64.6–76.1 ms/帧（P1 的 QueryMemory.update 7.3 ms/帧）。
- **对 0b 结论的修正**：F0b.4「差距主要在排序」只在 oracle 合并下成立；真实管线上**先学合并**，
  且阈值要在几何包含度尺度上重标——checkpoint 的 0.65 与包含度不可比，`average_linkage` 的簇间均值
  会把正确证据被帧内错误稀释。

【停】等用户确认后进入第 1 阶段。

---

## 4. 第 1 阶段：训练数据与探针

### 4.1 连续帧缓存

- `tools/merge_build_cache.py` 加参数 `--consecutive K`（默认 0 = 现有行为）与 `--out-subdir`：
  帧号取 `cache.frame_ids(root/scene, dataset.frame_stride, 0)[:K]`（与 `tools/stream_build_cache.py` 相同的取法），场景为训练列表去掉验证场景。
- 运行：`K=12`，输出 `work_dirs/merge_cache/seq/train/`，三卡分片（`--shard i/3`）。格式与 `merge_cache/train` 相同（`CacheBuilder.build`）。
- 记录磁盘占用与耗时到 `work_dirs/merge_cache/seq/train/manifest_*.json`。

### 4.2 留出场景

训练列表（去掉验证场景）排序后**最后 60 个**为留出场景，名单写入 `configs/merge_net.yaml: stream_v2.holdout_scenes`（只写一次，之后不改）。阈值标定与探针只用它们。

### 4.3 探针 `tools/window_probe.py`

在留出场景上用 oracle 打分跑 `WindowMemory`（W=5），收集每帧的图与边标签（§5.2 的标签规则），报：

- 每个 `pair_feature_names()` 的单列 AUC；
- 逻辑回归（按场景 5 折）AUC；
- 以上按**边类型**（本帧–本帧 / 本帧–过去 / 涉及定稿）和**关系**分层：同一物体再观测（同 GT，`max(contain) < 0.8`）、部件–整体（同 GT，`max(contain) ≥ 0.8`）、相邻不同物体（不同 GT）。

输出 `work_dirs/stream_v2/probe/summary.md`。不设门槛，结果写进最终文档。

---

## 5. 第 2 阶段：统一模型与训练

### 5.1 模型 `freesam3d/merging/model.py::WindowMergeNet`

在 `GroupMergeNet` 基础上：

- 节点输入 = `node_geom`(11) + `Linear(C→16)(feat)` + 节点类型 one-hot(3：本帧碎片 / 过去块 / 定稿物体) + `log1p(age)` + `log1p(n_frames)`；
- 边输入 = `[h_a + h_b, |h_a − h_b|, LayerNorm(pair_features)]`，`D = len(pair_feature_names())`；
- 一轮消息传递；三个头：边 logit、节点 objectness（thing 物体）、节点 impurity（纯度 < 0.6）。
- 参数量与 `GroupMergeNet full` 同量级（< 100k）。

### 5.2 图数据生成 `freesam3d/streaming/window_data.py`

对每个训练场景按帧在线跑 `WindowMemory`，每帧保存一张图：节点特征、边、边特征、标签。

- 边标签：两端纯度均 ≥ 0.6 且多数 GT 相同 → 1；均 ≥ 0.6 且不同 → 0；否则不参与。
- objectness 标签：纯度 ≥ 0.6 且多数语义不是 wall/floor/ceiling → 1；未标注不作负样本（同 `labels.objectness_labels`）。
- impurity 标签：纯度 < 0.6 且已标注比例 ≥ 0.3 → 1。
- **A 轮**：记忆由 oracle 打分驱动；**B 轮**：记忆由 A 轮模型驱动（阈值用 §5.4 标定值）。
- 存 `work_dirs/stream_v2/graphs/{A,B}/<scene>.pt`。

### 5.3 可选特征 `raw_cover`（消融）

对每条边：本帧原始 FastSAM mask 中同时覆盖两端本帧部分各 ≥ 80% 的那些 mask 的最高打分器分数（无则 0；需要 `merge_cache/proposals` 对应帧）。默认关闭，做一次开/关对比。

### 5.4 训练与标定 `tools/window_train.py`

- 损失：边 BCE（按边类型重加权，使三类边总权重相等）+ 0.5 × objectness BCE + 0.5 × impurity BCE。
- 先在 A 轮图上训练，再在 A ∪ B 上继续训练。三个随机种子，记录方差。
- 阈值：在 60 个留出场景上扫 `threshold ∈ {0.3, …, 0.9}`，指标为**体素级场景 AP50**：
  `streaming/evaluate.py::voxel_scene_eval`，体素 GT = 跨帧点投票的多数 GT，thing 且 ≥ 20 体素；预测 = 物体体素集合。写入 checkpoint。
- 物体分数（排序用）= 成员 objectness 的点数加权均值。
- 输出 `work_dirs/stream_v2/model/<tag>/model.pt`。

### 5.5 退化检查 G1

`--window 0`（每帧独立、无历史）在验证集逐帧协议（`merging/evaluate.py` 的口径，3D GT）上跑 `WindowMergeNet`：
平均最佳 IoU 应 ≥ 0.60（`GroupMergeNet full` 为 0.627）。低于 0.60 先查 bug 再继续，并在汇报里说明。

---

## 6. 第 3 阶段：在线评估、消融、计时

1. 主结果：`tools/window_eval.py --memory window --score model --checkpoint <tag> --window 5 --segment-smooth`，
   对比现有默认（AP50 0.365）、ESAM-E（0.589）、第 0 阶段的 oracle 上限。
2. 消融：窗口 1/3/5；`past_unit` block/fragment；`--no-object-merge`；`raw_cover` 开/关；去掉 `feat_cos`。
3. 逐帧副产品：用 `update` 返回的本帧标签算逐帧平均最佳 IoU（带历史），与 0.627 / 0.688 / ESAM 0.754 并列报告。
4. 计时：`tools/merge_benchmark.py` 加 `--memory window`，报逐阶段 ms/帧（建图、打分、分组、编号分配、滑窗）。目标记忆阶段 ≤ 20 ms/帧，未达标如实报告并给出瓶颈。

---

## 7. 第 4 阶段：文档

- 把第 0–3 阶段结果写进 `docs/stream_merge_v2.zh-CN.md` 新增的「结果」一节（人话 + 表），不删已有内容。
- README：进度表更新状态；「与 ESAM 的差异」加：窗口改错、体素渲染、ESAM 规则重写的偏差。
- AGENTS.md 目录说明里补上新文件（`window_memory.py`、`pair_features.py`、`esam_merge.py`、`window_data.py`）。

---

## 8. 汇报节点汇总

| 节点 | 时机 | 内容 | 是否等待用户 |
|---|---|---|---|
| G0 | 第 0 阶段完成 | §3.4 表 + 两个问题的答案 | **是** |
| G1 | 退化检查完成 | 逐帧 IoU；若 < 0.60 说明原因 | 否（< 0.60 时是） |
| G2 | 第 3 阶段完成 | 主结果、消融、计时 | 是 |

## 9. 不要做的事

- 不覆盖、不重建 `work_dirs/merge_cache/val`（不可再生）。
- 不在 16 个验证场景上调任何阈值或超参。
- 不修改 ESAM 仓库或其数据目录。
- 不改变现有工具和模块的默认行为（`QueryMemory`、`GroupMerger`、`stream_eval.py`）。
- 第 0 阶段结果出来前不写训练代码。
