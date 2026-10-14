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

在留出场景上用**留出场景选出的方案**跑 `WindowMemory`（W=1），收集每帧的图与边标签（§5.2 的标签规则），报：

- 每个 `pair_feature_names()` 的单列 AUC；
- 逻辑回归（按场景 5 折）AUC；
- 以上按**边类型**（本帧–本帧 / 本帧–过去 / 涉及定稿）和**关系**分层：同一物体再观测（同 GT，`max(contain) < 0.8`）、部件–整体（同 GT，`max(contain) ≥ 0.8`）、相邻不同物体（不同 GT）。

输出 `work_dirs/stream_v2/phase1/probe/{summary.md,probe.json}`。不设门槛，结果写进最终文档。

**实际执行时改了三点**（第 1 阶段已按改后的口径跑完）：

1. 打分用**留出场景选出的方案**，不是 oracle：oracle 打分下每条边的去留都由 GT 决定，
   探针就只剩"GT 能不能被特征复现"，与第 2 阶段要学的东西脱节。
2. window 用 **1**，不是 5：第 2 阶段的跨帧打分器是在 window=1 的图上训的。
3. "按关系分层"改成**两个关系之间**的对比：`relation_of` 是标签的函数（不同 GT → 相邻；
   同 GT → 再观测 / 部件–整体），单个关系内部全是同一类，**层内 AUC 没有定义**。
   而且"部件–整体"由包含度 ≥ 0.8 定义，包含度本身是特征列，那两行接近同义反复。
   见 `findings.zh-CN.md` F1.8。

---

### 4.4 第 1 阶段的结论（已完成，2026-10-13）

- **数据**：见 `work_dirs/stream_v2/phase1/data_inventory.md`。下载的镜像只有 ScanNet 2D 预处理
  （1513 场景，逐帧 RGB-D + 位姿 + 2D 实例/语义 PNG）；网格、3D 传播 GT、场景级 GT、
  super_points 只有 16 个验证场景有。→ 留出场景只能用**体素级代理指标**，与网格协议不可比。
- **训练缓存**：`work_dirs/merge_cache/seq/train`，1201 场景 × 12 帧 = 14182 帧，6.09 GiB，
  三卡约 40 分钟。标签用 2D 实例 PNG（场景内 id 一致，已验证）。
- **留出场景选出的方案**：`two_step` + 规则 `p2a` + 跨帧阈值 0.2（`phase1/selected.json`）。
  两个修复（跨帧阈值单独标定 / 两步分组）在留出场景上把体素 AP50 从 0.0450 提到 **0.0920**（约 2 倍）。
- **验证集一次跑**（网格协议、段平滑、S_nobs+nms）：AP50 **0.287**，没打赢 P1 的 0.311/0.369。
  但合并侧全面变好（IoU 0.553、召回 0.598、overmerge 0.081、frags/GT 1.00），
  **同一批 mask 换成 oracle 排序是 0.598（≈ ESAM 0.589）** → **瓶颈从合并转到排序**，
  这修正了 F0c.4 / F0c.9 的方向。
- **探针**（60 留出场景、277028 条跨帧边）：15 个跨帧列的逻辑回归 AUC 0.8198；
  现用规则在最难的"接触到的负类"上只有 **0.4840**，同样特征能到 0.8579；
  top-1 选对历史物体的准确率 0.8099，但 18.1% 的碎片候选里根本没有正确物体。
- **对第 2 阶段的结论**：训练必要，且要学的是**跨帧边打分器 / 物体排序**，
  不是继续改合并几何。主力特征 `feat_cos`、`inter1_frac_min`、`inter0_frac_min`、
  `contain_a_in_b`、`iou0`、`contain_b_in_a`、`centroid_dist`、`height_diff`；
  注意帧内 14 列在跨帧边上恒为 0。
- 全部细节：`work_dirs/stream_v2/phase1/summary.md`、`docs/findings.zh-CN.md` 阶段 1（F1.1–F1.11）。

---

## 5. 第 2 阶段：统一模型与训练（已完成，2026-10-14）

**目标**：用一个 `UnifiedMergeNet` 取代 GroupMergeNet —— 帧内合并、跨帧合并、物体合并、物体排序共用一个
模型；不新增第二个模型；阈值只在 60 个留出场景上标定，16 个验证场景只做最终汇报。

### 5.1 模型 `freesam3d/merging/model.py::UnifiedMergeNet`（已实现）

- 节点输入 = `node_geom`(11) + `Linear(640→16)(feat)` + 节点类型 one-hot(3：本帧碎片 / 本帧 block /
  历史物体) + `log1p(n_frames)` + `log1p(age)`；
- 边输入 = `[h_a + h_b, |h_a − h_b|, LayerNorm(pair_feat(29))]` + 边类型 one-hot(3：帧内 / block↔object /
  object↔object)；`pair_feat` = 14 列帧内特征（非帧内边恒为 0）+ 15 列跨帧特征（**每条边都算**）；
- 一轮消息传递；两个头：边 logit、节点质量（预测该节点与它 GT 实例的 IoU）。
- 41 548 参数（< 10 万）；共享部分从 `work_dirs/merge_net/full/model.pt` 初始化
  （`init_from_group_merge_net`，实测复制 18/18 个张量）。`GroupMergeNet` 保留不删。

### 5.2 分组流程 `freesam3d/streaming/unified_flow.py`（已实现）

1. 本帧缓存的帧内边过模型 → 平均连接成 **block**（阈值 `threshold`）；
2. 每个 block 与接触到的物体逐对过模型 → 并入分数最高且 ≥ `bo_threshold` 的物体（多个 block 可并入
   同一物体）；都没过阈值时，点数 ≥ `admit_min_points` 的 block 开新物体；
3. 接触的物体对过模型 → ≥ `oo_threshold` 时合并；
4. 输出按**节点质量**排序，只做 NMS，**不丢弃单帧物体**。

`RuleAdapter` 让第 1 阶段的规则（GroupMergeNet 帧内概率 + p2a 包含度 + 物体不互相合并）驱动同一段代码，
因此 A 轮采集与推理走的是同一条路径（差异口径见 findings F2.2）。

### 5.3 训练图采集 `freesam3d/streaming/window_data.py` + `tools/window_unified_collect.py`（已完成）

- 每个训练场景按帧在线跑上面的流程，每帧存两张图：帧内图（本帧碎片）与跨帧图（block + 历史物体），
  含节点/边张量与标签，`node_ptr`/`edge_ptr` 按帧分片，外观特征存 fp16。
- 边标签：两端纯度 ≥ 0.6 且多数 GT 相同 → 1，均 ≥ 0.6 且不同 → 0，否则**掩掉**（不当负类）。
- 节点标签：历史物体 → 场景结束时对体素 GT 的最佳 IoU（排序目标）；碎片与 block → objectness
  （`merging.labels.objectness_labels` 的口径）。合并事件用并查集折到最终物体上。
- A 轮由第 1 阶段的规则驱动（`--round A`）；B 轮由 A 轮模型驱动（`--round B --checkpoint ...`）；
  另有 `--round intra_extra`：从 `work_dirs/merge_cache/train` 采纯帧内图（无记忆），补充第 1 步的数据。
- 输出 `work_dirs/stream_v2/phase2/graphs/{A,B,intra_extra}/<scene>.npz`：A 轮 1141 个非留出场景，
  加 intra_extra 共 2282 个文件 / 35 816 帧。留出场景与 16 个验证场景都不参与训练。

### 5.4 训练 `freesam3d/streaming/unified_train.py` + `tools/window_unified_train.py`（已完成）

- 损失 = 边 BCE（**三类边各自先取均值再平均**，三类权重相等）+ 节点质量 MSE（sigmoid 输出对目标）。
- 一个训练步把 32 帧拼成一张不相交图（边是帧内的，只需按帧偏移节点编号）。
- 三个随机种子（0/1/2），配置见 `configs/merge_net.yaml::phase2`；输出
  `work_dirs/stream_v2/phase2/model/<round>/seed<k>/model.pt` 与 `train_summary.json`（含逐类边 AUC、
  节点质量 Spearman 诊断）。
- 先训 A 轮；B 轮采集后合并 A∪B 继续训练。

### 5.5 阈值标定 `freesam3d/streaming/unified_sweep.py` + `tools/window_unified_sweep.py`（已完成）

在 60 个留出场景上扫阈值，指标是体素代理 AP50（`voxel_scene_eval`，与第 1 阶段同口径；任务 0 已验证
它与网格 AP50 排序一致，Spearman 0.979，见 F2.1）。先三类边共用一个阈值，再各自一个阈值作对照；
每个三元组跑一遍全部留出场景，结果写 `work_dirs/stream_v2/phase2/sweep/*.json`，`--report` 选最优并写
`selected.json`。三个种子各跑一遍共享网格用于选种子。

### 5.6 退化检查 G1 `tools/window_unified_degrade.py`（已完成）

关掉历史、只用单帧：统一模型套 `SingleFrameAdapter`（节点类型 0、单次观测、只用帧内边、无记忆）跑
`merging/evaluate.py` 的逐帧协议（3D GT、thing ≥ 100 点、810 帧）。同一批帧上 GroupMergeNet 的
mean best IoU 实测 **0.6326**（F2.4），统一模型应 ≥ 0.62。

### 5.7 最终汇报 `tools/window_phase2_final.py`（已完成）

16 个验证场景、ESAM 网格协议、segment_smooth 之后：AP50/AP25/mean best IoU/recall/masks/dup_rate/
overmerge/frags-per-GT，逐行附完美排序（GT IoU）上界；按"只做跨帧打分 / ＋物体-物体合并 / ＋节点质量排序"
拆项；并把没有拿到物体的 block 归因成"物体尚未出现 / 存在但不在候选中 / 是候选但分数低 / block 自身
GT 不纯"（`streaming/unified_probe.py`）。输出 `work_dirs/stream_v2/phase2/final/`。

### 5.8 没做 / 改动

- 原计划的 `raw_cover` 消融特征：暂不做，先看统一模型的收益，必要时再加。
- 原计划的"节点 impurity 头"改成"节点质量头（预测与 GT 实例的 IoU）"：一个头同时承担 objectness 与
  排序，少一份标签口径（碎片/block 的 objectness 就是它与 GT 的 IoU 的特例）。
- A 轮记忆不用 oracle 驱动，改用第 1 阶段的规则（`RuleAdapter`）：oracle 会把 GT 信息带进记忆状态，
  而第 1 阶段的规则是真实可用的驱动，采到的图更贴近推理分布。

---

### 5.9 标定结果（2026-10-14）

- **阈值**：三类边在 60 个留出场景上选为 **intra 0.80 / block↔object 0.60 / object↔object 1.01**（第 3 步
  等于关掉）。分边标定只比一个统一阈值（0.80/0.80/0.80）好 0.0018 AP50，真正决定结果的是 intra 阈值。
- **物体↔物体合并（第 3 步）在最终工作点上不划算**：留出代理上 bo 0.60 时打开它低 0.0014（bo 0.80 时高
  0.0087）；16 个验证场景上（A 轮）打开它让 AP50 0.445 → 0.436、mean best IoU 0.577 → 0.547。它的作用是
  「用掩码精度换检出」，因此最终配置关掉。若以后要重开，应先把 oo 边头的精度做上去（当前它会并错物体）。
- **两轮自举数据（A∪B）没有在验证集上兑现**：16 场景 AP50 0.422 vs 只 A 轮 0.436（各自用自己留出选出的阈值）；留出代理上 A∪B 略好
  （0.3099 vs 0.3053），单帧能力也略低（0.6308 vs 0.6338）。差距在种子噪声内 → 下一步优先投排序头与掩码
  精度，而不是再加一轮数据。
- **验收**：退化检查通过（单帧 0.6308 vs GroupMergeNet 0.6340，门槛 ≥0.62）；16 场景 AP50 0.422 超过任务
  目标 0.369；阶段 2 模块 49.5 ms/帧、41 548 参数。
- 四个问题的完整回答：`work_dirs/stream_v2/phase2/summary.md`；逐条发现（含证据、口径、出处）：
  `docs/findings.zh-CN.md` 的 F2.1–F2.14。

## 6. 第 3 阶段：排序诊断、去重与关系线索（2026-10-14）

> 旧的第 3 阶段计划（在线评估 / 窗口 1-3-5 消融 / `window_eval.py --memory window`）作废：窗口记忆
> 与统一模型在第 2 阶段已合并成一条在线流程（`unified_flow`），窗口大小不再是变量。第 3 阶段的题目
> 改为任务书指定的四件事 + 一个条件项：诊断排序头为什么只学到 41%、免费的包含度去重、扩候选（两案）、
> 关系线索 + 场景内成对排序损失；条件项是物体级自注意力（只有第 3 步留出增益 ≥ +0.03 才加）。

### 6.1 排序诊断（已完成）
`freesam3d/streaming/phase3_diag.py` + `tools/window_phase3_diag.py`：训练侧（A+B+intra_extra 抽样
200 个场景、其中 137 个有跨帧节点）与留出侧（60 场景跑到终态）分别统计质量头 vs 最终体素 IoU 的
Spearman/AUC，并把排在最前的错误答案归类为"重复 / 是更高分正确 mask 的一部分"。结论见 F3.1：
不是过拟合，是信息不足；顶层错误不是重复。

### 6.2 包含度去重（已完成）
`freesam3d/streaming/unified_dedup.py`：NMS 之后按分数贪心、丢弃被更高分 mask 包含 ≥0.8 的 mask；
阈值 {0.6,0.7,0.8,0.9} 只在留出场景上选（`tools/window_phase3_sweep.py --dedup`），验证集不参与。
结果见 F3.2。

### 6.3 扩候选（已完成，两案均否）
接触判定膨胀 1→2（`--touch-dilation 2`）；"质心距离 <0.3 m 且外观余弦 > 阈值"作为 block↔object
候选（`WindowConfig.cand_dist/cand_cos`，在 `unified_flow.assign` 里并入候选集，只作用于
block↔object 对）。两案在 60 个留出场景上的 AP50/dup_rate 变化见 F3.3：膨胀 2 让 mask 变差
（oracle 上限同步下降），质心+外观规则与基线逐位相同（新增边极少、且从不改变任何决策）。

### 6.4 关系线索 + 场景内成对排序损失（修正后重做）
- 节点新增 8 列关系线索（`pair_features.REL_FEATURE_NAMES`：与相交物体的最大 IoU、双向包含度、
  相交物体数、是否其中最大、join 次数与 join 时分数的均值/最小值），在 `unified_flow.assign` 里
  与推理同一代码路径计算；A 轮训练图全量重采（1141 场景，覆盖口径见 F3.4）。
- 训练加场景内成对排序损失（`unified_train.rank_loss`，RankNet，帧内成对、平局跳过）。
- **第一版**把 rel 拼进共享主干 → 经节点嵌入泄漏进边决策、mask 变差（F3.5，两臂均低于 baseline）。
  按任务书"只在排序头的节点输入上加线索"的原意改为质量头输入上的加法 `rel_emb`
  （`UnifiedMergeNet.quality_input`；边路径完全不带 rel，单测断言 rel 变化时边 logits 逐位不变），
  重训 3 个 seed（relq 臂）后的对比见 F3.5 末尾。复现（每 seed 一个进程；YAML 默认关，命令行显式开）：`for s in 0 1 2; do python tools/window_phase3_train.py --rel --rank-weight 0.5 --seeds $s --out work_dirs/stream_v2/phase3/model/relq_s$s & done; wait`。

### 6.5 物体级自注意力：不加（条件未满足）

任务书条件是"第 3 步把留出 AP 提高 ≥ +0.03 才加"。实测 relq 相对 baseline 是 **−0.013**
（0.2899 vs 0.3031，per-seed 区间不重叠，F3.5 末尾）→ 条件未满足，不加。旁证：（a）关系线索三种
接线（主干 / 主干+排序损失 / 仅排序头）全部 ≤ baseline，且 relq 训练侧 Spearman 已是四臂最高
（0.680）仍未兑现 → 瓶颈不在节点侧信息量与排序头容量；（b）候选掩码空间仍是主约束（留出 oracle
上限 0.4519 vs quality 0.3069）。注意力通路已实现并冒烟验证（`ObjectAttention`，1 层单头
dim 64→inner 32，7 392 参数；含修掉的推理期 `frame_id=None` 掩码 bug 与回归单测）；若后续扩了
候选空间可再启用，按同一留出协议判。

### 6.6 最终配置与 16 场景结果（已完成）

最终配置（全部在留出上选）：**A 轮 seed0**（`phase2/model/A/seed0/model.pt`）@ intra 0.75 /
block-object 0.60 / object-object 1.01 + **包含度去重 0.7**。16 场景一次运行
（`tools/window_phase3_final.py`，mesh 协议）：quality 排序 AP50 **0.422（不去重）/ 0.413（去重）**，
oracle 上界 0.635 / 0.636；去重在验证集上 masks 74.4→60.2、dup_rate 0.240→0.168、frags/GT
1.21→0.97、平均最佳 IoU 0.578→0.547、AP50 −0.009（与留出 +0.003 方向相反，两者都小）→ 结论：
**去重是"清 mask"的免费工具，AP 中性**（详见 F3.6）。关系线索（relq −0.013）、扩候选（无正贡献）、
自注意力（条件未满足）均未纳入。对照第 2 阶段 0.422、ESAM 0.589：oracle 0.636 已超 ESAM，差距
结构与第 2 阶段一致（排序为主、mask 质量为次）。

### 6.7 耗时（已完成）

阶段 3 模块 **53.6 ms/帧**（16 场景报告：freeze 10.6±3.9 ＋ build 9.9±1.8 ＋ score 15.0±4.4 ＋
cluster 0.4±0.1 ＋ assign 17.7±5.1，按场景均值±标准差）；隔离基准 54.9 ms
（`tools/merge_benchmark.py --unified`，50 帧、scene0050_00、114 物体）。第 2 阶段同口径 49.5 ms。
模型与第 2 阶段相同（41 548 参数）、去重在输出端（≈0 成本）→ 差值为运行波动，阶段 3 没有增加
推理成本。

---

## 7. 第 4 阶段：文档

- 把第 0–3 阶段结果写进 `docs/stream_merge_v2.zh-CN.md` 新增的「结果」一节（人话 + 表），不删已有内容。
- README：进度表更新状态；「与 ESAM 的差异」加：窗口改错、体素渲染、ESAM 规则重写的偏差。
- AGENTS.md 目录说明里补上新文件（`window_memory.py`、`pair_features.py`、`esam_merge.py`、`window_data.py`）。

---

## 7b. 阶段 4：2D 实例特征头（2026-10-14，已完成；门槛未过，未接入）

- **协议**：冻结 FastSAM 六层特征（backbone / neck_p3 / neck_p4 / neck_p5 / proto_hidden / proto）
  训一个 258 592 参数的 2D 实例头（PartField 式多正样本 InfoNCE + 3D 与特征困难负样本，3 个种子）；
  探针在 60 个留出场景上用与 F1.7 完全相同的配对，门槛 = 最难配对 ≥0.85 且帧内不低于 GroupMergeNet。
- **结果**：门槛 3/3 种子未过（最难 0.672±0.003 vs 手工 0.792；帧内 0.719 vs 0.869）；逐层消融
  （F4.6）显示换层组合救不了；判定训练配方为主因、纯外观路线有真实上限（F4.7）。
- **按规则停止**：任务书 §五「过门槛才接入」——第 5 步三臂对比与 16 个验证场景重跑都未做；统一流程
  最终配置与第 3 阶段相同（AP50 0.422 / oracle 0.635 / ESAM 0.589），`merge_cache/val` 未被改动。
- **产物**：`freesam3d/models/instance_head.py`、`freesam3d/streaming/phase4_probe.py`、
  `freesam3d/vis/phase4.py`、`tools/phase4_{train_head,build_feats,probe,ablate_layers,timing}.py`、
  `tests/test_instance_head.py`（20 个 CPU 测试）；结果在 `work_dirs/stream_v2/phase4/`；发现 F4.1–F4.7。

## 7c. 阶段 4b：2D 头配方修复（轨道 A）+ 3D 几何头判定（轨道 B）（2026-10-14，已完成；无门槛通过，未接入）

- **协议**：与阶段 4 完全相同的探针（window=1、`two_step`/`p2a`/跨帧 0.2、纯度 ≥0.6、60 个留出场景；
  边数逐位一致：跨帧 141 745、帧内 112 767、最难 134 844）。先做两个不训练的判定探针（P2D.0 原始层、
  P3D.0 几何描述子），再按判定决定后续范围；所有选择只在留出场景上做。
- **§一 判定**：P2D.0 最好原始层 backbone 碎片级 0.7208 ∈ [0.70, 0.78] → 轨道 A 只做第 0/1/3 条低成本
  修复；P3D.0 逐碎片几何描述子 0.6112 ≤ 0.70 → **不做碎片级 3D 头**（轨道 B 只剩物体级重编码或放弃）。
- **§二 轨道 A**：四点阶梯 none（阶段 4 配方）/ +0（concat）/ +0+1（concat + 中心化进损失）/
  +0+3（concat + 挖掘修复）/ +0+1+3（全部），各 1 个种子，逐点诊断与探针都跑。**结论：四点阶梯都没过
  §四 的 A 门槛**（要求最难配对 ≥ max(最好原始层 − 0.03, 手工 15 列 − 0.02) = 0.7723），而该门槛高于
  外观信息上限 0.7208 本身——与已弃用的 0.85 门槛同属标定错误（门槛定在信息上限之上）。
- **§三 轨道 B**：按 P3D.0 判定不做碎片级 3D 头；`freesam3d/models/instance_head3d.py` 只保留逐点几何
  （体素桶邻域、逐点 PCA 法线、局部线状/面状/散射度）与逐碎片 16 维描述子，供 P3D.0 与后续物体级
  重编码复用。缓存无位姿 → 法线一律用不依赖符号的形式。
- **§四 / §五**：A 未过；B/B2 未做（其门槛按 P3D.0 已不可达：碎片级几何 + 4 列接触量 0.6965 < 手工
  几何 10 列 0.7510）；C（融合）未做 → **§五 的条件性接入未触发，没有跑任何一臂，也没有特批**
  （特批条件要求「手工 15 列 + 学到余弦 + 接触量」≥ 手工 15 列 + 0.01，实测最大 +0.0072）。
- **§六**：最终配置未变 → **16 个验证场景未重跑**（仍为第 3 阶段的 AP50 0.422 / oracle 0.635 /
  ESAM 0.589）；`merge_cache/val` 未被覆盖或重建。
- **产物**：`freesam3d/models/instance_head3d.py`、`freesam3d/models/instance_head.py`（`fusion='concat'`、
  `build_raw_layer_cache`、`load_interior_mask`、训练配方开关）、`freesam3d/streaming/phase4_probe.py`
  （两行互补逻辑回归）、`tools/phase4b_{train_head,point_diag}.py`、`tools/phase4_build_feats.py`
  （`--layer` / `--descriptor` / `--no-center`）；结果在 `work_dirs/stream_v2/phase4b/`；发现 F4b.1–F4b.7；
  `pytest -q tests` 184 个 CPU 测试全绿。

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
