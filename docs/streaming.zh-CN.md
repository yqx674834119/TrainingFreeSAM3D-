# 流式 3D 场景 Query 分割与 Query Memory

对标 ESAM 的 `OnlineMerge`（`ESAM/oneformer3d/instance_merge.py:180`）：逐帧 RGB-D 进来，
每帧的实例 mask 与场景级的 query 记忆做匹配、合并、入库，随时可以把记忆渲染到任意点集
（重建网格）上评场景级 AP。本项目的记忆**按空间（体素）去重，不按帧累积**，大小随场景体积增长。

代码：`freesam3d/streaming/`（`voxel.py`、`memory.py`、`projection.py`、`matcher.py`、`evaluate.py`），
入口：`tools/stream_build_cache.py`、`tools/stream_train_projection.py`、`tools/stream_train_matcher.py`、
`tools/stream_eval.py`、`tools/merge_benchmark.py --memory`。配置：`configs/merge_net.yaml` 的 `streaming:`。

## 0. 结论

16 个验证场景，ESAM 协议：最终实例 mask 落在 ESAM 重建网格顶点上，thing 类 GT ≥ 100 顶点，
class-agnostic AP，**两边都做 ESAM 的 mesh-segment 平滑**（ESAM 导出的 `scene_masks` 本身就是平滑后的，
见 `mixformer3d.py:780-801`）。场景平均。

| 方法 | AP50 | AP25 | mean best IoU | recall@0.5 | masks/场景 |
|---|---|---|---|---|---|
| 逐帧堆叠、无记忆（ESAM `ins_cat` 做法，未平滑） | 0.080 | 0.265 | 0.402 | 0.357 | 183 |
| QueryMemory，纯几何规则匹配 | 0.272 | 0.450 | 0.436 | 0.436 | 34 |
| QueryMemory + 跨帧投影特征（**本方案默认**） | 0.365 | 0.511 | 0.458 | 0.498 | 36 |
| 同上，逐帧 NMS 包含阈值 0.65 | 0.373 | 0.512 | 0.462 | 0.501 | 35 |
| 上限：GT 提议喂同一记忆 | 0.513 | 0.645 | 0.567 | 0.647 | 33 |
| ESAM-E 官方导出（训练版，含 3D U-Net 时序细化） | 0.589 | 0.722 | 0.691 | 0.726 | 40 |

不平滑时：规则 0.180、本方案 0.220、GT 提议 0.356（平滑带来 +0.09 ~ +0.16 AP50，对 ESAM 同样成立）。

记忆与耗时（`tools/merge_benchmark.py --scorer scorer --memory proj --frames 60`，scene0050 前 60 帧，RTX 5090）：

| | 本方案 | ESAM `OnlineMerge` |
|---|---|---|
| 记忆结构 | 稀疏 (体素, query) 表：29,471 条记录（44 个 query，45,323 个场景体素），每条 20 B ≈ 0.59 MB；每 query 一个 128 维特征 | `cur_masks` (K, 20000·F) 布尔：44 × 20000 × 60 = 52.8 M 位 ≈ 6.6 MB，**随帧数线性增长** |
| 每帧 update | 7.6 ms（匹配 + 入库 + 维护，全矩阵运算，匈牙利在 ≤ 20×200 上） | 同量级（匈牙利 + cat） |
| 整条流水线 | 98.8 ms/帧（FastSAM 55 + 提升 13 + 特征 6 + 图 8 + 模型 2 + 聚类 2 + 提议打分 4 + 记忆 8） | — |
| 可训练参数 | 投影头 395 k（InfoNCE，50 min）；主干 FastSAM 冻结 | 3D U-Net + 解码器 + MergeHead 全训练 |

计划里的验收目标（规则 ≥ 0.40、学习匹配 ≥ 0.48，均指未平滑）**没有达到**：实际 0.18 / 0.22（平滑后 0.27 / 0.37）。
§6 分析差距主要来自逐帧提议质量（GT 提议喂同一记忆可到 0.513）。记忆大小（< 1 M 记录）、update
< 10 ms、整条流水线 ≤ 115 ms/帧 三条都达标。

## 1. 设计

### 1.1 数据流（每帧）

```
GroupMerger (merging/infer.py, 已有)
   帧的最终 mask：masks (P,N) + 打分器分数 (P) + 未中心化池化特征 feat_raw (P,640)
         │   嵌套 NMS: IoU 0.5, 包含 0.8 (比逐帧评估用的 1.01 严; 见 §5.3), 取 top-20
         ▼
QueryMemory.update(FrameInput)               streaming/memory.py
   1. 点 → 5 cm 体素 id（三维整数打包成一个 int64）；mask → 去重的 (owner, voxel) 对
   2. (P × K) 交集矩阵：torch.unique 把两边体素映射到公共索引，再 (V,P)^T @ (V,K)
   3. 配对分 S = max(包含度_p∈k, 包含度_k∈p) × cos(投影特征)            [contain_x_feat]
      只在体素有接触 (IoU > 0) 的配对上非零；记忆侧体素先膨胀 1 格容忍视角偏移
   4. 匈牙利指派 (scipy, P ≤ 20, K ≤ 200)，S ≥ 0.3 才接受
   5. 匹配上：体素并集（每条记录累计票数）、特征按观测数加权运行平均（同 ESAM）、分数取 max
      没匹配上：入库门槛 → 新 query
   6. 维护：体素 IoU > 0.5 的 query 合并；n_obs = 1 且 30 帧未见的淘汰；K 上限 200（ESAM inscat_topk_insts）
         │
         ▼
QueryMemory.render(mesh_xyz)   顶点量化到同一体素格，searchsorted 查表 → (K,V) 掩码
evaluate.segment_smooth        ESAM 的 mesh-segment 平滑（每段多数票 > 0.5）
```

### 1.2 存储

| 字段 | 形状 | 说明 |
|---|---|---|
| `owner, voxel, votes` | (M,) int64 / int64 / int32 | 整张记忆是一张稀疏 (体素, query) 表，按 (owner, voxel) 去重；votes = 多少帧把该体素投给该 query |
| `feat` | (K, 128) | 投影特征的观测数加权运行平均（比较时重新 L2 归一化） |
| `score, n_obs, first_seen, last_seen` | (K,) | 提议分数 max、观测帧数、首/末次观测 |
| `scene_voxels` | (V_scene,) int64 | 所有观测过的体素，用于入库门槛 |

质心、包围盒从体素集按需算（`index_add_` / `scatter_reduce`）。不存点：渲染时把目标点集量化到同一体素格查表。
`keep_points=True` 可切到 ESAM 式的"存所有点 + 最近点渲染"（`render_knn`，体素桶候选，不建 KD-tree），
实测与体素渲染 AP50 相同（0.336 vs 0.343，用 ESAM 的逐帧 mask），所以默认不存点。

设计决定与证据：
- **5 cm 体素**：同物体相邻帧体素 IoU 0.34、异物体 0.003；3 cm / 8 cm 的 AP50 都更低（§5.1）。
  GT 实例按 5 cm 体素化再渲染回网格的上限 AP50 = 0.788（计划里假设 ≥ 0.9，没达到；3 cm 因采样稀疏更差）。
- **入库门槛**：提议分数 ≥ 0.3、体素数 ≥ 20、≥ 80% 体素落在已观测场景体素 3 格邻域内（排除穿窗/深度离群点）。
- **匹配分用包含度而非 IoU**：新视角往往只看到物体的一部分，IoU 天然小；max(两向包含度) 在真 GT 提议上 0.537 → 0.574。
- **渲染不做独占、不设票数门槛**：两者都降 AP（§5.4）。

### 1.3 跨帧特征：投影头（对标 ESAM MergeHead）

ESAM 的匹配分 = 训练过的 `MergeHead` 特征余弦 × bbox IoU。我们的 FastSAM 主干特征跨帧很弱
（帧内中心化后同物体相邻帧 cos 0.20，AUC 0.72），所以第一版记忆只用几何。
实测几何记忆的天花板：把 **ESAM 自己的逐帧 mask** 喂进来只有 0.336（未平滑），而 ESAM 自己的合并
用 oracle 特征能到 0.48–0.51 —— 跨帧外观特征值 ~0.2 AP50，是缺的那一块。

`streaming/projection.py::ProjectionHead`：Linear 640→512, LayerNorm, ReLU, Linear→128, L2（395 k 参数），
输入是**未中心化**的逐提议池化主干特征（`feat_raw`：mask→组权重 @ 组特征，再 L2）。
损失 = 多正样本 InfoNCE（τ = 0.07，`multi_positive_infonce`），锚点与正样本是同场景相邻 1–3 个缓存帧中
同一 GT 实例的提议（IoU ≥ 0.5 的提议才参与），负样本是配对帧里的其它所有提议（包括同类邻居）。

| 留出 60 场景配对 AUC | 隔 1 帧 | 隔 4 帧 |
|---|---|---|
| 原始特征 cos | 0.896 | 0.805 |
| 投影 cos（40 epoch，1141 场景，50 min） | **0.952** | **0.891** |

注意原始特征在"好提议"（IoU ≥ 0.5）上的 AUC 0.896 远高于早先在原始分组上测的 0.72：
物体级 mask 池化把部件级噪声平均掉了。AUC 在第 10 epoch 后不再上升（0.955 → 0.952），
训练损失继续降到 0.01，有轻微过拟合，但下游 AP 用的就是最终模型。

### 1.4 训练数据：稠密流式缓存

现有 `merge_cache/train` 每场景 8 帧、间隔 200 帧，相邻缓存帧同物体体素 IoU 只有 0.017，不能学跨帧。
`tools/stream_build_cache.py` 从 ESAM 2D 导出建 `work_dirs/merge_cache/stream/train/<scene>/<frame>.npz`：
1205 个训练场景 × 12 连续帧（40 帧间隔）= 14,182 帧，3.7 GB，0.4 s/帧 分 3 卡。每帧存提议
（packed masks、20 个标量、中心化/未中心化特征、与 GT 的 IoU 和最佳 GT id）、位姿、GT、分组。
同场景内 GT 实例 id 跨帧一致（已验证 100%）。

## 2. 与 ESAM OnlineMerge 的逐项对照

| | ESAM `OnlineMerge` | 本项目 `QueryMemory` |
|---|---|---|
| 记忆内容 | 所有帧的 20k 点上的布尔 mask (K, 20000·F) | (体素, query) 稀疏表，随场景体积增长 |
| 匹配分 | 训练特征 cos × bbox IoU，类别一致 | max 包含度 × 投影特征 cos，体素接触门控 |
| 指派 | 匈牙利，> 0 接受 | 匈牙利，≥ 0.3 接受 |
| 融合 | mask 沿点维拼接；query 计数加权均值 | 体素并集 + 票数；特征计数加权均值；分数 max |
| 新实例 | 未匹配即追加 | 入库门槛（分数、大小、空间支持） |
| 维护 | 只按分数截 top-200 | 重叠合并 + 淘汰 + top-200 |
| 评估映射 | 所有点 KNN 到网格顶点（pointops）+ segment 平滑 | 顶点量化查表 + 同样的 segment 平滑（可选 `render_knn`） |
| 训练 | 3D U-Net（含时序记忆）+ 解码器 + MergeHead | 只有投影头（395 k）与 mask 打分器（28 k）、GroupMergeNet（40 k） |

与 ESAM 的差异需要写在 README 的「与 ESAM 的差异」里：体素记忆、包含度门控、入库门槛、维护合并。

## 3. 实验设置

- 验证：16 个场景、809 帧（40 帧间隔），提议来自 `work_dirs/merge_cache/proposals/val`
  （不重跑 FastSAM：原始 `.sens` 已删除，缓存已备份到 `/media/.../TrainingFreeSAM3D_backup/`）。
  打分器 `work_dirs/mask_scorer/scorer`。网格、GT、超点来自 `work_dirs/esam_eval/data/scannet200/`。
- 每帧耗时在 `stream_eval` 里只计 `update()`（含同步）；整条流水线用 `merge_benchmark.py --memory`。
- 命令：

```bash
source tools/env.sh
python tools/stream_build_cache.py --split train --frames-per-scene 12 --shard 0/3   # 三卡各一份
python tools/stream_train_projection.py --tag proj --epochs 40 --max-gap 3
python tools/stream_eval.py --scorer scorer --projection proj --match-score contain_x_feat \
    --match-dilation 1 --nms-contain 0.8 --segment-smooth --tag final_proj_nms08
python tools/merge_benchmark.py --scorer scorer --memory proj --frames 60
```

## 4. 匹配打分器（MatchScorer）为什么没用上

计划第 7 步训练了 12 个配对线索（体素 IoU、两向包含度、质心距、bbox IoU、大小比、两种特征 cos、
提议分、query 分、观测数、未见帧数）的 MLP（`streaming/matcher.py`，1.5 k 参数，310,715 个配对样本，
留出 AUC 0.863 vs 规则 0.817）。但喂进记忆后 AP50 0.140，低于规则 0.157：
配对 AUC 高不等于指派好——它给"部件 vs 整体"配对的概率也高，匈牙利后把部件挂到了错误的 query 上，
而且它学的是**原始**特征的 cos。投影头出来以后，几何 × 投影 cos 的两项乘积已经比它好，
所以默认不用；代码保留作消融（`--matcher matcher`）。

## 5. 消融（AP50，未平滑，除非注明；全部 16 场景）

### 5.1 记忆参数（纯几何规则，直接喂 top-20 提议、不做逐帧 NMS）

| 体素 | 膨胀 | 入库分数 | AP50 | AP25 | queries | 记录数 | ms |
|---|---|---|---|---|---|---|---|
| 3 cm | 1 | 0.3 | 0.120 | 0.316 | 68 | 145 k | 11.0 |
| **5 cm** | 0 | 0.3 | 0.162 | 0.393 | 75 | 56 k | 8.6 |
| 5 cm | 1 | 0.3 | 0.157 | 0.357 | 62 | 65 k | 9.4 |
| 5 cm | 2 | 0.3 | 0.112 | 0.301 | 57 | 72 k | 11.6 |
| 8 cm | 1 | 0.3 | 0.116 | 0.322 | 54 | 30 k | 7.9 |
| 5 cm | 1 | 0.0 | 0.143 | 0.361 | 74 | 78 k | 9.0 |
| 5 cm | 1 | 0.5 | 0.144 | 0.351 | 45 | 47 k | 8.3 |

另：匹配分用 IoU 0.185（最早的规则版，膨胀 0）、many-to-one 指派 0.121、
膨胀包含度做维护合并 0.31 → 严重欠分割。

### 5.2 投影特征进匹配分（5 cm，无逐帧 NMS）

| 匹配分 | AP50 | AP25 | IoU | R@0.5 | queries |
|---|---|---|---|---|---|
| max 包含度（规则） | 0.157 | 0.357 | 0.370 | 0.348 | 62 |
| 包含度 × cos，膨胀 0 | 0.169 | 0.395 | 0.405 | 0.400 | 100 |
| 包含度 × cos，膨胀 1 | 0.182 | 0.419 | 0.405 | 0.405 | 84 |
| 包含度 × cos²，膨胀 0 | 0.142 | 0.373 | 0.395 | 0.368 | 111 |
| 包含度 × cos，阈值 0.4 | 0.147 | 0.379 | 0.397 | 0.370 | 108 |
| 包含度 × cos，膨胀 1，**query 特征比较前重新归一化** | 0.193 | 0.426 | 0.404 | 0.410 | 81 |

运行平均的单位向量模长 < 1，不归一化会低估 cos（ESAM 也有同样的问题），归一化 +0.01。

### 5.3 进记忆前的逐帧嵌套 NMS（投影 + 膨胀 1）

| NMS 包含阈值 | AP50 | AP25 | IoU | R@0.5 | queries |
|---|---|---|---|---|---|
| 无 NMS（top-20 直接进） | 0.193 | 0.426 | 0.404 | 0.410 | 81 |
| 1.01（只去 IoU > 0.5 重复，逐帧评估的设置） | 0.176 | 0.420 | 0.368 | 0.348 | 61 |
| **0.8** | 0.220 | 0.474 | 0.350 | 0.322 | 37 |
| 0.65 | 0.226 | 0.474 | 0.351 | 0.331 | 36 |

逐帧 AP 最好的设置（部件与整体同时输出）对记忆是坏的：每个部件各开一个 query，
匈牙利 1:1 后同一物体被拆成多个 query（早先的精确标签检查：75 个 query 全对、0 个错、26 个重复），
query 数 81 → 37 才接近 ESAM 的 40。代价是召回下降（部件被压掉后，物体整体被漏分的帧就没了）。

### 5.4 维护合并与渲染（投影 + NMS 0.8）

| 变体 | AP50 | AP25 | IoU | R@0.5 | queries |
|---|---|---|---|---|---|
| 基线 | 0.220 | 0.474 | 0.350 | 0.322 | 37 |
| 接触 + cos ≥ 0.8 就合并（膨胀 1） | 0.194 | 0.403 | 0.299 | 0.275 | 25 |
| 接触 + cos ≥ 0.9 就合并 | 0.212 | 0.438 | 0.325 | 0.294 | 30 |
| 渲染只留票数 ≥ 30% 观测数的体素 | 0.144 | 0.389 | 0.299 | 0.262 | 37 |
| 匹配阈值 0.2 | 0.208 | 0.459 | 0.346 | 0.317 | 36 |
| ESAM 原式 cos × bbox IoU（bbox 接触门控） | 0.170 | 0.369 | 0.291 | 0.276 | 34 |

"接触且外观相似就合并"在 cos 0.8/0.9 都过合并（相邻同类物体的投影 cos 常 > 0.9，AUC 0.95 不等于
有干净的阈值），所以维护合并保持几何 IoU > 0.5。票数门槛说明问题不是"渗出"而是"覆盖不够"。
ESAM 的 bbox 门控在我们这种有重叠提议的输入上太松。

### 5.5 ESAM mesh-segment 平滑（§0 表）

规则 0.180 → 0.272，本方案 0.220 → 0.365，GT 提议 0.356 → 0.513。平滑把 mask 边界吸附到网格的
过分割段上，补上 5 cm 体素化与采样稀疏造成的孔洞，对召回帮助最大（0.322 → 0.498）。

## 6. 差距分析与下一步

1. **逐帧提议质量是主因**：同一记忆、GT 提议 0.513 vs 我们的提议 0.365；ESAM 自己的逐帧 mask
   喂我们的记忆 0.336（未平滑，vs 我们 0.220）。ESAM 的逐帧 mask 来自带时序记忆的 3D U-Net，
   每帧都被之前的帧细化过；我们的逐帧 mask 互不相关。不训练 3D 网络的前提下，可做的是把记忆
   反馈给逐帧阶段（例如用上一帧 query 的体素集作为提议打分器/NMS 的先验），这是下一步最值得做的。
2. **跨帧特征还不够干净**：投影 cos 对相邻同类物体仍常 > 0.9，所以只能做乘法门控，不能单独决定合并。
   更难的负样本（只取相邻物体）、或把几何一起喂进投影头，可能把 AUC 推到能设硬阈值的程度。
3. **第一版不拆 query**：ESAM 也不拆；但我们 22% 的 GT 需要拆（见 merge_net 文档），记忆里一旦并错就永久错。
4. 体素化上限 0.788（5 cm）本身就低于 ESAM 的 0.589 不远；更细的体素需要更密的采样（每帧 > 20k 点）。

## 7. 测试

`tests/test_streaming.py`（7 个，CPU 可跑）：体素哈希可逆与负坐标；交并矩阵 = Python 集合参考；
GPU = CPU；合成 3 物体序列恢复 3 个 query、幂等、记录数以场景体素为界；入库门槛拒绝漂浮点；
特征门控的接触合并（同外观并、异外观不并）；segment 平滑 = 每段多数票。
