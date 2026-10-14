# TrainingFreeSAM3D — 项目约定（所有 agent / 协作者必须遵守）

本项目是 ESAM（训练版）的 **training-free 平行实现**：只使用冻结的 FastSAM-x 作图像模型；
不训练 3D 骨干网络，只训练 <1M 参数的小头（2D 实例头 / 3D 几何头 / 图合并模型）。
每个模块都对标 ESAM 中的一个流程模块，先做 baseline，再迭代。

`agents.md` 是指向本文件的软链接，只维护 `AGENTS.md` 一份。

## 1. 目录结构（不要乱放代码）

```
TrainingFreeSAM3D/
├── AGENTS.md               # 本文件：项目规则（唯一的规则来源）
├── README.md               # 使用说明、当前进度、实验结论
├── configs/                # YAML 实验配置；路径与超参只写在这里
├── docs/                   # 设计/调研笔记（如 FastSAM 特征层分析），不放代码
├── freesam3d/              # 唯一的库代码包，按“职责”分子包
│   ├── config.py           #   读取 YAML + 命令行覆盖
│   ├── io/                 #   数据读写：RGB-D 帧、ScanNet GT（不做算法）
│   ├── models/             #   冻结模型封装：FastSAM 分割 + 特征 hook；实例特征头
│   │                       #     (instance_head = 2D 头, instance_head3d = 3D 几何头 + 描述子)
│   ├── lifting/            #   ESAM「Query Lifting」对标模块
│   │   ├── geometry.py     #     深度反投影、世界坐标、20k 采样
│   │   ├── grouping.py     #     重叠 mask → 互斥分组、未覆盖点补组
│   │   ├── pooling.py      #     逐点特征 → 分组(query)特征
│   │   └── query_lifting.py#     本阶段的编排（只编排，不写具体算法）
│   ├── analysis/           #   特征层面的分析（相似度、AUC、purity）
│   ├── evaluation/         #   实例层面的评估（IoU 匹配、class-agnostic AP、粒度诊断）
│   ├── esam/               #   调用官方 ESAM 的桥接层：数据准备、导出预测
│   │                       #     （只 import ESAM，不修改 ESAM 仓库的代码和数据）
│   ├── merging/            #   GroupMergeNet：组邻接图(graph=CPU 参考, graph_gpu=矩阵版)、标签(labels)、
│   │                       #     缓存(cache)、数据(data)、模型(model)、训练(train)、聚类(cluster)、
│   │                       #     评估(evaluate)、探针(probe)、单次推理入口(infer)、
│   │                       #     提议候选+嵌套 NMS(proposals=CPU 参考, proposals_gpu=矩阵版)、
│   │                       #     mask 质量打分器(scorer, proposal_data)
│   ├── streaming/          #   流式场景级 Query Memory（对标 ESAM OnlineMerge）：体素哈希与稀疏交并(voxel)、
│   │                       #     QueryMemory(memory)、跨帧特征投影头(projection)、匹配打分器(matcher)、
│   │                       #     场景级评估与 mesh-segment 平滑(evaluate)、窗口记忆与统一帧内/跨帧图
│   │                       #     (window_memory, pair_features, window_scores)、两步式分组(two_step)、
│   │                       #     ESAM 规则重写(esam_merge)、输出端后处理(postprocess)、
│   │                       #     混合碎片去向(fragment_fate)、留出场景评估与探针(holdout, window_probe)
│   │                       #     阶段 2 统一分组(unified_flow：帧内 / block<->object / object<->object
│   │                       #     三类边共用一个模型 + 节点质量头)、训练图采集(window_data)、
│   │                       #     统一模型训练(unified_train)、留出阈值扫描(unified_sweep)、
│   │                       #     未分配 block 归因(unified_probe)、阶段 4 探针(phase4_probe)
│   └── vis/                #   所有画图/导出 PLY 的代码（含阶段 4 训练曲线）
├── tools/                  # 命令行入口脚本：只做参数解析 + 调用 freesam3d
├── tests/                  # pytest 单元测试（不依赖 GPU/权重的部分必须可离线跑）
└── work_dirs/              # 所有生成结果（git 忽略）；绝不从这里 import 代码
```

后续模块（对标 ESAM 的 Query Refinement / Query Merging 等）在 `freesam3d/` 下
**新建同级子包**（如 `freesam3d/refinement/`、`freesam3d/merging/`），不要塞进
`lifting/`。

## 2. 放置规则

- 库代码只能在 `freesam3d/`；`tools/` 里的脚本不得定义可复用的算法函数。
- 画图代码只能在 `freesam3d/vis/`；指标计算只能在 `freesam3d/analysis/`。
  算法模块不 import `vis`。
- 新增特征来源：只改 `freesam3d/models/fastsam.py` 里的 `FEATURE_SPECS`，并在
  `docs/fastsam_features.zh-CN.md` 记录其对应关系。
- 新增池化方式（如几何感知池化）：在 `freesam3d/lifting/pooling.py` 中新增函数，
  通过配置 `pooling.method` 选择；保留 `mean_l2` baseline 不删。
- 一次性调试脚本放在会话临时目录，不要提交到仓库根目录。
- **实验发现（可写进论文的结论）统一记录在 `docs/findings.zh-CN.md`**：每完成一个阶段
  就追加一节（发现 + 证据 + 口径 + 出处），旧结论不删除；被推翻的结论就地标注「修正」。
  这是本项目面向论文的唯一发现清单，与 `docs/stream_merge_v2*.md`（设计说明）分工不同。
- 需要跑官方 ESAM 时，通过 `freesam3d/esam/` 调用，数据根目录放在本项目的
  `work_dirs/esam_eval/` 下。不要往 `/home/yqx/SegmentAnything3D/ESAM/data` 里写东西。
- 任何可训练模块：先写 `tools/*_probe.py` 回答「输入是否有信号」再训练；阈值只在留出的
  训练场景上标定，验证集（16 个场景）只用于最终汇报。新的边/节点特征加在
  `merging/graph.py` 的 `edge_feature_names()` / `NODE_GEOM` 中并同步更新测试。
- 推理路径优先用矩阵运算（`one_hot.T @ X`、`cdist`、`scatter_reduce`），不用 KD-tree /
  sklearn / 逐边 Python 循环；每个 GPU 实现必须有与 CPU 参考实现对照的单元测试，
  并用 `tools/merge_benchmark.py` 记录逐阶段耗时。
- 不复制第三方仓库或权重进本项目；外部路径（FastSAM 权重、ScanNet 原始数据、
  ESAM、Personalize-SAM）只在 `configs/*.yaml` 中引用。

## 3. 与 ESAM 对齐的约定

- 点坐标使用 ESAM 的 `points/*.bin` 约定：ScanNet **未对齐**世界坐标
  （`pose_c2w @ K^-1 [u,v,1]·d`），axis-align 只用于 GT 的 bbox 去噪。
- 互斥分组沿用 ESAM：mask 按面积降序依次写入，**小 mask 覆盖大 mask**。
- 未覆盖点：在采样后的点上 KMeans(20)（<20 点则并成一组），然后把编号压缩成
  `0..G-1`。ESAM 原版对 `[xyz, rgb(0-255)]` 聚类，默认保持一致（可配置）。
- 任何偏离 ESAM 的地方必须在 README 的「与 ESAM 的差异」里写明。

## 4. 运行环境

- 环境：`source tools/env.sh`。它激活 `AutoSeg3D/.venv`（torch、ultralytics 8.0.120、
  fastsam、sklearn、scipy、plyfile），并把与内核驱动匹配的 NVIDIA 用户态库
  （`/home/yqx/.cache/openjev/driver-580.173.02`）放到 `LD_LIBRARY_PATH` 最前面。
  不这样做时，CUDA 会报 804 或 driver/library mismatch。默认 `CUDA_VISIBLE_DEVICES=3`。
- `device: auto` 在 CUDA 不可用时退回 CPU，结果与 GPU 一致（差异在浮点误差内）。
- 模型下载优先 ModelScope，其次 Hugging Face。
- 查看大于 1M 像素的图片前先缩小或压缩。

## 5. 代码风格

- 函数尽量是纯函数（numpy/torch 输入输出），随机性一律通过 `seed` 传入。
- 结果文件命名带编号前缀，方便按阅读顺序浏览（如 `01_overview.png`）。
- 新功能附带最小单元测试；`pytest -q tests` 必须保持通过。
