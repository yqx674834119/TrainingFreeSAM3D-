# FastSAM 特征层分析：对照 SAM / Personalize-SAM

## 1. Personalize-SAM 是怎么取特征的

代码：`/home/yqx/SegmentAnything3D/Personalize-SAM`（`persam.py`、`persam_f.py`、
`per_segment_anything/predictor.py`、`per_segment_anything/modeling/image_encoder.py`）。

1. `SamPredictor.set_image(img, mask)`：最长边缩放到 1024，按像素均值/方差归一化，
   **右下角补零**到 1024×1024；参考 mask 做同样的变换。
2. `self.features = self.model.image_encoder(input_image)`。`ImageEncoderViT.forward`
   为：`patch_embed → (+pos_embed) → blocks → neck`，所以 `predictor.features` 是
   **neck 之后**的 `256×64×64`。neck = `Conv1x1(C→256) → LayerNorm2d → Conv3x3 → LayerNorm2d`。
3. **neck 之前**的特征（最后一个 Transformer block 的输出，`B×64×64×C`，ViT-H 中 C=1280）
   PerSAM 完全没有用到。如果要取，需要在 `image_encoder.blocks[-1]` 上挂 forward hook。
4. 目标特征：把 mask 双线性插值到 64×64，取 `mask>0` 位置的特征 → **平均** → **L2 归一化**
   （PerSAM-F 用 `0.5·max + 0.5·mean`）。测试图特征逐位置 L2 归一化后做点积，即 cosine
   相似度，再上采样回原图。

要点：SAM neck 以 LayerNorm 结尾，特征近似零均值，所以 cosine 有区分度。

## 2. FastSAM 有没有对应的层

FastSAM-x 是 **YOLOv8x-seg**，纯 CNN，没有 ViT block，也没有 SAM 的 256 维 neck。
从权重中读出的结构（`model.save = [4, 6, 9, 12, 15, 18, 21]`）：

| 层 | 模块 | 输出（输入 480×640） | 角色 |
|---|---|---|---|
| L0–L8 | Conv/C2f | — | backbone |
| **L9** | SPPF | 640ch, stride 32, 15×20 | **backbone 最后输出 = neck 之前** |
| L10–L14 | Upsample/Concat/C2f | — | PAN-FPN 自顶向下 |
| **L15** | C2f | 320ch, stride 8, 60×80 | **P3：neck 输出，Proto 的输入** |
| L18 | C2f | 640ch, stride 16 | P4 |
| L21 | C2f | 640ch, stride 32 | P5 |
| L22.proto.cv2 | Conv | 320ch, stride 4, 120×160 | mask 解码器内部 |
| L22.proto | Proto | 32ch, stride 4, 120×160 | mask 原型；实例 mask = sigmoid(系数·原型) |

和 SAM 的对应关系（本项目 `FEATURE_SPECS` 的命名）：

| 名称 | 来源 | SAM 中的类比 |
|---|---|---|
| `backbone` | L9 | 最后一个 Transformer block 的输出（neck 之前） |
| `neck_p3` | L15 | neck 输出：送给 mask 解码器（Proto）的特征 |
| `neck_fpn` | concat(L15, ↑L18, ↑L21)，1600ch @ s8 | 多尺度 neck（SAM 没有） |
| `proto_hidden` | Proto.cv2 | mask 解码器中间层 |
| `proto` | Proto 输出 | mask 原型（最接近「逐像素 mask embedding」） |

## 3. 实现细节

- 特征用 forward hook 在 **同一次** `FastSAM.predict` 里截取，所以 letterbox、RGB、/255
  等预处理和分割时完全一致。FastSAM 没有均值/方差归一化。
- 480×640 输入在 `imgsz=640` 下不缩放（gain=1），且各 stride 都能整除，不需要 padding；
  代码仍按 ultralytics 的 letterbox 规则计算 gain/pad，可以处理其它分辨率。
- 逐点特征：把点的像素中心映射到网络输入坐标，在特征图上 `grid_sample` 双线性采样
  （`align_corners=False`，`padding_mode='border'`）。不把 1600 维特征整图上采样到
  480×640，省内存。`padding_mode='zeros'` 会让图像边缘的点与零混合，单元测试覆盖了这一点。

## 4. 与 SAM 的关键差异：cosine 被「压扁」

FastSAM 的特征都在 SiLU 之后，基本为正，并且有很大的公共分量，导致所有 cosine 都偏高
（`neck_p3` 几乎所有 mask 对都在 0.85–1.0）。SAM 的 neck 以 LayerNorm 结尾，没有这个问题。
所以提供了 `pooling.method=center_mean_l2`：先减去本帧所有点的特征均值，再做平均池化和
L2 归一化。实验结果见 README。
