# 方案 B 训练链路说明

本文对应当前已经落地的代码改造，目标是把 Cellpose-SAM 的导出结果整理成适合 `UNet + MobileNetV3` 训练的双任务数据集，并让网络输出：

- `channel0`：细胞前景 logits
- `channel1`：距离图回归结果

## 1. 已新增或改造的脚本

### Cellpose 侧

- `E:\cellpose\cellpose\tools\prepare_unet_distance_dataset.py`

作用：

- 读取 `Cellpose-SAM` 已导出的 `images / semantic_masks / instance_masks`
- 生成训练工程所需的目录结构：
  - `images/`
  - `masks/`
  - `distance_maps/`
  - `debug/`
- 其中：
  - `masks` 直接使用 `semantic_masks`
  - `distance_maps` 由 `instance_masks` 逐实例生成

### U-Net 工程侧

目录：

- `E:\Tianlu\cell\deeplearing_algorithm\third_process_segment`

已改造文件：

- `config.py`
- `baseline.py`
- `dataset.py`
- `loss.py`
- `train.py`
- `inference_batch.py`
- `export_to_onnx.py`
- `output_model.py`
- `segmentation_utils.py`

## 2. 当前数据格式约定

训练工程默认读取：

```text
data/
  train/
    images/
    masks/
    distance_maps/
  val/
    images/
    masks/
    distance_maps/
  test/
    images/
    masks/
    distance_maps/
```

其中：

- `images/`：原图，允许灰度图，训练时会自动复制成 3 通道
- `masks/`：前景二值标签，前景=255，背景=0
- `distance_maps/`：单通道浮点 tif，范围 `[0, 1]`

## 3. 距离图是怎么生成的

当前采用的是“逐实例距离图”方案，而不是把整个 `semantic_mask` 当成一整团去做距离变换。

处理规则：

1. 从 `instance_masks` 中取出每个实例 ID
2. 对每个单独细胞做一次欧氏距离变换
3. 在该细胞内部把最大距离归一化到 `1`
4. 写回整张图，背景始终保持 `0`

这样做的好处是：

- 相邻细胞会各自形成独立亮峰
- 不容易退化成一整团细胞只剩一个中心峰
- 更适合后续做局部峰值检测或分水岭

## 4. 模型输出与损失

当前训练任务固定为 2 通道单头输出：

- `channel0`：前景 logits
- `channel1`：距离图回归

损失定义：

- 前景通道：`BCEWithLogits + Dice`
- 距离图通道：仅在前景区域内计算 `SmoothL1`
- 总损失：

```text
total_loss = seg_loss + distance_weight * dist_loss
```

默认 `distance_weight = 1.0`

## 5. 推理输出

`inference_batch.py` 当前会保存：

- `foreground_prob/*.tif`
- `distance_pred/*.tif`
- `foreground_binary/*.png`
- `overlays/*.png`
- `preview_panels/*.png`

这意味着本轮虽然还没有实现分水岭和实例恢复，但已经把后续需要的两个核心输入稳定保存下来了：

- 前景概率图
- 距离图

## 6. ONNX 导出

`export_to_onnx.py` 已改为导出 2 通道模型。

导出后约定保持不变：

- `output[:, 0, :, :]`：前景 logits
- `output[:, 1, :, :]`：距离图预测

这对你后续做 RK3568 / ONNX / RKNN 接口适配会更清楚。

## 7. 当前默认尺寸说明

代码里已经按你当前真实导出的样本尺寸写成：

```text
input_size = (1200, 400)   # (H, W)
```

原因是你现在 `E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\images` 里的样本就是这个方向。

如果你后续换成横图 `400x1200`，只需要改：

- `E:\Tianlu\cell\deeplearing_algorithm\third_process_segment\config.py`
  - `data["input_size"]`

## 8. 本轮没有做的内容

这轮没有继续实现：

- 分水岭后处理
- 局部峰值检测
- RK3568 端 C++ 后处理
- 3 类标签导出器（背景 / 内部 / 边界）
- `outline_masks` 的主任务训练

这些都可以作为下一轮继续补。
