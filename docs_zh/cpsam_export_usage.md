# `tools/run_cpsam_export.py` 使用说明

## 1. 脚本作用

[run_cpsam_export.py](/E:/cellpose/cellpose/tools/run_cpsam_export.py) 用于：

- 批量读取 2D 图像
- 调用 Cellpose-SAM 推理
- 导出实例 ID mask
- 导出语义分割二值 mask
- 导出“相邻细胞不粘连”的语义 mask
- 导出 GUI 同源的轮廓线 mask 与轮廓叠加图
- 导出实例标签彩色预览图，解决“实例 mask 看起来全黑”的问题
- 可选导出 `_seg.npy`
- 可选导出叠加可视化图
- 可选导出原图、flow RGB、gradXY、cellprob 等中间结果
- 可选导出 cellprob 二值候选图与逐阶段后处理结果
- 可选导出多边形轮廓坐标文本
- 可选导出 ImageJ ROI 压缩包

这份脚本默认面向你当前的数据形态：

- 角膜内皮细胞图像
- 2D 灰度图
- 常见尺寸约 `400 x 1200`

---

## 2. 为什么单独写这个脚本

虽然 Cellpose 工程本身已经支持：

- CLI 推理
- GUI 推理
- `_seg.npy`
- PNG / TIF mask
- ROI zip

但它默认更偏“通用工具”风格，不完全符合你当前的维护习惯：

- 你希望直接改代码顶部配置，而不是拼命令行参数
- 你希望中文日志更清晰
- 你希望输出更聚焦于训练自己的 `Unet + MobileNetV3`
- 你当前优先要“实例 ID mask”，而不是一堆通用输出

所以这里单独补了一份更适合你直接改、直接跑、后续自己维护的脚本。

---

## 3. 主要配置项在哪里改

直接打开脚本顶部的“配置区”修改即可：

```python
INPUT_DIR = PROJECT_ROOT / "data" / "cornea_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cpsam_export"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MODEL_NAME = "cpsam"
USE_GPU = True
BATCH_SIZE = 8
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
NITER = 200
MAX_SIZE_FRACTION = 0.4
MIN_SIZE = 15
SAVE_SEG_NPY = True
SAVE_INSTANCE_MASK = True
SAVE_INSTANCE_MASK_PREVIEW = True
SAVE_SEMANTIC_MASK = True
SAVE_SEPARATED_SEMANTIC_MASK = True
SAVE_OVERLAY = True
SAVE_OUTLINE_MASK = True
SAVE_OUTLINE_OVERLAY = True
SAVE_IMAGE = True
SAVE_FLOW_RGB = True
SAVE_GRAD_XY = True
SAVE_CELLPROB = True
SAVE_CELLPROB_BINARY_MASK = True
SAVE_RAW_DYNAMICS_MASK = True
SAVE_RAW_DYNAMICS_SEMANTIC_MASK = True
SAVE_RAW_DYNAMICS_OVERLAY = True
SAVE_AFTER_MAXSIZE_MASK = True
SAVE_AFTER_MAXSIZE_OVERLAY = True
SAVE_AFTER_FLOWQC_MASK = True
SAVE_AFTER_FLOWQC_OVERLAY = True
SAVE_DEBUG_SUMMARY = True
SAVE_POLYGON_TXT = True
POLYGON_APPROX_EPSILON_RATIO = 0.005
SAVE_ROIS = False
```

你最常改的通常就是下面几个：

### `INPUT_DIR`

- 输入图像目录
- 改成你自己的原始图像文件夹

### `OUTPUT_DIR`

- 输出根目录
- 建议单独放一个结果目录，避免和原图混在一起

### `USE_GPU`

- `True`：优先用 GPU
- `False`：强制用 CPU

### `FLOW_THRESHOLD`

- 越低一般会检出更多目标
- 越高一般越保守

### `CELLPROB_THRESHOLD`

- 越低越容易保留更多细胞
- 越高会更严格

### `MIN_SIZE`

- 用于滤掉太小的噪点实例

### `NITER`

- 动力学聚合迭代次数
- 越大一般越容易把像素聚到稳定中心

### `MAX_SIZE_FRACTION`

- 单个实例允许占整图的最大比例
- 如果你怀疑“大细胞团块”被过滤太多，可以先临时调大到 `1.0` 做对比

### `SAVE_INSTANCE_MASK_PREVIEW`

- 额外导出彩色实例预览图
- 主要是解决“实例 mask 明明有值，但打开看像全黑”的问题

### `SAVE_SEPARATED_SEMANTIC_MASK`

- 是否导出“细胞内部不粘连”的训练标签
- 如果你希望相邻细胞之间保留黑色边界线，这个开关建议保持 `True`

### `SAVE_OUTLINE_MASK`

- 是否导出二值轮廓图
- 这一份和 GUI 的 `outlines on` 属于同一个来源，都是由最终实例 mask 派生出来的

### `SAVE_OUTLINE_OVERLAY`

- 是否把轮廓线直接画到原图上
- 比普通 overlay 更适合肉眼检查边界是不是贴合

### `SAVE_POLYGON_TXT`

- 是否导出多边形轮廓顶点文本
- 适合你后续做自定义标签、几何分析或转成别的格式

### `POLYGON_APPROX_EPSILON_RATIO`

- 控制轮廓压缩成多边形时的简化强度
- 值越小，越贴近原始边界
- 值越大，顶点越少、边界越平滑

### `SAVE_DEBUG_SUMMARY`

- 是否导出逐阶段调试报告
- 建议保持 `True`
- 这会直接告诉你：实例数到底是在哪一步下降的

---

## 4. 输入是什么

输入是一个目录中的 2D 图像文件，支持：

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.bmp`

当前脚本默认：

- 不递归子目录
- 只处理当前目录下的一层文件
- 只支持 2D 图像

如果误传了 3D 栈或非常规维度图像，脚本会直接给出中文报错，避免默默跑错。

---

## 5. 输出是什么

脚本会在 `OUTPUT_DIR` 下创建以下子目录：

### `seg_npy`

保存 GUI 兼容的：

```text
xxx_seg.npy
```

适合后续：

- 回到 GUI 继续修标签
- 保留完整推理中间结果

### `instance_masks`

保存实例 ID mask：

```text
xxx_instance_mask.png
```

或当实例数非常多时保存为：

```text
xxx_instance_mask.tif
```

说明：

- 背景是 `0`
- 第 1 个实例是 `1`
- 第 2 个实例是 `2`
- 以此类推

这是你后续训练轻量网络时最核心的标签文件。

### `instance_mask_previews`

保存实例标签彩色预览图：

```text
xxx_instance_mask_preview.png
```

说明：

- 这不是训练真值，而是便于人眼查看的预览图
- 用来解决原始实例标签图打开像“全黑”的问题

### `semantic_masks`

保存语义分割二值 mask：

```text
xxx_semantic_mask.png
```

说明：

- 背景为 `0`
- 细胞前景为 `255`

如果你后续训练的是普通二分类语义分割网络，这一份通常是最直接的监督标签。

### `separated_semantic_masks`

保存“细胞内部不粘连”的语义 mask：

```text
xxx_separated_semantic_mask.png
```

说明：

- 细胞内部为 `255`
- 轮廓线和背景都为 `0`
- 相邻细胞之间会保留细黑边，不会像普通二值前景那样整片粘连

如果你后续想训练“细胞内部”语义分割，这一份通常比普通 `semantic_masks` 更适合。

### `overlays`

保存叠加可视化图：

```text
xxx_overlay.png
```

适合快速人工抽查推理质量。

### `outline_masks`

保存二值轮廓线 mask：

```text
xxx_outline_mask.png
```

说明：

- 轮廓像素为 `255`
- 其余为 `0`
- 这份轮廓来源于 Cellpose 最终实例标签，不是从 `flow_rgb` 直接阈值化得到

### `outline_overlays`

保存轮廓线叠加图：

```text
xxx_outline_overlay.png
```

说明：

- 会把最终轮廓线直接画到原图上
- 比普通 overlay 更适合看相邻细胞边界是否紧密贴合

### `images`

保存原图副本：

```text
xxx_image.tif
```

便于和后处理结果逐张对照。

### `flow_rgb`

保存 Cellpose 的 flow RGB 可视化图：

```text
xxx_flow_rgb.png
```

这份图更适合肉眼查看流场方向是否稳定。

### `grad_xy`

保存原始二维流场：

```text
xxx_gradXY.tif
```

说明：

- 这是 `flows[1]`
- 一般形状为 `[2, H, W]`
- 两个通道分别对应 `dY` 和 `dX`

### `cellprob`

保存 cell probability：

```text
xxx_cellprob.tif
xxx_cellprob_preview.png
```

说明：

- `*.tif` 是原始浮点值
- `*_preview.png` 是为了方便肉眼查看做的归一化预览图
- 当 `instance_masks` 看起来“不出结果”时，优先检查这里最有帮助

### `cellprob_binary`

保存通过 `cellprob_threshold` 后的二值候选图：

```text
xxx_cellprob_binary_mask.png
```

说明：

- 这一步还不是最终实例
- 它只表示哪些像素进入了后续 dynamics 聚合

### `raw_dynamics_instance_masks`

保存“只做 dynamics 聚合，不做 max_size / flow_qc / min_size 过滤”的实例标签：

```text
xxx_instance_mask.png
```

这一步最接近你在 `flow_rgb` 中肉眼看到的“大量候选细胞”。

### `raw_dynamics_semantic_masks`

保存 raw dynamics 阶段的语义 mask：

```text
xxx_semantic_mask.png
```

如果你想尝试“尽量保留更多候选细胞”的语义标签，这一份值得优先对照。

### `raw_dynamics_overlays`

保存 raw dynamics 阶段的叠加图，方便和最终 overlay 直接比较。

### `after_maxsize_instance_masks` / `after_maxsize_overlays`

保存做完 `max_size_fraction` 过滤后的中间结果。

### `after_flowqc_instance_masks` / `after_flowqc_overlays`

保存做完 `flow_threshold` 质量过滤后的中间结果。

### `debug_reports`

保存逐阶段调试报告：

```text
xxx_debug_summary.txt
```

报告里会直接写出：

- 通过 `cellprob_threshold` 的像素数
- 通过 `cellprob_threshold` 的连通域数
- `raw_dynamics` 实例数
- `after_maxsize` 实例数
- `after_flowqc` 实例数
- `final` 实例数

### `polygon_txt`

保存轮廓多边形文本：

```text
xxx_polygons.txt
```

说明：

- `contour_xy` 是 GUI 同源的原始轮廓像素点
- `polygon_xy` 是用多边形近似后的顶点
- 坐标顺序统一为 `x,y`

### `rois`

当 `SAVE_ROIS = True` 时，保存：

```text
xxx_rois.zip
```

适合导入 Fiji / ImageJ。

---

## 6. 运行方式

### 6.1 先进入环境

```powershell
conda activate cellpose
```

### 6.2 运行脚本

```powershell
python E:\cellpose\cellpose\tools\run_cpsam_export.py
```

脚本启动后会先打印当前配置，然后开始逐张图像处理，并输出中文日志，例如：

- 读取图像
- 输入形状
- 推理完成
- 检测到多少个实例
- 保存到哪个路径
- 哪张图失败
- 最终成功/失败统计

---

## 7. 这个脚本如何处理灰度图

你的图像是 2D 灰度图，而 Cellpose 2D 推理内部通常使用 `H x W x 3` 的形式。

脚本里没有手写一套“灰度转三通道”的野路子，而是直接复用了 Cellpose 官方的转换逻辑：

- 先读取原图
- 再用 `cellpose.transforms.convert_image(..., do_3D=False)` 统一整理成 2D 推理输入

这样做的好处是：

- 和 Cellpose GUI / CLI 的输入预处理更一致
- 后续你调阈值时更容易对齐官方行为

---

## 8. Cellpose-SAM 的后处理是怎么走的

你问到的这一点很关键，当前脚本里其实已经有这个阈值。

Cellpose-SAM 不是网络一输出就直接变成 mask，而是大致经过下面这条链路：

1. 网络输出流场 `gradXY`
2. 网络输出 `cellprob`
3. 先用 `cellprob_threshold` 筛出哪些像素有资格参与后处理
4. 再根据 `gradXY` 做动力学积分
5. 再结合 `flow_threshold`、`max_size_fraction`、`min_size` 等条件得到最终实例 mask

所以：

- `CELLPROB_THRESHOLD` 在当前脚本里是有的
- `FLOW_THRESHOLD` 在当前脚本里也是有的
- `NITER` 在当前脚本里也是显式可改的
- `MAX_SIZE_FRACTION` 在当前脚本里也是显式可改的
- 这两个值都会直接传进 `model.eval(...)`
- 除了这两个阈值，Cellpose 还会做“大目标过滤”“流场一致性过滤”“小目标过滤”等额外步骤

如果你发现“为什么最后没出实例”，最应该优先检查的不是最终 mask，而是：

1. `cellprob`
2. `gradXY`
3. 当前阈值设置
4. 最终 `outline_masks` 和 `outline_overlays`
5. `debug_reports`

---

## 9. 对你当前训练流程的建议

你现在的目标链路是：

1. 用 Cellpose-SAM 获得较强的伪标签或初始标签
2. 再训练你自己的 `Unet + MobileNetV3`
3. 最后部署到 `rk3568`

在这个链路里，当前脚本输出的优先级建议是：

1. `separated_semantic_masks`
2. `seg_npy`
3. `semantic_masks`
4. `raw_dynamics_semantic_masks`
5. `instance_masks`
6. `outline_masks`
7. `overlays`

原因如下：

- `separated_semantic_masks` 更适合做“细胞内部”语义分割，邻接细胞不会直接粘成整片
- `seg_npy` 可以回到 GUI 继续修标签
- `semantic_masks` 仍然适合普通二分类前景/背景训练
- `raw_dynamics_semantic_masks` 更接近 flow_rgb 中肉眼看到的候选细胞数量，但误检风险更高
- `instance_masks` 保留了更完整的实例信息，后续仍可派生边界图、前景图、距离图
- `outline_masks` 适合做边界监督或人工核查
- `overlays` 便于快速发现离谱样本

---

## 10. 为什么这次不导出 JSON

这次默认不导出 JSON，原因不是做不到，而是当前阶段没必要把流程搞复杂。

你已经确认本轮默认策略是：

- 主输出：实例 ID mask
- 不额外导出 JSON

这样做的好处：

- 输出最直接
- 后续更容易喂给你现有训练代码
- 避免额外引入 LabelMe / 自定义轮廓格式转换误差

如果后续你确实需要：

- LabelMe JSON
- 自定义 polygon JSON
- 自定义 area / bbox / contour 点列表

再作为第二阶段扩展会更合适。

---

## 11. 运行后会生成哪些文件

以输入文件：

```text
sample_001.png
```

为例，默认会生成：

```text
OUTPUT_DIR/seg_npy/sample_001_seg.npy
OUTPUT_DIR/instance_masks/sample_001_instance_mask.png
OUTPUT_DIR/instance_mask_previews/sample_001_instance_mask_preview.png
OUTPUT_DIR/semantic_masks/sample_001_semantic_mask.png
OUTPUT_DIR/separated_semantic_masks/sample_001_separated_semantic_mask.png
OUTPUT_DIR/overlays/sample_001_overlay.png
OUTPUT_DIR/outline_masks/sample_001_outline_mask.png
OUTPUT_DIR/outline_overlays/sample_001_outline_overlay.png
OUTPUT_DIR/images/sample_001_image.tif
OUTPUT_DIR/flow_rgb/sample_001_flow_rgb.png
OUTPUT_DIR/grad_xy/sample_001_gradXY.tif
OUTPUT_DIR/cellprob/sample_001_cellprob.tif
OUTPUT_DIR/cellprob/sample_001_cellprob_preview.png
OUTPUT_DIR/cellprob_binary/sample_001_cellprob_binary_mask.png
OUTPUT_DIR/raw_dynamics_instance_masks/sample_001_instance_mask.png
OUTPUT_DIR/raw_dynamics_semantic_masks/sample_001_semantic_mask.png
OUTPUT_DIR/raw_dynamics_overlays/sample_001_overlay.png
OUTPUT_DIR/after_maxsize_instance_masks/sample_001_instance_mask.png
OUTPUT_DIR/after_maxsize_overlays/sample_001_overlay.png
OUTPUT_DIR/after_flowqc_instance_masks/sample_001_instance_mask.png
OUTPUT_DIR/after_flowqc_overlays/sample_001_overlay.png
OUTPUT_DIR/debug_reports/sample_001_debug_summary.txt
OUTPUT_DIR/polygon_txt/sample_001_polygons.txt
```

如果开启 `SAVE_ROIS = True`，还会额外生成：

```text
OUTPUT_DIR/rois/sample_001_rois.zip
```

---

## 12. 风险点与注意事项

### 11.1 GPU 不可用时会自动退回 CPU

如果你把 `USE_GPU = True`，但当前环境实际上没有 CUDA 版 `torch`，脚本不会直接崩，而是会打印中文警告并自动退回 CPU。

这很适合你排查环境时使用，但正式批量跑数据前，建议先确认 GPU 已经真的启用。

### 11.2 Cellpose-SAM 模型本身很大

这个仓库里的量化 notebook 显示，Cellpose-SAM 模型体量非常大，不适合直接拿去做嵌入式部署主模型。

所以更合理的路线仍然是：

- 用它生成标签
- 再训练你的轻量网络

### 11.3 当前脚本只做 2D

如果你后面要处理：

- 3D 栈
- 多切片体数据
- 多目录递归批处理

建议另外再写专门脚本，不要在这份 2D 脚本里不断打补丁。

### 12.4 `instance_masks` 看起来像“空白”并不等于真的没有结果

实例 mask 保存的是实例编号，不是普通显示图。

例如：

- 背景是 0
- 第 1 个细胞是 1
- 第 2 个细胞是 2
- 第 190 个细胞是 190

如果它被保存成 `uint16 PNG`，普通图片查看器通常会按 `0~65535` 来显示，
那 `1~190` 这一段会显得非常黑，看起来就像“空白”。

这时应该这样判断：

1. 看文件的像素最大值是不是大于 0
2. 看 `semantic_masks`
3. 看 `overlay`
4. 看 `cellprob_preview`
5. 看 `outline_overlay`

---

## 13. 结论

这份脚本已经满足你当前这轮工作的核心要求：

- 顶部配置区即可运行
- 中文日志
- 输出实例 ID mask
- 输出普通语义 mask
- 输出不粘连语义 mask
- 输出 GUI 同源轮廓线与多边形文本
- 可选 `_seg.npy`
- 可选 overlay
- 可选 ROI
- 面向 2D 灰度角膜内皮细胞图像

如果你下一步要继续扩展，我建议优先顺序是：

1. 先确认你训练到底用 `semantic_masks` 还是 `separated_semantic_masks`
2. 再考虑是否把 `outline_masks` 作为额外监督分支
3. 如果后续有需要，再扩展 JSON 导出
