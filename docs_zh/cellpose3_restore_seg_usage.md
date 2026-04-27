# `tools/run_cellpose3_restore_seg.py` 使用说明

## 1. 这份脚本是做什么的

[run_cellpose3_restore_seg.py](/E:/cellpose/cellpose/tools/run_cellpose3_restore_seg.py) 用于：

- 批量读取 2D 图像
- 调用 `cellpose3` 的恢复 + 分割模型
- 保存恢复后的图像
- 保存实例 ID mask
- 保存语义二值 mask
- 保存 overlay 叠加图
- 保存 `flow RGB`
- 保存 `cellprob` 原图与预览图
- 保存 GUI 可直接打开的 `_seg.npy`

这份脚本更接近你当前的使用习惯：

- 顶部配置区直接改路径和参数
- 中文注释
- 中文日志
- 不依赖命令行参数
- 输出目录清晰，适合你自己后续维护和复用

---

## 2. 为什么要单独写这份 `cellpose3` 脚本

当前这个仓库主线已经偏向 `Cellpose-SAM / v4`。

但 `cellpose3` 的调用入口、模型类型、恢复逻辑都不是完全同一套，尤其是：

- `cellpose3` 侧重点是“图像恢复 + 分割”
- `Cellpose-SAM / v4` 侧重点是 `cpsam`
- 两者模型名、接口习惯、环境依赖都不应该混着理解

所以这份脚本专门做了两件事：

1. 明确按 `cellpose3` 的方式调用 `denoise.CellposeDenoiseModel(...)`
2. 启动时强制检查当前环境是不是 `cellpose 3.x`

这样能尽量避免你出现下面这种坑：

- 代码看起来像能跑
- 实际却导入了当前仓库的 v4 源码
- 最后参数和模型都跑偏了

---

## 3. 最重要的前提：`cellpose3` 要单独环境

这份脚本不是让你直接在当前仓库主线环境里混跑的。

推荐单独建一个环境，例如：

```powershell
conda create -n cellpose3 python=3.10
conda activate cellpose3
python -m pip install "opencv-python-headless>=4.9.0.80"
python -m pip install cellpose==3.1.1.2
```

如果你需要 GPU，还需要在这个环境里安装和你显卡/CUDA 匹配的 `torch` 版本。

脚本启动时会自动检查：

- 当前导入的 `cellpose` 是否真的是 `3.x`
- 是否误导入了当前工作区里的本地源码

如果不是 `3.x`，脚本会直接报中文错并退出。

---

## 4. `cellpose3` 的核心调用方式是什么

这份脚本参考了仓库里的 notebook 调用方式，核心入口是：

```python
from cellpose import denoise

model = denoise.CellposeDenoiseModel(
    gpu=True,
    model_type="cyto3",
    restore_type="denoise_cyto3",
)

masks, flows, styles, restored_image = model.eval(
    image,
    channels=[0, 0],
    diameter=30,
)
```

你可以把它理解成两步合在一起：

1. 先用 `restore_type` 对图像做恢复
2. 再用 `model_type` 做分割

返回值里最常用的是：

- `masks`：最终实例分割标签
- `flows`：中间 flow / cellprob 信息
- `restored_image`：恢复后的图像

---

## 5. 主要配置项在哪里改

直接改脚本顶部“配置区”即可：

```python
INPUT_DIR = PROJECT_ROOT / "outputs" / "cellpose3_demo_inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cellpose3_restore_seg"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
USE_GPU = True
MODEL_TYPE = "cyto3"
RESTORE_TYPE = "denoise_cyto3"
CHANNELS = [0, 0]
CHAN2_RESTORE = False
DIAMETER = 30
BATCH_SIZE = 8
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 15
NORMALIZE = True
SAVE_SEG_NPY = True
SAVE_INSTANCE_MASK = True
SAVE_SEMANTIC_MASK = True
SAVE_OVERLAY = True
SAVE_RESTORED_IMAGE = True
SAVE_INPUT_IMAGE_COPY = True
SAVE_FLOW_RGB = True
SAVE_CELLPROB = True
```

你最常改的通常是下面这些：

### `INPUT_DIR`

- 输入图像目录
- 改成你自己的图像文件夹

### `OUTPUT_DIR`

- 输出根目录
- 建议每次单独放一个结果目录，方便对比不同参数结果

### `MODEL_TYPE`

- 分割主模型类型
- 常用：
  - `"cyto3"`：细胞
  - `"nuclei"`：细胞核

### `RESTORE_TYPE`

- 图像恢复模型类型
- 常用：
  - `"denoise_cyto3"`：偏去噪
  - `"deblur_cyto3"`：偏去模糊
  - `"upsample_cyto3"`：偏低分辨率提升
  - `"denoise_nuclei"`：核图去噪
  - `"deblur_nuclei"`：核图去模糊
  - `"upsample_nuclei"`：核图上采样恢复

经验上建议：

- 图像噪声大：先试 `denoise_*`
- 图像模糊明显：先试 `deblur_*`
- 图像明显偏小、欠采样：先试 `upsample_*`

### `CHANNELS`

- 指定输入图像的通道语义
- 常见写法：
  - `[0, 0]`：灰度图
  - `[2, 3]`：G 是细胞，B 是核
  - `[2, 1]`：G 是细胞，R 是核

### `CHAN2_RESTORE`

- 如果你的第二通道确实是核通道，并且也希望它先做恢复，可以改成 `True`

### `DIAMETER`

- 这是很重要的参数
- 目标越小，值可以更小
- 目标越大，值可以更大

如果你发现：

- mask 太碎
- 检出太少
- 恢复尺度不对

优先就从 `DIAMETER` 开始调。

### `FLOW_THRESHOLD`

- 控制流场一致性过滤强度
- 越低通常保留更多候选实例
- 越高更保守

### `CELLPROB_THRESHOLD`

- 控制像素进入 mask 的概率阈值
- 越低一般越容易出更多前景
- 越高一般更严格

### `MIN_SIZE`

- 滤掉小噪点实例

---

## 6. 输入支持什么

当前脚本默认处理：

- 2D 灰度图
- 2D 多通道图像

支持的后缀：

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.bmp`

当前默认行为是：

- 只扫描 `INPUT_DIR` 当前这一层
- 不递归子目录
- 不处理 3D 栈

如果你后面要跑 3D、按 z-stack 跑、或做多目录批处理，建议另外写专门脚本，不建议在这份 2D 脚本上硬打补丁。

---

## 7. 输出会生成什么

脚本会在 `OUTPUT_DIR` 下创建这些子目录：

### `input_images`

保存输入图像副本：

```text
xxx_input_image.tif
```

### `restored_images`

保存 `cellpose3` 恢复后的图像：

```text
xxx_restored_image.png
```

这是 `cellpose3` 和普通分割脚本最大的区别之一。
现在默认保存为便于直接查看的 `png`。

### `instance_masks`

保存实例 ID mask：

```text
xxx_instance_mask.png
```

或当实例编号太多时：

```text
xxx_instance_mask.tif
```

说明：

- 背景为 `0`
- 实例从 `1, 2, 3 ...` 编号

### `semantic_masks`

保存语义二值 mask：

```text
xxx_semantic_mask.png
```

说明：

- 背景为 `0`
- 前景为 `255`

### `overlays`

保存叠加可视化图：

```text
xxx_overlay.png
```

适合你快速肉眼检查分割质量。

### `flow_rgb`

保存 flow RGB 可视化图：

```text
xxx_flow_rgb.png
```

适合粗看流场方向与结构是否正常。

### `cellprob`

保存 cell probability：

```text
xxx_cellprob.tif
xxx_cellprob_preview.png
```

说明：

- `*.tif`：原始浮点值
- `*_preview.png`：方便查看的归一化预览图

当结果“几乎没分出来”时，这一组文件很值得优先检查。

### `seg_npy`

保存 GUI 兼容文件：

```text
xxx_seg.npy
```

这份文件除了 mask，也会尽量保留恢复相关信息，方便你后续回 GUI 检查。

---

## 8. 运行方式

### 8.1 先激活独立环境

```powershell
conda activate cellpose3
```

### 8.2 运行脚本

```powershell
python E:\cellpose\cellpose\tools\run_cellpose3_restore_seg.py
```

运行时脚本会输出中文日志，例如：

- 当前配置
- 实际使用 CPU 还是 GPU
- 当前处理到哪一张图
- 图像形状
- 检测到多少实例
- 文件保存到了哪里
- 哪些图失败
- 最终统计

---

## 9. 什么时候该选哪种 `RESTORE_TYPE`

这是最常见的问题之一。

### `denoise_*`

适合：

- 噪声重
- 散斑明显
- 颗粒感强

### `deblur_*`

适合：

- 边界虚
- 成像发糊
- 轮廓拖尾

### `upsample_*`

适合：

- 图像本身采样太粗
- 目标显得过小
- 放大后细节明显不足

如果你一开始拿不准，建议优先试：

1. `denoise_cyto3`
2. `deblur_cyto3`
3. `upsample_cyto3`

然后对比：

- `restored_images`
- `overlays`
- `cellprob_preview`

哪一组看起来最好，就优先用哪一组参数继续跑批量数据。

---

## 10. 对灰度图怎么设置

如果你的图像是普通单通道灰度图，最常用的配置就是：

```python
MODEL_TYPE = "cyto3"
RESTORE_TYPE = "denoise_cyto3"
CHANNELS = [0, 0]
CHAN2_RESTORE = False
```

这也是这份脚本默认给你的起始配置。

---

## 11. 运行后会生成哪些文件

假设输入文件是：

```text
sample_001.png
```

默认会生成：

```text
OUTPUT_DIR/input_images/sample_001_input_image.tif
OUTPUT_DIR/restored_images/sample_001_restored_image.png
OUTPUT_DIR/instance_masks/sample_001_instance_mask.png
OUTPUT_DIR/semantic_masks/sample_001_semantic_mask.png
OUTPUT_DIR/overlays/sample_001_overlay.png
OUTPUT_DIR/flow_rgb/sample_001_flow_rgb.png
OUTPUT_DIR/cellprob/sample_001_cellprob.tif
OUTPUT_DIR/cellprob/sample_001_cellprob_preview.png
OUTPUT_DIR/seg_npy/sample_001_seg.npy
```

---

## 12. 风险点和注意事项

### 12.1 最大风险不是参数，而是环境混用

这份脚本最重要的风险点其实不是 `FLOW_THRESHOLD`，而是：

- 你以为自己在跑 `cellpose3`
- 实际导入的是当前仓库里的 `Cellpose-SAM / v4`

所以脚本已经内置了版本检查。

### 12.2 `MODEL_TYPE` 和 `RESTORE_TYPE` 尽量配套

例如：

- `MODEL_TYPE = "cyto3"` 时，优先配 `denoise_cyto3 / deblur_cyto3 / upsample_cyto3`
- `MODEL_TYPE = "nuclei"` 时，优先配 `denoise_nuclei / deblur_nuclei / upsample_nuclei`

不要随便交叉乱配，否则结果可能很怪。

### 12.3 `instance_mask` 看起来发黑，不代表没有结果

实例 mask 保存的是实例编号，不是普通灰度图。

例如：

- 背景 = 0
- 第 1 个实例 = 1
- 第 2 个实例 = 2
- 第 100 个实例 = 100

普通图片查看器经常会把它显示得很黑，这不等于真没分出来。

判断有没有结果，建议优先看：

1. `overlay`
2. `semantic_mask`
3. `cellprob_preview`
4. mask 的最大值是否大于 0

### 12.4 当前脚本默认只做 2D

如果你后面要：

- 做 3D 栈恢复 + 分割
- 做更复杂的多通道批处理
- 做多目录递归

建议单独再写脚本，不要把这份 2D 脚本不断塞复杂分支。

---

## 13. 结论

这份脚本的定位很明确：

- 用 `cellpose3` 做标准的“恢复 + 分割”批处理
- 顶部配置区直接改
- 中文日志
- 输出恢复图、mask、overlay、`_seg.npy`
- 显式防止和当前仓库的 v4 环境混用

如果你下一步还想继续扩展，我建议优先顺序是：

1. 先确定你自己的数据更适合 `denoise`、`deblur` 还是 `upsample`
2. 再把 `DIAMETER` 调到稳定
3. 最后再细调 `FLOW_THRESHOLD`、`CELLPROB_THRESHOLD` 和 `MIN_SIZE`
