# Cellpose 工程目录与功能总览

本文档用于快速回答这几个问题：

1. 这个工程目录里都有什么。
2. 它能做什么，不只是分割吗。
3. 有没有去噪、去模糊、恢复相关能力。
4. 哪些部分适合直接拿来用，哪些更偏论文/实验代码。

---

## 1. 项目根目录整体结构

项目根目录主要可以分成下面几类：

### `cellpose/`

这是核心 Python 包目录，绝大多数真正的功能代码都在这里。

你最需要关注的模块包括：

- `models.py`：模型封装与推理入口
- `core.py`：设备分配、GPU 检测
- `io.py`：读图、保存 `_seg.npy`、保存 mask、保存 ROI
- `plot.py`：可视化
- `train.py`：训练流程
- `transforms.py`：图像格式整理与预处理
- `denoise.py`：恢复模型，包括去噪、去模糊、上采样等
- `vit_sam.py`：Cellpose-SAM 相关网络定义
- `gui/`：GUI 界面与交互逻辑

### `docs/`

官方文档源码目录。

里面已经覆盖了：

- 安装
- GUI
- CLI
- 输出格式
- 模型
- 训练
- 3D
- 恢复模型
- 分布式处理

如果你想快速定位某个功能能不能做，先看这里通常最快。

### `notebooks/`

官方示例 notebook，偏“演示 / 教学 / 说明流程”。

你当前最相关的是：

- `run_Cellpose-SAM.ipynb`
- `test_Cellpose-SAM.ipynb`
- `train_Cellpose-SAM.ipynb`
- `run_cellpose3.ipynb`

这些 notebook 很适合参考参数和典型调用方式，但不一定适合直接作为长期维护脚本。

### `paper/`

论文相关分析与实验代码目录。

这里包含：

- 1.0 / 2.0 / 3.0 论文相关脚本
- `cpsam/` 中的 Cellpose-SAM 论文实验代码
- 各种 benchmark、分析、绘图、训练子集实验

这部分更偏研究复现实验，不建议直接当“生产脚本”用。

### `tests/`

单元测试与输出测试。

适合你：

- 看官方默认输出命名规则
- 看 CLI 怎么调用
- 看 `_seg.npy` / PNG / TIF 的测试方式

### `pyinstaller/`

打包相关配置，用于做 Windows/Mac 可执行包。

---

## 2. 这个工程是不是不只是分割

是的，这个工程 **不只是分割**。

它至少包括下面几大类能力：

### 2.1 通用细胞分割

这是最核心能力，包括：

- Cellpose 早期通用模型
- Cellpose2/3 路线
- Cellpose-SAM 路线

### 2.2 GUI 标注与交互修订

GUI 不只是“看图”，还支持：

- 运行模型
- 手工修掩膜
- 自动保存 `_seg.npy`
- 训练自定义模型

### 2.3 训练自定义分割模型

支持：

- 使用目录中的图像与标签训练
- 使用 `_seg.npy` 作为训练标签来源
- 用 GUI 的 human-in-the-loop 方式逐步细化模型

### 2.4 图像恢复

这个项目确实包含图像恢复相关能力，不仅仅是分割。

在 [denoise.py](/E:/cellpose/cellpose/cellpose/denoise.py) 和 [restore.rst](/E:/cellpose/cellpose/docs/restore.rst) 中，可以看到它支持：

- 去噪
- 去模糊
- 上采样
- one-click 恢复

所以你问“是不是不只是分割模型，还有去模糊、去噪模型”，答案是：

**是的，有。**

但要注意一件很关键的事：

- 这些恢复模型主要属于 **Cellpose3 路线**
- 不是 Cellpose-SAM 主线能力

换句话说：

- `Cellpose-SAM` 更偏“超强泛化分割”
- `Cellpose3` 更强调“恢复 + 分割联动”

### 2.5 分布式大图分割

工程里还有：

- `cellpose/contrib/distributed_segmentation.py`

这说明它还考虑了大图、多块、分布式切片推理场景。

### 2.6 模型导出与量化探索

项目中可以看到：

- `cellpose/export.py`
- `model_quantization.ipynb`

这说明项目并非完全不考虑部署，但当前已有内容更偏：

- BioImage.IO 导出
- TorchScript/BioImage.IO 生态
- 模型量化实验

而不是你这种 `rk3568 / RKNN` 的现成交付链路。

---

## 3. 你当前最值得关注的目录

如果你的目标是：

- 用 Cellpose-SAM 产标签
- 再训练自己的轻量化网络
- 最终部署到 `rk3568`

那么最重要的目录优先级建议如下。

### 第一优先级

- `cellpose/`
- `docs/`
- `notebooks/`

原因：

- 真正推理与导出逻辑都在 `cellpose/`
- 文档定义了官方输出格式与参数含义
- notebook 方便你对照官方示例

### 第二优先级

- `tests/`

原因：

- 里面能看到官方输出命名和典型 CLI 用法

### 第三优先级

- `paper/`

原因：

- 可以参考论文实验思路
- 但不建议直接照抄做工程入口

---

## 4. 关键模块分别干什么

### `cellpose/core.py`

负责：

- GPU 检测
- 设备分配
- CPU / CUDA / MPS 切换

这也是 GUI 能否点亮 `use GPU` 的关键链路之一。

### `cellpose/models.py`

负责：

- 初始化 Cellpose / Cellpose-SAM 模型
- 调用 `eval()` 做推理

你后续做脚本推理时最常用这里。

### `cellpose/io.py`

负责：

- 读取图像
- 保存 `_seg.npy`
- 保存 PNG / TIF mask
- 保存 ImageJ ROI

这部分对你当前“生成训练标签”最重要。

### `cellpose/gui/`

负责：

- GUI 窗口
- 菜单
- 保存逻辑
- 标注交互
- 帮助页面

如果你后续还想继续通过 GUI 修标签，这部分很关键。

### `cellpose/denoise.py`

负责：

- 图像恢复模型
- 去噪
- 去模糊
- 上采样
- 恢复后再分割

这说明工程确实不只是“裸分割”。

### `cellpose/export.py`

负责：

- 把模型打包为 BioImage.IO 格式

更适合通用生物图像模型共享，不是 RK3568 的现成部署方案。

---

## 5. 对你当前任务来说，哪些是现成可用的

### 现成可直接用的

- Cellpose-SAM 推理
- GUI 手工修标签
- `_seg.npy` 保存与加载
- PNG / TIF mask 导出
- ROI zip 导出
- Cellpose 自定义模型训练

### 有能力但不是你当前首选的

- 图像恢复模型
- 大图分布式处理
- BioImage.IO 导出
- 量化实验 notebook

### 当前没有直接现成满足你需求的

- “顶部配置区即可运行”的中文批处理脚本
- 面向你当前流程的实例标签导出工具
- 面向 `rk3568 / RKNN` 的现成导出链路
- 通用 LabelMe JSON 导出流程

所以本轮新增脚本和中文文档正好补的是这个空白。

---

## 6. 对部署到 RK3568 的现实建议

从工程现状来看，不建议你直接把 Cellpose-SAM 本体拿去做 `rk3568` 部署主模型，原因有三点：

### 6.1 模型很大

项目里的量化 notebook 显示，Cellpose-SAM 网络体量非常大。

这类模型：

- 显存占用高
- 推理慢
- 部署链复杂

对于 `rk3568` 这类边缘设备并不友好。

### 6.2 工程没有现成 RKNN 路线

仓库里没有看到：

- RKNN 导出脚本
- RK3568 专门优化代码
- NPU 前后处理适配

### 6.3 你的路线本来就更合理

你当前的想法其实是更合适的：

1. 用 Cellpose-SAM 生成高质量训练标签
2. 训练你自己的 `Unet + MobileNetV3`
3. 导出并部署你的轻量模型到 `rk3568`

这个路线比“硬上 Cellpose-SAM 原模型部署”更现实。

---

## 7. 我对这个工程的结论

这个仓库不是单一的“分割模型仓库”，而是一个比较完整的生物图像分割平台，至少包括：

- 分割推理
- GUI 标注
- 自定义训练
- 图像恢复
- 大图 / 分布式处理
- BioImage.IO 导出
- 论文实验与 benchmark

如果只从你当前任务角度看，最值得保留和复用的是：

1. `Cellpose-SAM` 作为强教师模型
2. `_seg.npy` 作为 GUI / 训练之间的桥接格式
3. 你自己的轻量模型作为最终部署模型

这也是当前最贴合 `rk3568` 目标的一条路线。
