# Cellpose GUI 使用说明与 GPU 无法启用排查

本文档合并回答两个问题：

1. 你当前的 Cellpose GUI 为什么不能使用 GPU。
2. Cellpose GUI 的帮助入口、界面结构、常用功能和输出文件分别是什么。

---

## 1. 当前 GPU 不能用的根因

这次排查已经确认，你的机器本身 **不是没有 GPU**，而是 **Cellpose 所在 Python 环境里装的是 CPU 版 PyTorch**。

已确认的事实：

- 显卡存在且驱动正常：`nvidia-smi` 能识别到 `NVIDIA GeForce RTX 4060`
- Cellpose 环境路径：`D:\Anaconda3\envs\cellpose\python.exe`
- 该环境中的 `torch` 版本：`2.11.0+cpu`
- 该环境中的 `torch.cuda.is_available()` 返回 `False`
- Cellpose GUI 的 GPU 复选框启用逻辑在 [gui.py](/E:/cellpose/cellpose/cellpose/gui/gui.py) 中，只有 `core.use_gpu()` 为真时才会点亮

对应代码逻辑可以概括为：

- `cellpose/gui/gui.py` 先调用 `self.check_gpu()`
- `self.check_gpu()` 再调用 `cellpose.core.use_gpu()`
- 如果 `use_gpu()` 返回 `False`，GUI 中的 `use GPU` 复选框就会被禁用并变灰

所以这不是 GUI 控件坏了，而是当前环境没有 CUDA 版 `torch`。

---

## 2. 为什么会这样

你当前 `cellpose` 环境中的 `torch` 来自 CPU 轮子：

```python
torch_version= 2.11.0+cpu
cuda_available= False
cuda_version= None
```

只要是这种状态，Cellpose GUI 无法勾选 GPU 是正常现象。

另外还有一个容易混淆的点：

- Cellpose 官方 Windows 可执行版 `cellpose.exe` 本身就不提供 GPU 支持
- 如果你是双击官方 exe 包，而不是在 Conda/Python 环境里运行 `python -m cellpose`，那么 GUI 也不会启用 GPU

---

## 3. 推荐修复方式

### 3.1 先进入正确环境

在 PowerShell 或 Anaconda Prompt 中执行：

```powershell
conda activate cellpose
python --version
where python
```

你需要确认当前解释器就是：

```text
D:\Anaconda3\envs\cellpose\python.exe
```

### 3.2 卸载当前 CPU 版 torch

```powershell
python -m pip uninstall -y torch torchvision torchaudio
```

### 3.3 安装 CUDA 版 torch

根据 PyTorch 官方安装页面与官方历史版本页面，Windows + pip 可以直接使用 CUDA 轮子安装。  
结合你当前已经安装的 Cellpose 版本与本次实际验证结果，推荐直接使用下面这一组：

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

说明：

- 这里安装的是 **CUDA 12.8 对应的官方 PyTorch 轮子**
- PyTorch pip 轮子自带 CUDA 运行时，**不会依赖你本机安装的完整 CUDA Toolkit 才能运行**
- 你的显卡驱动版本足够新，一般可以兼容这类 CUDA 轮子
- 这一组版本已经在你当前的 `D:\Anaconda3\envs\cellpose` 环境中实际安装并验证通过

### 3.4 安装完成后验证

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')"
python -c "from cellpose import core; print(core.use_gpu())"
```

理想结果：

- `torch.__version__` 不再带 `+cpu`
- `torch.version.cuda` 有值
- `torch.cuda.is_available()` 为 `True`
- `core.use_gpu()` 为 `True`

### 3.5 再启动 GUI

```powershell
python -m cellpose
```

如果环境已经修好，那么 GUI 左侧 `Segmentation` 区域中的 `use GPU` 复选框就会变为可用，并通常默认勾选。

---

## 4. GUI 的主要帮助入口

你可以通过以下几种方式查看 GUI 帮助：

1. GUI 内按 `Ctrl + H`
2. GUI 顶部菜单 `Help`
3. 项目内置帮助 HTML：
   [guihelpwindowtext.html](/E:/cellpose/cellpose/cellpose/gui/guihelpwindowtext.html)
4. 项目文档：
   [gui.rst](/E:/cellpose/cellpose/docs/gui.rst)
5. 在线文档：
   [Cellpose GUI 文档](https://cellpose.readthedocs.io/en/latest/gui.html)

如果你后续打包给别人用，我更建议同时保留：

- 这份中文文档
- `Ctrl + H` 的英文原始帮助

这样最不容易遗漏官方约定。

---

## 5. 你截图里的界面怎么理解

你截图中的主要区域从上到下可以这样看：

### 5.1 Views

- 控制显示模式，例如 RGB、灰度、单通道查看
- 可切换不同可视化层，例如原图、flow、cell probability
- 适合检查输入图像和推理中间结果

### 5.2 Drawing

- 用于手工画掩膜、删掩膜、批量删 ROI
- `MASKS ON [X]` 控制是否显示 mask
- `outlines on [Z]` 控制是否显示轮廓
- `single stroke` 控制 2D 单笔绘制方式

### 5.3 Segmentation

- `use GPU`：只有环境支持 CUDA/MPS 才能启用
- `run CPSAM`：运行 Cellpose-SAM 分割
- `0 ROIs`：当前图像识别或标注到的实例数
- `additional settings`：展开后可调阈值、直径、迭代等参数

### 5.4 user-trained models

- 这里可以加载或切换你自己训练的 Cellpose 模型
- 不影响内置 `run CPSAM`

### 5.5 Image filtering

- 这是图像恢复/滤波入口
- 可做锐化、平滑、tile normalization 等预处理
- 这一部分和 Cellpose3 的恢复模型能力有关，不只是“显示增强”

---

## 6. GUI 常用操作

### 6.1 启动 GUI

```powershell
python -m cellpose
```

如果是 3D 栈：

```powershell
python -m cellpose --Zstack
```

### 6.2 加载图像

- 直接把图像拖进窗口
- 或按 `Ctrl + L`

支持常见格式：

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.gif`

### 6.3 运行分割

1. 打开图像
2. 视情况确认 `use GPU`
3. 在 `Segmentation` 区点击 `run CPSAM`
4. 等待进度条完成
5. 如果 `MASKS ON` 勾选，结果会直接叠加显示

### 6.4 手工修订掩膜

- 右键开始绘制 mask
- 回到起点或再次右键结束绘制
- `Ctrl + 左键` 删除 mask
- `Alt + 左键` 合并最近两个 mask

---

## 7. 常用快捷键

最常用的是这些：

- `Ctrl + H`：打开帮助
- `Ctrl + L`：加载图像
- `Ctrl + S`：保存到 `_seg.npy`
- `Ctrl + N`：保存 masks 为 PNG
- `Ctrl + R`：保存 ImageJ ROI zip
- `Ctrl + F`：保存 flow 图
- `A / D` 或左右方向键：切换当前目录中的上一张/下一张图
- `X`：显示/隐藏 masks
- `Z`：显示/隐藏 outlines
- `PageUp / PageDown`：切换不同视图

---

## 8. GUI 会生成哪些输出文件

### 8.1 `_seg.npy`

这是最重要的 GUI 原生结果文件，通常和原图同目录，文件名类似：

```text
xxx_seg.npy
```

里面通常包含：

- `filename`
- `masks`
- `outlines`
- `flows`
- `ismanual`
- `diameter`

这个文件最适合：

- 回到 GUI 中继续编辑
- 作为 Cellpose 的训练输入之一
- 保留完整中间结果

### 8.2 mask 图像

你可以另存为：

- `*_cp_masks.png`
- `*_cp_masks.tif`

### 8.3 ROI 压缩包

可保存为：

```text
*_rois.zip
```

它可以直接用于 Fiji / ImageJ 的 ROI Manager。

### 8.4 flow 与 cellprob

GUI 也支持保存 flow 和 cell probability 相关图像，方便排查阈值问题。

---

## 9. 适合你当前项目的建议用法

你的目标不是长期依赖 Cellpose-SAM 直接部署到 `rk3568`，而是：

1. 先用 Cellpose-SAM 批量生成实例标签
2. 再用你自己的 `Unet + MobileNetV3` 训练轻量模型
3. 最后把轻量模型部署到 `rk3568`

对于这个目标，最推荐保留三类文件：

1. 实例 ID mask
2. `_seg.npy`
3. 叠加可视化图

原因：

- 实例 mask 是训练主标签
- `_seg.npy` 便于回 GUI 复查和修标签
- overlay 便于快速抽检质量

---

## 10. 本次结论

你的 GUI 当前不能使用 GPU，根因非常明确：

- **不是显卡问题**
- **不是 GUI 按钮问题**
- **而是 `cellpose` 环境中的 `torch` 被安装成了 CPU 版**

只要把 `D:\Anaconda3\envs\cellpose` 环境里的 `torch / torchvision / torchaudio` 换成官方 CUDA 轮子，再用同一个环境启动 `python -m cellpose`，GUI 的 `use GPU` 就应该恢复正常。
