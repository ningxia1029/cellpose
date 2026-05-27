# Cellpose 角膜内皮分割工作流

本仓库基于 [MouseLand/cellpose](https://github.com/MouseLand/cellpose) 的 Cellpose-SAM 主线代码整理，当前本地主要用途不是单纯保留官方示例，而是围绕角膜内皮细胞图像做批量分割、标签导出、调试结果保存，以及后续 U-Net / distance map 训练数据整理。

官方 Cellpose-SAM 的 GUI、API、命令行能力仍然保留；新增内容集中放在 `tools/` 目录下，默认遵循“直接修改脚本顶部配置区即可运行”的方式，方便在 Windows 本地工程中重复处理不同批次数据。

## 当前仓库定位

- 保留官方 Cellpose / Cellpose-SAM 源码，便于直接调用官方模型、GUI、训练与推理能力。
- 新增角膜内皮 2D 图像批量推理脚本，重点导出实例 mask、语义 mask、轮廓、overlay、flow、cellprob 和逐阶段调试结果。
- 新增 Cellpose-SAM 导出结果到 U-Net distance dataset 的整理脚本，支持按样本组切分 train / val / test，避免同一原始样本泄漏到不同数据集。
- 保留一个 Cellpose3 图像恢复加分割脚本，用于 noisy / blurry / 低分辨率图像的恢复实验，但它需要独立的 cellpose3 环境，不能直接混用当前 Cellpose-SAM / v4 主线环境。

## 本机常用环境

当前本地已验证的常用解释器：

```powershell
D:\Anaconda3\envs\cellpose\python.exe
```

当前环境版本：

- `cellpose 4.1.1`
- `python 3.10.20`
- `torch 2.11.0+cu128`
- CUDA 可用：`True`

如果需要确认当前环境：

```powershell
cd E:\cellpose\cellpose
D:\Anaconda3\envs\cellpose\python.exe -m cellpose --version
D:\Anaconda3\envs\cellpose\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果重新安装为可编辑源码模式：

```powershell
cd E:\cellpose\cellpose
D:\Anaconda3\envs\cellpose\python.exe -m pip install -e .
```

## 新增工具脚本

### 1. `tools/run_cpsam_export.py`

用途：使用 Cellpose-SAM 或自训练 Cellpose 模型对 2D 图像批量推理，并导出后续训练需要的标签与调试文件。

当前脚本默认面向角膜内皮细胞 ROI 图像，关键配置集中在文件顶部：

- `INPUT_DIR`：待分割图像目录。
- `OUTPUT_DIR`：本次 Cellpose-SAM 导出结果根目录。
- `MODEL_NAME`：官方 `cpsam` 或自训练模型路径。
- `USE_GPU`：是否优先使用 GPU。
- `FLOW_THRESHOLD`、`CELLPROB_THRESHOLD`：Cellpose 后处理阈值。
- `MIN_SIZE`、`MAX_SIZE_FRACTION`：小噪点和异常大实例过滤参数。
- `SAVE_*` 系列开关：控制是否保存实例 mask、语义 mask、overlay、flow、cellprob、polygon、ROI 等结果。

当前默认路径已经切到 `roi_batch_selected_top3_v3` 这批数据：

```text
INPUT_DIR  = E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\roi_batch_selected_top3_v3\segmentation_inputs
OUTPUT_DIR = E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\roi_batch_selected_top3_v3\cpsam_export_0_0.0_size120
MODEL_NAME = E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\roi_batch_selected_top3_v3\segmentation_inputs\models\cpsam_20260525_161131
```

运行：

```powershell
cd E:\cellpose\cellpose
D:\Anaconda3\envs\cellpose\python.exe tools\run_cpsam_export.py
```

主要输出：

- `seg_npy/`：Cellpose GUI 可回读的 `_seg.npy`。
- `instance_masks/`：实例 ID mask，后续做实例相关训练或分析时使用。
- `semantic_masks/`：二值语义 mask，前景 255，背景 0。
- `separated_semantic_masks/`：去掉轮廓粘连后的语义 mask，更适合训练细胞内部区域。
- `overlays/`、`outline_overlays/`：肉眼检查用叠加图。
- `flow_rgb/`、`grad_xy/`、`cellprob/`：Cellpose 中间结果，方便判断空 mask 或漏检原因。
- `raw_dynamics_*`、`after_maxsize_*`、`after_flowqc_*`：不同后处理阶段的中间 mask，方便调参数。
- `debug_reports/`：逐图统计报告，包括 cellprob 通过像素数、连通域数、最终实例数等。

### 2. `tools/prepare_unet_distance_dataset.py`

用途：把 `run_cpsam_export.py` 生成的 Cellpose-SAM 结果整理成后续分割训练工程需要的数据集结构。

这个脚本重点解决两个问题：

- 自动生成 `train / val / test` 三套目录。
- 按样本组切分，而不是按单帧随机切分，避免同一个原始样本的 top01 / top02 / top03 图像进入不同 split。

关键配置集中在文件顶部：

- `SOURCE_ROOT`：Cellpose-SAM 导出根目录，应至少包含 `images/`、`semantic_masks/`、`instance_masks/`。
- `TARGET_DATA_ROOT`：目标训练工程的数据目录。
- `CLEAR_EXISTING_SPLITS`：是否先清空旧的 train / val / test。
- `COPY_IMAGES`：是否拷贝 ROI 输入图。
- `COPY_SELECTED_ORIGINAL_FRAMES`：是否额外拷贝原始帧。
- `SPLIT_RATIO`：组级别数据集切分比例，当前默认 `[7, 2, 1]`。
- `GROUP_KEY_MODE`、`GROUP_PREFIX_SEPARATOR`：样本组提取规则。

当前默认路径：

```text
SOURCE_ROOT      = E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\roi_batch_selected_top3_v3\cpsam_export_0_0.0_size120
TARGET_DATA_ROOT = E:\Tianlu\cell\deeplearing_algorithm\third_process_segment\data
```

运行：

```powershell
cd E:\cellpose\cellpose
D:\Anaconda3\envs\cellpose\python.exe tools\prepare_unet_distance_dataset.py
```

目标输出结构：

```text
data/
  train/
    images/
    original_frames/
    masks/
    distance_maps/
    debug/
  val/
    images/
    original_frames/
    masks/
    distance_maps/
    debug/
  test/
    images/
    original_frames/
    masks/
    distance_maps/
    debug/
```

其中 `distance_maps/` 会根据实例 mask 生成归一化距离图，适合后续做方案 B 这类带距离监督的训练。

### 3. `tools/run_cellpose3_restore_seg.py`

用途：调用独立安装的 Cellpose3 做图像恢复加分割，适合 noisy、blurry、低分辨率图像的实验。

注意：当前仓库主线是 Cellpose-SAM / v4；这个脚本内部会主动检查导入版本，避免误把当前源码环境当作 cellpose3 使用。运行前需要准备独立的 cellpose3 环境。

关键配置集中在文件顶部：

- `INPUT_DIR`：待恢复和分割的图像目录。
- `OUTPUT_DIR`：输出目录。
- `MODEL_TYPE`：分割模型类型，例如 `cyto3`。
- `RESTORE_TYPE`：恢复模型类型，例如 `denoise_cyto3`。
- `DIAMETER`：目标直径。
- `FLOW_THRESHOLD`、`CELLPROB_THRESHOLD`、`MIN_SIZE`：后处理参数。

## 推荐执行顺序

1. 准备角膜内皮 ROI 输入图，确认文件已经放到 `run_cpsam_export.py` 的 `INPUT_DIR`。
2. 根据本批数据选择 `MODEL_NAME`，官方模型可用 `cpsam`，自训练模型填写完整路径。
3. 运行 `tools/run_cpsam_export.py`，先重点检查 `overlays/`、`outline_overlays/`、`debug_reports/`。
4. 如果实例数量过少或空 mask，优先查看 `cellprob/`、`cellprob_binary/`、`raw_dynamics_*`，再调整 `FLOW_THRESHOLD`、`CELLPROB_THRESHOLD`、`MIN_SIZE`。
5. 确认导出质量可用后，运行 `tools/prepare_unet_distance_dataset.py` 生成后续训练数据。
6. 在第三阶段训练工程中读取 `TARGET_DATA_ROOT` 下的 `train / val / test` 数据。

## 官方 Cellpose 功能仍然可用

打开 GUI：

```powershell
cd E:\cellpose\cellpose
D:\Anaconda3\envs\cellpose\python.exe -m cellpose
```

查看官方文档：

- 官方仓库：[MouseLand/cellpose](https://github.com/MouseLand/cellpose)
- 官方文档：[cellpose.readthedocs.io](https://cellpose.readthedocs.io/en/latest/)
- Cellpose-SAM 在线体验：[Hugging Face Space](https://huggingface.co/spaces/mouseland/cellpose)

官方示例 notebook 仍保留在 `notebooks/` 目录下，适合查官方 API 用法；本地角膜内皮批处理优先使用 `tools/` 目录里的脚本。

## 数据与提交注意事项

- 大批量图像、模型权重、导出结果和训练数据默认不要提交到 GitHub，只提交脚本、配置说明和必要文档。
- `tools/` 脚本中的路径是本机工程路径，换数据批次时优先改脚本顶部配置区。
- 如果要把脚本给其他机器使用，需要同步修改 `INPUT_DIR`、`OUTPUT_DIR`、`MODEL_NAME`、`SOURCE_ROOT`、`TARGET_DATA_ROOT`。
- 当前新增脚本默认输出中文日志，方便直接从 PowerShell 终端判断处理进度和异常原因。

## 引用与许可

本仓库保留官方 Cellpose 源码与许可文件，许可信息以 [LICENSE](LICENSE) 为准。

如果使用 Cellpose-SAM，请引用官方 Cellpose-SAM 论文：

Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv.

如果使用 Cellpose 1、2、3 或人机交互训练、图像恢复模型，请按官方 README 和文档中的说明引用对应论文。
