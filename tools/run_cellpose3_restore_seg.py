"""
使用 Cellpose3 进行 2D 图像批量“图像恢复 + 分割”。

设计目标：
1. 直接修改文件顶部配置区即可运行，不依赖命令行参数。
2. 默认面向 2D 图像，适合你把 noisy / blurry / 低分辨率图先恢复，再做分割。
3. 强制检查当前导入的是否为 cellpose3，避免把当前仓库主线的 Cellpose-SAM / v4 源码误当成 cellpose3 来跑。
4. 默认导出恢复图、实例 mask、语义 mask、overlay、_seg.npy 以及部分调试结果，方便你后续复查和再利用。
"""

from __future__ import annotations

import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


# ============================== 配置区：请优先修改这里 ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入图像目录：
# 用途：待处理的原始图像所在目录。
# 默认值：项目根目录下 outputs/cellpose3_demo_inputs。
# 可调范围：任意本地目录；当前脚本默认只扫描这一层目录，不递归子目录。

INPUT_DIR = Path(r"E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\ROI\0208_3_01_9\crops")

# 输出根目录：
# 用途：保存本次 cellpose3 恢复与分割结果。
# 默认值：项目根目录下 outputs/cellpose3_restore_seg。
# 可调范围：任意本地可写目录。
OUTPUT_DIR = Path(r"E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\output\roi_batch_selected_top2\output_cellpose3")

# 待处理图像后缀：
# 用途：脚本只会读取这些后缀名的文件。
# 默认值：常见 2D 图像格式。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# 是否优先使用 GPU：
# True：优先尝试 CUDA / MPS。
# False：强制使用 CPU。
# 注意：只有当前独立的 cellpose3 环境里 torch 本身支持 GPU 时，才会真正启用 GPU。
USE_GPU = True

# cellpose3 分割模型类型：
# 作用：指定恢复后用于分割的主模型。
# 默认值："cyto3"。
# 常见可选值："cyto3"、"nuclei"。
# 如果你有自己训练的 cellpose3 分割模型，也可以改成该模型路径。
MODEL_TYPE = "cyto3"

# cellpose3 恢复模型类型：
# 作用：指定先做哪一种恢复，再接分割。
# 默认值："denoise_cyto3"。
# 常见可选值：
#   - "denoise_cyto3"
#   - "deblur_cyto3"
#   - "upsample_cyto3"
#   - "denoise_nuclei"
#   - "deblur_nuclei"
#   - "upsample_nuclei"
# 注意：一般应和 MODEL_TYPE 配套使用，细胞图优先 cyto3，细胞核图优先 nuclei。
RESTORE_TYPE = "denoise_cyto3"

# channels 参数：
# 作用：告诉 cellpose3 输入图里“分割目标通道”和“核通道”分别是哪一通道。
# 默认值：[0, 0]，表示单通道灰度图，无单独核通道。
# 常见示例：
#   - [0, 0]：灰度图
#   - [2, 3]：G 通道是细胞，B 通道是细胞核
#   - [2, 1]：G 通道是细胞，R 通道是细胞核
CHANNELS = [0, 0]

# 是否对第二通道也执行恢复：
# 作用：当你输入的是“细胞通道 + 核通道”双通道图时，可让第二通道也做恢复。
# 默认值：False。
# 典型使用场景：例如 CHANNELS = [2, 3]，且第二通道确实是核图时，可改成 True。
CHAN2_RESTORE = False

# 目标直径 diameter：
# 作用：告诉 cellpose3 目标大概有多大，影响恢复尺度和分割后处理。
# 默认值：30。
# 可调建议：
#   - 细胞更小：可尝试 10 ~ 20
#   - 常规细胞：常见 20 ~ 40
#   - 更大目标：可尝试 40 ~ 80
# 如果你非常确定图像尺度稳定，建议显式填写，而不是依赖默认估计。
DIAMETER = 30

# 推理 batch_size：
# 作用：控制 cellpose3 内部按 patch 推理时的批大小，不是“图片张数批处理”。
# 默认值：8。
# 显存不足可尝试减到 4、2 或 1。
BATCH_SIZE = 8

# flow_threshold：
# 作用：控制流场一致性过滤强度。
# 默认值：0.4。
# 越低通常越容易保留更多实例，越高越保守。
FLOW_THRESHOLD = 0.4

# cellprob_threshold：
# 作用：控制像素进入最终 mask 的概率阈值。
# 默认值：0.0。
# 越低通常越容易保留更多更大的前景区域，越高越保守。
CELLPROB_THRESHOLD = 0.0

# 最小实例面积：
# 作用：滤掉太小的噪点实例。
# 默认值：15 像素。
MIN_SIZE = 15

# 是否启用归一化：
# 作用：决定是否按 cellpose 默认方式对图像做强度归一化。
# 默认值：True。
NORMALIZE = True

# 是否保存 GUI 可直接打开的 _seg.npy：
# 默认值：True。
# 建议：保持开启，后续回 GUI 继续检查和修订会很方便。
SAVE_SEG_NPY = True

# 是否保存实例 ID mask：
# 默认值：True。
SAVE_INSTANCE_MASK = True

# 是否保存语义二值 mask：
# 默认值：True。
SAVE_SEMANTIC_MASK = True

# 是否保存叠加可视化图：
# 默认值：True。
SAVE_OVERLAY = True

# 是否保存恢复后的图像：
# 默认值：True。
SAVE_RESTORED_IMAGE = True

# 是否保存输入图像副本：
# 默认值：True。
SAVE_INPUT_IMAGE_COPY = True

# 是否保存 flow RGB：
# 默认值：True。
SAVE_FLOW_RGB = True

# 是否保存 cellprob 原图与预览图：
# 默认值：True。
SAVE_CELLPROB = True
# ==============================================================================


def log(message: str) -> None:
    """输出中文日志。

    输入：
        message: 需要打印的日志文本。
    输出：
        无。日志会直接打印到终端，便于观察脚本进度与异常位置。
    注意事项：
        统一附带时间戳，后续排查“卡在哪一步”会更直观。
    """

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def sanitize_sys_path_for_installed_cellpose3() -> list[str]:
    """移除当前工作区路径，避免误导入本仓库里的 v4 源码。

    输入：
        无。
    输出：
        removed_paths: 被移除的路径列表，便于日志中记录。
    处理逻辑：
        1. 当前脚本位于本仓库的 tools 目录下。
        2. 如果直接在当前工作区运行，Python 可能优先把工作区里的本地源码当成 `cellpose` 包导入。
        3. 但本脚本想调用的是“独立环境里安装的 cellpose3”，因此需要先把工作区相关路径从 sys.path 中剔掉。
    """

    current_file = Path(__file__).resolve()
    workspace_paths = {
        current_file.parents[1],  # E:\cellpose\cellpose
        current_file.parents[2],  # E:\cellpose
    }

    kept_paths: list[str] = []
    removed_paths: list[str] = []

    for raw_path in sys.path:
        try:
            resolved_path = Path(raw_path or ".").resolve()
        except Exception:
            kept_paths.append(raw_path)
            continue

        if resolved_path in workspace_paths:
            removed_paths.append(str(resolved_path))
            continue
        kept_paths.append(raw_path)

    sys.path[:] = kept_paths
    return removed_paths


REMOVED_SYS_PATHS = sanitize_sys_path_for_installed_cellpose3()

try:
    import cellpose  # type: ignore
    from cellpose import core, denoise, io, plot  # type: ignore
except Exception as import_exc:  # pragma: no cover - 这里只做显式报错提示
    log("错误：导入 cellpose 失败。")
    log("请确认你当前激活的是独立的 cellpose3 环境，而不是当前仓库主线的 v4 环境。")
    log("建议先执行：")
    print("  conda activate cellpose3")
    print("  python -m pip install \"opencv-python-headless>=4.9.0.80\"")
    print("  python -m pip install cellpose==3.1.1.2")
    raise import_exc


def get_installed_cellpose_version() -> str:
    """获取当前环境中已安装的 cellpose 版本号。"""

    try:
        return version("cellpose")
    except PackageNotFoundError:
        return "unknown"


INSTALLED_CELLPOSE_VERSION = get_installed_cellpose_version()


def ensure_cellpose3_environment() -> None:
    """确认当前实际导入的是 cellpose3。

    输入：
        无。
    输出：
        无。若环境不符合要求，会直接抛出异常并给出中文提示。
    注意事项：
        当前仓库主线已经是 Cellpose-SAM / v4，和 cellpose3 不是同一套使用方式。
        如果这里不拦住，最容易出现“脚本能 import，但模型其实跑错版本”的隐蔽问题。
    """

    if not INSTALLED_CELLPOSE_VERSION.startswith("3."):
        raise RuntimeError(
            "当前环境里的 cellpose 不是 3.x，无法按 cellpose3 方式运行。\n"
            f"检测到的版本为：{INSTALLED_CELLPOSE_VERSION}\n"
            "请单独创建 cellpose3 环境，并安装例如：cellpose==3.1.1.2"
        )


def create_output_dirs(output_root: Path) -> dict[str, Path]:
    """创建本次推理需要的输出目录。"""

    output_dirs = {
        "root": output_root,
        "input_images": output_root / "input_images",
        "restored_images": output_root / "restored_images",
        "instance_masks": output_root / "instance_masks",
        "semantic_masks": output_root / "semantic_masks",
        "overlays": output_root / "overlays",
        "flow_rgb": output_root / "flow_rgb",
        "cellprob": output_root / "cellprob",
        "seg_npy": output_root / "seg_npy",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def collect_image_files(input_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """收集输入目录中的待处理图像。"""

    suffix_set = {suffix.lower() for suffix in suffixes}
    image_files = [
        file_path for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in suffix_set
    ]
    image_files.sort(key=lambda path: path.name.lower())
    return image_files


def prepare_image(image_path: Path) -> np.ndarray:
    """读取单张输入图像，并做基础合法性检查。

    输入：
        image_path: 输入图像路径。
    输出：
        image: 原始读取后的 numpy 数组。
    注意事项：
        当前脚本默认面向 2D 图像或 2D 多通道图像，不做 3D 栈处理。
    """

    image = io.imread(str(image_path))
    if image is None:
        raise ValueError(f"读取图像失败：{image_path}")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"当前脚本只支持 2D 图像或 2D 多通道图像，检测到维度为 {image.ndim}：{image_path}"
        )

    return image


def build_display_image(image: np.ndarray) -> np.ndarray:
    """把输入图整理成适合做 overlay 的 H x W x 3 图像。

    输入：
        image: 原始图像，可为 2D 灰度图，或 2D 多通道图。
    输出：
        display_image: 适合 `plot.mask_overlay` 使用的 3 通道可视化图像。
    关键说明：
        这里不改变用于模型推理的原图，只是单独生成一份“方便看 overlay 的显示图”。
    """

    if image.ndim == 2:
        converted = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        converted = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        converted = image[..., :3]
    else:
        raise ValueError(f"无法为 overlay 生成显示图，当前图像形状为：{image.shape}")

    return converted


def get_output_stub(image_path: Path, output_dir: Path) -> Path:
    """根据输入文件名生成输出基础路径。"""

    return output_dir / image_path.stem


def save_input_image_file(image: np.ndarray, output_stub: Path) -> Path:
    """保存输入图像副本。

    输入：
        image: 需要保存的图像数组。
        output_stub: 输出基础路径，不带后缀。
    输出：
        save_path: 实际保存路径。
    注意事项：
        输入图像副本仍然保存为 tif，尽量保留原始位深与像素精度。
    """

    save_path = output_stub.with_name(f"{output_stub.name}_input_image.tif")
    io.imsave(str(save_path), image)
    return save_path


def save_restored_image_file(restored_image: np.ndarray, output_stub: Path) -> Path:
    """保存恢复后的图像为 png。

    输入：
        restored_image: cellpose3 返回的恢复图像。
        output_stub: 输出基础路径，不带后缀。
    输出：
        save_path: 实际保存路径。
    处理逻辑：
        1. 如果恢复图本身已经是 `uint8`，则直接保存为 png。
        2. 如果是浮点图，则自动映射到 0~255 后保存为 png，方便你直接查看。
    注意事项：
        png 更适合日常浏览，但不如 tif 适合无损保留浮点恢复值。
        这里按你的要求改成 png，因此更偏“查看方便”而不是“保留原始数值”。
    """

    restored_to_save = restored_image

    if restored_to_save.dtype != np.uint8:
        restored_to_save = restored_to_save.astype(np.float32)
        min_value = float(restored_to_save.min())
        max_value = float(restored_to_save.max())

        # 关键说明：
        # cellpose3 返回的恢复图在不同版本/不同输入下，可能是 0~1 浮点，也可能是其他范围的浮点值。
        # 这里统一做一次线性拉伸到 0~255，目的是让 png 结果“直接可看”。
        if abs(max_value - min_value) < 1e-8:
            restored_to_save = np.zeros(restored_to_save.shape, dtype=np.uint8)
        else:
            restored_to_save = (restored_to_save - min_value) / (max_value - min_value)
            restored_to_save = np.clip(restored_to_save * 255.0, 0, 255).astype(np.uint8)

    save_path = output_stub.with_name(f"{output_stub.name}_restored_image.png")
    io.imsave(str(save_path), restored_to_save)
    return save_path


def save_instance_mask_file(masks: np.ndarray, output_stub: Path) -> Path:
    """保存实例 ID mask。"""

    masks_to_save = masks.astype(np.uint32 if masks.max() > 65535 else np.uint16)
    if masks.max() > 65535:
        save_path = output_stub.with_name(output_stub.name + "_instance_mask.tif")
    else:
        save_path = output_stub.with_name(output_stub.name + "_instance_mask.png")
    io.imsave(str(save_path), masks_to_save)
    return save_path


def save_semantic_mask_file(masks: np.ndarray, output_stub: Path) -> Path:
    """保存语义分割二值 mask。"""

    semantic_mask = (masks > 0).astype(np.uint8) * 255
    save_path = output_stub.with_name(output_stub.name + "_semantic_mask.png")
    io.imsave(str(save_path), semantic_mask)
    return save_path


def save_overlay_file(display_image: np.ndarray, masks: np.ndarray, output_stub: Path) -> Path:
    """保存分割叠加可视化图。"""

    overlay = plot.mask_overlay(display_image, masks)
    save_path = output_stub.with_name(output_stub.name + "_overlay.png")
    io.imsave(str(save_path), overlay)
    return save_path


def save_flow_rgb_file(flow_rgb: np.ndarray, output_stub: Path) -> Path:
    """保存 flow RGB 可视化图。"""

    save_path = output_stub.with_name(output_stub.name + "_flow_rgb.png")
    io.imsave(str(save_path), flow_rgb.astype(np.uint8))
    return save_path


def save_cellprob_file(cellprob: np.ndarray, output_stub: Path) -> tuple[Path, Path]:
    """保存 cellprob 原图与预览图。

    输入：
        cellprob: `flows[2]`，通常为 H x W 的浮点图。
        output_stub: 输出基础路径。
    输出：
        raw_path: 原始浮点 tif 路径。
        preview_path: 归一化预览 png 路径。
    """

    raw_path = output_stub.with_name(output_stub.name + "_cellprob.tif")
    preview_path = output_stub.with_name(output_stub.name + "_cellprob_preview.png")

    io.imsave(str(raw_path), cellprob.astype(np.float32))

    if np.allclose(cellprob.max(), cellprob.min()):
        preview = np.zeros(cellprob.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(cellprob, 1))
        high = float(np.percentile(cellprob, 99))
        if abs(high - low) < 1e-8:
            preview = np.zeros(cellprob.shape, dtype=np.uint8)
        else:
            preview = np.clip((cellprob - low) / (high - low), 0, 1)
        preview = (preview * 255).astype(np.uint8)

    io.imsave(str(preview_path), preview)
    return raw_path, preview_path


def save_seg_npy_file(model,
                      image: np.ndarray,
                      masks: np.ndarray,
                      flows: list[np.ndarray],
                      restored_image: np.ndarray,
                      source_image_path: Path,
                      seg_dir: Path) -> Path:
    """保存 GUI 可直接打开的 `_seg.npy`。

    输入：
        model: cellpose3 的 CellposeDenoiseModel 实例。
        image: 原始输入图像。
        masks: 实例标签图。
        flows: model.eval 返回的 flow 相关中间结果。
        restored_image: 恢复后的图像，会一并写入 `_seg.npy`。
        source_image_path: 原始图像路径。
        seg_dir: `_seg.npy` 输出目录。
    输出：
        seg_path: 实际生成的 `_seg.npy` 路径。
    关键说明：
        cellpose3 的 `_seg.npy` 不只保存 mask，还可以保存 restore 信息。
        这样你后续回 GUI 时，更容易对齐“恢复后再分割”的完整过程。
    """

    seg_reference_path = seg_dir / source_image_path.name
    restore_ratio = float(getattr(getattr(model, "dn", None), "ratio", 1.0))

    io.masks_flows_to_seg(
        image,
        masks,
        flows,
        str(seg_reference_path),
        channels=CHANNELS,
        imgs_restore=restored_image,
        restore_type=RESTORE_TYPE,
        ratio=restore_ratio,
    )

    seg_path = seg_dir / f"{source_image_path.stem}_seg.npy"
    seg_data = np.load(seg_path, allow_pickle=True).item()
    seg_data["filename"] = str(source_image_path)
    seg_data["source_image_path"] = str(source_image_path)
    seg_data["restore_type"] = RESTORE_TYPE
    seg_data["model_type"] = MODEL_TYPE
    seg_data["export_note"] = "由 tools/run_cellpose3_restore_seg.py 生成"
    np.save(seg_path, seg_data)
    return seg_path


def init_model():
    """初始化 cellpose3 模型，并返回模型对象与是否实际启用 GPU。"""

    gpu_available = core.use_gpu() if USE_GPU else False

    if USE_GPU and gpu_available:
        log("检测到可用 GPU，本次 cellpose3 推理将使用 GPU。")
    elif USE_GPU and not gpu_available:
        log("警告：配置中要求使用 GPU，但当前环境没有可用 GPU，将自动退回 CPU。")
    else:
        log("配置中已指定使用 CPU 推理。")

    model = denoise.CellposeDenoiseModel(
        gpu=(USE_GPU and gpu_available),
        model_type=MODEL_TYPE,
        restore_type=RESTORE_TYPE,
        chan2_restore=CHAN2_RESTORE,
    )
    return model, bool(USE_GPU and gpu_available)


def print_config() -> None:
    """打印当前配置，方便运行前快速核对。"""

    log("当前脚本配置如下：")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  图像后缀: {IMAGE_SUFFIXES}")
    print(f"  是否请求GPU: {USE_GPU}")
    print(f"  cellpose版本: {INSTALLED_CELLPOSE_VERSION}")
    print(f"  分割模型类型 MODEL_TYPE: {MODEL_TYPE}")
    print(f"  恢复模型类型 RESTORE_TYPE: {RESTORE_TYPE}")
    print(f"  channels: {CHANNELS}")
    print(f"  是否恢复第二通道: {CHAN2_RESTORE}")
    print(f"  目标直径 DIAMETER: {DIAMETER}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  flow_threshold: {FLOW_THRESHOLD}")
    print(f"  cellprob_threshold: {CELLPROB_THRESHOLD}")
    print(f"  min_size: {MIN_SIZE}")
    print(f"  normalize: {NORMALIZE}")
    print(f"  保存_seg.npy: {SAVE_SEG_NPY}")
    print(f"  保存实例mask: {SAVE_INSTANCE_MASK}")
    print(f"  保存语义mask: {SAVE_SEMANTIC_MASK}")
    print(f"  保存overlay: {SAVE_OVERLAY}")
    print(f"  保存恢复图: {SAVE_RESTORED_IMAGE}")
    print(f"  保存输入图副本: {SAVE_INPUT_IMAGE_COPY}")
    print(f"  保存flow RGB: {SAVE_FLOW_RGB}")
    print(f"  保存cellprob: {SAVE_CELLPROB}")
    if REMOVED_SYS_PATHS:
        print("  为避免误导入本仓库 v4 源码，已临时移除以下 sys.path：")
        for removed_path in REMOVED_SYS_PATHS:
            print(f"    - {removed_path}")


def main() -> int:
    """主函数：批量执行 cellpose3 恢复与分割。"""

    try:
        ensure_cellpose3_environment()
        print_config()

        if not INPUT_DIR.exists():
            log(f"错误：输入目录不存在，请先修改 INPUT_DIR。当前路径：{INPUT_DIR}")
            return 1
        if not INPUT_DIR.is_dir():
            log(f"错误：INPUT_DIR 不是目录。当前路径：{INPUT_DIR}")
            return 1

        output_dirs = create_output_dirs(OUTPUT_DIR)
        image_files = collect_image_files(INPUT_DIR, IMAGE_SUFFIXES)
        if not image_files:
            log(f"错误：在输入目录中没有找到待处理图像。目录：{INPUT_DIR}")
            return 1

        log(f"共找到 {len(image_files)} 张待处理图像。")
        model, using_gpu = init_model()
        log(f"模型初始化完成。实际推理设备：{'GPU' if using_gpu else 'CPU'}")

        success_count = 0
        failed_files: list[Path] = []
        total_masks = 0
        begin_time = time.time()

        for index, image_path in enumerate(image_files, start=1):
            log(f"开始处理第 {index}/{len(image_files)} 张图像：{image_path.name}")
            try:
                image = prepare_image(image_path)
                display_image = build_display_image(image)
                log(f"图像读取完成，图像形状：{image.shape}")

                masks, flows, _, restored_image = model.eval(
                    image,
                    batch_size=BATCH_SIZE,
                    channels=CHANNELS,
                    normalize=NORMALIZE,
                    diameter=DIAMETER,
                    flow_threshold=FLOW_THRESHOLD,
                    cellprob_threshold=CELLPROB_THRESHOLD,
                    min_size=MIN_SIZE,
                )

                mask_count = int(masks.max())
                total_masks += mask_count
                log(f"恢复与分割完成，识别到 {mask_count} 个实例。")

                if mask_count == 0:
                    log("警告：当前图像没有生成实例，建议优先检查 channels、DIAMETER、RESTORE_TYPE 和 cellprob 结果。")

                if SAVE_INPUT_IMAGE_COPY:
                    input_stub = get_output_stub(image_path, output_dirs["input_images"])
                    input_save_path = save_input_image_file(image, input_stub)
                    log(f"输入图副本已保存：{input_save_path}")

                if SAVE_RESTORED_IMAGE:
                    restored_stub = get_output_stub(image_path, output_dirs["restored_images"])
                    restored_save_path = save_restored_image_file(restored_image,
                                                                  restored_stub)
                    log(f"恢复图像已保存：{restored_save_path}")

                if SAVE_INSTANCE_MASK:
                    instance_stub = get_output_stub(image_path, output_dirs["instance_masks"])
                    instance_path = save_instance_mask_file(masks, instance_stub)
                    log(f"实例 ID mask 已保存：{instance_path}")

                if SAVE_SEMANTIC_MASK:
                    semantic_stub = get_output_stub(image_path, output_dirs["semantic_masks"])
                    semantic_path = save_semantic_mask_file(masks, semantic_stub)
                    log(f"语义二值 mask 已保存：{semantic_path}")

                if SAVE_OVERLAY:
                    overlay_stub = get_output_stub(image_path, output_dirs["overlays"])
                    overlay_path = save_overlay_file(display_image, masks, overlay_stub)
                    log(f"叠加可视化图已保存：{overlay_path}")

                if SAVE_FLOW_RGB:
                    flow_stub = get_output_stub(image_path, output_dirs["flow_rgb"])
                    flow_path = save_flow_rgb_file(flows[0], flow_stub)
                    log(f"flow RGB 已保存：{flow_path}")

                if SAVE_CELLPROB:
                    cellprob_stub = get_output_stub(image_path, output_dirs["cellprob"])
                    cellprob_raw_path, cellprob_preview_path = save_cellprob_file(
                        flows[2], cellprob_stub)
                    log(f"cellprob 原图已保存：{cellprob_raw_path}")
                    log(f"cellprob 预览图已保存：{cellprob_preview_path}")

                if SAVE_SEG_NPY:
                    seg_path = save_seg_npy_file(
                        model=model,
                        image=image,
                        masks=masks,
                        flows=flows,
                        restored_image=restored_image,
                        source_image_path=image_path,
                        seg_dir=output_dirs["seg_npy"],
                    )
                    log(f"GUI 兼容的 _seg.npy 已保存：{seg_path}")

                success_count += 1
                log(f"当前图像处理完成：{image_path.name}")

            except Exception as exc:
                failed_files.append(image_path)
                log(f"错误：处理图像失败：{image_path}")
                log(f"异常信息：{repr(exc)}")
                traceback.print_exc()

        elapsed = time.time() - begin_time
        log("全部图像处理结束。")
        log(f"成功数量：{success_count}")
        log(f"失败数量：{len(failed_files)}")
        log(f"累计实例数量：{total_masks}")
        log(f"总耗时：{elapsed:.2f} 秒")
        log(f"输出根目录：{OUTPUT_DIR}")

        if failed_files:
            log("以下文件处理失败，请重点检查：")
            for failed_file in failed_files:
                print(f"  - {failed_file}")
            return 1

        return 0

    except Exception as exc:
        log(f"脚本异常退出：{repr(exc)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
