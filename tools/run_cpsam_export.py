"""
使用 Cellpose-SAM 对 2D 图像批量推理，并导出训练标签。

设计目标：
1. 直接修改本文件顶部配置区即可运行，不依赖命令行参数。
2. 默认面向 2D 灰度图，尤其适合 400x1200 这类角膜内皮细胞图像。
3. 同时保留 Cellpose-SAM 的关键中间结果，方便你分析为什么有些图能出 mask、有些图出不来。
4. 默认同时导出：
   - 实例分割标签
   - 语义分割标签
   - GUI 兼容的 ``_seg.npy``
   - 原图 / flow RGB / gradXY / cellprob 等中间结果
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from cellpose import core, dynamics, io, models, plot, transforms, utils


# ============================== 配置区：请优先修改这里 ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入图像目录：
# 建议直接改成你自己的角膜内皮细胞灰度图目录。
# 默认值：项目根目录下的 data/cornea_images
# 可调范围：任意本地目录，只要里面放的是 2D 图像即可。
INPUT_DIR = Path(r"E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\output\roi_batch_selected_top2\segmentation_inputs")

# 输出根目录：
# 脚本会在该目录下自动创建 seg_npy / instance_masks / overlays / rois 等子目录。
# 默认值：项目根目录下的 outputs/cpsam_export
# 可调范围：任意本地可写目录。
OUTPUT_DIR = Path(r"E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\output\roi_batch_selected_top2\cpsam_export_0.8_0.0_size15")

# 待处理图像后缀：
# 只会扫描这些后缀的文件，统一转小写后匹配。
# 默认值：常见 2D 图像格式。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# 分割模型名称：
# 默认使用 Cellpose-SAM 的官方模型 cpsam。
# 如果后续你训练了自定义 Cellpose 模型，也可以改成模型名或完整模型路径。
MODEL_NAME = "cpsam"

# 是否优先使用 GPU：
# True 表示优先尝试 CUDA / MPS；False 表示强制使用 CPU。
# 注意：只有当前 Python 环境中的 torch 真正带 CUDA 时，GPU 才会启用。
USE_GPU = True

# Cellpose 内部推理批大小：
# 这是 Cellpose 在图像分块推理时使用的 batch_size，不是“图片张数批处理”。
# 默认值：8。显存不足可改小到 4、2、1；显存充足可适当调大。
BATCH_SIZE = 8

# flow_threshold：
# 作用：控制 Cellpose 对流场一致性的过滤强度。
# 默认值：0.4。调低通常会保留更多目标，调高会更严格。
FLOW_THRESHOLD = 0.8

# cellprob_threshold：
# 作用：控制细胞概率阈值。
# 默认值：0.0。调低会检出更多更大的目标，调高会更保守。
CELLPROB_THRESHOLD = 0.0

# 动力学积分迭代次数：
# 作用：控制像素沿 flow 场聚合的步数。
# 默认值：200，与当前这类 2D 图像、默认直径设置下的 Cellpose 常见行为一致。
# 可调范围：通常 100~400。越大越容易让像素汇聚到稳定中心，但也可能让边界更激进。
NITER = 200

# 最大目标面积占整图比例：
# 作用：过滤掉异常大的“吞并型”伪实例。
# 默认值：0.4，表示单个实例若超过整图 40% 面积，会被当作异常大目标移除。
# 如果你想尽量保留更多候选实例，可暂时改成 1.0。
MAX_SIZE_FRACTION = 0.4

# 最小目标面积（像素）：
# 小于该面积的实例会被过滤。
# 默认值：15。对于噪点较多的图像可适当调大。
MIN_SIZE = 15

# 是否保存 GUI 兼容的 _seg.npy：
# 推荐保持 True，这样后续你可以直接回到 Cellpose GUI 中复查与修订。
SAVE_SEG_NPY = True

# 是否保存实例 ID mask：
# 推荐保持 True，这是后续训练轻量化分割网络最核心的标签输出。
SAVE_INSTANCE_MASK = True

# 是否保存实例 ID mask 的彩色预览图：
# 原始实例标签图本质是 0/1/2/3... 的整数编号，普通图片查看器常会把它显示成“全黑”。
# 打开这个开关后，会额外保存一份彩色预览图，便于你肉眼确认实例是否真的存在。
SAVE_INSTANCE_MASK_PREVIEW = True

# 是否保存语义分割 mask：
# 语义分割 mask 由实例标签直接二值化得到，前景像素为 255，背景为 0。
# 如果你后续训练的是普通二分类 U-Net，这个输出会更直接。
SAVE_SEMANTIC_MASK = True

# 是否保存“相邻细胞不粘连”的语义 mask：
# 该输出会先取实例前景，再减去轮廓线区域。
# 结果是细胞内部为 255、背景和边界为 0，适合你做“细胞内部”语义分割训练。
SAVE_SEPARATED_SEMANTIC_MASK = True

# 是否保存叠加可视化图：
# 推荐保持 True，便于快速肉眼检查推理质量。
SAVE_OVERLAY = True

# 是否保存轮廓线 mask：
# 轮廓线来自 Cellpose 最终实例 mask，和 GUI 的 outlines on 属于同一来源。
# 输出为二值图，轮廓像素为 255，其余为 0。
SAVE_OUTLINE_MASK = True

# 是否保存轮廓线叠加图：
# 该图会把最终轮廓线直接画在原图上，比普通 overlay 更适合检查相邻细胞边界是否贴合。
SAVE_OUTLINE_OVERLAY = True

# 是否保存原图副本：
# 便于和 gradXY / cellprob / mask 做一一对应检查。
SAVE_IMAGE = True

# 是否保存 flow 的 RGB 可视化图：
# 对应 Cellpose 返回的 flows[0]，主要用于直观看流场方向。
SAVE_FLOW_RGB = True

# 是否保存 gradXY：
# 对应 Cellpose 返回的 flows[1]，是二维流场 dY / dX 的原始浮点输出。
SAVE_GRAD_XY = True

# 是否保存 cellprob：
# 对应 Cellpose 返回的 flows[2]，是生成 mask 前最关键的概率图。
# 当实例 mask 为空时，建议优先看这个输出。
SAVE_CELLPROB = True

# 是否保存 cellprob 阈值二值图：
# 该图只表示“哪些像素通过了 cellprob_threshold”，还不是最终实例。
# 它可以帮助你判断：问题究竟出在 cellprob 阈值，还是出在后续动力学与质量过滤。
SAVE_CELLPROB_BINARY_MASK = True

# 是否导出“只做动力学聚合、不做 max_size / flow_qc / min_size 过滤”的原始候选实例：
# 这一步最接近你在 flow_rgb 中肉眼看到的“很多细胞候选区域”。
# 如果你觉得最终 overlay 数量过少，优先对比这一组结果。
SAVE_RAW_DYNAMICS_MASK = True

# 是否导出 raw dynamics 的语义 mask：
# 如果你想尝试“保留更多候选细胞”来训练语义分割，可以先看这一份。
# 但要注意：这份结果更激进，误检风险也更高，因此不建议盲目当作唯一真值标签。
SAVE_RAW_DYNAMICS_SEMANTIC_MASK = True

# 是否导出 raw dynamics 的叠加可视化图：
# 便于直接和最终 overlay 对照，看动力学之后、过滤之前保留了多少细胞。
SAVE_RAW_DYNAMICS_OVERLAY = True

# 是否导出“经过 max_size_fraction 过滤后”的中间实例结果：
# 便于判断是否有大量实例是在“大目标过滤”这一步被合并或移除的。
SAVE_AFTER_MAXSIZE_MASK = True

# 是否导出“经过 max_size_fraction 过滤后”的叠加可视化图。
SAVE_AFTER_MAXSIZE_OVERLAY = True

# 是否导出“经过 flow_threshold 质量过滤后、但尚未做 min_size 过滤”的中间实例结果。
SAVE_AFTER_FLOWQC_MASK = True

# 是否导出“经过 flow_threshold 质量过滤后、但尚未做 min_size 过滤”的叠加可视化图。
SAVE_AFTER_FLOWQC_OVERLAY = True

# 是否保存逐阶段统计报告：
# 会把 cellprob 通过像素数、cellprob 连通域数、raw dynamics 实例数、after max_size 实例数、
# after flow_qc 实例数、final 实例数等写入文本，方便你调参数时有定量依据。
SAVE_DEBUG_SUMMARY = True

# 是否导出多边形轮廓坐标文本：
# 该输出会先拿到 GUI 同源的轮廓点，再用多边形近似压缩顶点数量，便于你后续做自定义标签。
SAVE_POLYGON_TXT = True

# 多边形轮廓近似强度：
# 作用：控制轮廓点压缩成多边形时的简化程度。
# 默认值：0.005，越小越贴近原始轮廓，越大越平滑、顶点更少。
# 对角膜内皮细胞这种近六边形结构，通常可以从 0.003 ~ 0.01 之间试。
POLYGON_APPROX_EPSILON_RATIO = 0.005

# 是否导出 ImageJ ROI zip：
# 默认值：False。只有在你确实需要给 Fiji / ImageJ 使用时再打开。
SAVE_ROIS = False
# ==============================================================================


def log(message: str) -> None:
    """输出中文日志。

    输入：
        message: 需要打印的日志文本。

    输出：
        无。日志会直接打印到终端，便于观察处理进度。

    注意事项：
        这里统一加上时间戳，方便排查卡在哪一步。
    """

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def create_output_dirs(output_root: Path) -> dict[str, Path]:
    """创建本次推理需要的输出目录。

    输入：
        output_root: 输出根目录。

    输出：
        返回包含各类输出子目录路径的字典。

    处理逻辑：
        1. 无论是否真的保存某类结果，先把常用目录都创建好，保证路径稳定。
        2. 后续按开关决定是否真正写入文件。
    """

    output_dirs = {
        "root": output_root,
        "seg_npy": output_root / "seg_npy",
        "instance_masks": output_root / "instance_masks",
        "instance_mask_previews": output_root / "instance_mask_previews",
        "semantic_masks": output_root / "semantic_masks",
        "separated_semantic_masks": output_root / "separated_semantic_masks",
        "overlays": output_root / "overlays",
        "outline_masks": output_root / "outline_masks",
        "outline_overlays": output_root / "outline_overlays",
        "images": output_root / "images",
        "flow_rgb": output_root / "flow_rgb",
        "grad_xy": output_root / "grad_xy",
        "cellprob": output_root / "cellprob",
        "cellprob_binary": output_root / "cellprob_binary",
        "raw_dynamics_instance_masks": output_root / "raw_dynamics_instance_masks",
        "raw_dynamics_semantic_masks": output_root / "raw_dynamics_semantic_masks",
        "raw_dynamics_overlays": output_root / "raw_dynamics_overlays",
        "after_maxsize_instance_masks": output_root / "after_maxsize_instance_masks",
        "after_maxsize_overlays": output_root / "after_maxsize_overlays",
        "after_flowqc_instance_masks": output_root / "after_flowqc_instance_masks",
        "after_flowqc_overlays": output_root / "after_flowqc_overlays",
        "debug_reports": output_root / "debug_reports",
        "polygon_txt": output_root / "polygon_txt",
        "rois": output_root / "rois",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def collect_image_files(input_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """收集输入目录中的待处理图像文件。

    输入：
        input_dir: 图像目录。
        suffixes: 允许处理的图像后缀集合。

    输出：
        按文件名排序后的图像路径列表。

    注意事项：
        当前脚本只扫描当前目录，不递归子目录。
        这样做是为了让路径结构更直观，也更方便你自己维护。
    """

    suffix_set = {suffix.lower() for suffix in suffixes}
    image_files = [
        file_path for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in suffix_set
    ]
    image_files.sort(key=lambda path: path.name.lower())
    return image_files


def prepare_2d_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并整理 2D 图像，生成推理输入与可视化原图。

    输入：
        image_path: 单张图像路径。

    输出：
        original_image: 原始图像数组，保留读取后的原始形态，供叠加可视化使用。
        model_input: 经过 Cellpose 2D 输入规范转换后的图像，保证为 H x W x 3。

    处理逻辑：
        1. 读取原图。
        2. 仅允许 2D 或 2D+通道图像，显式拒绝 3D 栈，避免误把体数据按 2D 跑坏。
        3. 使用 Cellpose 官方的 convert_image 做通道整理，减少自定义预处理偏差。
    """

    original_image = io.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"读取图像失败：{image_path}")

    if original_image.ndim not in (2, 3):
        raise ValueError(
            f"当前脚本仅支持 2D 图像，检测到图像维度为 {original_image.ndim}，文件：{image_path}"
        )

    if original_image.ndim == 3 and min(original_image.shape) > 4:
        raise ValueError(
            f"检测到疑似 3D 或非常规维度图像，形状为 {original_image.shape}，请改用专门的 3D 流程。"
        )

    # 关键说明：
    # 这里不自己手写“灰度复制到 3 通道”的逻辑，而是直接走 Cellpose 官方转换函数，
    # 这样可以与 GUI / CLI 的输入预处理保持尽量一致。
    model_input = transforms.convert_image(original_image, do_3D=False)
    return original_image, model_input


def get_output_stub(image_path: Path, output_dir: Path) -> Path:
    """根据输入文件名生成输出基础路径。"""

    return output_dir / image_path.stem


def save_instance_mask_file(masks: np.ndarray, output_stub: Path) -> Path:
    """保存实例 ID mask。

    输入：
        masks: Cellpose 输出的实例标签图，0 表示背景，1..N 表示不同实例。
        output_stub: 输出基础路径，不带后缀。

    输出：
        返回实际保存的实例 mask 文件路径。

    注意事项：
        1. 若实例编号不超过 65535，则优先保存 16 位 PNG，便于查看和后处理。
        2. 若实例数过多，则退回保存为 TIF，避免 PNG 精度不够。
    """

    masks_to_save = masks.astype(np.uint32 if masks.max() > 65535 else np.uint16)
    if masks.max() > 65535:
        save_path = output_stub.with_name(output_stub.name + "_instance_mask.tif")
    else:
        save_path = output_stub.with_name(output_stub.name + "_instance_mask.png")

    io.imsave(str(save_path), masks_to_save)
    return save_path


def build_instance_mask_preview(masks: np.ndarray) -> np.ndarray:
    """把实例标签图转成便于查看的彩色预览图。

    输入：
        masks: 实例标签图，背景为 0，实例为 1..N。

    输出：
        preview_rgb: uint8 彩色图。背景为黑色，不同实例会被映射成不同颜色。

    关键说明：
        原始实例标签图保存的是整数 ID，不是普通灰度图。
        因此即使文件里真的有 1~200 这样的实例编号，很多看图软件也会把它显示得近乎全黑。
        这里额外导出一份彩色预览，就是为了解决你当前看到的“全黑但其实不为空”的问题。
    """

    if masks.max() == 0:
        return np.zeros((*masks.shape, 3), dtype=np.uint8)

    hsv = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.float32)
    for instance_id in range(1, int(masks.max()) + 1):
        pixels = masks == instance_id
        if not np.any(pixels):
            continue
        hue = (instance_id * 0.61803398875) % 1.0
        hsv[pixels, 0] = hue
        hsv[pixels, 1] = 0.85
        hsv[pixels, 2] = 0.95
    preview_rgb = (utils.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return preview_rgb


def save_instance_mask_preview_file(masks: np.ndarray, output_stub: Path) -> Path:
    """保存实例标签图的彩色预览版本。"""

    preview_rgb = build_instance_mask_preview(masks)
    save_path = output_stub.with_name(output_stub.name + "_instance_mask_preview.png")
    io.imsave(str(save_path), preview_rgb)
    return save_path


def save_semantic_mask_file(masks: np.ndarray, output_stub: Path) -> Path:
    """保存语义分割二值 mask。

    输入：
        masks: 实例标签图，背景为 0，实例为 1..N。
        output_stub: 输出基础路径。

    输出：
        返回实际保存的语义 mask 文件路径。

    关键说明：
        这里直接把“是否属于任意细胞实例”转成前景/背景二值图，
        更适合普通语义分割网络直接训练。
    """

    semantic_mask = (masks > 0).astype(np.uint8) * 255
    save_path = output_stub.with_name(output_stub.name + "_semantic_mask.png")
    io.imsave(str(save_path), semantic_mask)
    return save_path


def build_cellprob_binary_mask(cellprob: np.ndarray, threshold: float) -> np.ndarray:
    """根据 cellprob_threshold 生成 cellprob 二值候选图。"""

    return (cellprob > float(threshold)).astype(np.uint8) * 255


def save_cellprob_binary_mask_file(cellprob_binary_mask: np.ndarray,
                                   output_stub: Path) -> Path:
    """保存 cellprob 二值候选图。"""

    save_path = output_stub.with_name(output_stub.name + "_cellprob_binary_mask.png")
    io.imsave(str(save_path), cellprob_binary_mask.astype(np.uint8))
    return save_path


def count_binary_connected_components(binary_mask: np.ndarray) -> int:
    """统计二值图中的连通域数量。

    输入：
        binary_mask: 0/255 的二值图。

    输出：
        连通域数量，不包含背景。
    """

    binary_uint8 = (binary_mask > 0).astype(np.uint8)
    component_count, _ = cv2.connectedComponents(binary_uint8)
    return max(int(component_count) - 1, 0)


def get_effective_niter(niter: int | None) -> int:
    """得到本脚本实际用于 dynamics 的迭代次数。"""

    return 200 if niter is None or int(niter) <= 0 else int(niter)


def compute_stage_masks(dP: np.ndarray,
                        cellprob: np.ndarray,
                        device,
                        niter: int,
                        cellprob_threshold: float,
                        flow_threshold: float,
                        max_size_fraction: float,
                        min_size: int) -> dict[str, np.ndarray | int]:
    """按 Cellpose 后处理链路逐阶段计算中间结果。

    输入：
        dP: 原始二维流场 ``flows[1]``。
        cellprob: 原始细胞概率图 ``flows[2]``。
        device: Cellpose 当前推理设备。
        niter: dynamics 迭代次数。
        cellprob_threshold: cellprob 阈值。
        flow_threshold: flow 质量过滤阈值。
        max_size_fraction: 最大目标面积占整图比例。
        min_size: 最小目标面积。

    输出：
        返回包含多阶段 mask 与统计信息的字典。

    阶段说明：
        1. raw_dynamics:
           只做 cellprob_threshold + dynamics 聚合，不做 max_size / flow_qc / min_size 过滤。
        2. after_maxsize:
           在 raw_dynamics 基础上，再做 max_size_fraction 过滤。
        3. after_flowqc:
           在 after_maxsize 基础上，再做 flow_threshold 质量过滤。
        4. final:
           在 after_flowqc 基础上，再做 min_size 小目标过滤。
    """

    cellprob_binary_mask = build_cellprob_binary_mask(cellprob, cellprob_threshold)
    raw_dynamics_masks = dynamics.compute_masks(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=0.0,
        min_size=-1,
        max_size_fraction=1.0,
        device=device,
    )
    after_maxsize_masks = dynamics.compute_masks(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=0.0,
        min_size=-1,
        max_size_fraction=max_size_fraction,
        device=device,
    )
    after_flowqc_masks = dynamics.compute_masks(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=-1,
        max_size_fraction=max_size_fraction,
        device=device,
    )
    final_masks = dynamics.compute_masks(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
        device=device,
    )

    return {
        "cellprob_binary_mask": cellprob_binary_mask,
        "cellprob_binary_pixel_count": int((cellprob_binary_mask > 0).sum()),
        "cellprob_binary_component_count": count_binary_connected_components(
            cellprob_binary_mask),
        "raw_dynamics_masks": raw_dynamics_masks,
        "after_maxsize_masks": after_maxsize_masks,
        "after_flowqc_masks": after_flowqc_masks,
        "final_masks": final_masks,
    }


def save_debug_summary_file(output_stub: Path,
                            image_path: Path,
                            dP: np.ndarray,
                            cellprob: np.ndarray,
                            stage_results: dict[str, np.ndarray | int],
                            final_masks: np.ndarray) -> Path:
    """保存逐阶段统计报告文本。"""

    save_path = output_stub.with_name(output_stub.name + "_debug_summary.txt")
    lines = [
        f"图像路径: {image_path}",
        f"FLOW_THRESHOLD: {FLOW_THRESHOLD}",
        f"CELLPROB_THRESHOLD: {CELLPROB_THRESHOLD}",
        f"MAX_SIZE_FRACTION: {MAX_SIZE_FRACTION}",
        f"MIN_SIZE: {MIN_SIZE}",
        f"NITER: {get_effective_niter(NITER)}",
        "",
        f"cellprob 最小值: {float(cellprob.min()):.6f}",
        f"cellprob 最大值: {float(cellprob.max()):.6f}",
        f"gradXY 最小值: {float(dP.min()):.6f}",
        f"gradXY 最大值: {float(dP.max()):.6f}",
        "",
        f"通过 cellprob_threshold 的像素数: {stage_results['cellprob_binary_pixel_count']}",
        f"通过 cellprob_threshold 的连通域数: {stage_results['cellprob_binary_component_count']}",
        f"仅做 dynamics 聚合后的实例数(raw_dynamics): {int(stage_results['raw_dynamics_masks'].max())}",
        f"做完 max_size_fraction 过滤后的实例数(after_maxsize): {int(stage_results['after_maxsize_masks'].max())}",
        f"做完 flow_threshold 过滤后的实例数(after_flowqc): {int(stage_results['after_flowqc_masks'].max())}",
        f"做完 min_size 过滤后的最终实例数(final): {int(final_masks.max())}",
        "",
        "说明：",
        "1. flow_rgb 只是 dP 的彩色可视化，不是显式实例标签。",
        "2. 如果 raw_dynamics 很多，而 final 明显变少，说明数量主要是在后处理过滤阶段被压下来的。",
        "3. 如果 cellprob_binary 连通域本身就很少，说明问题更多出在 cellprob，而不是后面的 flow_threshold。",
    ]
    save_path.write_text("\n".join(lines), encoding="utf-8")
    return save_path


def build_outline_mask(masks: np.ndarray) -> np.ndarray:
    """根据最终实例标签生成二值轮廓 mask。

    输入：
        masks: Cellpose 最终输出的实例标签图，0 为背景，1..N 为不同细胞实例。

    输出：
        outline_mask: uint8 类型的二值轮廓图，轮廓像素为 255，其余为 0。

    关键说明：
        1. 这里直接复用 Cellpose 官方的 ``utils.masks_to_outlines``。
        2. 因此生成逻辑与 GUI 里的 outlines on 保持同源，而不是从 flow_rgb 直接阈值化。
        3. 如果 masks 本身为空，会返回全 0 轮廓图。
    """

    if masks.max() == 0:
        return np.zeros(masks.shape, dtype=np.uint8)
    outline_mask = utils.masks_to_outlines(masks).astype(np.uint8) * 255
    return outline_mask


def save_outline_mask_file(outline_mask: np.ndarray, output_stub: Path) -> Path:
    """保存二值轮廓线 mask。"""

    save_path = output_stub.with_name(output_stub.name + "_outline_mask.png")
    io.imsave(str(save_path), outline_mask.astype(np.uint8))
    return save_path


def save_separated_semantic_mask_file(masks: np.ndarray, outline_mask: np.ndarray,
                                      output_stub: Path) -> Path:
    """保存“细胞内部不粘连”的语义 mask。

    输入：
        masks: 实例标签图。
        outline_mask: 由最终实例标签派生出的轮廓二值图，轮廓像素为 255。
        output_stub: 输出基础路径。

    输出：
        返回保存路径。

    关键说明：
        1. 普通 semantic mask 会把所有细胞前景都连成一片，因此相邻细胞在二值图里会粘连。
        2. 这个输出会把轮廓线区域从前景里挖掉，只保留每个细胞内部。
        3. 如果你后续训练的是“细胞内部 vs 非细胞内部”的 U-Net，这一版通常更适合。
    """

    separated_mask = ((masks > 0) & (outline_mask == 0)).astype(np.uint8) * 255
    save_path = output_stub.with_name(output_stub.name + "_separated_semantic_mask.png")
    io.imsave(str(save_path), separated_mask)
    return save_path


def normalize_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """把输入图像整理成便于绘制轮廓的 RGB uint8 图。"""

    if image.ndim == 2:
        normalized = np.clip(transforms.normalize99(image), 0, 1)
        image_uint8 = (normalized * 255).astype(np.uint8)
        return np.stack([image_uint8, image_uint8, image_uint8], axis=-1)

    if image.ndim == 3 and image.shape[-1] == 3:
        if image.dtype == np.uint8:
            return image.copy()
        normalized = np.clip(transforms.normalize99(image), 0, 1)
        return (normalized * 255).astype(np.uint8)

    raise ValueError(f"不支持的图像形状，无法生成轮廓叠加图：{image.shape}")


def save_outline_overlay_file(image: np.ndarray, outline_mask: np.ndarray,
                              output_stub: Path) -> Path:
    """把轮廓线直接叠加绘制到原图上。

    输入：
        image: 原始图像或用于可视化的图像。
        outline_mask: 二值轮廓图，轮廓像素为 255。
        output_stub: 输出基础路径。

    输出：
        返回保存路径。

    关键说明：
        这里使用红色描边，让你更容易检查相邻细胞是否被正确分开。
    """

    overlay = normalize_image_to_uint8(image)
    outline_pixels = outline_mask > 0
    overlay[outline_pixels] = np.array([255, 0, 0], dtype=np.uint8)
    save_path = output_stub.with_name(output_stub.name + "_outline_overlay.png")
    io.imsave(str(save_path), overlay)
    return save_path


def build_polygon_records(masks: np.ndarray,
                          epsilon_ratio: float) -> list[dict[str, object]]:
    """根据最终实例标签提取轮廓，并生成多边形顶点。

    输入：
        masks: Cellpose 最终实例标签图。
        epsilon_ratio: 多边形近似系数，相对于轮廓周长的比例。

    输出：
        polygon_records: 每个实例对应一个字典，包含实例编号、原始轮廓点、多边形顶点等信息。

    关键说明：
        1. 先使用 Cellpose 官方的 ``utils.outlines_list`` 提取 GUI 同源轮廓点。
        2. 再用 OpenCV 的 ``approxPolyDP`` 做多边形近似，便于后续自定义标签使用。
        3. 这里保留的是 xy 坐标顺序，方便与你后续常见标注格式对齐。
    """

    polygon_records: list[dict[str, object]] = []
    if masks.max() == 0:
        return polygon_records

    outline_points_list = utils.outlines_list(masks, multiprocessing=False)
    for instance_id, outline_points in enumerate(outline_points_list, start=1):
        if outline_points.size == 0:
            continue

        contour_xy = np.asarray(outline_points, dtype=np.int32).reshape(-1, 1, 2)
        contour_length = cv2.arcLength(contour_xy, True)
        epsilon = max(contour_length * float(epsilon_ratio), 1.0)
        polygon_xy = cv2.approxPolyDP(contour_xy, epsilon, True).reshape(-1, 2)

        polygon_records.append({
            "instance_id": instance_id,
            "contour_xy": contour_xy.reshape(-1, 2).tolist(),
            "polygon_xy": polygon_xy.tolist(),
            "contour_point_count": int(contour_xy.shape[0]),
            "polygon_point_count": int(polygon_xy.shape[0]),
        })
    return polygon_records


def save_polygon_txt_file(polygon_records: list[dict[str, object]],
                          output_stub: Path) -> Path:
    """保存多边形轮廓坐标文本。

    输入：
        polygon_records: ``build_polygon_records`` 生成的轮廓与多边形信息列表。
        output_stub: 输出基础路径。

    输出：
        返回保存路径。

    注意事项：
        为了方便你直接打开查看，这里使用可读性更高的纯文本格式，而不是额外引入 JSON。
    """

    save_path = output_stub.with_name(output_stub.name + "_polygons.txt")
    lines = [
        "# 说明：",
        "# 1. contour_xy 是 GUI 同源的原始轮廓像素点，坐标顺序为 x,y。",
        "# 2. polygon_xy 是对 contour_xy 做多边形近似后的顶点，坐标顺序同样为 x,y。",
        "# 3. 每个实例之间用空行分隔。",
        "",
    ]

    for record in polygon_records:
        contour_xy = record["contour_xy"]
        polygon_xy = record["polygon_xy"]
        lines.append(f"实例ID: {record['instance_id']}")
        lines.append(f"原始轮廓点数: {record['contour_point_count']}")
        lines.append(f"多边形顶点数: {record['polygon_point_count']}")
        lines.append(
            "contour_xy: " + "; ".join(f"{x},{y}" for x, y in contour_xy))
        lines.append(
            "polygon_xy: " + "; ".join(f"{x},{y}" for x, y in polygon_xy))
        lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    return save_path


def save_image_file(image: np.ndarray, output_stub: Path) -> Path:
    """保存输入图像副本。

    输入：
        image: 原始读取后的图像。
        output_stub: 输出基础路径。

    输出：
        返回保存路径。

    注意事项：
        为尽量保留原始灰度值，这里统一保存为 tif。
    """

    save_path = output_stub.with_name(output_stub.name + "_image.tif")
    io.imsave(str(save_path), image)
    return save_path


def save_overlay_file(display_image: np.ndarray, masks: np.ndarray,
                      output_stub: Path) -> Path:
    """保存分割叠加可视化图。

    输入：
        display_image: 用于可视化的 2D 图像，建议传入已经整理成 H x W x 3 的推理输入。
        masks: 实例标签图。
        output_stub: 输出基础路径。

    输出：
        返回实际保存的叠加图路径。
    """

    overlay = plot.mask_overlay(display_image, masks)
    save_path = output_stub.with_name(output_stub.name + "_overlay.png")
    io.imsave(str(save_path), overlay)
    return save_path


def save_flow_rgb_file(flow_rgb: np.ndarray, output_stub: Path) -> Path:
    """保存 Cellpose 的 flow RGB 可视化图。"""

    save_path = output_stub.with_name(output_stub.name + "_flow_rgb.png")
    io.imsave(str(save_path), flow_rgb.astype(np.uint8))
    return save_path


def save_grad_xy_file(grad_xy: np.ndarray, output_stub: Path) -> Path:
    """保存 gradXY 原始浮点结果。

    输入：
        grad_xy: ``flows[1]``，形状通常为 [2, H, W]，分别对应 dY 和 dX。
        output_stub: 输出基础路径。

    输出：
        返回保存路径。

    注意事项：
        这里保存的是原始浮点值，不做阈值化，方便后续分析后处理。
    """

    save_path = output_stub.with_name(output_stub.name + "_gradXY.tif")
    io.imsave(str(save_path), grad_xy.astype(np.float32))
    return save_path


def save_cellprob_file(cellprob: np.ndarray, output_stub: Path) -> tuple[Path, Path]:
    """保存 cellprob 原始值和便于查看的预览图。

    输入：
        cellprob: ``flows[2]``，通常是 H x W 的浮点图，范围大致在 -10 到 10。
        output_stub: 输出基础路径。

    输出：
        返回两个路径：
        1. 原始浮点 tif
        2. 归一化预览 png
    """

    raw_path = output_stub.with_name(output_stub.name + "_cellprob.tif")
    preview_path = output_stub.with_name(output_stub.name + "_cellprob_preview.png")

    io.imsave(str(raw_path), cellprob.astype(np.float32))

    if np.allclose(cellprob.max(), cellprob.min()):
        preview = np.zeros(cellprob.shape, dtype=np.uint8)
    else:
        preview = np.clip(transforms.normalize99(cellprob), 0, 1)
        preview = (preview * 255).astype(np.uint8)
    io.imsave(str(preview_path), preview)
    return raw_path, preview_path


def save_seg_npy_file(model_input: np.ndarray, masks: np.ndarray, flows: list[np.ndarray],
                      source_image_path: Path, seg_dir: Path) -> Path:
    """保存 GUI 兼容的 ``_seg.npy``。

    输入：
        model_input: 实际送入 Cellpose 的 2D 图像。
        masks: 分割实例标签图。
        flows: Cellpose 返回的 flow / cellprob 等中间结果。
        source_image_path: 原始输入图像路径。
        seg_dir: ``_seg.npy`` 保存目录。

    输出：
        返回生成的 ``_seg.npy`` 路径。

    关键逻辑：
        1. 先调用 Cellpose 官方的 ``io.masks_flows_to_seg`` 生成标准结构。
        2. 再把 ``filename`` 字段回填成原始输入图像路径，方便后续追溯数据来源。
    """

    seg_reference_path = seg_dir / source_image_path.name
    io.masks_flows_to_seg(model_input, masks, flows, str(seg_reference_path))
    seg_path = seg_dir / f"{source_image_path.stem}_seg.npy"

    seg_data = np.load(seg_path, allow_pickle=True).item()
    seg_data["filename"] = str(source_image_path)
    seg_data["source_image_path"] = str(source_image_path)
    seg_data["export_note"] = "由 tools/run_cpsam_export.py 生成"
    np.save(seg_path, seg_data)
    return seg_path


def save_roi_zip_file(masks: np.ndarray, image_path: Path, roi_dir: Path) -> Path:
    """保存 ImageJ/Fiji 可读的 ROI zip。"""

    roi_reference = roi_dir / image_path.name
    io.save_rois(masks, str(roi_reference))
    return roi_dir / f"{image_path.stem}_rois.zip"


def init_model(use_gpu: bool, model_name: str) -> tuple[models.CellposeModel, bool]:
    """初始化 Cellpose-SAM 模型，并根据环境决定是否真的启用 GPU。"""

    gpu_available = core.use_gpu() if use_gpu else False
    if use_gpu and gpu_available:
        log("检测到可用 GPU，本次推理将使用 GPU。")
    elif use_gpu and not gpu_available:
        log("警告：配置中要求使用 GPU，但当前环境没有可用 CUDA/MPS，脚本将自动退回 CPU。")
    else:
        log("配置中已指定使用 CPU 推理。")

    model = models.CellposeModel(gpu=(use_gpu and gpu_available),
                                 pretrained_model=model_name)
    return model, bool(use_gpu and gpu_available)


def print_config() -> None:
    """打印当前配置，方便确认是否改对路径和参数。"""

    log("当前脚本配置如下：")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  图像后缀: {IMAGE_SUFFIXES}")
    print(f"  模型名称: {MODEL_NAME}")
    print(f"  请求使用GPU: {USE_GPU}")
    print(f"  推理batch_size: {BATCH_SIZE}")
    print(f"  flow_threshold: {FLOW_THRESHOLD}")
    print(f"  cellprob_threshold: {CELLPROB_THRESHOLD}")
    print(f"  dynamics niter: {NITER}")
    print(f"  max_size_fraction: {MAX_SIZE_FRACTION}")
    print(f"  min_size: {MIN_SIZE}")
    print(f"  保存_seg.npy: {SAVE_SEG_NPY}")
    print(f"  保存实例mask: {SAVE_INSTANCE_MASK}")
    print(f"  保存实例mask预览图: {SAVE_INSTANCE_MASK_PREVIEW}")
    print(f"  保存语义mask: {SAVE_SEMANTIC_MASK}")
    print(f"  保存不粘连语义mask: {SAVE_SEPARATED_SEMANTIC_MASK}")
    print(f"  保存叠加图: {SAVE_OVERLAY}")
    print(f"  保存轮廓mask: {SAVE_OUTLINE_MASK}")
    print(f"  保存轮廓叠加图: {SAVE_OUTLINE_OVERLAY}")
    print(f"  保存原图副本: {SAVE_IMAGE}")
    print(f"  保存flow RGB: {SAVE_FLOW_RGB}")
    print(f"  保存gradXY: {SAVE_GRAD_XY}")
    print(f"  保存cellprob: {SAVE_CELLPROB}")
    print(f"  保存cellprob二值图: {SAVE_CELLPROB_BINARY_MASK}")
    print(f"  保存raw dynamics实例mask: {SAVE_RAW_DYNAMICS_MASK}")
    print(f"  保存raw dynamics语义mask: {SAVE_RAW_DYNAMICS_SEMANTIC_MASK}")
    print(f"  保存raw dynamics叠加图: {SAVE_RAW_DYNAMICS_OVERLAY}")
    print(f"  保存after max_size实例mask: {SAVE_AFTER_MAXSIZE_MASK}")
    print(f"  保存after max_size叠加图: {SAVE_AFTER_MAXSIZE_OVERLAY}")
    print(f"  保存after flow_qc实例mask: {SAVE_AFTER_FLOWQC_MASK}")
    print(f"  保存after flow_qc叠加图: {SAVE_AFTER_FLOWQC_OVERLAY}")
    print(f"  保存逐阶段调试报告: {SAVE_DEBUG_SUMMARY}")
    print(f"  保存多边形轮廓文本: {SAVE_POLYGON_TXT}")
    print(f"  多边形近似系数: {POLYGON_APPROX_EPSILON_RATIO}")
    print(f"  保存ROI压缩包: {SAVE_ROIS}")


def main() -> int:
    """主函数：批量推理并导出标签。"""

    try:
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
        model, using_gpu = init_model(USE_GPU, MODEL_NAME)
        log(f"模型初始化完成。实际推理设备：{'GPU' if using_gpu else 'CPU'}")

        success_count = 0
        failed_files: list[Path] = []
        total_masks = 0
        begin_time = time.time()

        for index, image_path in enumerate(image_files, start=1):
            log(f"开始处理第 {index}/{len(image_files)} 张图像：{image_path.name}")
            try:
                original_image, model_input = prepare_2d_image(image_path)
                log(f"图像读取完成，原始形状：{original_image.shape}，推理输入形状：{model_input.shape}")

                masks, flows, _ = model.eval(
                    model_input,
                    batch_size=BATCH_SIZE,
                    flow_threshold=FLOW_THRESHOLD,
                    cellprob_threshold=CELLPROB_THRESHOLD,
                    max_size_fraction=MAX_SIZE_FRACTION,
                    min_size=MIN_SIZE,
                    niter=NITER,
                    normalize=True,
                )
                dP = flows[1]
                cellprob = flows[2]

                mask_count = int(masks.max())
                total_masks += mask_count
                log(f"推理完成，识别到 {mask_count} 个实例。")

                stage_results = None
                if any([
                        SAVE_CELLPROB_BINARY_MASK,
                        SAVE_RAW_DYNAMICS_MASK,
                        SAVE_RAW_DYNAMICS_SEMANTIC_MASK,
                        SAVE_RAW_DYNAMICS_OVERLAY,
                        SAVE_AFTER_MAXSIZE_MASK,
                        SAVE_AFTER_MAXSIZE_OVERLAY,
                        SAVE_AFTER_FLOWQC_MASK,
                        SAVE_AFTER_FLOWQC_OVERLAY,
                        SAVE_DEBUG_SUMMARY,
                ]):
                    stage_results = compute_stage_masks(
                        dP=dP,
                        cellprob=cellprob,
                        device=model.device,
                        niter=get_effective_niter(NITER),
                        cellprob_threshold=CELLPROB_THRESHOLD,
                        flow_threshold=FLOW_THRESHOLD,
                        max_size_fraction=MAX_SIZE_FRACTION,
                        min_size=MIN_SIZE,
                    )
                    if not np.array_equal(stage_results["final_masks"], masks):
                        log("警告：逐阶段重算得到的 final masks 与 model.eval 返回结果存在差异，请重点检查 NITER / MAX_SIZE_FRACTION 配置。")

                outline_mask = None
                if (SAVE_OUTLINE_MASK or SAVE_OUTLINE_OVERLAY or
                        SAVE_SEPARATED_SEMANTIC_MASK or SAVE_POLYGON_TXT):
                    outline_mask = build_outline_mask(masks)
                    log(f"已根据最终实例标签生成轮廓线，共 {int((outline_mask > 0).sum())} 个轮廓像素。")

                # 关键说明：
                # Cellpose-SAM 的后处理并不是“只看 mask”，而是会先利用 cellprob_threshold
                # 筛出参与动力学积分的像素，再结合 gradXY / flow_threshold 生成最终实例。
                # 因此当 mask 数量为 0 时，最应该优先检查的就是 cellprob 和 gradXY。
                if mask_count == 0:
                    log("警告：当前图像没有生成任何实例。建议优先查看导出的 cellprob 与 gradXY，确认是否是阈值过严或图像域偏差导致。")

                if SAVE_IMAGE:
                    image_stub = get_output_stub(image_path, output_dirs["images"])
                    image_output_path = save_image_file(original_image, image_stub)
                    log(f"原图副本已保存：{image_output_path}")

                if SAVE_FLOW_RGB:
                    flow_rgb_stub = get_output_stub(image_path, output_dirs["flow_rgb"])
                    flow_rgb_path = save_flow_rgb_file(flows[0], flow_rgb_stub)
                    log(f"flow RGB 已保存：{flow_rgb_path}")

                if SAVE_GRAD_XY:
                    grad_xy_stub = get_output_stub(image_path, output_dirs["grad_xy"])
                    grad_xy_path = save_grad_xy_file(flows[1], grad_xy_stub)
                    log(f"gradXY 已保存：{grad_xy_path}")

                if SAVE_CELLPROB:
                    cellprob_stub = get_output_stub(image_path, output_dirs["cellprob"])
                    cellprob_raw_path, cellprob_preview_path = save_cellprob_file(
                        cellprob, cellprob_stub)
                    log(f"cellprob 原始图已保存：{cellprob_raw_path}")
                    log(f"cellprob 预览图已保存：{cellprob_preview_path}")

                if SAVE_CELLPROB_BINARY_MASK and stage_results is not None:
                    cellprob_binary_stub = get_output_stub(image_path,
                                                           output_dirs["cellprob_binary"])
                    cellprob_binary_path = save_cellprob_binary_mask_file(
                        stage_results["cellprob_binary_mask"], cellprob_binary_stub)
                    log(f"cellprob 二值候选图已保存：{cellprob_binary_path}")

                if SAVE_INSTANCE_MASK:
                    instance_stub = get_output_stub(image_path, output_dirs["instance_masks"])
                    instance_path = save_instance_mask_file(masks, instance_stub)
                    log(f"实例 ID mask 已保存：{instance_path}")

                if SAVE_INSTANCE_MASK_PREVIEW:
                    instance_preview_stub = get_output_stub(
                        image_path, output_dirs["instance_mask_previews"])
                    instance_preview_path = save_instance_mask_preview_file(
                        masks, instance_preview_stub)
                    log(f"实例 mask 彩色预览图已保存：{instance_preview_path}")

                if SAVE_SEMANTIC_MASK:
                    semantic_stub = get_output_stub(image_path, output_dirs["semantic_masks"])
                    semantic_path = save_semantic_mask_file(masks, semantic_stub)
                    log(f"语义分割 mask 已保存：{semantic_path}")

                if SAVE_SEPARATED_SEMANTIC_MASK:
                    separated_stub = get_output_stub(image_path,
                                                     output_dirs["separated_semantic_masks"])
                    separated_path = save_separated_semantic_mask_file(
                        masks, outline_mask, separated_stub)
                    log(f"不粘连语义 mask 已保存：{separated_path}")

                if SAVE_OVERLAY:
                    overlay_stub = get_output_stub(image_path, output_dirs["overlays"])
                    overlay_path = save_overlay_file(model_input, masks, overlay_stub)
                    log(f"叠加可视化图已保存：{overlay_path}")

                if SAVE_RAW_DYNAMICS_MASK and stage_results is not None:
                    raw_dynamics_stub = get_output_stub(
                        image_path, output_dirs["raw_dynamics_instance_masks"])
                    raw_dynamics_path = save_instance_mask_file(
                        stage_results["raw_dynamics_masks"], raw_dynamics_stub)
                    log(f"raw dynamics 实例 mask 已保存：{raw_dynamics_path}")

                if SAVE_RAW_DYNAMICS_SEMANTIC_MASK and stage_results is not None:
                    raw_dynamics_semantic_stub = get_output_stub(
                        image_path, output_dirs["raw_dynamics_semantic_masks"])
                    raw_dynamics_semantic_path = save_semantic_mask_file(
                        stage_results["raw_dynamics_masks"], raw_dynamics_semantic_stub)
                    log(f"raw dynamics 语义 mask 已保存：{raw_dynamics_semantic_path}")

                if SAVE_RAW_DYNAMICS_OVERLAY and stage_results is not None:
                    raw_dynamics_overlay_stub = get_output_stub(
                        image_path, output_dirs["raw_dynamics_overlays"])
                    raw_dynamics_overlay_path = save_overlay_file(
                        model_input, stage_results["raw_dynamics_masks"],
                        raw_dynamics_overlay_stub)
                    log(f"raw dynamics 叠加图已保存：{raw_dynamics_overlay_path}")

                if SAVE_AFTER_MAXSIZE_MASK and stage_results is not None:
                    after_maxsize_stub = get_output_stub(
                        image_path, output_dirs["after_maxsize_instance_masks"])
                    after_maxsize_path = save_instance_mask_file(
                        stage_results["after_maxsize_masks"], after_maxsize_stub)
                    log(f"after max_size 实例 mask 已保存：{after_maxsize_path}")

                if SAVE_AFTER_MAXSIZE_OVERLAY and stage_results is not None:
                    after_maxsize_overlay_stub = get_output_stub(
                        image_path, output_dirs["after_maxsize_overlays"])
                    after_maxsize_overlay_path = save_overlay_file(
                        model_input, stage_results["after_maxsize_masks"],
                        after_maxsize_overlay_stub)
                    log(f"after max_size 叠加图已保存：{after_maxsize_overlay_path}")

                if SAVE_AFTER_FLOWQC_MASK and stage_results is not None:
                    after_flowqc_stub = get_output_stub(
                        image_path, output_dirs["after_flowqc_instance_masks"])
                    after_flowqc_path = save_instance_mask_file(
                        stage_results["after_flowqc_masks"], after_flowqc_stub)
                    log(f"after flow_qc 实例 mask 已保存：{after_flowqc_path}")

                if SAVE_AFTER_FLOWQC_OVERLAY and stage_results is not None:
                    after_flowqc_overlay_stub = get_output_stub(
                        image_path, output_dirs["after_flowqc_overlays"])
                    after_flowqc_overlay_path = save_overlay_file(
                        model_input, stage_results["after_flowqc_masks"],
                        after_flowqc_overlay_stub)
                    log(f"after flow_qc 叠加图已保存：{after_flowqc_overlay_path}")

                if SAVE_OUTLINE_MASK:
                    outline_stub = get_output_stub(image_path, output_dirs["outline_masks"])
                    outline_path = save_outline_mask_file(outline_mask, outline_stub)
                    log(f"轮廓线 mask 已保存：{outline_path}")

                if SAVE_OUTLINE_OVERLAY:
                    outline_overlay_stub = get_output_stub(image_path,
                                                           output_dirs["outline_overlays"])
                    outline_overlay_path = save_outline_overlay_file(
                        original_image, outline_mask, outline_overlay_stub)
                    log(f"轮廓线叠加图已保存：{outline_overlay_path}")

                if SAVE_POLYGON_TXT:
                    polygon_stub = get_output_stub(image_path, output_dirs["polygon_txt"])
                    polygon_records = build_polygon_records(
                        masks, POLYGON_APPROX_EPSILON_RATIO)
                    polygon_txt_path = save_polygon_txt_file(polygon_records, polygon_stub)
                    log(f"多边形轮廓文本已保存：{polygon_txt_path}")

                if SAVE_DEBUG_SUMMARY and stage_results is not None:
                    debug_report_stub = get_output_stub(image_path,
                                                        output_dirs["debug_reports"])
                    debug_report_path = save_debug_summary_file(
                        debug_report_stub, image_path, dP, cellprob, stage_results, masks)
                    log(f"逐阶段调试报告已保存：{debug_report_path}")

                if SAVE_SEG_NPY:
                    seg_path = save_seg_npy_file(model_input, masks, flows, image_path,
                                                 output_dirs["seg_npy"])
                    log(f"GUI 兼容的 _seg.npy 已保存：{seg_path}")

                if SAVE_ROIS:
                    roi_path = save_roi_zip_file(masks, image_path, output_dirs["rois"])
                    log(f"ImageJ ROI 压缩包已保存：{roi_path}")

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
