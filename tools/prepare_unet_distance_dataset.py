"""
根据 Cellpose-SAM 导出结果，整理方案 B 训练数据集。

本版本支持：
1. 只给一个输入目录，脚本自动切分 train / val / test
2. 按“样本组”切分，而不是按单帧切分，避免同一样本的多帧图像泄漏到不同数据集
3. 同一组样本（例如 `0208_3_01_4__top01__...` / `0208_3_01_4__top03__...`）强制进入同一个 split

固定输出目录结构：
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
"""

from __future__ import annotations

import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# ==================== 顶部配置区：直接改这里即可运行 ====================

# Cellpose-SAM 导出根目录：
# 目录内应至少包含 images / semantic_masks / instance_masks
SOURCE_ROOT = Path(r"E:\Tianlu\cell\deeplearing_algorithm\second_process_enhanced\roi_batch_selected_top3\cpsam_export_0.8_0.0_size120")

# 目标训练工程的数据根目录：
# 脚本会在这里自动创建 train / val / test 的
# images、original_frames、masks、distance_maps、debug
TARGET_DATA_ROOT = Path(r"E:\Tianlu\cell\deeplearing_algorithm\third_process_segment\data")

# 是否清空旧的 train / val / test 目录后再重新生成：
# 推荐保持 True，避免你多次运行后旧文件残留造成混淆
CLEAR_EXISTING_SPLITS = True

# 是否拷贝原图到训练工程中
COPY_IMAGES = True

# 是否额外保存 selected_original_frames 中的原始帧：
# 推荐保持 True，这样后续你可以同时拿到：
# 1. Cellpose 推理时真正输入的 ROI 图（images）
# 2. 对应挑选出的原始帧图（original_frames）
COPY_SELECTED_ORIGINAL_FRAMES = True

# 是否保存调试图
SAVE_DEBUG = True

# 调试叠加图中二值 mask 的阈值，仅用于可视化
DEBUG_MASK_THRESHOLD = 0.5

# 随机种子：
# 为了保证每次切分结果稳定一致，默认固定一个随机种子
RANDOM_SEED = 20260427

# 组级别切分比例：
# 当前默认使用 7:2:1，更适合你现在这种“样本组数还不算特别大”的情况。
# 如果后续样本组明显变多（例如 >100 组），也可以改成 [8, 1, 1]。
SPLIT_RATIO = [7, 2, 1]  # [train, val, test]

# 样本组提取规则：
# auto：优先按 "__top" 之前的前缀分组，例如
#       0208_3_01_4__top01__8_0_endothelial_cells_0.79_image.tif
#       -> 组键 0208_3_01_4__
# manual_prefix：按 GROUP_PREFIX_SEPARATOR 手动切分
GROUP_KEY_MODE = "auto"
GROUP_PREFIX_SEPARATOR = "__top"


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def ensure_split_dirs(target_data_root: Path, split_name: str) -> Dict[str, Path]:
    split_root = target_data_root / split_name
    output_dirs = {
        "root": split_root,
        "images": split_root / "images",
        "original_frames": split_root / "original_frames",
        "masks": split_root / "masks",
        "distance_maps": split_root / "distance_maps",
        "debug": split_root / "debug",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def clear_existing_split_dirs(target_data_root: Path) -> None:
    """
    清理旧的 train / val / test 目录，避免重复运行后旧文件残留。
    """
    if not CLEAR_EXISTING_SPLITS:
        return

    for split_name in ["train", "val", "test"]:
        split_root = target_data_root / split_name
        if split_root.exists():
            shutil.rmtree(split_root)
            log(f"已清空旧目录: {split_root}")


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def to_bgr_display(image: np.ndarray) -> np.ndarray:
    image_uint8 = normalize_to_uint8(image)
    if image_uint8.ndim == 2:
        return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 1:
        return cv2.cvtColor(image_uint8[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
        return image_uint8
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 4:
        return cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"不支持的图像形状: {image.shape}")


def render_distance_preview(distance_map: np.ndarray) -> np.ndarray:
    distance_uint8 = (np.clip(distance_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(distance_uint8, cv2.COLORMAP_JET)


def blend_mask_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    color_bgr=(0, 255, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    base = image_bgr.astype(np.float32)
    mask = (binary_mask > 0).astype(np.float32)[..., np.newaxis]
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - alpha * mask) + color * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def derive_sample_key_from_image_name(image_name: str) -> str:
    """
    从导出图像名中提取 sample key，用于找到对应的 semantic / instance 标签。
    """
    stem = Path(image_name).stem
    if stem.endswith("_image"):
        return stem[:-6]
    return stem


def derive_group_key_from_image_name(image_name: str) -> str:
    """
    从图像名中提取“同一原始样本”的分组键。

    例子：
    - `0208_3_01_4__top01__8_0_endothelial_cells_0.79_image.tif`
      -> `0208_3_01_4__`
    - `0208_3_01_4__top03__15_0_endothelial_cells_0.78_image.tif`
      -> `0208_3_01_4__`

    这样可以确保同一个样本的多帧图像始终进入同一个 split，避免数据集污染。
    """
    stem = Path(image_name).stem
    if stem.endswith("_image"):
        stem = stem[:-6]

    if GROUP_KEY_MODE == "manual_prefix":
        if GROUP_PREFIX_SEPARATOR in stem:
            return stem.split(GROUP_PREFIX_SEPARATOR)[0] + GROUP_PREFIX_SEPARATOR.replace("top", "")
        return stem

    # auto 模式：
    # 1. 优先识别 "__top"
    # 2. 识别失败时，再尝试按第一个双下划线之前切分
    if "__top" in stem:
        return stem.split("__top", 1)[0] + "__"
    if "__" in stem:
        return stem.split("__", 1)[0] + "__"
    return stem


def find_file_with_suffix(root_dir: Path, base_name: str, suffix_candidates: List[str]) -> Optional[Path]:
    for suffix in suffix_candidates:
        candidate = root_dir / f"{base_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def resolve_selected_original_frames_dir(source_root: Path) -> Optional[Path]:
    """
    查找 selected_original_frames 目录。

    兼容两种常见结构：
    1. SOURCE_ROOT / selected_original_frames
    2. SOURCE_ROOT.parent / selected_original_frames
    """
    candidates = [
        source_root / "selected_original_frames",
        source_root.parent / "selected_original_frames",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def collect_samples(source_root: Path) -> List[Dict[str, object]]:
    """
    收集样本，并为每个样本附带 group_key。
    """
    images_dir = source_root / "images"
    semantic_masks_dir = source_root / "semantic_masks"
    instance_masks_dir = source_root / "instance_masks"
    selected_original_frames_dir = resolve_selected_original_frames_dir(source_root)

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到 images 目录: {images_dir}")
    if not semantic_masks_dir.exists():
        raise FileNotFoundError(f"未找到 semantic_masks 目录: {semantic_masks_dir}")
    if not instance_masks_dir.exists():
        raise FileNotFoundError(f"未找到 instance_masks 目录: {instance_masks_dir}")

    samples: List[Dict[str, object]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        sample_key = derive_sample_key_from_image_name(image_path.name)
        group_key = derive_group_key_from_image_name(image_path.name)
        semantic_path = find_file_with_suffix(semantic_masks_dir, f"{sample_key}_semantic_mask", [".png", ".tif", ".tiff"])
        instance_path = find_file_with_suffix(instance_masks_dir, f"{sample_key}_instance_mask", [".png", ".tif", ".tiff"])
        original_frame_path = None
        if selected_original_frames_dir is not None:
            original_frame_path = find_file_with_suffix(
                selected_original_frames_dir,
                sample_key,
                [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
            )

        if semantic_path is None or instance_path is None:
            log(f"跳过样本，缺少 semantic 或 instance 标签: {image_path.name}")
            continue

        samples.append(
            {
                "image": image_path,
                "semantic": semantic_path,
                "instance": instance_path,
                "original_frame": original_frame_path,
                "sample_key": sample_key,
                "group_key": group_key,
            }
        )

    return samples


def split_group_keys(group_keys: List[str], split_ratio: List[int], random_seed: int) -> Dict[str, List[str]]:
    """
    按组键切分 train / val / test。

    关键点：
    1. 先对 group 做随机打乱
    2. 再按比例切分
    3. 如果组数足够，会尽量保证 val / test 至少各 1 组
    """
    if len(split_ratio) != 3:
        raise ValueError("SPLIT_RATIO 必须是长度为 3 的列表，例如 [7, 2, 1]")
    if any(value < 0 for value in split_ratio):
        raise ValueError("SPLIT_RATIO 中不能出现负数")

    unique_groups = sorted(set(group_keys))
    if not unique_groups:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(random_seed)
    rng.shuffle(unique_groups)

    total_groups = len(unique_groups)
    ratio_sum = sum(split_ratio)
    train_ratio, val_ratio, test_ratio = split_ratio

    train_count = int(round(total_groups * train_ratio / ratio_sum))
    val_count = int(round(total_groups * val_ratio / ratio_sum))
    test_count = total_groups - train_count - val_count

    # 组数足够时，尽量保证验证集和测试集各至少 1 组
    if total_groups >= 3:
        if val_count <= 0:
            val_count = 1
            train_count = max(train_count - 1, 1)
        if test_count <= 0:
            test_count = 1
            train_count = max(train_count - 1, 1)

    # 修正因 round 带来的越界
    while train_count + val_count + test_count > total_groups:
        if train_count >= max(val_count, test_count) and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < total_groups:
        train_count += 1

    train_groups = unique_groups[:train_count]
    val_groups = unique_groups[train_count:train_count + val_count]
    test_groups = unique_groups[train_count + val_count:]

    return {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }


def build_distance_map_from_instance_mask(instance_mask: np.ndarray) -> np.ndarray:
    """
    逐实例计算欧氏距离变换，并在每个实例内部归一化到 [0, 1]。
    """
    if instance_mask.ndim != 2:
        raise ValueError(f"instance_mask 必须是 2D，当前为 {instance_mask.shape}")

    distance_map = np.zeros(instance_mask.shape, dtype=np.float32)
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]

    for instance_id in instance_ids:
        instance_binary = (instance_mask == instance_id).astype(np.uint8)
        if int(instance_binary.sum()) == 0:
            continue

        distance = cv2.distanceTransform(instance_binary, distanceType=cv2.DIST_L2, maskSize=5)
        max_value = float(distance.max())
        if max_value > 0:
            distance = distance / max_value

        distance_map[instance_binary > 0] = distance[instance_binary > 0]

    return np.clip(distance_map, 0.0, 1.0).astype(np.float32)


def save_float_tiff(save_path: Path, array: np.ndarray) -> None:
    ok = cv2.imwrite(str(save_path), array.astype(np.float32))
    if not ok:
        raise RuntimeError(f"保存浮点 tif 失败: {save_path}")


def process_single_sample(sample: Dict[str, object], output_dirs: Dict[str, Path]) -> Dict[str, object]:
    image_path = sample["image"]
    semantic_path = sample["semantic"]
    instance_path = sample["instance"]
    original_frame_path = sample["original_frame"]

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    semantic_mask = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
    instance_mask = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"无法读取原图: {image_path}")
    if semantic_mask is None:
        raise ValueError(f"无法读取 semantic mask: {semantic_path}")
    if instance_mask is None:
        raise ValueError(f"无法读取 instance mask: {instance_path}")

    if instance_mask.ndim == 3:
        instance_mask = instance_mask[:, :, 0]

    semantic_mask = (semantic_mask > 127).astype(np.uint8) * 255

    if image.shape[:2] != semantic_mask.shape[:2] or image.shape[:2] != instance_mask.shape[:2]:
        raise ValueError(
            f"尺寸不一致: image={image.shape[:2]}, semantic={semantic_mask.shape[:2]}, instance={instance_mask.shape[:2]}"
        )

    distance_map = build_distance_map_from_instance_mask(instance_mask.astype(np.int32))

    output_stem = Path(image_path).stem
    target_image_path = output_dirs["images"] / image_path.name
    target_original_frame_path = None
    target_mask_path = output_dirs["masks"] / f"{output_stem}.png"
    target_distance_path = output_dirs["distance_maps"] / f"{output_stem}.tif"

    if COPY_IMAGES:
        shutil.copy2(str(image_path), str(target_image_path))

    if COPY_SELECTED_ORIGINAL_FRAMES and original_frame_path is not None:
        target_original_frame_path = output_dirs["original_frames"] / Path(original_frame_path).name
        shutil.copy2(str(original_frame_path), str(target_original_frame_path))

    cv2.imwrite(str(target_mask_path), semantic_mask)
    save_float_tiff(target_distance_path, distance_map)

    if SAVE_DEBUG:
        image_bgr = to_bgr_display(image)
        distance_preview = render_distance_preview(distance_map)
        overlay = blend_mask_overlay(image_bgr, semantic_mask > int(DEBUG_MASK_THRESHOLD * 255))
        cv2.imwrite(str(output_dirs["debug"] / f"{output_stem}_distance_preview.png"), distance_preview)
        cv2.imwrite(str(output_dirs["debug"] / f"{output_stem}_mask_overlay.png"), overlay)

    instance_count = int(np.max(instance_mask))
    foreground_pixels = int(np.count_nonzero(semantic_mask))
    return {
        "image_name": image_path.name,
        "original_frame_name": Path(original_frame_path).name if original_frame_path is not None else None,
        "sample_key": sample["sample_key"],
        "group_key": sample["group_key"],
        "shape_hw": list(image.shape[:2]),
        "instance_count": instance_count,
        "foreground_pixels": foreground_pixels,
        "distance_min": float(distance_map.min()),
        "distance_max": float(distance_map.max()),
        "saved_original_frame": target_original_frame_path is not None,
    }


def summarize_groups(samples: List[Dict[str, object]]) -> Dict[str, int]:
    group_counter: Dict[str, int] = defaultdict(int)
    for sample in samples:
        group_counter[str(sample["group_key"])] += 1
    return dict(sorted(group_counter.items(), key=lambda item: item[0]))


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"输入目录不存在: {SOURCE_ROOT}")

    TARGET_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    clear_existing_split_dirs(TARGET_DATA_ROOT)

    log(f"开始扫描输入目录: {SOURCE_ROOT}")
    original_frames_dir = resolve_selected_original_frames_dir(SOURCE_ROOT)
    if COPY_SELECTED_ORIGINAL_FRAMES:
        if original_frames_dir is not None:
            log(f"已找到原始帧目录: {original_frames_dir}")
        else:
            log("未找到 selected_original_frames 目录，本次不会额外保存 original_frames")

    all_samples = collect_samples(SOURCE_ROOT)
    if not all_samples:
        raise RuntimeError(f"未在目录中找到可用样本: {SOURCE_ROOT}")

    group_summary = summarize_groups(all_samples)
    split_group_map = split_group_keys(
        group_keys=list(group_summary.keys()),
        split_ratio=SPLIT_RATIO,
        random_seed=RANDOM_SEED,
    )

    log(f"共找到 {len(all_samples)} 帧样本，属于 {len(group_summary)} 个样本组")
    log(
        f"当前切分比例为 train:val:test = {SPLIT_RATIO[0]}:{SPLIT_RATIO[1]}:{SPLIT_RATIO[2]} | "
        f"随机种子 = {RANDOM_SEED}"
    )
    log(
        f"组数分配 -> train={len(split_group_map['train'])}, "
        f"val={len(split_group_map['val'])}, test={len(split_group_map['test'])}"
    )

    manifest = {
        "source_root": str(SOURCE_ROOT),
        "target_data_root": str(TARGET_DATA_ROOT),
        "split_ratio": SPLIT_RATIO,
        "random_seed": RANDOM_SEED,
        "group_key_mode": GROUP_KEY_MODE,
        "group_prefix_separator": GROUP_PREFIX_SEPARATOR,
        "num_total_frames": len(all_samples),
        "num_total_groups": len(group_summary),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        allowed_groups = set(split_group_map[split_name])
        split_samples = [sample for sample in all_samples if sample["group_key"] in allowed_groups]
        output_dirs = ensure_split_dirs(TARGET_DATA_ROOT, split_name)

        split_records = []
        for sample in split_samples:
            record = process_single_sample(sample, output_dirs)
            split_records.append(record)
            log(
                f"已生成 {split_name} 样本: {record['image_name']} | "
                f"group={record['group_key']} | 实例数={record['instance_count']}"
            )

        manifest["splits"][split_name] = {
            "num_frames": len(split_records),
            "num_groups": len(allowed_groups),
            "group_keys": sorted(allowed_groups),
            "records": split_records,
        }
        log(
            f"split={split_name} 处理完成 | "
            f"frames={len(split_records)} | groups={len(allowed_groups)}"
        )

    manifest_path = TARGET_DATA_ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"全部完成，数据清单已保存: {manifest_path}")


if __name__ == "__main__":
    main()
