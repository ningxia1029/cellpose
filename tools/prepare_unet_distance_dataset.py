"""
根据 Cellpose-SAM 导出结果，整理方案 B 训练数据集。

固定输出目录结构：
data/
  train/
    images/
    masks/
    distance_maps/
    debug/
  val/
  test/

标签规则：
1. `masks/` 直接来自 `semantic_masks`
2. `distance_maps/` 由 `instance_masks` 逐实例计算距离变换后生成
3. `debug/` 保存伪彩色距离图和叠加图，便于抽查
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# ==================== 顶部配置区：直接改这里即可运行 ====================

# Cellpose-SAM 导出根目录：
# 目录内应至少包含 images / semantic_masks / instance_masks
TRAIN_SOURCE_ROOT = Path(r"E:\Tianlu\data_raw\cpsam_export_thr0.2_size5")
VAL_SOURCE_ROOT = Path(r"")
TEST_SOURCE_ROOT = Path(r"")

# 目标训练工程的数据根目录：
# 脚本会在这里自动创建 train / val / test 的 images、masks、distance_maps、debug
TARGET_DATA_ROOT = Path(r"E:\Tianlu\cell\deeplearing_algorithm\third_process_segment\data")

# 是否拷贝原图到训练工程中
COPY_IMAGES = True

# 是否保存调试图
SAVE_DEBUG = True

# 调试叠加图中二值 mask 的阈值，仅用于可视化
DEBUG_MASK_THRESHOLD = 0.5


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def build_split_configs() -> List[Dict[str, Path]]:
    split_configs: List[Dict[str, Path]] = []

    if str(TRAIN_SOURCE_ROOT).strip() not in {"", "."}:
        split_configs.append({"name": "train", "source_root": TRAIN_SOURCE_ROOT})
    if str(VAL_SOURCE_ROOT).strip() not in {"", "."}:
        split_configs.append({"name": "val", "source_root": VAL_SOURCE_ROOT})
    if str(TEST_SOURCE_ROOT).strip() not in {"", "."}:
        split_configs.append({"name": "test", "source_root": TEST_SOURCE_ROOT})

    if not split_configs:
        raise ValueError("请至少填写一个有效的 TRAIN_SOURCE_ROOT / VAL_SOURCE_ROOT / TEST_SOURCE_ROOT")
    return split_configs


def ensure_split_dirs(target_data_root: Path, split_name: str) -> Dict[str, Path]:
    split_root = target_data_root / split_name
    output_dirs = {
        "root": split_root,
        "images": split_root / "images",
        "masks": split_root / "masks",
        "distance_maps": split_root / "distance_maps",
        "debug": split_root / "debug",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


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
    stem = Path(image_name).stem
    if stem.endswith("_image"):
        return stem[:-6]
    return stem


def find_file_with_suffix(root_dir: Path, base_name: str, suffix_candidates: List[str]) -> Optional[Path]:
    for suffix in suffix_candidates:
        candidate = root_dir / f"{base_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(source_root: Path) -> List[Dict[str, Path]]:
    images_dir = source_root / "images"
    semantic_masks_dir = source_root / "semantic_masks"
    instance_masks_dir = source_root / "instance_masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到 images 目录: {images_dir}")
    if not semantic_masks_dir.exists():
        raise FileNotFoundError(f"未找到 semantic_masks 目录: {semantic_masks_dir}")
    if not instance_masks_dir.exists():
        raise FileNotFoundError(f"未找到 instance_masks 目录: {instance_masks_dir}")

    samples: List[Dict[str, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        sample_key = derive_sample_key_from_image_name(image_path.name)
        semantic_path = find_file_with_suffix(semantic_masks_dir, f"{sample_key}_semantic_mask", [".png", ".tif", ".tiff"])
        instance_path = find_file_with_suffix(instance_masks_dir, f"{sample_key}_instance_mask", [".png", ".tif", ".tiff"])

        if semantic_path is None or instance_path is None:
            log(f"跳过样本，缺少 semantic 或 instance 标签: {image_path.name}")
            continue

        samples.append(
            {
                "image": image_path,
                "semantic": semantic_path,
                "instance": instance_path,
            }
        )

    return samples


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


def process_single_sample(sample: Dict[str, Path], output_dirs: Dict[str, Path]) -> Dict[str, object]:
    image_path = sample["image"]
    semantic_path = sample["semantic"]
    instance_path = sample["instance"]

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

    output_stem = image_path.stem
    target_image_path = output_dirs["images"] / image_path.name
    target_mask_path = output_dirs["masks"] / f"{output_stem}.png"
    target_distance_path = output_dirs["distance_maps"] / f"{output_stem}.tif"

    if COPY_IMAGES:
        shutil.copy2(str(image_path), str(target_image_path))

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
        "shape_hw": list(image.shape[:2]),
        "instance_count": instance_count,
        "foreground_pixels": foreground_pixels,
        "distance_min": float(distance_map.min()),
        "distance_max": float(distance_map.max()),
    }


def main() -> None:
    split_configs = build_split_configs()
    TARGET_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {"target_data_root": str(TARGET_DATA_ROOT), "splits": {}}

    for split_cfg in split_configs:
        split_name = split_cfg["name"]
        source_root = split_cfg["source_root"]
        log(f"开始处理 split={split_name} | 来源目录: {source_root}")

        if not source_root.exists():
            log(f"来源目录不存在，跳过 split={split_name}: {source_root}")
            continue

        output_dirs = ensure_split_dirs(TARGET_DATA_ROOT, split_name)
        samples = collect_samples(source_root)
        if not samples:
            log(f"split={split_name} 未找到可用样本，跳过")
            continue

        split_records = []
        for sample in samples:
            record = process_single_sample(sample, output_dirs)
            split_records.append(record)
            log(
                f"已生成 {split_name} 样本: {record['image_name']} | "
                f"尺寸={record['shape_hw']} | 实例数={record['instance_count']}"
            )

        manifest["splits"][split_name] = {
            "source_root": str(source_root),
            "num_samples": len(split_records),
            "records": split_records,
        }
        log(f"split={split_name} 处理完成，共 {len(split_records)} 个样本")

    manifest_path = TARGET_DATA_ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"全部完成，数据清单已保存: {manifest_path}")


if __name__ == "__main__":
    main()
