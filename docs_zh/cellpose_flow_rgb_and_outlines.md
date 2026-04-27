# `flow_rgb`、`outlines on`、最终 mask 的关系说明

## 1. 先说结论

你现在看到的这几类结果，来源并不一样：

1. `flow_rgb`
   这是网络输出流场 `gradXY` 的彩色可视化图，只是为了让人看流场方向和局部结构，不是最终标签。
2. `outlines on`
   这是根据最终实例 mask 再提取出来的轮廓线，属于后处理结果，不是网络直接输出。
3. `semantic_masks`
   这是把最终实例 mask 直接二值化后的前景图，所以相邻细胞会粘在一起，看起来会比 `flow_rgb` 少很多“独立小块”。
4. `separated_semantic_masks`
   这是在语义前景的基础上，把轮廓线区域挖空后的结果。它更接近你说的“每个细胞区域本身不互相粘连”的训练标签。

所以：

- 不建议直接把 `flow_rgb` 当作 Unet 的训练 mask
- 更合理的做法是从最终实例 mask 派生出：
  - `semantic_masks`
  - `separated_semantic_masks`
  - `outline_masks`
- 如果你确实想要“更接近 flow_rgb 里肉眼看到的更多细胞”，优先看新导出的 `raw_dynamics_semantic_masks`，而不是直接对 `flow_rgb` 本身做阈值

---

## 2. 为什么 `flow_rgb` 里看起来细胞很多，但 `overlay` 或 `semantic_masks` 里明显变少

这是因为 Cellpose-SAM 不会把 `flow_rgb` 直接当结果，而是还会继续做一整套后处理。

大致流程是：

1. 网络输出 `gradXY`
2. 网络输出 `cellprob`
3. 先用 `cellprob_threshold` 筛像素
4. 让像素沿着流场做动力学积分
5. 根据像素最终汇聚位置生成实例
6. 再做流场一致性过滤 `flow_threshold`
7. 再做大目标过滤 `max_size_fraction`
8. 再做小目标过滤 `min_size`
9. 最后得到最终实例 mask

也就是说，`flow_rgb` 只是“候选结构的方向图”，而不是“最后保留下来的实例标签”。

你当前这张图我实际做了一次单图诊断：

- 图像：`E:\Tianlu\data_raw\0208_3_03_6_seg.png`
- `cellprob > 0` 的像素数：`248948`
- `cellprob > 0` 的连通域数：`19`
- 不做 `flow QC`、不做 `min_size` 时的实例数：`263`
- 做了 `flow_threshold = 0.2` 和 `min_size = 5` 后的实例数：`190`

这说明中间确实有进一步过滤，不是只有你手动设置的两个阈值那么简单。

---

## 3. 为什么 `semantic_masks` 看起来细胞比 `flow_rgb` 少很多

这通常不是“实例真的少了”，而是因为表达方式变了。

`semantic_masks` 的逻辑是：

- 只要像素属于任意细胞实例，就统一记为前景 `255`
- 不再保留“第几个细胞”的编号

这样一来，相邻细胞如果彼此贴边，在二值图里就会连成一整片白色区域。

因此你看到的现象往往是：

- `instance_masks` 里其实还有很多实例编号
- 但 `semantic_masks` 视觉上会变成少量大块区域

这不是标签丢了，而是实例信息在二值化时被合并了。

---

## 4. 对 Unet 训练来说，哪一种 mask 更合适

### 情况 A：你做普通二分类语义分割

推荐先用：

- `semantic_masks`

含义是：

- 前景：细胞
- 背景：非细胞

这是最直接、最容易接你现有训练代码的。

### 情况 B：你希望相邻细胞不要粘在一起

推荐优先试：

- `separated_semantic_masks`

它的逻辑是：

- 细胞内部：`255`
- 背景和边界：`0`

这样相邻细胞之间会保留一条黑色分界线，更适合学习“单个细胞内部区域”。

### 情况 C：你后续还想做边界监督

可以同时使用：

- `separated_semantic_masks`
- `outline_masks`

一种常见做法是：

- 主输出学细胞内部
- 辅助输出学边界

---

## 5. GUI 的 `outlines on` 到底是什么

GUI 里的 `outlines on` 不是直接显示网络原始输出，而是基于最终实例 mask 再提取轮廓。

这条链路在源码里很清楚：

1. `cellpose.models.CellposeModel.eval(...)` 先返回最终 `masks` 和 `flows`
2. GUI 保存 outline 时，会调用 `cellpose.gui.io` 里的 `outlines_list(parent.cellpix[0])`
3. `outlines_list(...)` / `masks_to_outlines(...)` 再从最终实例 mask 里提取轮廓点

所以二者关系是：

- `flow_rgb`：流场可视化
- `outlines on`：最终实例轮廓

它们不是同一个阶段的输出。

---

## 6. 现在脚本里是怎么实现“紧密相连轮廓”和“多边形轮廓”的

当前脚本 [run_cpsam_export.py](/E:/cellpose/cellpose/tools/run_cpsam_export.py) 里已经补了这几步：

1. 先拿最终实例 `masks`
2. 用 `cellpose.utils.masks_to_outlines(masks)` 生成二值轮廓图
3. 用 `cellpose.utils.outlines_list(masks)` 提取每个实例的轮廓点
4. 再用 `cv2.approxPolyDP(...)` 把原始轮廓压缩成多边形顶点

所以现在会新增这些输出：

- `outline_masks/*.png`
- `outline_overlays/*.png`
- `separated_semantic_masks/*.png`
- `raw_dynamics_instance_masks/*.png`
- `raw_dynamics_semantic_masks/*.png`
- `debug_reports/*.txt`
- `polygon_txt/*.txt`

其中：

- `outline_masks` 最接近 GUI 的 `outlines on`
- `raw_dynamics_semantic_masks` 最接近“flow_rgb 里看起来很多、但尚未经过质量过滤”的候选标签
- `polygon_txt` 适合你看每个细胞的多边形坐标

---

## 7. 关键代码片段

下面这段逻辑就是核心：

```python
outline_mask = utils.masks_to_outlines(masks).astype(np.uint8) * 255
separated_mask = ((masks > 0) & (outline_mask == 0)).astype(np.uint8) * 255

outline_points_list = utils.outlines_list(masks, multiprocessing=False)
for outline_points in outline_points_list:
    contour_xy = np.asarray(outline_points, dtype=np.int32).reshape(-1, 1, 2)
    epsilon = max(cv2.arcLength(contour_xy, True) * POLYGON_APPROX_EPSILON_RATIO, 1.0)
    polygon_xy = cv2.approxPolyDP(contour_xy, epsilon, True).reshape(-1, 2)
```

含义分别是：

- `masks_to_outlines`：得到二值轮廓图
- `separated_mask`：得到不粘连的细胞内部区域
- `outlines_list`：拿到每个实例的轮廓点
- `approxPolyDP`：把轮廓点压缩成多边形

---

## 8. 你现在最推荐怎么用

如果你是为了训练部署到 `rk3568` 的 `Unet + MobileNetV3`，我建议这样试：

1. 第一版先用 `separated_semantic_masks` 训练
2. 抽查 `outline_overlays`，确认边界是否合理
3. 如果你觉得最终标签太保守，再对比 `raw_dynamics_semantic_masks`
4. 如果后续发现相邻细胞仍容易黏连，再把 `outline_masks` 加入辅助监督
5. `instance_masks` 和 `polygon_txt` 先保留，作为后续扩展使用

---

## 9. 对当前现象的直接回答

### 问题 1：`flow_rgb` 能不能转成更适合 Unet 的 mask

可以做“更适合 Unet 的 mask”，但不建议直接从 `flow_rgb` 本身去阈值化。

更推荐：

- 从最终实例 `masks` 生成 `separated_semantic_masks`

这样更稳定，也和 GUI/官方流程一致。

### 问题 2：为什么 `flow_rgb` 多、`overlay` 和 `semantic_masks` 少

因为 `flow_rgb` 只是流场可视化，最终还会经过：

- `cellprob_threshold`
- 动力学聚合
- `flow_threshold`
- `max_size_fraction`
- `min_size`

等步骤。

### 问题 3：`flow_rgb` 和 `outlines on` 是什么关系

不是一回事：

- `flow_rgb` 是网络原始流场的彩色展示
- `outlines on` 是最终实例 mask 的后处理轮廓

---

## 10. 新增输出目录对照

以 `0208_3_03_6_seg.png` 为例，你现在会看到：

```text
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\flow_rgb\0208_3_03_6_seg_flow_rgb.png
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\semantic_masks\0208_3_03_6_seg_semantic_mask.png
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\separated_semantic_masks\0208_3_03_6_seg_separated_semantic_mask.png
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\outline_masks\0208_3_03_6_seg_outline_mask.png
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\outline_overlays\0208_3_03_6_seg_outline_overlay.png
E:\Tianlu\data_raw\cpsam_export_thr0.2_size5\polygon_txt\0208_3_03_6_seg_polygons.txt
```

建议你先重点看这三个：

1. `separated_semantic_masks`
2. `outline_overlays`
3. `polygon_txt`

这三份最接近你现在要解决的“相邻细胞边界”和“训练标签表达方式”问题。
