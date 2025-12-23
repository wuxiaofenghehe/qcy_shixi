import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from tqdm import tqdm

class YOLOAugmentation:
    """
    同时支持 YOLO 分割（polygon）与 YOLO 检测（bbox）的数据增强工具
    usage:
        task='segmentation' 或 'detection'
    """
    def __init__(self, source_images_dir, source_labels_dir,
                 output_images_dir, output_labels_dir,
                 task='segmentation'):
        self.source_images_dir = Path(source_images_dir)
        self.source_labels_dir = Path(source_labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        self.task = task.lower()
        assert self.task in ('segmentation', 'detection'), \
            "task 只能是 'segmentation' 或 'detection'"

        # 创建输出目录
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 统一读写入口 --------------------
    def read_yolo_label(self, label_path):
        return (self.read_yolo_segmentation(label_path) if self.task == 'segmentation'
                else self.read_yolo_detection(label_path))

    def save_yolo_label(self, label_path, data):
        (self.save_yolo_segmentation(label_path, data) if self.task == 'segmentation'
         else self.save_yolo_detection(label_path, data))

    # -------------------- 分割格式 --------------------
    def read_yolo_segmentation(self, label_path):
        polygons = []
        if not os.path.exists(label_path):
            return polygons
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                polygons.append({'class_id': int(parts[0]),
                                 'coords': list(map(float, parts[1:]))})
        return polygons

    def save_yolo_segmentation(self, label_path, polygons):
        with open(label_path, 'w') as f:
            for p in polygons:
                coords_str = ' '.join([f'{c:.6f}' for c in p['coords']])
                f.write(f"{p['class_id']} {coords_str}\n")

    # -------------------- 检测格式 --------------------
    def read_yolo_detection(self, label_path):
        bboxes = []
        if not os.path.exists(label_path):
            return bboxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                bboxes.append({'class_id': class_id, 'bbox': bbox})
        return bboxes

    def save_yolo_detection(self, label_path, bboxes):
        with open(label_path, 'w') as f:
            for item in bboxes:
                class_id, (x_c, y_c, w, h) = item['class_id'], item['bbox']
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # -------------------- 增强方法 --------------------
    def random_mosaic_augmentation(self, image, num_patches=2, min_size=50, max_size=200,
                                   mosaic_cell_size=10, feather_radius=15):
        copper_color = [0, 26, 101]
        white_color = [122, 215, 241]
        light_copper_color = [0, 52, 123]
        copper_ratio = random.random()
        total_copper_cells = 18
        copper_count = int(total_copper_cells * copper_ratio)
        light_copper_count = total_copper_cells - copper_count
        colors = [light_copper_color] * light_copper_count + \
                 [copper_color] * copper_count + [white_color]

        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        if img is None:
            raise ValueError("无法读取图像")
        h, w = img.shape[:2]
        num_patches = max(1, num_patches)

        # 自适应参数
        if min(h, w) <= 160:
            adaptive_min_size = int(min(h, w) * 0.25)
            adaptive_max_size = int(min(h, w) * 0.5)
            adaptive_patch_w_min = 15
            adaptive_patch_w_max = 25
            adaptive_cell_size = max(1, mosaic_cell_size // 2)
            adaptive_feather = max(2, feather_radius // 2)
        else:
            adaptive_min_size = min_size
            adaptive_max_size = max_size
            adaptive_patch_w_min = 30
            adaptive_patch_w_max = 50
            adaptive_cell_size = mosaic_cell_size
            adaptive_feather = feather_radius

        for _ in range(num_patches):
            patch_h = random.randint(adaptive_min_size, adaptive_max_size)
            patch_w = random.randint(adaptive_patch_w_min, adaptive_patch_w_max)
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            x2, y2 = min(x + patch_w, w), min(y + patch_h, h)
            patch_w, patch_h = x2 - x, y2 - y

            mosaic_region = np.zeros_like(img[y:y2, x:x2])
            cy = 0
            while cy < patch_h:
                cell_h = random.randint(max(1, adaptive_cell_size // 2),
                                        max(1, adaptive_cell_size * 2))
                cell_h = min(cell_h, patch_h - cy)
                cx = 0
                while cx < patch_w:
                    cell_w = random.randint(max(1, adaptive_cell_size // 2),
                                            max(1, adaptive_cell_size * 2))
                    cell_w = min(cell_w, patch_w - cx)
                    color = random.choice(colors)
                    if color == copper_color or color == light_copper_color:
                        strip_h = max(1, cell_h // 3)
                        strip_y = cy + random.randint(0, cell_h - strip_h)
                        mosaic_region[strip_y:strip_y + strip_h,
                                      cx:cx + cell_w] = color
                        fill_color = light_copper_color if color == copper_color else copper_color
                        if strip_y > cy:
                            mosaic_region[cy:strip_y, cx:cx + cell_w] = fill_color
                        if strip_y + strip_h < cy + cell_h:
                            mosaic_region[strip_y + strip_h:cy + cell_h,
                                          cx:cx + cell_w] = fill_color
                    else:
                        mosaic_region[cy:cy + cell_h, cx:cx + cell_w] = color
                    cx += cell_w
                cy += cell_h

            # 羽化
            mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
            actual_feather_radius = min(adaptive_feather, min(patch_h, patch_w) // 3)
            if actual_feather_radius > 0:
                inner_top = actual_feather_radius
                inner_bottom = patch_h - actual_feather_radius
                inner_left = actual_feather_radius
                inner_right = patch_w - actual_feather_radius
                if inner_bottom > inner_top and inner_right > inner_left:
                    mask[inner_top:inner_bottom, inner_left:inner_right] = 255
            mask_inv = cv2.bitwise_not(mask)
            dist_transform = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
            if actual_feather_radius > 0:
                feather_mask = np.clip(1.0 - dist_transform / actual_feather_radius, 0, 1)
            else:
                feather_mask = np.ones((patch_h, patch_w), dtype=np.float32)
            if actual_feather_radius > 2:
                blur_size = actual_feather_radius // 2
                if blur_size % 2 == 0:
                    blur_size += 1
                feather_mask = cv2.GaussianBlur(feather_mask, (blur_size, blur_size), 0)
            feather_mask = np.clip(feather_mask, 0, 1)
            original_region = img[y:y2, x:x2].astype(np.float32)
            mosaic_region_float = mosaic_region.astype(np.float32)
            mask_3d = np.stack([feather_mask] * 3, axis=2)
            blended = (original_region * (1 - mask_3d) +
                       mosaic_region_float * mask_3d).astype(np.uint8)
            img[y:y2, x:x2] = blended
        return img

    def random_color_fill_augmentation_percentile(self, image, num_patches=3,
                                                  min_size=100, max_size=200,
                                                  surrounding_margin=5,
                                                  feather_radius=10, percentile=10):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        h, w = img.shape[:2]

        if min(h, w) <= 160:
            adaptive_min_size = int(min(h, w) * 0.2)
            adaptive_max_size = int(min(h, w) * 0.6)
            adaptive_num_patches = max(1, num_patches // 2)
        else:
            adaptive_min_size = min_size
            adaptive_max_size = min(max_size, int(min(h, w) * 0.8))
            adaptive_num_patches = num_patches

        for _ in range(adaptive_num_patches):
            patch_h = random.randint(adaptive_min_size, adaptive_max_size)
            patch_w = random.randint(int(adaptive_min_size * 0.4),
                                     int(adaptive_max_size * 0.6))
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))

            margin = surrounding_margin
            x1_outer = max(0, x - margin)
            y1_outer = max(0, y - margin)
            x2_outer = min(w, x + patch_w + margin)
            y2_outer = min(h, y + patch_h + margin)
            surrounding_mask = np.ones((y2_outer - y1_outer,
                                        x2_outer - x1_outer), dtype=bool)
            local_y1 = y - y1_outer
            local_y2 = local_y1 + patch_h
            local_x1 = x - x1_outer
            local_x2 = local_x1 + patch_w
            surrounding_mask[local_y1:local_y2, local_x1:local_x2] = False
            surrounding_region = img[y1_outer:y2_outer, x1_outer:x2_outer]
            if np.any(surrounding_mask):
                percentile_color = np.percentile(
                    surrounding_region[surrounding_mask], percentile, axis=0)
            else:
                percentile_color = np.percentile(img, percentile, axis=(0, 1))

            mask = np.zeros((patch_h, patch_w), dtype=np.float32)
            actual_feather_radius = min(feather_radius, min(patch_h, patch_w) // 4)
            if actual_feather_radius > 0:
                core_h = int(patch_h * 0.7)
                core_w = int(patch_w * 0.7)
                core_y = (patch_h - core_h) // 2
                core_x = (patch_w - core_w) // 2
                mask[core_y:core_y + core_h, core_x:core_x + core_w] = 1.0
                mask = cv2.GaussianBlur(mask, (0, 0), actual_feather_radius)
            mask = np.clip(mask, 0, 1)
            fill_region = np.full_like(img[y:y + patch_h, x:x + patch_w],
                                       percentile_color.astype(np.uint8))
            original_region = img[y:y + patch_h, x:x + patch_w]
            mask_3d = np.stack([mask] * 3, axis=2)
            blended = (original_region * (1 - mask_3d) +
                       fill_region * mask_3d).astype(np.uint8)
            img[y:y + patch_h, x:x + patch_w] = blended
        return img

    # -------------------- 单张处理 --------------------
    def augment_single_image(self, image_path, label_path, index):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        labels = self.read_yolo_label(label_path)

        original_name = image_path.stem
        original_ext = image_path.suffix

        # 原样复制
        shutil.copy(image_path,
                    self.output_images_dir / f"{original_name}{original_ext}")
        self.save_yolo_label(
            self.output_labels_dir / f"{original_name}.txt", labels)

        # 随机增强
        aug_method = random.choice(['mosaic', 'percentile'])
        try:
            if aug_method == 'mosaic':
                aug_img = self.random_mosaic_augmentation(
                    image,
                    num_patches=random.randint(2, 4),
                    min_size=40, max_size=100,
                    mosaic_cell_size=3, feather_radius=6)
            else:
                aug_img = self.random_color_fill_augmentation_percentile(
                    image,
                    num_patches=3, min_size=30, max_size=300,
                    surrounding_margin=6, feather_radius=15, percentile=15)

            aug_img_name = f"{original_name}_aug{original_ext}"
            cv2.imwrite(str(self.output_images_dir / aug_img_name), aug_img)
            self.save_yolo_label(
                self.output_labels_dir / f"{original_name}_aug.txt", labels)
        except Exception as e:
            print(f"{aug_method}增强 {image_path.name} 时出错: {e}")

    # -------------------- 主流程 --------------------
    def augment_dataset(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.source_images_dir.glob(f"*{ext}"))

        print(f"找到 {len(image_files)} 张图片")
        print(f"每张图片将随机选择一种增强方法")
        print(f"预计总共生成 {len(image_files) * 2} 张图片（原图+1增强）\n")

        for idx, image_path in enumerate(tqdm(image_files, desc="数据增强进度")):
            label_path = self.source_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"\n警告: 未找到标签文件 {label_path}")
                continue
            self.augment_single_image(image_path, label_path, idx)

        print(f"\n数据增强完成!")
        print(f"输出图片目录: {self.output_images_dir}")
        print(f"输出标签目录: {self.output_labels_dir}")


# -------------------- 主函数示例 --------------------
def main():
    # 1. 检测任务示例
    task = 'detection'
    # 2. 分割任务示例
    # task = 'segmentation'

    SOURCE_IMAGES_DIR = r"E:\qcy\new-data\new-data-20251125\20251125-split-data\classify-data\20251125-yolo-seg-data\images"
    SOURCE_LABELS_DIR = r"E:\qcy\new-data\new-data-20251125\20251125-split-data\classify-data\20251125-yolo-seg-data\labels"
    OUTPUT_IMAGES_DIR = r"E:\qcy\new-data\new-data-20251125\20251125-split-data\classify-data\20251125-yolo-seg-data\images\train-augment"
    OUTPUT_LABELS_DIR = r"E:\qcy\new-data\new-data-20251125\20251125-split-data\classify-data\20251125-yolo-seg-data\labels\train-augment"

    augmentor = YOLOAugmentation(
        source_images_dir=SOURCE_IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR,
        task=task
    )
    augmentor.augment_dataset()


if __name__ == "__main__":
    main()