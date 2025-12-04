import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from tqdm import tqdm

class YOLOSegmentationAugmentation:
    """YOLO格式分割数据集增强工具 - 使用自定义增强方法"""
    
    def __init__(self, source_images_dir, source_labels_dir, 
                 output_images_dir, output_labels_dir):
        """
        参数:
            source_images_dir: 原始图片目录
            source_labels_dir: 原始标签目录（YOLO分割格式）
            output_images_dir: 输出图片目录
            output_labels_dir: 输出标签目录
        """
        self.source_images_dir = Path(source_images_dir)
        self.source_labels_dir = Path(source_labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # 创建输出目录
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def random_mosaic_augmentation(self, image, 
                                  num_patches=2,
                                  min_size=50, 
                                  max_size=200,
                                  mosaic_cell_size=10,
                                  feather_radius=15):
        """
        在图像随机位置生成马赛克区域，使用铜色、白色和浅铜色三种颜色，边缘羽化处理
        
        参数:
            image: 输入图像 (numpy数组或路径)
            num_patches: 马赛克块的数量(1-2)
            min_size: 马赛克区域最小尺寸
            max_size: 马赛克区域最大尺寸
            mosaic_cell_size: 马赛克单元格大小(像素)
            feather_radius: 羽化半径(像素)
        
        返回:
            添加马赛克后的图像
        """
        # 定义三种颜色 (BGR格式)
        copper_color = [0, 26, 101]      # 铜色 (BGR)
        white_color = [122, 215, 241]      # 白色 (BGR)
        light_copper_color = [0, 52, 123]  # 浅铜色 (BGR)
        # 随机决定铜色和浅铜色的比例
        # 随机生成一个0到1之间的值，决定铜色的比例
        copper_ratio = random.random()  # 0.0到1.0之间
        
        # 根据比例决定颜色数量，总共有19个单元格，其中1个白色
        total_copper_cells = 18
        copper_count = int(total_copper_cells * copper_ratio)
        light_copper_count = total_copper_cells - copper_count
        
        # 创建颜色列表，白色比例更小，铜色和浅铜色比例随机
        colors = [light_copper_color] * light_copper_count + [copper_color] * copper_count + [white_color]
        
        # 读取图像
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        if img is None:
            raise ValueError("无法读取图像")
        
        h, w = img.shape[:2]
        num_patches = max(1, num_patches)
        
        # 根据图像尺寸自适应调整参数
        if min(h, w) <= 160:
            # 小图片参数
            adaptive_min_size = int(min(h, w) * 0.25)  # 约40像素
            adaptive_max_size = int(min(h, w) * 0.5)   # 约80像素
            adaptive_patch_w_min = 15
            adaptive_patch_w_max = 25
            adaptive_cell_size = max(1, mosaic_cell_size // 2)  # 减小马赛克单元格
            adaptive_feather = max(2, feather_radius // 2)  # 减小羽化半径
        else:
            # 大图片参数
            adaptive_min_size = min_size
            adaptive_max_size = max_size
            adaptive_patch_w_min = 30
            adaptive_patch_w_max = 50
            adaptive_cell_size = mosaic_cell_size
            adaptive_feather = feather_radius
        
        # 随机生成多个马赛克块
        for _ in range(num_patches):
            # 随机大小
            patch_h = random.randint(adaptive_min_size, adaptive_max_size)
            patch_w = random.randint(adaptive_patch_w_min, adaptive_patch_w_max)
            
            # 随机位置
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            
            # 确保区域在图像范围内
            x2 = min(x + patch_w, w)
            y2 = min(y + patch_h, h)
            
            # 创建一个临时图像用于生成马赛克
            mosaic_region = np.zeros_like(img[y:y2, x:x2])
            
            # 生成马赛克单元格，可以是长方形
            current_y = 0
            while current_y < (y2 - y):
                # 随机决定当前行单元格的高度
                # 确保随机范围有效（最小值至少为1）
                cell_min = max(1, adaptive_cell_size // 2)
                cell_max = max(cell_min, adaptive_cell_size * 2)
                cell_height = random.randint(cell_min, cell_max)
                cell_height = min(cell_height, (y2 - y) - current_y)
                
                current_x = 0
                while current_x < (x2 - x):
                    # 随机决定当前单元格的宽度
                    cell_min = max(1, adaptive_cell_size // 2)
                    cell_max = max(cell_min, adaptive_cell_size * 2)
                    cell_width = random.randint(cell_min, cell_max)
                    cell_width = min(cell_width, (x2 - x) - current_x)
                    
                    # 计算单元格位置
                    cell_y1 = current_y
                    cell_y2 = current_y + cell_height
                    cell_x1 = current_x
                    cell_x2 = current_x + cell_width
                    
                    # 随机选择一种颜色
                    color = random.choice(colors)
                    
                    # 如果是铜色或浅铜色，生成细长矩形
                    if color == copper_color or color == light_copper_color:
                        # 保持原有宽度，但缩小高度
                        strip_height = max(1, cell_height // 3)
                        strip_y = current_y + random.randint(0, cell_height - strip_height)
                        mosaic_region[strip_y:strip_y + strip_height, cell_x1:cell_x2] = color
                        # 填充剩余部分为浅铜色（避免黑色区域）
                        fill_color = light_copper_color if color == copper_color else copper_color
                        if strip_y > current_y:
                            mosaic_region[current_y:strip_y, cell_x1:cell_x2] = fill_color
                        if strip_y + strip_height < current_y + cell_height:
                            mosaic_region[strip_y + strip_height:current_y + cell_height, cell_x1:cell_x2] = fill_color
                    else:
                        # 白色保持原来的方块形状
                        mosaic_region[cell_y1:cell_y2, cell_x1:cell_x2] = color
                    
                    # 移动到下一个单元格
                    current_x += cell_width
                
                # 移动到下一行
                current_y += cell_height
            # 创建距离变换遮罩
            region_h, region_w = y2 - y, x2 - x
            mask = np.zeros((region_h, region_w), dtype=np.uint8)
            
            # 使用自适应羽化半径
            actual_feather_radius = min(adaptive_feather, min(region_h, region_w) // 3)
            
            # 填充中心区域为白色（距离边缘 > feather_radius 的部分）
            if actual_feather_radius > 0:
                # 计算内部矩形（完全不透明的区域）
                inner_top = actual_feather_radius
                inner_bottom = region_h - actual_feather_radius
                inner_left = actual_feather_radius
                inner_right = region_w - actual_feather_radius
                
                # 如果区域足够大，才填充中心
                if inner_bottom > inner_top and inner_right > inner_left:
                    mask[inner_top:inner_bottom, inner_left:inner_right] = 255
            
            # 对遮罩进行距离变换
            # 这会计算每个像素到最近的非零像素的距离
            mask_inv = cv2.bitwise_not(mask)
            dist_transform = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)
            
            # 归一化距离变换结果到 0-1 范围
            # 在 feather_radius 范围内创建平滑过渡
            if actual_feather_radius > 0:
                feather_mask = np.clip(dist_transform / actual_feather_radius, 0, 1)
                feather_mask = 1.0 - feather_mask  # 反转：边缘为0，中心为1
            else:
                feather_mask = np.ones((region_h, region_w), dtype=np.float32)
            
            # 可选：应用额外的高斯模糊使过渡更平滑
            if actual_feather_radius > 2:
                blur_size = actual_feather_radius // 2
                if blur_size % 2 == 0:
                    blur_size += 1
                feather_mask = cv2.GaussianBlur(feather_mask, (blur_size, blur_size), 0)
            
            # 确保值在0-1范围内
            feather_mask = np.clip(feather_mask, 0, 1)
            
            # 应用羽化遮罩混合原图和马赛克区域
            original_region = img[y:y2, x:x2].copy().astype(np.float32)
            mosaic_region_float = mosaic_region.astype(np.float32)
            
            # 将mask扩展到3个通道
            mask_3d = np.stack([feather_mask, feather_mask, feather_mask], axis=2)
            
            # 混合原图和马赛克区域
            blended_region = (original_region * (1 - mask_3d) + mosaic_region_float * mask_3d).astype(np.uint8)
            
            # 将混合结果放回原图
            img[y:y2, x:x2] = blended_region
        
        return img
    
    def random_color_fill_augmentation_percentile(self, image, 
                                              num_patches=3,
                                              min_size=100, 
                                              max_size=200,
                                              surrounding_margin=5,
                                              feather_radius=10,
                                              percentile=10):
        """
        在图像随机位置用周围区域的百分位数像素值填充作为数据增强，边缘羽化处理
        
        参数:
            image: 输入图像 (numpy数组或路径)
            num_patches: 填充块的数量
            min_size: 填充块最小尺寸
            max_size: 填充块最大尺寸
            surrounding_margin: 周围区域的边距(用于计算百分位数像素值)
            feather_radius: 羽化半径(像素)
            percentile: 百分位数值(0-100),默认为10,值越大颜色越亮
        
        返回:
            增强后的图像
        """
        # 读取图像
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        
        # 根据图像尺寸自适应调整patch参数
        # 对于小图片(160x160)，使用较小的patch
        # 对于大图片(360x360及以上)，使用较大的patch
        if min(h, w) <= 160:
            # 小图片参数
            adaptive_min_size = int(min(h, w) * 0.2)  # 约32像素
            adaptive_max_size = int(min(h, w) * 0.6)  # 约96像素
            adaptive_num_patches = max(1, num_patches // 2)  # 减少patch数量
        else:
            # 大图片参数
            adaptive_min_size = min_size
            adaptive_max_size = min(max_size, int(min(h, w) * 0.8))  # 不超过图像尺寸的80%
            adaptive_num_patches = num_patches
        
        # 随机生成多个填充块
        for _ in range(adaptive_num_patches):
            # 自适应调整patch尺寸
            patch_h = random.randint(adaptive_min_size, adaptive_max_size)
            patch_w = random.randint(int(adaptive_min_size * 0.4), int(adaptive_max_size * 0.6))
            
            # 随机位置
            x = random.randint(0, max(0, w - patch_w))
            y = random.randint(0, max(0, h - patch_h))
            
            # 计算周围区域的边界
            margin = surrounding_margin
            x1_outer = max(0, x - margin)
            y1_outer = max(0, y - margin)
            x2_outer = min(w, x + patch_w + margin)
            y2_outer = min(h, y + patch_h + margin)
            
            # 创建遮罩:标记周围区域(不包括中心填充区域)
            surrounding_mask = np.ones((y2_outer - y1_outer, x2_outer - x1_outer), dtype=bool)
            
            # 计算中心区域在局部坐标系中的位置
            local_y1 = y - y1_outer
            local_y2 = local_y1 + patch_h
            local_x1 = x - x1_outer
            local_x2 = local_x1 + patch_w
            
            # 排除中心填充区域
            surrounding_mask[local_y1:local_y2, local_x1:local_x2] = False
            
            # 提取周围区域
            surrounding_region = img[y1_outer:y2_outer, x1_outer:x2_outer]
            
            # 计算周围区域的百分位数像素值（比最小值稍亮）
            if np.any(surrounding_mask):
                # 使用指定百分位数，比最小值稍亮一些
                percentile_color = np.percentile(surrounding_region[surrounding_mask], percentile, axis=0)
            else:
                # 如果没有周围区域(边界情况),使用整个图像的指定百分位数
                percentile_color = np.percentile(img, percentile, axis=(0, 1))
            
            # 创建羽化遮罩
            # 首先创建一个矩形遮罩
            mask = np.zeros((patch_h, patch_w), dtype=np.float32)
            
            # 在矩形中心创建一个较小的矩形区域作为核心区域
            core_size_factor = 0.7  # 核心区域占整体区域的比例
            core_h = int(patch_h * core_size_factor)
            core_w = int(patch_w * core_size_factor)
            core_y = (patch_h - core_h) // 2
            core_x = (patch_w - core_w) // 2
            
            # 核心区域设为1.0
            mask[core_y:core_y+core_h, core_x:core_x+core_w] = 1.0
            
            # 应用高斯模糊创建羽化效果
            # 确保羽化半径不会太大
            actual_feather_radius = min(feather_radius, min(patch_h, patch_w) // 4)
            mask = cv2.GaussianBlur(mask, (0, 0), actual_feather_radius)
            
            # 确保值在0-1范围内
            mask = np.clip(mask, 0, 1)
            
            # 创建填充区域
            fill_region = np.zeros_like(img[y:y+patch_h, x:x+patch_w])
            fill_region[:, :] = percentile_color.astype(np.uint8)
            
            # 应用羽化遮罩混合原图和填充区域
            original_region = img[y:y+patch_h, x:x+patch_w].copy()
            
            # 将mask扩展到3个通道
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            # 混合原图和填充区域
            blended_region = (original_region * (1 - mask_3d) + fill_region * mask_3d).astype(np.uint8)
            
            # 将混合结果放回原图
            img[y:y+patch_h, x:x+patch_w] = blended_region
        
        return img
    
    def read_yolo_segmentation(self, label_path):
        """读取YOLO分割格式标签"""
        polygons = []
        if not os.path.exists(label_path):
            return polygons
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                # 将归一化坐标转换为像素坐标存储
                polygons.append({
                    'class_id': class_id,
                    'coords': coords  # 保持归一化格式
                })
        return polygons
    
    def save_yolo_segmentation(self, label_path, polygons):
        """保存YOLO分割格式标签"""
        with open(label_path, 'w') as f:
            for poly in polygons:
                class_id = poly['class_id']
                coords = poly['coords']
                coords_str = ' '.join([f'{c:.6f}' for c in coords])
                f.write(f"{class_id} {coords_str}\n")
    
    def denormalize_coords(self, coords, img_height, img_width):
        """反归一化坐标"""
        keypoints = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i+1] * img_height
            keypoints.append((x, y))
        return keypoints
    
    def normalize_coords(self, keypoints, img_height, img_width):
        """归一化坐标"""
        coords = []
        for x, y in keypoints:
            coords.append(x / img_width)
            coords.append(y / img_height)
        return coords
    
    def augment_single_image(self, image_path, label_path, index):
        """增强单张图片 - 使用两种自定义增强方法"""
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 读取标签
        polygons = self.read_yolo_segmentation(label_path)
        
        # 先保存原始图片和标签
        original_name = image_path.stem
        original_ext = image_path.suffix
        
        # 1. 保存原始文件
        shutil.copy(image_path, self.output_images_dir / f"{original_name}{original_ext}")
        shutil.copy(label_path, self.output_labels_dir / f"{original_name}.txt")
        
        # 2. 随机选择一种增强方法（50%概率选择马赛克，50%概率选择颜色填充）
        augmentation_method = random.choice(['mosaic', 'percentile'])
        
        if augmentation_method == 'mosaic':
            # 使用马赛克增强方法
            try:
                aug_image = self.random_mosaic_augmentation(
                    image,
                    num_patches=random.randint(2, 4),
                    min_size=40,
                    max_size=100,
                    mosaic_cell_size=3,
                    feather_radius=6
                )
                aug_image_name = f"{original_name}_aug{original_ext}"
                cv2.imwrite(str(self.output_images_dir / aug_image_name), aug_image)
                # 标签文件保持不变
                shutil.copy(label_path, self.output_labels_dir / f"{original_name}_aug.txt")
                
            except Exception as e:
                print(f"马赛克增强 {image_path.name} 时出错: {str(e)}")
        
        else:  # percentile
            # 使用颜色填充增强方法
            try:
                aug_image = self.random_color_fill_augmentation_percentile(
                    image,
                    num_patches=3,
                    min_size=30,
                    max_size=300,
                    surrounding_margin=6,
                    feather_radius=15,
                    percentile=15
                )
                aug_image_name = f"{original_name}_aug{original_ext}"
                cv2.imwrite(str(self.output_images_dir / aug_image_name), aug_image)
                # 标签文件保持不变
                shutil.copy(label_path, self.output_labels_dir / f"{original_name}_aug.txt")
                
            except Exception as e:
                print(f"颜色填充增强 {image_path.name} 时出错: {str(e)}")
    
    def augment_dataset(self):
        """增强整个数据集 - 每张图片随机选择一种增强方法(马赛克或颜色填充)"""
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.source_images_dir.glob(f'*{ext}'))
        
        print(f"找到 {len(image_files)} 张图片")
        print(f"每张图片将随机选择一种增强方法")
        print(f"预计总共生成 {len(image_files) * 2} 张图片 (原图 + 1个随机增强版本)\n")
        
        # 处理每张图片
        for idx, image_path in enumerate(tqdm(image_files, desc="数据增强进度")):
            label_path = self.source_labels_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                print(f"\n警告: 未找到标签文件 {label_path}")
                continue
            
            self.augment_single_image(image_path, label_path, idx)
        
        print(f"\n数据增强完成!")
        print(f"输出图片目录: {self.output_images_dir}")
        print(f"输出标签目录: {self.output_labels_dir}")


def main():
    """主函数 - 使用示例"""
    
    # ========== 配置参数 ==========
    SOURCE_IMAGES_DIR = r"E:\qcy\new-data\new-data-yolo-20251110\images"  # 原始图片目录
    SOURCE_LABELS_DIR = r"E:\qcy\new-data\new-data-yolo-20251110\labels"  # 原始标签目录
    OUTPUT_IMAGES_DIR = r"E:\qcy\train-augment\images"  # 输出图片目录
    OUTPUT_LABELS_DIR = r"E:\qcy\train-augment\labels"  # 输出标签目录
    
    # ========== 执行数据增强 ==========
    # 每张图片会生成2个版本：原图 + 随机选择的一种增强（马赛克或颜色填充，各50%概率）
    augmentor = YOLOSegmentationAugmentation(
        source_images_dir=SOURCE_IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR
    )
    
    augmentor.augment_dataset()


if __name__ == "__main__":
    main()
