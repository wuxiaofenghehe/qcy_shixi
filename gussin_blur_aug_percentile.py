import cv2
import numpy as np
import random


def random_color_fill_augmentation_percentile(image, 
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
        percentile: 百分位数值(0-100)，默认为10，值越大颜色越亮
    
    返回:
        增强后的图像
    """
    # 读取图像
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    h, w = img.shape[:2]
    
    # 随机生成多个填充块
    for _ in range(num_patches):
        # 随机大小
        patch_h = random.randint(200, max_size)
        patch_w = random.randint(40, 60)
        
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
        core_size_factor = 0.8  # 核心区域占整体区域的比例
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

# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例1: 单张图片 - 矩形填充
    img = cv2.imread(r"E:\qcy\train_1808\images\train\BZ26-6-2_743.png")
    
    # 示例2: 使用不同百分位数的对比
    result1 = random_color_fill_augmentation_percentile(
        img, 
        num_patches=3,
        min_size=30,
        max_size=300,
        surrounding_margin=6,
        feather_radius=15,
        percentile=25  # 使用25%分位数，更亮一些
    )
    cv2.imwrite('augmented_color_fill_percentile_25.jpg', result1)
    
    # 示例3: 批量处理
    # batch_augmentation('input_images/', 'augmented_images/', num_augmented=5)
    # 示例4: 可视化对比
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 25%分位数填充
    axes[2].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    axes[2].set_title('25th Percentile Fill')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_percentile.jpg')
    print("对比图已保存!")