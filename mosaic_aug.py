import cv2
import numpy as np
import random


def random_mosaic_augmentation(image, 
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
    
    # 随机生成多个马赛克块
    for _ in range(num_patches):
        # 随机大小
        patch_h = random.randint(min_size, max_size)
        patch_w = random.randint(30, 50)
        
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
            cell_height = random.randint(mosaic_cell_size // 2, mosaic_cell_size * 2)
            cell_height = min(cell_height, (y2 - y) - current_y)
            
            current_x = 0
            while current_x < (x2 - x):
                # 随机决定当前单元格的宽度
                cell_width = random.randint(mosaic_cell_size // 2, mosaic_cell_size * 2)
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
        
        # 确保羽化半径合理
        actual_feather_radius = min(feather_radius, min(region_h, region_w) // 3)
        
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


# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例1: 单张图片处理
    img = cv2.imread(r"E:\qcy\train_1808\images\train\BZ26-6-2_743.png")
    
    if img is not None:
        # 生成1-2个马赛克区域
        result = random_mosaic_augmentation(
            img, 
            num_patches=random.randint(2, 4),
            min_size=40,
            max_size=100,
            mosaic_cell_size=3,
            feather_radius=6  # 可以调大一点看效果更明显
        )
        cv2.imwrite('mosaic_augmented2.jpg', result)
        print("马赛克增强图像已保存为 'mosaic_augmented.jpg'")
        
        # 示例2: 可视化对比
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原图
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 马赛克增强图
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Mosaic Augmentation')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('mosaic_comparison.jpg')
        print("对比图已保存为 'mosaic_comparison.jpg'")
    else:
        print("无法加载图像，请检查路径是否正确")
