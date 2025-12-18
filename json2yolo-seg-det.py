import os
import json
import cv2
import numpy as np
import shutil

# 支持的图片后缀
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_for_json(img_dir, base_name):
    """根据 base_name（无扩展名）在 img_dir 中查找存在的图片，返回完整路径或 None。"""
    for ext in IMG_EXTS:
        candidate = os.path.join(img_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    # 宽松查找
    for f in os.listdir(img_dir):
        if os.path.splitext(f)[0] == base_name:
            return os.path.join(img_dir, f)
    return None

def imread_unicode(path):
    """以二进制方式读取图片文件，使用 cv2.imdecode 解码，解决中文路径问题。"""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def json_to_yolo(json_path, img_path, save_path, class_names=None, format_type='auto'):
    """将JSON格式的标签转换为YOLO格式
    
    Args:
        json_path: JSON标签文件路径
        img_path: 图片文件路径
        save_path: 输出的YOLO标签文件路径
        class_names: 类别名称列表
        format_type: 输出格式类型
            - 'auto': 自动检测（rectangle转bbox格式，polygon转分割格式）
            - 'detection': 强制转换为目标检测格式（bbox）
            - 'segmentation': 强制转换为实例分割格式（polygon）
    
    YOLO格式说明：
        - 目标检测: class_id x_center y_center width height
        - 实例分割: class_id x1 y1 x2 y2 x3 y3 ... (多个点的坐标)
        - 所有坐标都归一化到 0-1 范围
    """
    img = imread_unicode(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return False

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}, 路径: {img_path}")
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败 {json_path}: {e}")
        return False
    
    # 创建类别名到ID的映射
    if class_names:
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        print(f"[信息] 类别映射: {class_to_id}")
    else:
        class_to_id = {}
    
    yolo_lines = []
    shapes = json_data.get('shapes', [])
    print(f"[信息] JSON中包含 {len(shapes)} 个标注对象")
    
    # 处理每个形状
    for idx, shape in enumerate(shapes):
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'rectangle')
        
        print(f"  [对象{idx+1}] 类别: {label}, 形状类型: {shape_type}, 点数: {len(points)}")
        
        # 获取类别ID
        if class_names:
            if label in class_to_id:
                class_id = class_to_id[label]
            else:
                print(f"未知类别: {label}，跳过")
                continue
        else:
            try:
                class_id = int(label)
            except ValueError:
                print(f"无法解析类别ID: {label}，跳过")
                continue
        
        # 根据format_type和shape_type决定输出格式
        output_as_bbox = False
        output_as_polygon = False
        
        if format_type == 'detection':
            output_as_bbox = True
        elif format_type == 'segmentation':
            output_as_polygon = True
        else:  # auto
            if shape_type == 'rectangle':
                output_as_bbox = True
            elif shape_type == 'polygon':
                output_as_polygon = True
            else:
                print(f"未知的形状类型: {shape_type}，跳过")
                continue
        
        # ===== 处理矩形框（目标检测格式）=====
        if output_as_bbox:
            if shape_type == 'rectangle' and len(points) == 2:
                # 标准矩形：两个对角点
                x1, y1 = points[0]
                x2, y2 = points[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
            elif shape_type == 'polygon' and len(points) >= 3:
                # 多边形转矩形：使用外接矩形
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
            else:
                print(f"无法转换为bbox格式，点数不足，跳过")
                continue
            
            # 计算YOLO bbox格式：x_center, y_center, width, height（归一化）
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bbox_width = (xmax - xmin) / w
            bbox_height = (ymax - ymin) / h
            
            # 边界检查
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            bbox_width = max(0, min(1, bbox_width))
            bbox_height = max(0, min(1, bbox_height))
            
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            print(f"转换为YOLO bbox格式: {yolo_line}")
            yolo_lines.append(yolo_line)
        
        # ===== 处理多边形（实例分割格式）=====
        elif output_as_polygon:
            if shape_type == 'rectangle' and len(points) == 2:
                # 矩形转多边形：生成4个角点
                x1, y1 = points[0]
                x2, y2 = points[1]
                polygon_points = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]
            elif shape_type == 'polygon' and len(points) >= 3:
                # 直接使用多边形点
                polygon_points = points
            else:
                print(f"无法转换为polygon格式，点数不足，跳过")
                continue
            
            # 归一化多边形坐标
            normalized_coords = []
            for point in polygon_points:
                x, y = point
                norm_x = max(0, min(1, x / w))
                norm_y = max(0, min(1, y / h))
                normalized_coords.extend([norm_x, norm_y])
            
            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
            print(f"转换为YOLO polygon格式: class={class_id}, {len(polygon_points)}个点")
            yolo_lines.append(yolo_line)
    
    # 写入YOLO格式文件
    if not yolo_lines:
        print(f"警告: 没有有效的标注对象，生成空文件")
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        print(f"已写入 {len(yolo_lines)} 个标注到: {save_path}\n")
        return True
    except Exception as e:
        print(f"错误: 写入YOLO文件失败 {save_path}: {e}")  
        return False

def batch_convert(img_dir, json_dir, out_dir, class_names=None, format_type='auto'):
    """批量转换JSON格式标签为YOLO格式
    
    Args:
        img_dir: 图片文件夹路径
        json_dir: JSON标签文件夹路径
        out_dir: 输出文件夹路径
        class_names: 类别名称列表
        format_type: 输出格式类型 ('auto', 'detection', 'segmentation')
    """
    os.makedirs(out_dir, exist_ok=True)
    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    label_out_dir = os.path.join(out_dir, "labels")
    os.makedirs(label_out_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    print(f"\n{'='*60}")
    print(f"开始批量转换，共找到 {len(json_files)} 个JSON文件")
    print(f"转换模式: {format_type}")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for name in json_files:
        print(f"{'='*60}")
        print(f"处理文件: {name}")
        print(f"{'='*60}")
        
        json_path = os.path.join(json_dir, name)
        base = os.path.splitext(name)[0]
        
        img_path = find_image_for_json(img_dir, base)
        if img_path is None:
            print(f"错误: 找不到对应图片，跳过: {base}\n")
            fail_count += 1
            continue
        
        save_path = os.path.join(label_out_dir, base + ".txt")
        ok = json_to_yolo(json_path, img_path, save_path, class_names, format_type)
        
        if ok:
            # 复制图片到输出目录
            dst_img_path = os.path.join(img_out_dir, os.path.basename(img_path))
            shutil.copy2(str(img_path), str(dst_img_path))
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"转换完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # 对于分割和检测需要更改类别名称
    batch_convert(
        img_dir=r"E:\qcy\new-data\20251110-object\Object_Detection",  # 图片文件夹路径
        json_dir=r"E:\qcy\new-data\20251110-object\Object_Detection",  # JSON标签文件夹路径
        out_dir=r"E:\qcy\new-data\20251110-object\Object_Detection\yolo-object-data",  # 输出YOLO格式标签的文件夹路径
        class_names=["bl", "yl","YZ"],  # 类别名称列表，可以包含中文
        format_type='auto'  # 自动检测
    )
