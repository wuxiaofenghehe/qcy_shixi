import os
import json
import cv2
import numpy as np
import shutil  # 新增

# 支持的图片后缀（按需可增删）
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_for_json(img_dir, base_name):
    """根据 base_name（无扩展名）在 img_dir 中查找存在的图片，返回完整路径或 None。"""
    for ext in IMG_EXTS:
        candidate = os.path.join(img_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    # 有时候标签名本身可能包含扩展或不同大小写，做一个更宽松的查找
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

def json_to_yolo(json_path, img_path, save_path, class_names=None):
    """将JSON格式的标签转换为YOLO格式"""
    img = imread_unicode(img_path)
    if img is None:
        print("无法读取图片:", img_path)
        return False

    h, w = img.shape[:2]
    
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
    else:
        class_to_id = {}
    
    yolo_lines = []
    
    # 处理每个形状
    for shape in json_data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'polygon')
        
        # 只处理多边形
        if shape_type != 'polygon' or len(points) < 3:
            continue
        
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
        
        # 将多边形坐标转换为归一化坐标
        normalized_points = []
        for point in points:
            x, y = point
            # 归一化坐标
            norm_x = x / w
            norm_y = y / h
            normalized_points.extend([norm_x, norm_y])
        
        # 构建YOLO格式行
        yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
        yolo_lines.append(yolo_line)
    
    # 写入YOLO格式文件
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        return True
    except Exception as e:
        print(f"写入YOLO文件失败 {save_path}: {e}")
        return False

def batch_convert(img_dir, json_dir, out_dir, class_names=None):
    """批量转换JSON格式标签为YOLO格式"""
    os.makedirs(out_dir, exist_ok=True)
    img_out_dir = os.path.join(out_dir, "images")  # 新增
    os.makedirs(img_out_dir, exist_ok=True)       # 新增
    
    for name in os.listdir(json_dir):
        if not name.lower().endswith(".json"):
            continue
        
        json_path = os.path.join(json_dir, name)
        base = os.path.splitext(name)[0]
        
        img_path = find_image_for_json(img_dir, base)
        if img_path is None:
            print("找不到对应图片，跳过:", base)
            continue
        
        save_path = os.path.join(out_dir, base + ".txt")
        ok = json_to_yolo(json_path, img_path, save_path, class_names)
        
        if ok:
            # ---------------- 3. 同步图片 ----------------
            dst_img_path = os.path.join(img_out_dir, os.path.basename(img_path))
            shutil.copy2(str(img_path), str(dst_img_path))  # 复制
            print("转换+输出成功:", save_path, "=> 图片:", dst_img_path)
        else:
            print("转换失败:", json_path)

if __name__ == "__main__":
    # 示例调用，修改为你的路径和类别名称（类别名可以包含中文）
    batch_convert(
        img_dir=r"E:\qcy\new-data\gdf_gzf",  # 图片文件夹路径
        json_dir=r"E:\qcy\new-data\gdf_gzf",  # JSON标签文件夹路径
        out_dir=r"E:\qcy\new-data\new-data-label",  # 输出YOLO格式标签的文件夹路径
        class_names=["gdf", "gzf"]  # 类别名称列表，可以包含中文
    )