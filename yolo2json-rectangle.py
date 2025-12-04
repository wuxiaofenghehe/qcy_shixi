import os
import json
import cv2
import numpy as np
from typing import List, Optional

# 支持的图片后缀
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def find_image_for_label(img_dir: str, base_name: str) -> Optional[str]:
    """根据 base_name（无扩展名）在 img_dir 中查找存在的图片。"""
    for ext in IMG_EXTS:
        candidate = os.path.join(img_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    for f in os.listdir(img_dir):
        if os.path.splitext(f)[0] == base_name:
            return os.path.join(img_dir, f)
    return None


def imread_unicode(path: str) -> Optional[np.ndarray]:
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


def yolo_bbox_to_labelme(img_path: str, txt_path: str, save_path: str, class_names: Optional[List[str]] = None) -> bool:
    """
    将 YOLO 目标检测框 (class_id xc yc w h) 转换为 LabelMe "rectangle" JSON 格式。
    要求每行标签必须是 5 个数值。
    """
    img = imread_unicode(img_path)
    if img is None:
        print("无法读取图片:", img_path)
        return False

    h, w = img.shape[:2]  # 获取图片高度和宽度
    shapes = []

    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading label file {txt_path}: {e}")
        return False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        data = line.split()

        # 严格要求：必须是 类别ID + 4个BBox坐标 (总共 5 个元素)
        if len(data) != 5:
            print(f"⚠️ 跳过非标准 BBox 格式行 (元素: {len(data)}) : {txt_path} -> {line}")
            continue

        try:
            cls = int(float(data[0]))
            # 提取 4 个归一化坐标
            xc, yc, bw, bh = list(map(float, data[1:]))
        except:
            print(f"❌ 坐标解析失败，跳过: {txt_path} -> {line}")
            continue

        # --- BBox 坐标转换逻辑 ---

        # 转换为像素坐标
        pxc = xc * w
        pyc = yc * h
        pbw = bw * w
        pbh = bh * h

        # 计算左上角 (x1, y1) 和右下角 (x2, y2)
        x1 = pxc - pbw / 2
        y1 = pyc - pbh / 2
        x2 = pxc + pbw / 2
        y2 = pyc + pbh / 2

        # LabelMe 矩形框数据结构
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)

        shapes.append({
            "label": label,
            # LabelMe 矩形只需要左上角和右下角两个点
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": None,
            "shape_type": "rectangle",  # 明确指定为矩形
            "flags": {}
        })

    # 构建 LabelMe JSON 结构
    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    # 写 json，保证中文写入正确
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error writing JSON file {save_path}: {e}")
        return False


def batch_convert_bbox(img_dir: str, label_dir: str, out_dir: str, class_names: Optional[List[str]] = None):
    """批量转换 YOLO BBox 标签到 LabelMe JSON。"""
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- 🚀 开始批量转换 YOLO BBox -> LabelMe JSON ---")

    for name in os.listdir(label_dir):
        if not name.lower().endswith(".txt"):
            continue

        txt_path = os.path.join(label_dir, name)
        base = os.path.splitext(name)[0]

        img_path = find_image_for_label(img_dir, base)
        if img_path is None:
            print(f"🔴 找不到对应图片，跳过: {base}")
            continue

        save_path = os.path.join(out_dir, base + ".json")

        ok = yolo_bbox_to_labelme(img_path, txt_path, save_path, class_names)

        if ok:
            print(f"✅ Converted BBox: {os.path.basename(save_path)}")
        else:
            print(f"❌ 转换失败: {name}")

    print("--- 转换完成 ---")


if __name__ == "__main__":
    # 📢 示例调用：请修改为你的实际路径和类别名称

    MY_CLASS_NAMES = ["bl", "yl", "yz"]

    batch_convert_bbox(
        img_dir=r"E:\qcy\样本数据集\检测数据集\1999_三次合并\images",  # 你的图片目录
        label_dir=r"E:\qcy\样本数据集\检测数据集\1999_三次合并\labels",  # 你的 YOLO BBox TXT 标签目录
        out_dir=r"E:\qcy\样本数据集\检测数据集\1999_三次合并\json-data",  # 转换后的 JSON 输出目录
        class_names=MY_CLASS_NAMES
    )