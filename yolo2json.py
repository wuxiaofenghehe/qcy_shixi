import os
import json
import cv2
import numpy as np

# 支持的图片后缀（按需可增删）
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_for_label(img_dir, base_name):
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

def yolo_to_labelme(img_path, txt_path, save_path, class_names=None):
    img = imread_unicode(img_path)
    if img is None:
        print("无法读取图片:", img_path)
        return False

    h, w = img.shape[:2]

    shapes = []
    # 以 utf-8 打开 txt，遇到非法字符用 replace（更健壮）
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = line.split()
        # 至少要有 class_id 和一对坐标
        if len(data) < 3:
            print(f"跳过格式错误的行: {txt_path} -> {line}")
            continue
        try:
            cls = int(float(data[0]))  # 有些文件 class 写成 0.0
        except:
            print(f"无法解析类别 id: {data[0]} in {txt_path}")
            continue
        points = data[1:]
        # 如果是 bbox（4 个数）而不是多点，跳过或可扩展处理
        if len(points) % 2 != 0:
            print(f"点数量不是偶数，跳过: {txt_path} -> {line}")
            continue

        # 将归一化坐标变回像素坐标（如果坐标已经是像素值，这里也会按 float 转换）
        polygon = []
        try:
            points = list(map(float, points))
        except:
            print(f"坐标解析失败，跳过: {txt_path} -> {line}")
            continue

        for i in range(0, len(points), 2):
            x = points[i]
            y = points[i+1]
            # 检测是否为归一化坐标（通常在 0..1 之间），若是则转为像素
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                px = x * w
                py = y * h
            else:
                px = x
                py = y
            polygon.append([float(px), float(py)])

        shapes.append({
            "label": class_names[cls] if class_names and cls < len(class_names) else str(cls),
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

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
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    return True

def batch_convert(img_dir, label_dir, out_dir, class_names=None):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(label_dir):
        if not name.lower().endswith(".txt"):
            continue
        txt_path = os.path.join(label_dir, name)
        base = os.path.splitext(name)[0]
        img_path = find_image_for_label(img_dir, base)
        if img_path is None:
            print("找不到对应图片，跳过:", base)
            continue

        save_path = os.path.join(out_dir, base + ".json")
        ok = yolo_to_labelme(img_path, txt_path, save_path, class_names)
        if ok:
            print("Converted:", save_path)
        else:
            print("转换失败:", txt_path)

if __name__ == "__main__":
    # 示例调用，修改为你的路径和类别名称（类别名可以包含中文）
    batch_convert(
        img_dir=r"E:\qcy\data\train_1808\images\test",
        label_dir=r"E:\qcy\data\train_1808\labels\test",
        out_dir=r"E:\qcy\data\train_1808\labelme_json\1808-test",
        class_names=["gdf", "gzf"]  # 你可以把类别名改成中文
    )
