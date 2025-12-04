import os
import json
import shutil

# --- 配置部分 ---
# 1. 更改为你的数据所在的根文件夹路径
INPUT_DIR = r"E:\qcy\new-data\new_data_20251110_slice"
# 2. 更改为你想保存分类结果的输出文件夹路径
OUTPUT_DIR = r"E:\qcy\new-data\20251110-detect-json"

# 目标检测 (Object Detection) 标签
OD_LABELS = {"bl", "yl", "YZ"}
# 分割 (Segmentation) 标签
SEG_LABELS = {"gdf", "gzf"}


# --- 分类函数 ---
def classify_file(input_dir, output_dir):
    """
    遍历输入目录，并根据JSON标签将图片及其标签文件分类到不同的子文件夹。
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义子文件夹路径
    od_dir = os.path.join(output_dir, "Object_Detection")
    seg_dir = os.path.join(output_dir, "Segmentation")
    background_dir = os.path.join(output_dir, "Background")

    # 创建子文件夹
    os.makedirs(od_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    # 存储已处理的图片文件名（避免重复处理）
    processed_images = set()

    print(f"--- 开始分类 ---")
    print(f"源目录: {input_dir}")
    print(f"目标目录: {output_dir}\n")

    # 1. 首先处理带有JSON标签的文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)

            # 找到对应的图片文件名（假设图片名和json名相同，只是扩展名不同）
            # 兼容常见的图片格式，这里假设是.jpg, .png等
            base_name = filename[:-5]  # 移除 .json

            # 尝试匹配可能的图片文件
            image_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:  # 常见的图片扩展名
                potential_image = base_name + ext
                if os.path.exists(os.path.join(input_dir, potential_image)):
                    image_file = potential_image
                    break

            if image_file:
                image_path = os.path.join(input_dir, image_file)

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # **关键部分：解析JSON数据以找到所有标签**
                        # 假设标签列表在 'shapes' 键下，且每个标签对象有 'label' 键
                        labels = {shape.get('label') for shape in data.get('shapes', []) if shape.get('label')}

                        target_dir = None

                        # 检查是否有目标检测标签
                        if labels.intersection(OD_LABELS):
                            target_dir = od_dir
                        # 检查是否有分割标签
                        elif labels.intersection(SEG_LABELS):
                            target_dir = seg_dir

                        # 如果是带有标签的图像
                        if target_dir:
                            print(f"{image_file} (Label: {labels}) -> {os.path.basename(target_dir)}")

                            # 移动/复制图片和JSON文件
                            shutil.copy2(image_path, os.path.join(target_dir, image_file))
                            shutil.copy2(json_path, os.path.join(target_dir, filename))
                            processed_images.add(image_file)

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    # 2. 处理未被标记的图片（背景图像）
    print("\n--- 处理背景图像 ---")
    for filename in os.listdir(input_dir):
        # 检查是否是图片文件，并且没有被上面的步骤处理过
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) and filename not in processed_images:
            image_path = os.path.join(input_dir, filename)

            # 检查同名的JSON文件是否存在
            json_name = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(input_dir, json_name)

            # 如果没有对应的JSON文件，或者JSON文件不存在（已处理的图片已被排除）
            if not os.path.exists(json_path):
                print(f"{filename} -> Background")
                shutil.copy2(image_path, os.path.join(background_dir, filename))

            # 额外检查：如果JSON存在但没有有效标签，理论上也应归为背景，
            # 但为了简化，我们仅将“无JSON文件”的图片视为背景。

    print("\n--- 分类完成 ---")
if __name__ == "__main__":
        classify_file(INPUT_DIR, OUTPUT_DIR)