import os
import json
import shutil

# --- 配置部分 ---
# 1. 待分类数据所在的目录（包含图片和JSON标签文件）
INPUT_DIR = r"D:\wellimg\wellimg20260401\202260401sliced-data\WZ10-5-4_yt\classified_output\Background"
# 2. 分类结果的输出目录（子文件夹: Object_Detection / Segmentation / Background）
OUTPUT_DIR = r"D:\wellimg\wellimg20260401\202260401sliced-data\WZ10-5-4_yt\classified_output"
# 3. 如果目标文件已存在，是否覆盖（True=覆盖, False=跳过）
OVERWRITE = False

# 目标检测 (Object Detection) 标签
OD_LABELS = {"bl", "yl", "yz"}
# 分割 (Segmentation) 标签
SEG_LABELS = {"gdf", "gzf"}


def safe_move(src, dst, overwrite=False):
    """
    安全移动文件：
    - 如果 src 和 dst 是同一个文件，跳过
    - 如果 dst 已存在且 overwrite=False，跳过
    - 如果 dst 已存在且 overwrite=True，覆盖
    返回: (action, message)
        action: "moved", "skipped_same", "skipped_exists", "overwritten"
    """
    src_abs = os.path.normcase(os.path.abspath(src))
    dst_abs = os.path.normcase(os.path.abspath(dst))

    if src_abs == dst_abs:
        return ("skipped_same", "已在目标文件夹中，跳过")

    if os.path.exists(dst_abs):
        if overwrite:
            os.remove(dst_abs)
            shutil.move(src, dst)
            return ("overwritten", "目标已存在，已覆盖")
        else:
            return ("skipped_exists", "目标已存在，跳过")

    shutil.move(src, dst)
    return ("moved", "")


def classify_file(input_dir, output_dir, overwrite=False):
    """
    遍历输入目录，根据JSON标签将图片及标签文件移动到对应类别子文件夹。
    """
    input_dir = os.path.normpath(input_dir)
    output_dir = os.path.normpath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 子文件夹
    od_dir = os.path.join(output_dir, "Object_Detection")
    seg_dir = os.path.join(output_dir, "Segmentation")
    background_dir = os.path.join(output_dir, "Background")

    for d in [od_dir, seg_dir, background_dir]:
        os.makedirs(d, exist_ok=True)

    stats = {"od": 0, "seg": 0, "bg": 0, "skipped": 0, "errors": 0}
    moved_images = set()

    print(f"{'=' * 50}")
    print(f"源目录:   {input_dir}")
    print(f"目标目录: {output_dir}")
    print(f"覆盖模式: {'是' if overwrite else '否（已存在则跳过）'}")
    print(f"{'=' * 50}\n")

    # 检测输入目录是否在输出目录内部
    input_abs = os.path.normcase(os.path.abspath(input_dir))
    output_abs = os.path.normcase(os.path.abspath(output_dir))
    if input_abs.startswith(output_abs + os.sep) or input_abs == output_abs:
        print("⚠ 注意: 输入目录位于输出目录内部。")
        print("  文件将被移动到对应类别文件夹，已在正确位置的文件夹不会被移动。\n")

    all_files = os.listdir(input_dir)
    json_files = [f for f in all_files if f.endswith(".json")]
    total = len(json_files)

    # --- 1. 处理带有JSON标签的文件 ---
    for idx, filename in enumerate(json_files, 1):
        json_path = os.path.join(input_dir, filename)
        base_name = filename[:-5]  # 去掉 .json

        # 匹配图片文件
        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            potential = base_name + ext
            if os.path.exists(os.path.join(input_dir, potential)):
                image_file = potential
                break

        if not image_file:
            print(f"[{idx}/{total}] {filename} -> 未找到对应图片，跳过")
            stats["skipped"] += 1
            continue

        image_path = os.path.join(input_dir, image_file)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            labels = {shape.get('label') for shape in data.get('shapes', []) if shape.get('label')}

            if labels.intersection(OD_LABELS):
                target_dir, category = od_dir, "OD"
            elif labels.intersection(SEG_LABELS):
                target_dir, category = seg_dir, "SEG"
            else:
                # 有JSON但没有有效标签 → 归为背景
                target_dir, category = background_dir, "BG"

            # 移动图片
            dst_image = os.path.join(target_dir, image_file)
            action, msg = safe_move(image_path, dst_image, overwrite)

            # 移动JSON
            dst_json = os.path.join(target_dir, filename)
            _, msg_j = safe_move(json_path, dst_json, overwrite)
            if msg_j and not msg:
                msg = msg_j

            if action in ("moved", "overwritten"):
                if category == "OD":
                    stats["od"] += 1
                elif category == "SEG":
                    stats["seg"] += 1
                else:
                    stats["bg"] += 1
                moved_images.add(image_file)

            label_str = f"(Label: {labels})" if labels else "(无标签)"
            status = f"[{idx}/{total}] {image_file} {label_str} -> {os.path.basename(target_dir)}"
            if msg:
                status += f" | {msg}"
            print(status)

        except Exception as e:
            print(f"[{idx}/{total}] 处理 {filename} 时出错: {e}")
            stats["errors"] += 1

    # --- 2. 处理无JSON文件的图片（纯背景） ---
    print(f"\n--- 处理无标签背景图像 ---")
    for filename in sorted(all_files):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) and filename not in moved_images:
            json_name = os.path.splitext(filename)[0] + ".json"
            if not os.path.exists(os.path.join(input_dir, json_name)):
                image_path = os.path.join(input_dir, filename)
                dst_image = os.path.join(background_dir, filename)
                action, msg = safe_move(image_path, dst_image, overwrite)

                if action in ("moved", "overwritten"):
                    stats["bg"] += 1

                status = f"  {filename} -> Background"
                if msg:
                    status += f" | {msg}"
                print(status)

    # --- 汇总 ---
    print(f"\n{'=' * 50}")
    print(f"分类完成！")
    print(f"  目标检测 (Object_Detection): {stats['od']} 张")
    print(f"  分割 (Segmentation):        {stats['seg']} 张")
    print(f"  背景 (Background):          {stats['bg']} 张")
    if stats["skipped"]:
        print(f"  跳过:                       {stats['skipped']} 张")
    if stats["errors"]:
        print(f"  错误:                       {stats['errors']} 个")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    classify_file(INPUT_DIR, OUTPUT_DIR, OVERWRITE)
