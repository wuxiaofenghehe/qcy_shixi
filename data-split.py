from sklearn.model_selection import train_test_split
import os
import shutil


def split_dataset(images_dir, labels_dir, output_dir, test_size=0.1, val_size=0.1, random_seed=42):
    """
    将图像和 YOLO 标签文件划分为训练集、验证集和测试集。

    Args:
        images_dir (str): 原始图像文件所在的目录路径。
        labels_dir (str): 原始标签文件所在的目录路径。
        output_dir (str): 划分后数据集的输出目录路径。
        test_size (float): 测试集占原始数据集的比例。默认为 0.1。
        val_size (float): 验证集占原始数据集的比例。默认为 0.1。
        random_seed (int): 随机种子，用于保证划分结果可复现。默认为 42。
    """
    print("开始数据划分...")
    # 1. 获取所有图像文件名
    # 过滤掉非文件和非指定后缀的文件，提高鲁棒性
    images = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png') and os.path.isfile(os.path.join(images_dir, f))])

    if not images:
        print(f"在 {images_dir} 目录中未找到任何 .png 图像文件。请检查路径和文件。")
        return

    # 2. 第一次划分：分离出 测试集 (test)
    # 划分后的 train_val_images 包含了 训练集 和 验证集 的文件
    train_val_images, test_images = train_test_split(
        images,
        test_size=test_size,
        random_state=random_seed
    )

    # 3. 第二次划分：从剩余的 train_val_images 中分离出 验证集 (val) 和 训练集 (train)
    # 核心修正：计算新的 test_size (即 val_size_adjusted)。
    # val_size 是相对于原始数据集的比例 (e.g., 0.1)
    # 但第二次划分的 train_test_split 是相对于 train_val_images (剩余的 1.0 - test_size)
    # 新的比例 = 原始 val_size / 剩余数据集比例
    # 例如：原始 val_size=0.1, test_size=0.1, 则剩余比例为 0.9。新的比例 = 0.1 / 0.9 ≈ 0.1111
    val_size_adjusted = val_size / (1 - test_size)

    # 确保调整后的比例不超限
    if val_size_adjusted >= 1.0:
        print("错误：验证集和测试集的比例之和必须小于 1.0。请检查 val_size 和 test_size。")
        return

    train_images, val_images = train_test_split(
        train_val_images,
        test_size=val_size_adjusted,  # 使用调整后的比例
        random_state=random_seed
    )

    # 4. 统计结果
    total_count = len(images)
    print(f"\n数据集总数: {total_count} 个文件")
    print(f"--- 划分结果 ---")
    print(f"训练集 (train): {len(train_images)} 个文件 ({len(train_images) / total_count:.2%})")
    print(f"验证集 (val):   {len(val_images)} 个文件 ({len(val_images) / total_count:.2%})")
    print(f"测试集 (test):  {len(test_images)} 个文件 ({len(test_images) / total_count:.2%})")

    # 5. 复制文件到输出目录
    for subset, subset_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        # 定义目标子目录路径
        target_images_dir = os.path.join(output_dir, 'images', subset)
        target_labels_dir = os.path.join(output_dir, 'labels', subset)

        # 创建目录，exist_ok=True 避免重复创建时报错
        os.makedirs(target_images_dir, exist_ok=True)
        os.makedirs(target_labels_dir, exist_ok=True)

        print(f"\n📋 复制 {subset} 子集的文件...")

        for i, image in enumerate(subset_images):
            # 原始文件路径
            source_image_path = os.path.join(images_dir, image)
            label_file = image.replace('.png', '.txt')  # 假设标签文件名与图像名一致，只是后缀不同
            source_label_path = os.path.join(labels_dir, label_file)

            # 目标文件路径
            target_image_path = os.path.join(target_images_dir, image)
            target_label_path = os.path.join(target_labels_dir, label_file)

            # 检查标签文件是否存在，不存在则跳过，避免报错
            if not os.path.exists(source_label_path):
                print(f"   警告：图像 {image} 对应的标签文件 {label_file} 不存在，跳过此文件。")
                continue

            # 复制文件
            shutil.copy2(source_image_path, target_image_path)  # 使用 copy2 保留更多元数据
            shutil.copy2(source_label_path, target_label_path)

        print(f"   {subset} 子集复制完成。")

    print("\n✅ 数据划分和复制操作成功完成！")


if __name__ == '__main__':
    # 请确保以下路径存在且正确
    images_dir = r"D:\qcy\yolo11aug\new-data-yolo-20251110\images"
    labels_dir = r"D:\qcy\yolo11aug\new-data-yolo-20251110\labels"
    output_dir = r"D:\qcy\yolo11aug\new-data-yolo-20251110\split-data"

    # 默认划分比例：测试集 10%，验证集 10%，训练集 80%
    split_dataset(images_dir, labels_dir, output_dir, test_size=0.1, val_size=0.1)