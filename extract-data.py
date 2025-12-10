import os
import shutil

def extract_images_from_subfolders(source_dir, destination_dir, image_extensions=None):
    """
    遍历源文件夹下的所有子文件夹，将找到的图片文件复制到目标文件夹。

    Args:
        source_dir (str): 包含子文件夹的根目录。
        destination_dir (str): 所有图片将被复制到的目标目录。
        image_extensions (list, optional): 需要提取的图片文件扩展名列表。
                                           默认为常见的扩展名。
    """
    if image_extensions is None:
        # 默认的图片扩展名列表 (统一转为小写，方便匹配)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # 1. 创建目标目录（如果不存在）
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"创建目标目录: {destination_dir}")
    else:
        print(f"目标目录已存在: {destination_dir}")

    total_files_copied = 0
    
    # 2. 遍历源文件夹及其子目录
    # os.walk() 会生成一个三元组：(当前文件夹路径, [子文件夹列表], [文件列表])
    for root, dirs, files in os.walk(source_dir):
        # 排除目标文件夹，防止意外循环复制
        if root == destination_dir:
            continue
            
        print(f"正在检查目录: {root}")
        
        for file in files:
            # 获取文件扩展名并转为小写
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension in image_extensions:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)
                
                # 检查目标文件夹中是否已存在同名文件
                if os.path.exists(destination_path):
                    # 如果存在同名文件，我们添加一个计数器来重命名文件，防止覆盖
                    base_name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_filename = f"{base_name}_{counter}{ext}"
                        destination_path = os.path.join(destination_dir, new_filename)
                        counter += 1
                    print(f"文件名冲突，重命名为: {os.path.basename(destination_path)}")

                # 3. 复制文件
                try:
                    shutil.copy2(source_path, destination_path)
                    total_files_copied += 1
                    
                except Exception as e:
                    print(f"复制文件失败 {source_path}: {e}")

    print(f"\n提取完成！")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {destination_dir}")
    print(f"成功复制了 {total_files_copied} 个文件。")


# ---配置参数 ---
# 1. 包含子文件夹的根目录路径
SOURCE_DIRECTORY = r"E:\qcy\new-data\新样本20251125\20251125-sliced_images"

# 2. 所有图片最终要提取到的目标目录路径
DESTINATION_DIRECTORY = r"E:\qcy\new-data\新样本20251125\20251125data-json" 
# -------------------------


# 运行提取函数
if __name__ == "__main__":
    extract_images_from_subfolders(
        source_dir=SOURCE_DIRECTORY,
        destination_dir=DESTINATION_DIRECTORY
    )