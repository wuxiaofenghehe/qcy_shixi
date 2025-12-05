import shutil, os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    old_root = r"D:\qcy\yolo11aug\1808-label-aug"
    new_folder = r"D:\qcy\yolo11aug\new-data-yolo-20251110"
    merge_root = r"D:\qcy\yolo11aug\merge-data"
    ratio = (0.8, 0.1, 0.1)

    # 1. 收集全部图片绝对路径
    all_imgs = []

    # 旧数据
    for split in ['train', 'val', 'test']:
        img_dir = f'{old_root}/images/{split}'
        if os.path.isdir(img_dir):
            all_imgs += [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 新增数据
    new_img_dir = f'{new_folder}/images'
    if os.path.isdir(new_img_dir):
        all_imgs += [os.path.join(new_img_dir, f) for f in os.listdir(new_img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f'总共收集到 {len(all_imgs)} 张图片')

    # 2. 分层抽样
    train, tmp = train_test_split(all_imgs, test_size=sum(ratio[1:]), random_state=42)
    val, test = train_test_split(tmp, test_size=ratio[2] / sum(ratio[1:]), random_state=42)

    print(f'划分结果: train={len(train)}, val={len(val)}, test={len(test)}')


    # 3. 复制函数（增加错误处理）
    def copy_set(img_list, set_name):
        success = 0
        for img_path in img_list:
            # 标准化路径（统一为正斜杠或反斜杠）
            img_path = os.path.normpath(img_path)

            # 构建标签路径
            parts = img_path.split(os.sep)
            # 找到 'images' 并替换为 'labels'
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'labels'

            txt_path = os.sep.join(parts)
            txt_path = os.path.splitext(txt_path)[0] + '.txt'

            # 目标路径
            img_name = os.path.basename(img_path)
            txt_name = os.path.splitext(img_name)[0] + '.txt'

            dst_img = os.path.join(merge_root, 'images', set_name, img_name)
            dst_txt = os.path.join(merge_root, 'labels', set_name, txt_name)

            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_txt), exist_ok=True)

            try:
                shutil.copy(img_path, dst_img)
                if os.path.exists(txt_path):
                    shutil.copy(txt_path, dst_txt)
                else:
                    print(f'警告: 标签文件不存在 {txt_path}')
                success += 1
            except Exception as e:
                print(f'错误: 复制失败 {img_path}, 原因: {e}')

        print(f'{set_name} 集复制完成: {success}/{len(img_list)}')


    # 4. 执行
    copy_set(train, 'train')
    copy_set(val, 'val')
    copy_set(test, 'test')

    print('合并+重划分完成，输出在：', merge_root)  # 修正了括号
