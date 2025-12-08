# -*- coding: utf-8 -*-
"""
横向拼接长图，留 10px 白边
"""
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 1. 只需改这里：把待拼接的文件名（支持绝对/相对路径）按顺序填进列表
INPUT_FILES = [
    r"E:\qcy\ce\BZ26-6-6_old.png",
    r"E:\qcy\wellimg2025\Workspace\recognize\BZ26-6-6\results\BZ26-6-6_result.png",
    # '图C.png',   # 想拼几张就继续加
]

OUTPUT_FILE = r'E:\qcy\ce\BZ26-6-6-mergeout.png'   # 2. 输出文件名（可含路径）

GAP = 10            # 3. 白色边距像素
BG_COLOR = (255, 255, 255, 255)  # 白色背景 RGBA

def load_resize(im_path: str, target_h: int) -> Image.Image:
    im = Image.open(im_path)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    new_w = int(im.width * target_h / im.height)
    return im.resize((new_w, target_h), Image.LANCZOS)

def hmerge_white_gap(paths: list[str]) -> Image.Image:
    # 统一高度
    max_h = max(Image.open(p).height for p in paths)
    images = [load_resize(p, max_h) for p in paths]

    total_w = sum(im.width for im in images) + GAP * (len(images) - 1)
    canvas = Image.new('RGBA', (total_w, max_h), BG_COLOR)

    x = 0
    for im in images:
        canvas.paste(im, (x, 0), im)
        x += im.width + GAP
    return canvas

if __name__ == '__main__':
    merged = hmerge_white_gap(INPUT_FILES)
    merged.save(OUTPUT_FILE)
    print(f'拼接完成 → {OUTPUT_FILE}')