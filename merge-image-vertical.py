
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    # 1. 直接在这里写你的图片路径，想写几个就写几个
    pth1 = r"E:\qcy\IMAGE-WELL2-slicedimg\IMAGE-WELL2_slice_013.png"
    pth2 = r"E:\qcy\ce\BZ26-6-6-sliced\BZ26-6-6\BZ26-6-6_slice_113.png"
    pth3 = r"E:\qcy\ce\BZ26-6-6-sliced\BZ26-6-6-seg-yolo\images\BZ26-6-6_slice_287.png"
    pth4 = r"E:\qcy\new-data\new-data-20251125\20251125-yolo-seg-data\images\W4_2392-2832m_xf_slice_007.png"
    # 如果还有更多，继续 pth4 = r"..."

    paths = [pth1, pth2, pth3, pth4]          # 把变量全丢进列表

    # 2. 可选参数
    OUT_FILE = r"E:\qcy\vertical-merged.png"             # 输出文件名
    GAP      = 0                        # 图片间间隔（像素）

    # 3. 拼接逻辑
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    total_h = sum(i.height for i in imgs) + GAP * (len(imgs) - 1)
    max_w   = max(i.width  for i in imgs)

    canvas = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y), im)
        y += im.height + GAP

    canvas.save(OUT_FILE)
    print(f"纵向拼接完成 → {Path(OUT_FILE).absolute()}")