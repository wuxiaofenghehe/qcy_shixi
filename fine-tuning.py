from numpy.lib.twodim_base import flipud
from timm.utils import freeze

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"best.pt")  # 加载最佳模型权重
    results = model.train(
        resume=False,  # 不恢复优化器状态
        epochs=100,  # 新的训练轮次
        patience=10,
        lr0=1e-4,  # 自定义学习率
        optimizer='AdamW',
        data=r"ultralytics/cfg/datasets/oil_gdf_gzf_aug.yaml",
        project='yolov11s_oil',
        augment=True,
        copy_paste=0.3,
        mixup=0.3,
        cutmix=0.3,
        flipud=0.5,
        freeze=11,
    )
