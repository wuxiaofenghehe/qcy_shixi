from ultralytics import YOLO
import torch
#将pt文件转换为pth文件
if __name__ == "__main__":
    #直接填写pt文件路径
    print("开始转换")
    model = YOLO(r"E:\qcy\wellimg2026\model_logs\segmentation_weight_20250906\87.2-seg-use889-best.pt")
    torch.save(model.model.state_dict(), "yolo11s-seg-best.pth")
