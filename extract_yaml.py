from ultralytics import YOLO
import yaml
import os

def extract_yaml():
    weights_path = "best-copy.pt"
    output_path = "best-copy-model.yaml"
    
    if not os.path.exists(weights_path):
        print(f"错误: {weights_path} 未找到。")
        return

    try:
        print(f"正在从 {weights_path} 加载模型...")
        model = YOLO(weights_path)
        
        # 访问模型配置
        # 在 Ultralytics 中，model.model.yaml 包含架构字典
        if hasattr(model.model, 'yaml'):
            config = model.model.yaml
            
            # 保存到文件
            with open(output_path, "w", encoding='utf-8') as f:
                yaml.dump(config, f, sort_keys=False, allow_unicode=True)
            print(f"成功将模型配置保存到 {output_path}")
            
            # 打印配置的基本信息
            print("\n模型配置信息:")
            if isinstance(config, dict):
                for key in ['nc', 'depth_multiple', 'width_multiple', 'backbone', 'head']:
                    if key in config:
                        print(f"  {key}: {config[key]}")
        else:
            print("错误: 在 model.model 中未找到 'yaml' 属性")
            print("可用属性:", dir(model.model))

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_yaml()
