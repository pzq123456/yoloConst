from ultralytics import YOLO
import os

def export_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../model/best.pt")

    model = YOLO(model_path)

    model.export(
        format="engine", 
        half=True,        # 开启 FP16 量化，提速核心
        device=0,         # 必须在有 GPU 的设备上运行
        simplify=True     # 简化计算图
    )

    print("TensorRT 模型导出成功")

if __name__ == "__main__":
    export_model()