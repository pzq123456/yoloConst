# yoloCons/train.py
from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    # 2. 开始训练
    model.train(
        data="data/data.yaml",       # 修改为你的实际路径
        epochs=600,                  # 训练轮数
        imgsz=640,                   # 你的数据集预处理尺寸是 416，建议保持一致
        batch=32,                    # 若显存不足 (OOM)，请尝试减小到 8 或 4
        name="yolo26s_mocs",    # 保存的实验名称
        device=0,                    # 确保显卡驱动正确安装
        patience=30,                 # 增加耐心值，给模型更多收敛时间
        workers=4,                   # 数据加载线程，根据 CPU 核心数设置
    )

if __name__ == "__main__":
    main()