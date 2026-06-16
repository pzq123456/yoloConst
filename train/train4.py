# yoloCons/train.py
from ultralytics import YOLO
from pathlib import Path

def main():
    PATH_TO_MODEL = Path("runs/detect/yolo26n_finetune-2/weights/best.pt")
    model = YOLO(PATH_TO_MODEL)

    model.train(
        data="data6/data.yaml",
        epochs=100,                     # 1. 减少epochs
        lr0=0.001,                      # 2. 降低初始学习率
        imgsz=640,
        batch=32,
        name="yolo26n_finetune",
        device=0,
        patience=20,                    # 早停耐心值可以调低
        workers=4,
    )

if __name__ == "__main__":
    main()