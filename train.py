# yoloCons/train.py
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True

    PATH_TO_MODEL = Path("model/best.pt")
    model = YOLO(PATH_TO_MODEL)

    model.train(
        data="data7/data.yaml",
        epochs=500,
        imgsz=640,
        patience=50,
        batch=32,
        name="yolo26n_mocs",
        device=-1,
    )

if __name__ == "__main__":
    main()