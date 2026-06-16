# yoloCons/train.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True

    model = YOLO("yolo26n.pt")

    model.train(
        data="data7/data.yaml",
        epochs=300,
        imgsz=640,
        name="yolo26n_mocs",
        device=-1,
    )

if __name__ == "__main__":
    main()