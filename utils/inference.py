from ultralytics import YOLO
import os

def main():
    # 1. 设定路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 你的最优权重路径
    model_path = os.path.join(current_dir, "runs/detect/fyp_mocs/weights/best.pt")
    
    # 需要推理的图片路径列表
    image_paths = [
        os.path.join(current_dir, "test/image.png"),
        os.path.join(current_dir, "test/image copy.png"),
        os.path.join(current_dir, "test/image copy 2.png"),
        os.path.join(current_dir, "test/image1.png"),
    ]

    # 2. 加载模型
    # 如果路径不对，这里会报错，请确保 best.pt 确实存在于指定位置
    model = YOLO(model_path)

    # 3. 执行推理
    # save=True 会自动在 runs/detect/predict 文件夹下保存结果
    # conf=0.25 是置信度阈值，低于这个分数的框不会显示
    # imgsz=640 是推理时的尺寸，模型会自动缩放图片，通常保持默认或设为训练时的尺寸(如416)即可
    results = model.predict(
        source=image_paths, 
        save=True,          
        conf=0.3,          
        imgsz=416,          # 建议设为你训练时使用的 imgsz
    )

    # 4. 打印推理结果简单信息
    for i, r in enumerate(results):
        print(f"图片 {image_paths[i]} 推理完成，检测到 {len(r.boxes)} 个目标。")

if __name__ == "__main__":
    main()