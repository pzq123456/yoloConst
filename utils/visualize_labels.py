import cv2
import numpy as np
from pathlib import Path

CLASS_NAMES = {
    0: "equipment",
    1: "worker"
}

def visualize_bbox(root_dir, split="train", index=0, filename=None):
    """
    可视化检测框
    :param root_dir: 数据集根目录
    :param split: 子集 (train/valid/test)
    :param index: 当 filename 为 None 时，使用索引获取图片
    :param filename: 指定图片文件名 (例如 '000000.png')
    """
    root = Path(root_dir)
    img_dir = root / split / "images"
    
    # 确定目标图片路径
    if filename:
        img_path = img_dir / filename
        if not img_path.exists():
            print(f"❌ 未找到指定文件: {img_path}")
            return
    else:
        # 原有的索引查找逻辑
        img_list = sorted(list(img_dir.glob("*.[jJ][pP][gG]"))) + \
                   sorted(list(img_dir.glob("*.[pP][nN][gG]")))
        if not img_list or index >= len(img_list):
            print(f"❌ 索引 {index} 超出范围")
            return
        img_path = img_list[index]
    
    # 获取对应标签路径
    lbl_path = root / split / "labels" / img_path.with_suffix(".txt").name
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return
        
    h_img, w_img, _ = img.shape
    
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = list(map(float, line.split()))
                if len(parts) < 5: continue
                
                cls = int(parts[0])
                xc, yc, w, h = parts[1:5]
                
                x1 = int((xc - w/2) * w_img)
                y1 = int((yc - h/2) * h_img)
                x2 = int((xc + w/2) * w_img)
                y2 = int((yc + h/2) * h_img)
                
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label_text = CLASS_NAMES.get(cls, f"ID:{cls}")
                cv2.putText(img, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"✅ 正在显示: {img_path.name}")
    else:
        print(f"⚠️ 未找到对应标签文件: {lbl_path}")

    cv2.imshow("Bounding Box Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATASET_PATH = "data7"
    
    # 用法 1：通过文件名指定 (推荐)
    # visualize_bbox(DATASET_PATH, split="train", filename="000000.png")
    
    # 用法 2：通过索引指定 (保持原有功能)
    visualize_bbox(DATASET_PATH, split="train", index=20)