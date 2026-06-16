import os
import shutil
import random
from pathlib import Path

# 配置路径
BASE_DATA = "data3"
TARGET_ROOT = "data4"

def organize():
    # 1. 创建目标目录结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(TARGET_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, split, 'labels'), exist_ok=True)

    # 2. 收集所有数据对 (假设图片和标签文件名一一对应)
    # 遍历 train 和 val 目录，收集所有文件名
    data_pairs = []
    for split_type in ['train', 'val']:
        img_dir = Path(BASE_DATA) / 'images' / split_type
        lbl_dir = Path(BASE_DATA) / 'labels_processed' / split_type
        
        # 获取所有图片文件 (KITTI通常是.png)
        for img_path in img_dir.glob('*.png'):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                data_pairs.append((img_path, lbl_path))

    # 3. 打乱并重新划分
    random.shuffle(data_pairs)
    n = len(data_pairs)
    train_count = int(n * 0.8)
    val_count = int(n * 0.1)
    
    splits = {
        'train': data_pairs[:train_count],
        'valid': data_pairs[train_count:train_count+val_count],
        'test': data_pairs[train_count+val_count:]
    }

    # 4. 执行搬运
    for split_name, pairs in splits.items():
        print(f"正在整理 {split_name} 集合，共 {len(pairs)} 个文件对...")
        for img_src, lbl_src in pairs:
            shutil.copy(img_src, os.path.join(TARGET_ROOT, split_name, 'images', img_src.name))
            shutil.copy(lbl_src, os.path.join(TARGET_ROOT, split_name, 'labels', lbl_src.name))

    print(f"\n数据集整理完成！已保存至 {TARGET_ROOT} 目录下。")

if __name__ == "__main__":
    organize()