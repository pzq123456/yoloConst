import os
import shutil
import random
from pathlib import Path

# 配置
SOURCES = ["data5", "data6"]
TARGET = "data7"
RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}

def merge_and_split():
    # 1. 准备数据池
    all_data_pairs = []
    
    for src in SOURCES:
        src_path = Path(src)
        # 遍历该数据集中可能存在的 train/valid/test 目录
        for split_dir in ['train', 'valid', 'test']:
            img_dir = src_path / split_dir / "images"
            lbl_dir = src_path / split_dir / "labels"
            
            if not img_dir.exists(): continue
            
            for img_file in img_dir.glob("*"):
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    all_data_pairs.append((img_file, lbl_file))
    
    # 2. 打乱并分配
    random.shuffle(all_data_pairs)
    total = len(all_data_pairs)
    train_end = int(total * RATIOS["train"])
    valid_end = train_end + int(total * RATIOS["valid"])
    
    splits = {
        "train": all_data_pairs[:train_end],
        "valid": all_data_pairs[train_end:valid_end],
        "test": all_data_pairs[valid_end:]
    }
    
    # 3. 执行分发
    for split_name, pairs in splits.items():
        print(f"正在构建 {split_name} 集，包含 {len(pairs)} 个样本...")
        target_img_dir = Path(TARGET) / split_name / "images"
        target_lbl_dir = Path(TARGET) / split_name / "labels"
        
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, target_img_dir / img_src.name)
            shutil.copy2(lbl_src, target_lbl_dir / lbl_src.name)

    print(f"\n✅ 数据集融合完成，已生成 data7，结构如下：")
    print(f"训练集: {len(splits['train'])} | 验证集: {len(splits['valid'])} | 测试集: {len(splits['test'])}")

if __name__ == "__main__":
    merge_and_split()