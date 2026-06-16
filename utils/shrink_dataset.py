import os
import shutil
import random
from pathlib import Path

# 配置
SOURCE_DATA = "data4"  # 大数据集
TARGET_DATA = "data5"  # 小数据集
TOTAL_SAMPLES = 1000   # 你希望最终保留的总张数

def get_label_count(label_path):
    """计算单个标签文件中的目标个数"""
    if not label_path.exists():
        return 0
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return len([line for line in lines if line.strip()])

def shrink():
    all_pairs = []
    
    # 遍历所有分区（train/valid/test）收集图片及其标签个数
    for split in ['train', 'valid', 'test']:
        img_dir = Path(SOURCE_DATA) / split / "images"
        lbl_dir = Path(SOURCE_DATA) / split / "labels"
        
        for img_path in img_dir.glob('*'):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                count = get_label_count(lbl_path)
                all_pairs.append({
                    'img': img_path,
                    'lbl': lbl_path,
                    'split': split,
                    'count': count
                })

    # 核心逻辑：按标注数量从大到小排序，并给予一定程度的随机性
    # 先按 count 排序，让“标注密集”的排在前面
    all_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # 取出前 X 个标注最密集的图片，或者使用加权抽样
    # 这里我们简单取标注数量最多的前 TOTAL_SAMPLES 张
    selected = all_pairs[:TOTAL_SAMPLES]
    
    # 执行搬运
    print(f"准备将 {len(selected)} 张标注最丰富的图片复制到 {TARGET_DATA}...")
    for item in selected:
        dest_img_dir = Path(TARGET_DATA) / item['split'] / "images"
        dest_lbl_dir = Path(TARGET_DATA) / item['split'] / "labels"
        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        shutil.copy2(item['img'], dest_img_dir / item['img'].name)
        shutil.copy2(item['lbl'], dest_lbl_dir / item['lbl'].name)
    
    print("✅ 数据集瘦身完成！")

if __name__ == "__main__":
    shrink()