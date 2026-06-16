import os
import shutil
from pathlib import Path

# 定义数据集路径
SOURCE_DIR = "data2"  # 你原有的数据集
TARGET_DIR = "data4"  # 刚才整理好的数据集

def merge():
    # 遍历 train, valid, test
    for split in ['train', 'valid', 'test']:
        for sub_dir in ['images', 'labels']:
            source_path = Path(SOURCE_DIR) / split / sub_dir
            target_path = Path(TARGET_DIR) / split / sub_dir
            
            if not source_path.exists():
                print(f"警告: 源路径不存在，跳过: {source_path}")
                continue
                
            # 获取所有文件
            files = list(source_path.glob('*'))
            print(f"正在将 {len(files)} 个文件从 {source_path} 合并到 {target_path}...")
            
            # 拷贝文件
            for file in files:
                # 使用 copy2 保留文件元数据
                shutil.copy2(file, target_path / file.name)
    
    print("\n合并完成！所有数据已成功整合至 data4。")

if __name__ == "__main__":
    merge()