import os
import shutil
from pathlib import Path

# 定义映射逻辑
# Equipment: 0,1,2,3,5,6,7,8,9,11 -> 0
# Worker: 12 -> 1
# Ignore: 4, 10
MAP_DICT = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 11: 0, 12: 1}

def process_dataset(source_root, target_root):
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    for split in ['train', 'valid']:
        # 1. 创建目标目录结构
        img_dest = target_root / split / "images"
        lbl_dest = target_root / split / "labels"
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        # 2. 复制图片
        src_img_dir = source_root / split / "images"
        for img_file in src_img_dir.glob("*"):
            shutil.copy2(img_file, img_dest / img_file.name)
            
        # 3. 处理标签并转换
        src_lbl_dir = source_root / split / "labels"
        for lbl_file in src_lbl_dir.glob("*.txt"):
            with open(lbl_file, 'r') as f_in, open(lbl_dest / lbl_file.name, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    old_cls = int(parts[0])
                    if old_cls in MAP_DICT:
                        new_cls = MAP_DICT[old_cls]
                        new_line = f"{new_cls} {' '.join(parts[1:])}\n"
                        f_out.write(new_line)
        print(f"✅ {split} 数据处理完成。")

if __name__ == "__main__":
    # 假设你的原数据在 "data" 目录下
    process_dataset("data", "data6")
    print("所有转换已保存至 data6 文件夹。")