import os
from pathlib import Path

def check_integrity(root_dir):
    root = Path(root_dir)
    print(f"🔍 正在校验数据集完整性: {root_dir}\n" + "="*40)
    
    total_issues = 0
    
    for split in ['train', 'valid']:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        
        # 获取图片列表
        img_files = list(img_dir.glob("*"))
        
        for img_path in img_files:
            # 1. 检查标签文件是否存在
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                print(f"❌ [缺失标签] 图片: {img_path.name} 没有对应的标签文件")
                total_issues += 1
                continue
            
            # 2. 检查标签文件是否为空
            if lbl_path.stat().st_size == 0:
                print(f"⚠️ [空标签文件] {lbl_path.name} (该图片被视为背景图)")
                total_issues += 1
                continue
                
            # 3. 检查标签格式是否合规 (可选：简单检查是否有内容)
            with open(lbl_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"⚠️ [无效标签] {lbl_path.name} 内容为空")
                    total_issues += 1
    
    if total_issues == 0:
        print("✅ 数据集完整性校验通过！所有图片均有对应且有效的标签。")
    else:
        print(f"\n❌ 校验结束，共发现 {total_issues} 个潜在问题，请根据上述日志修复。")

if __name__ == "__main__":
    # 指向你刚才整理好的 data6 目录
    check_integrity("data5")