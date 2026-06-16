import os

# 映射逻辑
# 0: car, 1: van, 2: truck -> 0 (equipment)
# 3: pedestrian, 4: Person_sitting -> 1 (worker)
# 5: cyclist, 6: tram, 7: misc -> 忽略
MAP_DICT = {
    0: 0, 1: 0, 2: 0,
    3: 1, 4: 1
}

def convert_labels(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                original_class = int(parts[0])
                
                # 如果在映射字典中，则保留并修改 ID
                if original_class in MAP_DICT:
                    new_class = MAP_DICT[original_class]
                    new_line = f"{new_class} {' '.join(parts[1:])}\n"
                    f_out.write(new_line)

if __name__ == "__main__":
    # 请根据实际情况设置路径
    # 注意：运行前请备份原始 labels 文件夹！
    base_dir = "data3/labels"
    target_dir = "data3/labels_processed" # 建议输出到新目录，防止误操作
    
    # 分别处理 train 和 val
    for split in ['train', 'val']:
        in_path = os.path.join(base_dir, split)
        out_path = os.path.join(target_dir, split)
        print(f"正在转换 {split} 数据...")
        convert_labels(in_path, out_path)
    
    print("转换完成！请检查 labels_processed 目录。")