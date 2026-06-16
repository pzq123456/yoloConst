## 划分 kitti 数据

运行转换：确保先运行你之前的那个脚本生成 labels_processed。utils\convert_kitti.py
运行整理：执行此 organize_dataset.py 脚本。
检查验证：完成后，检查 data4 目录，确认是否符合 YOLO 预期的结构：

# 显示树状结构，但只显示文件夹
cmd /c "tree /A" | Where-Object { $_ -notmatch "\.\w+$" -and $_ -notmatch "^\s*$" }