import sys
import re
from pathlib import Path
import numpy as np
import cv2

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# ═══════════════════════════════════════════════════════════════
# 1. 精确解析清洗后的数据（严格过滤 # 开头的剔除点）
# ═══════════════════════════════════════════════════════════════
def load_cleaned_mapping(filepath):
    data = []
    pattern = re.compile(r"(-?\d+),(-?\d+)\s+(-?[\d.]+),(-?[\d.]+)")
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 关键：如果行以 # 开头或者是表头，直接跳过（从而过滤掉离群噪声点）
            if not line or line.startswith("#") or "x,y" in line or "画面" in line:
                continue
                
            match = pattern.match(line)
            if match:
                raw_px = int(match.group(1))
                raw_py = int(match.group(2))
                rx = float(match.group(3))
                ry = float(match.group(4))
                
                # 转换为 OpenCV 的左上角原点系统
                px = raw_px
                py = IMG_HEIGHT - raw_py
                data.append((px, py, rx, ry))
                
    if len(data) < 4:
        raise ValueError(f"需要至少4个有效控制点，当前只解析到 {len(data)} 个")
    return data

# ═══════════════════════════════════════════════════════════════
# 2. 标定矩阵计算
# ═══════════════════════════════════════════════════════════════
def compute_final_calibration(pixel_pts, world_pts):
    world_mean = world_pts.mean(axis=0)
    world_c = world_pts - world_mean

    # 因为输入的数据已经通过之前的步骤洗干净了，这里直接用全点集最小二乘拟合最优矩阵
    H, _ = cv2.findHomography(pixel_pts, world_c, 0)
    
    # 评估最终剩余点集的 RMSE 精度
    n = len(pixel_pts)
    ph = np.hstack([pixel_pts, np.ones((n, 1))])
    wp_h = (H @ ph.T).T
    wp = wp_h[:, :2] / wp_h[:, 2:] + world_mean
    errors = np.linalg.norm(wp - world_pts, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    return H, world_mean, rmse

# ═══════════════════════════════════════════════════════════════
# 3. 反向投影与透视网格绘制可视化
# ═══════════════════════════════════════════════════════════════
def draw_perspective_grid(H, world_mean, pixel_pts):
    """
    在虚拟黑底画布（模拟视频画面）上，绘制基于真实物理世界等间距的透视网格。
    同时把控制点标在上面，直观展示位置。
    """
    # 创建 1920x1080 三通道图像画布
    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    # 定义物理网格范围（米）：以世界坐标均值为中心，向四周辐射 30 米
    grid_size_meter = 2.0  # 每个格子长宽代表真实世界 2 米
    range_meter = 30.0     # 绘制覆盖前后左右 30 目的范围
    
    x_lines = np.arange(-range_meter, range_meter + 0.1, grid_size_meter)
    y_lines = np.arange(-range_meter, range_meter + 0.1, grid_size_meter)

    print("\n[Grid Generator] 正在投影物理空间世界网格到像素空间...")

    # 1. 绘制纵向平行线（固定真实世界 X，延伸 Y）
    for xl in x_lines:
        pts_img = []
        for yl in np.linspace(-range_meter, range_meter, 100): # 细分曲线避免透视大畸变错位
            # 物理世界坐标转换为相机像素坐标
            v_src = np.array([xl, yl, 1.0], dtype=np.float64)
            v_pix = H_inv @ v_src
            px, py = v_pix[0] / v_pix[2], v_pix[1] / v_pix[2]
            
            # 过滤掉超出屏幕范围太远的点
            if -2000 < px < 4000 and -2000 < py < 4000:
                pts_img.append((int(px), int(py)))
        
        # 绘制这条物理平行线在相机里的投影
        for i in range(len(pts_img) - 1):
            cv2.line(canvas, pts_img[i], pts_img[i+1], (60, 60, 60), 1, cv2.LINE_AA)

    # 2. 绘制横向平行线（固定真实世界 Y，延伸 X）
    for yl in y_lines:
        pts_img = []
        for xl in np.linspace(-range_meter, range_meter, 100):
            v_src = np.array([xl, yl, 1.0], dtype=np.float64)
            v_pix = H_inv @ v_src
            px, py = v_pix[0] / v_pix[2], v_pix[1] / v_pix[2]
            if -2000 < px < 4000 and -2000 < py < 4000:
                pts_img.append((int(px), int(py)))
                
        for i in range(len(pts_img) - 1):
            cv2.line(canvas, pts_img[i], pts_img[i+1], (60, 60, 60), 1, cv2.LINE_AA)

    # 3. 在网格上叠加绘制当前的有效标定点
    for idx, (px, py) in enumerate(pixel_pts):
        cx, cy = int(px), int(py)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(canvas, str(idx), (cx + 8, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # 4. 渲染辅助文本信息说明
    cv2.putText(canvas, f"Perspective Grid Space (Each cell = {grid_size_meter}x{grid_size_meter}m)", 
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Notice how cells look LARGE in foreground and SMALL in background", 
                (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Press 'Q' to Exit Visualization", 
                (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)

    # 弹窗显示结果
    cv2.namedWindow("Perspective Geometry Grid View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Perspective Geometry Grid View", 1280, 720)
    cv2.imshow("Perspective Geometry Grid View", canvas)
    print("提示: 画面弹出后，焦点选中窗口按键盘上的 'Q' 键可安全退出。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ═══════════════════════════════════════════════════════════════
# 4. 主流程主入口
# ═══════════════════════════════════════════════════════════════
def main():
    script_dir = Path(__file__).resolve().parent
    # 自动锁定你的清理后文件路径
    cleaned_file = script_dir / "371摄像头坐标映射_cleaned.txt"
    
    if not cleaned_file.exists():
        print(f"[Error] 未能找到清洗后的控制点文件: {cleaned_file}")
        sys.exit(1)
        
    print(f"[Step 1] 正在加载清洗后的数据文件: {cleaned_file.name}")
    raw_data = load_cleaned_mapping(cleaned_file)
    
    pixel_pts = np.array([[d[0], d[1]] for d in raw_data], dtype=np.float32)
    world_pts = np.array([[d[2], d[3]] for d in raw_data], dtype=np.float32)
    
    print(f"[Step 2] 正在计算最终的数学变换矩阵 (当前有效控制点: {len(raw_data)} 个)...")
    H, world_mean, final_rmse = compute_final_calibration(pixel_pts, world_pts)
    
    print("=" * 60)
    print("  标定数学参数计算成功报告")
    print("=" * 60)
    print(f"  清洗后剩余控制点数 : {len(raw_data)}")
    print(f"  最终核心模型精度RMSE: {final_rmse:.3f} 米 ({final_rmse*100:.1f} 厘米)")
    print("=" * 60)
    
    # 将高精度的标定参数固化保存为 npz，供后期的 YOLO 定位推理脚本随调随用
    out_matrix_path = script_dir / "camera_calib_371.npz"
    np.savez(
        out_matrix_path,
        H=H.astype(np.float64),
        world_mean=world_mean.astype(np.float64),
        img_height=IMG_HEIGHT
    )
    print(f"[存储] 标定参数已成功序列化写入: {out_matrix_path.name}")
    
    # [Step 3] 开启直观的格网近大远小特点透视大片可视化
    draw_perspective_grid(H, world_mean, pixel_pts)

if __name__ == "__main__":
    main()