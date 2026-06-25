import sys
import re
from pathlib import Path
import numpy as np
import cv2

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
RTSP_URL = "rtsp://118.140.130.26:8554/dahua1005189"

# ═══════════════════════════════════════════════════════════════
# 1. 数据解析（保持左下角到左上角的转换）
# ═══════════════════════════════════════════════════════════════
def load_mapping(filepath):
    data = []
    pattern = re.compile(r"(-?\d+),(-?\d+)\s+(-?[\d.]+),(-?[\d.]+)")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line_num, line in enumerate(lines, 1):
        line_str = line.strip()
        if not line_str or line_str.startswith("#") or "x,y" in line_str or "画面" in line_str:
            continue
        match = pattern.match(line_str)
        if match:
            raw_px, raw_py = int(match.group(1)), int(match.group(2))
            rx, ry = float(match.group(3)), float(match.group(4))
            # OpenCV 坐标
            px, py = raw_px, IMG_HEIGHT - raw_py
            data.append({
                "index": len(data),
                "line_content": line, # 保留原始行文本
                "cv_p": (px, py),
                "raw_p": (raw_px, raw_py),
                "world_p": (rx, ry)
            })
    return data

# ═══════════════════════════════════════════════════════════════
# 2. 核心标定与离群点识别
# ═══════════════════════════════════════════════════════════════
def calibrate_and_clean(data, thresh=1.5):
    pixel_pts = np.array([d["cv_p"] for d in data], dtype=np.float32)
    world_pts = np.array([d["world_p"] for d in data], dtype=np.float32)
    
    world_mean = world_pts.mean(axis=0)
    world_c = world_pts - world_mean

    H, mask = cv2.findHomography(pixel_pts, world_c, cv2.RANSAC, thresh)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(data), dtype=bool)

    # 计算重投影误差
    n = len(pixel_pts)
    ph = np.hstack([pixel_pts, np.ones((n, 1))])
    wp_h = (H @ ph.T).T
    wp = wp_h[:, :2] / wp_h[:, 2:] + world_mean
    errors = np.linalg.norm(wp - world_pts, axis=1)

    for i in range(len(data)):
        data[i]["error"] = errors[i]
        data[i]["is_inlier"] = inliers[i]
        
    return H, world_mean, inliers

# ═══════════════════════════════════════════════════════════════
# 3. 视频流绘制与实时叠加
# ═══════════════════════════════════════════════════════════════
def visualize_on_stream(data):
    """读取 RTSP 视频流并将标注点实时绘制在画面上"""
    print(f"\n[RTSP] 正在尝试连接视频流: {RTSP_URL}")
    print("提示: 画面弹出后，按 'Q' 键可退出播放。")
    
    cap = cv2.VideoCapture(RTSP_URL)
    
    # 调小缓冲区，降低实时流延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[Error] 无法打开 RTSP 视频流。请检查网络或 URL 是否有效。")
        return

    cv2.namedWindow("Calibration Points On-Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Points On-Stream", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] 未能获取到视频帧，正在重试...")
            continue
            
        # 确保画面尺寸一致
        if frame.shape[1] != IMG_WIDTH or frame.shape[0] != IMG_HEIGHT:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

        # 遍历所有点并绘制
        for d in data:
            # 注意：OpenCV 绘图使用的是其标准的左上角原点坐标系 (cv_p)
            cx, cy = d["cv_p"]
            raw_x, raw_y = d["raw_p"]
            
            if d["is_inlier"]:
                color = (0, 255, 0) # 内点绿色
                label = f"ID:{d['index']}"
            else:
                color = (0, 0, 255) # 离群点红色（警告色）
                label = f"ERR ID:{d['index']} ({d['error']:.1f}m)"

            # 画圆圈和中心点
            cv2.circle(frame, (cx, cy), 8, color, -1)
            cv2.circle(frame, (cx, cy), 15, color, 2)
            
            # 在圆圈旁书写文字标签
            cv2.putText(frame, label, (cx + 18, cy + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 渲染图例
        cv2.putText(frame, "GREEN: Inlier (Good)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "RED: Outlier (Bad/Need Check)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Calibration Points On-Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ═══════════════════════════════════════════════════════════════
# 4. 主程序入口
# ═══════════════════════════════════════════════════════════════
def main():
    script_dir = Path(__file__).resolve().parent
    input_file = script_dir / "371摄像头坐标映射_cleaned.txt"
    
    if not input_file.exists():
        print(f"[Error] 未找到源数据文件: {input_file}")
        sys.exit(1)

    # 1. 加载并运行标定过滤
    data = load_mapping(input_file)
    H, world_mean, inliers = calibrate_and_clean(data, thresh=1.5)
    
    # 2. 自动生成剔除离群点后的新txt文件
    clean_file = script_dir / "371摄像头坐标映射_cleaned.txt"
    outliers_count = len(data) - sum(inliers)
    
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write("# =========================================================\n")
        f.write("# 自动生成的标定控制点文件（已通过 RANSAC 剔除离群点）\n")
        f.write(f"# 原始总点数: {len(data)} | 已剔除噪声点: {outliers_count}\n")
        f.write("# =========================================================\n")
        f.write("画面左下角0,0\n")
        f.write("x,y\n")
        for d in data:
            if d["is_inlier"]:
                f.write(d["line_content"]) # 写入原本格式的一行
            else:
                f.write(f"# [剔除原因: 误差过大 {d['error']:.2f}米] {d['line_content']}")

    print(f"[成功] 过滤后的干净数据已写入: {clean_file.name}")
    print(f"       共保留了 {sum(inliers)} 个符合透视规则的控制点，注释掉了 {outliers_count} 个坏点。")

    # 3. 渲染视频流
    visualize_on_stream(data)

if __name__ == "__main__":
    main()