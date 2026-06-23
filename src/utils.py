# utils.py
import cv2
import numpy as np
import queue

def draw_foot_ellipse(frame, foot_point, bbox=None, color=(0, 255, 0), 
                      alpha=0.3, radius=None, person_height=None):
    """
    在人的脚底绘制椭圆形的安全区域
    
    参数:
        frame: 要绘制的图像
        foot_point: 脚底点坐标 (x, y)
        bbox: 人的边界框 [x1, y1, x2, y2]，用于计算人的高度（可选）
        color: 椭圆颜色 (B, G, R)
        alpha: 透明度 (0-1)
        radius: 自定义半径，如果为None则自动计算
        person_height: 人的高度（像素），如果提供则用于计算椭圆大小
    
    返回:
        (radius_x, radius_y): 椭圆的半轴长度
    """
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        person_height = y2 - y1
    
    # 确定半径
    if radius is not None:
        # 使用自定义半径
        base_radius = radius
    else:
        # 从配置中获取基础半径，或使用默认值
        base_radius = 80  # 默认值
    
    # 根据人的高度调整椭圆大小（近大远小）
    if person_height is not None and person_height > 0:
        height_scale = max(0.5, min(2.0, person_height / 200))  # 以200像素为基准
    else:
        height_scale = 1.0
    
    radius_x = int(base_radius * height_scale)
    radius_y = int(base_radius * height_scale * 0.3)  # 垂直方向更扁
    
    # 创建椭圆遮罩
    overlay = frame.copy()
    center = (int(foot_point[0]), int(foot_point[1]))
    
    # 绘制填充椭圆（半透明）
    cv2.ellipse(overlay, center, (radius_x, radius_y), 0, 0, 360, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 绘制椭圆边框
    cv2.ellipse(frame, center, (radius_x, radius_y), 0, 0, 360, color, 2)
    
    return radius_x, radius_y

def compute_bev_size(src_pts, scale=1.0):
    """
    自动计算BEV尺寸
    """
    top_width = np.linalg.norm(src_pts[1] - src_pts[0])
    bottom_width = np.linalg.norm(src_pts[2] - src_pts[3])
    width = (top_width + bottom_width) / 2
    left_height = np.linalg.norm(src_pts[3] - src_pts[0])
    right_height = np.linalg.norm(src_pts[2] - src_pts[1])
    height = (left_height + right_height) / 2
    bev_w = int(width * scale)
    bev_h = int(height * scale)
    bev_w = bev_w if bev_w % 2 == 0 else bev_w + 1
    bev_h = bev_h if bev_h % 2 == 0 else bev_h + 1
    return bev_w, bev_h

def compute_bev(frame, M, bev_size):
    """
    计算BEV图像
    """
    return cv2.warpPerspective(frame, M, bev_size)


def draw_src_region(frame, src_pts):
    """
    绘制源区域多边形
    """
    pts = src_pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    return frame

def draw_bev_grid_overlay(bev_img, grid_size, color=(128, 128, 128)):
    """
    在BEV图像上绘制网格
    """
    h, w = bev_img.shape[:2]
    for x in range(0, w, grid_size):
        cv2.line(bev_img, (x, 0), (x, h), color, 1)
    for y in range(0, h, grid_size):
        cv2.line(bev_img, (0, y), (w, y), color, 1)
    return bev_img

def video_reader(cap, frame_queue, stop_event=None):
    """
    视频读取线程（鲁棒版）
    - 捕获 C 层崩溃（ffmpeg assertion 等）
    - 跳过零星坏帧；连续坏帧过多则退出
    """
    bad_streak = 0
    MAX_BAD_STREAK = 10  # 连续 10 帧坏 → 流已死，退出

    while cap.isOpened():
        if stop_event and stop_event.is_set():
            break
        try:
            ret, frame = cap.read()
        except BaseException:
            break

        if not ret or frame is None or frame.size == 0:
            bad_streak += 1
            if bad_streak >= MAX_BAD_STREAK:
                break
            continue

        bad_streak = 0

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

def draw_fps_on_image(img, fps_display, position=(10, 30)):
    """
    在图像上绘制FPS
    """
    if fps_display:
        cv2.putText(img, fps_display, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img
