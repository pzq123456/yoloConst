# utils.py
import cv2
import numpy as np
import queue
import time

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

# utils.py
def update_grid(grid_map, detections, grid_size, decay=0.95):
    """
    更新占用网格
    detections: list of (points_list, color)
        points_list: 包含一个或多个点 [(x1,y1), (x2,y2), ...]
        color: (B,G,R)
    """
    cv2.addWeighted(grid_map, decay, np.zeros_like(grid_map), 1 - decay, 0, grid_map)
    h, w = grid_map.shape[:2]
    
    for points, color in detections:
        # 如果points是单个点的列表，占用一个网格
        # 如果是多个点，每个点占用一个网格（车辆占用多个网格）
        for point in points:
            if len(point) == 2:
                px, py = point
                px = np.clip(int(px), 0, w - 1)
                py = np.clip(int(py), 0, h - 1)
                gx = (px // grid_size) * grid_size
                gy = (py // grid_size) * grid_size
                gx = min(gx, w - grid_size)
                gy = min(gy, h - grid_size)
                cv2.rectangle(grid_map, (gx, gy), (gx + grid_size, gy + grid_size), color, -1)

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

def point_segment_distance(p, a, b):
    """
    计算点p到线段ab的最短欧氏距离（像素）
    """
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = wx * vx + wy * vy
    if c1 <= 0:
        return np.hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return np.hypot(px - bx, py - by)
    ratio = c1 / c2
    proj_x = ax + ratio * vx
    proj_y = ay + ratio * vy
    return np.hypot(px - proj_x, py - proj_y)

def video_reader(cap, frame_queue):
    """
    视频读取线程
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

def compute_fps(fps_counter, fps_timer):
    """
    计算FPS
    """
    fps_counter += 1
    current_time = time.time()
    if current_time - fps_timer >= 1.0:
        fps_display = f"FPS: {fps_counter:.1f}"
        fps_counter = 0
        fps_timer = current_time
        return fps_display, fps_counter, fps_timer
    return None, fps_counter, fps_timer

def draw_fps_on_image(img, fps_display, position=(10, 30)):
    """
    在图像上绘制FPS
    """
    if fps_display:
        cv2.putText(img, fps_display, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img

# utils.py 添加以下函数

def point_in_rectangle(point, rect):
    """
    判断点是否在矩形内
    
    参数:
        point: (x, y)
        rect: (x1, y1, x2, y2)
    """
    px, py = point
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def points_to_bbox(points):
    """
    将点列表转换为边界框
    
    参数:
        points: [(x1,y1), (x2,y2), ...]
    
    返回:
        (x1, y1, x2, y2)
    """
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))