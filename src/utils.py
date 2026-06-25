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

# ═══════════════════════════════════════════════════════════════
#  标定坐标转换（像素 ↔ 世界）
# ═══════════════════════════════════════════════════════════════

def pixel_to_world(px, py, H, world_mean):
    """OpenCV 像素坐标（左上角原点）→ UTM 世界坐标（米）"""
    p = np.array([px, py, 1.0])
    wp = H @ p
    w = wp[2]
    return wp[0] / w + world_mean[0], wp[1] / w + world_mean[1]


def pixel_to_local_meters(px, py, H):
    """OpenCV 像素坐标 → 局部物理米坐标（相对 world_mean，不加重心偏移）"""
    p = np.array([px, py, 1.0])
    wp = H @ p
    w = wp[2]
    return wp[0] / w, wp[1] / w


def world_to_pixel(wx, wy, H_inv, world_mean):
    """UTM 世界坐标（米）→ OpenCV 像素坐标"""
    wc = np.array([wx - world_mean[0], wy - world_mean[1], 1.0])
    pp = H_inv @ wc
    pz = pp[2]
    return pp[0] / pz, pp[1] / pz


def local_meters_to_bev(lx, ly, bev_range_x, bev_range_y, ppm):
    """
    局部物理米坐标 → BEV 图像像素索引（用于 grid_manager 和可视化）
    BEV 原点在左上角，(lx=0, ly=0) 对应 world_mean，映射到 BEV 中心
    """
    bev_px = lx * ppm + bev_range_x * ppm / 2.0
    bev_py = bev_range_y * ppm / 2.0 - ly * ppm   # Y 轴翻转（世界 Y↑ → 图像 Y↓）
    return bev_px, bev_py


# ═══════════════════════════════════════════════════════════════
#  相机坐标系变换（UTM → 相机对齐）
# ═══════════════════════════════════════════════════════════════

def local_to_camera(lx, ly, cam_forward, cam_right, cam_origin):
    """
    UTM-相对局部米坐标 → 相机对齐坐标系。
    cam_y = 相机前方（远离相机为正），cam_x = 相机右方。
    cam_origin = 相机地面投影的局部米坐标。
    """
    dx = lx - cam_origin[0]
    dy = ly - cam_origin[1]
    cam_x = dx * cam_right[0] + dy * cam_right[1]
    cam_y = dx * cam_forward[0] + dy * cam_forward[1]
    return cam_x, cam_y


def camera_to_local(cam_x, cam_y, cam_forward, cam_right, cam_origin):
    """相机对齐坐标 → UTM-相对局部米坐标"""
    lx = cam_x * cam_right[0] + cam_y * cam_forward[0] + cam_origin[0]
    ly = cam_x * cam_right[1] + cam_y * cam_forward[1] + cam_origin[1]
    return lx, ly


# ═══════════════════════════════════════════════════════════════
#  可视化工具
# ═══════════════════════════════════════════════════════════════

def draw_camera_perspective_grid(frame, H_inv, world_mean,
                                  cam_forward, cam_right, cam_origin,
                                  grid_cell_m=0.5, range_forward=25.0,
                                  range_back=15.0, range_side=20.0):
    """
    在摄像头画面上叠加标定地平面网格（半透明）。
    网格线沿相机前方/右方方向，每 grid_cell_m 米一条。
    """
    fw, fh = frame.shape[1], frame.shape[0]
    overlay = frame.copy()

    def _cam_point_to_pixel(cx, cy):
        lx, ly = camera_to_local(cx, cy, cam_forward, cam_right, cam_origin)
        wx = lx + world_mean[0]
        wy = ly + world_mean[1]
        wc = np.array([wx - world_mean[0], wy - world_mean[1], 1.0])
        pp = H_inv @ wc
        return pp[0] / pp[2], pp[1] / pp[2]

    def _draw_line(c1_start, c1_end, c2_vals, is_vertical, color, thickness):
        pts_list = []
        for c2 in c2_vals:
            pts = []
            for c1 in np.linspace(c1_start, c1_end, 80):
                if is_vertical:
                    cx, cy = c1, c2
                else:
                    cx, cy = c2, c1
                px, py = _cam_point_to_pixel(cx, cy)
                if -2000 < px < fw + 2000 and -2000 < py < fh + 2000:
                    pts.append((int(px), int(py)))
            if len(pts) >= 2:
                pts_list.append(pts)
                for i in range(len(pts) - 1):
                    cv2.line(overlay, pts[i], pts[i + 1], color, thickness,
                             cv2.LINE_AA)

    # 主要网格线（每 2m，白色，粗线）
    major_step = max(grid_cell_m * 4, 2.0)
    minor_step = grid_cell_m

    # 纵向线（固定 cam_x，沿 cam_y 延伸）— 代表"深度线"
    for cx in np.arange(-range_side, range_side + 0.01, minor_step):
        is_major = abs(cx % major_step) < 0.001 or abs(cx) < 0.001
        color = (200, 200, 200) if is_major else (80, 80, 80)
        thick = 1 if is_major else 1
        pts = []
        for cy in np.linspace(-range_back, range_forward, 100):
            px, py = _cam_point_to_pixel(cx, cy)
            if 0 <= px < fw and 0 <= py < fh:
                pts.append((int(px), int(py)))
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, thick, cv2.LINE_AA)

    # 横向线（固定 cam_y，沿 cam_x 延伸）— 代表"等高线"
    for cy in np.arange(-range_back, range_forward + 0.01, minor_step):
        is_major = abs(cy % major_step) < 0.001 or abs(cy) < 0.001
        color = (200, 200, 200) if is_major else (80, 80, 80)
        thick = 1 if is_major else 1
        pts = []
        for cx in np.linspace(-range_side, range_side, 100):
            px, py = _cam_point_to_pixel(cx, cy)
            if 0 <= px < fw and 0 <= py < fh:
                pts.append((int(px), int(py)))
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, thick, cv2.LINE_AA)

    # 半透明叠加
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_grid_highlight(frame, cam_x, cam_y, cam_forward, cam_right,
                         cam_origin, world_mean, H_inv,
                         color=(0, 255, 255), radius_m=0.5):
    """
    在摄像头画面上高亮一个检测目标对应的网格区域。
    在目标脚底绘制十字线 + 小圆圈表示其网格位置。
    """
    lx, ly = camera_to_local(
        cam_x, cam_y, cam_forward, cam_right, cam_origin,
    )
    wx = lx + world_mean[0]
    wy = ly + world_mean[1]
    wc = np.array([wx - world_mean[0], wy - world_mean[1], 1.0])
    pp = H_inv @ wc
    px, py = int(pp[0] / pp[2]), int(pp[1] / pp[2])

    # 十字线
    cv2.line(frame, (px - 15, py), (px + 15, py), color, 2, cv2.LINE_AA)
    cv2.line(frame, (px, py - 15), (px, py + 15), color, 2, cv2.LINE_AA)
    # 外圈
    cv2.circle(frame, (px, py), 18, color, 2, cv2.LINE_AA)


def draw_fps_on_image(img, fps_display, position=(10, 30)):
    """
    在图像上绘制FPS
    """
    if fps_display:
        cv2.putText(img, fps_display, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


# ═══════════════════════════════════════════════════════════════
#  视频画面叠加：占用网格 → 透视投影回摄像头画面
# ═══════════════════════════════════════════════════════════════

def draw_occupancy_overlay(frame, grid_manager, config):
    """
    将占用网格以半透明多边形方式叠加到摄像头原始画面上。
    只渲染有占用的单元格，通过 H_inv 将单元格角点反投影回像素坐标。

    颜色语义（BGR）：
      - 纯绿 (0,255,0)   = 仅人员
      - 纯红 (0,0,255)   = 仅设备
      - 品红 (255,0,255) = 人员+设备重叠（潜在危险）
    """
    fh, fw = frame.shape[:2]

    person_hits = grid_manager.person_hits
    equip_hits = grid_manager.equip_hits
    person_max = grid_manager.person_max
    equip_max = grid_manager.equip_max

    rows, cols = person_hits.shape
    cfg = config
    csz = cfg.GRID_CELL_SIZE_M
    half_x = cfg.GRID_X_RANGE_M / 2.0
    ppm = cfg.BEV_PIXELS_PER_METER

    # 预计算网格显示中的 cell 像素尺寸（用于决定颜色强度）
    grid_h, grid_w = grid_manager._display.shape[:2]
    cell_px_w = max(1.0, grid_w / float(cols))
    cell_px_h = max(1.0, grid_h / float(rows))

    overlay = frame.copy()

    for row in range(rows):
        for col in range(cols):
            p_val = person_hits[row, col]
            e_val = equip_hits[row, col]
            if p_val <= 0 and e_val <= 0:
                continue

            p_prob = min(p_val / float(person_max), 1.0)
            e_prob = min(e_val / float(equip_max), 1.0)

            # 决定颜色与透明度
            if p_prob > 0.01 and e_prob > 0.01:
                # 人+设备重叠 → 品红警告
                alpha = 0.25 + 0.2 * max(p_prob, e_prob)
                color = (255, 0, 255)
            elif p_prob > 0.01:
                alpha = 0.15 + 0.15 * p_prob
                color = (0, 255, 0)       # 绿 = 人
            else:
                alpha = 0.15 + 0.15 * e_prob
                color = (0, 0, 255)       # 红 = 设备

            # 单元格四角在相机坐标系中的位置
            cam_x_left   = (col - cols / 2.0) * csz
            cam_x_right  = (col + 1 - cols / 2.0) * csz
            cam_y_top    = cfg.GRID_Y_FAR_M - row * csz       # 远处
            cam_y_bottom = cfg.GRID_Y_FAR_M - (row + 1) * csz  # 近处

            corners_cam = [
                (cam_x_left,  cam_y_top),     # 左上（远左）
                (cam_x_right, cam_y_top),     # 右上（远右）
                (cam_x_right, cam_y_bottom),  # 右下（近右）
                (cam_x_left,  cam_y_bottom),  # 左下（近左）
            ]

            # 相机坐标 → 像素坐标
            pixel_pts = []
            for cx, cy in corners_cam:
                lx, ly = camera_to_local(
                    cx, cy, cfg.CAM_FORWARD, cfg.CAM_RIGHT, cfg.CAM_ORIGIN_WORLD)
                wc = np.array([lx, ly, 1.0])
                pp = cfg.H_inv @ wc
                px, py = pp[0] / pp[2], pp[1] / pp[2]
                if -2000 < px < fw + 2000 and -2000 < py < fh + 2000:
                    pixel_pts.append((int(px), int(py)))
                else:
                    pixel_pts.append(None)

            # 至少需要 3 个有效角点才能画多边形
            valid = [p for p in pixel_pts if p is not None]
            if len(valid) >= 3:
                pts_array = np.array(valid, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_array], color, cv2.LINE_AA)

            # 如果只有 1-2 个角点有效，画小圆点
            elif len(valid) > 0:
                for p in valid:
                    cv2.circle(overlay, p, 3, color, -1, cv2.LINE_AA)

    # 半透明叠加
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


# ═══════════════════════════════════════════════════════════════
#  网格显示：坐标系叠加
# ═══════════════════════════════════════════════════════════════

def draw_grid_coordinates(display, config):
    """
    在网格显示（BEV 像素分辨率）上绘制物理坐标系：
    - X 轴（cam_x = 0，纵线）和 Y 轴（cam_y = 0，横线）
    - 米刻度标记与数字标注
    - 方向标识（前/后/左/右）
    """
    h, w = display.shape[:2]
    cfg = config
    ppm = cfg.BEV_PIXELS_PER_METER
    half_x = cfg.GRID_X_RANGE_M / 2.0

    # ── 原点（cam_x=0, cam_y=0）在 BEV 像素中的位置 ──
    origin_x = int(half_x * ppm)                    # cam_x=0 → BEV 水平中心
    origin_y = int((cfg.GRID_Y_FAR_M - 0.0) * ppm)  # cam_y=0 的 BEV 行

    # ── 画 X 轴（纵线，cam_x = 0）─白色 ──
    if 0 <= origin_x < w:
        cv2.line(display, (origin_x, 0), (origin_x, h - 1),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "X=0", (origin_x + 4, origin_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── 画 Y 轴（横线，cam_y = 0）─白色 ──
    if 0 <= origin_y < h:
        cv2.line(display, (0, origin_y), (w - 1, origin_y),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Y=0", (origin_x + 6, origin_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── X 轴刻度（纵线，每隔 5m）─
    for cx_m in np.arange(-half_x, half_x + 0.01, 5.0):
        bx = int((cx_m + half_x) * ppm)
        if 0 <= bx < w:
            # 刻度线
            cv2.line(display, (bx, origin_y - 5), (bx, origin_y + 5),
                     (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display, f"{cx_m:.0f}m", (bx + 2, origin_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # ── Y 轴刻度（横线，每隔 5m）─
    for cy_m in np.arange(cfg.GRID_Y_NEAR_M, cfg.GRID_Y_FAR_M + 0.01, 5.0):
        by = int((cfg.GRID_Y_FAR_M - cy_m) * ppm)
        if 0 <= by < h:
            cv2.line(display, (origin_x - 5, by), (origin_x + 5, by),
                     (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display, f"{cy_m:.0f}m", (origin_x + 8, by + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # ── 方向标识（使用 ASCII 箭头，避免 OpenCV 渲染 Unicode 为 ???）──
    _draw_direction_label(display, "<- LEFT",    (10, h // 2),      (200, 200, 200))
    _draw_direction_label(display, "RIGHT ->",   (w - 80, h // 2),  (200, 200, 200))
    _draw_direction_label(display, f"FWD ^ ({cfg.GRID_Y_FAR_M:.0f}m)",
                          (w // 2 - 40, 14), (220, 220, 220))
    _draw_direction_label(display, f"v NEAR (cam, {cfg.GRID_Y_NEAR_M:.0f}m)",
                          (w // 2 - 50, h - 6), (220, 220, 220))


def _draw_direction_label(display, text, pos, color):
    """在 display 上绘制带阴影的方向标签"""
    x, y = pos
    cv2.putText(display, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(display, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
