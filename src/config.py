# config.py
import os
import numpy as np
import cv2

class Config:
    """配置管理类 — 基于标定单应性矩阵的物理米坐标系"""
    def __init__(self):
        # self.RTSP_URL = "rtsp://118.140.130.26:8554/dahua1003362"
        self.RTSP_URL = "rtsp://118.140.130.26:8554/dahua1005189" # 317
        self.MODEL_PATH = "../model/yolo26s_mocs_20260623_0216.pt"

        # ── 标定参数 ──
        calib_dir = os.path.join(os.path.dirname(__file__), "..", "calibration")
        self.CALIB_PATH = os.path.join(calib_dir, "camera_calib_371.npz")
        self.H = None           # 3x3 单应性矩阵（OpenCV像素 → 中心化UTM世界米）
        self.H_inv = None       # 逆矩阵（世界 → 像素）
        self.WORLD_MEAN = None  # UTM 坐标均值 [x, y]

        # ── 相机坐标系基向量（UTM → 相机对齐）──
        self.CAM_FORWARD = None  # 相机前方单位向量 [fx, fy]（图像↑对应世界方向）
        self.CAM_RIGHT = None    # 相机右方单位向量 [rx, ry]
        self.CAM_ORIGIN_WORLD = None  # 相机地面投影的局部米坐标 (lx, ly)

        # ── 物理网格参数（米制）──
        self.GRID_CELL_SIZE_M = 0.5            # 每个网格单元 = 0.5m × 0.5m
        self.EQUIPMENT_DANGER_RADIUS_M = 3.0   # 设备危险半径（米）
        self.GRID_X_RANGE_M = 40.0             # 左右覆盖 ±20m
        self.GRID_Y_NEAR_M = -5.0              # 网格底部（相机后方，cam_y 最小值）
        self.GRID_Y_FAR_M = 30.0               # 网格顶部（相机前方，cam_y 最大值）
        self.BEV_PIXELS_PER_METER = 25         # 可视化：1 米 = 25 像素

        # ── 检测参数 ──
        self.CONF_THRESH = 0.55
        self.IMGSZ = 640
        self.PERSON_CLASS = 1
        self.ALARM_CONTINUOUS_FRAMES = 3       # 报警防抖帧数
        self.ALARM_RISK_THRESHOLD = 0.15       # 风险值阈值 [0, 1]

        # ── Hit Counter 占用网格参数（置信度加权，抗闪烁）──
        self.PERSON_MAX_HITS = 3               # 人员：~3帧高置信度检测确认
        self.EQUIPMENT_MAX_HITS = 10           # 设备：~10帧确认
        self.PERSON_DECAY = 1.0               # 人员：每帧未检测到 -1.0（快速冷却）
        self.EQUIPMENT_DECAY = 0.25            # 设备：每帧未检测到 -0.25（缓慢衰减，~2.5s@10fps）

        # ── 显示 / 流参数 ──
        self.DEBUG_MODE = False                # 是否显示 BEV 窗口
        self.STREAM_WIDTH = None               # 流分辨率（None=自动探测）
        self.STREAM_HEIGHT = None

        # ── 运行时计算属性 ──
        self.M = None           # 保留旧透视变换（debug BEV warp 用）
        self.BEV_W = None
        self.BEV_H = None
        self.GRID_COLS = None
        self.GRID_ROWS = None
        self.DANGER_RADIUS_CELLS = None
        self.SRC_PTS = np.array(               # 保留旧 SRC_PTS（debug BEV warp 用）
            [[200, 100], [1700, 100], [1850, 1050], [70, 1050]],
            dtype=np.float32,
        )

    def initialize_calib(self):
        """加载标定矩阵并初始化所有派生参数（含相机坐标系对齐）"""
        calib = np.load(self.CALIB_PATH)
        self.H = calib["H"].astype(np.float64)
        self.WORLD_MEAN = calib["world_mean"].astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)

        # ── 计算相机坐标系基向量 ──
        # 图像中心往上 100px → "相机前方"方向（远离相机）
        img_cx, img_cy = 960.0, 540.0
        p_center = np.array([img_cx, img_cy, 1.0])
        p_up = np.array([img_cx, img_cy - 100.0, 1.0])  # 图像上方
        p_right = np.array([img_cx + 100.0, img_cy, 1.0])

        wc = self.H @ p_center
        wu = self.H @ p_up
        wr = self.H @ p_right

        cx, cy = wc[0] / wc[2], wc[1] / wc[2]
        ux, uy = wu[0] / wu[2], wu[1] / wu[2]
        rx, ry = wr[0] / wr[2], wr[1] / wr[2]

        fwd = np.array([ux - cx, uy - cy])
        fwd /= np.linalg.norm(fwd)
        rgt = np.array([rx - cx, ry - cy])
        rgt /= np.linalg.norm(rgt)
        # 正交化：right 减去在 forward 上的投影
        rgt = rgt - np.dot(rgt, fwd) * fwd
        rgt /= np.linalg.norm(rgt)

        self.CAM_FORWARD = fwd
        self.CAM_RIGHT = rgt

        # 相机地面投影原点：图像底部中心的局部米坐标
        p_bottom = np.array([img_cx, 1000.0, 1.0])
        wb = self.H @ p_bottom
        self.CAM_ORIGIN_WORLD = np.array([
            wb[0] / wb[2], wb[1] / wb[2],
        ])

        # ── 物理网格尺寸 ──
        grid_y_range = self.GRID_Y_FAR_M - self.GRID_Y_NEAR_M
        self.GRID_COLS = int(self.GRID_X_RANGE_M / self.GRID_CELL_SIZE_M)
        self.GRID_ROWS = int(grid_y_range / self.GRID_CELL_SIZE_M)
        self.DANGER_RADIUS_CELLS = int(
            self.EQUIPMENT_DANGER_RADIUS_M / self.GRID_CELL_SIZE_M,
        )

        # 网格行索引参数（cam_y → row）
        self._grid_y_scale = 1.0 / self.GRID_CELL_SIZE_M
        self._grid_y_offset = self.GRID_Y_NEAR_M

        # BEV 可视化画布 = 物理范围 × 像素密度
        self.BEV_W = int(self.GRID_X_RANGE_M * self.BEV_PIXELS_PER_METER)
        self.BEV_H = int(grid_y_range * self.BEV_PIXELS_PER_METER)

        # 保留旧 M 矩阵用于 debug BEV warp（warpPerspective 可视化）
        dst_pts = np.array([
            [0, 0], [self.BEV_W, 0],
            [self.BEV_W, self.BEV_H], [0, self.BEV_H],
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, dst_pts)

    def cam_to_grid_row(self, cam_y):
        """相机前方距离 → 网格行索引（0=顶部/远处, rows-1=底部/近处）"""
        row = int((self.GRID_Y_FAR_M - cam_y) * self._grid_y_scale)
        return max(0, min(self.GRID_ROWS - 1, row))

    def cam_to_grid_col(self, cam_x):
        """相机右侧距离 → 网格列索引（0=左侧, cols-1=右侧）"""
        col = int(cam_x * self._grid_y_scale + self.GRID_COLS / 2.0)
        return max(0, min(self.GRID_COLS - 1, col))