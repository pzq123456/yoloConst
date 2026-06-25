# config.py
import os
import json
import numpy as np
import cv2


# ── 模拟上游数据接口的本地 JSON 文件 ──
# 上游接口就绪后，删除此文件，改为从上游接口获取数据。
_MOCK_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "cameras.mock.json",
)


class Config:
    """配置管理类 — 基于标定单应性矩阵的物理米坐标系"""

    # ── 不从 JSON 读取的内部参数（保持硬编码） ──
    _MODEL_PATH = "../model/yolo26s_mocs_20260623_0216.pt"
    _CONF_THRESH = 0.55
    _IMGSZ = 640
    _PERSON_CLASS = 1
    _PERSON_MAX_HITS = 3
    _EQUIPMENT_MAX_HITS = 10
    _PERSON_DECAY = 1.0
    _EQUIPMENT_DECAY = 0.25
    _BEV_PIXELS_PER_METER = 25
    _DEBUG_MODE = False

    def __init__(self, camera_id=None):
        # ── 先设硬编码默认值（作为 JSON 缺失时的回退）──
        self.RTSP_URL = "rtsp://118.140.130.26:8554/dahua1005189"
        self.MODEL_PATH = self._MODEL_PATH

        calib_dir = os.path.join(os.path.dirname(__file__), "..", "calibration")
        self.CALIB_PATH = os.path.join(calib_dir, "camera_calib_371.npz")
        self.H = None
        self.H_inv = None
        self.WORLD_MEAN = None
        self.CAM_FORWARD = None
        self.CAM_RIGHT = None
        self.CAM_ORIGIN_WORLD = None

        self.GRID_CELL_SIZE_M = 0.5
        self.EQUIPMENT_DANGER_RADIUS_M = 3.0
        self.GRID_X_RANGE_M = 40.0
        self.GRID_Y_NEAR_M = -5.0
        self.GRID_Y_FAR_M = 30.0
        self.BEV_PIXELS_PER_METER = self._BEV_PIXELS_PER_METER

        self.CONF_THRESH = self._CONF_THRESH
        self.IMGSZ = self._IMGSZ
        self.PERSON_CLASS = self._PERSON_CLASS
        self.ALARM_CONTINUOUS_FRAMES = 3
        self.ALARM_RISK_THRESHOLD = 0.15

        self.PERSON_MAX_HITS = self._PERSON_MAX_HITS
        self.EQUIPMENT_MAX_HITS = self._EQUIPMENT_MAX_HITS
        self.PERSON_DECAY = self._PERSON_DECAY
        self.EQUIPMENT_DECAY = self._EQUIPMENT_DECAY

        self.DEBUG_MODE = self._DEBUG_MODE
        self.STREAM_WIDTH = None
        self.STREAM_HEIGHT = None

        self.M = None
        self.BEV_W = None
        self.BEV_H = None
        self.GRID_COLS = None
        self.GRID_ROWS = None
        self.DANGER_RADIUS_CELLS = None
        self.SRC_PTS = np.array(
            [[200, 100], [1700, 100], [1850, 1050], [70, 1050]],
            dtype=np.float32,
        )

        # ── 尝试从 JSON 覆盖（模拟上游数据接口）──
        self._camera_meta = {}
        self._try_load_json(camera_id)

    def _try_load_json(self, camera_id=None):
        """从本地 mock JSON 加载摄像头配置，覆盖硬编码默认值。

        上游接口就绪后，此方法替换为从上游接口获取数据。
        """
        if not os.path.exists(_MOCK_JSON_PATH):
            return

        with open(_MOCK_JSON_PATH, "r", encoding="utf-8") as f:
            cameras = json.load(f)

        if not cameras:
            return

        # 选择目标摄像头：指定 ID > 第一路 > 跳过
        target = None
        if camera_id:
            for cam in cameras:
                if cam.get("camera_id") == camera_id:
                    target = cam
                    break
        else:
            target = cameras[0]

        if target is None:
            return

        # ── 覆盖配置 ──
        self.RTSP_URL = target.get("rtsp_url", self.RTSP_URL)

        res = target.get("resolution", {})
        self.STREAM_WIDTH = res.get("width", self.STREAM_WIDTH)
        self.STREAM_HEIGHT = res.get("height", self.STREAM_HEIGHT)

        calib = target.get("calibration", {})
        if "H" in calib:
            self.H = np.array(calib["H"], dtype=np.float64)
        if "world_mean" in calib:
            self.WORLD_MEAN = np.array(calib["world_mean"], dtype=np.float64)

        grid = target.get("grid", {})
        if "cell_size_m" in grid:
            self.GRID_CELL_SIZE_M = float(grid["cell_size_m"])
        if "x_range_m" in grid:
            self.GRID_X_RANGE_M = float(grid["x_range_m"])
        if "y_near_m" in grid:
            self.GRID_Y_NEAR_M = float(grid["y_near_m"])
        if "y_far_m" in grid:
            self.GRID_Y_FAR_M = float(grid["y_far_m"])

        risk = target.get("risk", {})
        if "equipment_danger_radius_m" in risk:
            self.EQUIPMENT_DANGER_RADIUS_M = float(
                risk["equipment_danger_radius_m"],
            )
        if "alarm_continuous_frames" in risk:
            self.ALARM_CONTINUOUS_FRAMES = int(risk["alarm_continuous_frames"])
        if "alarm_risk_threshold" in risk:
            self.ALARM_RISK_THRESHOLD = float(risk["alarm_risk_threshold"])

        # 保存元信息（日志/报警溯源用）
        self._camera_meta = {
            "camera_id": target.get("camera_id", ""),
            "camera_name": target.get("camera_name", ""),
            "alarm_push_url": target.get("alarm_push_url"),
        }

    def initialize_calib(self):
        """加载标定矩阵并初始化所有派生参数（含相机坐标系对齐）

        H 和 WORLD_MEAN 优先从上游接口（当前为 JSON mock）获取；
        如果未提供则回退到本地 .npz 文件。
        """
        if self.H is None or self.WORLD_MEAN is None:
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