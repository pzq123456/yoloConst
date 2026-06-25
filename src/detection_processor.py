# detection_processor.py
import cv2
import numpy as np
from utils import draw_foot_ellipse, pixel_to_local_meters, local_to_camera


class DetectionProcessor:
    """检测结果处理器 — 基于标定 H 矩阵的相机对齐坐标系提取 + 风险感知标注"""

    PERSON_FOOTPRINT_RADIUS_M = 0.6   # 人员站立区域半径（米），覆盖~9个网格单元

    def __init__(self, config):
        self.config = config
        self.H = config.H
        self.H_inv = config.H_inv
        self.world_mean = config.WORLD_MEAN
        self.cam_forward = config.CAM_FORWARD
        self.cam_right = config.CAM_RIGHT
        self.cam_origin = config.CAM_ORIGIN_WORLD
        self.cell_size_m = config.GRID_CELL_SIZE_M
        self.grid_cols = config.GRID_COLS
        self.grid_rows = config.GRID_ROWS
        self.ellipse_config = {
            'alpha': 0.25,
            'color': (0, 255, 0),
            'radius': 50,
        }

    def extract_world_coordinates(self, results):
        """
        提取相机对齐坐标系下的物理米坐标。

        返回:
            person_points:  [(cam_x, cam_y), ...]     人员脚底（相机坐标）
            vehicle_areas: [(cx1,cy1,cx2,cy2), ...]   车辆矩形（相机坐标）
            detections:    [(points_list, color)]      网格更新数据（相机坐标）
            person_data:   [{bbox, foot_img, foot_cam}, ...]
        """
        person_points = []
        vehicle_areas = []
        detections = []
        person_data = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, clss, confs):
                x1, y1, x2, y2 = box
                foot_px = (x1 + x2) / 2.0
                foot_py = y2

                # 像素 → UTM局部米 → 相机对齐坐标
                lx, ly = pixel_to_local_meters(foot_px, foot_py, self.H)
                cam_x, cam_y = local_to_camera(
                    lx, ly, self.cam_forward, self.cam_right, self.cam_origin,
                )

                if cls == self.config.PERSON_CLASS:
                    color = (0, 255, 0)
                    # 人员：圆形足迹区域（~0.4m半径）
                    pts = self._circle_footprint(
                        cam_x, cam_y, self.PERSON_FOOTPRINT_RADIUS_M, float(conf),
                    )
                    detections.append((pts, color))
                    person_points.append((cam_x, cam_y))
                    person_data.append({
                        'bbox': box,
                        'foot_img': (foot_px, foot_py),
                        'foot_cam': (cam_x, cam_y),  # 相机对齐坐标
                        'foot_local': (lx, ly),       # UTM局部米（备用）
                    })

                else:
                    color = (0, 0, 255)
                    # 车辆：估算物理占用区域（相机坐标）
                    cx_min, cy_min, cx_max, cy_max = \
                        self._estimate_vehicle_camera_region(box)
                    vehicle_areas.append((cx_min, cy_min, cx_max, cy_max))

                    # 车辆占用网格单元
                    for gx, gy in self._get_cells_in_camera_region(
                        cx_min, cy_min, cx_max, cy_max,
                    ):
                        detections.append(([(gx, gy, float(conf))], color))

        return person_points, vehicle_areas, detections, person_data

    def annotate_frame(self, frame, person_data, risks, alarm_triggered):
        """根据风险等级绘制椭圆（保留原逻辑）"""
        for pd_, risk in zip(person_data, risks):
            if alarm_triggered and risk >= 0.1:
                color = (0, 0, 255)
            elif risk >= 0.15:
                color = (0, 140, 255)
            elif risk >= 0.01:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            draw_foot_ellipse(
                frame, pd_['foot_img'], bbox=pd_['bbox'],
                color=color,
                alpha=self.ellipse_config['alpha'],
                radius=self.ellipse_config['radius'],
            )

    # ── 足迹生成 ──────────────────────────────────────────────

    def _circle_footprint(self, cx, cy, radius_m, conf):
        """
        以单元格为中心的圆形足迹生成。
        直接遍历单元格索引，检查单元格中心是否在半径内，
        避免坐标采样法在 radius≈cell_size 时退化为十字线。
        """
        cells = []
        cfg = self.config
        csz = self.cell_size_m

        center_col = cfg.cam_to_grid_col(cx)
        center_row = cfg.cam_to_grid_row(cy)
        r_cells = int(radius_m / csz) + 1

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                col = center_col + dc
                row = center_row + dr
                if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
                    # 单元格中心在相机坐标系中的位置
                    gx = (col + 0.5 - self.grid_cols / 2.0) * csz
                    gy = cfg.GRID_Y_FAR_M - (row + 0.5) * csz
                    if (gx - cx) ** 2 + (gy - cy) ** 2 <= radius_m ** 2:
                        cells.append((gx, gy, conf))
        return cells

    # ── 车辆估算 ──────────────────────────────────────────────

    def _estimate_vehicle_camera_region(self, bbox):
        """
        估计车辆在相机对齐坐标系中的占用矩形。
        宽度：bbox 左右边 → 地平面投影距离。
        长度：由宽度推算（车辆长宽比 ~1.8:1），比 bbox 高度投影更稳定。
        """
        x1, y1, x2, y2 = bbox
        mid_px = (x1 + x2) / 2.0

        # 底边中点（车辆近端接触地面）→ 相机坐标
        near_lx, near_ly = pixel_to_local_meters(mid_px, y2, self.H)
        near_cx, near_cy = local_to_camera(
            near_lx, near_ly, self.cam_forward, self.cam_right, self.cam_origin)

        # 宽度：bbox 左右边在地面的投影距离（各用自己的 ly）
        left_lx, left_ly = pixel_to_local_meters(x1, y2, self.H)
        right_lx, right_ly = pixel_to_local_meters(x2, y2, self.H)
        left_cx, _ = local_to_camera(
            left_lx, left_ly, self.cam_forward, self.cam_right, self.cam_origin)
        right_cx, _ = local_to_camera(
            right_lx, right_ly, self.cam_forward, self.cam_right, self.cam_origin)
        phys_width = abs(right_cx - left_cx) * 0.85  # bbox 略宽于实际车宽
        phys_width = max(1.5, min(3.0, phys_width))

        # 长度：近大远小 — 用 bbox 高度估计深度，但用宽度约束上限
        far_lx, far_ly = pixel_to_local_meters(mid_px, y1, self.H)
        _, far_cy = local_to_camera(
            far_lx, far_ly, self.cam_forward, self.cam_right, self.cam_origin)
        depth_from_height = abs(far_cy - near_cy)
        length_from_width = phys_width * 1.8  # 典型车长宽比

        # 取两者中较小值（bbox 高度投影往往高估，宽度估计更可靠）
        phys_length = min(depth_from_height, length_from_width * 1.5)
        phys_length = max(2.5, min(7.0, phys_length))

        # 中心：近端往后半个车长
        cy_center = near_cy + phys_length / 2.0

        return (near_cx - phys_width / 2.0, cy_center - phys_length / 2.0,
                near_cx + phys_width / 2.0, cy_center + phys_length / 2.0)

    def _get_cells_in_camera_region(self, cx_min, cy_min, cx_max, cy_max):
        """相机坐标系矩形 → 覆盖的网格单元中心坐标"""
        cells = []
        csz = self.cell_size_m
        cfg = self.config

        start_col = cfg.cam_to_grid_col(cx_min)
        end_col = cfg.cam_to_grid_col(cx_max)
        # row: cy 越大 = 越远 = 行号越小 → 用 cy_max 取 start（小行号）
        start_row = cfg.cam_to_grid_row(cy_max)
        end_row = cfg.cam_to_grid_row(cy_min)

        for col in range(
            max(0, start_col), min(self.grid_cols, end_col + 1),
        ):
            for row in range(
                max(0, start_row), min(self.grid_rows, end_row + 1),
            ):
                # 单元中心的相机坐标
                gx = (col + 0.5 - self.grid_cols / 2.0) * csz
                gy = cfg.GRID_Y_FAR_M - (row + 0.5) * csz
                cells.append((gx, gy))
        return cells
