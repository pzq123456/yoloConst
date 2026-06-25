# frame_processor.py
import numpy as np
from grid_manager import GridManager
from utils import (draw_camera_perspective_grid, draw_occupancy_overlay,
                   draw_grid_coordinates)


class FrameProcessor:
    """帧处理器 — 工地人-设备距离报警管线（相机对齐坐标系）"""

    def __init__(self, config, model, bev_processor,
                 detection_processor, alarm_manager):
        self.config = config
        self.model = model
        self.bev_processor = bev_processor
        self.detection_processor = detection_processor
        self.alarm_manager = alarm_manager
        self.grid_manager = GridManager(config)

    def process_frame(self, frame):
        # 1. 模型推理
        results = self.model.track(
            frame, persist=True,
            conf=self.config.CONF_THRESH, verbose=False,
            imgsz=self.config.IMGSZ, iou=0.5,
        )

        # 2. 绘制原始图像 + 标定透视网格叠加
        annotated_frame = results[0].plot()
        draw_camera_perspective_grid(
            annotated_frame,
            self.config.H_inv, self.config.WORLD_MEAN,
            self.config.CAM_FORWARD, self.config.CAM_RIGHT,
            self.config.CAM_ORIGIN_WORLD,
            grid_cell_m=self.config.GRID_CELL_SIZE_M,
        )

        # 3. BEV（debug用）
        bev_img = self.bev_processor.process(frame)

        # 4. 提取相机对齐坐标（基于标定 H 矩阵）
        person_points, vehicle_areas, detections, person_data = \
            self.detection_processor.extract_world_coordinates(results)

        # 5. 更新占用网格 + 计算风险
        self.grid_manager.update(detections)
        danger_map = self.grid_manager.get_danger_map()
        global_risk = float(danger_map.max())

        # 6. 采样每个人的风险等级
        risks = self._sample_risks(danger_map, person_data)

        # 7. 报警防抖
        alarm_triggered = self.alarm_manager.update(global_risk)

        # 8. 视频画面叠加半透明占用网格（替代原来的椭圆绘制）
        draw_occupancy_overlay(annotated_frame, self.grid_manager, self.config)

        # 9. 网格显示（BEV 像素分辨率）+ 坐标系叠加
        display_grid = self.grid_manager.get_display_grid()
        draw_grid_coordinates(display_grid, self.config)
        self.alarm_manager.draw_grid_overlay(
            display_grid, alarm_triggered, global_risk,
        )

        return {
            'annotated_frame': annotated_frame,
            'bev_img': bev_img,
            'grid_map': display_grid,
        }

    def _sample_risks(self, danger_map, person_data):
        """从 danger_map 采样每个人员的风险值"""
        risks = []
        rows, cols = danger_map.shape
        cfg = self.config

        for pd_ in person_data:
            cam_x, cam_y = pd_['foot_cam']
            col = cfg.cam_to_grid_col(cam_x)
            row = cfg.cam_to_grid_row(cam_y)
            col = max(0, min(cols - 1, col))
            row = max(0, min(rows - 1, row))
            risks.append(float(danger_map[row, col]))
        return risks
