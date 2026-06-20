# detection_processor.py
import cv2
import numpy as np
from utils import draw_foot_ellipse

class DetectionProcessor:
    """检测结果处理器 — 坐标提取 + 风险感知标注"""

    def __init__(self, config):
        self.config = config
        self.M = config.M
        self.ellipse_config = {
            'alpha': 0.25,
            'color': (0, 255, 0),
            'radius': 50,
        }

    def extract_bev_coordinates(self, results):
        """
        提取BEV坐标（不做绘制，绘制在annotate_frame中根据风险等级完成）

        返回:
            person_points:  [(x,y), ...]          BEV人物位置
            vehicle_areas: [(x1,y1,x2,y2), ...]    BEV车辆矩形区域
            detections:     [(points_list, color)]  网格更新数据
            person_data:    [{bbox, foot_img, foot_bev}, ...]  用于后续风险标注
        """
        person_points = []
        vehicle_areas = []
        detections = []
        person_data = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = box
                foot_mid = [(x1 + x2) / 2, y2]

                if cls == self.config.PERSON_CLASS:
                    color = (0, 255, 0)
                    pts_img = np.array([foot_mid], dtype=np.float32).reshape(-1, 1, 2)
                    pts_bev = cv2.perspectiveTransform(pts_img, self.M).reshape(-1, 2)

                    detections.append((pts_bev.tolist(), color))
                    person_points.append(tuple(pts_bev[0]))
                    person_data.append({
                        'bbox': box,
                        'foot_img': foot_mid,
                        'foot_bev': tuple(pts_bev[0]),
                    })

                else:
                    color = (0, 0, 255)
                    vehicle_bbox = self._estimate_vehicle_bbox(box)
                    pts_img = np.array(vehicle_bbox, dtype=np.float32).reshape(-1, 1, 2)
                    pts_bev = cv2.perspectiveTransform(pts_img, self.M).reshape(-1, 2)

                    min_x, max_x = min(pts_bev[:, 0]), max(pts_bev[:, 0])
                    min_y, max_y = min(pts_bev[:, 1]), max(pts_bev[:, 1])
                    vehicle_areas.append((min_x, min_y, max_x, max_y))

                    for cell_x, cell_y in self._get_grid_cells_in_bbox(
                        min_x, min_y, max_x, max_y, self.config.GRID_SIZE
                    ):
                        detections.append(([(cell_x, cell_y)], color))

        return person_points, vehicle_areas, detections, person_data

    def annotate_frame(self, frame, person_data, risks, alarm_triggered):
        """
        根据风险等级绘制椭圆

        risk < 0.01  → 绿色（安全）
        risk 0.01-0.3 → 橙色（警告）
        risk >= 0.3   → 红色（危险）
        """
        for pd_, risk in zip(person_data, risks):
            if alarm_triggered and risk >= 0.1:
                color = (0, 0, 255)       # 红色
            elif risk >= 0.15:
                color = (0, 140, 255)     # 橙色
            elif risk >= 0.01:
                color = (0, 255, 255)     # 黄色
            else:
                color = (0, 255, 0)       # 绿色

            draw_foot_ellipse(
                frame, pd_['foot_img'], bbox=pd_['bbox'],
                color=color,
                alpha=self.ellipse_config['alpha'],
                radius=self.ellipse_config['radius'],
            )

    def _estimate_vehicle_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        width_scale = 0.8
        if height > 0:
            height_scale = min(1.5, max(0.5, height / 100))
            actual_width = width * width_scale * height_scale
        else:
            actual_width = width * width_scale
        depth_offset = min(actual_width * 0.8, height * 0.3)
        return [
            [x1, y2],
            [x2, y2],
            [x2 + actual_width * 0.1, y2 - depth_offset],
            [x1 - actual_width * 0.1, y2 - depth_offset],
        ]

    def _get_grid_cells_in_bbox(self, min_x, min_y, max_x, max_y, grid_size):
        cells = []
        start_gx = int(min_x // grid_size)
        end_gx = int(max_x // grid_size)
        start_gy = int(min_y // grid_size)
        end_gy = int(max_y // grid_size)
        for gx in range(start_gx, end_gx + 1):
            for gy in range(start_gy, end_gy + 1):
                center_x = gx * grid_size + grid_size // 2
                center_y = gy * grid_size + grid_size // 2
                cells.append((center_x, center_y))
        return cells
