# frame_processor.py
import numpy as np
from grid_manager import GridManager

class FrameProcessor:
    """帧处理器 — 工地人-设备距离报警管线"""

    def __init__(self, config, model, bev_processor, detection_processor, alarm_manager):
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

        # 2. 绘制原始图像
        annotated_frame = results[0].plot()

        # 3. BEV（debug用）
        bev_img = self.bev_processor.process(frame)

        # 4. 提取BEV坐标（不绘制）
        person_points, vehicle_areas, detections, person_data = \
            self.detection_processor.extract_bev_coordinates(results)

        # 5. 更新网格 + 计算风险
        self.grid_manager.update(detections)
        danger_map = self.grid_manager.get_danger_map()
        global_risk = float(danger_map.max())

        # 6. 采样每个人的风险等级
        risks = self._sample_risks(danger_map, person_data)

        # 7. 报警防抖
        alarm_triggered = self.alarm_manager.update(global_risk)

        # 8. 标注：风险色椭圆
        self.detection_processor.annotate_frame(annotated_frame, person_data, risks, alarm_triggered)

        # 9. 网格显示
        display_grid = self.grid_manager.get_display_grid()
        self.alarm_manager.draw_grid_overlay(display_grid, alarm_triggered, global_risk)

        return {
            'annotated_frame': annotated_frame,
            'bev_img': bev_img,
            'grid_map': display_grid,
        }

    def _sample_risks(self, danger_map, person_data):
        risks = []
        h, w = danger_map.shape
        for pd_ in person_data:
            px, py = pd_['foot_bev']
            ix = int(np.clip(px, 0, w - 1))
            iy = int(np.clip(py, 0, h - 1))
            risks.append(float(danger_map[iy, ix]))
        return risks
