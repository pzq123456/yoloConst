# frame_processor.py
import numpy as np
from utils import draw_src_region, update_grid

class FrameProcessor:
    """帧处理器 - 处理每一帧的完整逻辑"""
    
    def __init__(self, config, model, bev_processor, detection_processor, alarm_manager):
        self.config = config
        self.model = model
        self.bev_processor = bev_processor
        self.detection_processor = detection_processor
        self.alarm_manager = alarm_manager
        
        # 初始化网格地图
        self.grid_map = np.zeros((config.BEV_H, config.BEV_W, 3), dtype=np.uint8)
        
        # 椭圆绘制参数
        self.ellipse_config = {
            'color': (0, 255, 0),
            'alpha': 0.25,
            'radius': 50,
        }
    
    def process_frame(self, frame):
        """
        处理单帧图像
        """
        # 1. 模型推理
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.config.CONF_THRESH,
            verbose=False, 
            imgsz=self.config.IMGSZ,
            iou=0.5,
        )
        
        # 2. 绘制原始图像
        annotated_frame = results[0].plot()
        annotated_frame = draw_src_region(annotated_frame, self.config.SRC_PTS)
        
        # 3. 生成BEV图像
        bev_img = self.bev_processor.process(frame)
        
        # 4. 提取BEV坐标（现在车辆返回的是区域而不是线段）
        person_points, vehicle_areas, detections, person_bboxes = \
            self.detection_processor.extract_bev_coordinates(results)
        
        # 5. 绘制人物脚底椭圆
        self._draw_person_ellipses(annotated_frame, person_bboxes)
        
        # 可选：绘制车辆占用区域（调试用）
        # 注意：需要从results中获取车辆bbox
        # self._draw_vehicle_areas(annotated_frame, results)
        
        # 6. 更新网格（车辆会占用多个网格）
        update_grid(self.grid_map, detections, self.config.GRID_SIZE)
        
        # 7. 报警检测（现在使用车辆区域而不是线段）
        danger_detected = self.alarm_manager.check_danger(person_points, vehicle_areas)
        alarm_triggered = self.alarm_manager.update(danger_detected)
        self.alarm_manager.draw_warning(self.grid_map, bev_img)
        
        # 返回所有结果
        return {
            'annotated_frame': annotated_frame,
            'bev_img': bev_img,
            'grid_map': self.grid_map,
            'person_points': person_points,
            'vehicle_areas': vehicle_areas,
            'danger_detected': danger_detected,
            'alarm_triggered': alarm_triggered
        }
    
    def _draw_person_ellipses(self, frame, person_bboxes):
        """在原始图像上绘制人物脚底椭圆"""
        self.detection_processor.draw_person_ellipses(
            frame,
            person_bboxes,
            color=self.ellipse_config['color'],
            alpha=self.ellipse_config['alpha'],
            radius=self.ellipse_config['radius']
        )
    
    def reset_grid(self):
        """重置网格地图"""
        self.grid_map.fill(0)