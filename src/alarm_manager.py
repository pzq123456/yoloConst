# alarm_manager.py
import cv2
import numpy as np

class AlarmManager:
    """报警管理器"""
    
    def __init__(self, config):
        self.config = config
        self.alarm_counter = 0
        self.alarm_triggered = False
        self.alarm_pixel_thresh = config.ALARM_GRID_THRESH * config.GRID_SIZE
    
    def check_danger(self, person_points, vehicle_areas):
        """
        检测是否有危险情况
        
        参数:
            person_points: 人物点列表 [(x,y), ...]
            vehicle_areas: 车辆区域列表 [(x1,y1,x2,y2), ...]
        """
        for p in person_points:
            for area in vehicle_areas:
                x1, y1, x2, y2 = area
                # 计算点到矩形区域的距离
                dist = self._point_to_rectangle_distance(p, (x1, y1, x2, y2))
                if dist < self.alarm_pixel_thresh:
                    return True
        return False
    
    def _point_to_rectangle_distance(self, point, rect):
        """
        计算点到矩形的最短距离
        
        参数:
            point: (x, y)
            rect: (x1, y1, x2, y2)
        """
        px, py = point
        x1, y1, x2, y2 = rect
        
        # 如果点在矩形内部，距离为0
        if x1 <= px <= x2 and y1 <= py <= y2:
            return 0
        
        # 计算到四条边的距离
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        
        return np.hypot(dx, dy)
    
    def update(self, danger_detected):
        """更新报警状态（带防抖）"""
        if danger_detected:
            self.alarm_counter += 1
        else:
            self.alarm_counter = 0
        
        if self.alarm_counter >= self.config.ALARM_CONTINUOUS_FRAMES:
            self.alarm_triggered = True
        else:
            self.alarm_triggered = False
        
        return self.alarm_triggered
    
    def draw_warning(self, grid_map, bev_img):
        """在图像上绘制警告信息"""
        if self.alarm_triggered:
            cv2.putText(grid_map, "!!! WARNING: Person too close to Vehicle !!!",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(bev_img, "!!! WARNING !!!",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)