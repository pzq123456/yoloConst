# detection_processor.py
import cv2
import numpy as np
from utils import draw_foot_ellipse

class DetectionProcessor:
    """检测结果处理器"""
    
    def __init__(self, config):
        self.config = config
        self.M = config.M
        self.ellipse_config = {
            'alpha': 0.3,
            'color': (0, 255, 0),
            'radius': 80,
        }
        # 车辆占用网格数量（根据车辆大小估算）
        self.vehicle_grid_size = 3  # 默认3x3网格
    
    def extract_bev_coordinates(self, results):
        """
        提取BEV坐标：人物点和车辆区域
        
        返回:
            person_points: 人物点列表 [(x,y), ...]
            vehicle_areas: 车辆区域列表 [(x1,y1,x2,y2), ...] 在BEV坐标系中
            detections: 用于更新网格的检测数据
            person_bboxes: 人物边界框列表
        """
        person_points = []
        vehicle_areas = []  # 存储车辆的矩形区域 (x1,y1,x2,y2)
        detections = []
        person_bboxes = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = box
                foot_mid = [(x1 + x2) / 2, y2]
                color = (0, 255, 0) if cls == self.config.PERSON_CLASS else (0, 0, 255)
                
                if cls == self.config.PERSON_CLASS:
                    # 人物：一个点
                    points = [foot_mid]
                    person_bboxes.append((box, foot_mid))
                    
                    # 投影到BEV
                    pts_img = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                    pts_bev = cv2.perspectiveTransform(pts_img, self.M).reshape(-1, 2)
                    
                    # 人物占用1个网格
                    detections.append((pts_bev.tolist(), color))
                    person_points.append(tuple(pts_bev[0]))
                    
                else:
                    # 车辆：估算底部矩形区域
                    vehicle_bbox = self._estimate_vehicle_bbox(box)
                    points = vehicle_bbox  # 四个角点
                    
                    # 投影到BEV
                    pts_img = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                    pts_bev = cv2.perspectiveTransform(pts_img, self.M).reshape(-1, 2)
                    
                    # 计算BEV中的矩形边界
                    min_x = min(pts_bev[:, 0])
                    max_x = max(pts_bev[:, 0])
                    min_y = min(pts_bev[:, 1])
                    max_y = max(pts_bev[:, 1])
                    
                    # 保存车辆区域（用于报警检测）
                    vehicle_areas.append((min_x, min_y, max_x, max_y))
                    
                    # 车辆占用多个网格 - 使用矩形区域
                    vehicle_grid_cells = self._get_grid_cells_in_bbox(
                        min_x, min_y, max_x, max_y, self.config.GRID_SIZE
                    )
                    
                    # 为每个网格单元添加检测
                    for cell_x, cell_y in vehicle_grid_cells:
                        detections.append(([(cell_x, cell_y)], color))
        
        return person_points, vehicle_areas, detections, person_bboxes
    
    def _estimate_vehicle_bbox(self, bbox):
        """
        根据车辆的边界框估算底部矩形区域
        
        参数:
            bbox: [x1, y1, x2, y2] 车辆边界框
            
        返回:
            list: 底部矩形的四个角点 [[x1,y2], [x2,y2], [x2+offset,y2], [x1-offset,y2]]
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 估算车辆的实际宽度（底部）
        # 根据车辆高度估算宽度比例（近大远小）
        width_scale = 0.8  # 基础宽度比例
        
        # 根据高度调整：近处的车（高度大）宽度也大
        if height > 0:
            height_scale = min(1.5, max(0.5, height / 100))
            actual_width = width * width_scale * height_scale
        else:
            actual_width = width * width_scale
        
        # 底部矩形的四个角点（在图像坐标系中）
        # 脚底线的两个端点
        foot_left = [x1, y2]
        foot_right = [x2, y2]
        
        # 根据车辆宽度估算前后深度（在图像中表现为y方向偏移）
        depth_offset = min(actual_width * 0.8, height * 0.3)
        
        # 底部矩形（实际是车辆在地面的投影）
        # 注意：在图像中，车辆的前部（靠近摄像头）在下方
        # 后部（远离摄像头）在上方
        bottom_pts = [
            [x1, y2],                    # 左下（靠近摄像头）
            [x2, y2],                    # 右下（靠近摄像头）
            [x2 + actual_width*0.1, y2 - depth_offset],  # 右上（远离摄像头）
            [x1 - actual_width*0.1, y2 - depth_offset]   # 左上（远离摄像头）
        ]
        
        return bottom_pts
    
    def _get_grid_cells_in_bbox(self, min_x, min_y, max_x, max_y, grid_size):
        """
        获取矩形区域内所有网格单元的中心点
        
        参数:
            min_x, min_y, max_x, max_y: 矩形边界（BEV坐标）
            grid_size: 网格大小
            
        返回:
            list: 网格单元中心点列表 [(x, y), ...]
        """
        cells = []
        
        # 计算覆盖的网格范围
        start_gx = int(min_x // grid_size)
        end_gx = int(max_x // grid_size)
        start_gy = int(min_y // grid_size)
        end_gy = int(max_y // grid_size)
        
        # 遍历所有网格
        for gx in range(start_gx, end_gx + 1):
            for gy in range(start_gy, end_gy + 1):
                # 计算网格中心点
                center_x = gx * grid_size + grid_size // 2
                center_y = gy * grid_size + grid_size // 2
                cells.append((center_x, center_y))
        
        return cells
    
    def draw_person_ellipses(self, frame, person_bboxes, 
                            color=(0, 255, 0), alpha=0.3, radius=None):
        """
        在原始图像上为每个人绘制脚底椭圆
        """
        for bbox, foot_point in person_bboxes:
            x1, y1, x2, y2 = bbox
            person_height = y2 - y1
            
            color = color or self.ellipse_config['color']
            alpha = alpha or self.ellipse_config['alpha']
            radius = radius or self.ellipse_config['radius']
            
            draw_foot_ellipse(
                frame, 
                foot_point, 
                bbox=bbox,
                color=color,
                alpha=alpha,
                radius=radius,
                person_height=person_height
            )
    
    def draw_vehicle_areas(self, frame, vehicle_bboxes, color=(0, 0, 255), alpha=0.2):
        """
        在原始图像上绘制车辆占用区域（调试用）
        
        参数:
            frame: 要绘制的图像
            vehicle_bboxes: 车辆边界框列表
            color: 颜色
            alpha: 透明度
        """
        for bbox in vehicle_bboxes:
            x1, y1, x2, y2 = bbox
            
            # 估算车辆底部区域
            vehicle_pts = self._estimate_vehicle_bbox(bbox)
            
            # 绘制多边形
            pts = np.array(vehicle_pts, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            
            # 填充半透明区域
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)