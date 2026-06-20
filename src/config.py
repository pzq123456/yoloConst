# config.py
import numpy as np
import cv2

class Config:
    """配置管理类"""
    def __init__(self):
        self.RTSP_URL = "rtsp://118.140.130.26:8554/dahua1003362"
        self.MODEL_PATH = "../model/yolo26s_mocs.pt"
        self.BEV_SCALE = 0.5
        self.GRID_SIZE = 20
        self.SRC_PTS = np.array([[200, 100], [1700, 100], [1850, 1050], [70, 1050]], dtype=np.float32)
        self.CONF_THRESH = 0.35
        self.IMGSZ = 640
        self.PERSON_CLASS = 1
        self.ALARM_CONTINUOUS_FRAMES = 5
        self.ALARM_RISK_THRESHOLD = 0.3        # 风险值阈值 [0,1]
        self.EQUIPMENT_DECAY = 0.95            # 设备网格衰减（持久化）
        self.PERSON_DECAY = 0.85                # 人员网格衰减（瞬态）
        self.EQUIPMENT_DANGER_RADIUS_CELLS = 5  # 设备危险半径（网格单元数）
        self.DEBUG_MODE = False                 # 是否显示BEV窗口
        
        # 运行时计算的属性
        self.M = None
        self.BEV_W = None
        self.BEV_H = None
    
    def initialize_bev(self, bev_w, bev_h):
        """初始化BEV相关参数"""
        self.BEV_W = bev_w
        self.BEV_H = bev_h
        dst_pts = np.array([
            [0, 0],
            [bev_w, 0],
            [bev_w, bev_h],
            [0, bev_h]
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, dst_pts)