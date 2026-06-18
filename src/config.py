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
        self.ALARM_GRID_THRESH = 5
        self.ALARM_CONTINUOUS_FRAMES = 5
        
        # 运行时计算的属性
        self.DST_PTS = None
        self.M = None
        self.M_inv = None
        self.BEV_W = None
        self.BEV_H = None
    
    def initialize_bev(self, bev_w, bev_h):
        """初始化BEV相关参数"""
        self.BEV_W = bev_w
        self.BEV_H = bev_h
        self.DST_PTS = np.array([
            [0, 0],
            [bev_w, 0],
            [bev_w, bev_h],
            [0, bev_h]
        ], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.M_inv = cv2.getPerspectiveTransform(self.DST_PTS, self.SRC_PTS)