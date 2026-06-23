# config.py
import numpy as np
import cv2

class Config:
    """配置管理类"""
    def __init__(self):
        self.RTSP_URL = "rtsp://118.140.130.26:8554/dahua1003362"
        self.MODEL_PATH = "../model/yolo26s_mocs_20260623_0216.pt"
        self.BEV_SCALE = 0.5
        self.GRID_SIZE = 20
        self.SRC_PTS = np.array([[200, 100], [1700, 100], [1850, 1050], [70, 1050]], dtype=np.float32)
        self.CONF_THRESH = 0.35
        self.IMGSZ = 640
        self.PERSON_CLASS = 1
        self.ALARM_CONTINUOUS_FRAMES = 3    # 报警防抖帧数（hit counter已提供一级抗闪烁，此处可放宽）
        self.ALARM_RISK_THRESHOLD = 0.15       # 风险值阈值 [0,1]（降低以提高灵敏度）
        self.EQUIPMENT_DANGER_RADIUS_CELLS = 5  # 设备危险半径（网格单元数）

        # Hit Counter 占用网格参数（置信度加权，抗闪烁）
        self.PERSON_MAX_HITS = 3               # 人员：~3帧高置信度检测确认（conf加权）
        self.EQUIPMENT_MAX_HITS = 10           # 设备：~10帧确认（车辆稳定，可更严）
        self.DEBUG_MODE = False                 # 是否显示BEV窗口
        self.STREAM_WIDTH = None                # 流分辨率（None=自动探测）
        self.STREAM_HEIGHT = None

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