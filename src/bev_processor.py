# bev_processor.py
import cv2
import numpy as np
from utils import compute_bev, draw_bev_grid_overlay

class BEVProcessor:
    """BEV图像处理器"""
    
    def __init__(self, config):
        self.config = config
        self.bev_w = config.BEV_W
        self.bev_h = config.BEV_H
        self.M = config.M
    
    def process(self, frame):
        """生成BEV图像并添加网格"""
        bev_img = compute_bev(frame, self.M, (self.bev_w, self.bev_h))
        bev_with_grid = draw_bev_grid_overlay(bev_img.copy(), self.config.GRID_SIZE * 2)
        
        # 添加BEV尺寸信息
        cv2.putText(bev_with_grid, f"BEV: {self.bev_w}x{self.bev_h}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return bev_with_grid
    
