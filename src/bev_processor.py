# bev_processor.py
import cv2
import numpy as np
from utils import compute_bev, draw_bev_grid_overlay, world_to_pixel


class BEVProcessor:
    """BEV 图像处理器 — warp 透视 + 物理网格叠加"""

    def __init__(self, config):
        self.config = config
        self.bev_w = config.BEV_W
        self.bev_h = config.BEV_H
        self.M = config.M
        self.grid_cell_m = config.GRID_CELL_SIZE_M
        self.grid_x_range = config.GRID_X_RANGE_M
        self.grid_y_near = config.GRID_Y_NEAR_M
        self.grid_y_far = config.GRID_Y_FAR_M
        self.ppm = config.BEV_PIXELS_PER_METER
        self._grid_lines = self._build_physical_grid_lines()

    def process(self, frame):
        """生成 BEV 图像：warp 透视 + 物理网格叠加"""
        bev_img = compute_bev(frame, self.M, (self.bev_w, self.bev_h))

        for (x1, y1), (x2, y2) in self._grid_lines:
            cv2.line(bev_img, (int(x1), int(y1)), (int(x2), int(y2)),
                     (80, 80, 80), 1, cv2.LINE_AA)

        gy_range = self.grid_y_far - self.grid_y_near
        cv2.putText(bev_img,
                    f"BEV: {self.bev_w}x{self.bev_h}px  |  "
                    f"{self.grid_x_range:.0f}x{gy_range:.0f}m  |  "
                    f"cell={self.grid_cell_m:.1f}m",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return bev_img

    def _build_physical_grid_lines(self):
        """预计算 BEV 空间中的物理网格线（像素坐标）"""
        lines = []
        bw, bh = self.bev_w, self.bev_h
        half_x = self.grid_x_range / 2.0
        gy_range = self.grid_y_far - self.grid_y_near

        # 纵向线：固定 X
        for cx in np.arange(-half_x, half_x + 0.01, self.grid_cell_m):
            bev_x = (cx + half_x) * self.ppm
            if 0 <= bev_x <= bw:
                lines.append(((bev_x, 0), (bev_x, bh)))

        # 横向线：固定 cam_y（相机前方距离）
        for cy in np.arange(self.grid_y_near, self.grid_y_far + 0.01,
                            self.grid_cell_m):
            # cy = grid_y_near → bev bottom; cy = grid_y_far → bev top
            ratio = (cy - self.grid_y_near) / gy_range
            bev_y = bh - 1 - ratio * (bh - 1)
            if 0 <= bev_y <= bh:
                lines.append(((0, bev_y), (bw, bev_y)))

        return lines
