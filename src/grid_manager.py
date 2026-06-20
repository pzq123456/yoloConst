# grid_manager.py
import cv2
import numpy as np

class GridManager:
    """双层占用网格管理器 — 工地人-设备距离报警的计算核心"""

    def __init__(self, config):
        h, w = config.BEV_H, config.BEV_W
        self.person_grid = np.zeros((h, w), dtype=np.float32)
        self.equipment_grid = np.zeros((h, w), dtype=np.float32)
        self.person_decay = config.PERSON_DECAY
        self.equipment_decay = config.EQUIPMENT_DECAY
        self.danger_radius_cells = config.EQUIPMENT_DANGER_RADIUS_CELLS
        self.grid_size = config.GRID_SIZE
        self._display = np.zeros((h, w, 3), dtype=np.uint8)

    def update(self, detections):
        """衰减 + 写入检测点"""
        self.person_grid *= self.person_decay
        self.equipment_grid *= self.equipment_decay

        h, w = self.person_grid.shape
        gs = self.grid_size

        for points, color in detections:
            is_equipment = color[2] > color[1]
            target = self.equipment_grid if is_equipment else self.person_grid

            for pt in points:
                if len(pt) == 2:
                    px = int(np.clip(pt[0], 0, w - 1))
                    py = int(np.clip(pt[1], 0, h - 1))
                    gx = (px // gs) * gs
                    gy = (py // gs) * gs
                    gx = min(gx, w - gs)
                    gy = min(gy, h - gs)
                    target[gy:gy + gs, gx:gx + gs] = 1.0

    def get_danger_map(self):
        """返回逐像素风险图 [0, 1] — person_grid × dilate(equipment_grid)"""
        if self.person_grid.max() < 0.01 or self.equipment_grid.max() < 0.01:
            return np.zeros_like(self.person_grid)

        k = self.danger_radius_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(self.equipment_grid, kernel)
        return self.person_grid * dilated

    def get_danger_level(self):
        """全局最大风险值 → [0, 1]"""
        return float(self.get_danger_map().max())

    def get_display_grid(self):
        """合并为可视化图像：绿=人，红=设备，黄=危险重叠"""
        self._display.fill(0)
        # 基础层：人=绿，设备=红
        self._display[:, :, 1] = (self.person_grid * 255).astype(np.uint8)
        self._display[:, :, 2] = (self.equipment_grid * 255).astype(np.uint8)

        # 危险区域：半透明黄色叠加
        danger = self.get_danger_map()
        if danger.max() > 0.01:
            mask = danger > 0.01
            self._display[mask, 0] = (danger[mask] * 200).astype(np.uint8)  # 蓝 → 黄
            self._display[mask, 1] = np.maximum(
                self._display[mask, 1],
                (danger[mask] * 200).astype(np.uint8)
            )

        return self._display
