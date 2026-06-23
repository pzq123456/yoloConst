# grid_manager.py
import cv2
import numpy as np


class GridManager:
    """双层占用网格管理器 — Hit Counter 抗闪烁 + YOLO置信度加权"""

    def __init__(self, config):
        h, w = config.BEV_H, config.BEV_W

        # ── 浮点计数器网格（置信度加权） ──
        self.person_hits = np.zeros((h, w), dtype=np.float32)
        self.equip_hits = np.zeros((h, w), dtype=np.float32)

        # ── 确认阈值 ──
        self.person_max = config.PERSON_MAX_HITS
        self.equip_max = config.EQUIPMENT_MAX_HITS

        # ── 其余 ──
        self.grid_size = config.GRID_SIZE
        self.danger_radius_cells = config.EQUIPMENT_DANGER_RADIUS_CELLS
        self._display = np.zeros((h, w, 3), dtype=np.uint8)

        # ── 预分配权重数组（每帧复用） ──
        self._person_weight = np.zeros((h, w), dtype=np.float32)
        self._equip_weight = np.zeros((h, w), dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════
    #  更新
    # ═══════════════════════════════════════════════════════════════

    def update(self, detections):
        """
        逐帧更新占用网格。

        detections: [(points_list, color), ...]
            points_list: [(x, y, conf), ...]  带置信度的BEV坐标
            color: (B, G, R)

        每帧逻辑:
          1. 统计每个格子本帧检测到的最大置信度
          2. 没看到 + hits>0 → -1（冷却）
          3. 看到了 → hits += max_conf（不超过 MAX）
        """
        h, w = self.person_hits.shape
        gs = self.grid_size

        # 1. 清空本帧权重
        self._person_weight.fill(0.0)
        self._equip_weight.fill(0.0)

        # 2. 遍历检测，取每个格子的最大置信度
        for points, color in detections:
            is_equip = color[2] > color[1]
            weight = self._equip_weight if is_equip else self._person_weight

            for pt in points:
                if len(pt) == 2:
                    px, py = pt
                    conf = 1.0
                else:
                    px, py, conf = float(pt[0]), float(pt[1]), float(pt[2])

                gx = (int(np.clip(px, 0, w - 1)) // gs) * gs
                gy = (int(np.clip(py, 0, h - 1)) // gs) * gs
                gx = min(gx, w - gs)
                gy = min(gy, h - gs)

                block = weight[gy:gy + gs, gx:gx + gs]
                block[:] = np.maximum(block, conf)

        seen_person = self._person_weight > 0
        seen_equip = self._equip_weight > 0

        # 3. 看不到 → -1
        cool_person = (self.person_hits > 0) & ~seen_person
        self.person_hits[cool_person] -= 1.0

        cool_equip = (self.equip_hits > 0) & ~seen_equip
        self.equip_hits[cool_equip] -= 1.0

        # 4. 看到 → 按置信度累加（不超过 MAX）
        self.person_hits += self._person_weight
        np.clip(self.person_hits, 0.0, float(self.person_max),
                out=self.person_hits)

        self.equip_hits += self._equip_weight
        np.clip(self.equip_hits, 0.0, float(self.equip_max),
                out=self.equip_hits)

    # ═══════════════════════════════════════════════════════════════
    #  风险图
    # ═══════════════════════════════════════════════════════════════

    def get_danger_map(self):
        """返回逐像素风险图 [0, 1] — person_prob × dilate(equipment_prob)"""
        person_prob = self._hits_to_prob(self.person_hits, self.person_max)
        equip_prob = self._hits_to_prob(self.equip_hits, self.equip_max)

        if person_prob.max() < 0.01 or equip_prob.max() < 0.01:
            return np.zeros_like(person_prob)

        k = self.danger_radius_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(equip_prob, kernel)
        return person_prob * dilated

    # ═══════════════════════════════════════════════════════════════
    #  可视化
    # ═══════════════════════════════════════════════════════════════

    def get_display_grid(self):
        """合并为可视化图像：绿=人，红=设备，黄=危险重叠"""
        person_prob = self._hits_to_prob(self.person_hits, self.person_max)
        equip_prob = self._hits_to_prob(self.equip_hits, self.equip_max)

        self._display.fill(0)
        self._display[:, :, 1] = (person_prob * 255).astype(np.uint8)
        self._display[:, :, 2] = (equip_prob * 255).astype(np.uint8)

        danger = self.get_danger_map()
        if danger.max() > 0.01:
            mask = danger > 0.01
            self._display[mask, 0] = (danger[mask] * 200).astype(np.uint8)
            self._display[mask, 1] = np.maximum(
                self._display[mask, 1],
                (danger[mask] * 200).astype(np.uint8),
            )

        return self._display

    # ═══════════════════════════════════════════════════════════════
    #  工具
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _hits_to_prob(hits, max_hits):
        """计数 → [0, 1] 概率"""
        return hits.astype(np.float32) / float(max_hits)
