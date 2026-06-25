# grid_manager.py
import cv2
import numpy as np


class GridManager:
    """双层占用网格管理器 — Hit Counter 抗闪烁 + YOLO置信度加权"""

    def __init__(self, config):
        cols, rows = config.GRID_COLS, config.GRID_ROWS

        # ── 浮点计数器网格（置信度加权，相机对齐坐标） ──
        self.person_hits = np.zeros((rows, cols), dtype=np.float32)
        self.equip_hits = np.zeros((rows, cols), dtype=np.float32)

        # ── 参数引用 ──
        self.config = config
        self.cell_size_m = config.GRID_CELL_SIZE_M
        self.cols = cols
        self.rows = rows

        # ── 确认阈值 ──
        self.person_max = config.PERSON_MAX_HITS
        self.equip_max = config.EQUIPMENT_MAX_HITS

        # ── 危险半径 ──
        self.danger_radius_cells = config.DANGER_RADIUS_CELLS

        # ── 显示画布（BEV 像素尺寸）──
        self._display = np.zeros(
            (config.BEV_H, config.BEV_W, 3), dtype=np.uint8,
        )

        # ── 预分配权重数组（每帧复用） ──
        self._person_weight = np.zeros((rows, cols), dtype=np.float32)
        self._equip_weight = np.zeros((rows, cols), dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════
    #  更新
    # ═══════════════════════════════════════════════════════════════

    def update(self, detections):
        """
        逐帧更新占用网格。

        detections: [(points_list, color), ...]
            points_list: [(cx, cy, conf), ...]
                cx, cy: 相机对齐坐标（米）
            color: (B, G, R)

        每帧逻辑:
          1. 统计每个格子本帧检测到的最大置信度
          2. 没看到 + hits>0 → -1（冷却）
          3. 看到了 → hits += max_conf（不超过 MAX）
        """
        rows, cols = self.rows, self.cols
        cfg = self.config

        # 1. 清空本帧权重
        self._person_weight.fill(0.0)
        self._equip_weight.fill(0.0)

        # 2. 遍历检测，相机坐标 → 网格行列
        for points, color in detections:
            is_equip = color[2] > color[1]
            weight = self._equip_weight if is_equip else self._person_weight

            for pt in points:
                if len(pt) == 2:
                    cx, cy = pt
                    conf = 1.0
                else:
                    cx, cy, conf = float(pt[0]), float(pt[1]), float(pt[2])

                col = cfg.cam_to_grid_col(cx)
                row = cfg.cam_to_grid_row(cy)
                if 0 <= col < cols and 0 <= row < rows:
                    if conf > weight[row, col]:
                        weight[row, col] = conf

        seen_person = self._person_weight > 0
        seen_equip = self._equip_weight > 0

        # 3. 看不到 → 衰减（冷却速率可配置）
        cool_person = (self.person_hits > 0) & ~seen_person
        self.person_hits[cool_person] -= cfg.PERSON_DECAY

        cool_equip = (self.equip_hits > 0) & ~seen_equip
        self.equip_hits[cool_equip] -= cfg.EQUIPMENT_DECAY

        # 4. 看到 → 按置信度累加（不超过 MAX）
        self.person_hits += self._person_weight
        np.clip(self.person_hits, 0.0, float(self.person_max),
                out=self.person_hits)

        self.equip_hits += self._equip_weight
        np.clip(self.equip_hits, 0.0, float(self.equip_max),
                out=self.equip_hits)

    # ═══════════════════════════════════════════════════════════════
    #  风险图（网格分辨率）
    # ═══════════════════════════════════════════════════════════════

    def get_danger_map(self):
        """返回逐单元格风险图 [0, 1] — person_prob × dilate(equipment_prob)"""
        person_prob = self._hits_to_prob(self.person_hits, self.person_max)
        equip_prob = self._hits_to_prob(self.equip_hits, self.equip_max)

        if person_prob.max() < 0.01 or equip_prob.max() < 0.01:
            return np.zeros_like(person_prob)

        k = self.danger_radius_cells * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(equip_prob, kernel)
        return person_prob * dilated

    # ═══════════════════════════════════════════════════════════════
    #  可视化（BEV 像素分辨率）
    # ═══════════════════════════════════════════════════════════════

    def get_display_grid(self):
        """将占用网格上采样到 BEV 可视化分辨率"""
        person_prob = self._hits_to_prob(self.person_hits, self.person_max)
        equip_prob = self._hits_to_prob(self.equip_hits, self.equip_max)

        # 上采样到 BEV 显示分辨率
        display_w, display_h = self._display.shape[1], self._display.shape[0]
        person_full = cv2.resize(person_prob, (display_w, display_h),
                                 interpolation=cv2.INTER_LINEAR)
        equip_full = cv2.resize(equip_prob, (display_w, display_h),
                                interpolation=cv2.INTER_LINEAR)

        self._display.fill(0)
        self._display[:, :, 1] = (person_full * 255).astype(np.uint8)
        self._display[:, :, 2] = (equip_full * 255).astype(np.uint8)

        danger = self.get_danger_map()
        if danger.max() > 0.01:
            danger_full = cv2.resize(danger, (display_w, display_h),
                                     interpolation=cv2.INTER_LINEAR)
            mask = danger_full > 0.01
            self._display[mask, 0] = (danger_full[mask] * 200).astype(np.uint8)
            self._display[mask, 1] = np.maximum(
                self._display[mask, 1],
                (danger_full[mask] * 200).astype(np.uint8),
            )

        # 方向标注：上 = 远处（{:.0f}m），下 = 近处
        dh, dw = self._display.shape[:2]
        cv2.putText(self._display,
                    f"FAR ({self.config.GRID_Y_FAR_M:.0f}m)",
                    (dw // 2 - 30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(self._display,
                    "NEAR (Camera)", (dw // 2 - 40, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return self._display

    # ═══════════════════════════════════════════════════════════════
    #  工具
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _hits_to_prob(hits, max_hits):
        """计数 → [0, 1] 概率"""
        return hits.astype(np.float32) / float(max_hits)
