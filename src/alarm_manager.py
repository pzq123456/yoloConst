# alarm_manager.py
import cv2
import numpy as np

# 风险等级配色：BGR
RISK_COLORS = {
    'safe':    (0, 255, 0),    # 绿
    'caution': (0, 255, 255),  # 黄
    'warning': (0, 140, 255),  # 橙
    'danger':  (0, 0, 255),    # 红
}
BORDER_WIDTH = 12


class AlarmManager:
    """报警管理器 — 防抖 + 分层可视化警告"""

    def __init__(self, config):
        self.config = config
        self.alarm_counter = 0
        self.alarm_triggered = False
        self.risk_threshold = config.ALARM_RISK_THRESHOLD

    # ── 状态更新 ──────────────────────────────────────────

    def update(self, danger_level):
        if danger_level >= self.risk_threshold:
            self.alarm_counter += 1
        else:
            self.alarm_counter = max(0, self.alarm_counter - 1)

        if self.alarm_counter >= self.config.ALARM_CONTINUOUS_FRAMES:
            self.alarm_triggered = True
        elif self.alarm_counter == 0:
            self.alarm_triggered = False

        return self.alarm_triggered

    # ── 摄像机画面标注 ─────────────────────────────────────

    def draw_camera_overlay(self, frame, alarm_triggered, danger_level):
        """摄像机画面：风险色边框 + 等级条"""
        h, w = frame.shape[:2]
        color = self._risk_color(danger_level, alarm_triggered)

        # 边框
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, BORDER_WIDTH)

        # 顶部风险条
        bar_h = 30
        bar_w = int(w * min(danger_level * 1.5, 1.0))  # 非线性放大
        cv2.rectangle(frame, (0, 0), (bar_w, bar_h), color, -1)

        # 文字
        if alarm_triggered:
            label = "!!! ALARM !!!"
            cv2.putText(frame, label, (w // 2 - 120, bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        elif danger_level > 0.05:
            label = f"Risk: {danger_level:.2f}"
            cv2.putText(frame, label, (10, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    # ── 网格标注 ──────────────────────────────────────────

    def draw_grid_overlay(self, display_grid, alarm_triggered, danger_level):
        """网格画面：等级条 + 报警文字"""
        h, w = display_grid.shape[:2]
        color = self._risk_color(danger_level, alarm_triggered)

        # 底部风险条
        bar_h = 16
        bar_w = int(w * min(danger_level * 1.5, 1.0))
        cv2.rectangle(display_grid, (0, h - bar_h), (bar_w, h), color, -1)

        # 等级文字
        level_text = self._level_text(danger_level, alarm_triggered)
        cv2.putText(display_grid, level_text, (10, h - bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 报警文字
        if alarm_triggered:
            cv2.putText(display_grid, "!!! WARNING: Person too close to Equipment !!!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return display_grid

    # ── 内部 ──────────────────────────────────────────────

    def _risk_color(self, danger_level, triggered):
        if triggered:
            return RISK_COLORS['danger']
        if danger_level >= 0.3:
            return RISK_COLORS['danger']
        if danger_level >= 0.15:
            return RISK_COLORS['warning']
        if danger_level >= 0.01:
            return RISK_COLORS['caution']
        return RISK_COLORS['safe']

    def _level_text(self, danger_level, triggered):
        if triggered:
            return "LEVEL: DANGER"
        if danger_level >= 0.3:
            return "LEVEL: DANGER"
        if danger_level >= 0.15:
            return "LEVEL: WARNING"
        if danger_level >= 0.01:
            return "LEVEL: CAUTION"
        return "LEVEL: SAFE"
