# monitor.py — 后台持续监控，报警时自动保存截图
import os
import sys
import time
from datetime import datetime
import cv2
from ultralytics import YOLO
from config import Config
from bev_processor import BEVProcessor
from detection_processor import DetectionProcessor
from alarm_manager import AlarmManager
from video_stream import VideoStream
from frame_processor import FrameProcessor
from utils import compute_bev_size


def main():
    # ── 初始化 ──────────────────────────────────────────
    config = Config()
    bev_w, bev_h = compute_bev_size(config.SRC_PTS, config.BEV_SCALE)
    print(f"[INIT] BEV={bev_w}x{bev_h}")
    config.initialize_bev(bev_w, bev_h)

    model = YOLO(config.MODEL_PATH)
    video_stream = VideoStream(
        config.RTSP_URL,
        width=config.STREAM_WIDTH,
        height=config.STREAM_HEIGHT,
    )
    bev_processor = BEVProcessor(config)
    detection_processor = DetectionProcessor(config)
    alarm_manager = AlarmManager(config)
    frame_processor = FrameProcessor(config, model, bev_processor,
                                     detection_processor, alarm_manager)

    # ── 输出目录 ────────────────────────────────────────
    output_root = os.path.join(os.path.dirname(__file__), "..", "alarms")
    os.makedirs(output_root, exist_ok=True)

    # ── 启动视频流 ──────────────────────────────────────
    try:
        video_stream.start()
        print("[INIT] 视频流已连接，开始监控...")
        print("[INIT] 按 Ctrl+C 退出\n")
    except Exception as e:
        print(f"[FATAL] 启动视频流失败: {e}")
        return

    alarm_was_on = False
    frame_count = 0

    try:
        while True:
            frame = video_stream.get_frame()
            if frame is None:
                print("[WARN] 流超时，重连中...")
                try:
                    video_stream.reconnect()
                except Exception as e:
                    print(f"[FATAL] 重连失败: {e}")
                    break
                continue

            frame_count += 1
            result = frame_processor.process_frame(frame)

            # ── 报警状态变更检测 ──────────────────────────
            alarm_now = alarm_manager.alarm_triggered

            if alarm_now and not alarm_was_on:
                # 报警开始：保存截图
                danger_map = frame_processor.grid_manager.get_danger_map()
                global_risk = float(danger_map.max())

                now = datetime.now()
                date_dir = now.strftime("%Y-%m-%d")
                ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

                day_dir = os.path.join(output_root, date_dir)
                os.makedirs(day_dir, exist_ok=True)

                cam_path = os.path.join(day_dir, f"{ts}_cam.jpg")
                grid_path = os.path.join(day_dir, f"{ts}_grid.jpg")

                cv2.imwrite(cam_path, result['annotated_frame'])
                cv2.imwrite(grid_path, result['grid_map'])

                print(f"[ALARM] {ts}  risk={global_risk:.3f}"
                      f"  |  saved → {date_dir}/")

            elif not alarm_now and alarm_was_on:
                # 报警结束
                now = datetime.now()
                ts = now.strftime("%H:%M:%S")
                danger_map = frame_processor.grid_manager.get_danger_map()
                print(f"[CLEAR] {ts}  risk={float(danger_map.max()):.3f}")

            alarm_was_on = alarm_now

    except KeyboardInterrupt:
        print(f"\n[DONE] 共处理 {frame_count} 帧，正常退出")

    finally:
        video_stream.release()
        print("[DONE] 资源已释放")


if __name__ == "__main__":
    main()
