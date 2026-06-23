# main.py — 显示窗口 + 报警截图
import os
from datetime import datetime
import cv2
from ultralytics import YOLO
from config import Config
from bev_processor import BEVProcessor
from detection_processor import DetectionProcessor
from alarm_manager import AlarmManager
from video_stream import VideoStream
from frame_processor import FrameProcessor
from display_manager import DisplayManager
from fps_manager import FPSManager
from utils import compute_bev_size


def main():
    # 1. 初始化配置
    config = Config()

    # 2. 计算BEV尺寸
    bev_w, bev_h = compute_bev_size(config.SRC_PTS, config.BEV_SCALE)
    print(f"BEV尺寸: {bev_w}x{bev_h}")
    config.initialize_bev(bev_w, bev_h)

    # 3. 初始化各个模块
    model = YOLO(config.MODEL_PATH)
    video_stream = VideoStream(
        config.RTSP_URL,
        width=config.STREAM_WIDTH,
        height=config.STREAM_HEIGHT,
    )
    bev_processor = BEVProcessor(config)
    detection_processor = DetectionProcessor(config)
    alarm_manager = AlarmManager(config)
    display_manager = DisplayManager(show_bev=config.DEBUG_MODE)
    fps_manager = FPSManager()

    # 4. 初始化帧处理器
    frame_processor = FrameProcessor(
        config, model, bev_processor,
        detection_processor, alarm_manager,
    )

    # 5. 报警截图目录
    output_root = os.path.join(os.path.dirname(__file__), "..", "alarms")
    os.makedirs(output_root, exist_ok=True)

    # 6. 启动视频流
    try:
        video_stream.start()
    except Exception as e:
        print(f"启动视频流失败: {e}")
        return

    alarm_was_on = False

    try:
        while True:
            # 7. 获取视频帧
            frame = video_stream.get_frame()
            if frame is None:
                print("流超时，尝试重连...")
                try:
                    video_stream.reconnect()
                except Exception as e:
                    print(f"重连失败: {e}")
                    break
                continue

            # 8. 处理帧
            result = frame_processor.process_frame(frame)

            # 9. 更新FPS
            fps_manager.update()

            # 10. 报警截图
            alarm_now = alarm_manager.alarm_triggered
            if alarm_now and not alarm_was_on:
                global_risk = float(
                    frame_processor.grid_manager.get_danger_map().max()
                )
                now = datetime.now()
                date_dir = now.strftime("%Y-%m-%d")
                ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                day_dir = os.path.join(output_root, date_dir)
                os.makedirs(day_dir, exist_ok=True)

                cv2.imwrite(
                    os.path.join(day_dir, f"{ts}_cam.jpg"),
                    result['annotated_frame'],
                )
                cv2.imwrite(
                    os.path.join(day_dir, f"{ts}_grid.jpg"),
                    result['grid_map'],
                )
                print(f"[ALARM] {ts}  risk={global_risk:.3f}  |  {date_dir}/")
            elif not alarm_now and alarm_was_on:
                global_risk = float(
                    frame_processor.grid_manager.get_danger_map().max()
                )
                print(f"[CLEAR] {datetime.now().strftime('%H:%M:%S')}"
                      f"  risk={global_risk:.3f}")

            alarm_was_on = alarm_now

            # 11. 显示
            display_manager.update(
                result['annotated_frame'],
                result['bev_img'],
                result['grid_map'],
                fps_manager.get_display(),
            )

            # 12. 检查退出
            if display_manager.check_exit():
                break

    except KeyboardInterrupt:
        print("用户中断")

    finally:
        # 13. 清理资源
        video_stream.release()
        display_manager.close_all()
        print("程序退出")


if __name__ == "__main__":
    main()
