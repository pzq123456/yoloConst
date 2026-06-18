# main.py
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
    print(f"BEV尺寸: {bev_w}×{bev_h}")
    config.initialize_bev(bev_w, bev_h)
    
    # 3. 初始化各个模块
    model = YOLO(config.MODEL_PATH)
    video_stream = VideoStream(config.RTSP_URL)
    bev_processor = BEVProcessor(config)
    detection_processor = DetectionProcessor(config)
    alarm_manager = AlarmManager(config)
    display_manager = DisplayManager()
    fps_manager = FPSManager()
    
    # 4. 初始化帧处理器
    frame_processor = FrameProcessor(
        config, model, bev_processor, 
        detection_processor, alarm_manager
    )
    
    # 5. 启动视频流
    try:
        video_stream.start()
    except Exception as e:
        print(f"启动视频流失败: {e}")
        return
    
    try:
        while True:
            # 6. 获取视频帧
            frame = video_stream.get_frame()
            if frame is None:
                print("流超时，尝试重连...")
                try:
                    video_stream.reconnect()
                except Exception as e:
                    print(f"重连失败: {e}")
                    break
                continue
            
            # 7. 处理帧
            result = frame_processor.process_frame(frame)
            
            # 8. 更新FPS
            fps_manager.update()
            
            # 9. 显示
            display_manager.update(
                result['annotated_frame'],
                result['bev_img'],
                result['grid_map'],
                fps_manager.get_display()
            )
            
            # 10. 检查退出
            if display_manager.check_exit():
                break
    
    except KeyboardInterrupt:
        print("用户中断")
    
    finally:
        # 11. 清理资源
        video_stream.release()
        display_manager.close_all()
        print("程序退出")

if __name__ == "__main__":
    main()