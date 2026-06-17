import cv2
import threading
from ultralytics import YOLO
from pathlib import Path
import time

class RTSPStreamer:
    """多线程 RTSP 读取类，确保永远获取最新的一帧"""
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "model" / "best.engine"
    rtsp_url = "rtsp://118.140.130.26:8554/dahua1003362"
    output_video = script_dir / "output_video_10s.mp4"
    
    model = YOLO(str(model_path), task="detect")
    
    # 使用多线程读取
    streamer = RTSPStreamer(rtsp_url)
    time.sleep(1) # 等待启动
    
    frame = streamer.read()
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    
    print("--- 开始优化处理 ---")
    frame_count = 0
    total_frames = 250 # 10秒 @ 25fps
    start_all = time.time()
    
    try:
        while frame_count < total_frames:
            t1 = time.time()
            
            # 从缓存读取最新帧，没有阻塞
            frame = streamer.read()
            
            # 使用 predict 替代 track (如果你不需要 ID，这能快 20%+)
            # 如果必须用 track，persist=True 必须保留
            results = model.predict(
                frame, conf=0.35, verbose=False, half=True, device=0
            )
            
            t2 = time.time()
            
            # 仅在需要时绘制
            annotated_frame = results[0].plot()
            
            inference_ms = (t2 - t1) * 1000
            cv2.putText(annotated_frame, f"Inf: {inference_ms:.1f}ms", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"进度: {frame_count}/{total_frames} | 推理延迟: {inference_ms:.1f}ms")
                
    finally:
        streamer.stop()
        out.release()
        print(f"完成！平均耗时: {(time.time()-start_all)/frame_count*1000:.1f}ms/帧")

if __name__ == "__main__":
    main()