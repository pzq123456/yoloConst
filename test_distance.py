import cv2
import threading
import queue
from ultralytics import YOLO
from pathlib import Path

def video_reader(cap, frame_queue):
    """读取线程：专注于从 RTSP 获取最新帧"""
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # 保持队列中始终只有最新的一帧
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def main():
    # 1. 配置路径
    model_path = Path(r"model\best.pt")
    rtsp_url = "rtsp://118.140.130.26:8554/dahua1003362"

    # 2. 初始化模型
    model = YOLO(model_path)
    
    # 3. 初始化 OpenCV
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not cap.isOpened():
        print("错误：无法连接到 RTSP 流。")
        return

    # 初始化队列与线程
    frame_queue = queue.Queue(maxsize=1)
    reader_thread = threading.Thread(target=video_reader, args=(cap, frame_queue), daemon=True)
    reader_thread.start()

    window_name = "YOLO Real-Time Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--- 系统已启动 (异步读取/跟踪模式) ---")
    print("提示：按 'q' 键退出。")

    try:
        while True:
            # 非阻塞获取最新帧
            if not frame_queue.empty():
                frame = frame_queue.get()
                
                # 4. 推理
                results = model.track(frame, conf=0.35, persist=True, verbose=False)
                
                # 5. 绘制结果
                annotated_frame = results[0].plot()
                cv2.imshow(window_name, annotated_frame)

            # 保持窗口响应
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cv2.destroyAllWindows()
        print("\n--- 程序已安全退出 ---")

if __name__ == "__main__":
    main()