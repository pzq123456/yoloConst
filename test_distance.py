import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # 1. 配置路径
    # model_path = Path(r"runs\detect\fyp_mocs\weights\best.pt")
    # runs\detect\fyp_mocs_max\weights\best.pt
    # model_path = Path(r"runs\detect\fyp_mocs_max\weights\best.pt")
    # runs\detect\yolo26n_self_dataset\weights\best.pt
    # model_path = Path(r"runs\detect\yolo26n_self_dataset\weights\best.pt")
    # runs\detect\yolo26n_finetune\weights\best.pt
    # model_path = Path(r"runs\detect\yolo26n_finetune-3\weights\best.pt")
    # runs\detect\yolo26s_mocs\weights\best.pt
    model_path = Path(r"runs\detect\yolo26s_mocs\weights\best.pt")
    rtsp_url = "rtsp://118.140.130.26:8554/dahua1003362"


    # 2. 初始化模型
    model = YOLO(model_path)
    
    # 3. 初始化 OpenCV 视频捕获
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not cap.isOpened():
        print("错误：无法连接到 RTSP 流。")
        return

    window_name = "YOLO Real-Time Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--- 系统已启动 (YOLO 跟踪模式) ---")
    print("提示：按 'q' 键退出。")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 4. 使用 track 模式代替 predict
            # persist=True 启用跨帧跟踪，tracker 选择默认的 'bytetrack.yaml'
            results = model.track(frame, conf=0.35, persist=True, verbose=False)
            
            # 5. 绘制结果 (track 结果会自动包含追踪 ID)
            annotated_frame = results[0].plot()

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n--- 程序已安全退出 ---")

if __name__ == "__main__":
    main()