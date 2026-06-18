# video_stream.py
import cv2
import threading
import queue
from utils import video_reader

class VideoStream:
    """视频流管理器"""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.reader_thread = None
        self.is_running = False
    
    def start(self):
        """启动视频流"""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise Exception("无法打开视频流")
        self.is_running = True
        self.reader_thread = threading.Thread(
            target=video_reader, 
            args=(self.cap, self.frame_queue), 
            daemon=True
        )
        self.reader_thread.start()
    
    def get_frame(self, timeout=1.0):
        """获取一帧图像"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def reconnect(self):
        """重新连接"""
        self.cap.release() # type: ignore
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise Exception("重连失败")
        self.reader_thread = threading.Thread(
            target=video_reader, 
            args=(self.cap, self.frame_queue), 
            daemon=True
        )
        self.reader_thread.start()
    
    def release(self):
        """释放资源"""
        self.is_running = False
        if self.cap:
            self.cap.release()