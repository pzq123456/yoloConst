# video_stream.py
import os
import time
import threading
import queue
import cv2

# 超时 5 秒 + TCP 传输
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'


class VideoStream:
    """
    视频流管理器。

    关键原则：主线程绝不碰 cap 对象。
    读帧线程拥有 cap 的全部生命周期（打开→读取→释放）。
    重连时主线程只发信号、等线程退出、再起新线程。
    """

    def __init__(self, rtsp_url, width=None, height=None):
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None
        self.last_frame_time = 0.0

    # ── 公开接口 ─────────────────────────────────────────

    def start(self):
        self.stop_event.clear()
        self._launch_reader()

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            # 情况1：读帧线程已死 → 通知上层重连
            if self.reader_thread and not self.reader_thread.is_alive():
                return None
            # 情况2：线程还活着但超过 10s 没产出 → 假死 → 通知上层重连
            if time.time() - self.last_frame_time > 10:
                return None
            return None

    def reconnect(self):
        """等旧线程完全退出后，起新线程"""
        # 1. 通知旧线程停止
        self.stop_event.set()
        # 2. 等它自己退出（cap.release() 在线程内部完成）
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=10)
        # 3. 清空残留帧
        self._drain_queue()
        # 4. 指数退避重试
        for attempt in range(1, 100):
            delay = min(2.0 ** attempt, 32.0)
            print(f"[RECONNECT] 第 {attempt} 次重试，等待 {delay:.0f}s...")
            time.sleep(delay)
            if self._launch_reader():
                print("[RECONNECT] 重连成功")
                return
        raise Exception("重连失败：已达最大重试次数")

    def release(self):
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=5)

    # ── 内部 ─────────────────────────────────────────────

    def _launch_reader(self):
        """起一个新的读帧线程"""
        self.stop_event.clear()
        self.reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        # 等一小段时间看它能不能跑起来
        self.reader_thread.join(timeout=1.0)
        return self.reader_thread.is_alive()

    def _reader_loop(self):
        """
        读帧线程 —— 完全拥有 cap 对象。
        从打开到释放都在这个线程里，主线程绝不插手。
        """
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            return

        bad_streak = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    bad_streak += 1
                    if bad_streak >= 3:
                        break
                    continue
                bad_streak = 0

                self.last_frame_time = time.time()
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
        except Exception:
            pass
        finally:
            # 释放只在线程内部发生
            try:
                cap.release()
            except Exception:
                pass

    def _drain_queue(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
