# fps_manager.py
import time

class FPSManager:
    """FPS管理器"""
    
    def __init__(self):
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.fps_display = "FPS: --"
    
    def update(self):
        """
        更新FPS计数
        
        返回:
            str: 如果更新了FPS显示，返回新的FPS字符串，否则返回None
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            fps_display = f"FPS: {self.fps_counter:.1f}"
            self.fps_counter = 0
            self.fps_timer = current_time
            self.fps_display = fps_display
            return fps_display
        
        return None
    
    def get_display(self):
        """获取当前FPS显示字符串"""
        return self.fps_display