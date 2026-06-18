# display_manager.py
import cv2
from utils import draw_fps_on_image

class DisplayManager:
    """显示管理器 - 管理所有窗口显示"""
    
    def __init__(self):
        self.window_names = {
            'camera': "Camera View (ROI)",
            'bev': "BEV View",
            'grid': "Occupancy Grid"
        }
        self.fps_display = "FPS: --"
    
    def update(self, annotated_frame, bev_img, grid_map, fps_display=None):
        """
        更新所有显示窗口
        
        参数:
            annotated_frame: 标注后的原始图像
            bev_img: BEV图像
            grid_map: 网格地图
            fps_display: FPS显示字符串（可选）
        """
        # 更新FPS显示
        if fps_display:
            self.fps_display = fps_display
        
        # 在图像上绘制FPS
        draw_fps_on_image(bev_img, self.fps_display)
        draw_fps_on_image(grid_map, self.fps_display)
        
        # 显示所有窗口
        cv2.imshow(self.window_names['camera'], annotated_frame)
        cv2.imshow(self.window_names['bev'], bev_img)
        cv2.imshow(self.window_names['grid'], grid_map)
    
    def check_exit(self):
        """检查是否按下退出键"""
        return cv2.waitKey(1) & 0xFF == ord('q')
    
    def close_all(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()