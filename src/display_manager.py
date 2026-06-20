# display_manager.py
import cv2
from utils import draw_fps_on_image

class DisplayManager:
    """显示管理器"""

    def __init__(self, show_bev=False):
        self.show_bev = show_bev
        self.window_names = {
            'camera': "Camera View (ROI)",
            'bev': "BEV View",
            'grid': "Occupancy Grid"
        }
        self.fps_display = "FPS: --"

    def update(self, annotated_frame, bev_img, grid_map, fps_display=None):
        if fps_display:
            self.fps_display = fps_display

        draw_fps_on_image(grid_map, self.fps_display)

        cv2.imshow(self.window_names['camera'], annotated_frame)
        if self.show_bev:
            draw_fps_on_image(bev_img, self.fps_display)
            cv2.imshow(self.window_names['bev'], bev_img)
        cv2.imshow(self.window_names['grid'], grid_map)

    def check_exit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close_all(self):
        cv2.destroyAllWindows()
