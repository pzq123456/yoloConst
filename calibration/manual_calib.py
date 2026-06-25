"""
manual_calib.py — RTK像素标定工具
在视频画面中手动标记RTK打点对应的像素坐标。
自包含脚本，不依赖外部项目模块。
"""

import cv2
import threading
import queue
import csv
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

RTSP_URL = "rtsp://118.140.130.26:8554/dahua1003362"
SCRIPT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════
# 加载 RTK 数据
# ═══════════════════════════════════════════════════════════════


def load_rtk_data(dat_path):
    """读取 .dat 文件，兼容两种格式:
       PtN,,real_x,real_y,real_z  (5列，第二列为空)
       PtN,real_x,real_y,real_z   (4列)
    """
    points = []
    with open(dat_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            # 5列: PtN,,x,y,z  → 索引 [0][2][3][4]
            # 4列: PtN,x,y,z   → 索引 [0][1][2][3]
            if len(parts) >= 5:
                id_, rx, ry, rz = parts[0], float(parts[2]), float(parts[3]), float(parts[4])
            elif len(parts) == 4:
                id_, rx, ry, rz = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            else:
                continue
            points.append({
                "id": id_,
                "real_x": rx,
                "real_y": ry,
                "real_z": rz,
                "pixel_x": None,
                "pixel_y": None,
            })
    return points


def find_dat_file():
    """自动查找脚本同级目录下第一个 .dat 文件"""
    dat_files = sorted(SCRIPT_DIR.glob("*.dat"))
    if not dat_files:
        print("[错误] 未找到 .dat 文件，请将其放入 calibration/ 目录")
        sys.exit(1)
    return dat_files[0]


def save_result(points, output_path):
    """保存标定结果到 CSV"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "real_x", "real_y", "real_z", "pixel_x", "pixel_y"])
        for p in points:
            writer.writerow([
                p["id"], p["real_x"], p["real_y"], p["real_z"],
                p["pixel_x"] if p["pixel_x"] is not None else "",
                p["pixel_y"] if p["pixel_y"] is not None else "",
            ])
    marked = sum(1 for p in points if p["pixel_x"] is not None)
    print(f"[保存] {output_path.name} ({marked}/{len(points)})")


# ═══════════════════════════════════════════════════════════════
# 视频读取线程（与 test_distance.py 同款）
# ═══════════════════════════════════════════════════════════════


def video_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set() and cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()


# ═══════════════════════════════════════════════════════════════
# 绘制叠加层
# ═══════════════════════════════════════════════════════════════

CROSS_SIZE = 14
CROSS_THICK = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, points, current_idx, paused, blink_on):
    h, w = frame.shape[:2]

    # 所有已标注点：绿色十字 + 编号
    for p in points:
        if p["pixel_x"] is None:
            continue
        px, py = int(p["pixel_x"]), int(p["pixel_y"])
        cv2.drawMarker(frame, (px, py), (0, 255, 0),
                       cv2.MARKER_CROSS, CROSS_SIZE, CROSS_THICK)
        cv2.putText(frame, p["id"], (px + 16, py - 8),
                    FONT, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

    # 当前选中点：黄色大十字（非暂停时也显示，暂停时闪烁）
    if 0 <= current_idx < len(points):
        cp = points[current_idx]
        if cp["pixel_x"] is not None:
            px, py = int(cp["pixel_x"]), int(cp["pixel_y"])
        else:
            px, py = w // 2, h // 2

        if not paused or blink_on:
            cv2.drawMarker(frame, (px, py), (0, 255, 255),
                           cv2.MARKER_CROSS, CROSS_SIZE + 6, CROSS_THICK + 1)
            label = cp["id"]
            if cp["pixel_x"] is not None:
                label += f" ({cp['pixel_x']},{cp['pixel_y']})"
            cv2.putText(frame, label, (px + 20, py - 12),
                        FONT, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    # 底部状态栏
    marked = sum(1 for p in points if p["pixel_x"] is not None)
    status = "|| PAUSED" if paused else ">> PLAYING"
    cp_id = points[current_idx]["id"] if 0 <= current_idx < len(points) else "?"
    bar = f" {status}  |  [{cp_id}]  |  marked: {marked}/{len(points)}  |  S:save  Q:quit"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, dst=frame)
    cv2.putText(frame, bar, (10, h - 7), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════


def main():
    # 1. 加载数据
    dat_path = find_dat_file()
    points = load_rtk_data(dat_path)
    print(f"[加载] {dat_path.name} — {len(points)} 个点")

    # 2. 视频流
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[错误] 无法连接 RTSP 流")
        sys.exit(1)

    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    threading.Thread(target=video_reader, args=(cap, frame_queue, stop_event),
                     daemon=True).start()

    # 3. 运行状态
    paused = False
    current_idx = 0
    blink_counter = 0
    blink_on = True
    last_frame = None     # 暂停时冻结的帧
    live_frame = None     # 最新的实时帧

    window_name = "RTK Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 鼠标回调
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and paused:
            if 0 <= current_idx < len(points):
                points[current_idx]["pixel_x"] = int(x)
                points[current_idx]["pixel_y"] = int(y)
                print(f"[已标] {points[current_idx]['id']} → 像素({int(x)}, {int(y)})")

    cv2.setMouseCallback(window_name, on_mouse)

    print("─" * 50)
    print("[空格]暂停/播放  [↑↓]切换点  [鼠标点击]标记  [S]保存  [Q]退出")
    print("─" * 50)

    # 4. 主循环
    try:
        while True:
            # ── 取帧 ──
            if not frame_queue.empty():
                live_frame = frame_queue.get()
                if not paused:
                    last_frame = live_frame.copy()

            display_frame = None
            if paused and last_frame is not None:
                display_frame = last_frame.copy()
            elif live_frame is not None:
                display_frame = live_frame.copy()

            if display_frame is None:
                if cv2.waitKeyEx(1) == ord("q"):
                    break
                continue

            # ── 闪烁 ──
            blink_counter += 1
            if blink_counter >= 15:
                blink_counter = 0
                blink_on = not blink_on

            # ── 绘制 & 显示 ──
            draw_overlay(display_frame, points, current_idx, paused, blink_on)
            cv2.imshow(window_name, display_frame)

            # ── 键盘（waitKeyEx 可捕获方向键） ──
            key = cv2.waitKeyEx(1)

            if key == ord("q") or key == 27:  # Q 或 ESC
                break

            elif key == ord(" "):
                paused = not paused
                marked = sum(1 for p in points if p["pixel_x"] is not None)
                state = "暂停" if paused else "播放"
                cp_id = points[current_idx]["id"]
                print(f"[{state}] 当前: {cp_id} | 已标: {marked}/{len(points)}")
                if paused and points[current_idx]["pixel_x"] is None:
                    print(f"  → 请在画面中点击 {cp_id} 的位置")
                blink_on = True
                blink_counter = 0

            elif key == ord("s"):
                save_result(points, SCRIPT_DIR / "calib_result.csv")

            elif key == 2490368:  # ↑
                current_idx = (current_idx - 1) % len(points)
                cp = points[current_idx]
                status = "✓" if cp["pixel_x"] is not None else "?"
                print(f"[切换] {status} {cp['id']}")
                blink_on = True
                blink_counter = 0

            elif key == 2621440:  # ↓
                current_idx = (current_idx + 1) % len(points)
                cp = points[current_idx]
                status = "✓" if cp["pixel_x"] is not None else "?"
                print(f"[切换] {status} {cp['id']}")
                blink_on = True
                blink_counter = 0

    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        print("[退出]")

    # 5. 退出时自动保存
    marked = sum(1 for p in points if p["pixel_x"] is not None)
    if marked > 0:
        save_result(points, SCRIPT_DIR / "calib_result.csv")


if __name__ == "__main__":
    main()
