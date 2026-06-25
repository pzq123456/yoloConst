# YOLO Construction Site Safety Monitor

基于 YOLO 目标检测 + 单应性标定的工地人-设备距离报警系统。通过平面单应性矩阵将 2D 像素坐标映射到物理地面坐标系，在真实世界尺度上计算人员与设备之间的空间占用关系。

---

## 系统架构

```
RTSP 视频流
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│ video_stream │───▶│ frame_processor  │───▶│ display_mgr  │
└─────────────┘    └────────┬─────────┘    └──────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │  YOLO    │    │ detection_   │    │  grid_       │
   │  model   │    │ processor    │    │  manager     │
   │  .track()│    │ 像素→世界坐标 │    │ 占用网格     │
   └──────────┘    └──────┬───────┘    └──────┬───────┘
                          │                   │
                          ▼                   ▼
                   ┌──────────┐        ┌──────────────┐
                   │  utils   │        │  alarm_       │
                   │  坐标变换 │        │  manager     │
                   │  可视化   │        │  报警防抖     │
                   └──────────┘        └──────────────┘
```

### 模块说明

| 文件 | 职责 |
|------|------|
| `main.py` | 程序入口，初始化配置、各模块、显示、报警截图 |
| `config.py` | 全局配置管理（网格参数、检测参数、标定加载与派生计算） |
| `frame_processor.py` | 每帧处理管线：推理 → 坐标提取 → 网格更新 → 可视化叠加 |
| `detection_processor.py` | YOLO 检测结果处理：像素坐标 → 相机对齐物理坐标，足迹生成 |
| `grid_manager.py` | 双层占用网格（人员/设备），Hit Counter 抗闪烁，风险图计算 |
| `alarm_manager.py` | 报警防抖（连续帧确认），风险等级可视化 |
| `bev_processor.py` | BEV 鸟瞰图生成（debug 用） |
| `video_stream.py` | RTSP 视频流读取，支持断线重连 |
| `utils.py` | 坐标变换工具函数 + 可视化工具（透视网格、占用叠加、坐标系） |
| `display_manager.py` | 多窗口显示管理 |
| `fps_manager.py` | FPS 统计 |
| `monitor.py` | 进程看门狗 |

---

## 核心原理

### 1. 平面单应性（Plane Homography）

假设地面为平坦平面，像素坐标 $(u,v)$ 与局部物理米坐标 $(l_x, l_y)$ 之间存在 3×3 单应性映射 $H$：

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad l_x = \frac{x'}{w'}, \quad l_y = \frac{y'}{w'}$$

UTM 绝对坐标：$W_x = l_x + \text{world\_mean}_x,\quad W_y = l_y + \text{world\_mean}_y$

其中 $\text{world\_mean}$ 是标定控制点 UTM 坐标的均值，中心化避免浮点精度下溢。

**逆向**（世界 → 像素，用于叠加渲染）：

$$\begin{bmatrix} u' \\ v' \\ w' \end{bmatrix} = H^{-1} \cdot \begin{bmatrix} l_x \\ l_y \\ 1 \end{bmatrix}, \quad u = \frac{u'}{w'}, \quad v = \frac{v'}{w'}$$

### 2. 坐标系链

```
OpenCV 像素 (px, py)          ← 图像左上角原点, x→右, y↓
        │
        ▼  H @ [px, py, 1]^T
局部米 (lx, ly)               ← 相对 world_mean 的偏移
        │
        ▼  local_to_camera()
相机对齐 (cam_x, cam_y)       ← cam_y = 前方, cam_x = 右方
        │
        ▼  cam_to_grid_col / cam_to_grid_row
网格索引 (row, col)            ← row 0 = 远处, row N-1 = 近处
```

**关键变换函数**（均在 `utils.py`）：

| 函数 | 方向 |
|------|------|
| `pixel_to_local_meters(px, py, H)` | 像素 → 局部米 |
| `pixel_to_world(px, py, H, world_mean)` | 像素 → UTM 世界米 |
| `world_to_pixel(wx, wy, H_inv, world_mean)` | UTM 世界 → 像素 |
| `local_to_camera(lx, ly, ...)` | 局部米 → 相机对齐坐标 |
| `camera_to_local(cam_x, cam_y, ...)` | 相机对齐 → 局部米（反向） |

### 3. 相机坐标系（cam_x, cam_y）

相机坐标系的原点 $(0, 0)$ 对应摄像头在地面的**近似投影点**（由图像下边缘中心通过 $H$ 映射到地面得到）。

- `cam_y`：相机**前方**距离（远离相机为正），图像上方 ≈ cam_y 增大
- `cam_x`：相机**右方**距离，图像右方 ≈ cam_x 增大

基向量 `CAM_FORWARD` 和 `CAM_RIGHT` 由图像中心、正上方、正右方三个参考点通过 $H$ 投影后在局部米空间中计算，并做 Gram-Schmidt 正交化保证正交性。

### 4. 占用网格（Occupancy Grid）

#### Hit Counter 机制

每个单元格维护一个浮点数计数，遵循"看到 +conf / 看不到 -decay"规则：

```
每帧流程:
  1. 清空本帧权重
  2. 遍历检测结果 → 相机坐标 → 网格行列 → 记录每个cell本帧最大置信度
  3. 未检测到的cell → hits -= decay（冷却）
  4. 检测到的cell → hits += conf（不超过 MAX）
```

关键参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `PERSON_MAX_HITS` | 3 | 人员 3 帧确认（~0.1s） |
| `EQUIPMENT_MAX_HITS` | 10 | 设备 10 帧确认 |
| `PERSON_DECAY` | 1.0/帧 | 人员消失快（离开即清零） |
| `EQUIPMENT_DECAY` | 0.25/帧 | 设备缓慢衰减（避免短暂遮挡导致闪烁） |

> **设计理念**：人员快速移动，需要快速响应；设备静止，需要抗遮挡。两者衰减速率不同。

#### 风险计算

```
风险图 = person_prob ⊙ dilate(equip_prob, 危险半径)
```

- `person_prob = person_hits / PERSON_MAX_HITS`（归一化到 [0, 1]）
- `equip_prob = equip_hits / EQUIPMENT_MAX_HITS`
- 对 equip_prob 做椭圆形态学膨胀（半径 = `EQUIPMENT_DANGER_RADIUS_M / GRID_CELL_SIZE_M` 个 cell）
- 两图逐元素相乘 → 人员靠近设备时风险值 > 0

#### 足迹生成

人员检测到后，在 camera 坐标系中以其脚底位置为圆心、`PERSON_FOOTPRINT_RADIUS_M = 0.6m` 为半径，填充所有中心在圆内的网格单元。算法以 cell 为中心遍历（而非坐标采样），**避免 radius ≈ cell_size 时退化为中心十字线**。

### 5. 可视化对齐验证

系统提供两种互补的可视化，用于验证坐标变换链路的正确性：

| 视图 | 内容 |
|------|------|
| **摄像头画面叠加** (`draw_occupancy_overlay`) | 占用网格 cell 反投影到原始画面，半透明填充。绿色=人，红色=设备，品红=重叠 |
| **网格显示** (`draw_grid_coordinates`) | BEV 网格 + 物理坐标系（X/Y 轴、米制刻度、方向标识） |

> **对齐验证方法**：观察摄像头画面上的彩色 cell 是否与人/设备的实际位置重合。如果绿色格子始终覆盖在人的脚底，说明 $H$ 矩阵标定正确。

---

## 配置参数参考

```python
# ── 标定 ──
CALIB_PATH = "calibration/camera_calib_371.npz"   # H 矩阵 + world_mean

# ── 物理网格 ──
GRID_CELL_SIZE_M = 0.5           # 单元格 = 0.5m × 0.5m
GRID_X_RANGE_M = 40.0            # 左右 ±20m
GRID_Y_NEAR_M = -5.0             # 相机后方 5m
GRID_Y_FAR_M = 30.0              # 相机前方 30m（对应标定有效范围 0~15m + buffer）
BEV_PIXELS_PER_METER = 25        # 可视化缩放（1m = 25px）
EQUIPMENT_DANGER_RADIUS_M = 3.0  # 设备危险半径

# ── 检测 ──
CONF_THRESH = 0.55               # YOLO 置信度阈值
IMGSZ = 640                      # YOLO 输入尺寸
PERSON_CLASS = 1                 # COCO 类别 ID（1 = person）

# ── Hit Counter ──
PERSON_MAX_HITS = 3              # 人员确认帧数
EQUIPMENT_MAX_HITS = 10          # 设备确认帧数
PERSON_DECAY = 1.0               # 人员衰减速率（/帧）
EQUIPMENT_DECAY = 0.25           # 设备衰减速率（/帧，约 2.5s @10fps）

# ── 报警 ──
ALARM_CONTINUOUS_FRAMES = 3      # 连续报警帧数（防抖）
ALARM_RISK_THRESHOLD = 0.15      # 风险阈值 [0, 1]

# ── 其他 ──
DEBUG_MODE = False               # 是否显示 BEV 调试窗口
```

### 参数调优指南

| 场景 | 调整建议 |
|------|----------|
| 设备消失太快 / 闪烁 | 减小 `EQUIPMENT_DECAY`（如 0.1~0.15）或增大 `EQUIPMENT_MAX_HITS` |
| 人员足迹太小 | 增大 `PERSON_FOOTPRINT_RADIUS_M`（如 0.7） |
| 报警太敏感 | 增大 `ALARM_RISK_THRESHOLD`（如 0.2~0.3）或 `ALARM_CONTINUOUS_FRAMES` |
| 报警响应太慢 | 减小 `ALARM_CONTINUOUS_FRAMES` |
| 需要更大监控范围 | 增大 `GRID_Y_FAR_M`（注意：>20m 标定精度下降） |
| 网格显示太密/太疏 | 调整 `BEV_PIXELS_PER_METER` |

---

## 运行方法

### 环境要求

- Python 3.10+
- OpenCV, NumPy, PyTorch, Ultralytics YOLO

### 启动

```bash
# 确保标定文件存在
ls calibration/camera_calib_371.npz

# 启动主程序
cd src
python main.py
```

### 标定流程（如需重新标定）

```bash
# Step 1: 加载原始控制点，RANSAC 清洗离群点，视频流验证
python calibration/compute_calib.py

# Step 2: 用清洗后数据计算最终 H 矩阵，可视化透视网格，保存 .npz
python calibration/compute_and_grid_vis.py
```

详见 `calibration/README.md`。

---

## 已知限制与注意事项

### 1. 鱼眼镜头畸变

监控摄像头通常为广角镜头，存在明显的径向畸变。单应性矩阵假设**完美的针孔投影**，无法补偿畸变。表现为：

> 画面左侧的占用网格向左偏移，右侧的向右偏移（离光轴越远偏差越大）。

**未来改进**：用棋盘格标定获取畸变系数 $(k_1, k_2, p_1, p_2)$，在推理前用 `cv2.undistort()` 校正每一帧。

### 2. 标定精度

| 距离范围 | 定位精度 | 说明 |
|----------|----------|------|
| 0 ~ 15 m | **~15 cm** | 适用精确距离报警 |
| 15 ~ 20 m | **~60 cm** | 仅适合区域判断 |
| > 20 m | 不可靠 | 超出控制点覆盖范围（当前 `GRID_Y_FAR_M=30m` 的远端仅做参考） |

整体 RMSE ≈ 34 cm，控制点 23 对（RANSAC 剔除 3 个离群点后）。

### 3. 地面平面假设

单应性模型假设地面**绝对平坦**。工地虽有硬化路面，但局部 ±5~10 cm 的不平整会导致该区域的定位产生额外误差。

### 4. BEV 调试视图

`bev_processor.py` 中的 BEV warping 使用旧的硬编码 SRC_PTS（非标定 $H$ 矩阵），仅用于 `DEBUG_MODE` 下的粗略参考。**正式的坐标变换和占用计算使用标定的 $H$ 矩阵**，BEV 视图与占用网格不完全一致是正常的。

### 5. 网格方向约定

- 网格显示（`grid_map`）：上方 = 远处（FWD），下方 = 近处（Camera），左侧 = LEFT，右侧 = RIGHT
- 坐标系原点（cam_x=0, cam_y=0）在网格下方约 86% 处（因为仅包含 5m 后方 vs 30m 前方）
- 所有方向标签使用 ASCII 字符（`<-` `->` `^` `v`），避免 OpenCV 默认字体渲染 Unicode 箭头为 `???`

### 6. 车牌/车辆处理

当前系统对非 person 类别的检测统一视为"设备"（红色），使用简单的 bbox 地面投影估算占用区域。车辆的长宽比固定为 1.8:1，宽度裁剪到 1.5~3.0m。对于精确的车辆姿态估计，需引入 3D 检测或实例分割。

---

## 文件结构

```
yoloConst/
├── README.md                          # 本文件
├── calibration/
│   ├── README.md                      # 标定模块详细文档
│   ├── 371摄像头坐标映射.txt           # 原始控制点数据（26 对）
│   ├── 371摄像头坐标映射_cleaned.txt    # RANSAC 清洗后（23 对）
│   ├── camera_calib_371.npz           # 最终标定参数（H + world_mean）
│   ├── compute_calib.py               # 标定 Step 1：RANSAC 清洗
│   ├── compute_and_grid_vis.py        # 标定 Step 2：最终矩阵 + 可视化
│   └── 20260623180436.dat             # 原始 RTK 采集数据
├── model/
│   └── yolo26s_mocs_20260623_0216.pt  # YOLO 模型权重
├── src/
│   ├── main.py                        # 主入口
│   ├── config.py                      # 全局配置
│   ├── frame_processor.py             # 帧处理管线
│   ├── detection_processor.py         # 检测结果处理 + 坐标提取
│   ├── grid_manager.py                # 双层占用网格
│   ├── alarm_manager.py               # 报警管理
│   ├── bev_processor.py               # BEV 鸟瞰图生成
│   ├── utils.py                       # 坐标变换 + 可视化工具
│   ├── video_stream.py                # RTSP 视频流
│   ├── display_manager.py             # 显示窗口管理
│   ├── fps_manager.py                 # FPS 统计
│   ├── monitor.py                     # 进程看门狗
│   └── launcher.py                    # 启动器
└── alarms/                            # 报警截图输出目录
    └── YYYY-MM-DD/
        ├── HHMMSS_xxx_cam.jpg         # 报警时摄像头画面
        └── HHMMSS_xxx_grid.jpg        # 报警时占用网格
```
