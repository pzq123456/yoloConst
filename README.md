# yolo 工地目标检测

tensorboard --logdir /runs 
uv run train.py
tar -xf data7.tar -C data7

# Install or upgrade the ultralytics package from PyPI
pip install ultralytics

## 数据集：
> - https://universe.roboflow.com/fyp-rk465/fyp_mocsdataset/dataset/3

The dataset includes 9985 images.
Excavators are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 1 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down



## 监控对象类别说明
| 类别 (Name) | 对应含义 | 对你项目的意义 |
|---|---|---|
| Worker | 工人 | 核心目标：监控人员是否处于危险区域。 |
| Excavator | 挖掘机 | 高频大型器械，最需要重点防护的设备。 |
| Bulldozer | 推土机 | 大型器械，作业半径大。 |
| Truck | 卡车/渣土车 | 工地物流，常在人员周围穿梭。 |
| Concrete mixer | 搅拌车 | 大型车辆。 |
| Loader | 装载机 | 大型器械。 |
| Pump truck | 泵车 | 施工关键大型设备。 |
| Crane | 起重机 (通用) | 涉及高空吊装，危险性极高。 |
| Static crane | 固定式起重机 | 塔吊等，需监控作业覆盖范围。 |
| Pile driving | 打桩机 | 施工机械，震动和噪声大。 |
| Roller | 压路机 | 压实机械。 |
| Hanging head | 吊钩/吊具 | 关键细节：吊装作业的核心风险点。 |
| Other vehicle | 其他车辆 | 兜底类别。 |

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130



## 自己准备第二份数据集

equipment
worker
