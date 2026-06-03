# 马匹行为识别 — 多帧时序推理 (TCN)

## 概述

多帧时序识别使用 **TCN（Temporal Convolutional Network）** 对马的行为进行窗口级推理。
每 8 FPS 采样，滑动窗口 32 帧（4 秒），一次推理输出一个行为标签。

支持四类行为：`standing`（站立）、`eating`（进食）、`drinking`（喝水）、`lying`（卧倒）。

## 模型文件

| 文件 | 用途 |
| --- | --- |
| `runs/timesequence/tcn_behavior_8fps_4s/best.pt` | TCN 行为分类模型 |
| `runs/multiframes/horse_multiframe_detect/weights/best.pt` | YOLO 检测模型（horse/head/water/person/feces） |
| `runs/multiframes/horse_multiframe_segment/weights/best.pt` | YOLO 分割模型（stall 马厩） |
| `dataset/timesequence_8fps_4s/normalization.npz` | 特征归一化参数 |
| `config/feed_regions.yaml` | 固定食槽区域配置 |

## 快速开始

```powershell
.\.venv\Scripts\python.exe -m horse_behavior.infer_temporal_behavior `
  --source video\stable_20260523_105109.mp4 `
  --tcn-model runs\timesequence\tcn_behavior_8fps_4s\best.pt `
  --dataset-dir dataset\timesequence_8fps_4s `
  --sample-fps 8 `
  --render-mode full `
  --output-fps 30 `
  --no-display
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--source` | `video/stable_20260523_105109.mp4` | 输入视频路径 |
| `--output` | `outputs/tcn_behavior_infer.mp4` | 输出标注视频 |
| `--csv` | `outputs/tcn_behavior_infer.csv` | 帧级行为 CSV（传空字符串关闭） |
| `--tcn-model` | TCN 模型路径 | TCN 行为 checkpoint |
| `--det-model` | YOLO 检测权重 | 目标检测模型 |
| `--segment-model` | YOLO 分割权重 | 马厩分割模型 |
| `--dataset-dir` | 时序数据集目录 | 需包含 `normalization.npz` |
| `--sample-fps` | `2.0` | TCN 采样帧率（推荐 8） |
| `--start-sec` | `0` | 起始时间（秒） |
| `--end-sec` | `0` | 结束时间（秒），0 表示视频末尾 |
| `--max-seconds` | `0` | 从起始时间起最大处理秒数 |
| `--render-mode` | `full` | `full` 插帧输出 / `sampled` 仅采样帧 |
| `--output-fps` | `30` | 输出视频帧率 |
| `--no-display` | — | 关闭实时预览窗口 |
| `--det-device` | — | YOLO 设备，如 `0`（GPU）或 `cpu` |
| `--device` | `auto` | TCN 设备 |
| `--conf` | `0.05` | YOLO 检测置信度阈值 |
| `--draw-conf` | `0.25` | 非马检测框绘制最低置信度 |
| `--min-drinking-water-overlap` | `0.10` | 喝水护栏：head 与水槽重叠阈值（10%） |
| `--smooth-windows` | `3` | TCN 输出窗口多数投票平滑 |
| `--smooth-frames` | `5` | 帧级滑动窗口平滑过滤异常帧 |
| `--draw-water-regions` | `True` | 绘制水槽 ROI（`--no-draw-water-regions` 关闭） |
| `--draw-stall-regions` | `False` | 绘制马厩 ROI（`--draw-stall-regions` 开启） |

## 指定时间段推理

```powershell
# 从第 600 秒到第 720 秒
.\.venv\Scripts\python.exe -m horse_behavior.infer_temporal_behavior `
  --source video\stable_20260522_105023.mp4 `
  --tcn-model runs\timesequence\tcn_behavior_8fps_4s\best.pt `
  --dataset-dir dataset\timesequence_8fps_4s `
  --sample-fps 8 `
  --start-sec 600 `
  --end-sec 720 `
  --render-mode full `
  --output-fps 30 `
  --no-display
```

## 推理流程

```
视频 → 按 sample-fps 采帧 → 首帧冻结（马厩分割 + 水槽自动标定）
  → 每采样帧: YOLO检测 → 去重 → 提取特征 → 推入32帧窗口
  → 窗口满: TCN推理 → 窗口平滑 → 喝水护栏 → 帧平滑
  → 画框写标签 → 输出 MP4 + CSV
```

## 护栏机制

- **喝水护栏**：TCN 预测 `drinking` 时，要求 head 检测框与自动标定的水槽区域重叠 ≥ 10%，否则回退到上次非喝水行为
- **水槽自动标定**：未指定 `--water-regions` 时，首帧用 YOLO Detect 自动检测水槽位置并冻结
- **人员入侵检测**：检测到 person 框与 stall 马厩多边形重叠时，画面四周显示红色边框 + "INTRUSION: Person in Stall" 告警

## 输出 CSV 字段

| 字段 | 说明 |
| --- | --- |
| `frame` | 帧序号 |
| `time_sec` | 时间戳（秒） |
| `behavior` | 最终行为标签 |
| `confidence` | 置信度 |
| `raw_behavior` | TCN 原始预测 |
| `raw_confidence` | TCN 原始置信度 |
| `water_guard_overlap` | head 与水槽重叠比例 |
| `guarded_from_drinking` | 该帧是否被喝水护栏拦截 |
| `detections` | 当前帧检测目标摘要 |