## 重要文件

| 类型 | 默认路径 |
| --- | --- |
| YOLO 检测模型 | `runs\detect\runs\detect\horse_behavior_yolo\weights\best.pt` |
| ROI 行为分类模型 | `runs\behavior_yolo_roi_cls\horse_behavior_yolo_roi_cls\weights\best.pt` |
| LightGBM 模型 | `runs\behavior_cls\lightgbm_behavior.joblib` |
| LightGBM 标签编码器 | `runs\behavior_cls\label_encoder.joblib` |
| LightGBM 特征列 | `runs\behavior_cls\feature_columns.txt` |
| 吃饭区域配置 | `config\feed_regions.yaml` |
| 喝水区域配置 | `config\water_regions.yaml` |

## 使用 ROI rules 推理脚本

入口脚本：

```powershell
.\.venv\Scripts\python.exe infer_roi_rules.py `
  --det-model runs\detect\runs\detect\horse_behavior_yolo\weights\best.pt `
  --cls-model runs\behavior_yolo_roi_cls\horse_behavior_yolo_roi_cls\weights\best.pt `
  --feed-regions config\feed_regions.yaml `
  --water-regions config\water_regions.yaml `
  --source video\stable_20260522_105023.mp4 `
  --output outputs\roi_rules_check.mp4 `
  --csv outputs\roi_rules_check.csv `
  --start-sec 1260 `
  --end-sec 1380 `
  --max-frames 0 `
  --no-display
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--det-model` | YOLO 检测模型路径。 |
| `--cls-model` | ROI 行为分类模型路径。 |
| `--feed-regions` | 固定吃饭区域配置。 |
| `--water-regions` | 固定喝水区域配置。 |
| `--source` | 输入视频路径。 |
| `--output` | 输出标注视频路径。 |
| `--csv` | 输出逐帧结果 CSV。传空字符串可关闭 CSV。 |
| `--start-sec` | 从视频第几秒开始处理。 |
| `--end-sec` | 处理到视频第几秒结束。 |
| `--max-frames` | 最多处理帧数。`0` 表示处理完整选中区间。 |
| `--smooth-window` | 行为标签多数投票平滑窗口。 |
| `--drinking-smooth-window` | 喝水进入/退出使用的短平滑窗口。 |
| `--debug` | 绘制调试信息。 |
| `--no-display` | 不打开实时预览窗口，服务器或批处理时建议开启。 |

输出 CSV 主要字段：

| 字段 | 说明 |
| --- | --- |
| `behavior` | 平滑后的最终行为。 |
| `raw_behavior` | 融合后但未平滑的行为。 |
| `source` | 当前行为来源，例如 ROI、rule 或 fusion。 |
| `confidence` | 融合置信度。 |
| `roi_behavior` | ROI 分类模型输出。 |
| `rule_behavior` | 规则护栏输出。 |
| `rule_reason` | 规则命中原因。 |
| `detections` | 当前帧检测目标摘要。 |

## 使用 LightGBM 推理脚本

推荐通过统一入口 `infer.py` 调用：

```powershell
.\.venv\Scripts\python.exe infer.py --method lightgbm `
  --model runs\detect\runs\detect\horse_behavior_yolo\weights\best.pt `
  --behavior-model runs\behavior_cls\lightgbm_behavior.joblib `
  --label-encoder runs\behavior_cls\label_encoder.joblib `
  --feature-columns runs\behavior_cls\feature_columns.txt `
  --feed-regions config\feed_regions.yaml `
  --water-regions config\water_regions.yaml `
  --feature-history-window 5 `
  --smooth-window 10 `
  --event-min-frames 8 `
  --source video\stable_20260522_105023.mp4 `
  --output outputs\lightgbm_check.mp4 `
  --csv outputs\lightgbm_check.csv `
  --start-sec 1260 `
  --end-sec 1380 `
  --max-frames 0 `
  --no-display
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--model` | YOLO 检测模型路径。 |
| `--behavior-model` | LightGBM 行为分类模型。 |
| `--label-encoder` | LightGBM 标签编码器。 |
| `--feature-columns` | 训练时保存的特征列顺序。 |
| `--feed-regions` | 固定吃饭区域配置，用于特征和校正。 |
| `--water-regions` | 固定喝水区域配置，用于特征和校正。 |
| `--feature-history-window` | 特征提取时使用的历史帧窗口。 |
| `--smooth-window` | LightGBM 概率平滑窗口。 |
| `--event-min-frames` | 切换最终事件前要求连续确认的帧数。 |
| `--source` | 输入视频路径。 |
| `--output` | 输出标注视频路径。 |
| `--csv` | 输出逐帧结果 CSV。 |
| `--start-sec` | 从视频第几秒开始处理。 |
| `--end-sec` | 处理到视频第几秒结束。 |
| `--max-frames` | 最多处理帧数。`0` 表示处理完整选中区间。 |
| `--debug` | 绘制所有检测框和模型置信度。 |
| `--no-display` | 不打开实时预览窗口。 |

LightGBM CSV 会同时保留原始模型结果、规则校正结果和最终平滑结果：

| 字段 | 说明 |
| --- | --- |
| `behavior` | 最终输出行为。 |
| `confidence` | 最终置信度。 |
| `calibrated_behavior` | ROI/规则校正后的行为。 |
| `calibrated_confidence` | 校正后的置信度。 |
| `raw_behavior` | LightGBM 原始预测行为。 |
| `raw_confidence` | LightGBM 原始预测置信度。 |
| `calibration_reason` | 校正原因，例如 `fixed_feed_region_contact`。 |
| `probabilities` | 各类别概率。 |
| `detections` | 当前帧检测目标摘要。 |

## 区间处理和输出

两个脚本都支持按时间区间截取视频：

```powershell
--start-sec 1550 --end-sec 1710
```

如果只想限制帧数，也可以使用：

```powershell
--max-frames 1800
```

输出视频和 CSV 会自动创建父目录，例如 `outputs\xxx.mp4` 和 `outputs\xxx.csv`。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| `Missing YOLO model` 或 `Missing behavior model` | 检查模型路径是否存在。 |
| 窗口无法打开或远程运行报错 | 加上 `--no-display`。 |
| 输出行为跳变多 | 增大 `--smooth-window` 或 `--event-min-frames`。 |
| 吃饭/喝水位置判断不对 | 检查 `config\feed_regions.yaml` 和 `config\water_regions.yaml`。 |
| 路径里有空格 | 使用英文双引号包住路径。 |
