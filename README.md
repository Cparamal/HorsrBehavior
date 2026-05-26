# HorseBehavior

_基于YOLO模型的马匹视频行为识别开发进度展示文档。_

---

## 目标行为

计划识别 6 类行为：

| 行为 | 含义 | 主要判断依据 |
| --- | --- | --- |
| `吃饭` | 马接触草料 | `head` 与 `grass` 距离小于阈值，连续约 2 秒；`grass` 可位于固定 `feed_region` 内或画面中任意可识别草料位置 |
| `低头喝` | 马低头并靠近水桶、水槽或喝水区域 | `head` 连续一段时间靠近 `water` |
| `站立` | 马正常站立 | 检测到 `horse`，且没有触发更高优先级行为 |
| `躺卧` | 马躺在地上 | 检测到 `lying_horse`，或全身框形态符合躺卧 |
| `坐下` | 马呈坐姿，后腿弯曲、前腿直立或身体后倾 | 检测到 `sitting_horse`，或全身框形态符合坐姿 |
| `低头` | 马低头，但不是吃饭或喝水 | `head` 在 `horse` 框内位置较低，且不满足吃饭/喝水规则 |


## 当前进度

| 模块   | 状态 | 说明                |
|------| --- |-------------------|
| 项目目标定义 | 已完成 | 明确输入、输出、最终画面展示规则  |
| 行为类别设计 | 已完成 | 已确定 6 类目标行为       |
| 数据抽帧 | 未开始 | 待确认输入视频路径并抽取代表帧   |
| 数据标注 | 未开始 | 计划先标注 150-250 张图片 |
| 模型训练 | 未开始 | 待数据集导出后训练模型       |
| 行为规则脚本 | 未开始 | 待模型输出格式稳定后开发      |
| 视频导出 | 未开始 | 待推理和绘制逻辑完成后生成     |

## 开发计划

开发计划按照“当前进度”中的模块顺序推进，先完成数据准备，再进入模型训练、行为规则开发和最终视频导出。

### 1. 项目目标定义

状态：已完成

已明确项目输入、输出和最终展示方式：

- 输入一段马匹视频
- 输出一段带行为识别结果的视频

### 2. 行为类别设计

状态：已完成

当前计划识别 6 类行为：

- `吃饭`
- `低头喝`
- `站立`
- `躺卧`
- `坐下`
- `低头`

后续模型和规则开发都以这 6 类行为为输出目标。模型本身不直接训练这些行为类别，而是先检测马匹、马头、草料、水源、躺卧和坐姿状态，再结合固定候选投喂区域判断最终行为。

### 3. 数据抽帧

状态：已完成

计划步骤：

1. 确认输入视频路径和可用视频片段
2. 记录视频分辨率、帧率和时长
3. 使用 FFmpeg 按固定间隔抽帧
4. 人工筛选清晰、有代表性的帧
5. 保留覆盖站立、低头、吃饭、喝水、躺卧等情况的图片


```bash
ffmpeg -i input.mp4 -vf fps=1 frames/frame_%05d.jpg
```


### 4. 数据标注

状态：已完成

计划步骤：

1. 创建标注类别：`horse/head/grass/water/lying_horse/sitting_horse`
2. 标注第一批图片
3. 检查 `horse` 全身框是否完整
4. 检查 `head` 是否稳定覆盖马头和嘴鼻区域
5. 检查 `grass` 是否只覆盖可见草料主体
6. 配置固定候选投喂区域 `feed_regions`
7. 检查 `water` 是否能代表喝水位置

阶段输出：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
config/
  feed_regions.yaml
```


### 5. 模型训练

状态：已完成

计划步骤：

1. 编写 `data.yaml`
2. 使用轻量模型训练第一版检测模型
3. 在验证集上查看检测效果
4. 记录误检、漏检和框不准确的样本
5. 根据错误案例补充或修正标注
6. 重新训练第二版模型


### 6. 行为规则脚本

状态：已完成

计划步骤：

1. 读取模型每帧检测结果
2. 分离 `horse/head/grass/water/lying_horse/sitting_horse` 检测框
3. 选择最合适的 `horse` 全身框用于最终展示
4. `grass` 位于固定 `feed_regions` 内直接参与判断；区域外的 `grass` 也参与，可设更严格的距离阈值
5. 使用 `head` 或嘴鼻区域与 `grass` 的距离判断吃饭
6. 使用 `head` 与 `water` 的距离或重叠关系判断喝水
7. 使用 `head` 在 `horse` 框中的位置判断普通低头
8. 使用 `lying_horse` 或全身框形态判断躺卧
9. 使用 `sitting_horse` 或全身框形态判断坐下
10. 增加 2 秒左右的时间平滑，减少行为文字闪烁

行为优先级：

```text
躺卧
坐下
吃饭
低头喝
低头
站立
```

阶段输出：

- 行为分类函数
- 检测框筛选逻辑
- 时间平滑逻辑
- 每帧行为结果

### 7. 视频导出

状态：已完成

计划步骤：

1. 对完整输入视频运行模型推理
2. 将每帧检测结果交给行为规则脚本
3. 在画面中添加当前行为文字
4. 导出可播放的 demo 视频

阶段输出：

- 原始推理结果
- 带行为识别结果的视频
- `output_demo.mp4`

完成标准：

- 视频可以从头到尾正常播放
- 画面中出现马匹全身框
- 行为文字清晰可读

## 推理脚本使用方法

统一推理入口是 `infer.py`，通过 `--method` 选择行为判断方式：

| 方法 | 命令值 | 说明 |
| --- | --- | --- |
| 规则判断 | `rules` | 使用 YOLO 检测框和人工规则判断行为，适合做可解释 baseline。 |
| LightGBM | `lightgbm` | 使用 YOLO 检测框提取结构化特征，再用 LightGBM 分类。当前推荐作为主对比模型。 |
| ROI YOLO 分类 | `roi-yolo` | 先用 YOLO 检测马体框，裁剪 ROI，再用 YOLO 分类器判断行为。 |

### 通用参数

| 参数 | 说明 |
| --- | --- |
| `--source` | 输入视频路径，默认 `video/stable_20260523_105109.mp4`。 |
| `--output` | 输出标注视频路径。 |
| `--csv` | 输出逐帧结果 CSV 路径；传空字符串可关闭 CSV。 |
| `--max-frames` | 最多处理多少帧；`0` 表示处理完整视频。 |
| `--no-display` | 不打开实时预览窗口，批量生成视频时建议加上。 |
| `--imgsz` | YOLO 检测输入尺寸，默认 `640`。 |

### LightGBM 推理

```powershell
.\.venv\Scripts\python.exe infer.py `
  --method lightgbm `
  --source video/stable_20260523_105109.mp4 `
  --output outputs/behavior_lightgbm_1800.mp4 `
  --csv outputs/behavior_lightgbm_1800.csv `
  --max-frames 1800 `
  --no-display
```

常用可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt` | YOLO 检测器权重。 |
| `--behavior-model` | `runs/behavior_cls/lightgbm_behavior.joblib` | LightGBM 行为分类模型。 |
| `--label-encoder` | `runs/behavior_cls/label_encoder.joblib` | 行为标签编码器。 |
| `--feature-columns` | `runs/behavior_cls/feature_columns.txt` | 训练时保存的特征列顺序。 |
| `--smooth-window` | `15` | 对最近多少帧的分类概率做平均平滑。 |

### ROI YOLO 分类推理

```powershell
.\.venv\Scripts\python.exe infer.py `
  --method roi-yolo `
  --source video/stable_20260523_105109.mp4 `
  --output outputs/behavior_yolo_roi_cls_1800.mp4 `
  --csv outputs/behavior_yolo_roi_cls_1800.csv `
  --max-frames 1800 `
  --no-display
```

常用可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--det-model` | `runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt` | YOLO 检测器权重。 |
| `--cls-model` | `runs/behavior_yolo_roi_cls/horse_behavior_yolo_roi_cls/weights/best.pt` | ROI 行为分类器权重。 |
| `--crop-padding` | `0.15` | 马体检测框外扩比例，裁剪 ROI 时使用。 |
| `--cls-imgsz` | `224` | ROI 分类器输入尺寸。 |

### 规则推理

规则推理默认只预览画面；如果要保存视频，需要额外加 `--save-output`。

```powershell
.\.venv\Scripts\python.exe infer.py `
  --method rules `
  --source video/stable_20260523_105109.mp4 `
  --output outputs/behavior_rules_1800.mp4 `
  --csv outputs/behavior_rules_1800.csv `
  --max-frames 1800 `
  --save-output `
  --no-display
```

常用可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt` | YOLO 检测器权重。 |
| `--feed-regions` | `config/feed_regions.yaml` | 固定投喂区域配置。 |
| `--smooth-seconds` | `2.0` | 规则结果时间平滑窗口。 |
| `--debug` | 关闭 | 绘制更多规则调试辅助线和区域。 |

### 快速测试命令

只跑 1 帧用于确认环境、模型路径和输出路径是否正常：

```powershell
.\.venv\Scripts\python.exe infer.py --method lightgbm --max-frames 1 --no-display --output outputs/smoke_lightgbm.mp4 --csv outputs/smoke_lightgbm.csv
.\.venv\Scripts\python.exe infer.py --method roi-yolo --max-frames 1 --no-display --output outputs/smoke_roi.mp4 --csv outputs/smoke_roi.csv
.\.venv\Scripts\python.exe infer.py --method rules --max-frames 1 --save-output --no-display --output outputs/smoke_rules.mp4 --csv outputs/smoke_rules.csv
```
