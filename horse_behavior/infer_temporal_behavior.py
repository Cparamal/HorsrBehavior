import argparse
import csv
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from horse_behavior.behavior_features import BehaviorFeatureHistory, extract_behavior_features  # noqa: E402
from horse_behavior.infer_behavior import (  # noqa: E402
    Detection,
    box_area,
    detections_from_result,
    draw_detection_box,
    draw_label,
    load_feed_regions,
    load_regions,
    resize_for_display,
    select_largest_box,
)
from horse_behavior.train_temporal_tcn import TCNBehaviorClassifier  # noqa: E402
from horse_behavior.train_yolo import ensure_ultralytics_config_dir  # noqa: E402


DEFAULT_SOURCE = "video/stable_20260523_105109.mp4"
DEFAULT_OUTPUT = "outputs/tcn_behavior_infer.mp4"
DEFAULT_CSV = "outputs/tcn_behavior_infer.csv"
DEFAULT_DETECTOR_MODEL = "runs/multiframes/horse_multiframe_detect/weights/best.pt"
DEFAULT_SEGMENT_MODEL = "runs/multiframes/horse_multiframe_segment/weights/best.pt"
DEFAULT_TCN_MODEL = "runs/timesequence/tcn_behavior/best.pt"
DEFAULT_DATASET_DIR = "dataset/timesequence"
BOX_COLORS = {
    "horse": (30, 180, 80),
    "head": (80, 160, 230),
    "water": (230, 160, 40),
    "feces": (130, 90, 210),
    "person": (40, 210, 210),
    "grass": (40, 200, 40),
    "lying_horse": (30, 180, 80),
    "stall": (255, 210, 60),
    "water_region": (255, 180, 40),
}
BEHAVIOR_DISPLAY_NAMES = {
    "standing": "standing",
    "eating": "eating",
    "drinking": "drinking",
    "lying": "lying",
}


@dataclass(frozen=True)
class WindowPrediction:
    behavior: str
    confidence: float
    raw_behavior: str
    raw_confidence: float


@dataclass(frozen=True)
class InferenceState:
    prediction: WindowPrediction | None
    horse: Detection | None
    detections: list[Detection]
    water_guard_overlap: float
    guarded_from_drinking: bool
    person_intrusion: bool


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_tcn_checkpoint(path: Path, device: torch.device) -> tuple[TCNBehaviorClassifier, list[str], list[str], int]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    label_names = list(checkpoint["label_names"])
    feature_names = list(checkpoint["feature_names"])
    input_shape = tuple(int(value) for value in checkpoint.get("input_shape", (0, len(feature_names))))
    model_config = dict(checkpoint.get("model_config", {}))
    model = TCNBehaviorClassifier(
        feature_dim=len(feature_names),
        num_classes=len(label_names),
        hidden_channels=int(model_config.get("hidden_channels", 64)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        dilations=tuple(int(value) for value in model_config.get("dilations", [1, 2, 4, 8])),
        dropout=float(model_config.get("dropout", 0.20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    time_steps = input_shape[0] if input_shape and input_shape[0] > 0 else 16
    return model, label_names, feature_names, time_steps


def load_normalization(dataset_dir: Path, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(dataset_dir / "normalization.npz")
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    if mean.shape[0] != feature_count or std.shape[0] != feature_count:
        raise RuntimeError(
            f"Normalization feature count mismatch: expected {feature_count}, "
            f"got mean={mean.shape[0]} std={std.shape[0]}"
        )
    std = std.copy()
    std[std < 1e-6] = 1.0
    return mean, std


def box_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    intersection = box_intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def smaller_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    smaller_area = min(box_area(a), box_area(b))
    if smaller_area <= 0:
        return 0.0
    return box_intersection_area(a, b) / smaller_area


def box_overlap_ratio(
    box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    area = box_area(box)
    if area <= 0:
        return 0.0
    return box_intersection_area(box, region) / area


def max_box_overlap_ratio(
    box: tuple[float, float, float, float] | None,
    regions: list[tuple[float, float, float, float]],
) -> float:
    if box is None or not regions:
        return 0.0
    return max(box_overlap_ratio(box, region) for region in regions)



def is_box_in_polygon(box: tuple[float, float, float, float] | None, polygons: list[np.ndarray]) -> bool:
    if box is None or not polygons:
        return False
    for polygon in polygons:
        if polygon.size < 6:
            continue
        min_x = float(np.min(polygon[:, 0]))
        min_y = float(np.min(polygon[:, 1]))
        max_x = float(np.max(polygon[:, 0]))
        max_y = float(np.max(polygon[:, 1]))
        if box_overlap_ratio(box, (min_x, min_y, max_x, max_y)) > 0:
            return True
    return False

def select_best_head(detections: list[Detection], horse: Detection | None) -> Detection | None:
    heads = [d for d in detections if d.name == "head"]
    if not heads:
        return None
    if horse is None:
        return max(heads, key=lambda d: (d.conf, box_area(d.xyxy)))

    horse_x1, horse_y1, horse_x2, horse_y2 = horse.xyxy
    horse_width = max(1.0, horse_x2 - horse_x1)
    horse_height = max(1.0, horse_y2 - horse_y1)
    expanded = (
        horse_x1 - horse_width * 0.25,
        horse_y1 - horse_height * 0.20,
        horse_x2 + horse_width * 0.10,
        horse_y2 + horse_height * 0.10,
    )
    candidates = [
        head
        for head in heads
        if box_intersection_area(head.xyxy, horse.xyxy) > 0 or box_intersection_area(head.xyxy, expanded) > 0
    ]
    return max(candidates or heads, key=lambda d: (d.conf, box_area(d.xyxy)))


def detection_overlap(a: Detection, b: Detection, metric: str) -> float:
    if metric == "iou":
        return box_iou(a.xyxy, b.xyxy)
    return smaller_overlap_ratio(a.xyxy, b.xyxy)


def dedupe_detections(
    detections: list[Detection],
    threshold: float,
    metric: str,
    class_agnostic: bool,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: (d.conf, box_area(d.xyxy)), reverse=True):
        duplicate = False
        for kept_detection in kept:
            if not class_agnostic and detection.name != kept_detection.name:
                continue
            if detection_overlap(detection, kept_detection, metric) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def choose_sample_times(
    source_fps: float,
    total_frames: int,
    sample_fps: float,
    start_sec: float,
    end_sec: float,
    max_seconds: float,
) -> np.ndarray:
    duration_sec = total_frames / source_fps if total_frames > 0 else 0.0
    start = max(0.0, float(start_sec))
    if duration_sec > 0:
        start = min(start, duration_sec)

    if end_sec > 0:
        end = float(end_sec)
    elif duration_sec > 0:
        end = duration_sec
    else:
        end = start

    if max_seconds > 0:
        end = min(end, start + float(max_seconds))
    if duration_sec > 0:
        end = min(end, duration_sec)
    if end <= start:
        return np.empty((0,), dtype=np.float64)
    return np.arange(start, end, 1.0 / sample_fps, dtype=np.float64)


def read_sampled_frame(capture, target_frame: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def read_next_indexed_frame(capture, current_frame_index: int) -> tuple[int, np.ndarray] | None:
    ok, frame = capture.read()
    if not ok:
        return None
    return current_frame_index, frame


def advance_to_next_render_frame(
    capture,
    current_frame_index: int,
    render_step: int,
    end_frame: int,
) -> int:
    next_frame_index = current_frame_index + render_step
    while current_frame_index + 1 < next_frame_index and current_frame_index + 1 <= end_frame:
        if not capture.grab():
            return end_frame + 1
        current_frame_index += 1
    return next_frame_index


def feature_row_to_array(row: dict[str, object], feature_names: list[str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in feature_names], dtype=np.float32)


def predict_behavior_window(
    model: TCNBehaviorClassifier,
    rows: deque[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    label_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    X = np.stack(list(rows), axis=0).astype(np.float32)
    X = ((X - mean) / std).astype(np.float32)
    tensor = torch.from_numpy(X[None, :, :]).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    label_id = int(probabilities.argmax())
    return label_names[label_id], float(probabilities[label_id])


class PredictionSmoother:
    def __init__(self, window_size: int):
        self.history: deque[tuple[str, float]] = deque(maxlen=max(1, int(window_size)))

    def update(self, behavior: str, confidence: float) -> WindowPrediction:
        self.history.append((behavior, confidence))
        counts = Counter(label for label, _ in self.history)
        smoothed = counts.most_common(1)[0][0]
        smoothed_confidences = [conf for label, conf in self.history if label == smoothed]
        smoothed_conf = sum(smoothed_confidences) / max(1, len(smoothed_confidences))
        return WindowPrediction(
            behavior=smoothed,
            confidence=float(smoothed_conf),
            raw_behavior=behavior,
            raw_confidence=float(confidence),
        )


class DrinkingGuard:
    def __init__(self, fallback_behavior: str = "standing"):
        self.last_non_drinking = fallback_behavior

    def apply(self, prediction: WindowPrediction | None, overlap: float, min_overlap: float) -> tuple[WindowPrediction | None, bool]:
        if prediction is None:
            return None, False
        if prediction.behavior != "drinking":
            self.last_non_drinking = prediction.behavior
            return prediction, False
        if overlap >= min_overlap:
            return prediction, False

        guarded = WindowPrediction(
            behavior=self.last_non_drinking,
            confidence=prediction.confidence,
            raw_behavior=prediction.raw_behavior,
            raw_confidence=prediction.raw_confidence,
        )
        return guarded, True


class FrameSmoother:
    def __init__(self, window_size: int, default_behavior: str = "standing"):
        self.window_size = max(1, int(window_size))
        self.default_behavior = default_behavior
        self.history: deque[str] = deque(maxlen=self.window_size)

    def update(self, behavior: str | None) -> str:
        if behavior is None:
            if not self.history:
                return self.default_behavior
            return Counter(self.history).most_common(1)[0][0]
        self.history.append(behavior)
        return Counter(self.history).most_common(1)[0][0]


def detection_summary(detections: list[Detection]) -> str:
    return ";".join(
        f"{d.name}:{d.conf:.3f}@{int(d.xyxy[0])},{int(d.xyxy[1])},{int(d.xyxy[2])},{int(d.xyxy[3])}"
        for d in detections
    )


def draw_boxes_and_behavior(
    frame: np.ndarray,
    detections: list[Detection],
    behavior: str | None,
    confidence: float | None,
    draw_conf: float,
    selected_horse: Detection | None,
    show_behavior_conf: bool,
) -> None:
    for detection in detections:
        if detection.conf < draw_conf:
            continue
        thickness = 3 if detection.name == "horse" else 2
        draw_detection_box(frame, detection, color=BOX_COLORS.get(detection.name, (180, 180, 180)), thickness=thickness)

    if selected_horse is None or behavior is None:
        return

    x1, y1, x2, y2 = [int(round(value)) for value in selected_horse.xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLORS["horse"], 3)
    display_behavior = BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)
    suffix = f" {confidence:.2f}" if show_behavior_conf and confidence is not None else ""
    draw_label(frame, f"{display_behavior}{suffix}", (max(8, x1), max(32, y1 - 8)), color=BOX_COLORS["horse"])


def draw_regions(
    frame: np.ndarray,
    water_regions: list[tuple[float, float, float, float]],
    stall_polygons: list[np.ndarray],
    draw_water: bool,
    draw_stall: bool,
) -> None:
    if draw_stall:
        for polygon in stall_polygons:
            if polygon.size >= 6:
                cv2.polylines(frame, [polygon.astype(np.int32)], isClosed=True, color=BOX_COLORS["stall"], thickness=3)
                x, y = polygon.astype(np.int32).reshape(-1, 2).min(axis=0)
                draw_label(frame, "stall", (int(x), max(32, int(y) - 8)), color=BOX_COLORS["stall"])

    if draw_water:
        for index, (x1, y1, x2, y2) in enumerate(water_regions, start=1):
            cv2.rectangle(
                frame,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                BOX_COLORS["water_region"],
                3,
            )
            draw_label(frame, "water", (int(round(x1)), max(32, int(round(y1)) - 8)), color=BOX_COLORS["water_region"])


def extract_stall_polygons(seg_result, conf_threshold: float) -> list[np.ndarray]:
    if seg_result is None or seg_result.masks is None:
        return []

    names = seg_result.names
    polygons: list[np.ndarray] = []
    boxes = list(seg_result.boxes) if seg_result.boxes is not None else []
    xy_polygons = list(seg_result.masks.xy)
    for index, polygon in enumerate(xy_polygons):
        if index >= len(boxes):
            continue
        box = boxes[index]
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        if conf < conf_threshold or names[class_id] != "stall":
            continue
        if polygon is None or len(polygon) < 3:
            continue
        polygons.append(np.asarray(polygon, dtype=np.float32))
    return polygons


def calibrate_water_regions_from_frame(frame, detector, args):
    if frame is None:
        return []
    kwargs = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.det_device:
        kwargs["device"] = args.det_device
    result = detector.predict(frame, **kwargs)[0]
    water_boxes = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if names[class_id] == "water":
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                water_boxes.append((x1, y1, x2, y2))
    return water_boxes


def infer_stall_polygons(
    segment_model,
    frame: np.ndarray,
    imgsz: int,
    conf: float,
    device: str,
) -> list[np.ndarray]:
    if segment_model is None:
        return []
    kwargs = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device:
        kwargs["device"] = device
    result = segment_model.predict(frame, **kwargs)[0]
    return extract_stall_polygons(result, conf_threshold=conf)


def write_csv_header(writer) -> None:
    writer.writerow(
        [
            "frame",
            "time_sec",
            "behavior",
            "confidence",
            "raw_behavior",
            "raw_confidence",
            "water_guard_overlap",
            "guarded_from_drinking",
            "horse_conf",
            "detections",
        ]
    )


def write_csv_row(
    writer,
    frame_index: int,
    time_sec: float,
    state: InferenceState | None,
) -> None:
    prediction = None if state is None else state.prediction
    horse = None if state is None else state.horse
    detections = [] if state is None else state.detections
    writer.writerow(
        [
            frame_index,
            f"{time_sec:.3f}",
            "" if prediction is None else prediction.behavior,
            "" if prediction is None else f"{prediction.confidence:.4f}",
            "" if prediction is None else prediction.raw_behavior,
            "" if prediction is None else f"{prediction.raw_confidence:.4f}",
            "" if state is None else f"{state.water_guard_overlap:.4f}",
            "" if state is None else int(state.guarded_from_drinking),
            "" if horse is None else f"{horse.conf:.4f}",
            detection_summary(detections),
        ]
    )


def update_inference_state(
    frame: np.ndarray,
    frame_index: int,
    source_name: str,
    detector,
    feature_history: BehaviorFeatureHistory,
    feature_buffer: deque[np.ndarray],
    smoother: PredictionSmoother,
    model: TCNBehaviorClassifier,
    mean: np.ndarray,
    std: np.ndarray,
    label_names: list[str],
    feature_names: list[str],
    tcn_device: torch.device,
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]],
    stall_polygons: list[np.ndarray],
    image_size: tuple[int, int],
    drinking_guard: DrinkingGuard,
    args: argparse.Namespace,
) -> InferenceState:
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
    }
    if args.det_device:
        predict_kwargs["device"] = args.det_device

    result = detector.predict(frame, **predict_kwargs)[0]
    raw_detections = detections_from_result(result, conf_threshold=args.conf)
    detections = dedupe_detections(
        raw_detections,
        threshold=args.dedupe_threshold,
        metric=args.dedupe_metric,
        class_agnostic=args.class_agnostic_dedupe,
    )

    row = extract_behavior_features(
        detections=detections,
        image_size=image_size,
        split="infer",
        image=f"{source_name}:frame_{int(frame_index)}",
        label="",
        feed_regions=feed_regions,
        water_regions=water_regions,
        history=feature_history,
    )
    feature_buffer.append(feature_row_to_array(row, feature_names))

    prediction = None
    if len(feature_buffer) == feature_buffer.maxlen:
        raw_behavior, raw_confidence = predict_behavior_window(
            model=model,
            rows=feature_buffer,
            mean=mean,
            std=std,
            label_names=label_names,
            device=tcn_device,
        )
        prediction = smoother.update(raw_behavior, raw_confidence)
    horse = select_largest_box(detections, "horse")
    head = select_best_head(detections, horse)
    water_guard_overlap = max_box_overlap_ratio(None if head is None else head.xyxy, water_regions)
    prediction, guarded = drinking_guard.apply(

        prediction,
        overlap=water_guard_overlap,
        min_overlap=args.min_drinking_water_overlap,
    )

    person_box = select_largest_box(detections, "person")
    person_intrusion = is_box_in_polygon(None if person_box is None else person_box.xyxy, stall_polygons)

    return InferenceState(
        prediction=prediction,
        horse=horse,
        detections=detections,
        water_guard_overlap=water_guard_overlap,
        guarded_from_drinking=guarded,
        person_intrusion=person_intrusion,
    )


def draw_state(
    frame: np.ndarray,
    state: InferenceState | None,
    water_regions: list[tuple[float, float, float, float]],
    stall_polygons: list[np.ndarray],
    frame_smoother: FrameSmoother,
    args: argparse.Namespace,
) -> None:
    raw_behavior = None if state is None or state.prediction is None else state.prediction.behavior
    smoothed_behavior = frame_smoother.update(raw_behavior)
    draw_regions(
        frame=frame,
        water_regions=water_regions,
        stall_polygons=stall_polygons,
        draw_water=args.draw_water_regions,
        draw_stall=args.draw_stall_regions,
    )
    if state is None:
        return
    draw_boxes_and_behavior(
        frame=frame,
        detections=state.detections,
        behavior=smoothed_behavior,
        confidence=None if state.prediction is None else state.prediction.confidence,
        draw_conf=args.draw_conf,
        selected_horse=state.horse,
        show_behavior_conf=args.show_behavior_conf,
    )

    if state is not None and state.person_intrusion:
        ih, iw = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (iw, ih), (0, 0, 255), 8)
        alert_text = "INTRUSION: Person in Stall"
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, alert_text, ((iw - tw) // 2, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def run(args: argparse.Namespace) -> int:
    if args.sample_fps <= 0:
        raise RuntimeError("--sample-fps must be greater than 0")
    if args.dedupe_threshold < 0 or args.dedupe_threshold > 1:
        raise RuntimeError("--dedupe-threshold must be between 0 and 1")

    ensure_ultralytics_config_dir(PROJECT_ROOT)
    os.environ.setdefault("YOLO_VERBOSE", "False")

    source = resolve_project_path(args.source)
    output = resolve_project_path(args.output)
    csv_path = resolve_project_path(args.csv) if args.csv else None
    detector_path = resolve_project_path(args.det_model)
    segment_path = resolve_project_path(args.segment_model)
    tcn_path = resolve_project_path(args.tcn_model)
    dataset_dir = resolve_project_path(args.dataset_dir)
    feed_regions = load_feed_regions(resolve_project_path(args.feed_regions))
    water_regions = load_regions(resolve_project_path(args.water_regions)) if args.water_regions else []

    auto_calibrated_water_regions = not args.water_regions
    for path, label in (
        (source, "source video"),
        (detector_path, "YOLO detector model"),
        (tcn_path, "TCN model"),
        (dataset_dir / "normalization.npz", "normalization file"),
    ):
        if not path.exists():
            raise RuntimeError(f"Missing {label}: {path}")
    if not segment_path.exists():
        raise RuntimeError(f"Missing YOLO segment model: {segment_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Could not import ultralytics. Install project requirements first.") from exc

    tcn_device = load_device(args.device)
    model, label_names, feature_names, time_steps = load_tcn_checkpoint(tcn_path, tcn_device)
    mean, std = load_normalization(dataset_dir, len(feature_names))
    detector = YOLO(str(detector_path))
    segmenter = YOLO(str(segment_path))

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video size: {source}")

    sample_times = choose_sample_times(
        source_fps=source_fps,
        total_frames=total_frames,
        sample_fps=args.sample_fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_seconds=args.max_seconds,
    )
    if sample_times.size == 0:
        raise RuntimeError("No sampled frames selected. Check --start-sec, --end-sec, and --max-seconds.")
    frame_indices = np.rint(sample_times * source_fps).astype(np.int64)
    if total_frames > 0:
        frame_indices = np.clip(frame_indices, 0, max(0, total_frames - 1))

    stall_frame = read_sampled_frame(capture, int(frame_indices[0]))
    stall_polygons = infer_stall_polygons(
        segment_model=segmenter,
        frame=stall_frame,
        imgsz=args.segment_imgsz,
        conf=args.segment_conf,
        device=args.det_device,
    ) if stall_frame is not None else []
    # Stall polygons always computed (used for behavior rules); drawing is optional
    print(f"Stall polygons: {len(stall_polygons)}")

    if auto_calibrated_water_regions and stall_frame is not None:
        water_regions = calibrate_water_regions_from_frame(stall_frame, detector, args)
        print(f"Auto-calibrated water regions: {len(water_regions)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_fps > 0:
        output_fps = args.output_fps
    elif args.render_mode == "full":
        output_fps = min(source_fps, args.max_render_fps) if args.max_render_fps > 0 else source_fps
    else:
        output_fps = args.sample_fps
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

    feature_history = BehaviorFeatureHistory(window_size=args.feature_history_window)
    feature_buffer: deque[np.ndarray] = deque(maxlen=time_steps)
    smoother = PredictionSmoother(args.smooth_windows)
    drinking_guard = DrinkingGuard(fallback_behavior=args.drinking_fallback)
    frame_smoother = FrameSmoother(window_size=args.smooth_frames, default_behavior="standing")
    sampled_processed = 0
    rendered_frames = 0
    next_progress_report = 50

    try:
        if args.render_mode == "sampled":
            for time_sec, frame_index in zip(sample_times, frame_indices):
                frame = read_sampled_frame(capture, int(frame_index))
                if frame is None:
                    break

                last_state = update_inference_state(
                    frame=frame,
                    frame_index=int(frame_index),
                    source_name=source.name,
                    detector=detector,
                    feature_history=feature_history,
                    feature_buffer=feature_buffer,
                    smoother=smoother,
                    model=model,
                    mean=mean,
                    std=std,
                    label_names=label_names,
                    feature_names=feature_names,
                    tcn_device=tcn_device,
                    feed_regions=feed_regions,
                    water_regions=water_regions,
                    stall_polygons=stall_polygons,
                    image_size=(width, height),
                    drinking_guard=drinking_guard,
                    args=args,
                )
                draw_state(frame, last_state, water_regions, stall_polygons, frame_smoother, args)
                writer.write(frame)
                rendered_frames += 1

                if csv_writer is not None:
                    write_csv_row(
                        csv_writer,
                        int(frame_index),
                        float(time_sec),
                        last_state,
                    )

                if not args.no_display:
                    cv2.imshow("TCN Horse Behavior", resize_for_display(frame, args.display_scale))
                    key = cv2.waitKey(max(1, int(1000 / output_fps))) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break

                sampled_processed += 1
                if sampled_processed >= next_progress_report:
                    print(f"Processed {sampled_processed}/{len(sample_times)} sampled frames")
                    next_progress_report += 50
        else:
            start_frame = int(frame_indices[0])
            end_frame = int(frame_indices[-1])
            if args.end_sec > 0:
                end_frame = int(round(args.end_sec * source_fps))
            elif args.max_seconds > 0:
                end_frame = int(round((args.start_sec + args.max_seconds) * source_fps))
            elif total_frames > 0:
                end_frame = total_frames - 1
            if total_frames > 0:
                end_frame = min(end_frame, total_frames - 1)

            render_step = max(1, int(round(source_fps / output_fps)))
            sample_cursor = 0
            current_frame_index = start_frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while current_frame_index <= end_frame:
                indexed_frame = read_next_indexed_frame(capture, current_frame_index)
                if indexed_frame is None:
                    break
                actual_frame_index, frame = indexed_frame

                while (
                    sample_cursor < len(frame_indices)
                    and int(frame_indices[sample_cursor]) <= actual_frame_index + render_step // 2
                ):
                    last_state = update_inference_state(
                        frame=frame,
                        frame_index=int(frame_indices[sample_cursor]),
                        source_name=source.name,
                        detector=detector,
                        feature_history=feature_history,
                        feature_buffer=feature_buffer,
                        smoother=smoother,
                        model=model,
                        mean=mean,
                        std=std,
                        label_names=label_names,
                        feature_names=feature_names,
                        tcn_device=tcn_device,
                        feed_regions=feed_regions,
                        water_regions=water_regions,
                        stall_polygons=stall_polygons,
                        image_size=(width, height),
                        drinking_guard=drinking_guard,
                        args=args,
                    )
                    if csv_writer is not None:
                        write_csv_row(
                            csv_writer,
                            int(frame_indices[sample_cursor]),
                            float(sample_times[sample_cursor]),
                            last_state,
                        )
                    sample_cursor += 1
                    sampled_processed += 1

                draw_state(frame, last_state, water_regions, stall_polygons, frame_smoother, args)
                writer.write(frame)
                rendered_frames += 1

                if not args.no_display:
                    cv2.imshow("TCN Horse Behavior", resize_for_display(frame, args.display_scale))
                    key = cv2.waitKey(max(1, int(1000 / output_fps))) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break

                if sampled_processed >= next_progress_report:
                    print(f"Processed {sampled_processed}/{len(sample_times)} sampled frames")
                    next_progress_report += 50
                current_frame_index = advance_to_next_render_frame(
                    capture,
                    actual_frame_index,
                    render_step,
                    end_frame,
                )
    finally:
        capture.release()
        writer.release()
        if csv_file is not None:
            csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Output video: {output.resolve()}")
    if csv_path is not None:
        print(f"Frame CSV: {csv_path.resolve()}")
    print(f"Processed sampled frames: {sampled_processed}")
    print(f"Rendered frames: {rendered_frames} at {output_fps:.3f} FPS")
    print(f"TCN labels: {label_names}; window frames: {time_steps}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TCN temporal horse behavior inference on a video.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Input video path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Annotated sampled-output video path.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--det-model", default=DEFAULT_DETECTOR_MODEL, help="YOLO detector weights path.")
    parser.add_argument("--segment-model", default=DEFAULT_SEGMENT_MODEL, help="YOLO segment weights path for stall ROI drawing.")
    parser.add_argument("--tcn-model", default=DEFAULT_TCN_MODEL, help="TCN behavior checkpoint path.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Temporal dataset directory with normalization.npz.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional fixed feed region YAML.")
    parser.add_argument("--water-regions", default="", help="Optional fixed water region YAML. Empty uses first-frame YOLO detect auto-calibration.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling FPS used for TCN features.")
    parser.add_argument(
        "--render-mode",
        choices=["full", "sampled"],
        default="full",
        help="full keeps extra video frames for smoother playback; sampled writes only TCN sampled frames.",
    )
    parser.add_argument("--output-fps", type=float, default=30.0, help="Output video FPS. 0 uses render-mode default.")
    parser.add_argument("--max-render-fps", type=float, default=10.0, help="Default FPS cap for full render mode. 0 keeps source FPS.")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start timestamp in seconds.")
    parser.add_argument("--end-sec", type=float, default=0.0, help="End timestamp in seconds. 0 means video end.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Debug limit from start time. 0 means no limit.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO candidate confidence threshold for features.")
    parser.add_argument("--draw-conf", type=float, default=0.25, help="Minimum confidence for drawing non-horse boxes.")
    parser.add_argument("--segment-imgsz", type=int, default=640, help="YOLO segment image size for stall ROI.")
    parser.add_argument("--segment-conf", type=float, default=0.25, help="YOLO segment confidence threshold for stall ROI.")
    parser.add_argument(
        "--min-drinking-water-overlap",
        type=float,
        default=0.10,
        help="Required head-box area overlap with fixed water ROI before allowing drinking.",
    )
    parser.add_argument("--drinking-fallback", default="standing", help="Fallback behavior when drinking fails the water ROI guard.")
    parser.add_argument("--draw-water-regions", action=argparse.BooleanOptionalAction, default=True, help="Draw fixed/auto-calibrated water ROI.")
    parser.add_argument("--draw-stall-regions", action=argparse.BooleanOptionalAction, default=False, help="Draw stall segmentation ROI.")
    parser.add_argument("--dedupe-threshold", type=float, default=0.10, help="Overlap threshold for keeping one duplicate box.")
    parser.add_argument(
        "--dedupe-metric",
        choices=["smaller-overlap", "iou"],
        default="smaller-overlap",
        help="Overlap metric for duplicate box removal.",
    )
    parser.add_argument(
        "--class-agnostic-dedupe",
        action="store_true",
        help="Also suppress highly overlapping boxes across different classes.",
    )
    parser.add_argument("--feature-history-window", type=int, default=5, help="Recent frame count for existing temporal features.")
    parser.add_argument("--smooth-windows", type=int, default=3, help="Majority smoothing window over TCN window predictions.")
    parser.add_argument("--show-behavior-conf", action=argparse.BooleanOptionalAction, default=False, help="Show behavior confidence score on overlay.")
    parser.add_argument("--smooth-frames", type=int, default=5, help="Sliding-window smooth over frame-level behavior labels to filter outliers.")
    parser.add_argument("--device", default="auto", help="TCN device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--det-device", default="", help="Optional YOLO device, e.g. 0 or cpu. Empty lets Ultralytics choose.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a realtime preview window.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Realtime preview scale. Saved video keeps original size.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"TCN temporal inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
