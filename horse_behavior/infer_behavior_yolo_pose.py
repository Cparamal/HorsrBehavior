import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horse_behavior.infer_behavior import (
    DEFAULT_MODEL,
    Detection,
    behavior_display_name,
    box_center,
    detections_from_result,
    draw_clean_behavior_box,
    draw_label,
    effective_model_conf,
    head_touching_water,
    load_feed_regions,
    resize_for_display,
)
from horse_behavior.pose_schema import KEYPOINT_INDEX, SUPERANIMAL_QUADRUPED_KEYPOINTS, SUPERANIMAL_QUADRUPED_SKELETON
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_POSE_MODEL = "runs/pose/horse_pose_yolo/weights/best.pt"
DEFAULT_OUTPUT = "outputs/behavior_yolo_pose.mp4"
DEFAULT_CSV = "outputs/behavior_yolo_pose.csv"
BEHAVIOR_PRIORITY = ["lying", "drinking", "eating", "head_down", "standing", "unknown"]
BEHAVIOR_NAMES = {
    "standing": "站立",
    "eating": "吃饭",
    "drinking": "低头喝",
    "head_down": "低头",
    "lying": "躺卧",
    "unknown": "未知",
}
HEAD_KEYPOINTS = ["nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left", "right_eye", "left_eye"]
BODY_KEYPOINTS = ["neck_base", "neck_end", "back_base", "back_middle", "back_end", "belly_bottom"]


@dataclass(frozen=True)
class PoseInstance:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray


@dataclass(frozen=True)
class PoseBehaviorExplanation:
    behavior: str
    reason: str
    pose: PoseInstance | None
    head_point: tuple[float, float] | None
    detections: list[Detection]


@dataclass(frozen=True)
class PoseBehaviorDecision:
    behavior: str
    raw_behavior: str
    reason: str
    pose: PoseInstance | None
    detections: list[Detection]


class LabelSmoother:
    def __init__(self, window_size: int):
        self.history: deque[str] = deque(maxlen=max(1, int(window_size)))

    def update(self, behavior: str) -> str:
        self.history.append(behavior)
        counts: dict[str, int] = {}
        for value in self.history:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.items(), key=lambda item: (item[1], -BEHAVIOR_PRIORITY.index(item[0]) if item[0] in BEHAVIOR_PRIORITY else -99))[0]


def display_behavior(behavior: str) -> str:
    return BEHAVIOR_NAMES.get(behavior, behavior_display_name(behavior))


def pose_instances_from_result(result, min_pose_conf: float) -> list[PoseInstance]:
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(keypoints_xy.shape[:2])
    poses = []
    for box, conf, xy, scores in zip(boxes, confs, keypoints_xy, keypoints_conf):
        if float(conf) < min_pose_conf:
            continue
        keypoints = np.concatenate([xy.astype(np.float32), scores[..., None].astype(np.float32)], axis=1)
        poses.append(PoseInstance(bbox_xyxy=tuple(float(v) for v in box), confidence=float(conf), keypoints=keypoints))
    return poses


def select_main_pose(poses: list[PoseInstance]) -> PoseInstance | None:
    if not poses:
        return None
    return max(poses, key=lambda pose: (pose.confidence, (pose.bbox_xyxy[2] - pose.bbox_xyxy[0]) * (pose.bbox_xyxy[3] - pose.bbox_xyxy[1])))


def visible_points(pose: PoseInstance, names: list[str], threshold: float) -> np.ndarray:
    indexes = [KEYPOINT_INDEX[name] for name in names if name in KEYPOINT_INDEX]
    points = pose.keypoints[indexes]
    return points[points[:, 2] >= threshold]


def mean_point(points: np.ndarray) -> tuple[float, float] | None:
    if len(points) == 0:
        return None
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def point_box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def point_inside_regions(point: tuple[float, float], regions: list[tuple[float, float, float, float]]) -> bool:
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)


def head_low_ratio(head_point: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    _, y = head_point
    _, y1, _, y2 = bbox
    return (y - y1) / max(1.0, y2 - y1)


def back_flatness_ratio(pose: PoseInstance, threshold: float) -> float | None:
    body_points = visible_points(pose, ["neck_end", "back_base", "back_middle", "back_end", "tail_base"], threshold)
    if len(body_points) < 3:
        return None
    y_span = float(np.max(body_points[:, 1]) - np.min(body_points[:, 1]))
    x_span = float(np.max(body_points[:, 0]) - np.min(body_points[:, 0]))
    return y_span / max(1.0, x_span)


def classify_pose_behavior(
    pose: PoseInstance | None,
    detections: list[Detection],
    feed_regions: list[tuple[float, float, float, float]],
    args,
) -> PoseBehaviorExplanation:
    if pose is None:
        return PoseBehaviorExplanation("unknown", "no_pose", None, None, detections)

    head = mean_point(visible_points(pose, HEAD_KEYPOINTS, args.keypoint_threshold))
    if head is None:
        return PoseBehaviorExplanation("standing", "no_head_keypoints", pose, None, detections)

    bbox_width = max(1.0, pose.bbox_xyxy[2] - pose.bbox_xyxy[0])
    bbox_height = max(1.0, pose.bbox_xyxy[3] - pose.bbox_xyxy[1])
    flatness = back_flatness_ratio(pose, args.keypoint_threshold)
    if flatness is not None and flatness <= args.lying_flatness_ratio and bbox_height / bbox_width <= args.lying_aspect_ratio:
        return PoseBehaviorExplanation("lying", "flat_back_low_box", pose, head, detections)

    water_boxes = [d for d in detections if d.name == "water" and d.conf >= args.min_water_conf]
    head_box = Detection(name="head", conf=1.0, xyxy=(head[0] - 8, head[1] - 8, head[0] + 8, head[1] + 8))
    for water in water_boxes:
        if head_touching_water(head_box, water) or point_box_distance(head, water.xyxy) / max(bbox_width, bbox_height) <= args.drinking_distance_ratio:
            return PoseBehaviorExplanation("drinking", "head_near_water", pose, head, detections)

    grass_boxes = [d for d in detections if d.name == "grass" and d.conf >= args.min_grass_conf]
    for grass in grass_boxes:
        distance_ratio = point_box_distance(head, grass.xyxy) / max(bbox_width, bbox_height)
        grass_center = box_center(grass.xyxy)
        threshold = args.eating_distance_inside if point_inside_regions(grass_center, feed_regions) else args.eating_distance_outside
        if distance_ratio <= threshold:
            return PoseBehaviorExplanation("eating", "head_near_grass", pose, head, detections)

    if point_inside_regions(head, feed_regions):
        return PoseBehaviorExplanation("eating", "head_inside_feed_region", pose, head, detections)

    if head_low_ratio(head, pose.bbox_xyxy) >= args.head_down_ratio:
        return PoseBehaviorExplanation("head_down", "head_low_pose", pose, head, detections)

    return PoseBehaviorExplanation("standing", "pose_default", pose, head, detections)


def draw_pose(frame, pose: PoseInstance, threshold: float) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in pose.bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 80), 2)
    for a, b in SUPERANIMAL_QUADRUPED_SKELETON:
        ia = KEYPOINT_INDEX[a]
        ib = KEYPOINT_INDEX[b]
        if pose.keypoints[ia, 2] < threshold or pose.keypoints[ib, 2] < threshold:
            continue
        pa = tuple(int(round(v)) for v in pose.keypoints[ia, :2])
        pb = tuple(int(round(v)) for v in pose.keypoints[ib, :2])
        cv2.line(frame, pa, pb, (90, 180, 230), 2, cv2.LINE_AA)
    for index, (x, y, score) in enumerate(pose.keypoints):
        if score < threshold:
            continue
        color = (
            int(80 + (index * 37) % 175),
            int(80 + (index * 67) % 175),
            int(80 + (index * 97) % 175),
        )
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)


def draw_pose_behavior(frame, decision: PoseBehaviorDecision, explanation: PoseBehaviorExplanation, debug: bool) -> None:
    if not debug:
        horse = None
        if decision.pose is not None:
            horse = Detection(name="horse", conf=decision.pose.confidence, xyxy=decision.pose.bbox_xyxy)
        draw_clean_behavior_box(frame, horse, display_behavior(decision.behavior))
        return

    if decision.pose is not None:
        draw_pose(frame, decision.pose, 0.10)
    for detection in decision.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.xyxy]
        color = (40, 140, 255) if detection.name in {"grass", "water"} else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        draw_label(frame, f"{detection.name} {detection.conf:.2f}", (max(8, x1), max(24, y1 - 4)), color=color)
    if explanation.head_point is not None:
        hx, hy = [int(round(v)) for v in explanation.head_point]
        cv2.circle(frame, (hx, hy), 7, (0, 255, 255), 2, cv2.LINE_AA)
    draw_label(
        frame,
        f"Final:{display_behavior(decision.behavior)} raw:{decision.raw_behavior} reason:{decision.reason}",
        (8, 40),
        color=(40, 210, 210),
    )


def keypoints_to_json(pose: PoseInstance | None) -> str:
    if pose is None:
        return "[]"
    rows = []
    for name, (x, y, score) in zip(SUPERANIMAL_QUADRUPED_KEYPOINTS, pose.keypoints):
        rows.append({"name": name, "x": float(x), "y": float(y), "score": float(score)})
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def write_csv_header(writer) -> None:
    writer.writerow(["frame", "time_sec", "behavior", "raw_behavior", "reason", "pose_confidence", "pose_box", "detections", "keypoints"])


def write_csv_row(writer, frame_index: int, fps: float, decision: PoseBehaviorDecision) -> None:
    pose_conf = "" if decision.pose is None else f"{decision.pose.confidence:.4f}"
    pose_box = "" if decision.pose is None else json.dumps([float(v) for v in decision.pose.bbox_xyxy], separators=(",", ":"))
    detections = ";".join(f"{d.name}:{d.conf:.3f}" for d in decision.detections)
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            decision.behavior,
            decision.raw_behavior,
            decision.reason,
            pose_conf,
            pose_box,
            detections,
            keypoints_to_json(decision.pose),
        ]
    )


def process_frame(frame, pose_model, det_model, feed_regions, smoother: LabelSmoother, args) -> tuple[PoseBehaviorDecision, PoseBehaviorExplanation]:
    pose_result = pose_model.predict(frame, imgsz=args.pose_imgsz, conf=args.pose_conf, verbose=False)[0]
    poses = pose_instances_from_result(pose_result, args.pose_conf)
    pose = select_main_pose(poses)

    detections: list[Detection] = []
    if det_model is not None:
        det_result = det_model.predict(frame, imgsz=args.det_imgsz, conf=effective_model_conf(args), verbose=False)[0]
        detections = detections_from_result(det_result, effective_model_conf(args))

    explanation = classify_pose_behavior(pose, detections, feed_regions, args)
    behavior = smoother.update(explanation.behavior)
    return (
        PoseBehaviorDecision(
            behavior=behavior,
            raw_behavior=explanation.behavior,
            reason=explanation.reason,
            pose=pose,
            detections=detections,
        ),
        explanation,
    )


def run_video(args, pose_model, det_model, feed_regions) -> int:
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = args.max_frames if args.max_frames and args.max_frames > 0 else total_frames
    limit = min(limit, total_frames) if total_frames > 0 and limit else limit
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)
    smoother = LabelSmoother(args.smooth_window)
    frame_index = 0
    try:
        while True:
            if limit and frame_index >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            decision, explanation = process_frame(frame, pose_model, det_model, feed_regions, smoother, args)
            draw_pose_behavior(frame, decision, explanation, args.debug)
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, decision)
            if not args.no_display:
                cv2.imshow("YOLO Pose Behavior", resize_for_display(frame, args.display_scale))
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{limit or '?'} frames")
    finally:
        capture.release()
        writer.release()
        if csv_file:
            csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Output video: {output.resolve()}")
    if args.csv:
        print(f"Frame CSV: {Path(args.csv).resolve()}")
    print(f"Processed frames: {frame_index}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run behavior inference from a YOLO pose model.")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help="YOLO pose weights path.")
    parser.add_argument("--det-model", default=DEFAULT_MODEL, help="Optional YOLO object detector for grass/water context.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output annotated video path.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Optional frame CSV path. Empty disables CSV.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level detector candidate threshold.")
    parser.add_argument("--min-grass-conf", type=float, default=0.18)
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10)
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05)
    parser.add_argument("--min-water-conf", type=float, default=0.45)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    parser.add_argument("--head-down-ratio", type=float, default=0.58)
    parser.add_argument("--eating-distance-inside", type=float, default=0.16)
    parser.add_argument("--eating-distance-outside", type=float, default=0.10)
    parser.add_argument("--drinking-distance-ratio", type=float, default=0.06)
    parser.add_argument("--lying-flatness-ratio", type=float, default=0.10)
    parser.add_argument("--lying-aspect-ratio", type=float, default=0.70)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--no-detector", action="store_true", help="Disable grass/water detector context.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)
    if not Path(args.pose_model).exists():
        print(f"Missing pose model: {args.pose_model}", file=sys.stderr)
        return 2
    if not Path(args.source).exists():
        print(f"Missing source: {args.source}", file=sys.stderr)
        return 2
    if not args.no_detector and not Path(args.det_model).exists():
        print(f"Missing detector model: {args.det_model}", file=sys.stderr)
        return 2
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1
    pose_model = YOLO(str(Path(args.pose_model)))
    det_model = None if args.no_detector else YOLO(str(Path(args.det_model)))
    feed_regions = load_feed_regions(Path(args.feed_regions))
    return run_video(args, pose_model, det_model, feed_regions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"YOLO pose behavior inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
