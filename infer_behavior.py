import argparse
import csv
import math
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BEHAVIOR_PRIORITY = ["躺卧", "坐下", "吃饭", "低头喝", "低头", "站立", "未知"]
DEFAULT_MODEL = "runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


@dataclass(frozen=True)
class Detection:
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameDecision:
    raw_behavior: str
    behavior: str
    horse: Detection | None
    detections: list[Detection]


class BehaviorSmoother:
    def __init__(self, window_size: int, threshold: float = 0.6):
        self.window_size = max(1, int(window_size))
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.history: deque[str] = deque(maxlen=self.window_size)

    def update(self, behavior: str) -> str:
        self.history.append(behavior)
        counts = Counter(self.history)
        total = len(self.history)

        for candidate in BEHAVIOR_PRIORITY:
            if counts.get(candidate, 0) / total >= self.threshold:
                return candidate
        return behavior


def ensure_ultralytics_config_dir(project_root: Path) -> Path:
    config_dir = project_root / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(a: Detection, b: Detection) -> float:
    ax, ay = box_center(a.xyxy)
    bx, by = box_center(b.xyxy)
    return math.hypot(ax - bx, ay - by)


def boxes_overlap(a: Detection, b: Detection) -> bool:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def point_in_regions(point: tuple[float, float], regions: Iterable[tuple[float, float, float, float]]) -> bool:
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)


def box_overlaps_regions(box: tuple[float, float, float, float], regions: Iterable[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = box
    for rx1, ry1, rx2, ry2 in regions:
        if max(x1, rx1) < min(x2, rx2) and max(y1, ry1) < min(y2, ry2):
            return True
    return False


def box_contains_point(box: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def needs_pil_text(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def load_display_font(size: int) -> ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def select_largest_box(detections: list[Detection], class_name: str | None = None) -> Detection | None:
    candidates = [d for d in detections if class_name is None or d.name == class_name]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (box_area(d.xyxy), d.conf))


def select_best_head(detections: list[Detection], horse: Detection | None) -> Detection | None:
    heads = [d for d in detections if d.name == "head"]
    if not heads:
        return None
    if horse is None:
        return max(heads, key=lambda d: d.conf)

    horse_x1, horse_y1, horse_x2, horse_y2 = horse.xyxy
    horse_width = max(1.0, horse_x2 - horse_x1)
    horse_height = max(1.0, horse_y2 - horse_y1)
    expanded_horse = (
        horse_x1 - horse_width * 0.25,
        horse_y1 - horse_height * 0.20,
        horse_x2 + horse_width * 0.10,
        horse_y2 + horse_height * 0.10,
    )
    candidates = [d for d in heads if boxes_overlap(d, horse) or box_contains_point(expanded_horse, box_center(d.xyxy))]
    if candidates:
        return max(candidates, key=lambda d: d.conf)
    return max(heads, key=lambda d: d.conf)


def head_near_horse_front_edge(
    head: Detection,
    horse: Detection,
    front_margin_ratio: float = 0.20,
    top_margin_ratio: float = 0.25,
) -> bool:
    horse_x1, horse_y1, horse_x2, horse_y2 = horse.xyxy
    horse_width = max(1.0, horse_x2 - horse_x1)
    horse_height = max(1.0, horse_y2 - horse_y1)
    head_x, head_y = box_center(head.xyxy)

    near_left_front = head_x <= horse_x1 + horse_width * front_margin_ratio
    near_top_front = head_y <= horse_y1 + horse_height * top_margin_ratio
    return near_left_front or near_top_front


def head_near_regions(
    head: Detection,
    regions: Iterable[tuple[float, float, float, float]],
    distance_threshold: float,
) -> bool:
    if not regions:
        return False
    if point_in_regions(box_center(head.xyxy), regions):
        return True
    if box_overlaps_regions(head.xyxy, regions):
        return True
    hx, hy = box_center(head.xyxy)
    for x1, y1, x2, y2 in regions:
        nearest_x = min(max(hx, x1), x2)
        nearest_y = min(max(hy, y1), y2)
        if math.hypot(hx - nearest_x, hy - nearest_y) <= distance_threshold:
            return True
    return False


def head_touching_water(
    head: Detection,
    water: Detection,
    min_head_overlap_ratio: float = 0.08,
    min_water_overlap_ratio: float = 0.08,
) -> bool:
    overlap = intersection_area(head.xyxy, water.xyxy)
    if overlap <= 0:
        return False
    head_ratio = overlap / max(1.0, box_area(head.xyxy))
    water_ratio = overlap / max(1.0, box_area(water.xyxy))
    return head_ratio >= min_head_overlap_ratio or water_ratio >= min_water_overlap_ratio


def classify_behavior(
    detections: list[Detection],
    image_size: tuple[int, int],
    feed_regions: list[tuple[float, float, float, float]] | None = None,
    eating_threshold_inside: float = 0.15,
    eating_threshold_outside: float = 0.12,
    drinking_threshold: float = 0.12,
    head_down_ratio: float = 0.58,
    min_grass_conf: float = 0.18,
    min_feed_region_grass_conf: float = 0.10,
    min_overlap_grass_conf: float = 0.05,
    min_water_conf: float = 0.45,
    min_pose_conf: float = 0.35,
    front_head_margin_ratio: float = 0.20,
    top_head_margin_ratio: float = 0.25,
) -> str:
    feed_regions = feed_regions or []
    image_width, image_height = image_size
    scale = max(image_width, image_height)

    horse = select_largest_box(detections, "horse")
    head = select_best_head(detections, horse)
    grass_boxes = [d for d in detections if d.name == "grass" and d.conf >= min(min_grass_conf, min_feed_region_grass_conf, min_overlap_grass_conf)]
    water_boxes = [d for d in detections if d.name == "water" and d.conf >= min_water_conf]
    lying = select_largest_box(detections, "lying_horse")
    sitting = select_largest_box(detections, "sitting_horse")

    if lying is not None and lying.conf >= min_pose_conf:
        return "躺卧"
    if sitting is not None and sitting.conf >= min_pose_conf:
        return "坐下"

    if head is not None:
        for grass in grass_boxes:
            grass_in_feed_region = point_in_regions(box_center(grass.xyxy), feed_regions)
            distance_conf = min_feed_region_grass_conf if grass_in_feed_region else min_grass_conf
            threshold_ratio = eating_threshold_inside if grass_in_feed_region else eating_threshold_outside
            if grass.conf >= distance_conf and center_distance(head, grass) <= threshold_ratio * scale:
                return "吃饭"
            if grass.conf >= min_overlap_grass_conf and boxes_overlap(head, grass):
                return "吃饭"

        for water in water_boxes:
            if head_touching_water(head, water):
                return "低头喝"

        if horse is not None:
            _, head_y = box_center(head.xyxy)
            _, horse_y1, _, horse_y2 = horse.xyxy
            if head_y >= horse_y1 + (horse_y2 - horse_y1) * head_down_ratio:
                return "低头"
            if head_near_horse_front_edge(head, horse, front_head_margin_ratio, top_head_margin_ratio) and box_contains_point(
                horse.xyxy,
                box_center(head.xyxy),
            ):
                return "低头"

    if horse is not None:
        return "站立"
    return "未知"


def detections_from_result(result, conf_threshold: float) -> list[Detection]:
    names = result.names
    detections = []
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        class_id = int(box.cls[0])
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        detections.append(Detection(name=names[class_id], conf=conf, xyxy=xyxy))
    return detections


def effective_model_conf(args) -> float:
    values = [
        float(args.model_conf),
        float(args.conf),
        float(args.min_grass_conf),
        float(args.min_feed_region_grass_conf),
        float(args.min_overlap_grass_conf),
    ]
    return max(0.001, min(values))


def load_regions(path: Path | None) -> list[tuple[float, float, float, float]]:
    if path is None or not path.exists():
        return []

    raw_text = path.read_text(encoding="utf-8")
    compact = " ".join(
        line.split("#", 1)[0].strip()
        for line in raw_text.splitlines()
        if line.split("#", 1)[0].strip()
    )

    if "regions:" in compact and "[" in compact and "]" in compact:
        inside = compact.split("regions:", 1)[1]
        inside = inside[inside.find("[") + 1 : inside.rfind("]")]
        parts = [part.strip() for part in inside.split(",") if part.strip()]
        if len(parts) == 4:
            try:
                return [tuple(float(part) for part in parts)]
            except ValueError:
                pass

    regions = []
    in_regions = False
    current = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "regions:":
            in_regions = True
            continue
        if not in_regions:
            continue
        if stripped.startswith("-"):
            if {"x1", "y1", "x2", "y2"} <= current.keys():
                regions.append((current["x1"], current["y1"], current["x2"], current["y2"]))
            current = {}
            stripped = stripped[1:].strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"x1", "y1", "x2", "y2"}:
                current[key] = float(value)
    if {"x1", "y1", "x2", "y2"} <= current.keys():
        regions.append((current["x1"], current["y1"], current["x2"], current["y2"]))
    return regions


def load_feed_regions(path: Path | None) -> list[tuple[float, float, float, float]]:
    return load_regions(path)


def draw_label(frame, text: str, origin: tuple[int, int], color=(30, 180, 80)) -> None:
    x, y = origin
    if needs_pil_text(text):
        font = load_display_font(28)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        top = max(0, y - height - 16)
        left = max(0, x)
        draw.rectangle((left, top, left + width + 16, top + height + 16), fill=tuple(int(c) for c in color[::-1]))
        draw.text((left + 8, top + 7), text, font=font, fill=(0, 0, 0))
        frame[:, :] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - height - 12), (x + width + 12, y + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 6, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_decision(frame, decision: FrameDecision) -> None:
    horse = decision.horse
    if horse is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in horse.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 80), 3)
        label_y = max(32, y1 - 8)
        draw_label(frame, f"行为：{decision.behavior}", (max(8, x1), label_y))
    else:
        draw_label(frame, f"行为：{decision.behavior}", (20, 40), color=(80, 160, 230))


def resize_for_display(frame, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return frame
    height, width = frame.shape[:2]
    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    return cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)


def decide_frame(
    result,
    image_size: tuple[int, int],
    conf_threshold: float,
    smoother: BehaviorSmoother,
    feed_regions: list[tuple[float, float, float, float]],
    args,
) -> FrameDecision:
    detections = detections_from_result(result, conf_threshold)
    raw_behavior = classify_behavior(
        detections,
        image_size=image_size,
        feed_regions=feed_regions,
        eating_threshold_inside=args.eating_threshold_inside,
        eating_threshold_outside=args.eating_threshold_outside,
        drinking_threshold=args.drinking_threshold,
        head_down_ratio=args.head_down_ratio,
        min_grass_conf=args.min_grass_conf,
        min_feed_region_grass_conf=args.min_feed_region_grass_conf,
        min_overlap_grass_conf=args.min_overlap_grass_conf,
        min_water_conf=args.min_water_conf,
        min_pose_conf=args.min_pose_conf,
        front_head_margin_ratio=args.front_head_margin_ratio,
        top_head_margin_ratio=args.top_head_margin_ratio,
    )
    behavior = smoother.update(raw_behavior)
    horse = select_largest_box(detections, "horse")
    return FrameDecision(raw_behavior=raw_behavior, behavior=behavior, horse=horse, detections=detections)


def write_csv_header(writer) -> None:
    writer.writerow(["frame", "time_sec", "raw_behavior", "behavior", "horse_conf", "detections"])


def write_csv_row(writer, frame_index: int, fps: float, decision: FrameDecision) -> None:
    horse_conf = "" if decision.horse is None else f"{decision.horse.conf:.4f}"
    det_summary = ";".join(f"{d.name}:{d.conf:.3f}" for d in decision.detections)
    writer.writerow([frame_index, f"{frame_index / fps:.3f}" if fps else "", decision.raw_behavior, decision.behavior, horse_conf, det_summary])


def run_video(
    args,
    model,
    feed_regions: list[tuple[float, float, float, float]],
) -> int:
    source = Path(args.source)
    output = Path(args.output)
    if args.save_output:
        output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = args.max_frames if args.max_frames and args.max_frames > 0 else total_frames
    limit = min(limit, total_frames) if total_frames > 0 else limit

    writer = None
    if args.save_output:
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output}")

    smoother = BehaviorSmoother(window_size=max(1, int(fps * args.smooth_seconds)), threshold=args.smooth_threshold)
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

    frame_index = 0
    try:
        while True:
            if limit and frame_index >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
            decision = decide_frame(result, (width, height), effective_model_conf(args), smoother, feed_regions, args)
            draw_decision(frame, decision)
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Horse Behavior", resize_for_display(frame, args.display_scale))
                delay = max(1, int(1000 / fps))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, decision)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{limit or '?'} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

    if args.save_output:
        print(f"Output video: {output.resolve()}")
    if args.csv:
        print(f"Frame CSV: {Path(args.csv).resolve()}")
    return 0


def run_images(
    args,
    model,
    feed_regions: list[tuple[float, float, float, float]],
) -> int:
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    smoother = BehaviorSmoother(window_size=1, threshold=1.0)

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        height, width = frame.shape[:2]
        result = model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
        decision = decide_frame(result, (width, height), effective_model_conf(args), smoother, feed_regions, args)
        draw_decision(frame, decision)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"{image_path.name}: {decision.behavior}")
    print(f"Output images: {output_dir.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO inference and horse behavior rules.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video, image, or image directory.")
    parser.add_argument("--output", default="outputs/behavior_demo.mp4", help="Output video path or image output directory.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level YOLO candidate threshold before behavior rules filter classes.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--smooth-seconds", type=float, default=2.0, help="Temporal smoothing window for video.")
    parser.add_argument("--smooth-threshold", type=float, default=0.6, help="Majority threshold for smoothing.")
    parser.add_argument("--eating-threshold-inside", type=float, default=0.15, help="Head-to-grass distance ratio inside feed regions.")
    parser.add_argument("--eating-threshold-outside", type=float, default=0.12, help="Head-to-grass distance ratio outside feed regions.")
    parser.add_argument("--drinking-threshold", type=float, default=0.12, help="Head-to-water distance ratio.")
    parser.add_argument("--head-down-ratio", type=float, default=0.58, help="Head center must be below this fraction of horse height.")
    parser.add_argument("--front-head-margin-ratio", type=float, default=0.20, help="Head near the horse front/left edge is treated as head-down for overhead cameras.")
    parser.add_argument("--top-head-margin-ratio", type=float, default=0.25, help="Head near the horse top edge is treated as head-down for overhead cameras.")
    parser.add_argument("--min-grass-conf", type=float, default=0.18, help="Minimum grass confidence for distance-based eating rules.")
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10, help="Minimum grass confidence for distance-based eating rules inside fixed feed regions.")
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05, help="Minimum grass confidence when grass overlaps the selected head.")
    parser.add_argument("--min-water-conf", type=float, default=0.45, help="Minimum water confidence for drinking rules.")
    parser.add_argument("--min-pose-conf", type=float, default=0.35, help="Minimum lying/sitting confidence for pose rules.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit video frames for quick tests. 0 means full video.")
    parser.add_argument("--csv", default="outputs/behavior_frames.csv", help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--mode", choices=["auto", "video", "images"], default="auto", help="Input mode.")
    parser.add_argument("--save-output", action="store_true", help="Save annotated video while displaying realtime playback.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a realtime preview window.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Realtime preview scale. Saved video keeps original size.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    ensure_ultralytics_config_dir(project_root)

    source = Path(args.source)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Missing model weights: {model_path}", file=sys.stderr)
        return 2
    if not source.exists():
        print(f"Missing source: {source}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1

    feed_regions = load_feed_regions(Path(args.feed_regions))
    if feed_regions:
        print(f"Loaded {len(feed_regions)} feed region(s) from {args.feed_regions}")
    else:
        print("No feed regions loaded; grass outside fixed regions uses stricter distance threshold.")

    model = YOLO(str(model_path))
    mode = args.mode
    if mode == "auto":
        mode = "images" if source.is_dir() or source.suffix.lower() in {".jpg", ".jpeg", ".png"} else "video"

    if mode == "images":
        return run_images(args, model, feed_regions)
    return run_video(args, model, feed_regions)


if __name__ == "__main__":
    raise SystemExit(main())
