import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

from horse_behavior.behavior_roi import crop_behavior_roi
from horse_behavior.infer_behavior import (
    DEFAULT_MODEL,
    Detection,
    detections_from_result,
    draw_label,
    effective_model_conf,
    resize_for_display,
    select_largest_box,
)
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_ROI_CLS_MODEL = "runs/behavior_yolo_roi_cls/horse_behavior_yolo_roi_cls/weights/best.pt"


@dataclass(frozen=True)
class YoloRoiClsDecision:
    behavior: str
    confidence: float
    horse: Detection | None
    detections: list[Detection]
    roi_box: tuple[int, int, int, int]
    roi_source: str


def classify_from_yolo_result(result) -> tuple[str, float]:
    top1 = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    return str(result.names[top1]), confidence


def make_yolo_roi_cls_decision(
    frame,
    detections: list[Detection],
    cls_result,
    crop_padding: float = 0.15,
) -> YoloRoiClsDecision:
    behavior, confidence = classify_from_yolo_result(cls_result)
    roi = crop_behavior_roi(frame, detections, padding_ratio=crop_padding)
    return YoloRoiClsDecision(
        behavior=behavior,
        confidence=confidence,
        horse=select_largest_box(detections, "horse") or select_largest_box(detections, "lying_horse"),
        detections=detections,
        roi_box=roi.box,
        roi_source=roi.source,
    )


def classify_roi_frame(
    frame,
    detections: list[Detection],
    cls_model,
    cls_imgsz: int,
    crop_padding: float = 0.15,
) -> YoloRoiClsDecision:
    roi = crop_behavior_roi(frame, detections, padding_ratio=crop_padding)
    cls_result = cls_model.predict(roi.image, imgsz=cls_imgsz, verbose=False)[0]
    behavior, confidence = classify_from_yolo_result(cls_result)
    return YoloRoiClsDecision(
        behavior=behavior,
        confidence=confidence,
        horse=select_largest_box(detections, "horse") or select_largest_box(detections, "lying_horse"),
        detections=detections,
        roi_box=roi.box,
        roi_source=roi.source,
    )


def draw_yolo_roi_cls_decision(frame, decision: YoloRoiClsDecision) -> None:
    colors = {
        "horse": (30, 180, 80),
        "head": (80, 160, 230),
        "grass": (40, 200, 40),
        "water": (230, 160, 40),
        "lying_horse": (180, 90, 220),
    }
    for detection in decision.detections:
        color = colors.get(detection.name, (180, 180, 180))
        x1, y1, x2, y2 = [int(round(v)) for v in detection.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{detection.name} {detection.conf:.2f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    rx1, ry1, rx2, ry2 = decision.roi_box
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 220, 220), 3)
    label = f"ROI YOLO分类：{decision.behavior} {decision.confidence:.2f}"
    draw_label(frame, label, (max(8, rx1), max(32, ry1 - 8)), color=(0, 220, 220))


def write_csv_header(writer) -> None:
    writer.writerow(["frame", "time_sec", "behavior", "confidence", "horse_conf", "roi_source", "roi_box", "detections"])


def write_csv_row(writer, frame_index: int, fps: float, decision: YoloRoiClsDecision) -> None:
    horse_conf = "" if decision.horse is None else f"{decision.horse.conf:.4f}"
    det_summary = ";".join(f"{d.name}:{d.conf:.3f}" for d in decision.detections)
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            decision.behavior,
            f"{decision.confidence:.4f}",
            horse_conf,
            decision.roi_source,
            ",".join(str(v) for v in decision.roi_box),
            det_summary,
        ]
    )


def run_video(args, det_model, cls_model) -> int:
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

    frame_index = 0
    try:
        while True:
            if limit and frame_index >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            det_result = det_model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
            detections = detections_from_result(det_result, effective_model_conf(args))
            decision = classify_roi_frame(
                frame,
                detections=detections,
                cls_model=cls_model,
                cls_imgsz=args.cls_imgsz,
                crop_padding=args.crop_padding,
            )
            draw_yolo_roi_cls_decision(frame, decision)
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, decision)
            if not args.no_display:
                cv2.imshow("Horse Behavior ROI YOLO Classifier", resize_for_display(frame, args.display_scale))
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
    parser = argparse.ArgumentParser(description="Run detection boxes plus ROI-cropped YOLO classification behavior inference.")
    parser.add_argument("--det-model", default=DEFAULT_MODEL, help="YOLO detection weights path.")
    parser.add_argument("--cls-model", default=DEFAULT_ROI_CLS_MODEL, help="YOLO ROI classification weights path.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video path.")
    parser.add_argument("--output", default="outputs/behavior_yolo_roi_cls_demo.mp4", help="Output annotated video path.")
    parser.add_argument("--csv", default="outputs/behavior_yolo_roi_cls_demo.csv", help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level YOLO detection candidate threshold.")
    parser.add_argument("--min-grass-conf", type=float, default=0.18, help="Compatibility threshold for effective YOLO detection conf.")
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10, help="Compatibility threshold for effective YOLO detection conf.")
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05, help="Compatibility threshold for effective YOLO detection conf.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO detection image size.")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="YOLO classification image size.")
    parser.add_argument("--crop-padding", type=float, default=0.15, help="Padding ratio around selected horse ROI.")
    parser.add_argument("--max-frames", type=int, default=1800, help="Maximum frames to process.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a realtime preview window.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Realtime preview scale.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    for path_value, label in (
        (args.det_model, "detection model"),
        (args.cls_model, "classification model"),
        (args.source, "source video"),
    ):
        path = Path(path_value)
        if not path.exists():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1

    det_model = YOLO(str(Path(args.det_model)))
    cls_model = YOLO(str(Path(args.cls_model)))
    return run_video(args, det_model, cls_model)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"YOLO ROI classification behavior inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
