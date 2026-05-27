import argparse
import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from horse_behavior.behavior_features import extract_behavior_features
from horse_behavior.infer_behavior import (
    DEFAULT_MODEL,
    Detection,
    behavior_display_name,
    detections_from_result,
    draw_clean_behavior_box,
    draw_label,
    effective_model_conf,
    load_feed_regions,
    resize_for_display,
    select_largest_box,
)
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_BEHAVIOR_MODEL = "runs/behavior_cls/lightgbm_behavior.joblib"
DEFAULT_LABEL_ENCODER = "runs/behavior_cls/label_encoder.joblib"
DEFAULT_FEATURE_COLUMNS = "runs/behavior_cls/feature_columns.txt"


@dataclass(frozen=True)
class LightGBMFrameDecision:
    behavior: str
    confidence: float
    horse: Detection | None
    detections: list[Detection]


@dataclass(frozen=True)
class SmoothedPrediction:
    behavior: str
    confidence: float


class ProbabilitySmoother:
    def __init__(self, classes: list[str], window_size: int):
        self.classes = classes
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(window_size)))

    def update(self, probabilities) -> SmoothedPrediction:
        values = np.asarray(probabilities, dtype=float)
        self.history.append(values)
        averaged = np.mean(np.vstack(list(self.history)), axis=0)
        best_index = int(averaged.argmax())
        return SmoothedPrediction(behavior=str(self.classes[best_index]), confidence=float(averaged[best_index]))


def load_feature_columns(path: Path) -> list[str]:
    columns = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not columns:
        raise RuntimeError(f"No feature columns found in {path}")
    return columns


def predict_behavior(
    model,
    label_encoder,
    feature_row: dict[str, str | float | int],
    feature_columns: list[str],
) -> tuple[str, float]:
    missing = [column for column in feature_columns if column not in feature_row]
    if missing:
        raise RuntimeError(f"Feature row is missing trained columns: {', '.join(missing)}")

    frame = pd.DataFrame([{column: feature_row[column] for column in feature_columns}], columns=feature_columns).astype(float)
    probabilities = model.predict_proba(frame)[0]
    best_index = int(probabilities.argmax())
    behavior = str(label_encoder.classes_[best_index])
    confidence = float(probabilities[best_index])
    return behavior, confidence


def decide_lightgbm_frame(
    detections: list[Detection],
    image_size: tuple[int, int],
    behavior_model,
    label_encoder,
    feature_columns: list[str],
    feed_regions: list[tuple[float, float, float, float]],
    smoother: ProbabilitySmoother | None = None,
) -> LightGBMFrameDecision:
    feature_row = extract_behavior_features(
        detections,
        image_size=image_size,
        feed_regions=feed_regions,
    )
    if smoother is None:
        behavior, confidence = predict_behavior(
            model=behavior_model,
            label_encoder=label_encoder,
            feature_row=feature_row,
            feature_columns=feature_columns,
        )
    else:
        frame = pd.DataFrame([{column: feature_row[column] for column in feature_columns}], columns=feature_columns).astype(float)
        probabilities = behavior_model.predict_proba(frame)[0]
        smoothed = smoother.update(probabilities)
        behavior, confidence = smoothed.behavior, smoothed.confidence
    return LightGBMFrameDecision(
        behavior=behavior,
        confidence=confidence,
        horse=select_largest_box(detections, "horse"),
        detections=detections,
    )


def draw_lightgbm_decision(frame, decision: LightGBMFrameDecision, debug: bool = False) -> None:
    if not debug:
        draw_clean_behavior_box(frame, decision.horse, decision.behavior)
        return

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

    label = f"行为：{behavior_display_name(decision.behavior)} {decision.confidence:.2f}"
    if decision.horse is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in decision.horse.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 80), 3)
        draw_label(frame, label, (max(8, x1), max(32, y1 - 8)))
    else:
        draw_label(frame, label, (20, 40), color=(80, 160, 230))


def write_csv_header(writer) -> None:
    writer.writerow(["frame", "time_sec", "behavior", "confidence", "horse_conf", "detections"])


def write_csv_row(writer, frame_index: int, fps: float, decision: LightGBMFrameDecision) -> None:
    horse_conf = "" if decision.horse is None else f"{decision.horse.conf:.4f}"
    det_summary = ";".join(f"{d.name}:{d.conf:.3f}" for d in decision.detections)
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            decision.behavior,
            f"{decision.confidence:.4f}",
            horse_conf,
            det_summary,
        ]
    )


def run_video(args, yolo_model, behavior_model, label_encoder, feature_columns, feed_regions) -> int:
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

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
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
    smoother = ProbabilitySmoother(classes=list(label_encoder.classes_), window_size=args.smooth_window)
    try:
        while True:
            if limit and frame_index >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            result = yolo_model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
            detections = detections_from_result(result, effective_model_conf(args))
            decision = decide_lightgbm_frame(
                detections=detections,
                image_size=(width, height),
                behavior_model=behavior_model,
                label_encoder=label_encoder,
                feature_columns=feature_columns,
                feed_regions=feed_regions,
                smoother=smoother,
            )
            draw_lightgbm_decision(frame, decision, debug=args.debug)
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, decision)
            if not args.no_display:
                cv2.imshow("Horse Behavior LightGBM", resize_for_display(frame, args.display_scale))
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
    parser = argparse.ArgumentParser(description="Run realtime YOLO + LightGBM horse behavior inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path.")
    parser.add_argument("--behavior-model", default=DEFAULT_BEHAVIOR_MODEL, help="Saved LightGBM behavior model.")
    parser.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER, help="Saved label encoder.")
    parser.add_argument("--feature-columns", default=DEFAULT_FEATURE_COLUMNS, help="Saved feature column order.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video path.")
    parser.add_argument("--output", default="outputs/behavior_lightgbm_1800.mp4", help="Output annotated video path.")
    parser.add_argument("--csv", default="outputs/behavior_lightgbm_1800.csv", help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level YOLO candidate threshold before feature extraction.")
    parser.add_argument("--min-grass-conf", type=float, default=0.18, help="Compatibility threshold for effective YOLO conf.")
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10, help="Compatibility threshold for effective YOLO conf.")
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05, help="Compatibility threshold for effective YOLO conf.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--max-frames", type=int, default=1800, help="Maximum frames to process.")
    parser.add_argument("--smooth-window", type=int, default=15, help="Average LightGBM probabilities over this many recent frames.")
    parser.add_argument("--debug", action="store_true", help="Draw all YOLO detections and model confidence.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a realtime preview window.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Realtime preview scale.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    for path_value, label in (
        (args.model, "YOLO model"),
        (args.behavior_model, "behavior model"),
        (args.label_encoder, "label encoder"),
        (args.feature_columns, "feature columns"),
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

    yolo_model = YOLO(str(Path(args.model)))
    behavior_model = joblib.load(args.behavior_model)
    label_encoder = joblib.load(args.label_encoder)
    feature_columns = load_feature_columns(Path(args.feature_columns))
    feed_regions = load_feed_regions(Path(args.feed_regions))
    return run_video(args, yolo_model, behavior_model, label_encoder, feature_columns, feed_regions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"LightGBM behavior inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
