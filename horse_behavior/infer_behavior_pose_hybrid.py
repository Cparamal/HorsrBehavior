import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib

from horse_behavior.infer_behavior import (
    Detection,
    behavior_display_name,
    detections_from_result,
    draw_clean_behavior_box,
    draw_label,
    effective_model_conf,
    load_feed_regions,
    load_regions,
    resize_for_display,
)
from horse_behavior.pose_hybrid_context import DetectionContextCache, should_run_detector
from horse_behavior.pose_hybrid_features import (
    Core6Pose,
    PoseFeatureMemory,
    extract_pose_hybrid_features,
    pose_instances_from_result,
    select_main_pose,
)
from horse_behavior.pose_hybrid_fusion import (
    FusedPoseDecision,
    ModelSignal,
    fuse_rule_and_model,
    load_feature_columns,
    predict_pose_lightgbm,
)
from horse_behavior.pose_hybrid_rules import RuleSignal, classify_pose_rule
from horse_behavior.pose_hybrid_state import BehaviorStateMachine, StableBehaviorDecision
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_POSE_MODEL = "runs/pose/horse_pose_yolo_core6_crop/weights/best.pt"
DEFAULT_DET_MODEL = "runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt"
DEFAULT_BEHAVIOR_MODEL = "runs/behavior_pose_hybrid/lightgbm_pose_behavior.joblib"
DEFAULT_LABEL_ENCODER = "runs/behavior_pose_hybrid/label_encoder.joblib"
DEFAULT_FEATURE_COLUMNS = "runs/behavior_pose_hybrid/feature_columns.txt"
DEFAULT_OUTPUT = "outputs/behavior_pose_hybrid.mp4"
DEFAULT_CSV = "outputs/behavior_pose_hybrid.csv"


@dataclass
class PoseHybridRuntime:
    pose_model: object
    det_model: object | None
    behavior_model: object | None
    label_encoder: object | None
    feature_columns: list[str]
    feed_regions: list[tuple[float, float, float, float]]
    water_regions: list[tuple[float, float, float, float]]
    context_cache: DetectionContextCache
    state_machine: BehaviorStateMachine
    feature_memory: PoseFeatureMemory | None


@dataclass(frozen=True)
class StageTimings:
    pose_ms: float
    detector_ms: float
    feature_ms: float
    model_ms: float
    state_ms: float
    draw_ms: float = 0.0


@dataclass(frozen=True)
class PoseHybridFrameResult:
    decision: FusedPoseDecision
    stable: StableBehaviorDecision
    rule_signal: RuleSignal
    model_signal: ModelSignal | None
    pose: Core6Pose | None
    horse: Detection | None
    detections: list[Detection]
    feature_row: dict[str, float | int]
    keypoints_json: str
    timings: StageTimings


def process_frame(frame, frame_index: int, fps: float, runtime: PoseHybridRuntime, args) -> PoseHybridFrameResult:
    start = time.perf_counter()
    pose_result = runtime.pose_model.predict(frame, imgsz=args.pose_imgsz, conf=args.pose_conf, verbose=False)[0]
    pose = select_main_pose(pose_instances_from_result(pose_result, min_pose_conf=args.pose_conf))
    pose_ms = _elapsed_ms(start)

    detector_start = time.perf_counter()
    detections = _context_detections(frame, frame_index, runtime, args)
    detector_ms = _elapsed_ms(detector_start)

    feature_start = time.perf_counter()
    height, width = frame.shape[:2]
    feature_result = extract_pose_hybrid_features(
        pose=pose,
        detections=detections,
        image_size=(width, height),
        feed_regions=runtime.feed_regions,
        water_regions=runtime.water_regions,
        frame_index=frame_index,
        fps=fps,
        previous=runtime.feature_memory,
        keypoint_threshold=args.keypoint_threshold,
    )
    feature_ms = _elapsed_ms(feature_start)

    model_start = time.perf_counter()
    rule_signal = classify_pose_rule(feature_result.row)
    model_signal = None
    if not args.rules_only and _has_lightgbm_artifacts(runtime):
        model_signal = predict_pose_lightgbm(
            runtime.behavior_model,
            runtime.label_encoder,
            feature_result.row,
            runtime.feature_columns,
        )
    model_ms = _elapsed_ms(model_start)

    state_start = time.perf_counter()
    decision = fuse_rule_and_model(rule_signal, model_signal)
    stable = runtime.state_machine.update(decision)
    runtime.feature_memory = feature_result.memory
    state_ms = _elapsed_ms(state_start)

    return PoseHybridFrameResult(
        decision=decision,
        stable=stable,
        rule_signal=rule_signal,
        model_signal=model_signal,
        pose=pose,
        horse=feature_result.horse,
        detections=detections,
        feature_row=feature_result.row,
        keypoints_json=feature_result.keypoints_json,
        timings=StageTimings(
            pose_ms=pose_ms,
            detector_ms=detector_ms,
            feature_ms=feature_ms,
            model_ms=model_ms,
            state_ms=state_ms,
        ),
    )


def write_csv_header(writer) -> None:
    writer.writerow(
        [
            "frame",
            "time_sec",
            "behavior",
            "raw_behavior",
            "confidence",
            "source",
            "reason",
            "rule_behavior",
            "rule_reason",
            "rule_confidence",
            "rule_strength",
            "model_behavior",
            "model_confidence",
            "horse_conf",
            "detections",
            "keypoints",
            "pose_ms",
            "detector_ms",
            "feature_ms",
            "model_ms",
            "state_ms",
            "draw_ms",
        ]
    )


def write_csv_row(writer, frame_index: int, fps: float, result: PoseHybridFrameResult) -> None:
    model_behavior = "" if result.model_signal is None else result.model_signal.behavior
    model_confidence = "" if result.model_signal is None else f"{result.model_signal.confidence:.4f}"
    horse_conf = "" if result.horse is None else f"{result.horse.conf:.4f}"
    detections = ";".join(f"{d.name}:{d.conf:.3f}" for d in result.detections)
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            result.stable.stable_behavior,
            result.decision.behavior,
            f"{result.decision.confidence:.4f}",
            result.decision.source,
            result.decision.reason,
            result.rule_signal.behavior,
            result.rule_signal.reason,
            f"{result.rule_signal.confidence:.4f}",
            result.rule_signal.strength,
            model_behavior,
            model_confidence,
            horse_conf,
            detections,
            result.keypoints_json,
            f"{result.timings.pose_ms:.3f}",
            f"{result.timings.detector_ms:.3f}",
            f"{result.timings.feature_ms:.3f}",
            f"{result.timings.model_ms:.3f}",
            f"{result.timings.state_ms:.3f}",
            f"{result.timings.draw_ms:.3f}",
        ]
    )


def draw_pose_hybrid_result(frame, result: PoseHybridFrameResult, debug: bool = False) -> None:
    if not debug:
        draw_clean_behavior_box(frame, result.horse, result.stable.stable_behavior)
        return

    for detection in result.detections:
        color = (40, 200, 40) if detection.name == "grass" else (230, 160, 40) if detection.name == "water" else (160, 160, 160)
        x1, y1, x2, y2 = [int(round(v)) for v in detection.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        draw_label(frame, f"{detection.name} {detection.conf:.2f}", (max(8, x1), max(24, y1 - 4)), color=color)

    if result.horse is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in result.horse.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 80), 2)

    if result.pose is not None:
        for x, y, score in result.pose.keypoints:
            if float(score) >= 0.10:
                cv2.circle(frame, (int(round(x)), int(round(y))), 3, (90, 180, 230), -1, cv2.LINE_AA)

    label = (
        f"Final:{behavior_display_name(result.stable.stable_behavior)} "
        f"raw:{result.decision.behavior} rule:{result.rule_signal.behavior} "
        f"model:{result.decision.model_behavior}"
    )
    draw_label(frame, label, (8, 40), color=(40, 210, 210))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pose-hybrid horse behavior inference.")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--behavior-model", default=DEFAULT_BEHAVIOR_MODEL)
    parser.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER)
    parser.add_argument("--feature-columns", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml")
    parser.add_argument("--water-regions", default="")
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--model-conf", type=float, default=0.05)
    parser.add_argument("--min-grass-conf", type=float, default=0.18)
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10)
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05)
    parser.add_argument("--det-interval", type=int, default=8)
    parser.add_argument("--det-ttl", type=int, default=25)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--no-detector", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-scale", type=float, default=0.5)
    return parser


def run_video(args, runtime: PoseHybridRuntime) -> int:
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
            result = process_frame(frame, frame_index, fps, runtime, args)
            draw_start = time.perf_counter()
            draw_pose_hybrid_result(frame, result, debug=args.debug)
            result = PoseHybridFrameResult(
                decision=result.decision,
                stable=result.stable,
                rule_signal=result.rule_signal,
                model_signal=result.model_signal,
                pose=result.pose,
                horse=result.horse,
                detections=result.detections,
                feature_row=result.feature_row,
                keypoints_json=result.keypoints_json,
                timings=StageTimings(
                    pose_ms=result.timings.pose_ms,
                    detector_ms=result.timings.detector_ms,
                    feature_ms=result.timings.feature_ms,
                    model_ms=result.timings.model_ms,
                    state_ms=result.timings.state_ms,
                    draw_ms=_elapsed_ms(draw_start),
                ),
            )
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, result)
            if not args.no_display:
                cv2.imshow("Pose Hybrid Behavior", resize_for_display(frame, args.display_scale))
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


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    for path_value, label in ((args.pose_model, "pose model"), (args.source, "source video")):
        path = Path(path_value)
        if not path.exists():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 2

    if not args.no_detector and not Path(args.det_model).exists():
        print(f"Missing detector model: {args.det_model}", file=sys.stderr)
        return 2

    if not args.rules_only:
        missing = [
            (args.behavior_model, "behavior model"),
            (args.label_encoder, "label encoder"),
            (args.feature_columns, "feature columns"),
        ]
        for path_value, label in missing:
            path = Path(path_value)
            if not path.exists():
                print(f"Missing {label}: {path}", file=sys.stderr)
                return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1

    pose_model = YOLO(str(Path(args.pose_model)))
    det_model = None if args.no_detector else YOLO(str(Path(args.det_model)))
    behavior_model = None
    label_encoder = None
    feature_columns: list[str] = []
    if not args.rules_only:
        behavior_model = joblib.load(args.behavior_model)
        label_encoder = joblib.load(args.label_encoder)
        feature_columns = load_feature_columns(Path(args.feature_columns))

    runtime = PoseHybridRuntime(
        pose_model=pose_model,
        det_model=det_model,
        behavior_model=behavior_model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        feed_regions=load_feed_regions(Path(args.feed_regions)),
        water_regions=load_regions(Path(args.water_regions)) if args.water_regions else [],
        context_cache=DetectionContextCache(ttl_frames=args.det_ttl),
        state_machine=BehaviorStateMachine(),
        feature_memory=None,
    )
    return run_video(args, runtime)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Pose hybrid behavior inference failed: {exc}", file=sys.stderr)
        return 1


def _context_detections(frame, frame_index: int, runtime: PoseHybridRuntime, args) -> list[Detection]:
    if runtime.det_model is not None and should_run_detector(frame_index, args.det_interval):
        result = runtime.det_model.predict(frame, imgsz=args.det_imgsz, conf=effective_model_conf(args), verbose=False)[0]
        return runtime.context_cache.update(frame_index, detections_from_result(result, effective_model_conf(args)))
    return runtime.context_cache.current(frame_index)


def _has_lightgbm_artifacts(runtime: PoseHybridRuntime) -> bool:
    return (
        runtime.behavior_model is not None
        and runtime.label_encoder is not None
        and bool(runtime.feature_columns)
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
