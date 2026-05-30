import argparse
import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from horse_behavior.behavior_fusion import FusionConfig, FusedBehaviorDecision, RuleSignal, fuse_roi_and_rules
from horse_behavior.infer_behavior import (
    DEFAULT_MODEL,
    DEFAULT_MIN_WATER_HEAD_OVERLAP_RATIO,
    Detection,
    add_video_segment_args,
    behavior_display_name,
    compute_video_frame_range,
    draw_clean_behavior_box,
    draw_debug_overlay,
    draw_display_detections,
    draw_label,
    effective_model_conf,
    explain_behavior,
    load_feed_regions,
    load_regions,
    resize_for_display,
    seek_video_to_frame,
)
from horse_behavior.infer_behavior_yolo_roi_cls import (
    DEFAULT_ROI_CLS_MODEL,
    YoloRoiClsDecision,
    classify_roi_frame,
    draw_yolo_roi_cls_decision,
)
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_OUTPUT = "outputs/behavior_roi_rules_demo.mp4"
DEFAULT_CSV = "outputs/behavior_roi_rules_demo.csv"


@dataclass(frozen=True)
class RoiRulesFrameDecision:
    fused: FusedBehaviorDecision
    smoothed_behavior: str
    roi: YoloRoiClsDecision
    rule_signal: RuleSignal
    detections: list[Detection]


class LabelSmoother:
    def __init__(self, window_size: int, drinking_window_size: int | None = None):
        self.window_size = max(1, int(window_size))
        self.drinking_window_size = max(1, int(drinking_window_size if drinking_window_size is not None else self.window_size))
        self.history: deque[str] = deque(maxlen=self.window_size)
        self.current = ""

    def update(self, behavior: str) -> str:
        self.history.append(behavior)
        window_size = self.drinking_window_size if behavior == "drinking" or self.current == "drinking" else self.window_size
        values = list(self.history)[-window_size:]
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        self.current = max(counts.items(), key=lambda item: item[1])[0]
        return self.current


def make_rule_signal(
    detections: list[Detection],
    image_size: tuple[int, int],
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]],
    args,
) -> tuple[RuleSignal, object]:
    explanation = explain_behavior(
        detections,
        image_size=image_size,
        feed_regions=feed_regions,
        water_regions=water_regions,
        eating_threshold_inside=args.eating_threshold_inside,
        eating_threshold_outside=args.eating_threshold_outside,
        drinking_threshold=args.drinking_threshold,
        head_down_ratio=args.head_down_ratio,
        min_grass_conf=args.min_grass_conf,
        min_feed_region_grass_conf=args.min_feed_region_grass_conf,
        min_overlap_grass_conf=args.min_overlap_grass_conf,
        min_grass_overlap_ratio=args.min_grass_overlap_ratio,
        min_water_conf=args.min_water_conf,
        min_water_head_overlap_ratio=getattr(args, "min_water_head_overlap_ratio", DEFAULT_MIN_WATER_HEAD_OVERLAP_RATIO),
        min_pose_conf=args.min_pose_conf,
        front_head_margin_ratio=args.front_head_margin_ratio,
        top_head_margin_ratio=args.top_head_margin_ratio,
    )
    return RuleSignal(behavior=explanation.behavior, reason=explanation.reason), explanation


def make_fusion_config(args) -> FusionConfig:
    return FusionConfig(
        roi_accept_threshold=args.roi_accept_threshold,
        roi_low_threshold=args.roi_low_threshold,
        strong_rule_bonus=args.strong_rule_bonus,
        weak_rule_bonus=args.weak_rule_bonus,
        contact_rule_bonus=args.contact_rule_bonus,
    )


def decide_frame(
    frame,
    det_model,
    cls_model,
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]],
    fusion_config: FusionConfig,
    smoother: LabelSmoother,
    args,
) -> tuple[RoiRulesFrameDecision, object]:
    height, width = frame.shape[:2]
    det_result = det_model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
    from horse_behavior.infer_behavior import detections_from_result

    detections = detections_from_result(det_result, effective_model_conf(args))
    roi_decision = classify_roi_frame(
        frame,
        detections=detections,
        cls_model=cls_model,
        cls_imgsz=args.cls_imgsz,
        crop_padding=args.crop_padding,
    )
    rule_signal, explanation = make_rule_signal(detections, (width, height), feed_regions, water_regions, args)
    fused = fuse_roi_and_rules(
        roi_behavior=roi_decision.behavior,
        roi_confidence=roi_decision.confidence,
        rule_signal=rule_signal,
        config=fusion_config,
    )
    smoothed_behavior = smoother.update(fused.behavior)
    return (
        RoiRulesFrameDecision(
            fused=fused,
            smoothed_behavior=smoothed_behavior,
            roi=roi_decision,
            rule_signal=rule_signal,
            detections=detections,
        ),
        explanation,
    )


def draw_debug_roi_rules_decision(frame, decision: RoiRulesFrameDecision, explanation) -> None:
    draw_yolo_roi_cls_decision(frame, decision.roi, debug=True)
    if explanation is not None:
        draw_debug_overlay(frame, explanation)

    rx1, ry1, _, _ = decision.roi.roi_box
    label = (
        f"Final:{behavior_display_name(decision.smoothed_behavior)} "
        f"raw:{decision.fused.behavior} "
        f"src:{decision.fused.source} "
        f"roi:{decision.fused.roi_behavior} {decision.fused.roi_confidence:.2f} "
        f"rule:{decision.fused.rule_behavior}/{decision.fused.rule_reason}"
    )
    draw_label(frame, label, (max(8, rx1), max(64, ry1 - 42)), color=(40, 210, 210))


def draw_clean_roi_rules_decision(frame, decision: RoiRulesFrameDecision) -> None:
    if decision.roi.horse is not None:
        draw_display_detections(frame, decision.detections, horse=decision.roi.horse)
        draw_clean_behavior_box(frame, decision.roi.horse, decision.smoothed_behavior)
        return

    x1, y1, x2, y2 = decision.roi.roi_box
    fallback_horse = Detection(name="horse", conf=0.0, xyxy=(x1, y1, x2, y2))
    draw_display_detections(frame, decision.detections, horse=fallback_horse)
    draw_clean_behavior_box(frame, fallback_horse, decision.smoothed_behavior)


def draw_roi_rules_decision(frame, decision: RoiRulesFrameDecision, explanation, debug: bool = False) -> None:
    if debug:
        draw_debug_roi_rules_decision(frame, decision, explanation)
        return
    draw_clean_roi_rules_decision(frame, decision)


def write_csv_header(writer) -> None:
    writer.writerow(
        [
            "frame",
            "time_sec",
            "behavior",
            "raw_behavior",
            "source",
            "confidence",
            "roi_behavior",
            "roi_confidence",
            "rule_behavior",
            "rule_reason",
            "roi_source",
            "roi_box",
            "detections",
        ]
    )


def write_csv_row(writer, frame_index: int, fps: float, decision: RoiRulesFrameDecision) -> None:
    det_summary = ";".join(f"{d.name}:{d.conf:.3f}" for d in decision.detections)
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            decision.smoothed_behavior,
            decision.fused.behavior,
            decision.fused.source,
            f"{decision.fused.confidence:.4f}",
            decision.fused.roi_behavior,
            f"{decision.fused.roi_confidence:.4f}",
            decision.fused.rule_behavior,
            decision.fused.rule_reason,
            decision.roi.roi_source,
            ",".join(str(v) for v in decision.roi.roi_box),
            det_summary,
        ]
    )


def run_video(args, det_model, cls_model, feed_regions, water_regions) -> int:
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
    frame_range = compute_video_frame_range(
        total_frames=total_frames,
        fps=fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_frames=args.max_frames,
    )
    limit = frame_range.frame_limit
    seek_video_to_frame(capture, frame_range)

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

    fusion_config = make_fusion_config(args)
    smoother = LabelSmoother(args.smooth_window, drinking_window_size=args.drinking_smooth_window)
    processed_frames = 0
    frame_index = frame_range.start_frame
    try:
        while True:
            if limit is not None and processed_frames >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break
            decision, explanation = decide_frame(
                frame,
                det_model=det_model,
                cls_model=cls_model,
                feed_regions=feed_regions,
                water_regions=water_regions,
                fusion_config=fusion_config,
                smoother=smoother,
                args=args,
            )
            draw_roi_rules_decision(frame, decision, explanation, debug=args.debug)
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, frame_index, fps, decision)
            if not args.no_display:
                cv2.imshow("Horse Behavior ROI-primary + Rules", resize_for_display(frame, args.display_scale))
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            processed_frames += 1
            frame_index += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames}/{limit if limit is not None else '?'} frames")
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
    print(f"Processed frames: {processed_frames}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ROI-primary behavior inference with rules as safeguards.")
    parser.add_argument("--det-model", default=DEFAULT_MODEL, help="YOLO detection weights path.")
    parser.add_argument("--cls-model", default=DEFAULT_ROI_CLS_MODEL, help="YOLO ROI classification weights path.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output annotated video path.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--water-regions", default="config/water_regions.yaml", help="Optional fixed drinking region YAML. Regions affect drinking rules but are not drawn.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level YOLO detection candidate threshold.")
    parser.add_argument("--min-grass-conf", type=float, default=0.18, help="Minimum grass confidence for rules.")
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.10, help="Minimum feed-region grass confidence.")
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.05, help="Minimum overlap grass confidence.")
    parser.add_argument("--min-grass-overlap-ratio", type=float, default=0.08, help="Minimum head-area overlap ratio for eating.")
    parser.add_argument("--min-water-conf", type=float, default=0.45, help="Minimum water confidence for drinking.")
    parser.add_argument("--min-water-head-overlap-ratio", type=float, default=0.45, help="Minimum head-area overlap ratio for detected-water drinking.")
    parser.add_argument("--min-pose-conf", type=float, default=0.35, help="Minimum lying/sitting confidence for strong rules.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO detection image size.")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="YOLO classification image size.")
    parser.add_argument("--crop-padding", type=float, default=0.15, help="Padding ratio around selected horse ROI.")
    parser.add_argument("--eating-threshold-inside", type=float, default=0.15, help="Head-to-grass distance ratio inside feed regions.")
    parser.add_argument("--eating-threshold-outside", type=float, default=0.12, help="Head-to-grass distance ratio outside feed regions.")
    parser.add_argument("--drinking-threshold", type=float, default=0.12, help="Head-to-water distance ratio.")
    parser.add_argument("--head-down-ratio", type=float, default=0.58, help="Head center low threshold inside horse box.")
    parser.add_argument("--front-head-margin-ratio", type=float, default=0.20, help="Head near front edge rule.")
    parser.add_argument("--top-head-margin-ratio", type=float, default=0.25, help="Top edge guard for front-edge rule.")
    parser.add_argument("--roi-accept-threshold", type=float, default=0.55, help="ROI confidence threshold to accept ROI over weak rules.")
    parser.add_argument("--roi-low-threshold", type=float, default=0.45, help="Below this ROI confidence, rules can fallback.")
    parser.add_argument("--strong-rule-bonus", type=float, default=0.50, help="Score bonus for lying/sitting rules.")
    parser.add_argument("--contact-rule-bonus", type=float, default=0.35, help="Score bonus for eating/drinking contact rules.")
    parser.add_argument("--weak-rule-bonus", type=float, default=0.15, help="Score bonus for weak rules.")
    parser.add_argument("--smooth-window", type=int, default=15, help="Majority smoothing window in frames.")
    parser.add_argument("--drinking-smooth-window", type=int, default=5, help="Shorter majority smoothing window in frames when entering or exiting drinking.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process. 0 means full selected segment.")
    add_video_segment_args(parser)
    parser.add_argument("--debug", action="store_true", help="Draw rule-debug overlay.")
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

    feed_regions = load_feed_regions(Path(args.feed_regions))
    water_regions = load_regions(Path(args.water_regions))
    det_model = YOLO(str(Path(args.det_model)))
    cls_model = YOLO(str(Path(args.cls_model)))
    return run_video(args, det_model, cls_model, feed_regions, water_regions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ROI-primary behavior inference failed: {exc}", file=sys.stderr)
        return 1
