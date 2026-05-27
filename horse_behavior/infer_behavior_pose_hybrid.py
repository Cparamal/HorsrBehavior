import time
from dataclasses import dataclass

from horse_behavior.infer_behavior import Detection, detections_from_result, effective_model_conf
from horse_behavior.pose_hybrid_context import DetectionContextCache, should_run_detector
from horse_behavior.pose_hybrid_features import (
    Core6Pose,
    PoseFeatureMemory,
    extract_pose_hybrid_features,
    pose_instances_from_result,
    select_main_pose,
)
from horse_behavior.pose_hybrid_fusion import FusedPoseDecision, ModelSignal, fuse_rule_and_model, predict_pose_lightgbm
from horse_behavior.pose_hybrid_rules import RuleSignal, classify_pose_rule
from horse_behavior.pose_hybrid_state import BehaviorStateMachine, StableBehaviorDecision


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
    runtime.feature_memory = feature_result.memory
    feature_ms = _elapsed_ms(feature_start)

    model_start = time.perf_counter()
    rule_signal = classify_pose_rule(feature_result.row)
    model_signal = None
    if not args.rules_only:
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
    writer.writerow(["frame", "time_sec", "behavior", "stable_behavior", "confidence"])


def write_csv_row(writer, frame_index: int, fps: float, result: PoseHybridFrameResult) -> None:
    writer.writerow(
        [
            frame_index,
            f"{frame_index / fps:.3f}" if fps else "",
            result.decision.behavior,
            result.stable.stable_behavior,
            f"{result.decision.confidence:.4f}",
        ]
    )


def _context_detections(frame, frame_index: int, runtime: PoseHybridRuntime, args) -> list[Detection]:
    if runtime.det_model is not None and not args.rules_only and should_run_detector(frame_index, args.det_interval):
        result = runtime.det_model.predict(frame, imgsz=args.det_imgsz, conf=effective_model_conf(args), verbose=False)[0]
        return runtime.context_cache.update(frame_index, detections_from_result(result, effective_model_conf(args)))
    return runtime.context_cache.current(frame_index)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
