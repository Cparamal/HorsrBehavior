import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from horse_behavior.infer_behavior import Detection
from horse_behavior.infer_behavior_pose_hybrid import (
    PoseHybridFrameResult,
    PoseHybridRuntime,
    StageTimings,
    build_parser,
    draw_pose_hybrid_result,
    process_frame,
    write_csv_header,
    write_csv_row,
)
from horse_behavior.pose_hybrid_context import DetectionContextCache
from horse_behavior.pose_hybrid_fusion import FusedPoseDecision, ModelSignal
from horse_behavior.pose_hybrid_rules import RuleSignal
from horse_behavior.pose_hybrid_state import BehaviorStateMachine, StateMachineConfig
from horse_behavior.pose_hybrid_state import StableBehaviorDecision


class FakePoseModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, frame, imgsz, conf, verbose):
        self.calls += 1
        return [self.result]


class FakeDetModel:
    names = {0: "grass", 1: "water"}

    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def predict(self, frame, imgsz, conf, verbose):
        self.calls += 1
        return [FakeDetectionResult(self.detections)]


class FakeDetectionResult:
    def __init__(self, detections):
        self.names = {0: "grass", 1: "water"}
        self.boxes = [FakeBox(0 if d.name == "grass" else 1, d.conf, d.xyxy) for d in detections]


class FakeBox:
    def __init__(self, cls, conf, xyxy):
        import torch

        self.cls = torch.tensor([cls])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([list(xyxy)], dtype=torch.float32)


class FakePoseResult:
    def __init__(self):
        import torch

        self.boxes = type(
            "Boxes",
            (),
            {
                "xyxy": torch.tensor([[10.0, 20.0, 210.0, 120.0]]),
                "conf": torch.tensor([0.9]),
            },
        )()
        self.keypoints = type(
            "Keypoints",
            (),
            {
                "xy": torch.tensor(
                    [[[40.0, 95.0], [55.0, 90.0], [85.0, 55.0], [105.0, 50.0], [150.0, 52.0], [190.0, 58.0]]]
                ),
                "conf": torch.tensor([[0.91, 0.86, 0.77, 0.88, 0.82, 0.79]]),
            },
        )()


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.2, 0.7, 0.1]], dtype=float)


class FakeEncoder:
    classes_ = np.array(["standing", "eating", "drinking"], dtype=object)


def args():
    return Namespace(
        pose_imgsz=640,
        pose_conf=0.25,
        det_imgsz=640,
        conf=0.25,
        model_conf=0.05,
        min_grass_conf=0.18,
        min_feed_region_grass_conf=0.10,
        min_overlap_grass_conf=0.05,
        det_interval=8,
        det_ttl=25,
        keypoint_threshold=0.35,
        rules_only=False,
    )


class InferBehaviorPoseHybridTests(unittest.TestCase):
    def test_process_frame_runs_detector_on_interval_and_uses_cache_between_runs(self):
        runtime = PoseHybridRuntime(
            pose_model=FakePoseModel(FakePoseResult()),
            det_model=FakeDetModel([Detection("grass", 0.9, (30.0, 80.0, 70.0, 120.0))]),
            behavior_model=FakeModel(),
            label_encoder=FakeEncoder(),
            feature_columns=["pose_exists", "nose_box_y_ratio", "grass_exists"],
            feed_regions=[(25.0, 75.0, 80.0, 125.0)],
            water_regions=[],
            context_cache=DetectionContextCache(ttl_frames=25),
            state_machine=BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 1}, exit_frames={"eating": 1})),
            feature_memory=None,
        )
        frame = np.zeros((140, 240, 3), dtype=np.uint8)

        first = process_frame(frame, 0, 25.0, runtime, args())
        second = process_frame(frame, 1, 25.0, runtime, args())

        self.assertEqual(runtime.pose_model.calls, 2)
        self.assertEqual(runtime.det_model.calls, 1)
        self.assertEqual(first.decision.behavior, "eating")
        self.assertEqual(second.decision.behavior, "eating")
        self.assertGreater(first.timings.pose_ms, 0.0)

    def test_process_frame_rules_only_skips_lightgbm(self):
        runtime = PoseHybridRuntime(
            pose_model=FakePoseModel(FakePoseResult()),
            det_model=None,
            behavior_model=None,
            label_encoder=None,
            feature_columns=[],
            feed_regions=[],
            water_regions=[],
            context_cache=DetectionContextCache(ttl_frames=25),
            state_machine=BehaviorStateMachine(StateMachineConfig(enter_frames={"head_down": 1}, exit_frames={"head_down": 1})),
            feature_memory=None,
        )
        frame = np.zeros((140, 240, 3), dtype=np.uint8)
        local_args = args()
        local_args.rules_only = True

        result = process_frame(frame, 0, 25.0, runtime, local_args)

        self.assertEqual(result.model_signal, None)
        self.assertEqual(result.decision.behavior, "head_down")

    def test_process_frame_rules_only_still_runs_detector_and_uses_grass_context(self):
        runtime = PoseHybridRuntime(
            pose_model=FakePoseModel(FakePoseResult()),
            det_model=FakeDetModel([Detection("grass", 0.9, (30.0, 80.0, 70.0, 120.0))]),
            behavior_model=None,
            label_encoder=None,
            feature_columns=[],
            feed_regions=[],
            water_regions=[],
            context_cache=DetectionContextCache(ttl_frames=25),
            state_machine=BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 1}, exit_frames={"eating": 1})),
            feature_memory=None,
        )
        frame = np.zeros((140, 240, 3), dtype=np.uint8)
        local_args = args()
        local_args.rules_only = True

        result = process_frame(frame, 0, 25.0, runtime, local_args)

        self.assertEqual(runtime.det_model.calls, 1)
        self.assertEqual(result.model_signal, None)
        self.assertEqual(result.decision.behavior, "eating")

    def test_process_frame_missing_model_artifacts_falls_back_to_rules(self):
        runtime = PoseHybridRuntime(
            pose_model=FakePoseModel(FakePoseResult()),
            det_model=FakeDetModel([Detection("grass", 0.9, (30.0, 80.0, 70.0, 120.0))]),
            behavior_model=None,
            label_encoder=None,
            feature_columns=[],
            feed_regions=[],
            water_regions=[],
            context_cache=DetectionContextCache(ttl_frames=25),
            state_machine=BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 1}, exit_frames={"eating": 1})),
            feature_memory=None,
        )
        frame = np.zeros((140, 240, 3), dtype=np.uint8)

        result = process_frame(frame, 0, 25.0, runtime, args())

        self.assertEqual(result.model_signal, None)
        self.assertEqual(result.decision.behavior, "eating")
        self.assertEqual(result.decision.source, "rules_only")

    def test_process_frame_without_detector_uses_empty_context(self):
        runtime = PoseHybridRuntime(
            pose_model=FakePoseModel(FakePoseResult()),
            det_model=None,
            behavior_model=FakeModel(),
            label_encoder=FakeEncoder(),
            feature_columns=["pose_exists", "nose_box_y_ratio", "grass_exists"],
            feed_regions=[],
            water_regions=[],
            context_cache=DetectionContextCache(ttl_frames=25),
            state_machine=BehaviorStateMachine(StateMachineConfig(enter_frames={"head_down": 1}, exit_frames={"head_down": 1})),
            feature_memory=None,
        )
        frame = np.zeros((140, 240, 3), dtype=np.uint8)

        result = process_frame(frame, 0, 25.0, runtime, args())

        self.assertEqual(result.detections, [])
        self.assertEqual(result.feature_row["grass_exists"], 0)
        self.assertEqual(result.decision.rule_behavior, "head_down")


class InferBehaviorPoseHybridCliTests(unittest.TestCase):
    def test_parser_defaults_to_realtime_pose_hybrid_paths(self):
        parsed = build_parser().parse_args([])

        self.assertEqual(parsed.pose_model, "runs/pose/horse_pose_yolo_core6_crop/weights/best.pt")
        self.assertEqual(parsed.det_model, "runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt")
        self.assertEqual(parsed.behavior_model, "runs/behavior_pose_hybrid/lightgbm_pose_behavior.joblib")
        self.assertEqual(parsed.label_encoder, "runs/behavior_pose_hybrid/label_encoder.joblib")
        self.assertEqual(parsed.feature_columns, "runs/behavior_pose_hybrid/feature_columns.txt")
        self.assertEqual(parsed.source, "video/stable_20260523_105109.mp4")
        self.assertEqual(parsed.output, "outputs/behavior_pose_hybrid.mp4")
        self.assertEqual(parsed.csv, "outputs/behavior_pose_hybrid.csv")
        self.assertEqual(parsed.feed_regions, "config/feed_regions.yaml")
        self.assertEqual(parsed.water_regions, "")
        self.assertEqual(parsed.pose_imgsz, 640)
        self.assertEqual(parsed.pose_conf, 0.25)
        self.assertEqual(parsed.det_imgsz, 640)
        self.assertEqual(parsed.conf, 0.25)
        self.assertEqual(parsed.model_conf, 0.05)
        self.assertEqual(parsed.min_grass_conf, 0.18)
        self.assertEqual(parsed.min_feed_region_grass_conf, 0.10)
        self.assertEqual(parsed.min_overlap_grass_conf, 0.05)
        self.assertEqual(parsed.det_interval, 8)
        self.assertEqual(parsed.det_ttl, 25)
        self.assertEqual(parsed.keypoint_threshold, 0.35)
        self.assertEqual(parsed.max_frames, 1800)
        self.assertFalse(parsed.rules_only)
        self.assertFalse(parsed.no_detector)
        self.assertFalse(parsed.debug)
        self.assertFalse(parsed.no_display)
        self.assertEqual(parsed.display_scale, 0.5)

    def test_non_debug_draws_only_horse_box_and_final_behavior(self):
        frame = np.zeros((160, 260, 3), dtype=np.uint8)
        result = PoseHybridFrameResult(
            decision=FusedPoseDecision("eating", 0.91, "agreement", "test", "eating", "eating", {"eating": 0.91}),
            stable=StableBehaviorDecision("eating", "eating", 0.91, "eating", 2, "held"),
            rule_signal=RuleSignal("eating", "nose_near_feed", 0.9, "strong"),
            model_signal=None,
            pose=None,
            horse=Detection("horse", 0.9, (20.0, 30.0, 200.0, 140.0)),
            detections=[Detection("grass", 0.9, (210.0, 80.0, 250.0, 130.0))],
            feature_row={},
            keypoints_json="[]",
            timings=StageTimings(1, 0, 0, 0, 0),
        )

        draw_pose_hybrid_result(frame, result, debug=False)

        self.assertGreater(int(frame[30, 20].sum()), 0)
        self.assertEqual(int(frame[80, 210].sum()), 0)

    def test_csv_writes_behavior_rule_model_and_timing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.csv"
            result = PoseHybridFrameResult(
                decision=FusedPoseDecision("eating", 0.91, "agreement", "test", "eating", "eating", {"eating": 0.91}),
                stable=StableBehaviorDecision("eating", "eating", 0.91, "eating", 2, "held"),
                rule_signal=RuleSignal("eating", "nose_near_feed", 0.9, "strong"),
                model_signal=ModelSignal("eating", 0.8765, {"eating": 0.8765, "standing": 0.1235}),
                pose=None,
                horse=Detection("horse", 0.9, (20.0, 30.0, 200.0, 140.0)),
                detections=[],
                feature_row={},
                keypoints_json="[]",
                timings=StageTimings(1, 0, 0, 0, 0),
            )
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                write_csv_header(writer)
                write_csv_row(writer, 0, 25.0, result)

            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["behavior"], "eating")
            self.assertEqual(rows[0]["rule_behavior"], "eating")
            self.assertEqual(rows[0]["model_behavior"], "eating")
            self.assertEqual(rows[0]["model_confidence"], "0.8765")
            self.assertEqual(rows[0]["pose_ms"], "1.000")


if __name__ == "__main__":
    unittest.main()
