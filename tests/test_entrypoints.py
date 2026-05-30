import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import infer
from horse_behavior import infer_behavior
from horse_behavior import infer_behavior_lightgbm
from horse_behavior import infer_behavior_pose_hybrid
from horse_behavior import infer_behavior_roi_rules
from horse_behavior import infer_behavior_yolo_pose
from horse_behavior import infer_behavior_yolo_roi_cls
from horse_behavior import infer_pose_superanimal_yolo
from horse_behavior.infer_behavior import compute_video_frame_range


class EntrypointTests(unittest.TestCase):
    def test_infer_dispatches_to_selected_method(self):
        with patch("horse_behavior.infer_behavior_yolo_roi_cls.run", return_value=0) as run_roi:
            result = infer.main(["--method", "roi-yolo", "--start-sec", "2.5", "--end-sec", "4.0", "--max-frames", "1", "--no-display"])

        self.assertEqual(result, 0)
        run_roi.assert_called_once()
        self.assertEqual(run_roi.call_args.args[0].max_frames, 1)
        self.assertEqual(run_roi.call_args.args[0].start_sec, 2.5)
        self.assertEqual(run_roi.call_args.args[0].end_sec, 4.0)

    def test_video_frame_range_uses_start_end_and_max_frames(self):
        frame_range = compute_video_frame_range(
            total_frames=1000,
            fps=25.0,
            start_sec=10.0,
            end_sec=20.0,
            max_frames=100,
        )

        self.assertEqual(frame_range.start_frame, 250)
        self.assertEqual(frame_range.end_frame, 500)
        self.assertEqual(frame_range.frame_limit, 100)

    def test_video_frame_range_clamps_to_video_length(self):
        frame_range = compute_video_frame_range(
            total_frames=100,
            fps=25.0,
            start_sec=3.0,
            end_sec=10.0,
            max_frames=0,
        )

        self.assertEqual(frame_range.start_frame, 75)
        self.assertEqual(frame_range.end_frame, 100)
        self.assertEqual(frame_range.frame_limit, 25)

    def test_video_frame_range_uses_full_two_minute_segment_without_default_limit(self):
        frame_range = compute_video_frame_range(
            total_frames=30000,
            fps=30.0,
            start_sec=570.0,
            end_sec=690.0,
            max_frames=0,
        )

        self.assertEqual(frame_range.start_frame, 17100)
        self.assertEqual(frame_range.end_frame, 20700)
        self.assertEqual(frame_range.frame_limit, 3600)

    def test_video_frame_range_rounds_near_30fps_video_timestamps(self):
        frame_range = compute_video_frame_range(
            total_frames=54000,
            fps=29.999474454206773,
            start_sec=570.0,
            end_sec=690.0,
            max_frames=0,
        )

        self.assertEqual(frame_range.start_frame, 17100)
        self.assertEqual(frame_range.end_frame, 20700)
        self.assertEqual(frame_range.frame_limit, 3600)

    def test_all_video_inference_parsers_accept_time_segment_args(self):
        parsers = [
            infer_behavior.build_parser(),
            infer_behavior_lightgbm.build_parser(),
            infer_behavior_pose_hybrid.build_parser(),
            infer_behavior_roi_rules.build_parser(),
            infer_behavior_yolo_pose.build_parser(),
            infer_behavior_yolo_roi_cls.build_parser(),
            infer_pose_superanimal_yolo.build_parser(),
        ]

        for parser in parsers:
            parsed = parser.parse_args(["--start-sec", "12.5", "--end-sec", "18.0"])
            self.assertEqual(parsed.start_sec, 12.5)
            self.assertEqual(parsed.end_sec, 18.0)

    def test_video_inference_parsers_do_not_limit_frames_by_default(self):
        parsers = [
            infer_behavior.build_parser(),
            infer_behavior_lightgbm.build_parser(),
            infer_behavior_pose_hybrid.build_parser(),
            infer_behavior_roi_rules.build_parser(),
            infer_behavior_yolo_pose.build_parser(),
            infer_behavior_yolo_roi_cls.build_parser(),
            infer_pose_superanimal_yolo.build_parser(),
        ]

        for parser in parsers:
            with self.subTest(parser=parser.prog):
                parsed = parser.parse_args(["--start-sec", "570", "--end-sec", "690"])
                self.assertEqual(parsed.max_frames, 0)

    def test_infer_dispatches_to_pose_yolo_method(self):
        with patch("horse_behavior.infer_behavior_yolo_pose.run", return_value=0) as run_pose:
            result = infer.main(["--method", "pose-yolo", "--max-frames", "1", "--no-display"])

        self.assertEqual(result, 0)
        run_pose.assert_called_once()
        self.assertEqual(run_pose.call_args.args[0].max_frames, 1)

    def test_infer_dispatches_to_pose_hybrid_method(self):
        with patch("horse_behavior.infer_behavior_pose_hybrid.run", return_value=0) as run_hybrid:
            result = infer.main(["--method", "pose-hybrid", "--max-frames", "1", "--no-display", "--rules-only"])

        self.assertEqual(result, 0)
        run_hybrid.assert_called_once()
        self.assertEqual(run_hybrid.call_args.args[0].max_frames, 1)

    def test_project_root_keeps_only_supported_entrypoint_scripts(self):
        root = Path(__file__).resolve().parents[1]
        scripts = sorted(path.name for path in root.glob("*.py"))

        self.assertEqual(
            scripts,
            [
                "infer.py",
                "infer_pose_behavior.py",
                "infer_pose_superanimal_yolo.py",
                "infer_roi_rules.py",
                "prepare_pose_dataset.py",
                "prepare_roi_dataset.py",
                "train_detector.py",
                "train_lightgbm.py",
                "train_pose.py",
                "train_pose_lightgbm.py",
                "train_roi_classifier.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
