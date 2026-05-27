import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import infer


class EntrypointTests(unittest.TestCase):
    def test_infer_dispatches_to_selected_method(self):
        with patch("horse_behavior.infer_behavior_yolo_roi_cls.run", return_value=0) as run_roi:
            result = infer.main(["--method", "roi-yolo", "--max-frames", "1", "--no-display"])

        self.assertEqual(result, 0)
        run_roi.assert_called_once()
        self.assertEqual(run_roi.call_args.args[0].max_frames, 1)

    def test_infer_dispatches_to_pose_yolo_method(self):
        with patch("horse_behavior.infer_behavior_yolo_pose.run", return_value=0) as run_pose:
            result = infer.main(["--method", "pose-yolo", "--max-frames", "1", "--no-display"])

        self.assertEqual(result, 0)
        run_pose.assert_called_once()
        self.assertEqual(run_pose.call_args.args[0].max_frames, 1)

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
