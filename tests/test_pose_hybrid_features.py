import json
import unittest

import numpy as np

from horse_behavior.infer_behavior import Detection
from horse_behavior.pose_hybrid_features import (
    CORE6_NAMES,
    Core6Pose,
    POSE_HYBRID_FEATURE_COLUMNS,
    PoseFeatureMemory,
    extract_pose_hybrid_features,
    keypoints_to_json,
    pose_instances_from_result,
    select_main_pose,
)


class Boxes:
    def __init__(self):
        import torch

        self.xyxy = torch.tensor([[10.0, 20.0, 210.0, 120.0]])
        self.conf = torch.tensor([0.85])


class Keypoints:
    def __init__(self):
        import torch

        self.xy = torch.tensor(
            [
                [
                    [40.0, 95.0],
                    [55.0, 90.0],
                    [85.0, 55.0],
                    [105.0, 50.0],
                    [150.0, 52.0],
                    [190.0, 58.0],
                ]
            ]
        )
        self.conf = torch.tensor([[0.91, 0.86, 0.77, 0.88, 0.82, 0.79]])


class Result:
    boxes = Boxes()
    keypoints = Keypoints()


class LowConfidenceBoxes:
    def __init__(self):
        import torch

        self.xyxy = torch.tensor([[10.0, 20.0, 210.0, 120.0]])
        self.conf = torch.tensor([0.20])


class LowConfidenceResult:
    boxes = LowConfidenceBoxes()
    keypoints = Keypoints()


class NonCore6Keypoints:
    def __init__(self):
        import torch

        self.xy = torch.tensor(
            [
                [
                    [40.0, 95.0],
                    [55.0, 90.0],
                    [85.0, 55.0],
                    [105.0, 50.0],
                    [150.0, 52.0],
                ]
            ]
        )
        self.conf = torch.tensor([[0.91, 0.86, 0.77, 0.88, 0.82]])


class NonCore6Result:
    boxes = Boxes()
    keypoints = NonCore6Keypoints()


class LowConfidenceNonCore6Result:
    boxes = LowConfidenceBoxes()
    keypoints = NonCore6Keypoints()


class KeypointsWithoutConfidence:
    def __init__(self):
        import torch

        self.xy = torch.tensor(
            [
                [
                    [40.0, 95.0],
                    [55.0, 90.0],
                    [85.0, 55.0],
                    [105.0, 50.0],
                    [150.0, 52.0],
                    [190.0, 58.0],
                ]
            ]
        )
        self.conf = None


class ResultWithoutKeypointConfidence:
    boxes = Boxes()
    keypoints = KeypointsWithoutConfidence()


def make_pose(nose=(40.0, 95.0), confidence=0.9):
    keypoints = np.array(
        [
            [nose[0], nose[1], 0.91],
            [55.0, 90.0, 0.86],
            [85.0, 55.0, 0.77],
            [105.0, 50.0, 0.88],
            [150.0, 52.0, 0.82],
            [190.0, 58.0, 0.79],
        ],
        dtype=np.float32,
    )
    return Core6Pose(bbox_xyxy=(10.0, 20.0, 210.0, 120.0), confidence=confidence, keypoints=keypoints)


class PoseHybridFeatureTests(unittest.TestCase):
    def test_pose_instances_from_result_reads_core6_shape(self):
        poses = pose_instances_from_result(Result(), min_pose_conf=0.25)

        self.assertEqual(CORE6_NAMES, ["nose", "jaw", "withers", "neck_end", "mid_back", "croup"])
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0].keypoints.shape, (6, 3))
        self.assertAlmostEqual(poses[0].keypoints[0, 0], 40.0)

    def test_pose_instances_from_result_filters_by_min_pose_conf(self):
        poses = pose_instances_from_result(LowConfidenceResult(), min_pose_conf=0.25)

        self.assertEqual(poses, [])

    def test_pose_instances_from_result_rejects_non_core6_shape(self):
        with self.assertRaisesRegex(RuntimeError, f"Expected {len(CORE6_NAMES)} keypoints, got 5"):
            pose_instances_from_result(NonCore6Result(), min_pose_conf=0.25)

    def test_pose_instances_from_result_rejects_non_core6_shape_before_conf_filter(self):
        with self.assertRaisesRegex(RuntimeError, f"Expected {len(CORE6_NAMES)} keypoints, got 5"):
            pose_instances_from_result(LowConfidenceNonCore6Result(), min_pose_conf=0.25)

    def test_pose_instances_without_keypoint_confidence_remain_not_visible(self):
        poses = pose_instances_from_result(ResultWithoutKeypointConfidence(), min_pose_conf=0.25)

        result = extract_pose_hybrid_features(
            pose=poses[0],
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertEqual(result.row["nose_visible"], 0)
        self.assertEqual(result.row["jaw_visible"], 0)
        self.assertEqual(result.row["withers_visible"], 0)
        self.assertEqual(result.row["keypoint_conf_mean"], -1.0)
        self.assertEqual(result.row["nose_box_x_ratio"], -1.0)
        self.assertEqual(result.row["nose_backline_y_diff"], -1.0)

    def test_select_main_pose_prefers_confidence_then_area(self):
        small = Core6Pose((0.0, 0.0, 50.0, 50.0), 0.8, make_pose().keypoints)
        large = Core6Pose((0.0, 0.0, 100.0, 100.0), 0.8, make_pose().keypoints)

        selected = select_main_pose([small, large])

        self.assertEqual(selected.bbox_xyxy, large.bbox_xyxy)

    def test_extract_features_computes_pose_geometry_and_roi_distance(self):
        pose = make_pose()
        result = extract_pose_hybrid_features(
            pose=pose,
            detections=[Detection("grass", 0.8, (30.0, 80.0, 70.0, 120.0))],
            image_size=(240, 140),
            feed_regions=[(25.0, 75.0, 80.0, 125.0)],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        row = result.row
        self.assertEqual(row["pose_exists"], 1)
        self.assertEqual(row["nose_visible"], 1)
        self.assertAlmostEqual(row["nose_box_y_ratio"], 0.75)
        self.assertLess(row["nose_backline_y_diff"], 0.0)
        self.assertAlmostEqual(row["backline_flatness"], 6.0 / 105.0)
        self.assertEqual(row["nose_in_feed_region"], 1)
        self.assertEqual(row["grass_exists"], 1)
        self.assertEqual(row["water_exists"], 0)
        self.assertEqual(result.horse.name, "horse")

    def test_feed_distance_is_normalized_by_horse_scale_for_nearby_detection(self):
        pose = make_pose(nose=(40.0, 95.0))

        result = extract_pose_hybrid_features(
            pose=pose,
            detections=[Detection("grass", 0.8, (50.0, 85.0, 70.0, 105.0))],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertAlmostEqual(result.row["nose_to_feed_distance"], 0.05)

    def test_water_distance_is_normalized_by_horse_scale_for_nearby_region(self):
        pose = make_pose(nose=(40.0, 95.0))

        result = extract_pose_hybrid_features(
            pose=pose,
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[(50.0, 85.0, 70.0, 105.0)],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertAlmostEqual(result.row["nose_to_water_distance"], 0.05)

    def test_context_distances_preserve_missing_sentinel_when_no_context_exists(self):
        result = extract_pose_hybrid_features(
            pose=make_pose(),
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertEqual(result.row["nose_to_feed_distance"], -1.0)
        self.assertEqual(result.row["nose_to_water_distance"], -1.0)

    def test_backline_flatness_preserves_missing_sentinel(self):
        pose = make_pose()
        pose.keypoints[4, 2] = 0.10

        result = extract_pose_hybrid_features(
            pose=pose,
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertEqual(result.row["backline_flatness"], -1.0)

    def test_extract_features_returns_exact_pose_hybrid_columns(self):
        result = extract_pose_hybrid_features(
            pose=make_pose(),
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=None,
        )

        self.assertEqual(list(result.row), POSE_HYBRID_FEATURE_COLUMNS)

    def test_extract_features_uses_previous_memory_for_speed(self):
        previous = PoseFeatureMemory(
            nose=(30.0, 95.0),
            neck_end=(95.0, 50.0),
            frame_index=4,
            pose_missing_count=0,
        )

        result = extract_pose_hybrid_features(
            pose=make_pose(nose=(40.0, 95.0)),
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=5,
            fps=25.0,
            previous=previous,
        )

        self.assertAlmostEqual(result.row["nose_speed"], 250.0)
        self.assertEqual(result.memory.pose_missing_count, 0)

    def test_extract_features_accepts_positional_call_and_keypoint_threshold(self):
        pose = make_pose()

        result = extract_pose_hybrid_features(
            pose,
            [],
            (240, 140),
            [],
            [],
            5,
            25.0,
            None,
            keypoint_threshold=0.95,
        )

        self.assertEqual(result.row["pose_exists"], 1)
        self.assertEqual(result.row["nose_visible"], 0)
        self.assertEqual(result.row["nose_box_x_ratio"], -1.0)

    def test_missing_pose_returns_numeric_defaults(self):
        result = extract_pose_hybrid_features(
            pose=None,
            detections=[],
            image_size=(240, 140),
            feed_regions=[],
            water_regions=[],
            frame_index=6,
            fps=25.0,
            previous=PoseFeatureMemory(None, None, 5, 2),
        )

        self.assertEqual(result.row["pose_exists"], 0)
        self.assertEqual(result.row["recent_pose_missing_count"], 3)
        self.assertIsNone(result.horse)

    def test_keypoints_to_json_serializes_core6_names(self):
        payload = json.loads(keypoints_to_json(make_pose()))

        self.assertEqual(payload[0]["name"], "nose")
        self.assertEqual(payload[-1]["name"], "croup")


if __name__ == "__main__":
    unittest.main()
