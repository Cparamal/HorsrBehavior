import math
import unittest

import pandas as pd

from horse_behavior.pose_hybrid_rules import RuleSignal, classify_pose_rule


def row(**overrides):
    base = {
        "pose_exists": 1,
        "nose_visible": 1,
        "backline_visible": 1,
        "nose_backline_y_diff": -0.35,
        "nose_to_feed_distance": -1.0,
        "nose_to_water_distance": -1.0,
        "nose_in_feed_region": 0,
        "nose_in_water_region": 0,
        "backline_flatness": 0.10,
        "horse_box_aspect_ratio": 0.80,
        "grass_exists": 0,
        "water_exists": 0,
    }
    base.update(overrides)
    return base


class PoseHybridRuleTests(unittest.TestCase):
    def test_eating_strong_when_low_nose_near_feed(self):
        signal = classify_pose_rule(row(nose_to_feed_distance=0.04, grass_exists=1))

        self.assertEqual(signal, RuleSignal("eating", "nose_near_feed", 0.90, "strong"))

    def test_drinking_strong_when_low_nose_near_water(self):
        signal = classify_pose_rule(row(nose_to_water_distance=0.03, water_exists=1))

        self.assertEqual(signal.behavior, "drinking")
        self.assertEqual(signal.strength, "strong")

    def test_head_down_medium_when_low_without_context(self):
        signal = classify_pose_rule(row())

        self.assertEqual(signal.behavior, "head_down")
        self.assertEqual(signal.strength, "medium")

    def test_lying_strong_from_flat_back_and_flat_box(self):
        signal = classify_pose_rule(row(backline_flatness=0.02, horse_box_aspect_ratio=0.45))

        self.assertEqual(signal.behavior, "lying")
        self.assertEqual(signal.reason, "flat_back_low_box")

    def test_standing_weak_when_head_is_not_low(self):
        signal = classify_pose_rule(row(nose_backline_y_diff=0.10))

        self.assertEqual(signal.behavior, "standing")
        self.assertEqual(signal.strength, "weak")

    def test_unknown_when_pose_is_missing(self):
        signal = classify_pose_rule(row(pose_exists=0))

        self.assertEqual(signal.behavior, "unknown")
        self.assertEqual(signal.reason, "no_pose")

    def test_unknown_when_head_pose_evidence_is_missing(self):
        signal = classify_pose_rule(row(nose_visible=0, nose_backline_y_diff=-1.0))

        self.assertEqual(signal, RuleSignal("unknown", "missing_head_pose", 0.0, "weak"))

    def test_unknown_when_head_pose_values_are_none_or_nan(self):
        none_signal = classify_pose_rule(row(nose_backline_y_diff=None))
        nan_signal = classify_pose_rule(row(nose_backline_y_diff=math.nan))

        self.assertEqual(none_signal.reason, "missing_head_pose")
        self.assertEqual(nan_signal.reason, "missing_head_pose")

    def test_unknown_when_head_pose_value_is_pandas_na(self):
        signal = classify_pose_rule(row(nose_backline_y_diff=pd.NA))

        self.assertEqual(signal, RuleSignal("unknown", "missing_head_pose", 0.0, "weak"))

    def test_pandas_series_with_missing_pose_flag_is_unknown(self):
        signal = classify_pose_rule(pd.Series(row(pose_exists=pd.NA)))

        self.assertEqual(signal, RuleSignal("unknown", "no_pose", 0.0, "weak"))

    def test_pandas_series_with_missing_nose_flag_is_unknown(self):
        signal = classify_pose_rule(pd.Series(row(nose_visible=pd.NA)))

        self.assertEqual(signal, RuleSignal("unknown", "missing_head_pose", 0.0, "weak"))

    def test_head_low_and_feed_threshold_boundaries_are_inclusive(self):
        signal = classify_pose_rule(row(nose_backline_y_diff=-0.18, nose_to_feed_distance=0.10, grass_exists=1))

        self.assertEqual(signal.behavior, "eating")
        self.assertEqual(signal.reason, "nose_near_feed")

    def test_water_threshold_boundary_is_inclusive(self):
        signal = classify_pose_rule(row(nose_to_water_distance=0.08, water_exists=1))

        self.assertEqual(signal.behavior, "drinking")
        self.assertEqual(signal.reason, "nose_near_water")

    def test_water_wins_when_feed_and_water_both_match(self):
        signal = classify_pose_rule(
            row(
                grass_exists=1,
                water_exists=1,
                nose_to_feed_distance=0.04,
                nose_to_water_distance=0.03,
            )
        )

        self.assertEqual(signal.behavior, "drinking")
        self.assertEqual(signal.reason, "nose_near_water")

    def test_lying_priority_does_not_require_head_pose_evidence(self):
        signal = classify_pose_rule(
            row(
                nose_visible=0,
                nose_backline_y_diff=-1.0,
                backline_flatness=0.02,
                horse_box_aspect_ratio=0.45,
            )
        )

        self.assertEqual(signal.behavior, "lying")
        self.assertEqual(signal.reason, "flat_back_low_box")


if __name__ == "__main__":
    unittest.main()
