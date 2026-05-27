import tempfile
import unittest
from pathlib import Path

import numpy as np

from horse_behavior.pose_hybrid_fusion import (
    FusionConfig,
    ModelSignal,
    fuse_rule_and_model,
    load_feature_columns,
    predict_pose_lightgbm,
)
from horse_behavior.pose_hybrid_rules import RuleSignal


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities], dtype=float)
        self.seen_columns = None

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return self.probabilities


class FakeEncoder:
    classes_ = np.array(["standing", "eating", "drinking"], dtype=object)


class PoseHybridFusionTests(unittest.TestCase):
    def test_load_feature_columns_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "columns.txt"
            path.write_text("pose_exists\nnose_box_y_ratio\n", encoding="utf-8")

            self.assertEqual(load_feature_columns(path), ["pose_exists", "nose_box_y_ratio"])

    def test_predict_pose_lightgbm_uses_saved_order(self):
        model = FakeModel([0.1, 0.8, 0.1])
        signal = predict_pose_lightgbm(
            model=model,
            label_encoder=FakeEncoder(),
            feature_row={"nose_box_y_ratio": 0.7, "pose_exists": 1},
            feature_columns=["pose_exists", "nose_box_y_ratio"],
        )

        self.assertEqual(model.seen_columns, ["pose_exists", "nose_box_y_ratio"])
        self.assertEqual(signal.behavior, "eating")
        self.assertAlmostEqual(signal.confidence, 0.8)
        self.assertEqual(signal.probabilities["drinking"], 0.1)

    def test_strong_rule_overrides_high_model_standing(self):
        decision = fuse_rule_and_model(
            rule=RuleSignal("drinking", "nose_near_water", 0.92, "strong"),
            model=ModelSignal("standing", 0.96, {"standing": 0.96, "drinking": 0.04}),
            config=FusionConfig(),
        )

        self.assertEqual(decision.behavior, "drinking")
        self.assertEqual(decision.source, "strong_rule")

    def test_high_model_wins_against_medium_head_down(self):
        decision = fuse_rule_and_model(
            rule=RuleSignal("head_down", "nose_below_backline", 0.72, "medium"),
            model=ModelSignal("eating", 0.88, {"eating": 0.88, "head_down": 0.12}),
            config=FusionConfig(),
        )

        self.assertEqual(decision.behavior, "eating")
        self.assertEqual(decision.source, "model")

    def test_matching_rule_and_model_boosts_confidence(self):
        decision = fuse_rule_and_model(
            rule=RuleSignal("eating", "nose_near_feed", 0.90, "strong"),
            model=ModelSignal("eating", 0.70, {"eating": 0.70, "standing": 0.30}),
            config=FusionConfig(),
        )

        self.assertEqual(decision.behavior, "eating")
        self.assertEqual(decision.source, "agreement")
        self.assertGreater(decision.confidence, 0.90)


if __name__ == "__main__":
    unittest.main()
