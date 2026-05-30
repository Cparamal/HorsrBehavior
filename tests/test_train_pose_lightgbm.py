import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from horse_behavior.pose_hybrid_features import POSE_HYBRID_FEATURE_COLUMNS
from horse_behavior.train_pose_lightgbm import build_label_encoder, feature_columns_from_frame, load_feature_csv, train_and_save


def make_feature_row(split="train", image="a.jpg", label="standing", **overrides):
    row = {"split": split, "image": image, "label": label}
    row.update({column: 1.0 for column in POSE_HYBRID_FEATURE_COLUMNS})
    row.update(overrides)
    return row


class FakeClassifier:
    def fit(self, x_train, y_train):
        self.columns_ = list(x_train.columns)
        self.labels_ = list(y_train)
        return self

    def predict(self, x_val):
        return [0 for _ in range(len(x_val))]

    def predict_proba(self, x_val):
        return [[0.7, 0.3] for _ in range(len(x_val))]


class TrainPoseLightGBMTests(unittest.TestCase):
    def test_feature_columns_use_canonical_pose_contract(self):
        frame = pd.DataFrame([make_feature_row()])

        self.assertEqual(feature_columns_from_frame(frame), POSE_HYBRID_FEATURE_COLUMNS)

    def test_feature_columns_ignore_debug_and_leakage_columns(self):
        frame = pd.DataFrame(
            [
                make_feature_row(
                    frame=12,
                    time_sec=1.5,
                    keypoints="[]",
                    detections="[]",
                    behavior="standing",
                )
            ]
        )

        self.assertEqual(feature_columns_from_frame(frame), POSE_HYBRID_FEATURE_COLUMNS)

    def test_feature_columns_require_canonical_pose_columns(self):
        row = make_feature_row()
        row.pop(POSE_HYBRID_FEATURE_COLUMNS[-1])
        frame = pd.DataFrame([row])

        with self.assertRaisesRegex(RuntimeError, "Missing pose feature columns"):
            feature_columns_from_frame(frame)

    def test_load_feature_csv_requires_label_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            path.write_text("split,image,pose_exists\ntrain,a.jpg,1\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Missing required columns"):
                load_feature_csv(path)

    def test_train_and_save_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "model"
            train_df = pd.DataFrame(
                [
                    make_feature_row(split="train", image="a.jpg", label="standing", frame=1, behavior="standing"),
                    make_feature_row(split="train", image="b.jpg", label="eating", frame=2, behavior="eating"),
                ]
            )
            val_df = pd.DataFrame(
                [
                    make_feature_row(
                        split="val",
                        image="c.jpg",
                        label="standing",
                        frame=3,
                        time_sec=0.1,
                        keypoints="[]",
                        detections="[]",
                    )
                ]
            )
            encoder = build_label_encoder(None, labels=["standing", "eating"])

            result = train_and_save(train_df, val_df, FakeClassifier(), encoder, output_dir)

            self.assertTrue(result.model_path.exists())
            self.assertTrue(result.label_encoder_path.exists())
            self.assertTrue(result.feature_columns_path.exists())
            self.assertEqual(result.feature_columns, POSE_HYBRID_FEATURE_COLUMNS)
            self.assertEqual(result.feature_columns_path.read_text(encoding="utf-8").splitlines(), POSE_HYBRID_FEATURE_COLUMNS)

            reloaded_encoder = joblib.load(result.label_encoder_path)
            self.assertEqual(list(reloaded_encoder.classes_), ["standing", "eating"])


if __name__ == "__main__":
    unittest.main()
