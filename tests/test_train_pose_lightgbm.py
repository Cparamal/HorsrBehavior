import tempfile
import unittest
from pathlib import Path

import pandas as pd

from horse_behavior.train_pose_lightgbm import build_label_encoder, feature_columns_from_frame, load_feature_csv, train_and_save


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
    def test_feature_columns_exclude_metadata(self):
        frame = pd.DataFrame([{"split": "train", "image": "a.jpg", "label": "standing", "pose_exists": 1}])

        self.assertEqual(feature_columns_from_frame(frame), ["pose_exists"])

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
                    {"split": "train", "image": "a.jpg", "label": "standing", "pose_exists": 1.0},
                    {"split": "train", "image": "b.jpg", "label": "eating", "pose_exists": 1.0},
                ]
            )
            val_df = pd.DataFrame([{"split": "val", "image": "c.jpg", "label": "standing", "pose_exists": 1.0}])
            encoder = build_label_encoder(None, labels=["standing", "eating"])

            result = train_and_save(train_df, val_df, FakeClassifier(), encoder, output_dir)

            self.assertTrue(result.model_path.exists())
            self.assertTrue(result.label_encoder_path.exists())
            self.assertTrue(result.feature_columns_path.exists())
            self.assertEqual(result.feature_columns, ["pose_exists"])


if __name__ == "__main__":
    unittest.main()
