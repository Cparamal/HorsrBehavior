import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


TARGET_COLUMNS = ["split", "image", "label"]
DEFAULT_TRAIN_FEATURES = "dataset/behavior_features/train_features.csv"
DEFAULT_VAL_FEATURES = "dataset/behavior_features/val_features.csv"
DEFAULT_CLASSES = "dataset/behavior_labels/classes.txt"
DEFAULT_OUTPUT_DIR = "runs/behavior_cls"
DEFAULT_DETECTOR_MODEL = "runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt"


@dataclass(frozen=True)
class TrainingResult:
    model: object
    label_encoder: LabelEncoder
    feature_columns: list[str]
    classification_report: str
    confusion_matrix: object
    model_path: Path
    label_encoder_path: Path
    feature_columns_path: Path


def load_feature_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {', '.join(missing)}")
    if frame.empty:
        raise RuntimeError(f"Feature CSV is empty: {path}")
    return frame


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in TARGET_COLUMNS]


def build_label_encoder(classes_path: Path | None, labels: list[str] | None = None) -> LabelEncoder:
    if classes_path is not None and classes_path.exists():
        classes = [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif labels is not None:
        classes = list(dict.fromkeys(labels))
    else:
        raise RuntimeError("Either classes_path must exist or labels must be provided")

    if not classes:
        raise RuntimeError("No behavior classes were found")

    encoder = LabelEncoder()
    encoder.classes_ = pd.array(classes, dtype="object").to_numpy()
    return encoder


def validate_frames(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str], encoder: LabelEncoder) -> None:
    val_feature_columns = feature_columns_from_frame(val_df)
    if val_feature_columns != feature_columns:
        raise RuntimeError("Train and validation feature columns differ")

    known_classes = set(encoder.classes_)
    unknown_train = sorted(set(train_df["label"]) - known_classes)
    unknown_val = sorted(set(val_df["label"]) - known_classes)
    if unknown_train or unknown_val:
        unknown = sorted(set(unknown_train + unknown_val))
        raise RuntimeError(f"Unknown labels not present in classes file: {', '.join(unknown)}")


def train_and_save(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> TrainingResult:
    feature_columns = feature_columns_from_frame(train_df)
    validate_frames(train_df, val_df, feature_columns, label_encoder)

    x_train = train_df[feature_columns].astype(float)
    y_train = label_encoder.transform(train_df["label"])
    x_val = val_df[feature_columns].astype(float)
    y_val = label_encoder.transform(val_df["label"])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )
    matrix = confusion_matrix(y_val, y_pred, labels=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lightgbm_behavior.joblib"
    label_encoder_path = output_dir / "label_encoder.joblib"
    feature_columns_path = output_dir / "feature_columns.txt"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    feature_columns_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")

    return TrainingResult(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        classification_report=report,
        confusion_matrix=matrix,
        model_path=model_path,
        label_encoder_path=label_encoder_path,
        feature_columns_path=feature_columns_path,
    )


def build_lightgbm_model(args: argparse.Namespace, num_classes: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise RuntimeError(
            "Could not import lightgbm. Install it in the project venv first: "
            ".\\.venv\\Scripts\\python.exe -m pip install lightgbm"
        ) from exc

    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        class_weight="balanced",
        random_state=args.random_state,
        verbosity=-1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LightGBM behavior classifier from exported YOLO features.")
    parser.add_argument("--det-model", default=DEFAULT_DETECTOR_MODEL, help="YOLO detector weights for feature export.")
    parser.add_argument("--train-labels", default="dataset/behavior_labels/train.csv", help="Training behavior label CSV.")
    parser.add_argument("--val-labels", default="dataset/behavior_labels/val.csv", help="Validation behavior label CSV.")
    parser.add_argument("--train-features", default=DEFAULT_TRAIN_FEATURES, help="Training feature CSV.")
    parser.add_argument("--val-features", default=DEFAULT_VAL_FEATURES, help="Validation feature CSV.")
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help="Behavior classes.txt file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for saved model artifacts.")
    parser.add_argument("--feature-output-dir", default="dataset/behavior_features", help="Directory for exported feature CSV files.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size for feature export.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold for feature export.")
    parser.add_argument("--skip-feature-export", action="store_true", help="Use existing feature CSV files instead of exporting first.")
    parser.add_argument("--n-estimators", type=int, default=200, help="LightGBM n_estimators.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning_rate.")
    parser.add_argument("--num-leaves", type=int, default=15, help="LightGBM num_leaves.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser


def export_features_if_needed(args: argparse.Namespace) -> None:
    if args.skip_feature_export:
        return

    from horse_behavior.behavior_features import default_image_size_reader, make_yolo_predictor, resolve_image_path
    from horse_behavior.export_yolo_features import export_labeled_features
    from horse_behavior.infer_behavior import load_feed_regions
    from horse_behavior.labels import read_behavior_rows
    from horse_behavior.train_yolo import ensure_ultralytics_config_dir

    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)
    model_path = resolve_image_path(project_root, args.det_model)
    if not model_path.exists():
        raise RuntimeError(f"Missing detector model: {model_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import ultralytics. Use the project venv or install it first: "
            ".\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from exc

    output_dir = resolve_image_path(project_root, args.feature_output_dir)
    args.train_features = str(output_dir / "train_features.csv")
    args.val_features = str(output_dir / "val_features.csv")

    model = YOLO(str(model_path))
    predict_detections = make_yolo_predictor(model, imgsz=args.imgsz, conf=args.conf)
    feed_regions = load_feed_regions(resolve_image_path(project_root, args.feed_regions))

    for split, label_path, output_path in (
        ("train", resolve_image_path(project_root, args.train_labels), Path(args.train_features)),
        ("val", resolve_image_path(project_root, args.val_labels), Path(args.val_features)),
    ):
        rows = read_behavior_rows(label_path)
        written = export_labeled_features(
            rows=rows,
            output_path=output_path,
            predict_detections=predict_detections,
            image_size_reader=default_image_size_reader,
            feed_regions=feed_regions,
            project_root=project_root,
        )
        print(f"{split}: wrote {written} rows to {output_path.resolve()}")


def run(args: argparse.Namespace) -> int:
    export_features_if_needed(args)
    train_df = load_feature_csv(Path(args.train_features))
    val_df = load_feature_csv(Path(args.val_features))
    label_encoder = build_label_encoder(Path(args.classes), labels=list(train_df["label"]) + list(val_df["label"]))
    model = build_lightgbm_model(args, num_classes=len(label_encoder.classes_))

    result = train_and_save(
        train_df=train_df,
        val_df=val_df,
        model=model,
        label_encoder=label_encoder,
        output_dir=Path(args.output_dir),
    )

    print("Classification report:")
    print(result.classification_report)
    print("Confusion matrix:")
    matrix_df = pd.DataFrame(result.confusion_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(matrix_df.to_string())
    print(f"Model: {result.model_path.resolve()}")
    print(f"Label encoder: {result.label_encoder_path.resolve()}")
    print(f"Feature columns: {result.feature_columns_path.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Behavior classifier training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
