import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from horse_behavior.behavior_features import (
    BehaviorFeatureHistory,
    FEATURE_COLUMNS,
    default_image_size_reader,
    extract_behavior_features,
    make_yolo_predictor,
    resolve_image_path,
)
from horse_behavior.infer_behavior import Detection, load_feed_regions, load_regions
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_MODEL = "runs/detect/runs/detect/horse_behavior_yolo/weights/best.pt"


@dataclass(frozen=True)
class BehaviorLabelRow:
    split: str
    image: str
    label: str


def read_behavior_labels(path: Path) -> list[BehaviorLabelRow]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"split", "image", "label"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise RuntimeError(f"Behavior label CSV must contain columns: {', '.join(sorted(required))}")
        rows = []
        for index, row in enumerate(reader, 2):
            split = (row.get("split") or "").strip()
            image = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            if not split or not image or not label:
                raise RuntimeError(f"{path}:{index}: split, image and label are required")
            rows.append(BehaviorLabelRow(split=split, image=image, label=label))
    return rows


def export_labeled_features(
    rows: list[BehaviorLabelRow],
    output_path: Path,
    predict_detections: Callable[[Path], list[Detection]],
    image_size_reader: Callable[[Path], tuple[int, int]],
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]] | None = None,
    feature_history_window: int = 5,
    project_root: Path | None = None,
) -> int:
    project_root = project_root or Path.cwd()
    water_regions = water_regions or []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    history = BehaviorFeatureHistory(window_size=feature_history_window)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            image_path = resolve_image_path(project_root, row.image)
            image_size = image_size_reader(image_path)
            detections = predict_detections(image_path)
            feature_row = extract_behavior_features(
                detections,
                image_size=image_size,
                split=row.split,
                image=row.image,
                label=row.label,
                feed_regions=feed_regions,
                water_regions=water_regions,
                history=history,
            )
            writer.writerow(feature_row)
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export fixed-length behavior features from YOLO detections.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained YOLO weights path.")
    parser.add_argument("--train-labels", default="dataset/behavior_labels/train.csv", help="Training behavior label CSV.")
    parser.add_argument("--val-labels", default="dataset/behavior_labels/val.csv", help="Validation behavior label CSV.")
    parser.add_argument("--output-dir", default="dataset/behavior_features", help="Directory for exported feature CSV files.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional feed region YAML.")
    parser.add_argument("--water-regions", default="config/water_regions.yaml", help="Optional fixed drinking region YAML for LightGBM features.")
    parser.add_argument("--feature-history-window", type=int, default=5, help="Recent row window for temporal feature export.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold for keeping candidate boxes.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    model_path = resolve_image_path(project_root, args.model)
    if not model_path.exists():
        print(f"Missing model weights: {model_path}", file=sys.stderr)
        return 2

    train_labels = resolve_image_path(project_root, args.train_labels)
    val_labels = resolve_image_path(project_root, args.val_labels)
    if not train_labels.exists():
        print(f"Missing train labels: {train_labels}", file=sys.stderr)
        return 2
    if not val_labels.exists():
        print(f"Missing val labels: {val_labels}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import ultralytics. Use the project venv or install it first: "
            ".\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    predict_detections = make_yolo_predictor(model, imgsz=args.imgsz, conf=args.conf)
    feed_regions = load_feed_regions(resolve_image_path(project_root, args.feed_regions))
    water_regions = load_regions(resolve_image_path(project_root, args.water_regions))
    output_dir = resolve_image_path(project_root, args.output_dir)

    split_jobs = [
        ("train", read_behavior_labels(train_labels), output_dir / "train_features.csv"),
        ("val", read_behavior_labels(val_labels), output_dir / "val_features.csv"),
    ]

    for split, rows, output_path in split_jobs:
        written = export_labeled_features(
            rows=rows,
            output_path=output_path,
            predict_detections=predict_detections,
            image_size_reader=default_image_size_reader,
            feed_regions=feed_regions,
            water_regions=water_regions,
            feature_history_window=args.feature_history_window,
            project_root=project_root,
        )
        print(f"{split}: wrote {written} rows to {output_path.resolve()}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Feature export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
