import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

from horse_behavior.behavior_roi import crop_behavior_roi
from horse_behavior.infer_behavior import DEFAULT_MODEL, detections_from_result
from horse_behavior.labels import read_behavior_rows, read_classes
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_CLASSES = "dataset/behavior_labels/classes.txt"
DEFAULT_TRAIN_LABELS = "dataset/behavior_labels/train.csv"
DEFAULT_VAL_LABELS = "dataset/behavior_labels/val.csv"
DEFAULT_OUTPUT_DIR = "dataset/behavior_roi_cls"


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _write_metadata_header(writer) -> None:
    writer.writerow(
        [
            "split",
            "image",
            "label",
            "output_image",
            "roi_source",
            "roi_box",
            "selected_class",
            "selected_conf",
        ]
    )


def prepare_roi_classification_dataset(
    project_root: Path,
    label_csvs: list[Path],
    classes: list[str],
    output_dir: Path,
    predict_detections,
    crop_padding: float = 0.15,
    clean: bool = True,
) -> Counter:
    known_classes = set(classes)
    rows = []
    for label_csv in label_csvs:
        rows.extend(read_behavior_rows(label_csv))

    unknown = sorted({row.label for row in rows} - known_classes)
    if unknown:
        raise RuntimeError(f"Unknown labels not present in classes file: {', '.join(unknown)}")

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    splits = sorted({row.split for row in rows})
    for split in splits:
        for label in classes:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "roi_metadata.csv"
    summary = Counter()
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        metadata_writer = csv.writer(metadata_file)
        _write_metadata_header(metadata_writer)

        for row in rows:
            source = _resolve(project_root, row.image)
            if not source.exists():
                raise RuntimeError(f"Missing source image: {source}")
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"Could not read source image: {source}")

            detections = predict_detections(source, image)
            roi = crop_behavior_roi(image, detections, padding_ratio=crop_padding)
            target = output_dir / row.split / row.label / source.name
            if not cv2.imwrite(str(target), roi.image):
                raise RuntimeError(f"Could not write ROI image: {target}")

            selected_class = "" if roi.selected is None else roi.selected.name
            selected_conf = "" if roi.selected is None else f"{roi.selected.conf:.4f}"
            metadata_writer.writerow(
                [
                    row.split,
                    row.image,
                    row.label,
                    str(target.relative_to(output_dir)),
                    roi.source,
                    ",".join(str(v) for v in roi.box),
                    selected_class,
                    selected_conf,
                ]
            )
            summary[(row.split, row.label)] += 1
    return summary


def build_predictor(model, imgsz: int, conf: float):
    def predict(image_path: Path, image):
        del image_path
        result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        return detections_from_result(result, conf)

    return predict


def print_summary(summary: Counter, classes: list[str]) -> None:
    for split in sorted({split for split, _ in summary}):
        print(f"{split}:")
        for label in classes:
            print(f"  {label}: {summary.get((split, label), 0)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a YOLO classification dataset from detected horse ROIs.")
    parser.add_argument("--det-model", default=DEFAULT_MODEL, help="YOLO detection weights path.")
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help="Behavior classes.txt path.")
    parser.add_argument("--train-labels", default=DEFAULT_TRAIN_LABELS, help="Train behavior labels CSV.")
    parser.add_argument("--val-labels", default=DEFAULT_VAL_LABELS, help="Validation behavior labels CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output ROI classification dataset directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detection image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence threshold for ROI selection.")
    parser.add_argument("--crop-padding", type=float, default=0.15, help="Padding ratio around the selected horse box.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing output directory first.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    det_model_path = _resolve(project_root, args.det_model)
    if not det_model_path.exists():
        print(f"Missing detection model: {det_model_path}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1

    classes = read_classes(_resolve(project_root, args.classes))
    output_dir = _resolve(project_root, args.output_dir)
    det_model = YOLO(str(det_model_path))
    summary = prepare_roi_classification_dataset(
        project_root=project_root,
        label_csvs=[_resolve(project_root, args.train_labels), _resolve(project_root, args.val_labels)],
        classes=classes,
        output_dir=output_dir,
        predict_detections=build_predictor(det_model, imgsz=args.imgsz, conf=args.conf),
        crop_padding=args.crop_padding,
        clean=not args.no_clean,
    )
    print(f"YOLO ROI classification dataset: {output_dir.resolve()}")
    print_summary(summary, classes)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"YOLO ROI classification dataset preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
