import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from horse_behavior.train_yolo import ensure_ultralytics_config_dir, resolve_output_project


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA = "dataset/behavior_roi_cls"
DEFAULT_MODEL = "yolo11n-cls.pt"


def validate_classification_dataset(data_dir: Path) -> dict[str, Counter]:
    if not data_dir.exists():
        raise RuntimeError(f"Missing classification dataset: {data_dir}")

    summary: dict[str, Counter] = {}
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise RuntimeError(f"Missing {split} split directory: {split_dir}")
        counts = Counter()
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            counts[label_dir.name] = len([p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not counts:
            raise RuntimeError(f"No class directories found in {split_dir}")
        if sum(counts.values()) == 0:
            raise RuntimeError(f"No images found in {split_dir}")
        summary[split] = counts
    return summary


def print_summary(summary: dict[str, Counter]) -> None:
    for split, counts in summary.items():
        print(f"{split}:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO classification model for horse behavior labels.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="YOLO classification dataset directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO classification model.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Classification image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", default="runs/behavior_yolo_roi_cls", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="horse_behavior_yolo_roi_cls", help="Run name.")
    parser.add_argument("--device", default=None, help="Device string, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. 0 is safest on Windows.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and environment.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    summary = validate_classification_dataset(data_dir)
    print_summary(summary)
    if args.dry_run:
        print("Dry run complete. No training started.")
        return 0

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import ultralytics. Use the project venv or install it first: "
            ".\\.venv\\Scripts\\python.exe -m pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(resolve_output_project(project_root, args.project)),
        "name": args.name,
        "workers": args.workers,
        "task": "classify",
    }
    if args.device:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    print(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"YOLO classifier training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
