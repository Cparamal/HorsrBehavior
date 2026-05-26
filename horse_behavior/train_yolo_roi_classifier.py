import argparse
import sys

from horse_behavior.yolo_classifier_training import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO classification model on detected horse ROI crops.")
    parser.add_argument("--data", default="dataset/behavior_roi_cls", help="YOLO ROI classification dataset directory.")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="Base YOLO classification model.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Classification image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", default="runs/behavior_yolo_roi_cls", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="horse_behavior_yolo_roi_cls", help="Run name.")
    parser.add_argument("--device", default=None, help="Device string, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. 0 is safest on Windows.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and environment.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"YOLO ROI classifier training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
