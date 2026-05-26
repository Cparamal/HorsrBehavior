import argparse
import os
import sys
from collections import Counter
from pathlib import Path


class DatasetValidationError(RuntimeError):
    """Raised when the YOLO dataset layout or labels are not trainable."""


def ensure_ultralytics_config_dir(project_root: Path) -> Path:
    config_dir = project_root / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def _parse_simple_data_yaml(data_yaml: Path) -> dict:
    data = {}
    names = {}
    in_names = False

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names and raw_line.startswith((" ", "\t")):
            key, sep, value = line.strip().partition(":")
            if not sep:
                raise DatasetValidationError(f"Bad names entry in {data_yaml}: {raw_line}")
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError as exc:
                raise DatasetValidationError(f"Bad class id in {data_yaml}: {raw_line}") from exc
            continue

        in_names = False
        key, sep, value = line.partition(":")
        if sep:
            data[key.strip()] = value.strip().strip("'\"")

    data["names"] = names
    return data


def _resolve_split_path(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def resolve_output_project(project_root: Path, project_value: str) -> Path:
    project_path = Path(project_value)
    if project_path.is_absolute():
        return project_path
    return project_root / project_path


def validate_dataset(data_yaml: Path) -> dict:
    data_yaml = data_yaml.resolve()
    if not data_yaml.exists():
        raise DatasetValidationError(f"Missing data.yaml: {data_yaml}")

    data = _parse_simple_data_yaml(data_yaml)
    if "train" not in data or "val" not in data:
        raise DatasetValidationError("data.yaml must define both train and val paths")
    if not data["names"]:
        raise DatasetValidationError("data.yaml must define class names")

    dataset_root = Path(data.get("path") or data_yaml.parent)
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    class_ids = set(data["names"])
    class_counts = Counter()
    split_summary = {}
    issues = []

    for split in ("train", "val"):
        image_dir = _resolve_split_path(dataset_root, data[split])
        label_dir = Path(str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        if image_dir.name in {"train", "val"} and image_dir.parent.name == "images":
            label_dir = image_dir.parent.parent / "labels" / image_dir.name

        if not image_dir.exists():
            issues.append(f"Missing image directory for {split}: {image_dir}")
            continue
        if not label_dir.exists():
            issues.append(f"Missing label directory for {split}: {label_dir}")
            continue

        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} images have no label file, e.g. {missing_labels[:3]}")
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} labels have no image file, e.g. {missing_images[:3]}")
        if not images:
            issues.append(f"{split}: no images found in {image_dir}")

        objects = 0
        for label_file in labels:
            for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{label_file}:{line_number}: expected 5 YOLO columns, got {len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    issues.append(f"{label_file}:{line_number}: could not parse YOLO label values")
                    continue
                if class_id not in class_ids:
                    issues.append(f"{label_file}:{line_number}: class id {class_id} is not defined in data.yaml")
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    issues.append(f"{label_file}:{line_number}: normalized box values out of range")
                if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6:
                    issues.append(f"{label_file}:{line_number}: box extends past image width")
                if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1 + 1e-6:
                    issues.append(f"{label_file}:{line_number}: box extends past image height")
                class_counts[class_id] += 1
                objects += 1

        split_summary[split] = {
            "image_dir": image_dir,
            "label_dir": label_dir,
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
        }

    if issues:
        raise DatasetValidationError("\n".join(issues))

    return {
        "data_yaml": data_yaml,
        "dataset_root": dataset_root,
        "names": data["names"],
        "splits": split_summary,
        "class_counts": class_counts,
    }


def print_dataset_summary(summary: dict) -> None:
    print(f"Dataset: {summary['data_yaml']}")
    for split, info in summary["splits"].items():
        print(
            f"  {split}: {info['images']} images, "
            f"{info['labels']} labels, {info['objects']} objects"
        )
    print("  classes:")
    for class_id, name in sorted(summary["names"].items()):
        count = summary["class_counts"].get(class_id, 0)
        print(f"    {class_id}: {name} ({count} boxes)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for the HorseBehavior dataset.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model, e.g. yolo11n.pt or yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="horse_behavior_yolo", help="Run name under the project directory.")
    parser.add_argument("--device", default=None, help="Device string, e.g. 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. 0 is safest on Windows.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and environment.")
    return parser


def train(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = ensure_ultralytics_config_dir(project_root)
    summary = validate_dataset(Path(args.data))

    print(f"Ultralytics config: {config_dir}")
    print_dataset_summary(summary)

    missing_or_empty = [
        name
        for class_id, name in sorted(summary["names"].items())
        if summary["class_counts"].get(class_id, 0) == 0
    ]
    if missing_or_empty:
        print("Warning: these classes have zero boxes: " + ", ".join(missing_or_empty))

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
        "data": str(summary["data_yaml"]),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(resolve_output_project(project_root, args.project)),
        "name": args.name,
        "workers": args.workers,
    }
    if args.device:
        train_kwargs["device"] = args.device

    print("Starting YOLO training...")
    results = model.train(**train_kwargs)
    print("Training complete.")
    print(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return train(args)
    except DatasetValidationError as exc:
        print(f"Dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
