import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from horse_behavior.pose_schema import SUPERANIMAL_QUADRUPED_KEYPOINTS
from horse_behavior.train_yolo import ensure_ultralytics_config_dir, resolve_output_project


class PoseDatasetValidationError(RuntimeError):
    pass


def _parse_pose_data_yaml(data_yaml: Path) -> dict:
    data: dict[str, object] = {}
    names: dict[int, str] = {}
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
            if sep:
                names[int(key.strip())] = value.strip().strip("'\"")
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


def _parse_kpt_count(data: dict) -> int:
    raw = str(data.get("kpt_shape", "")).strip()
    if not raw:
        raise PoseDatasetValidationError("data.yaml must define kpt_shape")
    values = [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    if len(values) != 2:
        raise PoseDatasetValidationError(f"kpt_shape must look like [num_keypoints, 3], got {raw}")
    try:
        count, dims = int(values[0]), int(values[1])
    except ValueError as exc:
        raise PoseDatasetValidationError(f"Could not parse kpt_shape: {raw}") from exc
    if dims != 3:
        raise PoseDatasetValidationError(f"YOLO pose labels require kpt_shape second value 3, got {dims}")
    return count


def validate_pose_dataset(data_yaml: Path) -> dict:
    data_yaml = data_yaml.resolve()
    if not data_yaml.exists():
        raise PoseDatasetValidationError(f"Missing data.yaml: {data_yaml}")
    data = _parse_pose_data_yaml(data_yaml)
    if "train" not in data or "val" not in data:
        raise PoseDatasetValidationError("data.yaml must define both train and val paths")
    if not data["names"]:
        raise PoseDatasetValidationError("data.yaml must define class names")
    keypoint_count = _parse_kpt_count(data)
    expected_columns = 5 + keypoint_count * 3

    dataset_root = Path(str(data.get("path") or data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    issues = []
    split_summary = {}
    class_counts: Counter[int] = Counter()
    visible_keypoints = 0
    for split in ("train", "val"):
        image_dir = _resolve_split_path(dataset_root, str(data[split]))
        label_dir = image_dir.parent.parent / "labels" / image_dir.name if image_dir.parent.name == "images" else Path(
            str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        )
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
                if len(parts) != expected_columns:
                    issues.append(
                        f"{label_file}:{line_number}: expected {expected_columns} YOLO pose columns, got {len(parts)}"
                    )
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    issues.append(f"{label_file}:{line_number}: could not parse YOLO pose values")
                    continue
                if class_id not in data["names"]:
                    issues.append(f"{label_file}:{line_number}: class id {class_id} is not defined in data.yaml")
                box = values[:4]
                if not (0 <= box[0] <= 1 and 0 <= box[1] <= 1 and 0 < box[2] <= 1 and 0 < box[3] <= 1):
                    issues.append(f"{label_file}:{line_number}: normalized box values out of range")
                kpts = values[4:]
                for index in range(0, len(kpts), 3):
                    x, y, visibility = kpts[index : index + 3]
                    if visibility not in {0.0, 1.0, 2.0}:
                        issues.append(f"{label_file}:{line_number}: invalid visibility {visibility}")
                    if visibility > 0 and not (0 <= x <= 1 and 0 <= y <= 1):
                        issues.append(f"{label_file}:{line_number}: visible keypoint values out of range")
                    if visibility > 0:
                        visible_keypoints += 1
                class_counts[class_id] += 1
                objects += 1
        split_summary[split] = {"image_dir": image_dir, "label_dir": label_dir, "images": len(images), "labels": len(labels), "objects": objects}

    if issues:
        raise PoseDatasetValidationError("\n".join(issues))
    return {
        "data_yaml": data_yaml,
        "dataset_root": dataset_root,
        "names": data["names"],
        "keypoint_count": keypoint_count,
        "splits": split_summary,
        "class_counts": class_counts,
        "visible_keypoints": visible_keypoints,
    }


def print_pose_dataset_summary(summary: dict) -> None:
    print(f"Pose dataset: {summary['data_yaml']}")
    print(f"  keypoints: {summary['keypoint_count']}")
    for split, info in summary["splits"].items():
        print(f"  {split}: {info['images']} images, {info['labels']} labels, {info['objects']} poses")
    print(f"  visible keypoints: {summary['visible_keypoints']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO pose model from SuperAnimal pseudo labels.")
    parser.add_argument("--data", default="dataset/horse_pose_yolo/data.yaml", help="Path to YOLO pose data.yaml.")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Base YOLO pose checkpoint.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--name", default="horse_pose_yolo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def train(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = ensure_ultralytics_config_dir(project_root)
    summary = validate_pose_dataset(Path(args.data))
    print(f"Ultralytics config: {config_dir}")
    print_pose_dataset_summary(summary)
    if summary["keypoint_count"] != len(SUPERANIMAL_QUADRUPED_KEYPOINTS):
        print(
            f"Warning: expected {len(SUPERANIMAL_QUADRUPED_KEYPOINTS)} SuperAnimal keypoints, "
            f"got {summary['keypoint_count']}"
        )
    if args.dry_run:
        print("Dry run complete. No pose training started.")
        return 0

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Could not import ultralytics. Use the project venv or install ultralytics.") from exc

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
    print("Starting YOLO pose training...")
    results = model.train(**train_kwargs)
    print("YOLO pose training complete.")
    print(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return train(args)
    except PoseDatasetValidationError as exc:
        print(f"Pose dataset validation failed:\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"YOLO pose training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
