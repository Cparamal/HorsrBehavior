import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECT_CLASSES = ["head", "horse", "water", "feces", "person"]
SEGMENT_CLASSES = ["stall"]
EXCLUDED_CLASSES = {"grass_bucket"}


@dataclass(frozen=True)
class SourceItem:
    root: Path
    image_path: Path
    json_path: Path
    source_id: str


@dataclass(frozen=True)
class ConversionSummary:
    images: int
    label_files: int
    objects: Counter
    skipped: Counter


def _safe_source_id(root: Path) -> str:
    return root.name.replace(" ", "_")


def _read_image_size(image_path: Path, data: dict) -> tuple[int, int]:
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height

    with Image.open(image_path) as image:
        return image.size


def _find_image(json_path: Path, data: dict) -> Path | None:
    image_path_value = data.get("imagePath")
    candidates = []
    if image_path_value:
        candidates.append(json_path.parent / image_path_value)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(json_path.with_suffix(suffix))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def collect_source_items(source_roots: list[Path]) -> list[SourceItem]:
    items: list[SourceItem] = []
    for root in source_roots:
        if not root.exists():
            raise FileNotFoundError(f"Missing source directory: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Source is not a directory: {root}")

        source_id = _safe_source_id(root)
        for json_path in sorted(root.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            image_path = _find_image(json_path, data)
            if image_path is None:
                raise RuntimeError(f"Could not find image for annotation: {json_path}")
            items.append(SourceItem(root=root, image_path=image_path, json_path=json_path, source_id=source_id))
    if not items:
        raise RuntimeError("No annotation JSON files found in source directories.")
    return items


def split_items(items: list[SourceItem], val_ratio: float) -> dict[str, list[SourceItem]]:
    grouped: dict[str, list[SourceItem]] = {}
    for item in items:
        grouped.setdefault(item.source_id, []).append(item)

    splits = {"train": [], "val": []}
    for group_items in grouped.values():
        ordered = sorted(group_items, key=lambda item: item.image_path.name)
        val_count = max(1, round(len(ordered) * val_ratio)) if len(ordered) > 1 else 0
        val_indices = set()
        if val_count:
            if val_count == 1:
                val_indices.add(len(ordered) // 2)
            else:
                for index in range(val_count):
                    val_indices.add(round(index * (len(ordered) - 1) / (val_count - 1)))

        for index, item in enumerate(ordered):
            split = "val" if index in val_indices else "train"
            splits[split].append(item)
    return splits


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _shape_points(shape: dict) -> list[tuple[float, float]]:
    points = shape.get("points") or []
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            result.append((x, y))
    return result


def _box_from_points(points: list[tuple[float, float]], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = [_clip(point[0], 0.0, float(width)) for point in points]
    ys = [_clip(point[1], 0.0, float(height)) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 1 or box_height <= 1:
        return None
    return (
        (x1 + box_width / 2) / width,
        (y1 + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )


def _polygon_from_points(points: list[tuple[float, float]], width: int, height: int) -> list[float] | None:
    if len(points) < 3:
        return None
    normalized = []
    for x, y in points:
        normalized.extend((_clip(x, 0.0, float(width)) / width, _clip(y, 0.0, float(height)) / height))
    if len(normalized) < 6:
        return None
    return normalized


def _write_yaml(path: Path, dataset_root: Path, names: list[str]) -> None:
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reset_dataset_dir(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_detection_dataset(
    splits: dict[str, list[SourceItem]],
    output_dir: Path,
) -> ConversionSummary:
    class_to_id = {name: index for index, name in enumerate(DETECT_CLASSES)}
    objects = Counter()
    skipped = Counter()
    label_files = 0
    images = 0

    for split, items in splits.items():
        for item in items:
            data = json.loads(item.json_path.read_text(encoding="utf-8"))
            width, height = _read_image_size(item.image_path, data)
            stem = f"{item.source_id}_{item.image_path.stem}"
            target_image = output_dir / "images" / split / f"{stem}{item.image_path.suffix.lower()}"
            target_label = output_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(item.image_path, target_image)
            images += 1

            lines = []
            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label in EXCLUDED_CLASSES or label == "stall":
                    skipped[label] += 1
                    continue
                if label not in class_to_id:
                    skipped[label or "<empty>"] += 1
                    continue
                box = _box_from_points(_shape_points(shape), width, height)
                if box is None:
                    skipped[f"{label}:invalid_box"] += 1
                    continue
                lines.append(
                    " ".join([str(class_to_id[label]), *(_format_float(value) for value in box)])
                )
                objects[label] += 1

            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_files += 1

    _write_yaml(output_dir / "data.yaml", output_dir.resolve(), DETECT_CLASSES)
    return ConversionSummary(images=images, label_files=label_files, objects=objects, skipped=skipped)


def convert_segment_dataset(
    splits: dict[str, list[SourceItem]],
    output_dir: Path,
) -> ConversionSummary:
    class_to_id = {name: index for index, name in enumerate(SEGMENT_CLASSES)}
    objects = Counter()
    skipped = Counter()
    label_files = 0
    images = 0

    for split, items in splits.items():
        for item in items:
            data = json.loads(item.json_path.read_text(encoding="utf-8"))
            width, height = _read_image_size(item.image_path, data)
            stem = f"{item.source_id}_{item.image_path.stem}"
            target_image = output_dir / "images" / split / f"{stem}{item.image_path.suffix.lower()}"
            target_label = output_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(item.image_path, target_image)
            images += 1

            lines = []
            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                if label in EXCLUDED_CLASSES or label not in class_to_id:
                    skipped[label or "<empty>"] += 1
                    continue
                polygon = _polygon_from_points(_shape_points(shape), width, height)
                if polygon is None:
                    skipped[f"{label}:invalid_polygon"] += 1
                    continue
                lines.append(
                    " ".join([str(class_to_id[label]), *(_format_float(value) for value in polygon)])
                )
                objects[label] += 1

            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_files += 1

    _write_yaml(output_dir / "data.yaml", output_dir.resolve(), SEGMENT_CLASSES)
    return ConversionSummary(images=images, label_files=label_files, objects=objects, skipped=skipped)


def print_summary(name: str, summary: ConversionSummary) -> None:
    print(f"{name}: {summary.images} images, {summary.label_files} label files")
    print("  objects:")
    for label, count in sorted(summary.objects.items()):
        print(f"    {label}: {count}")
    print("  skipped:")
    for label, count in sorted(summary.skipped.items()):
        print(f"    {label}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare YOLO detect/segment datasets from X-AnyLabeling JSON files.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source directory with image/json pairs. Repeat for multiple directories.",
    )
    parser.add_argument("--detect-dir", default="dataset/detect", help="Output YOLO detection dataset directory.")
    parser.add_argument("--segment-dir", default="dataset/segment", help="Output YOLO segmentation dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per source directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.val_ratio < 0.5:
        print("--val-ratio must be greater than 0 and less than 0.5", file=sys.stderr)
        return 2

    source_roots = [Path(source).resolve() for source in args.source]
    items = collect_source_items(source_roots)
    splits = split_items(items, args.val_ratio)

    detect_dir = Path(args.detect_dir).resolve()
    segment_dir = Path(args.segment_dir).resolve()
    _reset_dataset_dir(detect_dir)
    _reset_dataset_dir(segment_dir)

    detect_summary = convert_detection_dataset(splits, detect_dir)
    segment_summary = convert_segment_dataset(splits, segment_dir)

    print(f"Sources: {len(source_roots)} directories, {len(items)} annotated images")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}")
    print_summary("Detection dataset", detect_summary)
    print_summary("Segmentation dataset", segment_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
