import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horse_behavior.infer_behavior import Detection
from horse_behavior.infer_pose_superanimal_yolo import POSE_MODEL_NAME, SUPERANIMAL_NAME, load_pose_runner, run_pose_for_frame
from horse_behavior.pose_schema import (
    CORE_6_KEYPOINTS,
    KEYPOINT_INDEX,
    SUPERANIMAL_QUADRUPED_KEYPOINTS,
    core_6_skeleton_indices,
    skeleton_indices,
)
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DEFAULT_OUTPUT = "dataset/horse_pose_yolo"


@dataclass(frozen=True)
class PoseKeypointSet:
    name: str
    output_names: tuple[str, ...]
    source_indices: tuple[int, ...]
    skeleton: list[list[int]]


FULL_KEYPOINT_SET = PoseKeypointSet(
    name="superanimal-39",
    output_names=tuple(SUPERANIMAL_QUADRUPED_KEYPOINTS),
    source_indices=tuple(range(len(SUPERANIMAL_QUADRUPED_KEYPOINTS))),
    skeleton=skeleton_indices(),
)

CORE_6_KEYPOINT_SET = PoseKeypointSet(
    name="core-6",
    output_names=tuple(name for name, _source_name in CORE_6_KEYPOINTS),
    source_indices=tuple(KEYPOINT_INDEX[source_name] for _name, source_name in CORE_6_KEYPOINTS),
    skeleton=core_6_skeleton_indices(),
)

KEYPOINT_SET_PRESETS = {
    FULL_KEYPOINT_SET.name: FULL_KEYPOINT_SET,
    CORE_6_KEYPOINT_SET.name: CORE_6_KEYPOINT_SET,
}


@dataclass(frozen=True)
class PoseSample:
    split: str
    image_path: Path
    label_path: Path
    frame_source: str
    frame_index: int
    visible_keypoints: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CropRegion:
    x1: int
    y1: int
    x2: int
    y2: int


def normalize_split_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def iter_image_sources(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def iter_video_sources(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return []


def label_values_from_keypoints(
    keypoints: np.ndarray,
    image_size: tuple[int, int],
    min_keypoint_conf: float,
    bbox_padding: float,
    keypoint_indices: tuple[int, ...] | None = None,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
) -> tuple[list[float], int, tuple[float, float, float, float]] | None:
    image_width, image_height = image_size
    visible_mask = keypoints[:, 2] >= min_keypoint_conf
    visible = keypoints[visible_mask]
    if len(visible) < 4:
        return None

    if bbox_xyxy is None:
        x1 = float(np.clip(np.min(visible[:, 0]), 0, image_width - 1))
        y1 = float(np.clip(np.min(visible[:, 1]), 0, image_height - 1))
        x2 = float(np.clip(np.max(visible[:, 0]), 0, image_width - 1))
        y2 = float(np.clip(np.max(visible[:, 1]), 0, image_height - 1))
    else:
        x1, y1, x2, y2 = bbox_xyxy
        x1 = float(np.clip(x1, 0, image_width - 1))
        y1 = float(np.clip(y1, 0, image_height - 1))
        x2 = float(np.clip(x2, 0, image_width - 1))
        y2 = float(np.clip(y2, 0, image_height - 1))
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if bbox_xyxy is None:
        pad_x = width * max(0.0, bbox_padding)
        pad_y = height * max(0.0, bbox_padding)
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(image_width - 1), x2 + pad_x)
        y2 = min(float(image_height - 1), y2 + pad_y)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

    values = [
        0.0,
        (x1 + x2) / 2.0 / image_width,
        (y1 + y2) / 2.0 / image_height,
        width / image_width,
        height / image_height,
    ]
    label_keypoints = keypoints[list(keypoint_indices)] if keypoint_indices is not None else keypoints
    visible_label_count = 0
    for x, y, score in label_keypoints:
        if score >= min_keypoint_conf and 0 <= x < image_width and 0 <= y < image_height:
            values.extend([float(x) / image_width, float(y) / image_height, 2.0])
            visible_label_count += 1
        else:
            values.extend([0.0, 0.0, 0.0])
    return values, visible_label_count, (x1, y1, x2, y2)


def pose_keypoint_indices(preset: str) -> tuple[int, ...]:
    return KEYPOINT_SET_PRESETS[preset].source_indices


def detection_from_yolo_label(
    label_path: Path,
    image_size: tuple[int, int],
    class_ids: set[int],
) -> Detection | None:
    if not label_path.exists():
        return None
    image_width, image_height = image_size
    best: Detection | None = None
    best_area = -1.0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if class_id not in class_ids:
            continue
        box_width = width * image_width
        box_height = height * image_height
        x1 = (x_center * image_width) - box_width / 2
        y1 = (y_center * image_height) - box_height / 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        x1 = max(0.0, min(float(image_width - 1), x1))
        y1 = max(0.0, min(float(image_height - 1), y1))
        x2 = max(0.0, min(float(image_width - 1), x2))
        y2 = max(0.0, min(float(image_height - 1), y2))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = Detection(name="horse", conf=1.0, xyxy=(x1, y1, x2, y2))
    return best


def source_label_path(image_path: Path, args) -> Path | None:
    if not args.box_label_root:
        return None
    split = source_split_from_path(image_path)
    if split is None:
        return None
    return Path(args.box_label_root) / split / f"{image_path.stem}.txt"


def write_yolo_pose_label(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_id = int(values[0])
    parts = [str(class_id)] + [f"{value:.6f}" for value in values[1:]]
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def sample_split(index: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    interval = max(2, round(1.0 / val_ratio))
    return "val" if index % interval == interval - 1 else "train"


def source_split_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return None


def choose_sample_split(
    sample_index: int,
    val_ratio: float,
    source_path: Path | None = None,
    preserve_source_split: bool = False,
) -> str:
    if preserve_source_split and source_path is not None:
        source_split = source_split_from_path(source_path)
        if source_split is not None:
            return source_split
    return sample_split(sample_index, val_ratio)


def write_data_yaml(output_root: Path, keypoint_set: PoseKeypointSet | tuple[int, ...] | None = None) -> None:
    if keypoint_set is None:
        resolved_keypoint_set = FULL_KEYPOINT_SET
    elif isinstance(keypoint_set, PoseKeypointSet):
        resolved_keypoint_set = keypoint_set
    else:
        indices = tuple(keypoint_set)
        if indices == CORE_6_KEYPOINT_SET.source_indices:
            resolved_keypoint_set = CORE_6_KEYPOINT_SET
        elif indices == FULL_KEYPOINT_SET.source_indices:
            resolved_keypoint_set = FULL_KEYPOINT_SET
        else:
            resolved_keypoint_set = PoseKeypointSet(
                name="custom",
                output_names=tuple(SUPERANIMAL_QUADRUPED_KEYPOINTS[index] for index in indices),
                source_indices=indices,
                skeleton=[],
            )
    keypoint_names = ", ".join(f"'{name}'" for name in resolved_keypoint_set.output_names)
    skeleton = json.dumps(resolved_keypoint_set.skeleton)
    content = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: horse",
            "",
            f"kpt_shape: [{len(resolved_keypoint_set.output_names)}, 3]",
            f"keypoint_names: [{keypoint_names}]",
            f"skeleton: {skeleton}",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def write_manifest(output_root: Path, samples: list[PoseSample]) -> None:
    manifest = output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "image", "label", "frame_source", "frame_index", "visible_keypoints", "bbox_xyxy"])
        for sample in samples:
            writer.writerow(
                [
                    sample.split,
                    sample.image_path.as_posix(),
                    sample.label_path.as_posix(),
                    sample.frame_source,
                    sample.frame_index,
                    sample.visible_keypoints,
                    json.dumps([float(v) for v in sample.bbox_xyxy], separators=(",", ":")),
                ]
            )


def full_frame_horse(image_size: tuple[int, int]) -> Detection:
    width, height = image_size
    return Detection(name="horse", conf=1.0, xyxy=(0.0, 0.0, float(width - 1), float(height - 1)))


def crop_frame_to_detection(frame, detection: Detection, padding: float = 0.0) -> tuple[np.ndarray, CropRegion]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * max(0.0, padding)
    pad_y = box_height * max(0.0, padding)
    crop_x1 = max(0, int(np.floor(x1 - pad_x)))
    crop_y1 = max(0, int(np.floor(y1 - pad_y)))
    crop_x2 = min(width, int(np.ceil(x2 + pad_x)))
    crop_y2 = min(height, int(np.ceil(y2 + pad_y)))
    if crop_x2 <= crop_x1:
        crop_x2 = min(width, crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(height, crop_y1 + 1)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), CropRegion(crop_x1, crop_y1, crop_x2, crop_y2)


def keypoints_crop_to_global(keypoints: np.ndarray, region: CropRegion) -> np.ndarray:
    global_keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    global_keypoints[:, 0] += float(region.x1)
    global_keypoints[:, 1] += float(region.y1)
    return global_keypoints


def draw_qc_image(frame, keypoints: np.ndarray, min_keypoint_conf: float, output_path: Path) -> None:
    for index, (x, y, score) in enumerate(keypoints):
        if score < min_keypoint_conf:
            continue
        color = (
            int(80 + (index * 37) % 175),
            int(80 + (index * 67) % 175),
            int(80 + (index * 97) % 175),
        )
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def save_sample(
    frame,
    keypoints: np.ndarray,
    source_name: str,
    frame_index: int,
    sample_index: int,
    output_root: Path,
    args,
    source_path: Path | None = None,
    horse_box: Detection | None = None,
) -> PoseSample | None:
    height, width = frame.shape[:2]
    label_data = label_values_from_keypoints(
        keypoints=keypoints,
        image_size=(width, height),
        min_keypoint_conf=args.min_keypoint_conf,
        bbox_padding=args.label_bbox_padding,
        keypoint_indices=args.keypoint_set.source_indices,
        bbox_xyxy=horse_box.xyxy if horse_box else None,
    )
    if label_data is None:
        return None
    values, visible_keypoints, bbox = label_data
    if visible_keypoints < args.min_visible_keypoints:
        return None

    split = choose_sample_split(
        sample_index=sample_index,
        val_ratio=normalize_split_ratio(args.val_ratio),
        source_path=source_path,
        preserve_source_split=args.preserve_source_split,
    )
    stem = f"{Path(source_name).stem}_frame_{frame_index:06d}"
    image_path = output_root / "images" / split / f"{stem}.jpg"
    label_path = output_root / "labels" / split / f"{stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), frame)
    write_yolo_pose_label(label_path, values)

    if args.qc_dir and sample_index < args.qc_limit:
        draw_qc_image(frame.copy(), keypoints, args.min_keypoint_conf, Path(args.qc_dir) / f"{stem}.jpg")

    return PoseSample(
        split=split,
        image_path=image_path,
        label_path=label_path,
        frame_source=source_name,
        frame_index=frame_index,
        visible_keypoints=visible_keypoints,
        bbox_xyxy=bbox,
    )


def process_images(args, pose_runner) -> list[PoseSample]:
    samples = []
    output_root = Path(args.output)
    image_paths = iter_image_sources(Path(args.source))
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            continue
        height, width = frame.shape[:2]
        horse = None
        label_path = source_label_path(image_path, args)
        if label_path is not None:
            horse = detection_from_yolo_label(label_path, image_size=(width, height), class_ids=args.box_label_class_ids)
        if horse is None:
            horse = full_frame_horse((width, height))
        pose_frame = frame
        pose_horse = horse
        crop_region = None
        if args.crop_to_box and label_path is not None:
            pose_frame, crop_region = crop_frame_to_detection(frame, horse, padding=args.crop_padding)
            crop_height, crop_width = pose_frame.shape[:2]
            pose_horse = full_frame_horse((crop_width, crop_height))
        keypoints, _ = run_pose_for_frame(
            pose_runner=pose_runner,
            frame=pose_frame,
            horse=pose_horse,
            image_size=(pose_frame.shape[1], pose_frame.shape[0]),
            bbox_padding=0.0,
        )
        if keypoints is None:
            continue
        if crop_region is not None:
            keypoints = keypoints_crop_to_global(keypoints, crop_region)
        sample = save_sample(
            frame,
            keypoints,
            image_path.name,
            0,
            len(samples),
            output_root,
            args,
            image_path,
            horse if label_path is not None else None,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def process_videos(args, pose_runner) -> list[PoseSample]:
    samples = []
    output_root = Path(args.output)
    for video_path in iter_video_sources(Path(args.source)):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Skipping unreadable video: {video_path}", file=sys.stderr)
            continue
        frame_index = 0
        try:
            while True:
                if args.max_frames and frame_index >= args.max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % max(1, args.frame_stride) != 0:
                    frame_index += 1
                    continue
                height, width = frame.shape[:2]
                keypoints, _ = run_pose_for_frame(
                    pose_runner=pose_runner,
                    frame=frame,
                    horse=full_frame_horse((width, height)),
                    image_size=(width, height),
                    bbox_padding=0.0,
                )
                if keypoints is not None:
                    sample = save_sample(frame, keypoints, video_path.name, frame_index, len(samples), output_root, args)
                    if sample is not None:
                        samples.append(sample)
                frame_index += 1
                if len(samples) and len(samples) % 100 == 0:
                    print(f"Saved {len(samples)} pose samples")
        finally:
            capture.release()
    return samples


def reset_output_dir(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a YOLO pose pseudo dataset with DeepLabCut SuperAnimal-Quadruped.")
    parser.add_argument("--source", default="video", help="Input video/image path or directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output YOLO pose dataset directory.")
    parser.add_argument("--pose-model", default=".cache/superanimal_quadruped/superanimal_quadruped_rtmpose_s.pt")
    parser.add_argument("--device", default="auto", help="Torch device for DeepLabCut, e.g. cuda:0 or cpu.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample every N video frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per video. 0 means full video.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument(
        "--preserve-source-split",
        action="store_true",
        help="For image datasets, use source train/val folder names instead of val-ratio splitting.",
    )
    parser.add_argument(
        "--keypoint-set",
        choices=sorted(KEYPOINT_SET_PRESETS),
        default="superanimal-39",
        help="Keypoint preset to export in YOLO pose labels.",
    )
    parser.add_argument("--min-keypoint-conf", type=float, default=0.35, help="Minimum DLC keypoint confidence.")
    parser.add_argument("--min-visible-keypoints", type=int, default=12, help="Minimum visible keypoints to keep a sample.")
    parser.add_argument("--label-bbox-padding", type=float, default=0.08, help="Padding around visible keypoint bbox.")
    parser.add_argument(
        "--box-label-root",
        default="",
        help="Optional YOLO detection label root with train/val subfolders. Horse boxes guide SuperAnimal and pose bboxes.",
    )
    parser.add_argument(
        "--crop-to-box",
        action="store_true",
        help="Run SuperAnimal on detected horse crops and convert crop keypoints back to full-image coordinates.",
    )
    parser.add_argument("--crop-padding", type=float, default=0.10, help="Padding around detected horse crops.")
    parser.add_argument(
        "--box-label-class-ids",
        default="0,4",
        help="Comma-separated source detection class ids to use as horse boxes. Defaults to horse and lying_horse.",
    )
    parser.add_argument("--pose-batch-size", type=int, default=1, help="DLC pose runner batch size.")
    parser.add_argument("--qc-dir", default="outputs/pose_dataset_qc", help="Optional QC image directory. Empty disables QC.")
    parser.add_argument("--qc-limit", type=int, default=80, help="Maximum QC images to write.")
    parser.add_argument("--no-reset", action="store_true", help="Append to existing output instead of recreating it.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)
    source = Path(args.source)
    output_root = Path(args.output)
    if not source.exists():
        print(f"Missing source: {source}", file=sys.stderr)
        return 2
    if not args.no_reset:
        reset_output_dir(output_root)
    args.keypoint_set = KEYPOINT_SET_PRESETS[args.keypoint_set]
    args.box_label_class_ids = {int(value.strip()) for value in args.box_label_class_ids.split(",") if value.strip()}

    if not iter_image_sources(source) and not iter_video_sources(source):
        print(f"No supported images or videos found under {source}", file=sys.stderr)
        return 2

    pose_runner = load_pose_runner(args)
    samples = []
    if iter_image_sources(source):
        samples.extend(process_images(args, pose_runner))
    if iter_video_sources(source):
        samples.extend(process_videos(args, pose_runner))

    write_data_yaml(output_root, args.keypoint_set)
    write_manifest(output_root, samples)
    print(f"YOLO pose dataset: {output_root.resolve()}")
    print(f"Saved samples: {len(samples)}")
    print(f"Keypoint set: {args.keypoint_set.name} ({len(args.keypoint_set.output_names)} points)")
    print(f"DeepLabCut source: {SUPERANIMAL_NAME}/{POSE_MODEL_NAME}")
    if len(samples) == 0:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Pose dataset generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
