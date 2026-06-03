import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from horse_behavior.behavior_features import (  # noqa: E402
    FEATURE_COLUMNS,
    BehaviorFeatureHistory,
    extract_behavior_features,
)
from horse_behavior.infer_behavior import detections_from_result, load_feed_regions, load_regions  # noqa: E402
from horse_behavior.train_yolo import ensure_ultralytics_config_dir  # noqa: E402


DEFAULT_VIDEO_DIR = "video"
DEFAULT_OUTPUT_DIR = "dataset/timesequence"
DEFAULT_DETECTOR_MODEL = "runs/multiframes/horse_multiframe_detect/weights/best.pt"
DEFAULT_LABEL_ORDER = ["standing", "eating", "drinking", "lying", "sitting", "head_down"]
REQUIRED_COLUMNS = ["video", "stall_id", "horse_id", "start_sec", "end_sec", "label"]
TARGET_COLUMNS = {"split", "image", "label"}
NUMERIC_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in TARGET_COLUMNS]
UNKNOWN_LABEL = "unknown"


@dataclass(frozen=True)
class AnnotationInterval:
    video: str
    stall_id: str
    horse_id: str
    start_sec: float
    end_sec: float
    label: str
    source_file: str
    source_format: str


@dataclass(frozen=True)
class FrameSample:
    video: str
    frame_index: int
    time_sec: float
    label: str
    stall_id: str
    horse_id: str
    source_file: str


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def read_annotation_frame(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, encoding="utf-8-sig"), "csv"
    except Exception as csv_exc:
        try:
            return pd.read_excel(path, engine="xlrd"), "excel"
        except Exception as excel_exc:
            raise RuntimeError(f"Could not read annotation file as CSV or Excel: {path}") from excel_exc or csv_exc


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "stand": "standing",
        "eat": "eating",
        "drink": "drinking",
        "lie": "lying",
        "lying_horse": "lying",
        "sitting_horse": "sitting",
    }
    return aliases.get(text, text)


def load_annotations(video_dir: Path) -> list[AnnotationInterval]:
    csv_paths = sorted(video_dir.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No annotation CSV files found in {video_dir}")

    intervals: list[AnnotationInterval] = []
    for path in csv_paths:
        frame, source_format = read_annotation_frame(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise RuntimeError(f"Missing columns in {path}: {', '.join(missing)}")

        for row_index, row in frame.iterrows():
            video = str(row["video"]).strip()
            label = normalize_label(row["label"])
            if not video or not label:
                raise RuntimeError(f"Bad annotation row {row_index + 2} in {path}")

            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if end_sec <= start_sec:
                raise RuntimeError(f"end_sec must be greater than start_sec in {path}, row {row_index + 2}")

            intervals.append(
                AnnotationInterval(
                    video=video,
                    stall_id=str(row["stall_id"]).strip(),
                    horse_id=str(row["horse_id"]).strip(),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=label,
                    source_file=str(path),
                    source_format=source_format,
                )
            )

    return sorted(intervals, key=lambda item: (item.video, item.start_sec, item.end_sec))


def validate_video_files(intervals: Iterable[AnnotationInterval], video_dir: Path) -> None:
    missing = []
    for video in sorted({interval.video for interval in intervals}):
        if not (video_dir / video).exists():
            missing.append(str(video_dir / video))
    if missing:
        raise RuntimeError("Missing video files:\n" + "\n".join(missing))


def annotation_at_time(intervals: list[AnnotationInterval], time_sec: float) -> AnnotationInterval | None:
    for interval in intervals:
        if interval.start_sec <= time_sec < interval.end_sec:
            return interval
    if intervals and abs(time_sec - intervals[-1].end_sec) < 1e-6:
        return intervals[-1]
    return None


def sample_times_for_video(
    intervals: list[AnnotationInterval],
    duration_sec: float,
    sample_fps: float,
    max_seconds: float,
) -> np.ndarray:
    start_sec = max(0.0, min(interval.start_sec for interval in intervals))
    end_sec = max(interval.end_sec for interval in intervals)
    if duration_sec > 0:
        end_sec = min(end_sec, duration_sec)
    if max_seconds > 0:
        end_sec = min(end_sec, start_sec + max_seconds)
    if end_sec <= start_sec:
        return np.empty((0,), dtype=np.float64)
    return np.arange(start_sec, end_sec, 1.0 / sample_fps, dtype=np.float64)


def iter_sampled_frames(capture, frame_indices: np.ndarray):
    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    for target_frame in frame_indices:
        target = int(target_frame)
        if target < current_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            current_frame = target
        while current_frame < target:
            if not capture.grab():
                return
            current_frame += 1
        ok, frame = capture.read()
        if not ok:
            return
        current_frame += 1
        yield frame


def predict_batch(model, frames: list[np.ndarray], imgsz: int, conf: float, device: str) -> list:
    kwargs = {
        "imgsz": imgsz,
        "conf": conf,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    return model.predict(frames, **kwargs)


def extract_video_features(
    video_path: Path,
    intervals: list[AnnotationInterval],
    model,
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]],
    sample_fps: float,
    imgsz: int,
    conf: float,
    batch_size: int,
    device: str,
    feature_history_window: int,
    max_seconds: float,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0:
        source_fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video size: {video_path}")

    duration_sec = total_frames / source_fps if total_frames > 0 else max(interval.end_sec for interval in intervals)
    sample_times = sample_times_for_video(intervals, duration_sec, sample_fps, max_seconds)
    if sample_times.size == 0:
        capture.release()
        return []

    frame_indices = np.rint(sample_times * source_fps).astype(np.int64)
    if total_frames > 0:
        frame_indices = np.clip(frame_indices, 0, max(0, total_frames - 1))

    rows: list[dict[str, object]] = []
    history = BehaviorFeatureHistory(window_size=feature_history_window)
    batch_frames: list[np.ndarray] = []
    batch_samples: list[FrameSample] = []
    frame_iter = iter_sampled_frames(capture, frame_indices)
    progress = tqdm(
        zip(sample_times, frame_indices, frame_iter),
        total=len(sample_times),
        desc=video_path.name,
        unit="frame",
    )

    def flush_batch() -> None:
        if not batch_frames:
            return
        results = predict_batch(model, batch_frames, imgsz=imgsz, conf=conf, device=device)
        for result, sample in zip(results, batch_samples):
            detections = detections_from_result(result, conf_threshold=conf)
            row = extract_behavior_features(
                detections=detections,
                image_size=(width, height),
                split="all",
                image=f"{sample.video}:frame_{sample.frame_index}",
                label=sample.label,
                feed_regions=feed_regions,
                water_regions=water_regions,
                history=history,
            )
            row.update(
                {
                    "video": sample.video,
                    "frame_index": sample.frame_index,
                    "time_sec": sample.time_sec,
                    "stall_id": sample.stall_id,
                    "horse_id": sample.horse_id,
                    "source_file": sample.source_file,
                }
            )
            rows.append(row)
        batch_frames.clear()
        batch_samples.clear()

    try:
        for time_sec, frame_index, frame in progress:
            interval = annotation_at_time(intervals, float(time_sec))
            label = UNKNOWN_LABEL if interval is None else interval.label
            sample = FrameSample(
                video=video_path.name,
                frame_index=int(frame_index),
                time_sec=float(time_sec),
                label=label,
                stall_id="" if interval is None else interval.stall_id,
                horse_id="" if interval is None else interval.horse_id,
                source_file="" if interval is None else interval.source_file,
            )
            batch_frames.append(frame)
            batch_samples.append(sample)
            if len(batch_frames) >= batch_size:
                flush_batch()
        flush_batch()
    finally:
        capture.release()

    return rows


def labels_from_window(labels: list[str], min_label_ratio: float) -> tuple[str | None, float]:
    if not labels or any(label == UNKNOWN_LABEL for label in labels):
        return None, 0.0
    label, count = Counter(labels).most_common(1)[0]
    ratio = count / len(labels)
    if ratio < min_label_ratio:
        return None, ratio
    return label, ratio


def most_common_text(values: Iterable[object]) -> str:
    texts = [str(value) for value in values if str(value)]
    if not texts:
        return ""
    return Counter(texts).most_common(1)[0][0]


def build_windows(
    frame_rows: list[dict[str, object]],
    sample_fps: float,
    window_sec: float,
    stride_sec: float,
    min_label_ratio: float,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row["video"])].append(row)

    window_size = max(1, int(round(sample_fps * window_sec)))
    stride = max(1, int(round(sample_fps * stride_sec)))
    windows: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, object]] = []

    for video, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: float(row["time_sec"]))
        if len(rows) < window_size:
            continue
        features = np.asarray(
            [[float(row[column]) for column in NUMERIC_FEATURE_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        row_labels = [str(row["label"]) for row in rows]
        for start in range(0, len(rows) - window_size + 1, stride):
            end = start + window_size
            label, label_ratio = labels_from_window(row_labels[start:end], min_label_ratio=min_label_ratio)
            if label is None:
                continue

            window_rows = rows[start:end]
            window_id = len(windows)
            windows.append(features[start:end])
            labels.append(label)
            metadata.append(
                {
                    "window_id": window_id,
                    "video": video,
                    "stall_id": most_common_text(row["stall_id"] for row in window_rows),
                    "horse_id": most_common_text(row["horse_id"] for row in window_rows),
                    "start_sec": float(window_rows[0]["time_sec"]),
                    "end_sec": float(window_rows[-1]["time_sec"]) + 1.0 / sample_fps,
                    "start_frame": int(window_rows[0]["frame_index"]),
                    "end_frame": int(window_rows[-1]["frame_index"]),
                    "label": label,
                    "label_ratio": float(label_ratio),
                    "source_file": most_common_text(row["source_file"] for row in window_rows),
                }
            )

    if not windows:
        raise RuntimeError("No trainable windows were generated from the annotations.")
    return np.stack(windows).astype(np.float32), labels, pd.DataFrame(metadata)


def ordered_label_names(labels: Iterable[str]) -> list[str]:
    seen = set(labels)
    ordered = [label for label in DEFAULT_LABEL_ORDER if label in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def split_indices(y: np.ndarray, val_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if val_size <= 0:
        return indices, np.empty((0,), dtype=np.int64)
    counts = Counter(int(value) for value in y)
    stratify = y if all(count >= 2 for count in counts.values()) else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def normalize_windows(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = X[train_idx].reshape(-1, X.shape[-1])
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((X - mean) / std).astype(np.float32), mean, std


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_split(output_dir: Path, name: str, X: np.ndarray, y: np.ndarray, indices: np.ndarray, metadata: pd.DataFrame) -> None:
    np.savez_compressed(output_dir / f"{name}.npz", X=X[indices], y=y[indices])
    split_meta = metadata.iloc[indices].copy()
    split_meta.insert(1, "split", name)
    split_meta.to_csv(output_dir / f"metadata_{name}.csv", index=False, encoding="utf-8")


def write_dataset(
    output_dir: Path,
    frame_rows: list[dict[str, object]],
    annotations: list[AnnotationInterval],
    X: np.ndarray,
    label_texts: list[str],
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_names = ordered_label_names(label_texts)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in label_texts], dtype=np.int64)

    train_idx, val_idx = split_indices(y, val_size=args.val_size, random_state=args.random_state)
    X_norm, mean, std = normalize_windows(X, train_idx)

    save_split(output_dir, "train", X_norm, y, train_idx, metadata)
    save_split(output_dir, "val", X_norm, y, val_idx, metadata)
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)

    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(output_dir / "frame_features.csv", index=False, encoding="utf-8")
    metadata.to_csv(output_dir / "metadata_all.csv", index=False, encoding="utf-8")
    pd.DataFrame([asdict(interval) for interval in annotations]).to_csv(
        output_dir / "annotations_normalized.csv",
        index=False,
        encoding="utf-8",
    )
    save_json(output_dir / "feature_names.json", NUMERIC_FEATURE_COLUMNS)
    save_json(output_dir / "label_names.json", label_names)

    split_counts = {
        "all": Counter(label_texts),
        "train": Counter(label_names[int(label_id)] for label_id in y[train_idx]),
        "val": Counter(label_names[int(label_id)] for label_id in y[val_idx]),
    }
    summary = {
        "sample_fps": args.sample_fps,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "window_size": int(round(args.sample_fps * args.window_sec)),
        "min_label_ratio": args.min_label_ratio,
        "features": len(NUMERIC_FEATURE_COLUMNS),
        "frames": len(frame_rows),
        "windows": int(len(y)),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "labels": label_names,
        "class_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "videos": sorted({interval.video for interval in annotations}),
        "split_strategy": "stratified_window",
        "note": "Window-level split is intended for a first usable baseline; use video/horse-level splits after collecting more videos per class.",
    }
    save_json(output_dir / "dataset_summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a temporal horse behavior dataset from video CSV annotations.")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory containing MP4 videos and behavior CSV files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output dataset directory.")
    parser.add_argument("--det-model", default=DEFAULT_DETECTOR_MODEL, help="YOLO detector weights used for frame features.")
    parser.add_argument("--feed-regions", default="config/feed_regions.yaml", help="Optional fixed feed region YAML.")
    parser.add_argument("--water-regions", default="config/water_regions.yaml", help="Optional fixed water region YAML.")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second sampled from each annotated video.")
    parser.add_argument("--window-sec", type=float, default=8.0, help="Seconds per TCN input window.")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Sliding-window stride in seconds.")
    parser.add_argument("--min-label-ratio", type=float, default=0.80, help="Required majority-label ratio in a window.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction for stratified window split.")
    parser.add_argument("--feature-history-window", type=int, default=5, help="Recent frame count for existing temporal features.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO candidate confidence threshold for feature extraction.")
    parser.add_argument("--batch-size", type=int, default=16, help="YOLO inference batch size.")
    parser.add_argument("--device", default="", help="Optional YOLO device, e.g. 0 or cpu. Empty lets Ultralytics choose.")
    parser.add_argument("--max-seconds-per-video", type=float, default=0.0, help="Debug limit. 0 processes all annotated seconds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/val split.")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.sample_fps <= 0:
        raise RuntimeError("--sample-fps must be greater than 0")
    if args.window_sec <= 0 or args.stride_sec <= 0:
        raise RuntimeError("--window-sec and --stride-sec must be greater than 0")
    if not 0 <= args.min_label_ratio <= 1:
        raise RuntimeError("--min-label-ratio must be between 0 and 1")

    ensure_ultralytics_config_dir(PROJECT_ROOT)
    os.environ.setdefault("YOLO_VERBOSE", "False")

    video_dir = resolve_project_path(args.video_dir)
    output_dir = resolve_project_path(args.output_dir)
    model_path = resolve_project_path(args.det_model)
    if not model_path.exists():
        raise RuntimeError(f"Missing detector model: {model_path}")

    annotations = load_annotations(video_dir)
    validate_video_files(annotations, video_dir)
    grouped_annotations: dict[str, list[AnnotationInterval]] = defaultdict(list)
    for interval in annotations:
        grouped_annotations[interval.video].append(interval)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Could not import ultralytics. Install project requirements first.") from exc

    model = YOLO(str(model_path))
    feed_regions = load_feed_regions(resolve_project_path(args.feed_regions))
    water_regions = load_regions(resolve_project_path(args.water_regions))

    all_frame_rows: list[dict[str, object]] = []
    for video_name, intervals in sorted(grouped_annotations.items()):
        rows = extract_video_features(
            video_path=video_dir / video_name,
            intervals=intervals,
            model=model,
            feed_regions=feed_regions,
            water_regions=water_regions,
            sample_fps=args.sample_fps,
            imgsz=args.imgsz,
            conf=args.conf,
            batch_size=max(1, args.batch_size),
            device=args.device,
            feature_history_window=max(1, args.feature_history_window),
            max_seconds=max(0.0, args.max_seconds_per_video),
        )
        all_frame_rows.extend(rows)

    X, label_texts, metadata = build_windows(
        all_frame_rows,
        sample_fps=args.sample_fps,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        min_label_ratio=args.min_label_ratio,
    )
    write_dataset(output_dir, all_frame_rows, annotations, X, label_texts, metadata, args)

    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    print(f"Dataset: {output_dir.resolve()}")
    print(f"Frames: {summary['frames']}, windows: {summary['windows']}")
    print(f"Train windows: {summary['train_windows']}, val windows: {summary['val_windows']}")
    print(f"Class counts: {summary['class_counts']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Temporal dataset build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
