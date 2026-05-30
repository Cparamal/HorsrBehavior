import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from horse_behavior.infer_behavior import (
    DEFAULT_MODEL,
    Detection,
    add_video_segment_args,
    box_area,
    compute_video_frame_range,
    detections_from_result,
    effective_model_conf,
    resize_for_display,
    seek_video_to_frame,
)
from horse_behavior.train_yolo import ensure_ultralytics_config_dir


DEFAULT_POSE_MODEL = ".cache/superanimal_quadruped/superanimal_quadruped_rtmpose_s.pt"
DEFAULT_OUTPUT = "outputs/pose_superanimal_yolo.mp4"
DEFAULT_CSV = "outputs/pose_superanimal_yolo.csv"
SUPERANIMAL_NAME = "superanimal_quadruped"
POSE_MODEL_NAME = "rtmpose_s"


@dataclass(frozen=True)
class PoseFrameResult:
    frame_index: int
    time_sec: float
    horse: Detection | None
    keypoints: np.ndarray | None
    pose_latency_ms: float
    yolo_latency_ms: float
    reused_pose: bool


def select_pose_horse(detections: list[Detection]) -> Detection | None:
    horses = [d for d in detections if d.name == "horse"]
    if not horses:
        return None
    return max(horses, key=lambda d: (d.conf, box_area(d.xyxy)))


def padded_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * max(0.0, padding)
    pad_y = height * max(0.0, padding)
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width - 1), x2 + pad_x),
        min(float(image_height - 1), y2 + pad_y),
    )


def make_pose_context(horse: Detection, image_size: tuple[int, int], padding: float) -> dict[str, np.ndarray]:
    box = padded_box(horse.xyxy, image_size, padding)
    return {
        "bboxes": np.asarray([box], dtype=np.float32),
        "bbox_scores": np.asarray([horse.conf], dtype=np.float32),
    }


def load_pose_runner(args):
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    from deeplabcut.pose_estimation_pytorch.modelzoo import load_super_animal_config

    pose_checkpoint = Path(args.pose_model)
    if not pose_checkpoint.exists():
        raise RuntimeError(f"Missing pose model checkpoint: {pose_checkpoint}")

    model_config = load_super_animal_config(
        super_animal=SUPERANIMAL_NAME,
        model_name=POSE_MODEL_NAME,
        detector_name="fasterrcnn_resnet50_fpn_v2",
        max_individuals=1,
        device=args.device,
    )
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_config,
        snapshot_path=str(pose_checkpoint),
        max_individuals=1,
        batch_size=max(1, int(args.pose_batch_size)),
        device=args.device,
        detector_path=None,
    )
    if detector_runner is not None:
        raise RuntimeError("Expected SuperAnimal pose runner without detector, but a detector runner was created.")
    return pose_runner


def run_pose_for_frame(
    pose_runner,
    frame,
    horse: Detection | None,
    image_size: tuple[int, int],
    bbox_padding: float,
) -> tuple[np.ndarray | None, float]:
    if horse is None:
        return None, 0.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    context = make_pose_context(horse, image_size, bbox_padding)
    started = time.perf_counter()
    prediction = pose_runner.inference([(frame_rgb, context)])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not prediction:
        return None, latency_ms

    keypoints = prediction[0].get("bodyparts")
    if keypoints is None or len(keypoints) == 0:
        return None, latency_ms
    return np.asarray(keypoints[0], dtype=np.float32), latency_ms


def draw_pose(
    frame,
    horse: Detection | None,
    keypoints: np.ndarray | None,
    pose_threshold: float,
) -> None:
    if horse is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in horse.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 180, 80), 2)
        cv2.putText(
            frame,
            f"horse {horse.conf:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 180, 80),
            2,
            cv2.LINE_AA,
        )

    if keypoints is None:
        return

    for index, (x, y, score) in enumerate(keypoints):
        if score < pose_threshold:
            continue
        color = (
            int(80 + (index * 37) % 175),
            int(80 + (index * 67) % 175),
            int(80 + (index * 97) % 175),
        )
        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(round(x)), int(round(y))), 5, (0, 0, 0), 1, cv2.LINE_AA)


def keypoints_to_json(keypoints: np.ndarray | None) -> str:
    if keypoints is None:
        return "[]"
    values = [
        {"index": int(i), "x": float(x), "y": float(y), "score": float(score)}
        for i, (x, y, score) in enumerate(keypoints)
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def write_csv_header(writer) -> None:
    writer.writerow(
        [
            "frame",
            "time_sec",
            "horse_conf",
            "horse_bbox_xyxy",
            "pose_latency_ms",
            "yolo_latency_ms",
            "reused_pose",
            "keypoints",
        ]
    )


def write_csv_row(writer, result: PoseFrameResult) -> None:
    if result.horse is None:
        horse_conf = ""
        horse_box = ""
    else:
        horse_conf = f"{result.horse.conf:.4f}"
        horse_box = json.dumps([float(v) for v in result.horse.xyxy], separators=(",", ":"))
    writer.writerow(
        [
            result.frame_index,
            f"{result.time_sec:.3f}",
            horse_conf,
            horse_box,
            f"{result.pose_latency_ms:.3f}",
            f"{result.yolo_latency_ms:.3f}",
            int(result.reused_pose),
            keypoints_to_json(result.keypoints),
        ]
    )


def process_frame(
    frame,
    frame_index: int,
    fps: float,
    yolo_model,
    pose_runner,
    args,
    last_keypoints: np.ndarray | None,
) -> tuple[PoseFrameResult, np.ndarray | None]:
    height, width = frame.shape[:2]
    started = time.perf_counter()
    result = yolo_model.predict(frame, imgsz=args.imgsz, conf=effective_model_conf(args), verbose=False)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yolo_latency_ms = (time.perf_counter() - started) * 1000.0
    detections = detections_from_result(result, effective_model_conf(args))
    horse = select_pose_horse(detections)

    should_run_pose = args.pose_every_n_frames <= 1 or frame_index % args.pose_every_n_frames == 0
    reused_pose = False
    pose_latency_ms = 0.0
    keypoints = last_keypoints
    if should_run_pose:
        keypoints, pose_latency_ms = run_pose_for_frame(
            pose_runner=pose_runner,
            frame=frame,
            horse=horse,
            image_size=(width, height),
            bbox_padding=args.bbox_padding,
        )
    else:
        reused_pose = keypoints is not None

    frame_result = PoseFrameResult(
        frame_index=frame_index,
        time_sec=(frame_index / fps) if fps else 0.0,
        horse=horse,
        keypoints=keypoints,
        pose_latency_ms=pose_latency_ms,
        yolo_latency_ms=yolo_latency_ms,
        reused_pose=reused_pose,
    )
    return frame_result, keypoints


def run_images(args, yolo_model, pose_runner) -> int:
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"No input images found: {source}")

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

    try:
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
                continue
            result, _ = process_frame(
                frame=frame,
                frame_index=index,
                fps=1.0,
                yolo_model=yolo_model,
                pose_runner=pose_runner,
                args=args,
                last_keypoints=None,
            )
            draw_pose(frame, result.horse, result.keypoints, args.pose_threshold)
            out_path = output_dir / f"pose_{image_path.name}"
            cv2.imwrite(str(out_path), frame)
            if csv_writer:
                write_csv_row(csv_writer, result)
            print(f"{image_path.name}: horse={result.horse.conf:.3f}" if result.horse else f"{image_path.name}: no horse")
    finally:
        if csv_file:
            csv_file.close()

    print(f"Output images: {output_dir.resolve()}")
    if args.csv:
        print(f"Frame CSV: {Path(args.csv).resolve()}")
    return 0


def run_video(args, yolo_model, pose_runner) -> int:
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_range = compute_video_frame_range(
        total_frames=total_frames,
        fps=fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_frames=args.max_frames,
    )
    limit = frame_range.frame_limit
    seek_video_to_frame(capture, frame_range)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

    processed_frames = 0
    frame_index = frame_range.start_frame
    last_keypoints = None
    latencies = []
    started_all = time.perf_counter()
    try:
        while True:
            if limit is not None and processed_frames >= limit:
                break
            ok, frame = capture.read()
            if not ok:
                break

            result, last_keypoints = process_frame(
                frame=frame,
                frame_index=frame_index,
                fps=fps,
                yolo_model=yolo_model,
                pose_runner=pose_runner,
                args=args,
                last_keypoints=last_keypoints,
            )
            latencies.append(result.yolo_latency_ms + result.pose_latency_ms)
            draw_pose(frame, result.horse, result.keypoints, args.pose_threshold)
            writer.write(frame)
            if csv_writer:
                write_csv_row(csv_writer, result)

            if not args.no_display:
                cv2.imshow("SuperAnimal YOLO-box Pose", resize_for_display(frame, args.display_scale))
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            processed_frames += 1
            frame_index += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames}/{limit if limit is not None else '?'} frames")
    finally:
        capture.release()
        writer.release()
        if csv_file:
            csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_all
    throughput = processed_frames / elapsed if elapsed > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    print(f"Output video: {output.resolve()}")
    if args.csv:
        print(f"Frame CSV: {Path(args.csv).resolve()}")
    print(f"Processed frames: {processed_frames}")
    print(f"Throughput FPS: {throughput:.2f}")
    print(f"Mean YOLO+pose latency: {mean_latency:.2f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SuperAnimal-Quadruped pose inference using YOLO horse boxes.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO horse detector weights path.")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help="SuperAnimal rtmpose_s checkpoint path.")
    parser.add_argument("--source", default="video/stable_20260523_105109.mp4", help="Input video, image, or image directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output video path or image output directory.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Optional frame-level CSV path. Empty disables CSV.")
    parser.add_argument("--mode", choices=["auto", "video", "images"], default="auto", help="Input mode.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device for pose inference.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--model-conf", type=float, default=0.05, help="Low-level YOLO candidate threshold.")
    parser.add_argument("--min-grass-conf", type=float, default=0.25, help="Compatibility option for effective YOLO conf.")
    parser.add_argument("--min-feed-region-grass-conf", type=float, default=0.25, help="Compatibility option for effective YOLO conf.")
    parser.add_argument("--min-overlap-grass-conf", type=float, default=0.25, help="Compatibility option for effective YOLO conf.")
    parser.add_argument("--bbox-padding", type=float, default=0.10, help="Padding ratio applied around the selected horse box.")
    parser.add_argument("--pose-threshold", type=float, default=0.15, help="Minimum keypoint score to draw.")
    parser.add_argument("--pose-every-n-frames", type=int, default=1, help="Run pose every N video frames and reuse keypoints between runs.")
    parser.add_argument("--pose-batch-size", type=int, default=1, help="Pose model batch size.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum video frames to process. 0 means full selected segment.")
    add_video_segment_args(parser)
    parser.add_argument("--no-display", action="store_true", help="Do not open a realtime preview window.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Realtime preview scale.")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    ensure_ultralytics_config_dir(project_root)

    for path_value, label in ((args.model, "YOLO model"), (args.pose_model, "pose model"), (args.source, "source")):
        path = Path(path_value)
        if not path.exists():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import ultralytics: {exc}", file=sys.stderr)
        return 1

    yolo_model = YOLO(str(Path(args.model)))
    pose_runner = load_pose_runner(args)

    source = Path(args.source)
    mode = args.mode
    if mode == "auto":
        mode = "images" if source.is_dir() or source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else "video"

    if mode == "images":
        return run_images(args, yolo_model, pose_runner)
    return run_video(args, yolo_model, pose_runner)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"SuperAnimal YOLO-box pose inference failed: {exc}", file=sys.stderr)
        return 1
