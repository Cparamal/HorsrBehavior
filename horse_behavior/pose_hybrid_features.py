import json
import math
from dataclasses import dataclass

import numpy as np

from horse_behavior.infer_behavior import Detection, box_center, point_in_regions


CORE6_NAMES = ["nose", "jaw", "withers", "neck_end", "mid_back", "croup"]
CORE6_INDEX = {name: index for index, name in enumerate(CORE6_NAMES)}

POSE_HYBRID_FEATURE_COLUMNS = [
    "pose_exists",
    "pose_confidence",
    "horse_cx",
    "horse_cy",
    "horse_w",
    "horse_h",
    "horse_box_aspect_ratio",
    "nose_visible",
    "jaw_visible",
    "withers_visible",
    "neck_end_visible",
    "mid_back_visible",
    "croup_visible",
    "keypoint_conf_mean",
    "nose_box_x_ratio",
    "nose_box_y_ratio",
    "jaw_box_x_ratio",
    "jaw_box_y_ratio",
    "nose_backline_y_diff",
    "head_vector_angle",
    "backline_angle",
    "backline_flatness",
    "grass_exists",
    "grass_conf",
    "water_exists",
    "water_conf",
    "nose_to_feed_distance",
    "nose_to_water_distance",
    "nose_in_feed_region",
    "nose_in_water_region",
    "nose_speed",
    "neck_end_speed",
    "recent_pose_missing_count",
]


@dataclass(frozen=True)
class Core6Pose:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray


@dataclass(frozen=True)
class PoseFeatureMemory:
    nose: tuple[float, float] | None
    neck_end: tuple[float, float] | None
    frame_index: int | None
    pose_missing_count: int = 0


@dataclass(frozen=True)
class PoseFeatureResult:
    row: dict[str, float | int]
    memory: PoseFeatureMemory
    horse: Detection | None
    keypoints_json: str


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


def pose_instances_from_result(result, min_pose_conf: float) -> list[Core6Pose]:
    if not hasattr(result, "boxes") or not hasattr(result, "keypoints"):
        return []

    boxes_xyxy = _as_numpy(result.boxes.xyxy)
    box_conf = _as_numpy(result.boxes.conf)
    keypoint_xy = _as_numpy(result.keypoints.xy)
    keypoint_conf = None
    if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
        keypoint_conf = _as_numpy(result.keypoints.conf)

    poses = []
    for index, (bbox, confidence, xy) in enumerate(zip(boxes_xyxy, box_conf, keypoint_xy)):
        if xy.shape[0] != len(CORE6_NAMES):
            raise RuntimeError(f"Expected {len(CORE6_NAMES)} keypoints, got {xy.shape[0]}")
        if float(confidence) < min_pose_conf:
            continue
        conf = keypoint_conf[index] if keypoint_conf is not None else np.full(len(CORE6_NAMES), -1.0, dtype=np.float32)
        keypoints = np.column_stack((xy, conf)).astype(np.float32, copy=False)
        poses.append(
            Core6Pose(
                bbox_xyxy=tuple(float(v) for v in bbox),
                confidence=float(confidence),
                keypoints=keypoints,
            )
        )
    return poses


def select_main_pose(poses: list[Core6Pose]) -> Core6Pose | None:
    if not poses:
        return None
    return max(poses, key=lambda pose: (pose.confidence, _box_area(pose.bbox_xyxy)))


def keypoints_to_json(pose: Core6Pose | None) -> str:
    if pose is None:
        return "[]"
    payload = []
    for name, point in zip(CORE6_NAMES, pose.keypoints):
        payload.append(
            {
                "name": name,
                "x": float(point[0]),
                "y": float(point[1]),
                "conf": float(point[2]),
            }
        )
    return json.dumps(payload)


def extract_pose_hybrid_features(
    pose: Core6Pose | None,
    detections: list[Detection],
    image_size: tuple[int, int],
    feed_regions: list[tuple[float, float, float, float]],
    water_regions: list[tuple[float, float, float, float]],
    frame_index: int,
    fps: float,
    previous: PoseFeatureMemory | None,
    keypoint_threshold: float = 0.35,
) -> PoseFeatureResult:
    if pose is None:
        missing_count = (previous.pose_missing_count if previous is not None else 0) + 1
        row = _empty_feature_row()
        row["recent_pose_missing_count"] = missing_count
        return PoseFeatureResult(
            row=row,
            memory=PoseFeatureMemory(None, None, frame_index, missing_count),
            horse=None,
            keypoints_json="[]",
        )

    image_width, image_height = image_size
    x1, y1, x2, y2 = pose.bbox_xyxy
    horse_width = max(1.0, x2 - x1)
    horse_height = max(1.0, y2 - y1)
    horse_scale = max(horse_width, horse_height)
    horse = Detection("horse", float(pose.confidence), pose.bbox_xyxy)
    nose = _visible_point(pose, "nose", keypoint_threshold)
    jaw = _visible_point(pose, "jaw", keypoint_threshold)
    neck_end = _visible_point(pose, "neck_end", keypoint_threshold)
    grass = _nearest_detection_to_point([d for d in detections if d.name == "grass"], nose)
    water = _nearest_detection_to_point([d for d in detections if d.name == "water"], nose)

    row = _empty_feature_row()
    row.update(
        {
            "pose_exists": 1,
            "pose_confidence": float(pose.confidence),
            "horse_cx": box_center(pose.bbox_xyxy)[0] / max(1.0, float(image_width)),
            "horse_cy": box_center(pose.bbox_xyxy)[1] / max(1.0, float(image_height)),
            "horse_w": max(0.0, x2 - x1) / max(1.0, float(image_width)),
            "horse_h": max(0.0, y2 - y1) / max(1.0, float(image_height)),
            "horse_box_aspect_ratio": horse_width / horse_height,
            "keypoint_conf_mean": float(np.mean(pose.keypoints[:, 2])),
            "grass_exists": int(grass is not None),
            "grass_conf": float(grass.conf) if grass is not None else -1.0,
            "water_exists": int(water is not None),
            "water_conf": float(water.conf) if water is not None else -1.0,
            "recent_pose_missing_count": 0,
        }
    )

    for name in CORE6_NAMES:
        row[f"{name}_visible"] = int(_visible_point(pose, name, keypoint_threshold) is not None)

    if nose is not None:
        row.update(
            {
                "nose_box_x_ratio": (nose[0] - x1) / horse_width,
                "nose_box_y_ratio": (nose[1] - y1) / horse_height,
                "nose_in_feed_region": int(point_in_regions(nose, feed_regions)),
                "nose_in_water_region": int(point_in_regions(nose, water_regions)),
                "nose_to_feed_distance": _normalized_distance_to_regions_or_box(nose, feed_regions, grass, horse_scale),
                "nose_to_water_distance": _normalized_distance_to_regions_or_box(nose, water_regions, water, horse_scale),
                "nose_speed": _point_speed(nose, previous.nose if previous else None, frame_index, previous.frame_index if previous else None, fps),
            }
        )
        backline_y = _backline_y_at_x(pose, nose[0], keypoint_threshold)
        if backline_y is not None:
            row["nose_backline_y_diff"] = (backline_y - nose[1]) / horse_height

    if jaw is not None:
        row.update(
            {
                "jaw_box_x_ratio": (jaw[0] - x1) / horse_width,
                "jaw_box_y_ratio": (jaw[1] - y1) / horse_height,
            }
        )
    if nose is not None and jaw is not None:
        row["head_vector_angle"] = _line_angle(jaw, nose)

    withers = _visible_point(pose, "withers", keypoint_threshold)
    mid_back = _visible_point(pose, "mid_back", keypoint_threshold)
    croup = _visible_point(pose, "croup", keypoint_threshold)
    if withers is not None and croup is not None:
        row["backline_angle"] = _line_angle(withers, croup)
    row["backline_flatness"] = _backline_flatness([withers, mid_back, croup])

    if neck_end is not None:
        row["neck_end_speed"] = _point_speed(
            neck_end,
            previous.neck_end if previous else None,
            frame_index,
            previous.frame_index if previous else None,
            fps,
        )

    return PoseFeatureResult(
        row={column: row[column] for column in POSE_HYBRID_FEATURE_COLUMNS},
        memory=PoseFeatureMemory(nose, neck_end, frame_index, 0),
        horse=horse,
        keypoints_json=keypoints_to_json(pose),
    )


def _empty_feature_row() -> dict[str, float | int]:
    row = {column: -1.0 for column in POSE_HYBRID_FEATURE_COLUMNS}
    for column in (
        "pose_exists",
        "nose_visible",
        "jaw_visible",
        "withers_visible",
        "neck_end_visible",
        "mid_back_visible",
        "croup_visible",
        "grass_exists",
        "water_exists",
        "nose_in_feed_region",
        "nose_in_water_region",
        "recent_pose_missing_count",
    ):
        row[column] = 0
    return row


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _visible_point(pose: Core6Pose, name: str, keypoint_threshold: float = 0.35) -> tuple[float, float] | None:
    point = pose.keypoints[CORE6_INDEX[name]]
    if float(point[2]) < keypoint_threshold:
        return None
    return (float(point[0]), float(point[1]))


def _line_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _backline_flatness(points: list[tuple[float, float] | None]) -> float:
    visible = [point for point in points if point is not None]
    if len(visible) < 3:
        return -1.0
    y_values = [point[1] for point in visible]
    x_values = [point[0] for point in visible]
    x_span = max(x_values) - min(x_values)
    if x_span <= 0:
        return -1.0
    return (max(y_values) - min(y_values)) / max(1.0, x_span)


def _nearest_detection_to_point(candidates: list[Detection], point: tuple[float, float] | None) -> Detection | None:
    if not candidates:
        return None
    if point is None:
        return max(candidates, key=lambda detection: (detection.conf, _box_area(detection.xyxy)))
    return min(candidates, key=lambda detection: (_point_box_distance(point, detection.xyxy), -detection.conf))


def _point_box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    nearest_x = min(max(x, x1), x2)
    nearest_y = min(max(y, y1), y2)
    return math.hypot(x - nearest_x, y - nearest_y)


def _point_region_distance(point: tuple[float, float], region: tuple[float, float, float, float]) -> float:
    return _point_box_distance(point, region)


def _distance_to_regions_or_box(
    point: tuple[float, float],
    regions: list[tuple[float, float, float, float]],
    detection: Detection | None,
) -> float:
    distances = [_point_region_distance(point, region) for region in regions]
    if detection is not None:
        distances.append(_point_box_distance(point, detection.xyxy))
    if not distances:
        return -1.0
    return min(distances)


def _normalized_distance_to_regions_or_box(
    point: tuple[float, float],
    regions: list[tuple[float, float, float, float]],
    detection: Detection | None,
    scale: float,
) -> float:
    distance = _distance_to_regions_or_box(point, regions, detection)
    if distance < 0.0:
        return distance
    return distance / max(1.0, scale)


def _point_speed(
    point: tuple[float, float],
    previous_point: tuple[float, float] | None,
    frame_index: int,
    previous_frame_index: int | None,
    fps: float,
) -> float:
    if previous_point is None or previous_frame_index is None or fps <= 0:
        return -1.0
    frame_delta = frame_index - previous_frame_index
    if frame_delta <= 0:
        return -1.0
    return math.hypot(point[0] - previous_point[0], point[1] - previous_point[1]) * fps / frame_delta


def _backline_y_at_x(pose: Core6Pose, x: float, keypoint_threshold: float = 0.35) -> float | None:
    withers = _visible_point(pose, "withers", keypoint_threshold)
    croup = _visible_point(pose, "croup", keypoint_threshold)
    if withers is None or croup is None:
        return None
    if abs(croup[0] - withers[0]) < 1e-9:
        return (withers[1] + croup[1]) / 2.0
    ratio = (x - withers[0]) / (croup[0] - withers[0])
    return withers[1] + ratio * (croup[1] - withers[1])
