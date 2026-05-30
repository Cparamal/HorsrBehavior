import math
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

from horse_behavior.infer_behavior import (
    DEFAULT_MIN_WATER_HEAD_OVERLAP_RATIO,
    Detection,
    box_area,
    box_center,
    head_low_in_horse,
    head_near_horse_front_edge,
    head_near_regions,
    intersection_area,
    select_best_head,
    select_largest_box,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_HEAD_LOW_RATIO = 0.58
DEFAULT_REGION_NEAR_RATIO = 0.12
DEFAULT_GRASS_OVERLAP_RATIO = 0.08
DEFAULT_GRASS_DISTANCE_INSIDE_RATIO = 0.15
DEFAULT_GRASS_DISTANCE_OUTSIDE_RATIO = 0.12
DEFAULT_TEMPORAL_WINDOW = 5

FEATURE_COLUMNS = [
    "split",
    "image",
    "label",
    "horse_exists",
    "horse_conf",
    "head_exists",
    "head_conf",
    "grass_exists",
    "grass_conf",
    "water_exists",
    "water_conf",
    "lying_horse_exists",
    "lying_horse_conf",
    "horse_cx",
    "horse_cy",
    "horse_w",
    "horse_h",
    "head_cx",
    "head_cy",
    "head_w",
    "head_h",
    "grass_cx",
    "grass_cy",
    "grass_w",
    "grass_h",
    "water_cx",
    "water_cy",
    "water_w",
    "water_h",
    "lying_horse_cx",
    "lying_horse_cy",
    "lying_horse_w",
    "lying_horse_h",
    "head_rel_x",
    "head_rel_y",
    "head_in_horse",
    "head_out_left",
    "head_out_right",
    "head_out_top",
    "head_out_bottom",
    "head_to_grass_dist",
    "head_to_water_dist",
    "head_grass_overlap",
    "head_water_overlap",
    "grass_in_feed_region",
    "head_area_ratio_to_horse",
    "grass_area_ratio_to_horse",
    "water_area_ratio_to_horse",
    "head_y_ratio",
    "head_front_edge_ratio",
    "head_bottom_to_horse_bottom",
    "grass_count",
    "water_count",
    "max_grass_conf",
    "max_water_conf",
    "nearest_grass_conf",
    "nearest_water_conf",
    "min_head_grass_dist",
    "min_head_water_dist",
    "max_head_grass_overlap",
    "max_head_water_overlap",
    "head_low_in_horse",
    "head_near_horse_front_edge",
    "detected_water_large_overlap",
    "detected_grass_overlap",
    "head_near_feed_region",
    "head_near_water_region",
    "water_region_head_low",
    "grass_distance_rule_hit",
    "water_overlap_rule_hit",
    "head_in_feed_region",
    "head_dist_to_feed_region",
    "grass_dist_to_feed_region",
    "head_in_water_region",
    "head_dist_to_water_region",
    "water_dist_to_water_region",
    "detection_count",
    "head_rel_y_mean_5",
    "head_rel_y_delta_5",
    "head_water_dist_min_5",
    "head_grass_dist_min_5",
    "water_overlap_mean_5",
    "grass_overlap_mean_5",
    "horse_box_stability_5",
]


class BehaviorFeatureHistory:
    def __init__(self, window_size: int = DEFAULT_TEMPORAL_WINDOW):
        self.window_size = max(1, int(window_size))
        self.history: deque[dict[str, float | int | str]] = deque(maxlen=self.window_size)

    def update(self, row: dict[str, float | int | str]) -> dict[str, float]:
        snapshot = {
            "head_rel_y": float(row["head_rel_y"]),
            "head_exists": int(row["head_exists"]),
            "head_to_water_dist": float(row["head_to_water_dist"]),
            "head_to_grass_dist": float(row["head_to_grass_dist"]),
            "head_water_overlap": float(row["head_water_overlap"]),
            "head_grass_overlap": float(row["head_grass_overlap"]),
            "horse_exists": int(row["horse_exists"]),
            "horse_cx": float(row["horse_cx"]),
            "horse_cy": float(row["horse_cy"]),
            "horse_w": float(row["horse_w"]),
            "horse_h": float(row["horse_h"]),
        }
        self.history.append(snapshot)
        return _temporal_features(list(self.history))


def _missing_box_features(prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}_exists": 0,
        f"{prefix}_conf": -1,
        f"{prefix}_cx": -1,
        f"{prefix}_cy": -1,
        f"{prefix}_w": -1,
        f"{prefix}_h": -1,
    }


def _box_features(prefix: str, detection: Detection | None, image_size: tuple[int, int]) -> dict[str, float | int]:
    if detection is None:
        return _missing_box_features(prefix)

    image_width, image_height = image_size
    x1, y1, x2, y2 = detection.xyxy
    cx, cy = box_center(detection.xyxy)
    return {
        f"{prefix}_exists": 1,
        f"{prefix}_conf": float(detection.conf),
        f"{prefix}_cx": cx / max(1.0, float(image_width)),
        f"{prefix}_cy": cy / max(1.0, float(image_height)),
        f"{prefix}_w": max(0.0, x2 - x1) / max(1.0, float(image_width)),
        f"{prefix}_h": max(0.0, y2 - y1) / max(1.0, float(image_height)),
    }


def _select_nearest_to_head(candidates: list[Detection], head: Detection | None) -> Detection | None:
    if not candidates:
        return None
    if head is None:
        return max(candidates, key=lambda d: (d.conf, box_area(d.xyxy)))

    head_x, head_y = box_center(head.xyxy)
    return min(
        candidates,
        key=lambda d: (
            math.hypot(box_center(d.xyxy)[0] - head_x, box_center(d.xyxy)[1] - head_y),
            -d.conf,
        ),
    )


def _center_inside_regions(center: tuple[float, float], regions: Iterable[tuple[float, float, float, float]]) -> int:
    x, y = center
    for x1, y1, x2, y2 in regions:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 1
    return 0


def _point_distance_to_regions(
    point: tuple[float, float] | None,
    regions: Iterable[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> float:
    regions = list(regions)
    if point is None or not regions:
        return -1
    px, py = point
    distances = []
    for x1, y1, x2, y2 in regions:
        nearest_x = min(max(px, x1), x2)
        nearest_y = min(max(py, y1), y2)
        distances.append(math.hypot(px - nearest_x, py - nearest_y))
    return min(distances) / max(1.0, float(max(image_size)))


def _overlap_ratio(reference: Detection | None, other: Detection | None) -> float:
    if reference is None or other is None:
        return -1
    reference_area = box_area(reference.xyxy)
    if reference_area <= 0:
        return -1
    return intersection_area(reference.xyxy, other.xyxy) / reference_area


def _max_overlap_ratio(reference: Detection | None, candidates: list[Detection]) -> float:
    if reference is None or not candidates:
        return -1
    return max(_overlap_ratio(reference, candidate) for candidate in candidates)


def _normalized_center_distance(a: Detection | None, b: Detection | None, image_size: tuple[int, int]) -> float:
    if a is None or b is None:
        return -1
    ax, ay = box_center(a.xyxy)
    bx, by = box_center(b.xyxy)
    return math.hypot(ax - bx, ay - by) / max(1.0, float(max(image_size)))


def _min_normalized_center_distance(a: Detection | None, candidates: list[Detection], image_size: tuple[int, int]) -> float:
    if a is None or not candidates:
        return -1
    return min(_normalized_center_distance(a, candidate, image_size) for candidate in candidates)


def _area_ratio(numerator: Detection | None, denominator: Detection | None) -> float:
    if numerator is None or denominator is None:
        return -1
    denominator_area = box_area(denominator.xyxy)
    if denominator_area <= 0:
        return -1
    return box_area(numerator.xyxy) / denominator_area


def _max_confidence(candidates: list[Detection]) -> float:
    if not candidates:
        return -1
    return max(float(candidate.conf) for candidate in candidates)


def _valid_values(rows: list[dict[str, float | int | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if float(row[key]) >= 0]


def _mean_valid(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = _valid_values(rows, key)
    if not values:
        return -1
    return sum(values) / len(values)


def _min_valid(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = _valid_values(rows, key)
    if not values:
        return -1
    return min(values)


def _normalized_horse_box(row: dict[str, float | int | str]) -> tuple[float, float, float, float] | None:
    if int(row["horse_exists"]) <= 0:
        return None
    cx = float(row["horse_cx"])
    cy = float(row["horse_cy"])
    width = float(row["horse_w"])
    height = float(row["horse_h"])
    if min(cx, cy, width, height) < 0:
        return None
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = intersection_area(a, b)
    union = box_area(a) + box_area(b) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _horse_box_stability(rows: list[dict[str, float | int | str]]) -> float:
    boxes = [_normalized_horse_box(row) for row in rows]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return -1
    current = boxes[-1]
    return min(_box_iou(current, box) for box in boxes)


def _temporal_features(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    head_rel_y_values = [
        float(row["head_rel_y"])
        for row in rows
        if int(row.get("head_exists", 1)) > 0 and int(row["horse_exists"]) > 0
    ]
    if head_rel_y_values:
        head_rel_y_mean = sum(head_rel_y_values) / len(head_rel_y_values)
        head_rel_y_delta = head_rel_y_values[-1] - head_rel_y_values[0]
    else:
        head_rel_y_mean = -1
        head_rel_y_delta = -1

    return {
        "head_rel_y_mean_5": head_rel_y_mean,
        "head_rel_y_delta_5": head_rel_y_delta,
        "head_water_dist_min_5": _min_valid(rows, "head_to_water_dist"),
        "head_grass_dist_min_5": _min_valid(rows, "head_to_grass_dist"),
        "water_overlap_mean_5": _mean_valid(rows, "head_water_overlap"),
        "grass_overlap_mean_5": _mean_valid(rows, "head_grass_overlap"),
        "horse_box_stability_5": _horse_box_stability(rows),
    }


def extract_behavior_features(
    detections: list[Detection],
    image_size: tuple[int, int],
    split: str = "",
    image: str = "",
    label: str = "",
    feed_regions: list[tuple[float, float, float, float]] | None = None,
    water_regions: list[tuple[float, float, float, float]] | None = None,
    history: BehaviorFeatureHistory | None = None,
) -> dict[str, str | float | int]:
    feed_regions = feed_regions or []
    water_regions = water_regions or []
    image_width, image_height = image_size
    scale = max(1.0, float(max(image_size)))

    horse = select_largest_box(detections, "horse")
    head = select_best_head(detections, horse)
    grass_candidates = [d for d in detections if d.name == "grass"]
    water_candidates = [d for d in detections if d.name == "water"]
    grass = _select_nearest_to_head(grass_candidates, head)
    water = _select_nearest_to_head(water_candidates, head)
    lying_horse = select_largest_box(detections, "lying_horse")

    row: dict[str, str | float | int] = {
        "split": split,
        "image": image,
        "label": label,
    }
    for prefix, detection in (
        ("horse", horse),
        ("head", head),
        ("grass", grass),
        ("water", water),
        ("lying_horse", lying_horse),
    ):
        row.update(_box_features(prefix, detection, image_size))

    if head is not None and horse is not None:
        horse_x1, horse_y1, horse_x2, horse_y2 = horse.xyxy
        horse_width = max(1.0, horse_x2 - horse_x1)
        horse_height = max(1.0, horse_y2 - horse_y1)
        head_x1, head_y1, head_x2, head_y2 = head.xyxy
        head_cx, head_cy = box_center(head.xyxy)
        horse_cx, horse_cy = box_center(horse.xyxy)
        row.update(
            {
                "head_rel_x": (head_cx - horse_cx) / horse_width,
                "head_rel_y": (head_cy - horse_cy) / horse_height,
                "head_in_horse": int(
                    head_x1 >= horse_x1 and head_y1 >= horse_y1 and head_x2 <= horse_x2 and head_y2 <= horse_y2
                ),
                "head_out_left": int(head_x1 < horse_x1),
                "head_out_right": int(head_x2 > horse_x2),
                "head_out_top": int(head_y1 < horse_y1),
                "head_out_bottom": int(head_y2 > horse_y2),
                "head_front_edge_ratio": (head_cx - horse_x1) / horse_width,
                "head_bottom_to_horse_bottom": (horse_y2 - head_y2) / horse_height,
            }
        )
    else:
        row.update(
            {
                "head_rel_x": -1,
                "head_rel_y": -1,
                "head_in_horse": 0,
                "head_out_left": 0,
                "head_out_right": 0,
                "head_out_top": 0,
                "head_out_bottom": 0,
                "head_front_edge_ratio": -1,
                "head_bottom_to_horse_bottom": -1,
            }
        )

    row.update(
        {
            "head_to_grass_dist": _normalized_center_distance(head, grass, image_size),
            "head_to_water_dist": _normalized_center_distance(head, water, image_size),
            "head_grass_overlap": _overlap_ratio(head, grass),
            "head_water_overlap": _overlap_ratio(head, water),
            "grass_in_feed_region": _center_inside_regions(box_center(grass.xyxy), feed_regions) if grass else 0,
            "head_area_ratio_to_horse": _area_ratio(head, horse),
            "grass_area_ratio_to_horse": _area_ratio(grass, horse),
            "water_area_ratio_to_horse": _area_ratio(water, horse),
            "head_y_ratio": box_center(head.xyxy)[1] / max(1.0, float(image_height)) if head else -1,
        }
    )

    head_center = box_center(head.xyxy) if head is not None else None
    grass_center = box_center(grass.xyxy) if grass is not None else None
    water_center = box_center(water.xyxy) if water is not None else None
    head_is_low = int(head_low_in_horse(head, horse, DEFAULT_HEAD_LOW_RATIO)) if head is not None else 0
    head_near_feed = int(head_near_regions(head, feed_regions, DEFAULT_REGION_NEAR_RATIO * scale)) if head else 0
    head_near_water = int(head_near_regions(head, water_regions, DEFAULT_REGION_NEAR_RATIO * scale)) if head else 0
    max_grass_overlap = _max_overlap_ratio(head, grass_candidates)
    max_water_overlap = _max_overlap_ratio(head, water_candidates)

    grass_distance_rule_hit = 0
    if head is not None:
        for candidate in grass_candidates:
            dist = _normalized_center_distance(head, candidate, image_size)
            in_feed = _center_inside_regions(box_center(candidate.xyxy), feed_regions)
            threshold = DEFAULT_GRASS_DISTANCE_INSIDE_RATIO if in_feed else DEFAULT_GRASS_DISTANCE_OUTSIDE_RATIO
            if dist >= 0 and dist <= threshold:
                grass_distance_rule_hit = 1
                break

    row.update(
        {
            "grass_count": len(grass_candidates),
            "water_count": len(water_candidates),
            "max_grass_conf": _max_confidence(grass_candidates),
            "max_water_conf": _max_confidence(water_candidates),
            "nearest_grass_conf": float(grass.conf) if grass else -1,
            "nearest_water_conf": float(water.conf) if water else -1,
            "min_head_grass_dist": _min_normalized_center_distance(head, grass_candidates, image_size),
            "min_head_water_dist": _min_normalized_center_distance(head, water_candidates, image_size),
            "max_head_grass_overlap": max_grass_overlap,
            "max_head_water_overlap": max_water_overlap,
            "head_low_in_horse": head_is_low,
            "head_near_horse_front_edge": int(head_near_horse_front_edge(head, horse)) if head is not None and horse is not None else 0,
            "detected_water_large_overlap": int(max_water_overlap >= DEFAULT_MIN_WATER_HEAD_OVERLAP_RATIO),
            "detected_grass_overlap": int(max_grass_overlap >= DEFAULT_GRASS_OVERLAP_RATIO),
            "head_near_feed_region": head_near_feed,
            "head_near_water_region": head_near_water,
            "water_region_head_low": int(head_near_water and head_is_low),
            "grass_distance_rule_hit": grass_distance_rule_hit,
            "water_overlap_rule_hit": int(max_water_overlap >= DEFAULT_MIN_WATER_HEAD_OVERLAP_RATIO and head_is_low),
            "head_in_feed_region": _center_inside_regions(head_center, feed_regions) if head_center else 0,
            "head_dist_to_feed_region": _point_distance_to_regions(head_center, feed_regions, image_size),
            "grass_dist_to_feed_region": _point_distance_to_regions(grass_center, feed_regions, image_size),
            "head_in_water_region": _center_inside_regions(head_center, water_regions) if head_center else 0,
            "head_dist_to_water_region": _point_distance_to_regions(head_center, water_regions, image_size),
            "water_dist_to_water_region": _point_distance_to_regions(water_center, water_regions, image_size),
            "detection_count": len(detections),
        }
    )

    temporal = history.update(row) if history is not None else _temporal_features([row])
    row.update(temporal)

    return {column: row[column] for column in FEATURE_COLUMNS}


def resolve_image_path(project_root: Path, image_value: str) -> Path:
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return project_root / image_path


def default_image_size_reader(image_path: Path) -> tuple[int, int]:
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = frame.shape[:2]
    return width, height


def make_yolo_predictor(model, imgsz: int, conf: float) -> Callable[[Path], list[Detection]]:
    def predict(image_path: Path) -> list[Detection]:
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        names = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    name=names[class_id],
                    conf=float(box.conf[0]),
                    xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                )
            )
        return detections

    return predict
