import math
from pathlib import Path
from typing import Callable, Iterable

from horse_behavior.infer_behavior import Detection, box_area, box_center, intersection_area, select_best_head, select_largest_box


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
]


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


def _overlap_ratio(reference: Detection | None, other: Detection | None) -> float:
    if reference is None or other is None:
        return -1
    reference_area = box_area(reference.xyxy)
    if reference_area <= 0:
        return -1
    return intersection_area(reference.xyxy, other.xyxy) / reference_area


def _normalized_center_distance(a: Detection | None, b: Detection | None, image_size: tuple[int, int]) -> float:
    if a is None or b is None:
        return -1
    ax, ay = box_center(a.xyxy)
    bx, by = box_center(b.xyxy)
    return math.hypot(ax - bx, ay - by) / max(1.0, float(max(image_size)))


def _area_ratio(numerator: Detection | None, denominator: Detection | None) -> float:
    if numerator is None or denominator is None:
        return -1
    denominator_area = box_area(denominator.xyxy)
    if denominator_area <= 0:
        return -1
    return box_area(numerator.xyxy) / denominator_area


def extract_behavior_features(
    detections: list[Detection],
    image_size: tuple[int, int],
    split: str = "",
    image: str = "",
    label: str = "",
    feed_regions: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, str | float | int]:
    feed_regions = feed_regions or []
    image_width, image_height = image_size

    horse = select_largest_box(detections, "horse")
    head = select_best_head(detections, horse)
    grass = _select_nearest_to_head([d for d in detections if d.name == "grass"], head)
    water = _select_nearest_to_head([d for d in detections if d.name == "water"], head)
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
