from dataclasses import dataclass

import numpy as np

from horse_behavior.infer_behavior import Detection, select_largest_box


@dataclass(frozen=True)
class BehaviorRoi:
    image: np.ndarray
    box: tuple[int, int, int, int]
    source: str
    selected: Detection | None


def expand_and_clip_box(
    xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding_ratio: float = 0.15,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * max(0.0, padding_ratio)
    pad_y = height * max(0.0, padding_ratio)

    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(image_width, int(np.ceil(x2 + pad_x)))
    bottom = min(image_height, int(np.ceil(y2 + pad_y)))

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def select_behavior_roi_box(detections: list[Detection]) -> Detection | None:
    horse = select_largest_box(detections, "horse")
    if horse is not None:
        return horse
    return select_largest_box(detections, "lying_horse")


def crop_behavior_roi(
    frame: np.ndarray,
    detections: list[Detection],
    padding_ratio: float = 0.15,
) -> BehaviorRoi:
    height, width = frame.shape[:2]
    selected = select_behavior_roi_box(detections)
    if selected is None:
        return BehaviorRoi(
            image=frame.copy(),
            box=(0, 0, width, height),
            source="full_frame",
            selected=None,
        )

    box = expand_and_clip_box(selected.xyxy, (width, height), padding_ratio)
    x1, y1, x2, y2 = box
    return BehaviorRoi(
        image=frame[y1:y2, x1:x2].copy(),
        box=box,
        source="detected",
        selected=selected,
    )
