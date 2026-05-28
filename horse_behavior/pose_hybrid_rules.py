import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSignal:
    behavior: str
    reason: str
    confidence: float
    strength: str


def _flag(value: object) -> bool:
    number = _finite_number(value)
    return number is not None and int(number) == 1


def _finite_number(value: object, missing_sentinel: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if missing_sentinel is not None and math.isclose(number, missing_sentinel):
        return None
    return number


def classify_pose_rule(
    row: dict[str, object],
    head_low_threshold: float = -0.18,
    head_standing_threshold: float = -0.02,
    nose_low_ratio_threshold: float = 0.62,
    nose_standing_ratio_threshold: float = 0.45,
    jaw_low_ratio_threshold: float = 0.58,
    jaw_standing_ratio_threshold: float = 0.48,
    downward_head_angle_threshold: float = 0.70,
    standing_head_angle_threshold: float = 0.20,
    feed_distance_threshold: float = 0.10,
    water_distance_threshold: float = 0.08,
    lying_flatness_threshold: float = 0.06,
    lying_aspect_ratio_threshold: float = 0.55,
) -> RuleSignal:
    if not _flag(row.get("pose_exists", 0)):
        return RuleSignal("unknown", "no_pose", 0.0, "weak")

    flatness = _finite_number(row.get("backline_flatness"))
    aspect = _finite_number(row.get("horse_box_aspect_ratio"))
    if (
        flatness is not None
        and aspect is not None
        and 0.0 <= flatness <= lying_flatness_threshold
        and 0.0 <= aspect <= lying_aspect_ratio_threshold
    ):
        return RuleSignal("lying", "flat_back_low_box", 0.92, "strong")

    nose_backline = _finite_number(row.get("nose_backline_y_diff"), missing_sentinel=-1.0)
    has_backline = True
    if "backline_visible" in row:
        has_backline = _flag(row.get("backline_visible"))
    if not _flag(row.get("nose_visible", 0)) or not has_backline or nose_backline is None:
        return RuleSignal("unknown", "missing_head_pose", 0.0, "weak")

    head_low_score = _head_low_score(
        row,
        nose_backline=nose_backline,
        head_low_threshold=head_low_threshold,
        nose_low_ratio_threshold=nose_low_ratio_threshold,
        jaw_low_ratio_threshold=jaw_low_ratio_threshold,
        downward_head_angle_threshold=downward_head_angle_threshold,
    )
    standing_score = _standing_score(
        row,
        nose_backline=nose_backline,
        head_standing_threshold=head_standing_threshold,
        nose_standing_ratio_threshold=nose_standing_ratio_threshold,
        jaw_standing_ratio_threshold=jaw_standing_ratio_threshold,
        standing_head_angle_threshold=standing_head_angle_threshold,
    )

    head_low = head_low_score >= 3
    if not head_low:
        if standing_score >= 3:
            return RuleSignal("standing", "head_high_score", 0.58, "weak")
        return RuleSignal("unknown", "ambiguous_head_pose", 0.0, "weak")

    water_distance = _finite_number(row.get("nose_to_water_distance"), missing_sentinel=-1.0)
    if _flag(row.get("water_exists", 0)) or _flag(row.get("nose_in_water_region", 0)):
        if (
            water_distance is not None
            and 0.0 <= water_distance <= water_distance_threshold
        ) or _flag(row.get("nose_in_water_region", 0)):
            return RuleSignal("drinking", "nose_near_water", 0.92, "strong")

    feed_distance = _finite_number(row.get("nose_to_feed_distance"), missing_sentinel=-1.0)
    if _flag(row.get("grass_exists", 0)) or _flag(row.get("nose_in_feed_region", 0)):
        if (
            feed_distance is not None
            and 0.0 <= feed_distance <= feed_distance_threshold
        ) or _flag(row.get("nose_in_feed_region", 0)):
            return RuleSignal("eating", "nose_near_feed", 0.90, "strong")

    return RuleSignal("head_down", "head_low_score", 0.74, "medium")


def _head_low_score(
    row: dict[str, object],
    nose_backline: float,
    head_low_threshold: float,
    nose_low_ratio_threshold: float,
    jaw_low_ratio_threshold: float,
    downward_head_angle_threshold: float,
) -> int:
    score = 0
    if nose_backline <= head_low_threshold:
        score += 2

    nose_y = _finite_number(row.get("nose_box_y_ratio"), missing_sentinel=-1.0)
    if nose_y is not None and nose_y >= nose_low_ratio_threshold:
        score += 1

    jaw_y = _finite_number(row.get("jaw_box_y_ratio"), missing_sentinel=-1.0)
    if _flag(row.get("jaw_visible", 0)) and jaw_y is not None and jaw_y >= jaw_low_ratio_threshold:
        score += 1

    head_angle = _finite_number(row.get("head_vector_angle"), missing_sentinel=-1.0)
    if head_angle is not None and head_angle >= downward_head_angle_threshold:
        score += 1

    if _flag(row.get("jaw_visible", 0)):
        score += 1
    return score


def _standing_score(
    row: dict[str, object],
    nose_backline: float,
    head_standing_threshold: float,
    nose_standing_ratio_threshold: float,
    jaw_standing_ratio_threshold: float,
    standing_head_angle_threshold: float,
) -> int:
    score = 0
    if nose_backline >= head_standing_threshold:
        score += 2

    nose_y = _finite_number(row.get("nose_box_y_ratio"), missing_sentinel=-1.0)
    if nose_y is not None and nose_y <= nose_standing_ratio_threshold:
        score += 1

    jaw_y = _finite_number(row.get("jaw_box_y_ratio"), missing_sentinel=-1.0)
    if _flag(row.get("jaw_visible", 0)) and jaw_y is not None and jaw_y <= jaw_standing_ratio_threshold:
        score += 1

    head_angle = _finite_number(row.get("head_vector_angle"), missing_sentinel=-1.0)
    if head_angle is not None and head_angle <= standing_head_angle_threshold:
        score += 1
    return score
