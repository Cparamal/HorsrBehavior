from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSignal:
    behavior: str
    reason: str
    confidence: float
    strength: str


def classify_pose_rule(
    row: dict[str, float | int],
    head_low_threshold: float = -0.18,
    feed_distance_threshold: float = 0.10,
    water_distance_threshold: float = 0.08,
    lying_flatness_threshold: float = 0.06,
    lying_aspect_ratio_threshold: float = 0.55,
) -> RuleSignal:
    if int(row.get("pose_exists", 0)) == 0:
        return RuleSignal("unknown", "no_pose", 0.0, "weak")

    flatness = float(row.get("backline_flatness", -1.0))
    aspect = float(row.get("horse_box_aspect_ratio", -1.0))
    if 0.0 <= flatness <= lying_flatness_threshold and 0.0 <= aspect <= lying_aspect_ratio_threshold:
        return RuleSignal("lying", "flat_back_low_box", 0.92, "strong")

    nose_backline = float(row.get("nose_backline_y_diff", 0.0))
    head_low = nose_backline <= head_low_threshold
    if not head_low:
        return RuleSignal("standing", "pose_default", 0.55, "weak")

    water_distance = float(row.get("nose_to_water_distance", -1.0))
    if int(row.get("water_exists", 0)) or int(row.get("nose_in_water_region", 0)):
        if 0.0 <= water_distance <= water_distance_threshold or int(row.get("nose_in_water_region", 0)):
            return RuleSignal("drinking", "nose_near_water", 0.92, "strong")

    feed_distance = float(row.get("nose_to_feed_distance", -1.0))
    if int(row.get("grass_exists", 0)) or int(row.get("nose_in_feed_region", 0)):
        if 0.0 <= feed_distance <= feed_distance_threshold or int(row.get("nose_in_feed_region", 0)):
            return RuleSignal("eating", "nose_near_feed", 0.90, "strong")

    return RuleSignal("head_down", "nose_below_backline", 0.72, "medium")
