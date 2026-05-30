from dataclasses import dataclass


STRONG_RULE_REASONS = {"lying_horse", "sitting_horse"}
MEDIUM_RULE_REASONS = {"head_low"}
CONTACT_RULE_REASONS = {"grass_distance", "grass_overlap", "water_region_head_low"}
GENERIC_ROI_BEHAVIORS = {"standing", "unknown", "未知", "站立", "鏈煡", "绔欑珛"}


@dataclass(frozen=True)
class RuleSignal:
    behavior: str
    reason: str


@dataclass(frozen=True)
class FusionConfig:
    roi_accept_threshold: float = 0.55
    roi_low_threshold: float = 0.45
    strong_rule_bonus: float = 0.50
    weak_rule_bonus: float = 0.15
    contact_rule_bonus: float = 0.35


@dataclass(frozen=True)
class FusedBehaviorDecision:
    behavior: str
    confidence: float
    source: str
    roi_behavior: str
    roi_confidence: float
    rule_behavior: str
    rule_reason: str


def _rule_bonus(reason: str, config: FusionConfig) -> float:
    if reason in STRONG_RULE_REASONS:
        return config.strong_rule_bonus
    if reason in CONTACT_RULE_REASONS:
        return config.contact_rule_bonus
    return config.weak_rule_bonus


def fuse_roi_and_rules(
    roi_behavior: str,
    roi_confidence: float,
    rule_signal: RuleSignal,
    config: FusionConfig | None = None,
) -> FusedBehaviorDecision:
    config = config or FusionConfig()
    roi_confidence = max(0.0, min(1.0, float(roi_confidence)))

    if rule_signal.reason in STRONG_RULE_REASONS:
        confidence = min(1.0, roi_confidence + config.strong_rule_bonus)
        return FusedBehaviorDecision(
            behavior=rule_signal.behavior,
            confidence=confidence,
            source="strong_rule",
            roi_behavior=roi_behavior,
            roi_confidence=roi_confidence,
            rule_behavior=rule_signal.behavior,
            rule_reason=rule_signal.reason,
        )

    if rule_signal.reason in MEDIUM_RULE_REASONS and roi_behavior in GENERIC_ROI_BEHAVIORS:
        confidence = min(1.0, max(roi_confidence, config.roi_accept_threshold))
        return FusedBehaviorDecision(
            behavior=rule_signal.behavior,
            confidence=confidence,
            source="medium_rule",
            roi_behavior=roi_behavior,
            roi_confidence=roi_confidence,
            rule_behavior=rule_signal.behavior,
            rule_reason=rule_signal.reason,
        )

    if rule_signal.reason == "water_region_head_low" and roi_behavior in GENERIC_ROI_BEHAVIORS:
        confidence = min(1.0, max(roi_confidence, config.roi_accept_threshold) + config.contact_rule_bonus)
        return FusedBehaviorDecision(
            behavior=rule_signal.behavior,
            confidence=confidence,
            source="contact_rule",
            roi_behavior=roi_behavior,
            roi_confidence=roi_confidence,
            rule_behavior=rule_signal.behavior,
            rule_reason=rule_signal.reason,
        )

    scores = {roi_behavior: roi_confidence}
    rule_score = scores.get(rule_signal.behavior, 0.0) + _rule_bonus(rule_signal.reason, config)
    scores[rule_signal.behavior] = rule_score
    best_behavior, best_score = max(scores.items(), key=lambda item: item[1])

    if rule_signal.reason in CONTACT_RULE_REASONS and roi_confidence < config.roi_accept_threshold:
        return FusedBehaviorDecision(
            behavior=rule_signal.behavior,
            confidence=min(1.0, max(rule_score, roi_confidence)),
            source="rule_boost",
            roi_behavior=roi_behavior,
            roi_confidence=roi_confidence,
            rule_behavior=rule_signal.behavior,
            rule_reason=rule_signal.reason,
        )

    if roi_confidence < config.roi_low_threshold and rule_signal.behavior:
        return FusedBehaviorDecision(
            behavior=rule_signal.behavior,
            confidence=max(rule_score, roi_confidence),
            source="rule_fallback",
            roi_behavior=roi_behavior,
            roi_confidence=roi_confidence,
            rule_behavior=rule_signal.behavior,
            rule_reason=rule_signal.reason,
        )

    if best_behavior == roi_behavior and roi_confidence >= config.roi_accept_threshold:
        source = "roi"
    elif best_behavior == rule_signal.behavior:
        source = "rule_boost"
    else:
        source = "roi"

    return FusedBehaviorDecision(
        behavior=best_behavior,
        confidence=min(1.0, float(best_score)),
        source=source,
        roi_behavior=roi_behavior,
        roi_confidence=roi_confidence,
        rule_behavior=rule_signal.behavior,
        rule_reason=rule_signal.reason,
    )
