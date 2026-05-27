from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from horse_behavior.pose_hybrid_rules import RuleSignal


@dataclass(frozen=True)
class ModelSignal:
    behavior: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class FusionConfig:
    strong_rule_threshold: float = 0.85
    model_high_threshold: float = 0.80
    agreement_bonus: float = 0.08
    weak_model_threshold: float = 0.45


@dataclass(frozen=True)
class FusedPoseDecision:
    behavior: str
    confidence: float
    source: str
    reason: str
    rule_behavior: str
    model_behavior: str
    model_probabilities: dict[str, float]


def load_feature_columns(path: Path) -> list[str]:
    columns = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not columns:
        raise RuntimeError(f"No feature columns found in {path}")
    return columns


def predict_pose_lightgbm(
    model,
    label_encoder,
    feature_row: dict[str, float | int],
    feature_columns: list[str],
) -> ModelSignal:
    missing = [column for column in feature_columns if column not in feature_row]
    if missing:
        raise RuntimeError(f"Feature row is missing trained columns: {', '.join(missing)}")

    frame = pd.DataFrame(
        [{column: feature_row[column] for column in feature_columns}],
        columns=feature_columns,
    ).astype(float)
    probabilities = model.predict_proba(frame)[0]
    classes = [str(value) for value in label_encoder.classes_]
    probability_map = {label: float(probability) for label, probability in zip(classes, probabilities)}
    behavior = max(probability_map.items(), key=lambda item: item[1])[0]
    return ModelSignal(
        behavior=behavior,
        confidence=probability_map[behavior],
        probabilities=probability_map,
    )


def fuse_rule_and_model(
    rule: RuleSignal,
    model: ModelSignal | None,
    config: FusionConfig | None = None,
) -> FusedPoseDecision:
    config = config or FusionConfig()
    if model is None:
        return FusedPoseDecision(
            rule.behavior,
            rule.confidence,
            "rules_only",
            rule.reason,
            rule.behavior,
            "unknown",
            {},
        )

    if rule.behavior == model.behavior and rule.behavior != "unknown":
        confidence = min(1.0, max(rule.confidence, model.confidence) + config.agreement_bonus)
        return FusedPoseDecision(
            rule.behavior,
            confidence,
            "agreement",
            rule.reason,
            rule.behavior,
            model.behavior,
            model.probabilities,
        )

    if rule.strength == "strong" and rule.confidence >= config.strong_rule_threshold:
        return FusedPoseDecision(
            rule.behavior,
            rule.confidence,
            "strong_rule",
            rule.reason,
            rule.behavior,
            model.behavior,
            model.probabilities,
        )

    if model.confidence >= config.model_high_threshold and rule.strength in {"medium", "weak"}:
        return FusedPoseDecision(
            model.behavior,
            model.confidence,
            "model",
            "high_model_confidence",
            rule.behavior,
            model.behavior,
            model.probabilities,
        )

    if model.confidence < config.weak_model_threshold and rule.behavior != "unknown":
        return FusedPoseDecision(
            rule.behavior,
            max(rule.confidence, model.confidence),
            "rule_fallback",
            rule.reason,
            rule.behavior,
            model.behavior,
            model.probabilities,
        )

    return FusedPoseDecision(
        model.behavior,
        model.confidence,
        "model",
        "default_model",
        rule.behavior,
        model.behavior,
        model.probabilities,
    )
