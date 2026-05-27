from dataclasses import dataclass, field

from horse_behavior.pose_hybrid_fusion import FusedPoseDecision


@dataclass(frozen=True)
class StateMachineConfig:
    enter_frames: dict[str, int] = field(
        default_factory=lambda: {"eating": 6, "drinking": 6, "head_down": 4, "lying": 8, "standing": 8}
    )
    exit_frames: dict[str, int] = field(default_factory=lambda: {"eating": 12, "drinking": 12, "head_down": 8, "lying": 20})
    default_behavior: str = "standing"


@dataclass(frozen=True)
class StableBehaviorDecision:
    stable_behavior: str
    raw_behavior: str
    confidence: float
    pending_behavior: str
    state_age_frames: int
    transition_reason: str


class BehaviorStateMachine:
    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self.stable_behavior = self.config.default_behavior
        self.pending_behavior = self.config.default_behavior
        self.pending_count = 0
        self.exit_count = 0
        self.state_age_frames = 0

    def update(self, decision: FusedPoseDecision) -> StableBehaviorDecision:
        raw = decision.behavior
        if raw == "unknown":
            raw = self.config.default_behavior
        transition = "held"

        if raw == self.stable_behavior:
            self.pending_behavior = raw
            self.pending_count = 0
            self.exit_count = 0
            self.state_age_frames += 1
            return self._result(decision, transition)

        if self.stable_behavior != self.config.default_behavior and raw == self.config.default_behavior:
            self.exit_count += 1
            needed = self.config.exit_frames.get(self.stable_behavior, self.config.enter_frames.get(raw, 1))
            if self.exit_count >= needed:
                old = self.stable_behavior
                self.stable_behavior = self.config.default_behavior
                self.pending_behavior = self.config.default_behavior
                self.pending_count = 0
                self.state_age_frames = 0
                self.exit_count = 0
                transition = f"exited:{old}"
            else:
                self.pending_behavior = self.config.default_behavior
                self.pending_count = 0
                self.state_age_frames += 1
            return self._result(decision, transition)

        self.exit_count = 0

        if raw != self.pending_behavior:
            self.pending_behavior = raw
            self.pending_count = 1
        else:
            self.pending_count += 1

        needed = self.config.enter_frames.get(raw, 1)
        if self.pending_count >= needed:
            self.stable_behavior = raw
            self.state_age_frames = 0
            self.exit_count = 0
            transition = f"entered:{raw}"
        else:
            self.state_age_frames += 1
        return self._result(decision, transition)

    def _result(self, decision: FusedPoseDecision, transition: str) -> StableBehaviorDecision:
        return StableBehaviorDecision(
            stable_behavior=self.stable_behavior,
            raw_behavior=decision.behavior,
            confidence=decision.confidence,
            pending_behavior=self.pending_behavior,
            state_age_frames=self.state_age_frames,
            transition_reason=transition,
        )
