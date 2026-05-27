import unittest

from horse_behavior.pose_hybrid_fusion import FusedPoseDecision
from horse_behavior.pose_hybrid_state import BehaviorStateMachine, StateMachineConfig


def decision(behavior, confidence=0.8):
    return FusedPoseDecision(behavior, confidence, "test", "test", behavior, behavior, {behavior: confidence})


class PoseHybridStateTests(unittest.TestCase):
    def test_behavior_enters_after_required_frames(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 2}, exit_frames={"eating": 2}, default_behavior="standing"))

        first = machine.update(decision("eating"))
        second = machine.update(decision("eating"))

        self.assertEqual(first.stable_behavior, "standing")
        self.assertEqual(second.stable_behavior, "eating")
        self.assertEqual(second.transition_reason, "entered:eating")

    def test_behavior_exits_after_required_frames(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 1}, exit_frames={"eating": 2}, default_behavior="standing"))
        machine.update(decision("eating"))

        first = machine.update(decision("standing"))
        second = machine.update(decision("standing"))

        self.assertEqual(first.stable_behavior, "eating")
        self.assertEqual(second.stable_behavior, "standing")
        self.assertEqual(second.transition_reason, "exited:eating")

    def test_exit_resets_pending_behavior_before_reentry(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"eating": 2}, exit_frames={"eating": 1}, default_behavior="standing"))
        machine.update(decision("eating"))
        machine.update(decision("eating"))
        machine.update(decision("standing"))

        state = machine.update(decision("eating"))

        self.assertEqual(state.stable_behavior, "standing")
        self.assertEqual(state.transition_reason, "held")

    def test_exit_requires_consecutive_default_frames(self):
        machine = BehaviorStateMachine(
            StateMachineConfig(
                enter_frames={"eating": 1, "drinking": 2},
                exit_frames={"eating": 2},
                default_behavior="standing",
            )
        )
        machine.update(decision("eating"))

        machine.update(decision("standing"))
        machine.update(decision("drinking"))
        state = machine.update(decision("standing"))

        self.assertEqual(state.stable_behavior, "eating")
        self.assertEqual(state.transition_reason, "held")

    def test_default_frame_resets_pending_non_default_entry(self):
        machine = BehaviorStateMachine(
            StateMachineConfig(
                enter_frames={"eating": 1, "drinking": 2},
                exit_frames={"eating": 3},
                default_behavior="standing",
            )
        )
        machine.update(decision("eating"))

        machine.update(decision("drinking"))
        machine.update(decision("standing"))
        state = machine.update(decision("drinking"))

        self.assertEqual(state.stable_behavior, "eating")
        self.assertEqual(state.transition_reason, "held")

        state = machine.update(decision("drinking"))

        self.assertEqual(state.stable_behavior, "drinking")
        self.assertEqual(state.transition_reason, "entered:drinking")

    def test_unknown_does_not_immediately_clear_stable_behavior(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"drinking": 1}, exit_frames={"drinking": 3}, default_behavior="standing"))
        machine.update(decision("drinking"))

        for _ in range(2):
            state = machine.update(decision("unknown", confidence=0.1))

        self.assertEqual(state.stable_behavior, "drinking")

    def test_unknown_holds_default_without_reentering_default(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"standing": 3}, default_behavior="standing"))

        for _ in range(3):
            state = machine.update(decision("unknown", confidence=0.1))

            self.assertEqual(state.stable_behavior, "standing")
            self.assertEqual(state.transition_reason, "held")
            self.assertEqual(state.raw_behavior, "unknown")


if __name__ == "__main__":
    unittest.main()
