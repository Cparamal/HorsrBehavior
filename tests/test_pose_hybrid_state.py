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

    def test_unknown_does_not_immediately_clear_stable_behavior(self):
        machine = BehaviorStateMachine(StateMachineConfig(enter_frames={"drinking": 1}, exit_frames={"drinking": 3}, default_behavior="standing"))
        machine.update(decision("drinking"))

        for _ in range(2):
            state = machine.update(decision("unknown", confidence=0.1))

        self.assertEqual(state.stable_behavior, "drinking")


if __name__ == "__main__":
    unittest.main()
