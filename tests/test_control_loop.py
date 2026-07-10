import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeClient:
    def __init__(self, poses, captures=None):
        self.poses = list(poses)
        self.captures = list(captures or [])
        self.moves = []

    def get_pose(self):
        return self.poses.pop(0) if self.poses else None

    def capture(self):
        return self.captures.pop(0) if self.captures else None

    def move(self, dx, dy, dz):
        self.moves.append((dx, dy, dz))

    def rotate(self, rx, ry, rz):
        self.moves.append((rx, ry, rz))


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate_action(self, observation):
        return self.outputs.pop(0)


class ControlLoopTests(unittest.TestCase):
    def test_repeated_pose_failures_stop_at_failure_budget(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        result = run_control_loop(
            client=FakeClient([None, None, None]),
            model=FakeModel([]),
            config=ControlLoopConfig(max_steps=5, max_consecutive_failures=2),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "pose_failure_budget_exhausted")
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.failure_count, 2)

    def test_repeated_capture_failures_stop_at_failure_budget(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        result = run_control_loop(
            client=FakeClient(
                poses=[
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0),
                ],
                captures=[None, None],
            ),
            model=FakeModel([]),
            config=ControlLoopConfig(max_steps=5, max_consecutive_failures=2),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "capture_failure_budget_exhausted")
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.failure_count, 2)

    def test_mixed_failures_do_not_mislabel_the_budget(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        result = run_control_loop(
            client=FakeClient(
                poses=[
                    (0, 0, 0, 0, 0, 0),
                    None,
                    (0, 0, 0, 0, 0, 0),
                ],
                captures=[None, None],
            ),
            model=FakeModel([]),
            config=ControlLoopConfig(max_steps=5, max_consecutive_failures=2),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "capture_failure_budget_exhausted")
        self.assertEqual(result.failure_count, 3)

    def test_invalid_model_output_is_a_policy_error(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        client = FakeClient(
            poses=[(0, 0, 0, 0, 0, 0)],
            captures=[object()],
        )
        result = run_control_loop(
            client=client,
            model=FakeModel(["move closer"]),
            config=ControlLoopConfig(max_steps=5),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "invalid_policy_output")
        self.assertEqual(result.steps, 0)
        self.assertEqual(client.moves, [])

    def test_final_pose_is_read_after_the_last_dispatched_action(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        client = FakeClient(
            poses=[
                (0, 0, 0, 0, 0, 0),
                (5, 0, 0, 0, 0, 0),
            ],
            captures=[object()],
        )
        result = run_control_loop(
            client=client,
            model=FakeModel(["AUTO_FORWARD"]),
            config=ControlLoopConfig(max_steps=1),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "max_steps_reached")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.final_pose, (5, 0, 0, 0, 0, 0))

    def test_failed_post_action_pose_is_reported(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        client = FakeClient(
            poses=[
                (0, 0, 0, 0, 0, 0),
                None,
            ],
            captures=[object()],
        )
        result = run_control_loop(
            client=client,
            model=FakeModel(["AUTO_FORWARD"]),
            config=ControlLoopConfig(max_steps=1),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "final_pose_unavailable")
        self.assertIsNone(result.final_pose)
        self.assertEqual(result.last_known_pose, (0, 0, 0, 0, 0, 0))
        self.assertEqual(result.failure_count, 1)

    def test_cleanup_runs_when_model_raises(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        class RaisingModel:
            def generate_action(self, observation):
                raise RuntimeError("model failed")

        cleaned = []
        result = run_control_loop(
            client=FakeClient([(0, 0, 0, 0, 0, 0)], [object()]),
            model=RaisingModel(),
            config=ControlLoopConfig(max_steps=1),
            make_observation=lambda pose, capture: (pose, capture),
            cleanup=lambda: cleaned.append(True),
        )

        self.assertEqual(result.status, "exception")
        self.assertEqual(result.error, "model failed")
        self.assertEqual(cleaned, [True])

    def test_exception_after_dispatched_action_preserves_progress(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        class RaisingPostActionPoseClient(FakeClient):
            def get_pose(self):
                if len(self.poses) == 1:
                    raise RuntimeError("pose read failed")
                return super().get_pose()

        result = run_control_loop(
            client=RaisingPostActionPoseClient(
                poses=[
                    (0, 0, 0, 0, 0, 0),
                    (5, 0, 0, 0, 0, 0),
                ],
                captures=[object()],
            ),
            model=FakeModel(["AUTO_FORWARD"]),
            config=ControlLoopConfig(max_steps=2),
            make_observation=lambda pose, capture: (pose, capture),
        )

        self.assertEqual(result.status, "exception")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.last_known_pose, (0, 0, 0, 0, 0, 0))
        self.assertEqual(result.error, "pose read failed")

    def test_cleanup_error_does_not_escape_or_hide_primary_error(self):
        from drone_event_nav.evaluation.control_loop import ControlLoopConfig, run_control_loop

        class RaisingModel:
            def generate_action(self, observation):
                raise RuntimeError("model failed")

        def failing_cleanup():
            raise RuntimeError("cleanup failed")

        result = run_control_loop(
            client=FakeClient([(0, 0, 0, 0, 0, 0)], [object()]),
            model=RaisingModel(),
            config=ControlLoopConfig(max_steps=1),
            make_observation=lambda pose, capture: (pose, capture),
            cleanup=failing_cleanup,
        )

        self.assertEqual(result.status, "exception")
        self.assertEqual(result.error, "model failed")
        self.assertEqual(result.cleanup_error, "cleanup failed")


if __name__ == "__main__":
    unittest.main()
