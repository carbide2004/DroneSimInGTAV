import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeClient:
    def __init__(self):
        self.calls = []

    def move(self, dx, dy, dz):
        self.calls.append(("move", dx, dy, dz))

    def rotate(self, rx, ry, rz):
        self.calls.append(("rotate", rx, ry, rz))


class ActionContractTests(unittest.TestCase):
    def test_parser_prefers_final_assistant_action(self):
        from drone_event_nav.actions import parse_action

        text = "Action Set: [AUTO_DOWN, AUTO_FORWARD]\nDecision:\nassistant\nAUTO_DOWN"
        self.assertEqual(parse_action(text), "AUTO_DOWN")

    def test_parser_returns_none_for_invalid_output(self):
        from drone_event_nav.actions import parse_action

        self.assertIsNone(parse_action("move toward the target"))

    def test_policy_and_terminal_outcomes_are_separate(self):
        from drone_event_nav.actions import POLICY_ACTIONS, TERMINAL_OUTCOMES

        self.assertNotIn("AUTO_STOP_FAILED", POLICY_ACTIONS)
        self.assertIn("AUTO_STOP_FAILED", TERMINAL_OUTCOMES)

    def test_dispatch_preserves_up_and_down_signs(self):
        from drone_event_nav.actions import dispatch_action

        client = FakeClient()
        dispatch_action(client, "AUTO_UP", up_step=7.0)
        dispatch_action(client, "AUTO_DOWN", down_step=3.0)

        self.assertEqual(client.calls, [
            ("move", 0.0, 0.0, 7.0),
            ("move", 0.0, 0.0, -3.0),
        ])

    def test_unknown_action_is_rejected(self):
        from drone_event_nav.actions import dispatch_action

        with self.assertRaisesRegex(ValueError, "Unknown action"):
            dispatch_action(FakeClient(), "AUTO_SIDEWAYS")


if __name__ == "__main__":
    unittest.main()
