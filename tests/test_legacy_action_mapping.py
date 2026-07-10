import inspect
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
AGENT_ROOT = REPO_ROOT / "agent_control"
for path in (SRC_ROOT, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakeClient:
    def __init__(self):
        self.calls = []

    def move(self, dx, dy, dz):
        self.calls.append(("move", dx, dy, dz))

    def rotate(self, rx, ry, rz):
        self.calls.append(("rotate", rx, ry, rz))


class LegacyActionMappingTests(unittest.TestCase):
    def test_public_api_remains_available(self):
        from action_mapping import ACTIONS, dispatch_action, parse_action

        self.assertEqual(len(ACTIONS), 6)
        self.assertEqual(parse_action("Decision:\nassistant\nAUTO_FORWARD"), "AUTO_FORWARD")
        self.assertEqual(
            list(inspect.signature(dispatch_action).parameters),
            ["cli", "action", "forward_step", "up_step", "down_step", "yaw_step"],
        )

    def test_legacy_dispatch_calls_client(self):
        from action_mapping import dispatch_action

        client = FakeClient()
        dispatch_action(client, "AUTO_YAW_LEFT", yaw_step=12.0)

        self.assertEqual(client.calls, [("rotate", 0.0, 0.0, 12.0)])


if __name__ == "__main__":
    unittest.main()
