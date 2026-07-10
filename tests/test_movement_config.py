import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class MovementConfigTests(unittest.TestCase):
    def test_argparse_adapter_preserves_independent_up_and_down_steps(self):
        from drone_event_nav.config import MovementConfig

        args = SimpleNamespace(
            forward_step=5.0,
            up_step=7.0,
            down_step=3.0,
            yaw_step=15.0,
        )

        config = MovementConfig.from_namespace(args)

        self.assertEqual(config.to_dispatch_kwargs(), {
            "forward_step": 5.0,
            "up_step": 7.0,
            "down_step": 3.0,
            "yaw_step": 15.0,
        })

    def test_legacy_namespace_without_up_step_uses_down_step(self):
        from drone_event_nav.config import MovementConfig

        args = SimpleNamespace(
            forward_step=5.0,
            down_step=3.0,
            yaw_step=15.0,
        )

        config = MovementConfig.from_namespace(args)

        self.assertEqual(config.up_step, 3.0)
        self.assertEqual(config.down_step, 3.0)

    def test_none_up_step_uses_down_step_for_cli_compatibility(self):
        from drone_event_nav.config import MovementConfig

        args = SimpleNamespace(
            forward_step=5.0,
            up_step=None,
            down_step=3.0,
            yaw_step=15.0,
        )

        config = MovementConfig.from_namespace(args)

        self.assertEqual(config.up_step, 3.0)

    def test_non_positive_steps_are_rejected(self):
        from drone_event_nav.config import MovementConfig

        with self.assertRaisesRegex(ValueError, "forward_step"):
            MovementConfig(forward_step=0.0, up_step=5.0, down_step=5.0, yaw_step=15.0)

    def test_non_finite_steps_are_rejected(self):
        from drone_event_nav.config import MovementConfig

        with self.assertRaisesRegex(ValueError, "yaw_step"):
            MovementConfig(forward_step=5.0, up_step=5.0, down_step=5.0, yaw_step=math.inf)


if __name__ == "__main__":
    unittest.main()
