import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
AGENT_ROOT = REPO_ROOT / "agent_control"
for path in (SRC_ROOT, AGENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep this compatibility test dependency-light. verification_runtime only needs
# these image helpers when an actual capture is processed.
if "rgbd_utils" not in sys.modules:
    rgbd_stub = types.ModuleType("rgbd_utils")
    rgbd_stub.depth_bytes_to_pil = lambda *args, **kwargs: None
    rgbd_stub.rgb_bytes_to_pil = lambda *args, **kwargs: None
    sys.modules["rgbd_utils"] = rgbd_stub


class LegacyVerificationAdapterTests(unittest.TestCase):
    def test_build_movement_params_uses_canonical_config(self):
        from verification_runtime import build_movement_params

        args = SimpleNamespace(
            forward_step=5.0,
            up_step=8.0,
            down_step=2.0,
            yaw_step=10.0,
        )

        self.assertEqual(build_movement_params(args), {
            "forward_step": 5.0,
            "up_step": 8.0,
            "down_step": 2.0,
            "yaw_step": 10.0,
        })


if __name__ == "__main__":
    unittest.main()
