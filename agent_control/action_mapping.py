import sys
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from drone_event_nav.actions import POLICY_ACTIONS, dispatch_action, parse_action


# Backward-compatible name used by existing training and evaluation scripts.
ACTIONS = POLICY_ACTIONS


if __name__ == "__main__":
    action = parse_action(
        "user\nTask: You are an outdoor exploration drone. Analyze the RGB and Depth observations to decide the next best move. "
        "Your current task is to find the closest burning car.\nObservations: <image><image>\n"
        "Current Pose: x=20.64, y=-1579.47, z=39.29, rz=0.0°.\n"
        "Action Set: [AUTO_DOWN, AUTO_FORWARD, AUTO_YAW_LEFT, AUTO_YAW_RIGHT, AUTO_STOP_REACHED].\n"
        "Requirement: You must output only one string from the action set.\nDecision:\nassistant\nAUTO_FORWARD"
    )
    print(action)
