import re
from typing import Optional


POLICY_ACTIONS = (
    "AUTO_DOWN",
    "AUTO_UP",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
)

TERMINAL_OUTCOMES = (
    "AUTO_STOP_FAILED",
    "AUTO_STOP_MAXSTEPS",
)

_ACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(action) for action in POLICY_ACTIONS) + r")\b",
    re.IGNORECASE,
)


def parse_action(model_text) -> Optional[str]:
    if model_text is None:
        return None
    text = str(model_text).strip()
    if not text:
        return None

    lower = text.lower()
    cut = -1
    for marker in ("decision:",):
        index = lower.rfind(marker)
        if index != -1:
            cut = max(cut, index + len(marker))
    for marker in ("\nassistant\n", "\nassistant:", "<|im_start|>assistant", "assistant\n"):
        index = lower.rfind(marker)
        if index != -1:
            cut = max(cut, index + len(marker))

    tail = text[cut:].strip() if cut != -1 else text
    tail_matches = list(_ACTION_RE.finditer(tail.replace("`", " ")))
    if tail_matches:
        return tail_matches[-1].group(1).upper()

    all_matches = list(_ACTION_RE.finditer(text))
    return all_matches[-1].group(1).upper() if all_matches else None


def dispatch_action(
    cli,
    action,
    forward_step=5.0,
    up_step=5.0,
    down_step=5.0,
    yaw_step=15.0,
):
    action = str(action).upper()
    if action == "AUTO_FORWARD":
        cli.move(float(forward_step), 0.0, 0.0)
        return
    if action == "AUTO_UP":
        cli.move(0.0, 0.0, float(up_step))
        return
    if action == "AUTO_DOWN":
        cli.move(0.0, 0.0, -float(down_step))
        return
    if action == "AUTO_YAW_LEFT":
        cli.rotate(0.0, 0.0, float(yaw_step))
        return
    if action == "AUTO_YAW_RIGHT":
        cli.rotate(0.0, 0.0, -float(yaw_step))
        return
    if action == "AUTO_STOP_REACHED":
        return
    raise ValueError(f"Unknown action: {action}")
