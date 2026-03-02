import re


ACTIONS = (
    "AUTO_DOWN",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
)

_ACTION_RE = re.compile(
    r"\b(AUTO_DOWN|AUTO_FORWARD|AUTO_YAW_LEFT|AUTO_YAW_RIGHT|AUTO_STOP_REACHED)\b",
    re.IGNORECASE,
)


def parse_action(model_text):
    if model_text is None:
        return None
    text = str(model_text).strip()
    if not text:
        return None

    text = text.replace("`", " ").replace("\n", " ").replace("\r", " ")
    m = _ACTION_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def dispatch_action(
    cli,
    action,
    forward_step=5.0,
    down_step=5.0,
    yaw_step=15.0,
):
    action = str(action).upper()
    if action == "AUTO_FORWARD":
        cli.move(float(forward_step), 0.0, 0.0)
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

