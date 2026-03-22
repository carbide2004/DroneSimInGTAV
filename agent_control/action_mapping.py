import re


ACTIONS = (
    "AUTO_DOWN",
    "AUTO_UP",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
)

_ACTION_RE = re.compile(
    r"\b(AUTO_DOWN|AUTO_UP|AUTO_FORWARD|AUTO_YAW_LEFT|AUTO_YAW_RIGHT|AUTO_STOP_REACHED)\b",
    re.IGNORECASE,
)


def parse_action(model_text):
    if model_text is None:
        return None
    raw = str(model_text)
    text = raw.strip()
    if not text:
        return None

    lower = text.lower()
    cut = -1

    for marker in ("decision:",):
        i = lower.rfind(marker)
        if i != -1:
            cut = max(cut, i + len(marker))

    for marker in ("\nassistant\n", "\nassistant:", "<|im_start|>assistant", "assistant\n"):
        i = lower.rfind(marker)
        if i != -1:
            cut = max(cut, i + len(marker))

    tail = text[cut:].strip() if cut != -1 else text
    tail = tail.replace("`", " ")

    matches = list(_ACTION_RE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(_ACTION_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    return None


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

if __name__ == "__main__":
    import sys

    action = parse_action("user\nTask: You are an outdoor exploration drone. Analyze the RGB and Depth observations to decide the next best move. Your current task is to find the closest burning car.\nObservations: <image><image>\nCurrent Pose: x=20.64, y=-1579.47, z=39.29, rz=0.0°.\nAction Set: [AUTO_DOWN, AUTO_FORWARD, AUTO_YAW_LEFT, AUTO_YAW_RIGHT, AUTO_STOP_REACHED].\nRequirement: You must output only one string from the action set.\nDecision:\nassistant\nAUTO_FORWARD")
    print(action)
