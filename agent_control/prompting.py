ACTION_SET_TEXT = "[AUTO_DOWN, AUTO_UP, AUTO_FORWARD, AUTO_YAW_LEFT, AUTO_YAW_RIGHT, AUTO_STOP_REACHED]"


def build_prompt(x, y, z, rz, task=None):
    if task is None:
        task = "find the closest burning car"
    
    return (
        f"Task: You are an outdoor exploration drone. Analyze the RGB and Depth observations to decide the next best move. Your current task is to {task}.\n"
        "Observations: <image><image>\n"
        f"Current Pose: x={float(x):.2f}, y={float(y):.2f}, z={float(z):.2f}, rz={rz}°.\n"
        f"Action Set: {ACTION_SET_TEXT}.\n"
        "Requirement: You must output only one string from the action set.\n"
        "Decision:"
    )
