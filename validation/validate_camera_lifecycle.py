"""Validate synchronous camera lifecycle commands without fixed sleeps.

One RGB-D frame is captured in memory after recreation. The script restores
the initial active/inactive state and camera pose before exiting.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    DroneSimCommandError,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stop, query, recreate, and capture from the scripted camera "
            "without client-side settling sleeps."
        )
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    client = DroneSimClient(args.host, args.port)
    initially_active = client.is_camera_active()
    original_pose = client.get_pose() if initially_active else None
    camera_id = 0
    recreated_pose = None
    frame = None
    try:
        client.stop_camera()
        if client.is_camera_active():
            raise RuntimeError(
                "Camera remained active after STOP_CAMERA success"
            )
        try:
            client.get_pose()
        except DroneSimCommandError as error:
            if error.status_name != "CAMERA_INACTIVE":
                raise RuntimeError(
                    "Inactive GET_POSE returned "
                    f"{error.status_name}, expected CAMERA_INACTIVE"
                ) from error
        else:
            raise RuntimeError("Inactive GET_POSE unexpectedly succeeded")

        camera_id = client.create_camera()
        if camera_id == 0:
            raise RuntimeError("CREATE_CAMERA returned a zero camera id")
        if not client.is_camera_active():
            raise RuntimeError(
                "Camera is inactive after CREATE_CAMERA success"
            )
        recreated_pose = client.get_pose()
        frame = client.capture(args.capture_timeout_ms)
    finally:
        active_now = client.is_camera_active()
        if initially_active:
            if not active_now:
                client.create_camera()
            client.set_camera_pose(
                original_pose[0],
                original_pose[1],
                original_pose[2],
                original_pose[5],
                collision_check=False,
            )
        elif active_now:
            client.stop_camera()

    print(
        "PASS "
        f"camera_id={camera_id} "
        f"recreated_pose={tuple(round(value, 3) for value in recreated_pose)} "
        f"frame={frame.frame_id}"
    )
    print("No RGB-D payload was written to disk.")


if __name__ == "__main__":
    main()
