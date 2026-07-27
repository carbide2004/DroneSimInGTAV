"""Validate synchronous camera-pose control against fresh Capture V3 metadata.

The script keeps every RGB-D payload in memory and writes no images or point
clouds. It restores the initial camera pose before exiting.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    DroneSimCommandError,
)


def _angle_error(actual, expected):
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def _assert_pose(actual, expected, position_tolerance, yaw_tolerance):
    position_error = float(
        np.linalg.norm(
            np.asarray(actual[:3], dtype=np.float64)
            - np.asarray(expected[:3], dtype=np.float64)
        )
    )
    yaw_error = _angle_error(actual[5], expected[5])
    if position_error > position_tolerance or yaw_error > yaw_tolerance:
        raise RuntimeError(
            "Pose mismatch: "
            f"position_error={position_error:.6f} m, "
            f"yaw_error={yaw_error:.6f} deg"
        )


def _assert_view_matches_pose(
    frame,
    pose,
    position_tolerance,
    direction_tolerance,
):
    view = np.asarray(frame.view_matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(view).all():
        raise RuntimeError("Capture view matrix contains non-finite values")
    inverse_view = np.linalg.inv(view)
    camera_center = inverse_view[:3, 3]
    center_error = float(
        np.linalg.norm(camera_center - np.asarray(pose[:3], dtype=np.float64))
    )
    if center_error > position_tolerance:
        raise RuntimeError(
            "Capture view matrix has the wrong camera center: "
            f"error={center_error:.6f} m"
        )

    pitch = math.radians(float(pose[3]))
    yaw = math.radians(float(pose[5]))
    expected_forward = np.asarray(
        (
            -math.sin(yaw) * abs(math.cos(pitch)),
            math.cos(yaw) * abs(math.cos(pitch)),
            math.sin(pitch),
        ),
        dtype=np.float64,
    )
    actual_forward = -view[2, :3]
    actual_forward /= np.linalg.norm(actual_forward)
    direction_error = float(
        np.linalg.norm(actual_forward - expected_forward)
    )
    if direction_error > direction_tolerance:
        raise RuntimeError(
            "Capture view matrix has the wrong camera orientation: "
            f"direction_error={direction_error:.6e}"
        )


def _validate_expected_collision(client, target, original_pose):
    try:
        client.set_camera_pose(
            target[0],
            target[1],
            target[2],
            original_pose[5],
            collision_check=True,
        )
    except DroneSimCommandError as error:
        if error.status_name != "COLLISION_BLOCKED":
            raise RuntimeError(
                "Expected COLLISION_BLOCKED, received "
                f"{error.status_name}"
            ) from error
    else:
        raise RuntimeError(
            "Expected collision target was accepted by the plugin"
        )
    after = client.get_pose()
    _assert_pose(after, original_pose, 1.0e-3, 1.0e-2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Alternate two nearby absolute poses, capture immediately, and "
            "verify command acknowledgement plus Capture V3 view metadata."
        )
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--captures", type=int, default=40)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--yaw-delta", type=float, default=120.0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--position-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--yaw-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--direction-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--collision-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable navigation collision checks for the alternating poses.",
    )
    parser.add_argument(
        "--expect-collision",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Additionally require this world-space target to produce "
            "COLLISION_BLOCKED without moving the camera."
        ),
    )
    parser.add_argument(
        "--expect-forward-collision",
        type=float,
        metavar="METERS",
        help=(
            "Require a target this many meters along the current camera "
            "forward direction to produce COLLISION_BLOCKED."
        ),
    )
    args = parser.parse_args()

    if args.captures <= 0:
        raise ValueError("--captures must be positive")
    for name in (
        "offset",
        "yaw_delta",
        "position_tolerance",
        "yaw_tolerance",
        "direction_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.expect_forward_collision is not None
        and (
            not math.isfinite(args.expect_forward_collision)
            or args.expect_forward_collision <= 0
        )
    ):
        raise ValueError(
            "--expect-forward-collision must be positive and finite"
        )
    if (
        args.expect_collision is not None
        and args.expect_forward_collision is not None
    ):
        raise ValueError(
            "Use only one of --expect-collision and "
            "--expect-forward-collision"
        )

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original = client.get_pose()
    x, y, z, pitch, roll, yaw = original
    yaw_radians = math.radians(yaw)
    right = (math.cos(yaw_radians), math.sin(yaw_radians))
    targets = (
        (
            x + right[0] * args.offset,
            y + right[1] * args.offset,
            z,
            pitch,
            roll,
            yaw + args.yaw_delta,
        ),
        (
            x - right[0] * args.offset,
            y - right[1] * args.offset,
            z,
            pitch,
            roll,
            yaw - args.yaw_delta,
        ),
    )

    frame_ids = []
    try:
        for index in range(args.captures):
            target = targets[index % 2]
            actual = client.set_camera_pose(
                target[0],
                target[1],
                target[2],
                target[5],
                collision_check=args.collision_check,
            )
            _assert_pose(
                actual,
                target,
                args.position_tolerance,
                args.yaw_tolerance,
            )
            observed = client.get_pose()
            _assert_pose(
                observed,
                actual,
                args.position_tolerance,
                args.yaw_tolerance,
            )
            frame = client.capture(args.capture_timeout_ms)
            if frame_ids and frame.frame_id <= frame_ids[-1]:
                raise RuntimeError(
                    f"Non-increasing frame_id {frame.frame_id} "
                    f"after {frame_ids[-1]}"
                )
            _assert_view_matches_pose(
                frame,
                actual,
                args.position_tolerance,
                args.direction_tolerance,
            )
            frame_ids.append(frame.frame_id)

        client.set_camera_pose(
            original[0],
            original[1],
            original[2],
            original[5],
            collision_check=False,
        )
        if args.expect_collision is not None:
            _validate_expected_collision(
                client,
                args.expect_collision,
                original,
            )
        elif args.expect_forward_collision is not None:
            pitch_radians = math.radians(original[3])
            yaw_radians = math.radians(original[5])
            distance = args.expect_forward_collision
            forward_target = (
                original[0]
                - math.sin(yaw_radians)
                * abs(math.cos(pitch_radians))
                * distance,
                original[1]
                + math.cos(yaw_radians)
                * abs(math.cos(pitch_radians))
                * distance,
                original[2] + math.sin(pitch_radians) * distance,
            )
            print(
                "testing forward collision target "
                f"({forward_target[0]:.2f}, "
                f"{forward_target[1]:.2f}, "
                f"{forward_target[2]:.2f})"
            )
            _validate_expected_collision(
                client,
                forward_target,
                original,
            )
    finally:
        client.set_camera_pose(
            original[0],
            original[1],
            original[2],
            original[5],
            collision_check=False,
        )

    print(
        f"PASS captures={len(frame_ids)} "
        f"frames={frame_ids[0]}..{frame_ids[-1]}"
    )
    print("No RGB-D payload was written to disk.")


if __name__ == "__main__":
    main()
