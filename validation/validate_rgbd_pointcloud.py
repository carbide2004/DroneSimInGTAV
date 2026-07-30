"""Validate live RGB-D calibration by building an in-memory 360-degree point cloud.

This script never writes RGB, depth, screenshots, videos, PLY, or PCD files.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import DroneSimClient
from validation.rgbd_geometry import (  # noqa: E402
    frame_to_world_points,
)


def main():
    parser = argparse.ArgumentParser(
        description="Rotate the current GTA V camera in 45-degree steps and "
        "display one merged in-memory colored point cloud."
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-view-depth", type=float, default=200.0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive")
    if not math.isfinite(args.max_view_depth) or args.max_view_depth <= 0:
        raise ValueError("--max-view-depth must be a positive finite value")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for interactive point-cloud validation"
        ) from exc

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    if original_pose is None:
        raise RuntimeError("No active GTA V rendering camera")
    x, y, z, pitch, roll, initial_yaw = original_pose

    point_batches = []
    color_batches = []
    frame_ids = []
    try:
        for step in range(8):
            yaw = initial_yaw + 45.0 * step
            target = (x, y, z, pitch, roll, yaw)
            client.set_camera_pose(
                target[0],
                target[1],
                target[2],
                target[5],
                collision_check=False,
            )
            frame = client.capture(args.capture_timeout_ms)
            if frame_ids and frame.frame_id <= frame_ids[-1]:
                raise RuntimeError(
                    f"Non-increasing frame_id: {frame.frame_id} "
                    f"after {frame_ids[-1]}"
                )
            points, colors = frame_to_world_points(
                frame,
                args.pixel_stride,
                args.max_view_depth,
            )
            frame_ids.append(frame.frame_id)
            point_batches.append(points)
            color_batches.append(colors)
            print(
                f"view {step + 1}/8 yaw={yaw:.1f} "
                f"frame={frame.frame_id} points={len(points)}"
            )
    finally:
        client.set_camera_pose(
            original_pose[0],
            original_pose[1],
            original_pose[2],
            original_pose[5],
            collision_check=False,
        )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(point_batches, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(color_batches, axis=0))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[x, y, z]
    )
    print(
        f"displaying {len(cloud.points)} in-memory points from frames {frame_ids}; "
        "nothing will be saved"
    )
    o3d.visualization.draw_geometries(
        [cloud, axes],
        window_name="DroneSim RGB-D 360-degree validation",
    )


if __name__ == "__main__":
    main()
