"""Validate live RGB-D calibration by building an in-memory 360-degree point cloud.

This script never writes RGB, depth, screenshots, videos, PLY, or PCD files.
"""

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import DroneSimClient


def _angle_error_degrees(actual, expected):
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def _wait_for_posture(client, target, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pose = client.get_pose()
        if pose is None:
            raise RuntimeError("GET_POSE returned no camera pose")
        position_error = np.linalg.norm(
            np.asarray(pose[:3], dtype=np.float64)
            - np.asarray(target[:3], dtype=np.float64)
        )
        rotation_error = max(
            _angle_error_degrees(pose[3], target[3]),
            _angle_error_degrees(pose[4], target[4]),
            _angle_error_degrees(pose[5], target[5]),
        )
        if position_error <= 1.0e-3 and rotation_error <= 1.0e-2:
            return pose
        time.sleep(0.01)
    raise TimeoutError(f"Camera did not reach requested posture {target}")


def _frame_to_world_points(frame, pixel_stride, max_view_depth):
    rgb = frame.rgb_array()
    depth = frame.depth_array()
    rows = np.arange(0, frame.height, pixel_stride, dtype=np.int32)
    columns = np.arange(0, frame.width, pixel_stride, dtype=np.int32)
    pixel_x, pixel_y = np.meshgrid(columns, rows)

    sampled_depth = depth[pixel_y, pixel_x]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (sampled_depth <= max_view_depth)
    )
    if not np.any(valid):
        raise RuntimeError("No valid depth samples remain for point-cloud display")

    pixel_x = pixel_x[valid].astype(np.float64)
    pixel_y = pixel_y[valid].astype(np.float64)
    sampled_depth = sampled_depth[valid].astype(np.float64)
    ndc_x = (2.0 / (frame.width - 1.0)) * pixel_x - 1.0
    ndc_y = (-2.0 / (frame.height - 1.0)) * pixel_y + 1.0

    projection = np.asarray(
        frame.projection_matrix, dtype=np.float64
    ).reshape(4, 4)
    view = np.asarray(frame.view_matrix, dtype=np.float64).reshape(4, 4)
    inverse_projection = np.linalg.inv(projection)
    inverse_view = np.linalg.inv(view)

    ndc = np.stack(
        (
            ndc_x,
            ndc_y,
            np.full_like(ndc_x, 0.5),
            np.ones_like(ndc_x),
        ),
        axis=0,
    )
    rays = inverse_projection @ ndc
    rays /= rays[3:4, :]
    ray_z = rays[2, :]
    if not np.isfinite(rays).all() or np.any(np.abs(ray_z) < 1.0e-12):
        raise RuntimeError("Projection inverse produced invalid camera rays")

    scale = -sampled_depth / ray_z
    view_points = rays[:3, :] * scale[None, :]
    view_points_h = np.vstack(
        (view_points, np.ones((1, view_points.shape[1]), dtype=np.float64))
    )
    world_points_h = inverse_view @ view_points_h
    world_points_h /= world_points_h[3:4, :]
    world_points = world_points_h[:3, :].T
    colors = rgb[pixel_y.astype(np.int32), pixel_x.astype(np.int32)].astype(
        np.float64
    ) / 255.0

    if not np.isfinite(world_points).all():
        raise RuntimeError("World-space point cloud contains non-finite values")
    return world_points, colors


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
    parser.add_argument("--posture-timeout", type=float, default=3.0)
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
            client.set_posture(*target)
            _wait_for_posture(client, target, args.posture_timeout)
            frame = client.capture(args.capture_timeout_ms)
            if frame_ids and frame.frame_id <= frame_ids[-1]:
                raise RuntimeError(
                    f"Non-increasing frame_id: {frame.frame_id} "
                    f"after {frame_ids[-1]}"
                )
            points, colors = _frame_to_world_points(
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
        client.set_posture(*original_pose)
        _wait_for_posture(client, original_pose, args.posture_timeout)

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
