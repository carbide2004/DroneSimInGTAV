"""Shared in-memory RGB-D geometry helpers for online validation."""

import numpy as np


def frame_to_world_points(
    frame,
    pixel_stride,
    max_view_depth,
):
    rgb = frame.rgb_array()
    depth = frame.depth_array()
    rows = np.arange(
        0,
        frame.height,
        pixel_stride,
        dtype=np.int32,
    )
    columns = np.arange(
        0,
        frame.width,
        pixel_stride,
        dtype=np.int32,
    )
    pixel_x, pixel_y = np.meshgrid(columns, rows)

    sampled_depth = depth[pixel_y, pixel_x]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (sampled_depth <= max_view_depth)
    )
    if not np.any(valid):
        raise RuntimeError(
            "No valid depth samples remain for point-cloud display"
        )

    pixel_x = pixel_x[valid].astype(np.float64)
    pixel_y = pixel_y[valid].astype(np.float64)
    sampled_depth = sampled_depth[valid].astype(np.float64)
    ndc_x = (
        (2.0 / (frame.width - 1.0)) * pixel_x - 1.0
    )
    ndc_y = (
        (-2.0 / (frame.height - 1.0)) * pixel_y + 1.0
    )

    projection = np.asarray(
        frame.projection_matrix,
        dtype=np.float64,
    ).reshape(4, 4)
    view = np.asarray(
        frame.view_matrix,
        dtype=np.float64,
    ).reshape(4, 4)
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
    if (
        not np.isfinite(rays).all()
        or np.any(np.abs(ray_z) < 1.0e-12)
    ):
        raise RuntimeError(
            "Projection inverse produced invalid camera rays"
        )

    scale = -sampled_depth / ray_z
    view_points = rays[:3, :] * scale[None, :]
    view_points_h = np.vstack(
        (
            view_points,
            np.ones(
                (1, view_points.shape[1]),
                dtype=np.float64,
            ),
        )
    )
    world_points_h = inverse_view @ view_points_h
    world_points_h /= world_points_h[3:4, :]
    world_points = world_points_h[:3, :].T
    colors = (
        rgb[
            pixel_y.astype(np.int32),
            pixel_x.astype(np.int32),
        ].astype(np.float64)
        / 255.0
    )

    if not np.isfinite(world_points).all():
        raise RuntimeError(
            "World-space point cloud contains non-finite values"
        )
    return world_points, colors
