"""Deterministic agent-centric local geometry derived from dual-view Depth.

The policy is allowed to consume this map because every value comes from the
current RGB-D observation and camera calibration.  No GTA collision query or
evaluation truth enters this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


LOCAL_GEOMETRY_CHANNELS = (
    "observed_free_level",
    "occupied_level",
    "occupied_below",
    "occupied_above",
    "mean_surface_height",
    "unknown",
)


@dataclass(frozen=True)
class LocalGeometryConfig:
    half_extent_m: float = 20.0
    cell_m: float = 1.0
    maximum_depth_m: float = 24.0
    occupied_pixel_stride: int = 8
    free_pixel_stride: int = 32
    level_half_height_m: float = 1.0
    vertical_extent_m: float = 4.0
    height_normalizer_m: float = 8.0

    @property
    def grid_size(self) -> int:
        return int(round(2.0 * self.half_extent_m / self.cell_m)) + 1

    def validate(self) -> None:
        finite_positive = (
            self.half_extent_m,
            self.cell_m,
            self.maximum_depth_m,
            self.level_half_height_m,
            self.vertical_extent_m,
            self.height_normalizer_m,
        )
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in finite_positive):
            raise ValueError("Local geometry scales must be finite and positive")
        if self.grid_size != 41:
            raise ValueError("Stage 3C local geometry must be a 41x41 grid")
        if self.occupied_pixel_stride <= 0 or self.free_pixel_stride <= 0:
            raise ValueError("Depth pixel strides must be positive")
        if self.vertical_extent_m <= self.level_half_height_m:
            raise ValueError("vertical_extent_m must exceed the level band")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LocalGeometryConfig":
        config = cls(**payload)
        config.validate()
        return config


def projection_matrix_from_spec(observation_spec) -> np.ndarray:
    """Reconstruct the exact GTA projection convention used by the recorder."""
    width = int(observation_spec["width"])
    height = int(observation_spec["height"])
    fov = float(observation_spec["fov_degrees"])
    near = float(observation_spec["near_clip"])
    far = float(observation_spec["far_clip"])
    if width < 2 or height < 2 or not 0.0 < fov < 180.0 or not 0.0 < near < far:
        raise ValueError("Invalid recorded observation calibration")
    tangent = math.tan(math.radians(fov) * 0.5)
    near_minus_far = near - far
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = (height / width) / tangent
    projection[1, 1] = 1.0 / tangent
    projection[2, 2] = -near / near_minus_far
    projection[2, 3] = -near * far / near_minus_far
    projection[3, 2] = -1.0
    return projection


def _body_view_matrix(pitch_degrees: float) -> np.ndarray:
    """Camera view at the body origin with body forward +Y and right +X."""
    pitch = math.radians(float(pitch_degrees))
    forward = np.asarray((0.0, abs(math.cos(pitch)), math.sin(pitch)))
    right = np.asarray((1.0, 0.0, 0.0))
    up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float64)
    result[0, :3] = right
    result[1, :3] = up
    result[2, :3] = -forward
    return result


def _sample_body_points(
    depth: np.ndarray,
    projection: np.ndarray,
    pitch_degrees: float,
    stride: int,
    maximum_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional array")
    height, width = depth.shape
    rows = np.arange(0, height, stride, dtype=np.int32)
    columns = np.arange(0, width, stride, dtype=np.int32)
    pixel_x, pixel_y = np.meshgrid(columns, rows)
    sampled = depth[pixel_y, pixel_x].astype(np.float64)
    valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled <= maximum_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    x = pixel_x[valid].astype(np.float64)
    y = pixel_y[valid].astype(np.float64)
    sampled = sampled[valid]
    ndc = np.stack(
        (
            2.0 * x / (width - 1.0) - 1.0,
            1.0 - 2.0 * y / (height - 1.0),
            np.full_like(x, 0.5),
            np.ones_like(x),
        ),
        axis=0,
    )
    rays = np.linalg.inv(projection) @ ndc
    rays /= rays[3:4]
    ray_z = rays[2]
    if not np.isfinite(rays).all() or np.any(np.abs(ray_z) < 1.0e-12):
        raise RuntimeError("Projection inverse produced invalid camera rays")
    scale = -sampled / ray_z
    view_points = rays[:3] * scale[None]
    body_h = np.linalg.inv(_body_view_matrix(pitch_degrees)) @ np.vstack(
        (view_points, np.ones((1, view_points.shape[1]), dtype=np.float64))
    )
    body_xyz = body_h[:3].T
    # The verified GTA camera basis uses +Y forward and +X right.  Expose the
    # research body convention as forward, right, up.
    points = body_xyz[:, (1, 0, 2)]
    if not np.isfinite(points).all():
        raise RuntimeError("Depth backprojection produced non-finite points")
    return points, sampled


def _cell_indices(points: np.ndarray, config: LocalGeometryConfig):
    row = np.rint(points[:, 0] / config.cell_m).astype(np.int64) + config.grid_size // 2
    column = np.rint(points[:, 1] / config.cell_m).astype(np.int64) + config.grid_size // 2
    valid = (
        (row >= 0)
        & (row < config.grid_size)
        & (column >= 0)
        & (column < config.grid_size)
    )
    return row[valid], column[valid], points[valid]


def build_local_geometry(
    views: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    config: LocalGeometryConfig | None = None,
) -> np.ndarray:
    """Build a 6x41x41 body-aligned geometry map.

    Each view is ``(depth, projection_matrix, pitch_degrees)``. Occupancy is
    taken from densely sampled ray endpoints. A coarser independent ray set
    marks observed free cells at four fractions before the measured surface.
    """
    config = LocalGeometryConfig() if config is None else config
    config.validate()
    size = config.grid_size
    free = np.zeros((size, size), dtype=np.float32)
    level = np.zeros_like(free)
    below = np.zeros_like(free)
    above = np.zeros_like(free)
    height_sum = np.zeros_like(free, dtype=np.float64)
    height_count = np.zeros_like(free, dtype=np.int32)

    for depth, projection, pitch in views:
        endpoints, _ = _sample_body_points(
            depth,
            np.asarray(projection, dtype=np.float64).reshape(4, 4),
            pitch,
            config.occupied_pixel_stride,
            config.maximum_depth_m,
        )
        rows, columns, selected = _cell_indices(endpoints, config)
        if selected.size:
            z = selected[:, 2]
            np.maximum.at(level, (rows, columns), (np.abs(z) <= config.level_half_height_m).astype(np.float32))
            np.maximum.at(
                below,
                (rows, columns),
                ((z < -config.level_half_height_m) & (z >= -config.vertical_extent_m)).astype(np.float32),
            )
            np.maximum.at(
                above,
                (rows, columns),
                ((z > config.level_half_height_m) & (z <= config.vertical_extent_m)).astype(np.float32),
            )
            np.add.at(height_sum, (rows, columns), np.clip(z, -config.height_normalizer_m, config.height_normalizer_m))
            np.add.at(height_count, (rows, columns), 1)

        free_endpoints, _ = _sample_body_points(
            depth,
            np.asarray(projection, dtype=np.float64).reshape(4, 4),
            pitch,
            config.free_pixel_stride,
            config.maximum_depth_m,
        )
        if free_endpoints.size:
            fractions = np.asarray((0.2, 0.4, 0.6, 0.8), dtype=np.float64)
            traversed = (free_endpoints[:, None, :] * fractions[None, :, None]).reshape(-1, 3)
            rows, columns, selected = _cell_indices(traversed, config)
            current_band = np.abs(selected[:, 2]) <= config.level_half_height_m
            np.maximum.at(free, (rows, columns), current_band.astype(np.float32))

    any_occupied = np.maximum.reduce((level, below, above))
    free[any_occupied > 0.0] = 0.0
    mean_height = np.zeros_like(free)
    observed_surface = height_count > 0
    mean_height[observed_surface] = (
        height_sum[observed_surface] / height_count[observed_surface]
    ).astype(np.float32) / config.height_normalizer_m
    unknown = ((free == 0.0) & (any_occupied == 0.0)).astype(np.float32)
    result = np.stack((free, level, below, above, mean_height, unknown), axis=0)
    if result.shape != (len(LOCAL_GEOMETRY_CHANNELS), size, size):
        raise RuntimeError("Local geometry shape invariant failed")
    if not np.isfinite(result).all():
        raise RuntimeError("Local geometry contains non-finite values")
    return result.astype(np.float32, copy=False)


def build_local_geometry_from_pair(pair, config: LocalGeometryConfig | None = None):
    return build_local_geometry(
        (
            (
                pair.oblique.depth_array(),
                np.asarray(pair.oblique.projection_matrix).reshape(4, 4),
                -45.0,
            ),
            (
                pair.nadir.depth_array(),
                np.asarray(pair.nadir.projection_matrix).reshape(4, 4),
                -90.0,
            ),
        ),
        config,
    )
