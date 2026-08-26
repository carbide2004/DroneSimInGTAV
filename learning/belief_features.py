"""Shared source-blind grounded-track feature encoding.

The encoder is deliberately stateful: motion is reconstructed only from the
same anonymous track in two adjacent frozen observations.  Both the offline
dataset loader and the online Spatial RNN runtime use this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SEMANTIC_CLASSES = (
    "PAD",
    "FIRE_TRUCK",
    "PEDESTRIAN",
    "FIRE_SOURCE",
    "UNKNOWN",
)
SEMANTIC_CLASS_TO_INDEX = {
    name: index for index, name in enumerate(SEMANTIC_CLASSES)
}

TRACK_FEATURE_NAMES = (
    "position_forward_over_radius",
    "position_right_over_radius",
    "position_up_over_vertical_bound",
    "motion_unit_forward",
    "motion_unit_right",
    "displacement_over_four_meters",
    "motion_valid",
    "evidence_age_over_horizon",
    "bbox_center_x_normalized",
    "bbox_center_y_normalized",
    "bbox_span_over_256_pixels",
    "view_is_nadir",
    "same_class_count_over_32",
)


@dataclass(frozen=True)
class GroundedTrackFeatureConfig:
    radius_m: float
    vertical_bound_m: float
    horizon_steps: int
    width: int
    height: int
    minimum_motion_m: float = 0.4

    def __post_init__(self) -> None:
        numeric = (
            self.radius_m,
            self.vertical_bound_m,
            self.minimum_motion_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in numeric):
            raise ValueError("Grounded-track feature scales must be finite and positive")
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        if int(self.width) < 2 or int(self.height) < 2:
            raise ValueError("RGB-D dimensions must be at least 2x2")


@dataclass(frozen=True)
class EncodedGroundedTrackStep:
    step_index: int
    track_features: np.ndarray
    track_classes: np.ndarray
    track_mask: np.ndarray
    source_visible_now: bool
    source_seen: bool
    motion_evidence_track_ids: tuple[int, ...]

    @property
    def has_motion_evidence(self) -> bool:
        return bool(self.motion_evidence_track_ids)


def _field(track, name):
    if isinstance(track, dict):
        return track[name]
    return getattr(track, name)


class StreamingGroundedTrackEncoder:
    """Encode one strictly ordered stream of grounded observations."""

    def __init__(self, config: GroundedTrackFeatureConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._previous_tracks: dict[int, tuple[int, str, np.ndarray]] = {}
        self._evidence_counts: dict[int, int] = {}
        self._source_seen = False
        self._last_step_index: int | None = None

    def encode(self, grounded_frame, tracks=None) -> EncodedGroundedTrackStep:
        if tracks is None:
            step_index = _field(grounded_frame, "step_index")
            tracks = _field(grounded_frame, "tracks")
        else:
            step_index = grounded_frame
        step_index = int(step_index)
        if step_index <= 0:
            raise ValueError("Grounded step_index must be positive")
        if self._last_step_index is not None and step_index <= self._last_step_index:
            raise ValueError(
                "Grounded step indices must be strictly increasing; "
                f"received {self._last_step_index} -> {step_index}"
            )
        tracks = tuple(tracks)
        source_visible_now = any(
            str(_field(track, "semantic_class")) == "FIRE_SOURCE"
            for track in tracks
        )
        self._source_seen = self._source_seen or source_visible_now

        class_counts: dict[str, int] = {}
        for track in tracks:
            semantic = str(_field(track, "semantic_class"))
            class_counts[semantic] = class_counts.get(semantic, 0) + 1

        encoded_features = []
        encoded_classes = []
        evidence_track_ids = []
        current_tracks: dict[int, tuple[int, str, np.ndarray]] = {}
        for track in tracks:
            track_id = int(_field(track, "track_id"))
            semantic = str(_field(track, "semantic_class"))
            position = np.asarray(_field(track, "position_local"), dtype=np.float32)
            bbox = np.asarray(_field(track, "projected_bbox"), dtype=np.float32)
            view_name = str(_field(track, "view_name"))
            if position.shape != (3,) or bbox.shape != (4,):
                raise ValueError("Grounded track position/bbox shape mismatch")
            if not np.isfinite(position).all() or not np.isfinite(bbox).all():
                raise ValueError("Grounded track contains non-finite values")
            if view_name not in ("oblique", "nadir"):
                raise ValueError(f"Unknown grounded view_name={view_name!r}")

            previous = self._previous_tracks.get(track_id)
            delta = np.zeros(2, dtype=np.float32)
            displacement = 0.0
            motion_valid = 0.0
            if (
                previous is not None
                and previous[0] == step_index - 1
                and previous[1] == semantic
            ):
                delta = position[:2] - previous[2][:2]
                displacement = float(np.linalg.norm(delta))
                if displacement >= self.config.minimum_motion_m:
                    motion_valid = 1.0
                    self._evidence_counts[track_id] = (
                        self._evidence_counts.get(track_id, 0) + 1
                    )
                    if semantic != "FIRE_SOURCE":
                        evidence_track_ids.append(track_id)
            unit = (
                delta / displacement
                if displacement > 1.0e-6
                else np.zeros(2, dtype=np.float32)
            )
            bbox_center_x = 0.5 * float(bbox[0] + bbox[2])
            bbox_center_y = 0.5 * float(bbox[1] + bbox[3])
            bbox_span = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
            encoded_features.append(
                np.asarray(
                    (
                        position[0] / self.config.radius_m,
                        position[1] / self.config.radius_m,
                        position[2] / self.config.vertical_bound_m,
                        unit[0],
                        unit[1],
                        displacement / 4.0,
                        motion_valid,
                        self._evidence_counts.get(track_id, 0)
                        / float(self.config.horizon_steps),
                        2.0 * bbox_center_x / float(self.config.width - 1) - 1.0,
                        2.0 * bbox_center_y / float(self.config.height - 1) - 1.0,
                        bbox_span / 256.0,
                        1.0 if view_name == "nadir" else 0.0,
                        class_counts[semantic] / 32.0,
                    ),
                    dtype=np.float32,
                )
            )
            encoded_classes.append(
                SEMANTIC_CLASS_TO_INDEX.get(
                    semantic, SEMANTIC_CLASS_TO_INDEX["UNKNOWN"]
                )
            )
            current_tracks[track_id] = (step_index, semantic, position)

        feature_count = len(TRACK_FEATURE_NAMES)
        if encoded_features:
            features = np.stack(encoded_features).astype(np.float32, copy=False)
            classes = np.asarray(encoded_classes, dtype=np.int64)
            mask = np.ones(len(encoded_features), dtype=np.bool_)
        else:
            features = np.empty((0, feature_count), dtype=np.float32)
            classes = np.empty((0,), dtype=np.int64)
            mask = np.empty((0,), dtype=np.bool_)

        self._previous_tracks = current_tracks
        self._last_step_index = step_index
        return EncodedGroundedTrackStep(
            step_index=step_index,
            track_features=features,
            track_classes=classes,
            track_mask=mask,
            source_visible_now=source_visible_now,
            source_seen=self._source_seen,
            motion_evidence_track_ids=tuple(sorted(evidence_track_ids)),
        )
