"""Schema-4 dataset for the explicit-belief Stage 3C action policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.belief_dataset import (
    BeliefDatasetError,
    BeliefEpisodeRecord,
    GroundedTrackBeliefDataset,
    _read_json,
    _read_jsonl,
    _world_to_start_local,
    apply_d4_transform,
    collate_belief_episodes,
)
from learning.policy_geometry import (
    LOCAL_GEOMETRY_CHANNELS,
    LocalGeometryConfig,
    build_local_geometry,
    projection_matrix_from_spec,
)


ACTION_NAMES = (
    "FORWARD",
    "ASCEND",
    "DESCEND",
    "TURNLEFT",
    "TURNRIGHT",
    "HOLD",
    "STOP",
)
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
ACTION_HISTORY_PAD = len(ACTION_NAMES)
ACTION_HISTORY_SIZE = len(ACTION_NAMES) + 1
ACTION_HISTORY_LENGTH = 4
POLICY_MAX_TRACKS = 34
SOURCE_FEATURE_NAMES = (
    "visible",
    "consecutive_age_over_horizon",
    "body_forward_over_radius",
    "body_right_over_radius",
    "body_up_over_vertical_bound",
    "horizontal_range_over_radius",
    "bearing_sine",
    "bearing_cosine",
    "bbox_center_x_normalized",
    "bbox_center_y_normalized",
    "bbox_span_over_256_pixels",
    "view_is_nadir",
)


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def source_features_from_track(
    track, odometry, width, height, radius, vertical_bound, age, horizon
):
    """Encode one directly grounded source using the offline/online contract."""
    source = np.asarray(_field(track, "position_local"), dtype=np.float64)
    agent = np.asarray(_field(odometry, "position_local"), dtype=np.float64)
    if source.shape != (3,) or agent.shape != (3,) or not np.isfinite(source).all():
        raise BeliefDatasetError("Invalid grounded FIRE_SOURCE position")
    yaw = math.radians(float(_field(odometry, "yaw_from_start_degrees")))
    delta = source - agent
    body_forward = math.cos(yaw) * delta[0] - math.sin(yaw) * delta[1]
    body_right = math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
    horizontal = math.hypot(body_forward, body_right)
    bearing = math.atan2(body_right, body_forward)
    bbox = np.asarray(_field(track, "projected_bbox"), dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise BeliefDatasetError("Invalid grounded FIRE_SOURCE bbox")
    center_x = 0.5 * (bbox[0] + bbox[2])
    center_y = 0.5 * (bbox[1] + bbox[3])
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    features = np.asarray(
        (
            1.0,
            min(age, horizon) / float(horizon),
            body_forward / radius,
            body_right / radius,
            delta[2] / vertical_bound,
            horizontal / radius,
            math.sin(bearing),
            math.cos(bearing),
            2.0 * center_x / (width - 1.0) - 1.0,
            2.0 * center_y / (height - 1.0) - 1.0,
            span / 256.0,
            float(_field(track, "view_name") == "nadir"),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise BeliefDatasetError("Non-finite FIRE_SOURCE features")
    return features, source.astype(np.float32)


def pose_features_from_odometry(
    odometry, radius_m, vertical_bound_m, horizon_steps, observation_step
):
    """Return the exact six pose features used by schema-4 policy training."""
    # BeliefEpisodeDataset quantizes recorded odometry to float32 before
    # normalization. Keep the online helper bit-compatible with that contract.
    position = np.asarray(_field(odometry, "position_local"), dtype=np.float32)
    yaw = float(_field(odometry, "yaw_from_start_degrees"))
    radius_m = float(radius_m)
    vertical_bound_m = float(vertical_bound_m)
    horizon_steps = int(horizon_steps)
    observation_step = int(observation_step)
    if (
        position.shape != (3,)
        or not np.isfinite(position).all()
        or not math.isfinite(yaw)
        or radius_m <= 0.0
        or vertical_bound_m <= 0.0
        or horizon_steps <= 0
        or not 1 <= observation_step <= horizon_steps
    ):
        raise BeliefDatasetError("Invalid online policy pose input")
    yaw_radians = math.radians(yaw)
    return np.asarray(
        (
            position[0] / radius_m,
            position[1] / radius_m,
            position[2] / vertical_bound_m,
            math.sin(yaw_radians),
            math.cos(yaw_radians),
            (horizon_steps - observation_step) / float(horizon_steps),
        ),
        dtype=np.float32,
    )


class StructuredBeliefPolicyDataset(Dataset):
    """One item is one aligned expert episode with structured policy inputs."""

    def __init__(
        self,
        records: Sequence[BeliefEpisodeRecord],
        augment_d4: bool = False,
        augmentation_seed: int = 0,
        geometry_config: LocalGeometryConfig | None = None,
        cache_items: bool = True,
    ):
        self.records = tuple(records)
        if not self.records:
            raise BeliefDatasetError("records must not be empty")
        self.belief_dataset = GroundedTrackBeliefDataset(self.records)
        self.grid_spec = self.belief_dataset.grid_spec
        self.augment_d4 = bool(augment_d4)
        self.augmentation_seed = int(augmentation_seed)
        self.geometry_config = LocalGeometryConfig() if geometry_config is None else geometry_config
        self.geometry_config.validate()
        self.cache_items = bool(cache_items)
        self._cache: dict[int, dict] = {}

    def __len__(self):
        return len(self.records)

    def _load(self, index: int) -> dict:
        record = self.records[index]
        root = record.episode_root
        item = self.belief_dataset[index]
        current_tracks = int(item["track_features"].shape[1])
        if current_tracks > POLICY_MAX_TRACKS:
            raise BeliefDatasetError(
                f"Policy episode declares {current_tracks} tracks; maximum is {POLICY_MAX_TRACKS}"
            )
        if current_tracks < POLICY_MAX_TRACKS:
            steps = int(item["length"])
            feature_count = int(item["track_features"].shape[2])
            features = torch.zeros(steps, POLICY_MAX_TRACKS, feature_count)
            classes = torch.zeros(steps, POLICY_MAX_TRACKS, dtype=torch.long)
            mask = torch.zeros(steps, POLICY_MAX_TRACKS, dtype=torch.bool)
            features[:, :current_tracks] = item["track_features"]
            classes[:, :current_tracks] = item["track_classes"]
            mask[:, :current_tracks] = item["track_mask"]
            item["track_features"] = features
            item["track_classes"] = classes
            item["track_mask"] = mask
        agent_episode = _read_json(root / "agent" / "episode.json")
        truth_episode = _read_json(root / "evaluation_truth" / "episode.json")
        agent_steps = _read_jsonl(root / "agent" / "steps.jsonl")
        awareness_steps = _read_jsonl(root / "teacher" / "awareness.jsonl")
        truth_steps = _read_jsonl(root / "evaluation_truth" / "steps.jsonl")
        length = int(item["length"])
        if not (len(agent_steps) == len(awareness_steps) == len(truth_steps) == length):
            raise BeliefDatasetError(f"Policy episode streams are misaligned: {root}")
        observation_spec = agent_episode["observation_spec"]
        episode_spec = agent_episode["episode_spec"]
        width = int(observation_spec["width"])
        height = int(observation_spec["height"])
        horizon = int(episode_spec["horizon_steps"])
        vertical_bound = float(episode_spec["activity_vertical_m"])
        projection = projection_matrix_from_spec(observation_spec)
        pitches = (
            float(observation_spec["oblique_pitch_degrees"]),
            float(observation_spec["nadir_pitch_degrees"]),
        )

        action_target = np.empty(length, dtype=np.int64)
        action_history = np.full(
            (length, ACTION_HISTORY_LENGTH), ACTION_HISTORY_PAD, dtype=np.int64
        )
        local_geometry = np.empty(
            (
                length,
                len(LOCAL_GEOMETRY_CHANNELS),
                self.geometry_config.grid_size,
                self.geometry_config.grid_size,
            ),
            dtype=np.float32,
        )
        source_features = np.zeros((length, len(SOURCE_FEATURE_NAMES)), dtype=np.float32)
        source_mask = np.zeros(length, dtype=np.bool_)
        source_position = np.zeros((length, 3), dtype=np.float32)
        action_prefix: list[int] = []
        source_track_id = None
        source_age = 0

        for sequence_index, (agent_row, awareness_row) in enumerate(
            zip(agent_steps, awareness_steps, strict=True)
        ):
            step = sequence_index + 1
            action_name = str(agent_row.get("action", {}).get("type", ""))
            if action_name not in ACTION_TO_INDEX:
                raise BeliefDatasetError(f"Unknown expert action {action_name!r} in {root}")
            action_target[sequence_index] = ACTION_TO_INDEX[action_name]
            history = action_prefix[-ACTION_HISTORY_LENGTH:]
            if history:
                action_history[sequence_index, -len(history):] = history
            action_prefix.append(ACTION_TO_INDEX[action_name])

            tracks = awareness_row.get("grounded_tracks")
            if not isinstance(tracks, list):
                raise BeliefDatasetError(f"grounded_tracks is not a list in {root}")
            sources = [track for track in tracks if track.get("semantic_class") == "FIRE_SOURCE"]
            if len(sources) > 1:
                raise BeliefDatasetError(f"Multiple FIRE_SOURCE tracks in {root} step {step}")
            if sources:
                source = sources[0]
                track_id = int(source["track_id"])
                source_age = source_age + 1 if source_track_id == track_id else 1
                source_track_id = track_id
                features, position = source_features_from_track(
                    source,
                    agent_row["odometry"],
                    width,
                    height,
                    self.grid_spec.radius_m,
                    vertical_bound,
                    source_age,
                    horizon,
                )
                source_features[sequence_index] = features
                source_position[sequence_index] = position
                source_mask[sequence_index] = True
            else:
                source_track_id = None
                source_age = 0

            stem = f"{step:03d}"
            depths = []
            for name, pitch in zip(("oblique", "nadir"), pitches, strict=True):
                path = root / "agent" / "depth" / f"{stem}_{name}.npz"
                try:
                    with np.load(path) as archive:
                        if set(archive.files) != {"depth"}:
                            raise BeliefDatasetError(f"Unexpected depth keys in {path}")
                        depth = np.asarray(archive["depth"], dtype=np.float32)
                except (OSError, ValueError) as error:
                    raise BeliefDatasetError(f"Could not read depth {path}: {error}") from error
                if depth.shape != (height, width):
                    raise BeliefDatasetError(f"Depth shape mismatch in {path}: {depth.shape}")
                depths.append((depth, projection, pitch))
            local_geometry[sequence_index] = build_local_geometry(
                tuple(depths), self.geometry_config
            )

        try:
            absolute_pose = truth_episode["start_blueprint"]["absolute_pose"]
            event_world = truth_steps[0]["event_position"]
        except (KeyError, TypeError) as error:
            raise BeliefDatasetError(f"Missing policy event truth in {root}") from error
        event_xyz = np.asarray(
            _world_to_start_local(absolute_pose, event_world), dtype=np.float32
        )
        value_target = (
            (length - np.arange(length, dtype=np.float32)) / float(horizon)
        )
        item.update(
            {
                "action_target": torch.from_numpy(action_target),
                "action_label_mask": torch.ones(length, dtype=torch.bool),
                "value_label_mask": torch.ones(length, dtype=torch.bool),
                "action_history": torch.from_numpy(action_history),
                "local_geometry": torch.from_numpy(local_geometry),
                "source_features": torch.from_numpy(source_features),
                "source_mask": torch.from_numpy(source_mask),
                "source_position_local": torch.from_numpy(source_position),
                "event_xyz": torch.from_numpy(event_xyz),
                "value_target": torch.from_numpy(value_target),
            }
        )
        return item

    def __getitem__(self, index: int) -> dict:
        if index not in self._cache:
            loaded = self._load(index)
            if self.cache_items:
                self._cache[index] = loaded
        else:
            loaded = self._cache[index]
        item = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in loaded.items()
        }
        if self.augment_d4:
            code = (self.augmentation_seed + index * 5) % 8
            item = apply_policy_d4_transform(item, code)
        else:
            item["d4_code"] = 0
        return item


def _transform_xy(values: torch.Tensor, code: int) -> torch.Tensor:
    result = values.clone()
    if code >= 4:
        result[..., 1] *= -1.0
    for _ in range(code % 4):
        forward = result[..., 0].clone()
        right = result[..., 1].clone()
        result[..., 0] = -right
        result[..., 1] = forward
    return result


def _swap_turns(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    left = ACTION_TO_INDEX["TURNLEFT"]
    right = ACTION_TO_INDEX["TURNRIGHT"]
    result[values == left] = right
    result[values == right] = left
    return result


def apply_policy_d4_transform(item: dict, code: int) -> dict:
    if not 0 <= int(code) < 8:
        raise ValueError("D4 code must lie in [0, 7]")
    result = apply_d4_transform(item, int(code))
    result["event_xyz"][:2] = _transform_xy(result["event_xyz"][:2], int(code))
    result["source_position_local"][..., :2] = _transform_xy(
        result["source_position_local"][..., :2], int(code)
    )
    if int(code) >= 4:
        result["local_geometry"] = result["local_geometry"].flip(-1)
        result["source_features"][..., 3] *= -1.0
        result["source_features"][..., 6] *= -1.0
        result["source_features"][..., 8] *= -1.0
        result["action_target"] = _swap_turns(result["action_target"])
        history = result["action_history"]
        valid = history != ACTION_HISTORY_PAD
        swapped = _swap_turns(history)
        result["action_history"] = torch.where(valid, swapped, history)
    result["d4_code"] = int(code)
    return result


def collate_policy_episodes(items: Iterable[dict]) -> dict:
    items = tuple(items)
    batch = collate_belief_episodes(items)
    batch_size = len(items)
    maximum_steps = int(batch["sequence_mask"].shape[1])
    geometry_shape = tuple(items[0]["local_geometry"].shape[1:])
    source_count = len(SOURCE_FEATURE_NAMES)
    action_target = torch.zeros(batch_size, maximum_steps, dtype=torch.long)
    action_history = torch.full(
        (batch_size, maximum_steps, ACTION_HISTORY_LENGTH),
        ACTION_HISTORY_PAD,
        dtype=torch.long,
    )
    local_geometry = torch.zeros(batch_size, maximum_steps, *geometry_shape)
    source_features = torch.zeros(batch_size, maximum_steps, source_count)
    source_mask = torch.zeros(batch_size, maximum_steps, dtype=torch.bool)
    source_position = torch.zeros(batch_size, maximum_steps, 3)
    value_target = torch.zeros(batch_size, maximum_steps)
    action_label_mask = torch.zeros(batch_size, maximum_steps, dtype=torch.bool)
    value_label_mask = torch.zeros(batch_size, maximum_steps, dtype=torch.bool)
    event_xyz = []
    for batch_index, item in enumerate(items):
        length = int(item["length"])
        action_target[batch_index, :length] = item["action_target"]
        action_label_mask[batch_index, :length] = item.get(
            "action_label_mask", torch.ones(length, dtype=torch.bool)
        )
        value_label_mask[batch_index, :length] = item.get(
            "value_label_mask", torch.zeros(length, dtype=torch.bool)
        )
        action_history[batch_index, :length] = item["action_history"]
        local_geometry[batch_index, :length] = item["local_geometry"]
        source_features[batch_index, :length] = item["source_features"]
        source_mask[batch_index, :length] = item["source_mask"]
        source_position[batch_index, :length] = item["source_position_local"]
        value_target[batch_index, :length] = item["value_target"]
        event_xyz.append(item["event_xyz"])
    batch.update(
        {
            "action_target": action_target,
            "action_label_mask": action_label_mask,
            "action_history": action_history,
            "local_geometry": local_geometry,
            "source_features": source_features,
            "source_mask": source_mask,
            "source_position_local": source_position,
            "event_xyz": torch.stack(event_xyz),
            "value_target": value_target,
            "value_label_mask": value_label_mask,
        }
    )
    return batch
