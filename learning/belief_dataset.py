"""Strict schema-4 dataset loader for the learned 2-D belief baseline.

The baseline consumes the structured tracks recovered by the existing RGB-D
grounder.  It deliberately does not consume teacher motion evidence, event
affiliation, GTA velocity, or world-coordinate camera state.  Consecutive
motion is reconstructed here from anonymous grounded track observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


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

# Track feature layout.  Keeping this public makes checkpoints and future
# perception front-ends able to declare the exact learned-updater contract.
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


class BeliefDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class BeliefEpisodeRecord:
    episode_root: Path
    anchor_name: str
    episode_id: str


@dataclass(frozen=True)
class BeliefGridSpec:
    radius_m: float
    cell_m: float
    size: int

    @property
    def coordinates(self) -> np.ndarray:
        return np.linspace(
            -self.radius_m,
            self.radius_m,
            self.size,
            dtype=np.float32,
        )

    @property
    def valid_mask(self) -> np.ndarray:
        coordinates = self.coordinates
        forward, right = np.meshgrid(coordinates, coordinates, indexing="ij")
        return forward * forward + right * right <= self.radius_m**2 + 1.0e-5


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BeliefDatasetError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BeliefDatasetError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise BeliefDatasetError(
            f"Could not read JSONL {path}: {error}"
        ) from error
    if not rows:
        raise BeliefDatasetError(f"JSONL file is empty: {path}")
    return rows


def discover_belief_episodes(dataset_root: str | Path) -> tuple[BeliefEpisodeRecord, ...]:
    root = Path(dataset_root).resolve()
    manifest_path = root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 4:
        raise BeliefDatasetError(
            "Learned belief training requires a schema-4 grouped dataset; "
            f"found schema={manifest.get('schema_version')!r} at {manifest_path}"
        )

    records = []
    for belief_path in sorted(root.glob("anchor_*/scene_*/episode_*/teacher/beliefs.npz")):
        episode_root = belief_path.parents[1]
        relative = episode_root.relative_to(root)
        if len(relative.parts) < 3 or not relative.parts[0].startswith("anchor_"):
            raise BeliefDatasetError(f"Unexpected episode path: {episode_root}")
        required = (
            episode_root / "agent" / "episode.json",
            episode_root / "agent" / "steps.jsonl",
            episode_root / "teacher" / "episode.json",
            episode_root / "teacher" / "awareness.jsonl",
            episode_root / "evaluation_truth" / "episode.json",
            episode_root / "evaluation_truth" / "steps.jsonl",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise BeliefDatasetError(
                f"Episode {episode_root} is incomplete; missing={missing}"
            )
        records.append(
            BeliefEpisodeRecord(
                episode_root=episode_root,
                anchor_name=relative.parts[0],
                episode_id="/".join(relative.parts[:3]),
            )
        )
    if not records:
        raise BeliefDatasetError(f"No complete schema-4 episodes found under {root}")
    return tuple(records)


def split_records_by_anchor(
    records: Sequence[BeliefEpisodeRecord],
    validation_anchors: Sequence[str] | None = None,
) -> tuple[tuple[BeliefEpisodeRecord, ...], tuple[BeliefEpisodeRecord, ...]]:
    anchors = sorted({record.anchor_name for record in records})
    if len(anchors) < 2:
        raise BeliefDatasetError(
            "At least two anchors are required for a location-disjoint train/validation split"
        )
    if validation_anchors is None:
        count = max(1, int(math.ceil(len(anchors) * 0.2)))
        selected = tuple(anchors[-count:])
    else:
        selected = tuple(validation_anchors)
        unknown = sorted(set(selected) - set(anchors))
        if unknown:
            raise BeliefDatasetError(f"Unknown validation anchors: {unknown}")
        if not selected:
            raise BeliefDatasetError("validation_anchors must not be empty")
    validation_set = set(selected)
    train = tuple(record for record in records if record.anchor_name not in validation_set)
    validation = tuple(record for record in records if record.anchor_name in validation_set)
    if not train or not validation:
        raise BeliefDatasetError(
            "Anchor split must leave at least one episode in both train and validation"
        )
    return train, validation


def _world_to_start_local(
    absolute_pose: Sequence[float], world_position: Sequence[float]
) -> tuple[float, float, float]:
    if len(absolute_pose) != 6 or len(world_position) != 3:
        raise BeliefDatasetError("Invalid pose or event position shape")
    yaw = math.radians(float(absolute_pose[5]))
    forward = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0), dtype=np.float64)
    right = np.asarray((math.cos(yaw), math.sin(yaw), 0.0), dtype=np.float64)
    delta = np.asarray(world_position, dtype=np.float64) - np.asarray(
        absolute_pose[:3], dtype=np.float64
    )
    if not np.isfinite(delta).all():
        raise BeliefDatasetError("Non-finite event transform")
    return (
        float(np.dot(delta, forward)),
        float(np.dot(delta, right)),
        float(delta[2]),
    )


class GroundedTrackBeliefDataset(Dataset):
    """One item is one complete variable-length expert episode."""

    def __init__(self, records: Sequence[BeliefEpisodeRecord]):
        if not records:
            raise BeliefDatasetError("records must not be empty")
        self.records = tuple(records)
        self.grid_spec = self._read_grid_spec(self.records[0])
        for record in self.records[1:]:
            if self._read_grid_spec(record) != self.grid_spec:
                raise BeliefDatasetError(
                    "All episodes must use one identical teacher belief grid"
                )

    @staticmethod
    def _read_grid_spec(record: BeliefEpisodeRecord) -> BeliefGridSpec:
        teacher = _read_json(record.episode_root / "teacher" / "episode.json")
        try:
            radius = float(teacher["belief_radius_m"])
            cell = float(teacher["belief_cell_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise BeliefDatasetError(
                f"Invalid teacher belief metadata in {record.episode_root}"
            ) from error
        size_float = 2.0 * radius / cell + 1.0
        size = int(round(size_float))
        if (
            not math.isfinite(radius)
            or not math.isfinite(cell)
            or radius <= 0.0
            or cell <= 0.0
            or abs(size - size_float) > 1.0e-6
        ):
            raise BeliefDatasetError("Invalid belief grid dimensions")
        return BeliefGridSpec(radius_m=radius, cell_m=cell, size=size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        root = record.episode_root
        agent_episode = _read_json(root / "agent" / "episode.json")
        truth_episode = _read_json(root / "evaluation_truth" / "episode.json")
        agent_steps = _read_jsonl(root / "agent" / "steps.jsonl")
        awareness_steps = _read_jsonl(root / "teacher" / "awareness.jsonl")
        truth_steps = _read_jsonl(root / "evaluation_truth" / "steps.jsonl")
        try:
            with np.load(root / "teacher" / "beliefs.npz") as archive:
                if set(archive.files) != {"belief"}:
                    raise BeliefDatasetError(
                        f"Unexpected belief archive keys in {root}: {archive.files}"
                    )
                teacher_belief = np.asarray(archive["belief"], dtype=np.float32)
        except (OSError, ValueError) as error:
            raise BeliefDatasetError(f"Could not load beliefs for {root}: {error}") from error

        lengths = {
            len(agent_steps),
            len(awareness_steps),
            len(truth_steps),
            int(teacher_belief.shape[0]),
        }
        if len(lengths) != 1:
            raise BeliefDatasetError(
                f"Episode streams are misaligned in {root}: lengths={sorted(lengths)}"
            )
        if teacher_belief.shape[1:] != (
            self.grid_spec.size,
            self.grid_spec.size,
        ):
            raise BeliefDatasetError(
                f"Belief shape mismatch in {root}: {teacher_belief.shape}"
            )
        step_count = len(agent_steps)
        observation = agent_episode.get("observation_spec", {})
        episode_spec = agent_episode.get("episode_spec", {})
        width = int(observation.get("width", 0))
        height = int(observation.get("height", 0))
        vertical_bound = float(episode_spec.get("activity_vertical_m", 0.0))
        horizon = int(episode_spec.get("horizon_steps", 0))
        if width < 2 or height < 2 or vertical_bound <= 0.0 or horizon <= 0:
            raise BeliefDatasetError(f"Invalid episode/observation spec in {root}")

        previous_tracks: dict[int, tuple[int, str, np.ndarray]] = {}
        evidence_counts: dict[int, int] = {}
        step_tracks = []
        maximum_tracks = 0
        poses = []

        for sequence_index, (agent_row, teacher_row) in enumerate(
            zip(agent_steps, awareness_steps)
        ):
            expected_step = sequence_index + 1
            if (
                int(agent_row.get("step_index", -1)) != expected_step
                or int(teacher_row.get("step_index", -1)) != expected_step
            ):
                raise BeliefDatasetError(
                    f"Non-contiguous step index in {root} at sequence index {sequence_index}"
                )
            grounded = teacher_row.get("grounded_tracks")
            if not isinstance(grounded, list):
                raise BeliefDatasetError(f"grounded_tracks is not a list in {root}")
            class_counts: dict[str, int] = {}
            for item in grounded:
                semantic = str(item.get("semantic_class", "UNKNOWN"))
                class_counts[semantic] = class_counts.get(semantic, 0) + 1

            encoded = []
            current_tracks: dict[int, tuple[int, str, np.ndarray]] = {}
            for item in grounded:
                try:
                    track_id = int(item["track_id"])
                    semantic = str(item["semantic_class"])
                    position = np.asarray(item["position_local"], dtype=np.float32)
                    bbox = np.asarray(item["projected_bbox"], dtype=np.float32)
                    view_name = str(item["view_name"])
                except (KeyError, TypeError, ValueError) as error:
                    raise BeliefDatasetError(
                        f"Invalid grounded track in {root} step {expected_step}"
                    ) from error
                if position.shape != (3,) or bbox.shape != (4,):
                    raise BeliefDatasetError("Grounded track shape mismatch")
                if not np.isfinite(position).all() or not np.isfinite(bbox).all():
                    raise BeliefDatasetError("Grounded track contains non-finite values")
                if view_name not in ("oblique", "nadir"):
                    raise BeliefDatasetError(f"Unknown view_name={view_name!r}")

                previous = previous_tracks.get(track_id)
                delta = np.zeros(2, dtype=np.float32)
                displacement = 0.0
                motion_valid = 0.0
                if (
                    previous is not None
                    and previous[0] == expected_step - 1
                    and previous[1] == semantic
                ):
                    delta = position[:2] - previous[2][:2]
                    displacement = float(np.linalg.norm(delta))
                    if displacement >= 0.4:
                        motion_valid = 1.0
                        evidence_counts[track_id] = evidence_counts.get(track_id, 0) + 1
                unit = (
                    delta / displacement
                    if displacement > 1.0e-6
                    else np.zeros(2, dtype=np.float32)
                )
                bbox_center_x = 0.5 * float(bbox[0] + bbox[2])
                bbox_center_y = 0.5 * float(bbox[1] + bbox[3])
                bbox_span = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
                features = np.asarray(
                    (
                        position[0] / self.grid_spec.radius_m,
                        position[1] / self.grid_spec.radius_m,
                        position[2] / vertical_bound,
                        unit[0],
                        unit[1],
                        displacement / 4.0,
                        motion_valid,
                        evidence_counts.get(track_id, 0) / float(horizon),
                        2.0 * bbox_center_x / float(width - 1) - 1.0,
                        2.0 * bbox_center_y / float(height - 1) - 1.0,
                        bbox_span / 256.0,
                        1.0 if view_name == "nadir" else 0.0,
                        class_counts[semantic] / 32.0,
                    ),
                    dtype=np.float32,
                )
                class_index = SEMANTIC_CLASS_TO_INDEX.get(
                    semantic, SEMANTIC_CLASS_TO_INDEX["UNKNOWN"]
                )
                encoded.append((features, class_index))
                current_tracks[track_id] = (expected_step, semantic, position)
            previous_tracks = current_tracks
            step_tracks.append(encoded)
            maximum_tracks = max(maximum_tracks, len(encoded))

            odometry = agent_row.get("odometry", {})
            position_local = np.asarray(odometry.get("position_local"), dtype=np.float32)
            yaw = float(odometry.get("yaw_from_start_degrees", math.nan))
            if position_local.shape != (3,) or not np.isfinite(position_local).all() or not math.isfinite(yaw):
                raise BeliefDatasetError(f"Invalid odometry in {root} step {expected_step}")
            yaw_radians = math.radians(yaw)
            poses.append(
                (
                    position_local[0] / self.grid_spec.radius_m,
                    position_local[1] / self.grid_spec.radius_m,
                    position_local[2] / vertical_bound,
                    math.sin(yaw_radians),
                    math.cos(yaw_radians),
                    (horizon - expected_step) / float(horizon),
                )
            )

        # Keep a real tensor dimension even when one step observes no tracks.
        maximum_tracks = max(1, maximum_tracks)
        features = np.zeros(
            (step_count, maximum_tracks, len(TRACK_FEATURE_NAMES)), dtype=np.float32
        )
        classes = np.zeros((step_count, maximum_tracks), dtype=np.int64)
        track_mask = np.zeros((step_count, maximum_tracks), dtype=np.bool_)
        for step_index, encoded in enumerate(step_tracks):
            for track_index, (track_features, class_index) in enumerate(encoded):
                features[step_index, track_index] = track_features
                classes[step_index, track_index] = class_index
                track_mask[step_index, track_index] = True

        teacher_belief = np.asarray(teacher_belief, dtype=np.float32)
        if not np.isfinite(teacher_belief).all() or np.any(teacher_belief < 0.0):
            raise BeliefDatasetError(f"Invalid teacher belief in {root}")
        totals = teacher_belief.reshape(step_count, -1).sum(axis=1)
        if not np.allclose(totals, 1.0, atol=2.0e-4):
            raise BeliefDatasetError(f"Teacher belief is not normalized in {root}")

        try:
            absolute_pose = truth_episode["start_blueprint"]["absolute_pose"]
            event_world = truth_steps[0]["event_position"]
        except (KeyError, TypeError) as error:
            raise BeliefDatasetError(f"Missing event transform truth in {root}") from error
        event_local = _world_to_start_local(absolute_pose, event_world)
        event_row = int(round((event_local[0] + self.grid_spec.radius_m) / self.grid_spec.cell_m))
        event_column = int(round((event_local[1] + self.grid_spec.radius_m) / self.grid_spec.cell_m))
        if not (
            0 <= event_row < self.grid_spec.size
            and 0 <= event_column < self.grid_spec.size
            and self.grid_spec.valid_mask[event_row, event_column]
        ):
            raise BeliefDatasetError(
                f"Event lies outside the belief grid in {root}: local={event_local}"
            )

        return {
            "episode_id": record.episode_id,
            "anchor_name": record.anchor_name,
            "track_features": torch.from_numpy(features),
            "track_classes": torch.from_numpy(classes),
            "track_mask": torch.from_numpy(track_mask),
            "pose_features": torch.tensor(poses, dtype=torch.float32),
            "teacher_belief": torch.from_numpy(teacher_belief),
            "event_cell": torch.tensor((event_row, event_column), dtype=torch.long),
            "event_xy": torch.tensor(event_local[:2], dtype=torch.float32),
            "length": step_count,
        }


def collate_belief_episodes(items: Iterable[dict]) -> dict:
    items = tuple(items)
    if not items:
        raise BeliefDatasetError("Cannot collate an empty batch")
    batch_size = len(items)
    maximum_steps = max(int(item["length"]) for item in items)
    maximum_tracks = max(int(item["track_features"].shape[1]) for item in items)
    feature_count = len(TRACK_FEATURE_NAMES)
    grid_shape = tuple(items[0]["teacher_belief"].shape[1:])

    track_features = torch.zeros(
        batch_size, maximum_steps, maximum_tracks, feature_count, dtype=torch.float32
    )
    track_classes = torch.zeros(
        batch_size, maximum_steps, maximum_tracks, dtype=torch.long
    )
    track_mask = torch.zeros(
        batch_size, maximum_steps, maximum_tracks, dtype=torch.bool
    )
    pose_features = torch.zeros(batch_size, maximum_steps, 6, dtype=torch.float32)
    teacher_belief = torch.zeros(
        batch_size, maximum_steps, *grid_shape, dtype=torch.float32
    )
    sequence_mask = torch.zeros(batch_size, maximum_steps, dtype=torch.bool)
    event_cells = []
    event_xy = []
    episode_ids = []
    anchor_names = []

    for batch_index, item in enumerate(items):
        length = int(item["length"])
        tracks = int(item["track_features"].shape[1])
        track_features[batch_index, :length, :tracks] = item["track_features"]
        track_classes[batch_index, :length, :tracks] = item["track_classes"]
        track_mask[batch_index, :length, :tracks] = item["track_mask"]
        pose_features[batch_index, :length] = item["pose_features"]
        teacher_belief[batch_index, :length] = item["teacher_belief"]
        sequence_mask[batch_index, :length] = True
        event_cells.append(item["event_cell"])
        event_xy.append(item["event_xy"])
        episode_ids.append(item["episode_id"])
        anchor_names.append(item["anchor_name"])

    return {
        "episode_id": tuple(episode_ids),
        "anchor_name": tuple(anchor_names),
        "track_features": track_features,
        "track_classes": track_classes,
        "track_mask": track_mask,
        "pose_features": pose_features,
        "teacher_belief": teacher_belief,
        "sequence_mask": sequence_mask,
        "event_cell": torch.stack(event_cells),
        "event_xy": torch.stack(event_xy),
    }
