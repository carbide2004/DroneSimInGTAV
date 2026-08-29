"""Compact Stage 3C DAgger shards and their training dataset.

Shards contain no RGB, Depth, point clouds, or teacher belief.  Local geometry
is quantized to uint8 and is decoded back to the policy's documented six-channel
range when training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from learning.belief_dataset import BeliefDatasetError
from learning.belief_features import TRACK_FEATURE_NAMES
from learning.policy_dataset import (
    ACTION_HISTORY_LENGTH,
    ACTION_NAMES,
    ACTION_TO_INDEX,
    POLICY_MAX_TRACKS,
    SOURCE_FEATURE_NAMES,
    apply_policy_d4_transform,
)
from learning.policy_geometry import LOCAL_GEOMETRY_CHANNELS


DAGGER_SHARD_FORMAT = 1
DAGGER_ARRAY_FILE = "steps.npz"
DAGGER_METADATA_FILE = "metadata.json"


def encode_geometry_uint8(geometry: np.ndarray) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float32)
    expected = (len(LOCAL_GEOMETRY_CHANNELS), 41, 41)
    if geometry.shape != expected or not np.isfinite(geometry).all():
        raise ValueError(f"DAgger geometry must be finite with shape {expected}")
    result = np.empty(expected, dtype=np.uint8)
    height_index = LOCAL_GEOMETRY_CHANNELS.index("mean_surface_height")
    for channel in range(geometry.shape[0]):
        values = geometry[channel]
        if channel == height_index:
            result[channel] = np.rint((np.clip(values, -1.0, 1.0) + 1.0) * 127.5)
        else:
            if np.any((values != 0.0) & (values != 1.0)):
                raise ValueError("Binary DAgger geometry channels must contain only 0/1")
            result[channel] = values.astype(np.uint8) * 255
    return result


def decode_geometry_uint8(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded)
    if encoded.ndim != 4 or encoded.shape[1:] != (
        len(LOCAL_GEOMETRY_CHANNELS), 41, 41
    ) or encoded.dtype != np.uint8:
        raise BeliefDatasetError("Invalid compressed DAgger geometry tensor")
    result = encoded.astype(np.float32) / 255.0
    height_index = LOCAL_GEOMETRY_CHANNELS.index("mean_surface_height")
    result[:, height_index] = result[:, height_index] * 2.0 - 1.0
    return result


class DaggerShardRecorder:
    """In-memory writer that atomically publishes one compact episode shard."""

    def __init__(self, output_path: str | Path, metadata: dict):
        self.output_path = Path(output_path).resolve()
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"DAgger shard path already exists: {self.output_path}")
        self.partial_path.mkdir(parents=True)
        self.metadata = dict(metadata)
        self.rows: list[dict] = []
        self._finished = False

    def record_step(
        self,
        *,
        step_index: int,
        training_row: dict,
        learner_action: str,
        expert_action: str | None,
        executed_by: str,
        event_local,
        event_cell,
    ) -> None:
        if self._finished or training_row is None:
            raise RuntimeError("DAgger recorder is unavailable or missing a training row")
        if learner_action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown learner action {learner_action!r}")
        if expert_action is not None and expert_action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown expert action {expert_action!r}")
        if executed_by not in ("LEARNER", "EXPERT"):
            raise ValueError("executed_by must be LEARNER or EXPERT")
        event_local = np.asarray(event_local, dtype=np.float32)
        event_cell = np.asarray(event_cell, dtype=np.int64)
        if event_local.shape != (3,) or event_cell.shape != (2,):
            raise ValueError("Invalid DAgger event truth shape")
        row = {
            key: np.asarray(training_row[key]).copy()
            for key in (
                "track_features", "track_classes", "track_mask",
                "pose_features", "action_history", "source_features",
                "source_position_local",
            )
        }
        row.update({
            "step_index": int(step_index),
            "inference_mask": bool(training_row["inference_mask"]),
            "motion_evidence": bool(training_row["motion_evidence"]),
            "source_mask": bool(training_row["source_mask"]),
            "local_geometry": encode_geometry_uint8(training_row["local_geometry"]),
            "learner_action": ACTION_TO_INDEX[learner_action],
            "expert_action": -1 if expert_action is None else ACTION_TO_INDEX[expert_action],
            "executed_by": 1 if executed_by == "EXPERT" else 0,
            "event_local": event_local,
            "event_cell": event_cell,
        })
        if self.rows and row["step_index"] <= self.rows[-1]["step_index"]:
            raise ValueError("DAgger step indices must strictly increase")
        self.rows.append(row)

    @staticmethod
    def _stack(rows, key, dtype=None):
        result = np.stack([row[key] for row in rows])
        return result if dtype is None else result.astype(dtype, copy=False)

    def finish(self, status: str, result=None) -> Path:
        if self._finished:
            raise RuntimeError("DAgger shard was already finalized")
        if not self.rows:
            raise RuntimeError("Cannot finalize an empty DAgger shard")
        event_local = self.rows[0]["event_local"]
        event_cell = self.rows[0]["event_cell"]
        if any(
            not np.array_equal(row["event_cell"], event_cell)
            or not np.allclose(row["event_local"], event_local, atol=1.0e-5)
            for row in self.rows[1:]
        ):
            raise RuntimeError("DAgger event truth changed within one episode")
        arrays = {
            "step_index": np.asarray([row["step_index"] for row in self.rows], dtype=np.int32),
            "track_features": self._stack(self.rows, "track_features", np.float32),
            "track_classes": self._stack(self.rows, "track_classes", np.int64),
            "track_mask": self._stack(self.rows, "track_mask", np.bool_),
            "inference_mask": np.asarray([row["inference_mask"] for row in self.rows], dtype=np.bool_),
            "motion_evidence_mask": np.asarray([row["motion_evidence"] for row in self.rows], dtype=np.bool_),
            "pose_features": self._stack(self.rows, "pose_features", np.float32),
            "action_history": self._stack(self.rows, "action_history", np.int64),
            "local_geometry_u8": self._stack(self.rows, "local_geometry", np.uint8),
            "source_features": self._stack(self.rows, "source_features", np.float32),
            "source_mask": np.asarray([row["source_mask"] for row in self.rows], dtype=np.bool_),
            "source_position_local": self._stack(self.rows, "source_position_local", np.float32),
            "learner_action": np.asarray([row["learner_action"] for row in self.rows], dtype=np.int64),
            "expert_action": np.asarray([row["expert_action"] for row in self.rows], dtype=np.int64),
            "executed_by": np.asarray([row["executed_by"] for row in self.rows], dtype=np.uint8),
            "event_xyz": event_local.astype(np.float32),
            "event_cell": event_cell.astype(np.int64),
        }
        np.savez_compressed(self.partial_path / DAGGER_ARRAY_FILE, **arrays)
        metadata = {
            **self.metadata,
            "format_version": DAGGER_SHARD_FORMAT,
            "status": str(status),
            "steps": len(self.rows),
            "expert_labels": int(np.sum(arrays["expert_action"] >= 0)),
            "executed_by_expert": int(np.sum(arrays["executed_by"])),
            "result": None if result is None else {
                "success": bool(result.success),
                "status": str(result.status),
                "actions": int(result.actions),
                "localization_error_m": result.localization_error_m,
            },
            "contains_rgb": False,
            "contains_depth": False,
            "geometry_encoding": "uint8-binary-plus-signed-height-v1",
        }
        with (self.partial_path / DAGGER_METADATA_FILE).open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(self.partial_path, self.output_path)
        self._finished = True
        return self.output_path

    def abort(self) -> None:
        if not self._finished and self.partial_path.is_dir():
            shutil.rmtree(self.partial_path)
        self._finished = True


def discover_dagger_shards(roots: Sequence[str | Path]) -> tuple[Path, ...]:
    shards = []
    for value in roots:
        root = Path(value).resolve()
        if not root.is_dir():
            raise BeliefDatasetError(f"DAgger root is not a directory: {root}")
        for metadata in sorted(root.rglob(DAGGER_METADATA_FILE)):
            shard = metadata.parent
            if shard.name.endswith(".partial"):
                continue
            if not (shard / DAGGER_ARRAY_FILE).is_file():
                raise BeliefDatasetError(f"Incomplete DAgger shard: {shard}")
            shards.append(shard)
    if not shards:
        raise BeliefDatasetError("No complete DAgger shards were discovered")
    return tuple(shards)


class DaggerPolicyDataset(Dataset):
    def __init__(self, shards: Sequence[Path], augment_d4=False, augmentation_seed=0):
        self.shards = tuple(Path(path).resolve() for path in shards)
        if not self.shards:
            raise BeliefDatasetError("DAgger dataset must not be empty")
        self.augment_d4 = bool(augment_d4)
        self.augmentation_seed = int(augmentation_seed)

    def __len__(self):
        return len(self.shards)

    @staticmethod
    def metadata(path: Path) -> dict:
        try:
            with (path / DAGGER_METADATA_FILE).open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise BeliefDatasetError(f"Could not read DAgger metadata {path}: {error}") from error
        if value.get("format_version") != DAGGER_SHARD_FORMAT:
            raise BeliefDatasetError(f"Unsupported DAgger shard format: {path}")
        return value

    def __getitem__(self, index):
        path = self.shards[index]
        metadata = self.metadata(path)
        try:
            with np.load(path / DAGGER_ARRAY_FILE) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError, KeyError) as error:
            raise BeliefDatasetError(f"Could not load DAgger shard {path}: {error}") from error
        required = {
            "step_index", "track_features", "track_classes", "track_mask",
            "inference_mask", "motion_evidence_mask", "pose_features",
            "action_history", "local_geometry_u8", "source_features",
            "source_mask", "source_position_local", "expert_action",
            "event_xyz", "event_cell",
        }
        if set(arrays) < required:
            raise BeliefDatasetError(f"DAgger shard is missing arrays: {sorted(required-set(arrays))}")
        length = int(arrays["step_index"].shape[0])
        if length <= 0 or arrays["track_features"].shape != (
            length, POLICY_MAX_TRACKS, len(TRACK_FEATURE_NAMES)
        ):
            raise BeliefDatasetError(f"DAgger track tensor mismatch: {path}")
        expert = arrays["expert_action"].astype(np.int64, copy=False)
        label_mask = expert >= 0
        target = np.where(label_mask, expert, 0)
        if np.any(target >= len(ACTION_NAMES)):
            raise BeliefDatasetError(f"DAgger expert action out of range: {path}")
        event_xyz = arrays["event_xyz"].astype(np.float32, copy=False)
        event_cell = arrays["event_cell"].astype(np.int64, copy=False)
        source_mask = arrays["source_mask"].astype(np.bool_, copy=False)
        inference = arrays["inference_mask"].astype(np.bool_, copy=False)
        horizon = int(metadata["horizon_steps"])
        item = {
            "episode_id": str(metadata.get("episode_id", path.name)),
            "anchor_name": str(metadata["anchor_name"]),
            "length": length,
            "track_features": torch.from_numpy(arrays["track_features"].astype(np.float32, copy=False)),
            "track_classes": torch.from_numpy(arrays["track_classes"].astype(np.int64, copy=False)),
            "track_mask": torch.from_numpy(arrays["track_mask"].astype(np.bool_, copy=False)),
            "pose_features": torch.from_numpy(arrays["pose_features"].astype(np.float32, copy=False)),
            "teacher_belief": torch.zeros(length, 61, 61),
            "source_visible_mask": torch.from_numpy(source_mask),
            "motion_evidence_mask": torch.from_numpy(arrays["motion_evidence_mask"].astype(np.bool_, copy=False)),
            "inference_mask": torch.from_numpy(inference),
            "belief_update_mask": torch.from_numpy(inference.copy()),
            "event_cell": torch.from_numpy(event_cell),
            "event_xy": torch.from_numpy(event_xyz[:2].copy()),
            "action_target": torch.from_numpy(target),
            "action_label_mask": torch.from_numpy(label_mask),
            "action_history": torch.from_numpy(arrays["action_history"].astype(np.int64, copy=False)),
            "local_geometry": torch.from_numpy(decode_geometry_uint8(arrays["local_geometry_u8"])),
            "source_features": torch.from_numpy(arrays["source_features"].astype(np.float32, copy=False)),
            "source_mask": torch.from_numpy(source_mask),
            "source_position_local": torch.from_numpy(arrays["source_position_local"].astype(np.float32, copy=False)),
            "event_xyz": torch.from_numpy(event_xyz),
            "value_target": torch.from_numpy((horizon - np.arange(length, dtype=np.float32)) / float(horizon)),
            "value_label_mask": torch.zeros(length, dtype=torch.bool),
            "d4_code": 0,
        }
        if self.augment_d4:
            item = apply_policy_d4_transform(
                item, (self.augmentation_seed + index * 5) % 8
            )
        return item
