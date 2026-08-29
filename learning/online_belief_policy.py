"""Stateful online runtime for the Stage 3C explicit-belief policy.

The only learned recurrent tensor owned here is the one-channel log-belief.
Grounded response tracks are consumed only by the embedded Spatial RNN.  The
action policy receives the resulting normalized belief and explicit current
state tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from learning.belief_features import (
    GroundedTrackFeatureConfig,
    StreamingGroundedTrackEncoder,
    TRACK_FEATURE_NAMES,
)
from learning.policy_dataset import (
    ACTION_HISTORY_LENGTH,
    ACTION_HISTORY_PAD,
    ACTION_NAMES,
    POLICY_MAX_TRACKS,
    SOURCE_FEATURE_NAMES,
    pose_features_from_odometry,
    source_features_from_track,
)
from learning.policy_geometry import build_local_geometry_from_pair
from learning.policy_runtime import load_policy_checkpoint, resolve_device


class OnlinePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnlinePolicySnapshot:
    step_index: int
    belief: np.ndarray
    log_belief: np.ndarray
    belief_updated: bool
    evidence_track_ids: tuple[int, ...]
    inference_started: bool
    source_visible_now: bool
    source_seen: bool
    entropy: float
    map_cell: tuple[int, int]
    map_local_xy: tuple[float, float]
    credible_areas_m2: tuple[float, float, float]
    raw_action_probabilities: tuple[float, ...]
    legal_action_probabilities: tuple[float, ...]
    selected_action_index: int
    selected_action_name: str
    remaining_value: float
    event_estimate_local: tuple[float, float, float]
    source_track_id: int | None
    source_age: int
    pose_features: tuple[float, ...]
    action_history: tuple[int, ...]


def _source_track(grounded):
    sources = tuple(
        track for track in grounded.tracks
        if str(track.semantic_class) == "FIRE_SOURCE"
    )
    if len(sources) > 1:
        raise OnlinePolicyError("MULTIPLE_GROUNDED_FIRE_SOURCES")
    return None if not sources else sources[0]


class OnlineExplicitBeliefPolicyRuntime:
    """One episode of strict streaming Stage 3C inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        feature_config: GroundedTrackFeatureConfig,
        horizon_steps: int,
        vertical_bound_m: float,
        device: str | torch.device = "auto",
    ):
        self.device = resolve_device(device) if isinstance(device, str) else torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint, self.model, self.geometry_config = load_policy_checkpoint(
            self.checkpoint_path, self.device
        )
        if abs(self.model.belief_config.radius_m - feature_config.radius_m) > 1.0e-6:
            raise RuntimeError("Checkpoint/online belief radius mismatch")
        self.feature_config = feature_config
        self.horizon_steps = int(horizon_steps)
        self.vertical_bound_m = float(vertical_bound_m)
        if self.horizon_steps <= 0 or self.vertical_bound_m <= 0.0:
            raise ValueError("Invalid online episode specification")
        contract = self.checkpoint.get("episode_contract")
        if not isinstance(contract, dict):
            raise RuntimeError("Stage 3C checkpoint is missing episode_contract")
        expected_episode = contract.get("episode_spec", {})
        expected_observation = contract.get("observation_spec", {})
        if int(expected_episode.get("horizon_steps", -1)) != self.horizon_steps:
            raise RuntimeError("Checkpoint/online action horizon mismatch")
        if abs(float(expected_episode.get("activity_vertical_m", -1.0)) - self.vertical_bound_m) > 1.0e-6:
            raise RuntimeError("Checkpoint/online vertical activity bound mismatch")
        if (
            int(expected_observation.get("width", -1)) != feature_config.width
            or int(expected_observation.get("height", -1)) != feature_config.height
        ):
            raise RuntimeError("Checkpoint/online RGB-D resolution mismatch")
        self.encoder = StreamingGroundedTrackEncoder(feature_config)
        self._log_belief = None
        self._inference_started = False
        self._action_history: list[int] = []
        self._source_track_id = None
        self._source_age = 0
        self._last_snapshot = None
        self._last_training_row = None
        self.reset()

    @property
    def last_snapshot(self):
        return self._last_snapshot

    @property
    def recurrent_log_belief(self):
        return self._log_belief

    @property
    def last_training_row(self):
        if self._last_training_row is None:
            return None
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._last_training_row.items()
        }

    def reset(self):
        self.encoder.reset()
        self._log_belief = self.model.belief_updater.initial_log_belief(
            1, torch.float32
        ).to(self.device)
        self._inference_started = False
        self._action_history = []
        self._source_track_id = None
        self._source_age = 0
        self._last_snapshot = None
        self._last_training_row = None

    def commit_action_name(self, action_name: str):
        action_name = str(action_name)
        if action_name not in ACTION_NAMES:
            raise OnlinePolicyError(f"UNKNOWN_EXECUTED_ACTION: {action_name}")
        self._action_history.append(ACTION_NAMES.index(action_name))

    def _track_tensors(self, encoded):
        count = len(encoded.track_classes)
        if count > POLICY_MAX_TRACKS:
            raise OnlinePolicyError(
                f"TOO_MANY_GROUNDED_TRACKS: {count}>{POLICY_MAX_TRACKS}"
            )
        padded = POLICY_MAX_TRACKS
        features = np.zeros((padded, len(TRACK_FEATURE_NAMES)), dtype=np.float32)
        classes = np.zeros(padded, dtype=np.int64)
        mask = np.zeros(padded, dtype=np.bool_)
        if count:
            features[:count] = encoded.track_features
            classes[:count] = encoded.track_classes
            mask[:count] = encoded.track_mask
        return features, classes, mask

    def _source_inputs(self, grounded, odometry):
        source = _source_track(grounded)
        features = np.zeros(len(SOURCE_FEATURE_NAMES), dtype=np.float32)
        position = np.zeros(3, dtype=np.float32)
        if source is None:
            self._source_track_id = None
            self._source_age = 0
            return features, False, position, None
        track_id = int(source.track_id)
        self._source_age = self._source_age + 1 if self._source_track_id == track_id else 1
        self._source_track_id = track_id
        features, position = source_features_from_track(
            source,
            odometry,
            self.feature_config.width,
            self.feature_config.height,
            self.feature_config.radius_m,
            self.vertical_bound_m,
            self._source_age,
            self.horizon_steps,
        )
        return features, True, position, track_id

    def _history_input(self):
        history = np.full(ACTION_HISTORY_LENGTH, ACTION_HISTORY_PAD, dtype=np.int64)
        selected = self._action_history[-ACTION_HISTORY_LENGTH:]
        if selected:
            history[-len(selected):] = selected
        return history

    @staticmethod
    def _belief_summary(model, belief):
        valid = model.belief_updater.valid_mask.detach().cpu().numpy()
        if (
            not np.isfinite(belief[valid]).all()
            or np.any(belief < 0.0)
            or np.any(belief[~valid] != 0.0)
            or not np.isclose(float(belief.sum()), 1.0, atol=2.0e-5)
        ):
            raise OnlinePolicyError("ONLINE_POLICY_BELIEF_INVALID")
        values = belief[valid]
        entropy = float(-np.sum(values[values > 0.0] * np.log(values[values > 0.0])))
        masked = np.where(valid, belief, -1.0)
        row, column = np.unravel_index(int(np.argmax(masked)), belief.shape)
        map_xy = (
            float(model.belief_updater.grid_forward[row, column].item()),
            float(model.belief_updater.grid_right[row, column].item()),
        )
        ordered = np.sort(values)[::-1]
        cumulative = np.cumsum(ordered)
        counts = tuple(
            int(np.searchsorted(cumulative, threshold, side="left") + 1)
            for threshold in (0.5, 0.8, 0.9)
        )
        cell_area = float(model.belief_config.cell_m) ** 2
        return entropy, (int(row), int(column)), map_xy, tuple(
            float(count * cell_area) for count in counts
        )

    @torch.no_grad()
    def update(self, grounded, observation, pair, action_count: int) -> OnlinePolicySnapshot:
        action_count = int(action_count)
        if not 0 <= action_count < self.horizon_steps:
            raise OnlinePolicyError("HORIZON_EXHAUSTED")
        encoded = self.encoder.encode(grounded)
        if encoded.has_motion_evidence:
            self._inference_started = True
        update_allowed = self._inference_started and not encoded.source_seen
        belief_updated = update_allowed and encoded.has_motion_evidence
        track_features, track_classes, track_mask = self._track_tensors(encoded)
        source_features, source_mask, source_position, source_track_id = (
            self._source_inputs(grounded, observation.odometry)
        )
        pose_features = pose_features_from_odometry(
            observation.odometry,
            self.feature_config.radius_m,
            self.vertical_bound_m,
            self.horizon_steps,
            action_count + 1,
        )
        action_history = self._history_input()
        local_geometry = build_local_geometry_from_pair(pair, self.geometry_config)

        def tensor(value, dtype=None):
            result = torch.from_numpy(value).unsqueeze(0)
            if dtype is not None:
                result = result.to(dtype=dtype)
            return result.to(self.device)

        output = self.model.forward_step(
            self._log_belief,
            tensor(track_features),
            tensor(track_classes),
            tensor(track_mask),
            torch.tensor([update_allowed], dtype=torch.bool, device=self.device),
            tensor(pose_features),
            tensor(action_history),
            tensor(local_geometry),
            tensor(source_features),
            torch.tensor([source_mask], dtype=torch.bool, device=self.device),
            tensor(source_position),
        )
        self._log_belief = output["log_belief"]
        belief = output["belief"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        log_belief = output["log_belief"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        raw_probabilities = output["action_probabilities"][0]
        legal_logits = output["action_logits"][0].clone()
        stop_index = ACTION_NAMES.index("STOP")
        if not source_mask:
            legal_logits[stop_index] = -torch.inf
        if action_count == self.horizon_steps - 1:
            if not source_mask:
                raise OnlinePolicyError(
                    "HORIZON_EXHAUSTED: final action slot has no grounded source"
                )
            forced = torch.full_like(legal_logits, -torch.inf)
            forced[stop_index] = legal_logits[stop_index]
            legal_logits = forced
        if not bool(torch.isfinite(legal_logits).any()):
            raise OnlinePolicyError("NO_LEGAL_POLICY_ACTION")
        legal_probabilities = torch.softmax(legal_logits, dim=-1)
        selected_index = int(torch.argmax(legal_logits).item())
        estimate = output["event_estimate_local"][0].detach().cpu().numpy()
        if not np.isfinite(estimate).all():
            raise OnlinePolicyError("NON_FINITE_EVENT_ESTIMATE")
        entropy, map_cell, map_xy, credible_areas = self._belief_summary(
            self.model, belief
        )
        self._last_training_row = {
            "track_features": track_features,
            "track_classes": track_classes,
            "track_mask": track_mask,
            "inference_mask": bool(update_allowed),
            "motion_evidence": bool(encoded.has_motion_evidence),
            "pose_features": pose_features,
            "action_history": action_history,
            "local_geometry": local_geometry,
            "source_features": source_features,
            "source_mask": bool(source_mask),
            "source_position_local": source_position,
        }
        snapshot = OnlinePolicySnapshot(
            step_index=int(encoded.step_index),
            belief=belief,
            log_belief=log_belief,
            belief_updated=bool(belief_updated),
            evidence_track_ids=tuple(encoded.motion_evidence_track_ids),
            inference_started=bool(self._inference_started),
            source_visible_now=bool(encoded.source_visible_now),
            source_seen=bool(encoded.source_seen),
            entropy=entropy,
            map_cell=map_cell,
            map_local_xy=map_xy,
            credible_areas_m2=credible_areas,
            raw_action_probabilities=tuple(float(v) for v in raw_probabilities.cpu()),
            legal_action_probabilities=tuple(float(v) for v in legal_probabilities.cpu()),
            selected_action_index=selected_index,
            selected_action_name=ACTION_NAMES[selected_index],
            remaining_value=float(output["remaining_value"][0].item()),
            event_estimate_local=tuple(float(v) for v in estimate),
            source_track_id=source_track_id,
            source_age=int(self._source_age),
            pose_features=tuple(float(v) for v in pose_features),
            action_history=tuple(int(v) for v in action_history),
        )
        self._last_snapshot = snapshot
        return snapshot
