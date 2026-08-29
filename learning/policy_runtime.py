"""Strict checkpoint I/O for the Stage 3C explicit-belief policy."""

from __future__ import annotations

from pathlib import Path

import torch

from learning.belief_features import SEMANTIC_CLASSES, TRACK_FEATURE_NAMES
from learning.explicit_belief_policy import (
    ExplicitBeliefActionPolicy,
    ExplicitBeliefPolicyConfig,
)
from learning.policy_dataset import (
    ACTION_NAMES,
    POLICY_MAX_TRACKS,
    SOURCE_FEATURE_NAMES,
)
from learning.policy_geometry import LOCAL_GEOMETRY_CHANNELS, LocalGeometryConfig
from learning.spatial_belief_model import SpatialRecurrentBeliefConfig


POLICY_CHECKPOINT_FORMAT = 3
POLICY_MODEL_NAME = "ExplicitBeliefActionPolicy"
POLICY_SPATIAL_ENCODER_CONTRACT = "coarse_4x4_layout_before_linear_embedding"
POLICY_OBSERVATION_FEATURE_CONTRACT = {
    "depth_geometry": "per_episode_projection_to_fixed_6x41x41_body_grid",
    "bbox_center": "normalized_by_episode_width_and_height",
    "bbox_span": "pixels_divided_by_256",
    "online_resolution": "canonical_training_resolution_exact_match",
}
POLICY_BELIEF_BOTTLENECK = "response_tracks_to_belief_only"
POLICY_RECURRENT_STATE = "one_channel_log_belief_only"
POLICY_D4_CONTRACT = {
    "global_rotation_keeps_body_actions": True,
    "reflection_axis": "start_local_right",
    "reflection_swaps_actions": ("TURNLEFT", "TURNRIGHT"),
    "local_geometry_frame": "agent_body",
}
POLICY_SUPERVISION_CONTRACT = {
    "action": "episode_balanced_expert_current_action",
    "belief": "source_blind_inference_window_event_cell_nll",
    "coordinate": "all_source_visible_frames_event_local",
    "value": "base_expert_remaining_actions_only",
}
POLICY_LOSS_WEIGHTS = {
    "action": 1.0, "belief": 1.0, "coordinate": 2.0, "value": 0.1,
}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_policy_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(Path(path).resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != POLICY_CHECKPOINT_FORMAT:
        raise RuntimeError("Unsupported Stage 3C checkpoint format")
    if payload.get("model") != POLICY_MODEL_NAME:
        raise RuntimeError(f"Unexpected Stage 3C model {payload.get('model')!r}")
    contracts = (
        ("semantic_classes", SEMANTIC_CLASSES),
        ("track_feature_names", TRACK_FEATURE_NAMES),
        ("action_names", ACTION_NAMES),
        ("source_feature_names", SOURCE_FEATURE_NAMES),
        ("geometry_channels", LOCAL_GEOMETRY_CHANNELS),
    )
    for key, expected in contracts:
        if tuple(payload.get(key, ())) != tuple(expected):
            raise RuntimeError(f"Stage 3C checkpoint {key} mismatch")
    if payload.get("policy_max_tracks") != POLICY_MAX_TRACKS:
        raise RuntimeError("Stage 3C checkpoint policy_max_tracks mismatch")
    scalar_contracts = (
        ("belief_bottleneck", POLICY_BELIEF_BOTTLENECK),
        ("recurrent_state", POLICY_RECURRENT_STATE),
        ("spatial_encoder_contract", POLICY_SPATIAL_ENCODER_CONTRACT),
    )
    for key, expected in scalar_contracts:
        if payload.get(key) != expected:
            raise RuntimeError(f"Stage 3C checkpoint {key} mismatch")
    mapping_contracts = (
        ("d4_contract", POLICY_D4_CONTRACT),
        ("supervision_contract", POLICY_SUPERVISION_CONTRACT),
        ("loss_weights", POLICY_LOSS_WEIGHTS),
    )
    for key, expected in mapping_contracts:
        if payload.get(key) != expected:
            raise RuntimeError(f"Stage 3C checkpoint {key} mismatch")
    episode_contract = payload.get("episode_contract")
    if not isinstance(episode_contract, dict) or not all(
        isinstance(episode_contract.get(key), dict)
        for key in ("episode_spec", "observation_spec")
    ):
        raise RuntimeError("Stage 3C checkpoint episode contract is invalid")
    resolution_contract = payload.get("observation_resolution_contract")
    if not isinstance(resolution_contract, dict):
        raise RuntimeError("Stage 3C checkpoint observation resolution contract is missing")
    if resolution_contract.get("feature_contract") != POLICY_OBSERVATION_FEATURE_CONTRACT:
        raise RuntimeError("Stage 3C checkpoint observation feature contract mismatch")
    canonical = resolution_contract.get("canonical_online_resolution")
    expected_observation = episode_contract["observation_spec"]
    if (
        not isinstance(canonical, dict)
        or int(canonical.get("width", -1)) != int(expected_observation.get("width", -2))
        or int(canonical.get("height", -1)) != int(expected_observation.get("height", -2))
    ):
        raise RuntimeError("Stage 3C checkpoint canonical online resolution is invalid")
    for split in ("train", "validation"):
        entries = resolution_contract.get(split)
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(
                f"Stage 3C checkpoint {split} observation resolutions are invalid"
            )
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Stage 3C checkpoint {split} resolution entry is invalid"
                )
            width = int(entry.get("width", 0))
            height = int(entry.get("height", 0))
            count = int(entry.get("episodes", 0))
            key = (width, height)
            if width <= 1 or height <= 1 or count <= 0 or key in seen:
                raise RuntimeError(
                    f"Stage 3C checkpoint {split} resolution entry is invalid"
                )
            seen.add(key)
    train_anchors = tuple(payload.get("train_anchors", ()))
    validation_anchors = tuple(payload.get("validation_anchors", ()))
    if (
        not train_anchors
        or not validation_anchors
        or set(train_anchors) & set(validation_anchors)
    ):
        raise RuntimeError("Stage 3C checkpoint anchor split is invalid")
    class_counts = payload.get("action_class_counts", {})
    class_weights = payload.get("action_class_weights", {})
    if set(class_counts) != set(ACTION_NAMES) or set(class_weights) != set(ACTION_NAMES):
        raise RuntimeError("Stage 3C checkpoint action statistics are incomplete")
    weights = torch.as_tensor([class_weights[name] for name in ACTION_NAMES])
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise RuntimeError("Stage 3C checkpoint action weights are invalid")
    initialization = payload.get("belief_initialization")
    if not isinstance(initialization, dict) or len(str(initialization.get("sha256", ""))) != 64:
        raise RuntimeError("Stage 3C checkpoint belief initialization is invalid")
    if int(payload.get("dagger_iteration", -1)) < 0:
        raise RuntimeError("Stage 3C checkpoint DAgger iteration is invalid")

    belief_config = SpatialRecurrentBeliefConfig.from_dict(payload["belief_config"])
    policy_config = ExplicitBeliefPolicyConfig.from_dict(payload["policy_config"])
    geometry_config = LocalGeometryConfig.from_dict(payload["geometry_config"])
    model = ExplicitBeliefActionPolicy(belief_config, policy_config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload, model, geometry_config
