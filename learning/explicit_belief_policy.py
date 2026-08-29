"""Explicit-belief bottleneck policy for Stage 3C.

Response tracks are accepted only by the belief updater.  The action policy
itself consumes the normalized belief, observation-derived geometry, explicit
odometry/action history, and a separate directly observed fire-source token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from learning.policy_dataset import (
    ACTION_HISTORY_LENGTH,
    ACTION_HISTORY_SIZE,
    ACTION_NAMES,
    SOURCE_FEATURE_NAMES,
)
from learning.policy_geometry import LOCAL_GEOMETRY_CHANNELS
from learning.spatial_belief_model import (
    SpatialRecurrentBeliefConfig,
    SpatialRecurrentBeliefUpdater,
)


@dataclass(frozen=True)
class ExplicitBeliefPolicyConfig:
    belief_feature_channels: int = 3
    belief_embedding_dim: int = 128
    geometry_embedding_dim: int = 96
    state_embedding_dim: int = 64
    source_embedding_dim: int = 64
    fusion_hidden_dim: int = 256
    policy_hidden_dim: int = 128
    dropout: float = 0.1
    spatial_pool_size: int = 4
    coordinate_residual_limit_m: float = 8.0

    def validate(self):
        integer_fields = (
            "belief_feature_channels",
            "belief_embedding_dim",
            "geometry_embedding_dim",
            "state_embedding_dim",
            "source_embedding_dim",
            "fusion_hidden_dim",
            "policy_hidden_dim",
            "spatial_pool_size",
        )
        if any(int(getattr(self, field)) <= 0 for field in integer_fields):
            raise ValueError("Policy dimensions must be positive")
        if self.belief_feature_channels != 3:
            raise ValueError("Agent-centric belief input must have three channels")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.coordinate_residual_limit_m) or self.coordinate_residual_limit_m <= 0:
            raise ValueError("coordinate residual limit must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        config = cls(**payload)
        config.validate()
        return config


def _conv_block(in_channels, out_channels, stride=1):
    groups = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


def _spatial_encoder(convolution, channels, embedding_dim, pool_size):
    """Preserve coarse spatial layout before producing a compact embedding."""
    return nn.Sequential(
        *convolution,
        nn.AdaptiveAvgPool2d((pool_size, pool_size)),
        nn.Flatten(),
        nn.Linear(channels * pool_size * pool_size, embedding_dim),
        nn.GELU(),
    )


class ExplicitBeliefActionPolicy(nn.Module):
    """Belief updater plus a memoryless policy over explicit state."""

    def __init__(
        self,
        belief_config: SpatialRecurrentBeliefConfig,
        policy_config: ExplicitBeliefPolicyConfig | None = None,
    ):
        super().__init__()
        belief_config.validate()
        self.policy_config = (
            ExplicitBeliefPolicyConfig() if policy_config is None else policy_config
        )
        self.policy_config.validate()
        self.belief_updater = SpatialRecurrentBeliefUpdater(belief_config)
        policy = self.policy_config
        self.belief_encoder = _spatial_encoder(
            (
                _conv_block(3, 32),
                _conv_block(32, 64, stride=2),
                _conv_block(64, 96, stride=2),
                _conv_block(96, policy.belief_embedding_dim, stride=2),
            ),
            policy.belief_embedding_dim,
            policy.belief_embedding_dim,
            policy.spatial_pool_size,
        )
        self.geometry_encoder = _spatial_encoder(
            (
                _conv_block(len(LOCAL_GEOMETRY_CHANNELS), 32),
                _conv_block(32, 64, stride=2),
                _conv_block(64, policy.geometry_embedding_dim, stride=2),
            ),
            policy.geometry_embedding_dim,
            policy.geometry_embedding_dim,
            policy.spatial_pool_size,
        )
        state_input_dim = 6 + ACTION_HISTORY_LENGTH * ACTION_HISTORY_SIZE
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, policy.state_embedding_dim),
            nn.GELU(),
            nn.Linear(policy.state_embedding_dim, policy.state_embedding_dim),
            nn.GELU(),
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(len(SOURCE_FEATURE_NAMES), policy.source_embedding_dim),
            nn.GELU(),
            nn.Linear(policy.source_embedding_dim, policy.source_embedding_dim),
            nn.GELU(),
        )
        fusion_input = (
            policy.belief_embedding_dim
            + policy.geometry_embedding_dim
            + policy.state_embedding_dim
            + policy.source_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, policy.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(policy.dropout),
            nn.Linear(policy.fusion_hidden_dim, policy.policy_hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(policy.policy_hidden_dim, len(ACTION_NAMES))
        self.value_head = nn.Linear(policy.policy_hidden_dim, 1)
        self.coordinate_head = nn.Linear(policy.policy_hidden_dim, 3)

        coordinates = torch.linspace(
            -belief_config.radius_m,
            belief_config.radius_m,
            belief_config.grid_size,
            dtype=torch.float32,
        )
        body_forward, body_right = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("body_grid_forward", body_forward.clone(), persistent=True)
        self.register_buffer("body_grid_right", body_right.clone(), persistent=True)

    @property
    def belief_config(self):
        return self.belief_updater.config

    def _agent_centric_belief(
        self,
        belief: torch.Tensor,
        log_belief: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> torch.Tensor:
        if belief.ndim != 3 or log_belief.shape != belief.shape:
            raise ValueError("Belief tensors must be [items, height, width]")
        if pose_features.shape != (belief.shape[0], 6):
            raise ValueError("pose_features must be [items, 6]")
        radius = float(self.belief_config.radius_m)
        position_forward = pose_features[:, 0] * radius
        position_right = pose_features[:, 1] * radius
        sine = pose_features[:, 3]
        cosine = pose_features[:, 4]
        body_forward = self.body_grid_forward[None]
        body_right = self.body_grid_right[None]
        start_forward = (
            position_forward[:, None, None]
            + cosine[:, None, None] * body_forward
            + sine[:, None, None] * body_right
        )
        start_right = (
            position_right[:, None, None]
            - sine[:, None, None] * body_forward
            + cosine[:, None, None] * body_right
        )
        sampling_grid = torch.stack((start_right / radius, start_forward / radius), dim=-1)
        valid = self.belief_updater.valid_mask[None].expand_as(belief)
        valid_count = self.belief_updater.valid_mask.sum().to(belief.dtype)
        uniform_log = -torch.log(valid_count)
        finite_log = torch.where(
            valid,
            log_belief,
            torch.full_like(log_belief, float(uniform_log.item()) - 12.0),
        )
        relative_log = ((finite_log - uniform_log) / 12.0).clamp(-1.0, 1.0)
        raw = torch.stack(
            (
                belief * valid_count,
                relative_log,
                valid.to(belief.dtype),
            ),
            dim=1,
        )
        return F.grid_sample(
            raw,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def policy_from_belief(
        self,
        belief: torch.Tensor,
        log_belief: torch.Tensor,
        pose_features: torch.Tensor,
        action_history: torch.Tensor,
        local_geometry: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        source_position_local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute actions without accepting response-track tensors."""
        item_count = belief.shape[0]
        if action_history.shape != (item_count, ACTION_HISTORY_LENGTH):
            raise ValueError("action_history has an invalid shape")
        if local_geometry.shape != (
            item_count,
            len(LOCAL_GEOMETRY_CHANNELS),
            41,
            41,
        ):
            raise ValueError("local_geometry has an invalid shape")
        if source_features.shape != (item_count, len(SOURCE_FEATURE_NAMES)):
            raise ValueError("source_features has an invalid shape")
        if source_mask.shape != (item_count,) or source_position_local.shape != (item_count, 3):
            raise ValueError("source tensors have invalid shapes")
        if bool(((action_history < 0) | (action_history >= ACTION_HISTORY_SIZE)).any()):
            raise ValueError("action_history contains an invalid class")
        belief_image = self._agent_centric_belief(belief, log_belief, pose_features)
        belief_embedding = self.belief_encoder(belief_image)
        geometry_embedding = self.geometry_encoder(local_geometry)
        history = F.one_hot(action_history, num_classes=ACTION_HISTORY_SIZE).to(pose_features.dtype)
        state = torch.cat((pose_features, history.flatten(1)), dim=1)
        state_embedding = self.state_encoder(state)
        visible_source = source_features * source_mask[:, None].to(source_features.dtype)
        source_embedding = self.source_encoder(visible_source)
        fused = self.fusion(
            torch.cat(
                (belief_embedding, geometry_embedding, state_embedding, source_embedding),
                dim=1,
            )
        )
        action_logits = self.action_head(fused)
        value = torch.sigmoid(self.value_head(fused)[:, 0])
        residual = (
            torch.tanh(self.coordinate_head(fused))
            * self.policy_config.coordinate_residual_limit_m
        )
        event_estimate = source_position_local + residual
        return {
            "action_logits": action_logits,
            "action_probabilities": torch.softmax(action_logits, dim=-1),
            "remaining_value": value,
            "event_estimate_local": event_estimate,
            "agent_centric_belief": belief_image,
        }

    def forward_step(
        self,
        previous_log_belief: torch.Tensor,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        update_allowed: torch.Tensor,
        pose_features: torch.Tensor,
        action_history: torch.Tensor,
        local_geometry: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        source_position_local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        belief_output = self.belief_updater.forward_step(
            previous_log_belief,
            track_features,
            track_classes,
            track_mask,
            update_allowed,
        )
        policy_output = self.policy_from_belief(
            belief_output["belief"],
            belief_output["log_belief"],
            pose_features,
            action_history,
            local_geometry,
            source_features,
            source_mask,
            source_position_local,
        )
        return {**belief_output, **policy_output}

    def forward_sequence(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        inference_mask: torch.Tensor,
        pose_features: torch.Tensor,
        action_history: torch.Tensor,
        local_geometry: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        source_position_local: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        belief_output = self.belief_updater(
            track_features,
            track_classes,
            track_mask,
            sequence_mask,
            inference_mask,
        )
        batch, steps = sequence_mask.shape
        if self.training:
            flattened = lambda tensor: tensor.reshape(batch * steps, *tensor.shape[2:])
            policy_output = self.policy_from_belief(
                flattened(belief_output["belief"]),
                flattened(belief_output["log_belief"]),
                flattened(pose_features),
                flattened(action_history),
                flattened(local_geometry),
                flattened(source_features),
                source_mask.reshape(-1),
                flattened(source_position_local),
            )
            reshaped = {
                key: value.reshape(batch, steps, *value.shape[1:])
                for key, value in policy_output.items()
            }
        else:
            # Evaluation uses the exact same batch shape as streaming steps.
            # This avoids CPU/GPU convolution-kernel rounding changes caused
            # solely by flattening time into a much larger batch.
            rows = []
            for step in range(steps):
                rows.append(self.policy_from_belief(
                    belief_output["belief"][:, step],
                    belief_output["log_belief"][:, step],
                    pose_features[:, step],
                    action_history[:, step],
                    local_geometry[:, step],
                    source_features[:, step],
                    source_mask[:, step],
                    source_position_local[:, step],
                ))
            reshaped = {
                key: torch.stack([row[key] for row in rows], dim=1)
                for key in rows[0]
            }
        return {
            "belief": belief_output["belief"],
            "log_belief": belief_output["log_belief"],
            **reshaped,
        }
