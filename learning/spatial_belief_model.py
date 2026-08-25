"""Belief-only spatial recurrent updater for source-blind event inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from learning.belief_dataset import (
    SEMANTIC_CLASSES,
    SEMANTIC_CLASS_TO_INDEX,
    TRACK_FEATURE_NAMES,
)


@dataclass(frozen=True)
class SpatialRecurrentBeliefConfig:
    radius_m: float = 120.0
    cell_m: float = 4.0
    grid_size: int = 61
    semantic_embedding_dim: int = 16
    pair_hidden_dim: int = 64
    evidence_channels: int = 16
    recurrent_hidden_dim: int = 32
    use_activation_checkpoint: bool = True

    def validate(self) -> None:
        expected = int(round(2.0 * self.radius_m / self.cell_m + 1.0))
        if self.radius_m <= 0.0 or self.cell_m <= 0.0:
            raise ValueError("Belief radius and cell size must be positive")
        if expected != self.grid_size:
            raise ValueError(
                f"grid_size={self.grid_size} does not match radius/cell ({expected})"
            )
        for name in (
            "semantic_embedding_dim",
            "pair_hidden_dim",
            "evidence_channels",
            "recurrent_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SpatialRecurrentBeliefConfig":
        config = cls(**payload)
        config.validate()
        return config


class SpatialRecurrentBeliefUpdater(nn.Module):
    """Update an explicit log-belief map with no hidden recurrent side state."""

    POSITION_FORWARD = 0
    POSITION_RIGHT = 1
    POSITION_UP = 2
    MOTION_FORWARD = 3
    MOTION_RIGHT = 4
    DISPLACEMENT = 5
    MOTION_VALID = 6
    EVIDENCE_AGE = 7
    BBOX_SPAN = 10
    VIEW_IS_NADIR = 11
    SAME_CLASS_COUNT = 12

    def __init__(self, config: SpatialRecurrentBeliefConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.semantic_embedding = nn.Embedding(
            len(SEMANTIC_CLASSES),
            config.semantic_embedding_dim,
            padding_idx=SEMANTIC_CLASS_TO_INDEX["PAD"],
        )
        scalar_feature_count = 6
        pair_input_dim = config.semantic_embedding_dim + scalar_feature_count + 3
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input_dim, config.pair_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pair_hidden_dim, config.evidence_channels),
            nn.GELU(),
        )
        aggregate_channels = 2 * config.evidence_channels + 1
        self.aggregate_projection = nn.Sequential(
            nn.Conv2d(aggregate_channels, config.evidence_channels, 1),
            nn.GELU(),
        )

        recurrent_input_channels = config.evidence_channels + 2
        self.gate_context = nn.Sequential(
            nn.Conv2d(
                recurrent_input_channels,
                config.recurrent_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                config.recurrent_hidden_dim,
                config.recurrent_hidden_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.GELU(),
        )
        self.gate_head = nn.Conv2d(config.recurrent_hidden_dim, 2, 1)
        self.candidate_context = nn.Sequential(
            nn.Conv2d(
                recurrent_input_channels,
                config.recurrent_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                config.recurrent_hidden_dim,
                config.recurrent_hidden_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.GELU(),
        )
        self.candidate_head = nn.Conv2d(config.recurrent_hidden_dim, 1, 1)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        with torch.no_grad():
            self.gate_head.bias[1] = -2.0
        nn.init.zeros_(self.candidate_head.weight)
        nn.init.zeros_(self.candidate_head.bias)

        coordinates = torch.linspace(
            -config.radius_m,
            config.radius_m,
            config.grid_size,
            dtype=torch.float32,
        )
        forward, right = torch.meshgrid(coordinates, coordinates, indexing="ij")
        valid = forward.square() + right.square() <= config.radius_m**2 + 1.0e-4
        self.register_buffer("grid_forward", forward.clone(), persistent=True)
        self.register_buffer("grid_right", right.clone(), persistent=True)
        self.register_buffer("valid_mask", valid, persistent=True)

    def initial_log_belief(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        uniform = -torch.log(self.valid_mask.sum().to(dtype=dtype))
        values = torch.full(
            (batch_size, self.config.grid_size, self.config.grid_size),
            -torch.inf,
            device=self.valid_mask.device,
            dtype=dtype,
        )
        return torch.where(self.valid_mask.unsqueeze(0), uniform, values)

    def _valid_track_mask(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> torch.Tensor:
        source = SEMANTIC_CLASS_TO_INDEX["FIRE_SOURCE"]
        pad = SEMANTIC_CLASS_TO_INDEX["PAD"]
        return (
            track_mask
            & (track_features[..., self.MOTION_VALID] > 0.5)
            & (track_classes != source)
            & (track_classes != pad)
        )

    def evidence_for_step(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if track_features.ndim != 3:
            raise ValueError("Step track_features must be [batch, tracks, features]")
        if track_features.shape[-1] != len(TRACK_FEATURE_NAMES):
            raise ValueError("Unexpected track feature layout")
        if track_classes.shape != track_mask.shape:
            raise ValueError("Track class and mask shapes do not match")
        valid_tracks = self._valid_track_mask(
            track_features, track_classes, track_mask
        )
        batch_size, track_count = track_classes.shape

        origin_forward = (
            track_features[..., self.POSITION_FORWARD] * self.config.radius_m
        )[..., None, None]
        origin_right = (
            track_features[..., self.POSITION_RIGHT] * self.config.radius_m
        )[..., None, None]
        delta_forward = self.grid_forward[None, None] - origin_forward
        delta_right = self.grid_right[None, None] - origin_right
        distance_m = torch.sqrt(
            delta_forward.square() + delta_right.square() + 1.0e-8
        )
        candidate_forward = delta_forward / distance_m
        candidate_right = delta_right / distance_m
        motion_forward = track_features[..., self.MOTION_FORWARD, None, None]
        motion_right = track_features[..., self.MOTION_RIGHT, None, None]
        cosine = candidate_forward * motion_forward + candidate_right * motion_right
        sine = motion_forward * candidate_right - motion_right * candidate_forward
        geometry = torch.stack(
            (
                (distance_m / (2.0 * self.config.radius_m)).clamp_max(1.0),
                cosine.clamp(-1.0, 1.0),
                sine.clamp(-1.0, 1.0),
            ),
            dim=-1,
        )

        scalar_indices = (
            self.DISPLACEMENT,
            self.POSITION_UP,
            self.BBOX_SPAN,
            self.VIEW_IS_NADIR,
            self.EVIDENCE_AGE,
            self.SAME_CLASS_COUNT,
        )
        scalar = track_features[..., scalar_indices]
        semantic = self.semantic_embedding(track_classes)
        token = torch.cat((semantic, scalar), dim=-1)
        height, width = self.grid_forward.shape
        token = token[..., None, None, :].expand(
            batch_size, track_count, height, width, token.shape[-1]
        )
        pair_input = torch.cat((token, geometry), dim=-1)
        pair = self.pair_encoder(pair_input)
        pair_mask = valid_tracks[..., None, None, None]
        pair_sum = torch.where(pair_mask, pair, torch.zeros_like(pair)).sum(dim=1)
        count = valid_tracks.sum(dim=1).clamp_min(1).to(pair.dtype)
        pair_mean = pair_sum / count[:, None, None, None]
        negative_infinity = torch.full_like(pair, -torch.inf)
        pair_max = torch.where(pair_mask, pair, negative_infinity).amax(dim=1)
        has_evidence = valid_tracks.any(dim=1)
        pair_max = torch.where(
            has_evidence[:, None, None, None],
            pair_max,
            torch.zeros_like(pair_max),
        )
        count_feature = (
            torch.log1p(valid_tracks.sum(dim=1).to(pair.dtype)) / math.log(33.0)
        )[:, None, None, None].expand(batch_size, height, width, 1)
        aggregate = torch.cat((pair_mean, pair_max, count_feature), dim=-1)
        evidence = self.aggregate_projection(aggregate.permute(0, 3, 1, 2))
        evidence = evidence * has_evidence[:, None, None, None].to(evidence.dtype)
        evidence = evidence * self.valid_mask[None, None].to(evidence.dtype)
        return evidence, has_evidence

    def _recurrent_step(
        self,
        previous_log_belief: torch.Tensor,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        update_allowed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(update_allowed.any()):
            batch_size = previous_log_belief.shape[0]
            height = self.config.grid_size
            evidence = previous_log_belief.new_zeros(
                batch_size, self.config.evidence_channels, height, height
            )
            gate = previous_log_belief.new_zeros(batch_size, height, height)
            return previous_log_belief, evidence, gate

        evidence, has_evidence = self.evidence_for_step(
            track_features, track_classes, track_mask
        )
        finite_previous = torch.where(
            self.valid_mask[None],
            previous_log_belief,
            torch.zeros_like(previous_log_belief),
        )
        valid_channel = self.valid_mask[None, None].to(evidence.dtype).expand(
            evidence.shape[0], 1, -1, -1
        )
        recurrent_input = torch.cat(
            (evidence, finite_previous[:, None], valid_channel), dim=1
        )
        gates = torch.sigmoid(self.gate_head(self.gate_context(recurrent_input)))
        reset_gate = gates[:, 0]
        update_gate = gates[:, 1]
        candidate_input = torch.cat(
            (evidence, (reset_gate * finite_previous)[:, None], valid_channel),
            dim=1,
        )
        candidate = self.candidate_head(self.candidate_context(candidate_input))[:, 0]
        mixed = (1.0 - update_gate) * finite_previous + update_gate * candidate
        mixed = torch.where(
            self.valid_mask[None],
            mixed,
            torch.full_like(mixed, -torch.inf),
        )
        normalized = mixed - torch.logsumexp(mixed.flatten(1), dim=1)[:, None, None]
        effective_update = update_allowed & has_evidence
        next_log_belief = torch.where(
            effective_update[:, None, None],
            normalized,
            previous_log_belief,
        )
        evidence = evidence * update_allowed[:, None, None, None].to(
            evidence.dtype
        )
        visible_gate = torch.where(
            effective_update[:, None, None],
            update_gate,
            torch.zeros_like(update_gate),
        )
        visible_gate = visible_gate * self.valid_mask[None].to(visible_gate.dtype)
        return next_log_belief, evidence, visible_gate

    def forward(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        inference_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if track_features.ndim != 4:
            raise ValueError("track_features must be [batch, time, tracks, features]")
        batch_size, time_steps = track_features.shape[:2]
        expected = (batch_size, time_steps)
        if sequence_mask.shape != expected or inference_mask.shape != expected:
            raise ValueError("Sequence/inference mask shape mismatch")
        if bool((inference_mask & ~sequence_mask).any()):
            raise ValueError("inference_mask must be a subset of sequence_mask")

        log_belief = self.initial_log_belief(batch_size, track_features.dtype)
        beliefs = []
        log_beliefs = []
        evidence_maps = []
        update_gates = []
        for step in range(time_steps):
            arguments = (
                log_belief,
                track_features[:, step],
                track_classes[:, step],
                track_mask[:, step],
                inference_mask[:, step],
            )
            if (
                self.training
                and self.config.use_activation_checkpoint
                and torch.is_grad_enabled()
            ):
                log_belief, evidence, gate = checkpoint(
                    self._recurrent_step, *arguments, use_reentrant=False
                )
            else:
                log_belief, evidence, gate = self._recurrent_step(*arguments)
            beliefs.append(log_belief.exp())
            log_beliefs.append(log_belief)
            evidence_maps.append(evidence.detach())
            update_gates.append(gate.detach())
        return {
            "belief": torch.stack(beliefs, dim=1),
            "log_belief": torch.stack(log_beliefs, dim=1),
            "evidence_map": torch.stack(evidence_maps, dim=1),
            "update_gate": torch.stack(update_gates, dim=1),
        }

