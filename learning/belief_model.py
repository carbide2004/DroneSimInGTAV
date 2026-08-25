"""Interpretable learned evidence updater for a 2-D event belief map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from learning.belief_dataset import SEMANTIC_CLASSES, TRACK_FEATURE_NAMES


@dataclass(frozen=True)
class BeliefUpdaterConfig:
    radius_m: float = 120.0
    cell_m: float = 4.0
    grid_size: int = 61
    semantic_embedding_dim: int = 16
    token_hidden_dim: int = 64
    minimum_sigma_degrees: float = 5.0
    maximum_sigma_degrees: float = 90.0
    maximum_update_weight: float = 2.0
    likelihood_floor: float = 1.0e-4

    def validate(self) -> None:
        expected = int(round(2.0 * self.radius_m / self.cell_m + 1.0))
        if self.radius_m <= 0.0 or self.cell_m <= 0.0:
            raise ValueError("Belief radius and cell size must be positive")
        if expected != self.grid_size:
            raise ValueError(
                f"grid_size={self.grid_size} does not match radius/cell ({expected})"
            )
        if not 0.0 < self.minimum_sigma_degrees < self.maximum_sigma_degrees < 180.0:
            raise ValueError("Invalid learned angular uncertainty range")
        if self.maximum_update_weight <= 0.0:
            raise ValueError("maximum_update_weight must be positive")
        if not 0.0 < self.likelihood_floor < 1.0:
            raise ValueError("likelihood_floor must lie in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BeliefUpdaterConfig":
        config = cls(**payload)
        config.validate()
        return config


class LearnedCueBeliefUpdater(nn.Module):
    """Learn cue direction/reliability, then perform explicit Bayesian fusion.

    The network never predicts a fresh posterior from scratch.  Each grounded
    moving track predicts an event direction relative to its measured motion,
    an angular uncertainty, and an update weight.  Those likelihoods are added
    to the previous log-belief and normalized.  The only temporal state is the
    externally visible belief map.
    """

    POSITION_FORWARD = 0
    POSITION_RIGHT = 1
    MOTION_FORWARD = 3
    MOTION_RIGHT = 4
    MOTION_VALID = 6

    def __init__(self, config: BeliefUpdaterConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.semantic_embedding = nn.Embedding(
            len(SEMANTIC_CLASSES), config.semantic_embedding_dim, padding_idx=0
        )
        input_dim = len(TRACK_FEATURE_NAMES) + config.semantic_embedding_dim + 6
        self.token_encoder = nn.Sequential(
            nn.Linear(input_dim, config.token_hidden_dim),
            nn.GELU(),
            nn.Linear(config.token_hidden_dim, config.token_hidden_dim),
            nn.GELU(),
        )
        # parallel, perpendicular, sigma logit, reliability logit
        self.parameter_head = nn.Linear(config.token_hidden_dim, 4)
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)
        with torch.no_grad():
            # Begin with broad, weak evidence.  Direction is intentionally not
            # initialized to the hand-written fire-truck/pedestrian rule.
            self.parameter_head.bias[2] = 1.0
            self.parameter_head.bias[3] = -2.0

        coordinates = torch.linspace(
            -config.radius_m, config.radius_m, config.grid_size, dtype=torch.float32
        )
        forward, right = torch.meshgrid(coordinates, coordinates, indexing="ij")
        valid = forward.square() + right.square() <= config.radius_m**2 + 1.0e-4
        self.register_buffer("grid_forward", forward, persistent=True)
        self.register_buffer("grid_right", right, persistent=True)
        self.register_buffer("valid_mask", valid, persistent=True)

    def initial_log_belief(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        valid_count = self.valid_mask.sum().to(dtype=dtype)
        uniform = -torch.log(valid_count)
        result = torch.full(
            (batch_size, self.config.grid_size, self.config.grid_size),
            -torch.inf,
            dtype=dtype,
            device=self.valid_mask.device,
        )
        return torch.where(self.valid_mask.unsqueeze(0), uniform, result)

    def cue_parameters(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if track_features.shape[:-1] != track_classes.shape:
            raise ValueError("Track feature/class shapes do not align")
        if pose_features.shape[-1] != 6:
            raise ValueError("pose_features must have six channels")
        pose = pose_features.unsqueeze(-2).expand(
            *track_features.shape[:-1], pose_features.shape[-1]
        )
        semantic = self.semantic_embedding(track_classes)
        encoded = self.token_encoder(torch.cat((track_features, semantic, pose), dim=-1))
        raw = self.parameter_head(encoded)

        # Predict an event direction in the measured motion's local basis.  A
        # positive parallel component means "event lies along the motion";
        # a negative value means "event lies opposite the motion".
        parallel = torch.tanh(raw[..., 0])
        perpendicular = torch.tanh(raw[..., 1])
        motion_forward = track_features[..., self.MOTION_FORWARD]
        motion_right = track_features[..., self.MOTION_RIGHT]
        direction_forward = parallel * motion_forward - perpendicular * motion_right
        direction_right = parallel * motion_right + perpendicular * motion_forward
        norm = torch.sqrt(
            direction_forward.square() + direction_right.square() + 1.0e-8
        )
        direction_forward = direction_forward / norm
        direction_right = direction_right / norm

        sigma_degrees = self.config.minimum_sigma_degrees + torch.sigmoid(
            raw[..., 2]
        ) * (
            self.config.maximum_sigma_degrees
            - self.config.minimum_sigma_degrees
        )
        reliability = (
            torch.sigmoid(raw[..., 3])
            * self.config.maximum_update_weight
            * track_features[..., self.MOTION_VALID]
        )
        return {
            "parallel": parallel,
            "perpendicular": perpendicular,
            "direction_forward": direction_forward,
            "direction_right": direction_right,
            "sigma_degrees": sigma_degrees,
            "reliability": reliability,
        }

    def evidence_for_step(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if track_features.ndim != 3:
            raise ValueError("Step track_features must be [batch, tracks, features]")
        if track_mask.shape != track_classes.shape:
            raise ValueError("track_mask and track_classes do not align")
        parameters = self.cue_parameters(track_features, track_classes, pose_features)

        origin_forward = (
            track_features[..., self.POSITION_FORWARD] * self.config.radius_m
        )[..., None, None]
        origin_right = (
            track_features[..., self.POSITION_RIGHT] * self.config.radius_m
        )[..., None, None]
        delta_forward = self.grid_forward[None, None] - origin_forward
        delta_right = self.grid_right[None, None] - origin_right
        distance = torch.sqrt(delta_forward.square() + delta_right.square() + 1.0e-8)
        candidate_forward = delta_forward / distance
        candidate_right = delta_right / distance
        cosine = (
            candidate_forward * parameters["direction_forward"][..., None, None]
            + candidate_right * parameters["direction_right"][..., None, None]
        ).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        sigma = torch.deg2rad(parameters["sigma_degrees"])[..., None, None]
        log_likelihood = -0.5 * (angle / sigma).square()
        log_likelihood = log_likelihood.clamp(min=math.log(self.config.likelihood_floor))
        weight = parameters["reliability"] * track_mask.to(track_features.dtype)
        log_likelihood = log_likelihood * weight[..., None, None]
        evidence = log_likelihood.sum(dim=1)
        evidence = torch.where(self.valid_mask.unsqueeze(0), evidence, torch.zeros_like(evidence))
        return evidence, parameters

    def forward(
        self,
        track_features: torch.Tensor,
        track_classes: torch.Tensor,
        track_mask: torch.Tensor,
        pose_features: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if track_features.ndim != 4:
            raise ValueError("track_features must be [batch, time, tracks, features]")
        batch_size, time_steps = track_features.shape[:2]
        if sequence_mask.shape != (batch_size, time_steps):
            raise ValueError("sequence_mask shape mismatch")
        log_belief = self.initial_log_belief(batch_size, track_features.dtype)
        beliefs = []
        log_beliefs = []
        evidence_maps = []
        parameter_rows: dict[str, list[torch.Tensor]] = {
            "parallel": [],
            "perpendicular": [],
            "sigma_degrees": [],
            "reliability": [],
        }
        for step in range(time_steps):
            evidence, parameters = self.evidence_for_step(
                track_features[:, step],
                track_classes[:, step],
                track_mask[:, step],
                pose_features[:, step],
            )
            candidate = log_belief + evidence
            candidate = torch.where(
                self.valid_mask.unsqueeze(0), candidate, torch.full_like(candidate, -torch.inf)
            )
            candidate = candidate - torch.logsumexp(
                candidate.flatten(1), dim=1
            )[:, None, None]
            active = sequence_mask[:, step, None, None]
            log_belief = torch.where(active, candidate, log_belief)
            beliefs.append(log_belief.exp())
            log_beliefs.append(log_belief)
            evidence_maps.append(torch.where(active, evidence, torch.zeros_like(evidence)))
            for name in parameter_rows:
                parameter_rows[name].append(parameters[name])
        result = {
            "belief": torch.stack(beliefs, dim=1),
            "log_belief": torch.stack(log_beliefs, dim=1),
            "evidence": torch.stack(evidence_maps, dim=1),
        }
        result.update(
            {name: torch.stack(rows, dim=1) for name, rows in parameter_rows.items()}
        )
        return result
