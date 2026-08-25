"""Shared checkpoint and evaluation helpers for the Spatial RNN baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning.belief_dataset import SEMANTIC_CLASSES, TRACK_FEATURE_NAMES
from learning.belief_objective import belief_metrics, inference_belief_objective
from learning.spatial_belief_model import (
    SpatialRecurrentBeliefConfig,
    SpatialRecurrentBeliefUpdater,
)


SPATIAL_CHECKPOINT_FORMAT = 1
SPATIAL_MODEL_NAME = "SpatialRecurrentBeliefUpdater"
METRIC_NAMES = (
    "teacher_kl",
    "event_nll",
    "map_error_m",
    "event_rank",
    "teacher_event_rank",
    "top_10_recall",
    "top_50_recall",
    "top_100_recall",
    "entropy",
    "coverage_50",
    "credible_area_50_m2",
    "coverage_80",
    "credible_area_80_m2",
    "coverage_90",
    "credible_area_90_m2",
)


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def spatial_forward(
    model: SpatialRecurrentBeliefUpdater, batch: dict
) -> dict[str, torch.Tensor]:
    return model(
        batch["track_features"],
        batch["track_classes"],
        batch["track_mask"],
        batch["sequence_mask"],
        batch["belief_update_mask"],
    )


def last_true_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("mask must be [batch, time]")
    counts = mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Every episode must contain at least one selected step")
    time = torch.arange(mask.shape[1], device=mask.device)[None, :]
    indices = torch.where(mask, time, torch.full_like(time, -1)).amax(dim=1)
    result = torch.zeros_like(mask)
    result[torch.arange(mask.shape[0], device=mask.device), indices] = True
    return result


def _accumulate_metrics(
    totals: dict[str, float],
    metrics: dict[str, float],
) -> None:
    weight = metrics["steps"]
    totals["steps"] += weight
    for name in METRIC_NAMES:
        totals[name] += metrics[name] * weight


@torch.no_grad()
def evaluate_spatial_model(
    model: SpatialRecurrentBeliefUpdater,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float] | float]:
    model.eval()
    subset_totals = {
        "inference": defaultdict(float),
        "last_source_blind": defaultdict(float),
    }
    loss_sum = 0.0
    episode_count = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        prediction = spatial_forward(model, batch)
        objective = inference_belief_objective(
            prediction, batch["event_cell"], batch["inference_mask"]
        )
        batch_episodes = int(batch["track_features"].shape[0])
        loss_sum += float(objective["episode_nll"].sum().item())
        episode_count += batch_episodes
        masks = {
            "inference": batch["inference_mask"],
            "last_source_blind": last_true_mask(batch["inference_mask"]),
        }
        for name, mask in masks.items():
            metrics = belief_metrics(
                prediction,
                batch["teacher_belief"],
                batch["event_cell"],
                batch["event_xy"],
                mask,
                model.grid_forward,
                model.grid_right,
            )
            _accumulate_metrics(subset_totals[name], metrics)
    if episode_count <= 0:
        raise RuntimeError("Evaluation loader produced no episodes")
    result: dict[str, dict[str, float] | float] = {
        "loss": loss_sum / episode_count,
        "episodes": float(episode_count),
    }
    for subset, totals in subset_totals.items():
        denominator = totals["steps"]
        if denominator <= 0:
            raise RuntimeError(f"Evaluation subset {subset} is empty")
        result[subset] = {
            name: totals[name] / denominator for name in METRIC_NAMES
        } | {"steps": denominator}
    return result


def load_spatial_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[dict, SpatialRecurrentBeliefUpdater]:
    checkpoint_payload = torch.load(
        Path(path).resolve(), map_location=device, weights_only=False
    )
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("format_version") != SPATIAL_CHECKPOINT_FORMAT
    ):
        raise RuntimeError("Unsupported Spatial RNN checkpoint format")
    if checkpoint_payload.get("model") != SPATIAL_MODEL_NAME:
        raise RuntimeError(
            f"Unexpected Spatial RNN model: {checkpoint_payload.get('model')!r}"
        )
    if tuple(checkpoint_payload.get("semantic_classes", ())) != SEMANTIC_CLASSES:
        raise RuntimeError("Checkpoint semantic-class contract does not match")
    if tuple(checkpoint_payload.get("track_feature_names", ())) != TRACK_FEATURE_NAMES:
        raise RuntimeError("Checkpoint track-feature contract does not match")
    if (
        checkpoint_payload.get("source_boundary")
        != "first_grounded_FIRE_SOURCE_exclusive"
    ):
        raise RuntimeError("Checkpoint source-boundary contract does not match")
    if (
        checkpoint_payload.get("recurrent_state")
        != "one_channel_log_belief_only"
    ):
        raise RuntimeError("Checkpoint recurrent-state contract does not match")
    if checkpoint_payload.get("supervision") != "source_blind_inference_nll":
        raise RuntimeError("Checkpoint supervision contract does not match")
    config = SpatialRecurrentBeliefConfig.from_dict(
        checkpoint_payload["model_config"]
    )
    model = SpatialRecurrentBeliefUpdater(config).to(device)
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    model.eval()
    return checkpoint_payload, model


def format_metric_row(
    model_name: str,
    subset_name: str,
    metrics: dict[str, float],
) -> str:
    return (
        f"model={model_name} subset={subset_name} steps={int(metrics['steps'])} "
        f"nll={metrics['event_nll']:.4f} "
        f"map_error={metrics['map_error_m']:.2f}m "
        f"rank={metrics['event_rank']:.1f} "
        f"top10/50/100={metrics['top_10_recall']:.3f}/"
        f"{metrics['top_50_recall']:.3f}/{metrics['top_100_recall']:.3f} "
        f"entropy={metrics['entropy']:.3f} "
        f"coverage50/80/90={metrics['coverage_50']:.3f}/"
        f"{metrics['coverage_80']:.3f}/{metrics['coverage_90']:.3f} "
        f"area50/80/90={metrics['credible_area_50_m2']:.1f}/"
        f"{metrics['credible_area_80_m2']:.1f}/"
        f"{metrics['credible_area_90_m2']:.1f}m2"
    )

