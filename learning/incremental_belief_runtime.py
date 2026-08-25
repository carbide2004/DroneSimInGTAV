"""Strict loading and source-blind evaluation for the incremental baseline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning.belief_dataset import SEMANTIC_CLASSES, TRACK_FEATURE_NAMES
from learning.belief_model import BeliefUpdaterConfig, LearnedCueBeliefUpdater
from learning.belief_objective import belief_metrics, inference_belief_objective
from learning.spatial_belief_runtime import METRIC_NAMES, last_true_mask, to_device


def load_incremental_checkpoint(
    path: str | Path, device: torch.device, require_source_blind: bool = False
) -> tuple[dict, LearnedCueBeliefUpdater]:
    payload = torch.load(Path(path).resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise RuntimeError("Unsupported incremental belief checkpoint format")
    if payload.get("model") != "LearnedCueBeliefUpdater":
        raise RuntimeError(f"Unexpected incremental model: {payload.get('model')!r}")
    if tuple(payload.get("semantic_classes", ())) != SEMANTIC_CLASSES:
        raise RuntimeError("Incremental semantic-class contract does not match")
    if tuple(payload.get("track_feature_names", ())) != TRACK_FEATURE_NAMES:
        raise RuntimeError("Incremental track-feature contract does not match")
    supervision = payload.get("supervision")
    if supervision is None:
        legacy = payload.get("training_config", {}).get("supervision")
        supervision = (
            "source_blind_inference_nll" if legacy == "inference" else legacy
        )
    if require_source_blind and supervision != "source_blind_inference_nll":
        raise RuntimeError(
            "Fair comparison requires an incremental checkpoint trained with "
            "--supervision inference"
        )
    config = BeliefUpdaterConfig.from_dict(payload["model_config"])
    model = LearnedCueBeliefUpdater(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload, model


@torch.no_grad()
def evaluate_incremental_model(
    model: LearnedCueBeliefUpdater,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    totals = {
        "inference": defaultdict(float),
        "last_source_blind": defaultdict(float),
    }
    loss_sum = 0.0
    episode_count = 0
    for raw in loader:
        batch = to_device(raw, device)
        prediction = model(
            batch["track_features"],
            batch["track_classes"],
            batch["track_mask"],
            batch["pose_features"],
            batch["sequence_mask"],
        )
        objective = inference_belief_objective(
            prediction, batch["event_cell"], batch["inference_mask"]
        )
        episodes = int(batch["sequence_mask"].shape[0])
        loss_sum += float(objective["episode_nll"].sum().item())
        episode_count += episodes
        masks = {
            "inference": batch["inference_mask"],
            "last_source_blind": last_true_mask(batch["inference_mask"]),
        }
        for subset, mask in masks.items():
            metrics = belief_metrics(
                prediction,
                batch["teacher_belief"],
                batch["event_cell"],
                batch["event_xy"],
                mask,
                model.grid_forward,
                model.grid_right,
            )
            weight = metrics["steps"]
            totals[subset]["steps"] += weight
            for name in METRIC_NAMES:
                totals[subset][name] += metrics[name] * weight
    if episode_count <= 0:
        raise RuntimeError("Evaluation loader produced no episodes")
    result = {"loss": loss_sum / episode_count, "episodes": float(episode_count)}
    for subset, row in totals.items():
        steps = row["steps"]
        if steps <= 0:
            raise RuntimeError(f"Evaluation subset {subset} is empty")
        result[subset] = {
            name: row[name] / steps for name in METRIC_NAMES
        } | {"steps": steps}
    return result
