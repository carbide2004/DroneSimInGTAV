"""Losses and compact metrics for the Stage 3C policy."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from learning.belief_objective import inference_belief_objective
from learning.policy_dataset import ACTION_NAMES


def _episode_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have equal [batch,time] shapes")
    counts = mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Every episode must contribute at least one supervised row")
    return ((values * mask.to(values.dtype)).sum(dim=1) / counts).mean()


def _optional_episode_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Episode-balanced mean that permits unlabeled DAgger episodes."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have equal [batch,time] shapes")
    counts = mask.sum(dim=1)
    valid = counts > 0
    if not bool(valid.any()):
        return values.sum() * 0.0
    rows = (values * mask.to(values.dtype)).sum(dim=1)
    return (rows[valid] / counts[valid]).mean()


def explicit_policy_objective(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    action_class_weights: torch.Tensor,
    belief_weight: float = 1.0,
    coordinate_weight: float = 2.0,
    value_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    sequence_mask = batch["sequence_mask"]
    logits = prediction["action_logits"]
    if logits.shape[:2] != sequence_mask.shape or logits.shape[-1] != len(ACTION_NAMES):
        raise ValueError("Action logits have an invalid shape")
    action_rows = F.cross_entropy(
        logits.transpose(1, 2),
        batch["action_target"],
        weight=action_class_weights,
        reduction="none",
    )
    action_mask = sequence_mask & batch.get("action_label_mask", sequence_mask)
    action_loss = _optional_episode_mean(action_rows, action_mask)
    inference_mask = batch["inference_mask"] & sequence_mask
    inference_episodes = inference_mask.sum(dim=1) > 0
    if bool(inference_episodes.any()):
        belief = inference_belief_objective(
            {"belief": prediction["belief"][inference_episodes]},
            batch["event_cell"][inference_episodes],
            inference_mask[inference_episodes],
        )["loss"]
    else:
        belief = prediction["belief"].sum() * 0.0

    scale = prediction["event_estimate_local"].new_tensor((120.0, 120.0, 40.0))
    target = batch["event_xyz"][:, None, :].expand_as(prediction["event_estimate_local"])
    coordinate_rows = F.smooth_l1_loss(
        prediction["event_estimate_local"] / scale,
        target / scale,
        reduction="none",
    ).mean(dim=-1)
    source_mask = batch["source_mask"] & sequence_mask
    if bool(source_mask.any()):
        supervised = []
        for row in range(source_mask.shape[0]):
            if bool(source_mask[row].any()):
                supervised.append(coordinate_rows[row][source_mask[row]].mean())
        coordinate_loss = torch.stack(supervised).mean()
    else:
        coordinate_loss = coordinate_rows.sum() * 0.0

    value_rows = F.smooth_l1_loss(
        prediction["remaining_value"], batch["value_target"], reduction="none"
    )
    value_mask = sequence_mask & batch.get(
        "value_label_mask", sequence_mask
    )
    value_loss = _optional_episode_mean(value_rows, value_mask)
    total = (
        action_loss
        + float(belief_weight) * belief
        + float(coordinate_weight) * coordinate_loss
        + float(value_weight) * value_loss
    )
    return {
        "loss": total,
        "action_loss": action_loss,
        "belief_loss": belief,
        "coordinate_loss": coordinate_loss,
        "value_loss": value_loss,
    }


@torch.no_grad()
def action_confusion(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    selected = prediction["action_logits"].argmax(dim=-1)
    target = batch["action_target"]
    mask = batch["sequence_mask"] & batch.get(
        "action_label_mask", batch["sequence_mask"]
    )
    matrix = torch.zeros(
        len(ACTION_NAMES), len(ACTION_NAMES), dtype=torch.long, device=selected.device
    )
    flat_index = target[mask] * len(ACTION_NAMES) + selected[mask]
    matrix.view(-1).scatter_add_(0, flat_index, torch.ones_like(flat_index))
    return matrix


def class_weights_from_counts(counts) -> torch.Tensor:
    values = torch.as_tensor(tuple(counts), dtype=torch.float32)
    if values.shape != (len(ACTION_NAMES),) or bool((values < 0).any()) or not bool((values > 0).any()):
        raise ValueError("Action counts must be non-negative and not all zero")
    total = values.sum()
    # Tiny smoke-test subsets may omit a rare vertical action.  A clipped
    # finite weight keeps the loader testable without inventing examples.
    return torch.sqrt(total / (len(ACTION_NAMES) * values.clamp_min(1.0))).clamp(0.5, 4.0)
