"""Losses and metrics for the explicit 2-D belief updater."""

from __future__ import annotations

import torch


def _final_step_indices(sequence_mask: torch.Tensor) -> torch.Tensor:
    if sequence_mask.ndim != 2:
        raise ValueError("sequence_mask must be [batch, time]")
    lengths = sequence_mask.sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("Every sequence must contain at least one active step")
    return lengths - 1


def belief_training_objective(
    prediction: dict[str, torch.Tensor],
    event_cell: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted = prediction["belief"].clamp_min(1.0e-30)
    if predicted.ndim != 4 or predicted.shape[:2] != sequence_mask.shape:
        raise ValueError("Predicted belief and sequence mask shapes do not match")
    if event_cell.shape != (predicted.shape[0], 2):
        raise ValueError("event_cell must be [batch, 2]")
    batch_indices = torch.arange(predicted.shape[0], device=predicted.device)
    final_steps = _final_step_indices(sequence_mask)
    event_probability = predicted[
        batch_indices,
        final_steps,
        event_cell[:, 0],
        event_cell[:, 1],
    ]
    event_nll = -event_probability.clamp_min(1.0e-30).log().mean()
    return {
        "loss": event_nll,
        "event_nll": event_nll,
        "event_probability": event_probability.mean(),
    }


def inference_belief_objective(
    prediction: dict[str, torch.Tensor],
    event_cell: torch.Tensor,
    inference_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Episode-balanced NLL over the source-blind inference interval."""
    predicted = prediction["belief"].clamp_min(1.0e-30)
    if predicted.ndim != 4 or predicted.shape[:2] != inference_mask.shape:
        raise ValueError("Predicted belief and inference mask shapes do not match")
    if event_cell.shape != (predicted.shape[0], 2):
        raise ValueError("event_cell must be [batch, 2]")
    counts = inference_mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Every episode must contain a source-blind inference step")
    batch_indices = torch.arange(predicted.shape[0], device=predicted.device)
    time_indices = torch.arange(predicted.shape[1], device=predicted.device)
    event_probability = predicted[
        batch_indices[:, None],
        time_indices[None, :],
        event_cell[:, 0, None],
        event_cell[:, 1, None],
    ]
    per_step_nll = -event_probability.clamp_min(1.0e-30).log()
    weights = inference_mask.to(per_step_nll.dtype)
    episode_nll = (per_step_nll * weights).sum(dim=1) / counts
    episode_probability = (event_probability * weights).sum(dim=1) / counts
    return {
        "loss": episode_nll.mean(),
        "event_nll": episode_nll.mean(),
        "event_probability": episode_probability.mean(),
        "episode_nll": episode_nll,
        "per_step_nll": per_step_nll,
    }


@torch.no_grad()
def belief_metrics(
    prediction: dict[str, torch.Tensor],
    teacher_belief: torch.Tensor,
    event_cell: torch.Tensor,
    event_xy: torch.Tensor,
    sequence_mask: torch.Tensor,
    grid_forward: torch.Tensor,
    grid_right: torch.Tensor,
) -> dict[str, float]:
    belief = prediction["belief"]
    target = teacher_belief.clamp_min(0.0)
    if belief.shape != target.shape:
        raise ValueError("Predicted and teacher belief shapes do not match")
    active = sequence_mask
    active_count = int(active.sum().item())
    if active_count <= 0:
        raise ValueError("No active steps for metrics")
    target_positive = target > 0.0
    target_log = torch.where(
        target_positive, target.clamp_min(1.0e-30).log(), 0.0
    )
    predicted_log = belief.clamp_min(1.0e-30).log()
    teacher_kl_per_step = torch.sum(
        torch.where(
            target_positive,
            target * (target_log - predicted_log),
            torch.zeros_like(target),
        ),
        dim=(-2, -1),
    )
    teacher_kl = teacher_kl_per_step[active].mean()

    flattened = belief.flatten(-2)
    map_index = flattened.argmax(dim=-1)
    grid_size = belief.shape[-1]
    map_row = torch.div(map_index, grid_size, rounding_mode="floor")
    map_column = map_index % grid_size
    map_forward = grid_forward[map_row, map_column]
    map_right = grid_right[map_row, map_column]
    map_error = torch.sqrt(
        (map_forward - event_xy[:, 0, None]).square()
        + (map_right - event_xy[:, 1, None]).square()
    )

    batch_indices = torch.arange(belief.shape[0], device=belief.device)
    event_probability = belief[
        batch_indices[:, None],
        torch.arange(belief.shape[1], device=belief.device)[None, :],
        event_cell[:, 0, None],
        event_cell[:, 1, None],
    ]
    event_greater = (flattened > event_probability[..., None]).sum(dim=-1)
    event_equal = (flattened == event_probability[..., None]).sum(dim=-1).clamp_min(1)
    event_rank = (
        1.0
        + event_greater.to(belief.dtype)
        + 0.5 * (event_equal.to(belief.dtype) - 1.0)
    )

    teacher_flat = teacher_belief.flatten(-2)
    teacher_event_probability = teacher_belief[
        batch_indices[:, None],
        torch.arange(belief.shape[1], device=belief.device)[None, :],
        event_cell[:, 0, None],
        event_cell[:, 1, None],
    ]
    teacher_greater = (
        teacher_flat > teacher_event_probability[..., None]
    ).sum(dim=-1)
    teacher_equal = (
        teacher_flat == teacher_event_probability[..., None]
    ).sum(dim=-1).clamp_min(1)
    teacher_event_rank = 1.0 + teacher_greater.to(belief.dtype) + 0.5 * (
        teacher_equal.to(belief.dtype) - 1.0
    )

    entropy = -torch.sum(
        torch.where(belief > 0.0, belief * predicted_log, torch.zeros_like(belief)),
        dim=(-2, -1),
    )
    sorted_probability = torch.sort(flattened, dim=-1, descending=True).values
    cumulative_probability = sorted_probability.cumsum(dim=-1)
    cell_m = float(torch.abs(grid_forward[1, 0] - grid_forward[0, 0]).item())
    credible = {}
    for percentage, level in ((50, 0.5), (80, 0.8), (90, 0.9)):
        region_size = 1 + (cumulative_probability < level).sum(dim=-1)
        region_size = region_size.clamp_max(flattened.shape[-1])
        covered = (
            (region_size.to(belief.dtype) - event_greater.to(belief.dtype))
            / event_equal.to(belief.dtype)
        ).clamp(0.0, 1.0)
        credible[f"coverage_{percentage}"] = float(
            covered[active].mean().item()
        )
        credible[f"credible_area_{percentage}_m2"] = float(
            (region_size[active].float() * (cell_m**2)).mean().item()
        )

    return {
        "teacher_kl": float(teacher_kl.item()),
        "event_nll": float(
            (-event_probability.clamp_min(1.0e-30).log())[active].mean().item()
        ),
        "map_error_m": float(map_error[active].mean().item()),
        "event_rank": float(event_rank[active].float().mean().item()),
        "teacher_event_rank": float(
            teacher_event_rank[active].float().mean().item()
        ),
        "top_10_recall": float(
            (
                (10 - event_greater).to(belief.dtype) / event_equal
            ).clamp(0.0, 1.0)[active].mean().item()
        ),
        "top_50_recall": float(
            (
                (50 - event_greater).to(belief.dtype) / event_equal
            ).clamp(0.0, 1.0)[active].mean().item()
        ),
        "top_100_recall": float(
            (
                (100 - event_greater).to(belief.dtype) / event_equal
            ).clamp(0.0, 1.0)[active].mean().item()
        ),
        "entropy": float(entropy[active].mean().item()),
        "steps": float(active_count),
        **credible,
    }
