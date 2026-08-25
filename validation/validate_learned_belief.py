"""Offline contract and numerical validation for the learned belief baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    GroundedTrackBeliefDataset,
    collate_belief_episodes,
    discover_belief_episodes,
    split_records_by_anchor,
)
from learning.belief_model import (  # noqa: E402
    BeliefUpdaterConfig,
    LearnedCueBeliefUpdater,
)
from learning.belief_objective import belief_training_objective  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate loader, explicit Bayesian update, and gradients offline."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("--episodes must be at least 2")
    return args


def main() -> None:
    args = _arguments()
    records = discover_belief_episodes(args.dataset_root)
    train, validation = split_records_by_anchor(records)
    selected = (train[: max(1, args.episodes - 1)] + validation[:1])[: args.episodes]
    dataset = GroundedTrackBeliefDataset(selected)
    batch = collate_belief_episodes([dataset[index] for index in range(len(dataset))])
    grid = dataset.grid_spec
    model = LearnedCueBeliefUpdater(
        BeliefUpdaterConfig(
            radius_m=grid.radius_m,
            cell_m=grid.cell_m,
            grid_size=grid.size,
        )
    )
    prediction = model(
        batch["track_features"],
        batch["track_classes"],
        batch["track_mask"],
        batch["pose_features"],
        batch["sequence_mask"],
    )
    active_beliefs = prediction["belief"][batch["sequence_mask"]]
    if not torch.isfinite(active_beliefs).all():
        raise RuntimeError("Learned belief contains non-finite probabilities")
    if (active_beliefs < 0.0).any():
        raise RuntimeError("Learned belief contains negative probabilities")
    totals = active_beliefs.sum(dim=(-2, -1))
    if not torch.allclose(totals, torch.ones_like(totals), atol=1.0e-5):
        raise RuntimeError("Learned belief is not normalized")
    invalid_probability = active_beliefs[:, ~model.valid_mask].abs().max()
    if float(invalid_probability.item()) != 0.0:
        raise RuntimeError("Learned belief leaked outside the circular map")

    objective = belief_training_objective(
        prediction, batch["event_cell"], batch["sequence_mask"]
    )
    lengths = batch["sequence_mask"].sum(dim=1)
    batch_indices = torch.arange(len(dataset))
    final_event_probability = prediction["belief"][
        batch_indices,
        lengths - 1,
        batch["event_cell"][:, 0],
        batch["event_cell"][:, 1],
    ]
    manual_final_nll = -final_event_probability.clamp_min(1.0e-30).log().mean()
    if not torch.allclose(
        objective["loss"], manual_final_nll, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError(
            "Training objective is not exactly the terminal event-cell NLL"
        )
    objective["loss"].backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not torch.isfinite(objective["loss"]) or gradient_sum <= 0.0:
        raise RuntimeError("Belief objective did not produce a finite non-zero gradient")

    motion_tracks = int(
        (
            batch["track_mask"]
            & batch["sequence_mask"].unsqueeze(-1)
            & (batch["track_features"][..., model.MOTION_VALID] > 0.5)
        ).sum().item()
    )
    print(
        f"PASS terminal_only=true episodes={len(dataset)} "
        f"steps={int(batch['sequence_mask'].sum())} "
        f"motion_tracks={motion_tracks} loss={float(objective['loss'].item()):.5f} "
        f"gradient_sum={gradient_sum:.3f} anchors="
        f"{sorted(set(batch['anchor_name']))}",
        flush=True,
    )
    print("No RGB-D, belief, or checkpoint payload was written to disk.", flush=True)


if __name__ == "__main__":
    main()
