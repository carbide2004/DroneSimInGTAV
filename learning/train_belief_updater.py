"""Train the structured-track 2-D belief updater on schema-4 episodes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    GroundedTrackBeliefDataset,
    SEMANTIC_CLASSES,
    TRACK_FEATURE_NAMES,
    collate_belief_episodes,
    discover_belief_episodes,
    split_records_by_anchor,
)
from learning.belief_model import (  # noqa: E402
    BeliefUpdaterConfig,
    LearnedCueBeliefUpdater,
)
from learning.belief_objective import (  # noqa: E402
    belief_metrics,
    belief_training_objective,
    inference_belief_objective,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an explicit 2-D event belief updater from RGB-D-grounded "
            "anonymous track observations."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning/checkpoints/stage3_belief_updater.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maximum-train-episodes", type=int)
    parser.add_argument("--maximum-validation-episodes", type=int)
    parser.add_argument(
        "--validation-anchors",
        nargs="+",
        help="Explicit anchor_NNN names; default is the final 20%% of anchors",
    )
    parser.add_argument(
        "--supervision",
        choices=("final", "inference"),
        default="final",
        help="Legacy final-step NLL or episode-normalized source-blind NLL",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("Invalid optimizer parameters")
    if args.gradient_clip <= 0.0:
        parser.error("--gradient-clip must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model: LearnedCueBeliefUpdater, batch: dict) -> dict:
    return model(
        batch["track_features"],
        batch["track_classes"],
        batch["track_mask"],
        batch["pose_features"],
        batch["sequence_mask"],
    )


def _objective(prediction: dict, batch: dict, supervision: str) -> dict:
    if supervision == "inference":
        return inference_belief_objective(
            prediction, batch["event_cell"], batch["inference_mask"]
        )
    return belief_training_objective(
        prediction, batch["event_cell"], batch["sequence_mask"]
    )


def _metric_mask(batch: dict, supervision: str, device: torch.device) -> torch.Tensor:
    if supervision == "inference":
        return batch["inference_mask"]
    final_mask = torch.zeros_like(batch["sequence_mask"])
    lengths = batch["sequence_mask"].sum(dim=1)
    final_mask[
        torch.arange(final_mask.shape[0], device=device), lengths - 1
    ] = True
    return final_mask


@torch.no_grad()
def _evaluate(
    model: LearnedCueBeliefUpdater,
    loader: DataLoader,
    device: torch.device,
    supervision: str,
) -> dict[str, float]:
    model.eval()
    objective_sum = 0.0
    episode_count = 0
    totals = {
        "teacher_kl": 0.0,
        "event_nll": 0.0,
        "map_error_m": 0.0,
        "event_rank": 0.0,
        "teacher_event_rank": 0.0,
        "steps": 0.0,
    }
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        prediction = _forward(model, batch)
        objective = _objective(prediction, batch, supervision)
        final_mask = _metric_mask(batch, supervision, device)
        metrics = belief_metrics(
            prediction,
            batch["teacher_belief"],
            batch["event_cell"],
            batch["event_xy"],
            final_mask,
            model.grid_forward,
            model.grid_right,
        )
        batch_episodes = int(batch["sequence_mask"].shape[0])
        objective_sum += float(objective["loss"].item()) * batch_episodes
        episode_count += batch_episodes
        weight = metrics["steps"]
        totals["steps"] += weight
        for name in (
            "teacher_kl",
            "event_nll",
            "map_error_m",
            "event_rank",
            "teacher_event_rank",
        ):
            totals[name] += metrics[name] * weight
    if episode_count <= 0:
        raise RuntimeError("Validation loader produced no episodes")
    if totals["steps"] <= 0:
        raise RuntimeError("Validation loader produced no complete episodes")
    metric_steps = totals["steps"]
    return {
        "loss": objective_sum / episode_count,
        **{
            name: value / metric_steps if name != "steps" else value
            for name, value in totals.items()
        },
    }


def main() -> None:
    args = _arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = discover_belief_episodes(args.dataset_root)
    train_records, validation_records = split_records_by_anchor(
        records, args.validation_anchors
    )
    if args.maximum_train_episodes is not None:
        train_records = train_records[: args.maximum_train_episodes]
    if args.maximum_validation_episodes is not None:
        validation_records = validation_records[: args.maximum_validation_episodes]
    if not train_records or not validation_records:
        raise RuntimeError("Episode limits removed the train or validation split")

    train_dataset = GroundedTrackBeliefDataset(train_records)
    validation_dataset = GroundedTrackBeliefDataset(validation_records)
    if train_dataset.grid_spec != validation_dataset.grid_spec:
        raise RuntimeError("Train and validation belief grids differ")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_belief_episodes,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_belief_episodes,
        pin_memory=torch.cuda.is_available(),
    )

    grid = train_dataset.grid_spec
    config = BeliefUpdaterConfig(
        radius_m=grid.radius_m,
        cell_m=grid.cell_m,
        grid_size=grid.size,
    )
    device = _device(args.device)
    model = LearnedCueBeliefUpdater(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial checkpoint exists: {temporary}")

    train_anchors = sorted({record.anchor_name for record in train_records})
    validation_anchors = sorted(
        {record.anchor_name for record in validation_records}
    )
    print(
        f"belief training device={device} episodes={len(train_records)}/"
        f"{len(validation_records)} anchors={train_anchors}/{validation_anchors} "
        f"grid={grid.size}x{grid.size}@{grid.cell_m:g}m "
        f"supervision={args.supervision}",
        flush=True,
    )

    best_validation = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            objective = _objective(prediction, batch, args.supervision)
            objective["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            episodes = int(batch["sequence_mask"].shape[0])
            train_loss_sum += float(objective["loss"].item()) * episodes
            train_steps += episodes
        validation = _evaluate(model, validation_loader, device, args.supervision)
        train_loss = train_loss_sum / max(1, train_steps)
        print(
            f"epoch={epoch:03d} train_{args.supervision}_nll={train_loss:.5f} "
            f"val_{args.supervision}_nll={validation['loss']:.5f} "
            f"metric_KL={validation['teacher_kl']:.5f} "
            f"metric_map_error={validation['map_error_m']:.2f}m "
            f"metric_rank={validation['event_rank']:.1f}",
            flush=True,
        )
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_epoch = epoch
            checkpoint = {
                "format_version": 2,
                "model": "LearnedCueBeliefUpdater",
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
                "semantic_classes": SEMANTIC_CLASSES,
                "track_feature_names": TRACK_FEATURE_NAMES,
                "train_anchors": train_anchors,
                "validation_anchors": validation_anchors,
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "epoch": epoch,
                "validation_metrics": validation,
                "supervision": (
                    "source_blind_inference_nll"
                    if args.supervision == "inference"
                    else "final_event_cell_nll"
                ),
                "training_config": {
                    "supervision": args.supervision,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "gradient_clip": args.gradient_clip,
                },
            }
            torch.save(checkpoint, temporary)
            os.replace(temporary, output)

    print(
        f"PASS best_epoch={best_epoch} best_val_loss={best_validation:.5f} "
        f"checkpoint={output}",
        flush=True,
    )
    print(
        "This is a structured RGB-D-grounded track baseline; no raw image or "
        "depth payload was copied.",
        flush=True,
    )


if __name__ == "__main__":
    main()
