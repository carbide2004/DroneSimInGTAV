"""Train the source-blind belief-only Spatial RNN on schema-4 episodes."""

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
from learning.belief_objective import inference_belief_objective  # noqa: E402
from learning.spatial_belief_model import (  # noqa: E402
    SpatialRecurrentBeliefConfig,
    SpatialRecurrentBeliefUpdater,
)
from learning.spatial_belief_runtime import (  # noqa: E402
    SPATIAL_CHECKPOINT_FORMAT,
    SPATIAL_MODEL_NAME,
    evaluate_spatial_model,
    spatial_forward,
    to_device,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the belief-only source-blind Spatial RNN."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning/checkpoints/stage3_spatial_rnn.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
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
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--disable-activation-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("--epochs, --patience and --batch-size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("Invalid optimizer parameters")
    if args.gradient_clip <= 0.0 or args.workers < 0:
        parser.error("Invalid gradient clip or worker count")
    for name in ("maximum_train_episodes", "maximum_validation_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _save_checkpoint(
    output: Path,
    temporary: Path,
    model: SpatialRecurrentBeliefUpdater,
    epoch: int,
    validation: dict,
    train_anchors: list[str],
    validation_anchors: list[str],
    args: argparse.Namespace,
) -> None:
    checkpoint_payload = {
        "format_version": SPATIAL_CHECKPOINT_FORMAT,
        "model": SPATIAL_MODEL_NAME,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "semantic_classes": SEMANTIC_CLASSES,
        "track_feature_names": TRACK_FEATURE_NAMES,
        "supervision": "source_blind_inference_nll",
        "source_boundary": "first_grounded_FIRE_SOURCE_exclusive",
        "recurrent_state": "one_channel_log_belief_only",
        "train_anchors": train_anchors,
        "validation_anchors": validation_anchors,
        "dataset_root": str(args.dataset_root.resolve()),
        "epoch": epoch,
        "validation_metrics": validation,
        "training_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "d4_augmentation": True,
        },
    }
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, output)


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

    train_dataset = GroundedTrackBeliefDataset(
        train_records, augment_d4=True, augmentation_seed=args.seed
    )
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
    config = SpatialRecurrentBeliefConfig(
        radius_m=grid.radius_m,
        cell_m=grid.cell_m,
        grid_size=grid.size,
        use_activation_checkpoint=not args.disable_activation_checkpoint,
    )
    device = _device(args.device)
    model = SpatialRecurrentBeliefUpdater(config).to(device)
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
        f"spatial belief training device={device} episodes={len(train_records)}/"
        f"{len(validation_records)} anchors={train_anchors}/{validation_anchors} "
        f"grid={grid.size}x{grid.size}@{grid.cell_m:g}m "
        f"activation_checkpoint={config.use_activation_checkpoint}",
        flush=True,
    )

    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_episode_count = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = spatial_forward(model, batch)
            objective = inference_belief_objective(
                prediction, batch["event_cell"], batch["inference_mask"]
            )
            objective["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            episodes = int(batch["track_features"].shape[0])
            train_loss_sum += float(objective["loss"].item()) * episodes
            train_episode_count += episodes

        validation = evaluate_spatial_model(model, validation_loader, device)
        train_loss = train_loss_sum / train_episode_count
        inference = validation["inference"]
        last = validation["last_source_blind"]
        print(
            f"epoch={epoch:03d} train_inference_nll={train_loss:.5f} "
            f"val_inference_nll={validation['loss']:.5f} "
            f"val_last_nll={last['event_nll']:.5f} "
            f"map_error={inference['map_error_m']:.2f}m "
            f"rank={inference['event_rank']:.1f}",
            flush=True,
        )
        if float(validation["loss"]) < best_validation:
            best_validation = float(validation["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                output,
                temporary,
                model,
                epoch,
                validation,
                train_anchors,
                validation_anchors,
                args,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping epoch={epoch} patience={args.patience}",
                flush=True,
            )
            break

    print(
        f"PASS best_epoch={best_epoch} best_val_inference_nll="
        f"{best_validation:.5f} checkpoint={output}",
        flush=True,
    )
    print("No RGB-D or teacher-belief payload was copied.", flush=True)


if __name__ == "__main__":
    main()

