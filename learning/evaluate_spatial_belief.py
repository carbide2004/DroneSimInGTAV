"""Evaluate a source-blind Spatial RNN belief checkpoint offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    GroundedTrackBeliefDataset,
    collate_belief_episodes,
    discover_belief_episodes,
)
from learning.spatial_belief_runtime import (  # noqa: E402
    evaluate_spatial_model,
    format_metric_row,
    load_spatial_checkpoint,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the belief-only source-blind Spatial RNN."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "train", "all"), default="validation"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--maximum-episodes", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("Invalid batch or worker count")
    if args.maximum_episodes is not None and args.maximum_episodes <= 0:
        parser.error("--maximum-episodes must be positive")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def selected_records(records: tuple, checkpoint: dict, split: str) -> tuple:
    if split == "all":
        return records
    key = "validation_anchors" if split == "validation" else "train_anchors"
    anchors = checkpoint.get(key)
    if not isinstance(anchors, list) or not anchors:
        raise RuntimeError(f"Checkpoint does not declare a non-empty {key}")
    available = {record.anchor_name for record in records}
    missing = sorted(set(anchors) - available)
    if missing:
        raise RuntimeError(f"Dataset is missing checkpoint {split} anchors: {missing}")
    selected = tuple(record for record in records if record.anchor_name in set(anchors))
    if not selected:
        raise RuntimeError(f"No episodes selected for split={split}")
    return selected


def validate_grid(dataset: GroundedTrackBeliefDataset, model) -> None:
    grid = dataset.grid_spec
    config = model.config
    if (grid.radius_m, grid.cell_m, grid.size) != (
        config.radius_m,
        config.cell_m,
        config.grid_size,
    ):
        raise RuntimeError("Dataset belief grid does not match checkpoint")


@torch.no_grad()
def main() -> None:
    args = _arguments()
    device = _device(args.device)
    checkpoint, model = load_spatial_checkpoint(args.checkpoint, device)
    records = selected_records(
        discover_belief_episodes(args.dataset_root), checkpoint, args.split
    )
    if args.maximum_episodes is not None:
        records = records[: args.maximum_episodes]
    dataset = GroundedTrackBeliefDataset(records)
    validate_grid(dataset, model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_belief_episodes,
        pin_memory=device.type == "cuda",
    )
    result = evaluate_spatial_model(model, loader, device)
    print(
        f"PASS split={args.split} device={device} episodes={len(records)} "
        f"checkpoint_epoch={checkpoint.get('epoch')}",
        flush=True,
    )
    print(format_metric_row("spatial_rnn", "inference", result["inference"]))
    print(
        format_metric_row(
            "spatial_rnn", "last_source_blind", result["last_source_blind"]
        )
    )
    print(
        "teacher_KL is diagnostic only; checkpoint selection uses inference NLL.",
        flush=True,
    )
    print("No RGB-D or prediction payload was written to disk.", flush=True)


if __name__ == "__main__":
    main()
