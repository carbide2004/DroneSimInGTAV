"""Compare uniform, incremental, and Spatial RNN beliefs on one split."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from learning.belief_objective import belief_metrics  # noqa: E402
from learning.evaluate_spatial_belief import selected_records, validate_grid  # noqa: E402
from learning.incremental_belief_runtime import (  # noqa: E402
    evaluate_incremental_model,
    load_incremental_checkpoint,
)
from learning.spatial_belief_runtime import (  # noqa: E402
    METRIC_NAMES,
    evaluate_spatial_model,
    format_metric_row,
    last_true_mask,
    load_spatial_checkpoint,
    to_device,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("incremental_checkpoint", type=Path)
    parser.add_argument("spatial_checkpoint", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "train"), default="validation"
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


@torch.no_grad()
def _evaluate_uniform(loader: DataLoader, spatial_model, device: torch.device) -> dict:
    rows = {
        "inference": defaultdict(float),
        "last_source_blind": defaultdict(float),
    }
    valid = spatial_model.valid_mask.to(device)
    uniform = valid.to(torch.float32) / valid.sum()
    for raw in loader:
        batch = to_device(raw, device)
        batch_size, time_steps = batch["sequence_mask"].shape
        belief = uniform[None, None].expand(batch_size, time_steps, -1, -1)
        prediction = {"belief": belief}
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
                spatial_model.grid_forward,
                spatial_model.grid_right,
            )
            weight = metrics["steps"]
            rows[subset]["steps"] += weight
            for name in METRIC_NAMES:
                rows[subset][name] += metrics[name] * weight
    result = {}
    for subset, totals in rows.items():
        steps = totals["steps"]
        if steps <= 0:
            raise RuntimeError(f"Uniform subset {subset} is empty")
        result[subset] = {
            name: totals[name] / steps for name in METRIC_NAMES
        } | {"steps": steps}
    return result


def _contract_anchors(payload: dict, split: str) -> tuple[str, ...]:
    key = "validation_anchors" if split == "validation" else "train_anchors"
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Checkpoint is missing {key}")
    return tuple(value)


@torch.no_grad()
def main() -> None:
    args = _arguments()
    device = _device(args.device)
    incremental_payload, incremental = load_incremental_checkpoint(
        args.incremental_checkpoint, device, require_source_blind=True
    )
    spatial_payload, spatial = load_spatial_checkpoint(args.spatial_checkpoint, device)
    incremental_anchors = _contract_anchors(incremental_payload, args.split)
    spatial_anchors = _contract_anchors(spatial_payload, args.split)
    if incremental_anchors != spatial_anchors:
        raise RuntimeError(
            "Checkpoint anchor splits differ: "
            f"incremental={incremental_anchors}, spatial={spatial_anchors}"
        )
    if (
        incremental.config.radius_m,
        incremental.config.cell_m,
        incremental.config.grid_size,
    ) != (
        spatial.config.radius_m,
        spatial.config.cell_m,
        spatial.config.grid_size,
    ):
        raise RuntimeError("Checkpoint belief grids differ")

    records = selected_records(
        discover_belief_episodes(args.dataset_root), spatial_payload, args.split
    )
    if args.maximum_episodes is not None:
        records = records[: args.maximum_episodes]
    dataset = GroundedTrackBeliefDataset(records)
    validate_grid(dataset, spatial)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_belief_episodes,
        pin_memory=device.type == "cuda",
    )
    uniform_result = _evaluate_uniform(loader, spatial, device)
    incremental_result = evaluate_incremental_model(incremental, loader, device)
    spatial_result = evaluate_spatial_model(spatial, loader, device)

    print(
        f"PASS split={args.split} device={device} episodes={len(records)} "
        f"anchors={list(spatial_anchors)}",
        flush=True,
    )
    for subset in ("inference", "last_source_blind"):
        print(format_metric_row("uniform", subset, uniform_result[subset]))
        print(
            format_metric_row(
                "incremental", subset, incremental_result[subset]
            )
        )
        print(format_metric_row("spatial_rnn", subset, spatial_result[subset]))
    print(
        "Model ranking is reported, not enforced; teacher_KL remains diagnostic.",
        flush=True,
    )
    print("No RGB-D or prediction payload was written to disk.", flush=True)


if __name__ == "__main__":
    main()
