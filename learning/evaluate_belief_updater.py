"""Evaluate a learned 2-D belief updater without connecting to GTA V."""

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
    SEMANTIC_CLASSES,
    TRACK_FEATURE_NAMES,
    collate_belief_episodes,
    discover_belief_episodes,
)
from learning.belief_model import (  # noqa: E402
    BeliefUpdaterConfig,
    LearnedCueBeliefUpdater,
)
from learning.belief_objective import belief_metrics  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the structured-track 2-D belief updater."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "train", "all"), default="validation"
    )
    parser.add_argument("--batch-size", type=int, default=4)
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


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict, LearnedCueBeliefUpdater]:
    checkpoint = torch.load(
        path.resolve(), map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise RuntimeError("Unsupported learned-belief checkpoint format")
    if checkpoint.get("model") != "LearnedCueBeliefUpdater":
        raise RuntimeError(f"Unexpected model: {checkpoint.get('model')!r}")
    if tuple(checkpoint.get("semantic_classes", ())) != SEMANTIC_CLASSES:
        raise RuntimeError("Checkpoint semantic-class contract does not match this code")
    if tuple(checkpoint.get("track_feature_names", ())) != TRACK_FEATURE_NAMES:
        raise RuntimeError("Checkpoint track-feature contract does not match this code")
    config = BeliefUpdaterConfig.from_dict(checkpoint["model_config"])
    model = LearnedCueBeliefUpdater(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return checkpoint, model


def _selected_records(records: tuple, checkpoint: dict, split: str) -> tuple:
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


@torch.no_grad()
def main() -> None:
    args = _arguments()
    device = _device(args.device)
    checkpoint, model = _load_checkpoint(args.checkpoint, device)
    records = discover_belief_episodes(args.dataset_root)
    records = _selected_records(records, checkpoint, args.split)
    if args.maximum_episodes is not None:
        records = records[: args.maximum_episodes]
    dataset = GroundedTrackBeliefDataset(records)
    expected = model.config
    grid = dataset.grid_spec
    if (grid.radius_m, grid.cell_m, grid.size) != (
        expected.radius_m,
        expected.cell_m,
        expected.grid_size,
    ):
        raise RuntimeError("Dataset belief grid does not match checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_belief_episodes,
        pin_memory=device.type == "cuda",
    )

    metric_names = (
        "teacher_kl", "event_nll", "map_error_m", "event_rank", "teacher_event_rank"
    )
    subset_totals = {name: defaultdict(float) for name in ("all", "post_cue", "final")}
    class_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for raw in loader:
        batch = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in raw.items()
        }
        prediction = model(
            batch["track_features"],
            batch["track_classes"],
            batch["track_mask"],
            batch["pose_features"],
            batch["sequence_mask"],
        )
        valid_motion = (
            batch["track_mask"]
            & batch["sequence_mask"].unsqueeze(-1)
            & (batch["track_features"][..., model.MOTION_VALID] > 0.5)
        )
        motion_step = valid_motion.any(dim=-1)
        post_cue = batch["sequence_mask"] & (motion_step.cumsum(dim=1) > 0)
        final_step = torch.zeros_like(batch["sequence_mask"])
        lengths = batch["sequence_mask"].sum(dim=1)
        final_step[
            torch.arange(final_step.shape[0], device=device), lengths - 1
        ] = True
        subsets = {
            "all": batch["sequence_mask"],
            "post_cue": post_cue,
            "final": final_step,
        }
        for subset_name, subset_mask in subsets.items():
            if not bool(subset_mask.any()):
                continue
            metrics = belief_metrics(
                prediction,
                batch["teacher_belief"],
                batch["event_cell"],
                batch["event_xy"],
                subset_mask,
                model.grid_forward,
                model.grid_right,
            )
            weight = metrics["steps"]
            totals = subset_totals[subset_name]
            totals["steps"] += weight
            for name in metric_names:
                totals[name] += metrics[name] * weight

        for class_index in range(1, len(SEMANTIC_CLASSES)):
            mask = valid_motion & (batch["track_classes"] == class_index)
            count = int(mask.sum().item())
            if count == 0:
                continue
            row = class_totals[class_index]
            row["count"] += count
            for name in ("parallel", "perpendicular", "sigma_degrees", "reliability"):
                row[name] += float(prediction[name][mask].sum().item())

    if subset_totals["all"]["steps"] <= 0:
        raise RuntimeError("Evaluation produced no active steps")
    print(
        f"PASS split={args.split} device={device} episodes={len(records)}", flush=True
    )
    for subset_name, totals in subset_totals.items():
        denominator = totals["steps"]
        if denominator <= 0:
            continue
        print(
            f"metrics subset={subset_name} steps={int(denominator)} "
            f"KL={totals['teacher_kl'] / denominator:.5f} "
            f"event_nll={totals['event_nll'] / denominator:.3f} "
            f"map_error={totals['map_error_m'] / denominator:.2f}m "
            f"event_rank={totals['event_rank'] / denominator:.1f} "
            f"teacher_event_rank={totals['teacher_event_rank'] / denominator:.1f}",
            flush=True,
        )
    for class_index, row in sorted(class_totals.items()):
        count = row["count"]
        print(
            f"cue_class={SEMANTIC_CLASSES[class_index]} count={int(count)} "
            f"parallel={row['parallel'] / count:+.3f} "
            f"perpendicular={row['perpendicular'] / count:+.3f} "
            f"sigma={row['sigma_degrees'] / count:.1f}deg "
            f"reliability={row['reliability'] / count:.3f}",
            flush=True,
        )
    print("No RGB-D or prediction payload was written to disk.", flush=True)


if __name__ == "__main__":
    main()
