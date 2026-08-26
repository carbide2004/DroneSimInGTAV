"""Offline contract checks for Stage 3B streaming Spatial RNN inference.

No RGB-D payload is read or written.  The validator reconstructs features from
the compact grounded-track JSONL, compares them with the training Dataset, and
checks that recurrent ``forward_step`` is numerically identical to the sequence
``forward`` implementation used during training.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    GroundedTrackBeliefDataset,
    _read_json,
    _read_jsonl,
    discover_belief_episodes,
)
from learning.belief_features import (  # noqa: E402
    GroundedTrackFeatureConfig,
    SEMANTIC_CLASS_TO_INDEX,
    StreamingGroundedTrackEncoder,
)
from learning.spatial_belief_runtime import load_spatial_checkpoint  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Stage 3B streaming features and one-step recurrence."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=REPOSITORY_ROOT / "dataset/stage2e_5x20_v4",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "learning/checkpoints/stage3_spatial_rnn.pt",
    )
    parser.add_argument(
        "--devices",
        choices=("cpu", "cuda", "both", "auto"),
        default="auto",
        help="auto validates CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--parity-episodes",
        type=int,
        default=3,
        help="episodes per device for full-sequence versus step parity",
    )
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.dataset.is_dir():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.parity_episodes <= 0:
        parser.error("--parity-episodes must be positive")
    if args.devices in ("cuda", "both") and not torch.cuda.is_available():
        parser.error("CUDA validation was requested but CUDA is unavailable")
    return args


def _encoder_config(record, radius_m):
    metadata = _read_json(record.episode_root / "agent" / "episode.json")
    episode = metadata["episode_spec"]
    observation = metadata["observation_spec"]
    return GroundedTrackFeatureConfig(
        radius_m=float(radius_m),
        vertical_bound_m=float(episode["activity_vertical_m"]),
        horizon_steps=int(episode["horizon_steps"]),
        width=int(observation["width"]),
        height=int(observation["height"]),
    )


def _validate_streaming_features(dataset, records):
    maximum_difference = 0.0
    total_steps = 0
    for episode_index, record in enumerate(records):
        item = dataset[episode_index]
        rows = _read_jsonl(record.episode_root / "teacher" / "awareness.jsonl")
        encoder = StreamingGroundedTrackEncoder(
            _encoder_config(record, dataset.grid_spec.radius_m)
        )
        if len(rows) != int(item["length"]):
            raise RuntimeError(f"Feature stream length mismatch in {record.episode_id}")
        for sequence_index, row in enumerate(rows):
            expected_step = sequence_index + 1
            encoded = encoder.encode(expected_step, row["grounded_tracks"])
            count = len(encoded.track_classes)
            expected_mask = item["track_mask"][sequence_index]
            if int(expected_mask.sum()) != count:
                raise RuntimeError(
                    f"Track count mismatch in {record.episode_id} step {expected_step}"
                )
            actual_features = item["track_features"][sequence_index, :count].numpy()
            if count:
                difference = float(
                    np.max(np.abs(actual_features - encoded.track_features))
                )
                maximum_difference = max(maximum_difference, difference)
                if difference != 0.0:
                    raise RuntimeError(
                        "Streaming/Dataset feature mismatch in "
                        f"{record.episode_id} step {expected_step}: {difference}"
                    )
                if not np.array_equal(
                    item["track_classes"][sequence_index, :count].numpy(),
                    encoded.track_classes,
                ):
                    raise RuntimeError("Streaming/Dataset class mismatch")
            if bool(item["source_visible_mask"][sequence_index]) != encoded.source_seen:
                raise RuntimeError("Streaming/Dataset source boundary mismatch")
            if (
                bool(item["motion_evidence_mask"][sequence_index])
                != encoded.has_motion_evidence
            ):
                raise RuntimeError("Streaming/Dataset motion mask mismatch")
            total_steps += 1
    print(
        f"streaming feature parity PASS episodes={len(records)} steps={total_steps} "
        f"max_abs_diff={maximum_difference:.1e}",
        flush=True,
    )


def _selected_episode_indices(dataset, count):
    lengths = [(int(dataset[index]["length"]), index) for index in range(len(dataset))]
    ordered = [index for _length, index in sorted(lengths)]
    choices = [ordered[0], ordered[-1]]
    if len(ordered) > 2:
        choices.append(ordered[len(ordered) // 2])
    for index in ordered:
        if len(choices) >= count:
            break
        if index not in choices:
            choices.append(index)
    return tuple(choices[: min(count, len(ordered))])


@torch.no_grad()
def _validate_model_contract(dataset, checkpoint_path, device, episode_indices):
    _checkpoint, model = load_spatial_checkpoint(checkpoint_path, device)
    model.eval()
    tolerance = 1.0e-5 if device.type == "cuda" else 1.0e-6
    maximum_difference = 0.0
    for episode_index in episode_indices:
        item = dataset[episode_index]
        features = item["track_features"].unsqueeze(0).to(device)
        classes = item["track_classes"].unsqueeze(0).to(device)
        mask = item["track_mask"].unsqueeze(0).to(device)
        update = item["belief_update_mask"].unsqueeze(0).to(device)
        sequence = torch.ones_like(update)
        full = model(features, classes, mask, sequence, update)
        previous = model.initial_log_belief(1, features.dtype)
        step_outputs = {name: [] for name in full}
        for step in range(features.shape[1]):
            output = model.forward_step(
                previous,
                features[:, step],
                classes[:, step],
                mask[:, step],
                update[:, step],
            )
            previous = output["log_belief"]
            for name, value in output.items():
                step_outputs[name].append(value)
        for name, expected in full.items():
            actual = torch.stack(step_outputs[name], dim=1)
            difference = float((actual - expected).abs().max().item())
            maximum_difference = max(maximum_difference, difference)
            if not torch.allclose(actual, expected, atol=tolerance, rtol=0.0):
                raise RuntimeError(
                    f"forward_step mismatch for {name} on {device}: {difference}"
                )

        # Once source is visible the Dataset update mask is false forever.
        source_indices = torch.nonzero(item["source_visible_mask"], as_tuple=False)
        first_source = int(source_indices[0].item())
        if first_source > 0:
            frozen = full["belief"][:, first_source:]
            reference = full["belief"][:, first_source - 1 : first_source]
            if not torch.equal(frozen, reference.expand_as(frozen)):
                raise RuntimeError("Belief changed after the first grounded source")

    # The evidence encoder must be permutation-invariant and source-blind.
    item = dataset[episode_indices[0]]
    step = int(torch.nonzero(item["belief_update_mask"], as_tuple=False)[0].item())
    features = item["track_features"][step].unsqueeze(0).to(device)
    classes = item["track_classes"][step].unsqueeze(0).to(device)
    mask = item["track_mask"][step].unsqueeze(0).to(device)
    previous = model.initial_log_belief(1, features.dtype)
    allowed = torch.ones(1, dtype=torch.bool, device=device)
    reference = model.forward_step(previous, features, classes, mask, allowed)
    permutation = torch.arange(features.shape[1] - 1, -1, -1, device=device)
    permuted = model.forward_step(
        previous,
        features[:, permutation],
        classes[:, permutation],
        mask[:, permutation],
        allowed,
    )
    if not torch.allclose(
        reference["belief"], permuted["belief"], atol=tolerance, rtol=0.0
    ):
        raise RuntimeError("Track permutation changed the Spatial RNN belief")

    source_features = torch.zeros(1, 2, features.shape[-1], device=device)
    source_features[:, 0, 6] = 1.0
    source_features[:, 1, 6] = 1.0
    source_classes = torch.tensor(
        [[SEMANTIC_CLASS_TO_INDEX["PEDESTRIAN"], SEMANTIC_CLASS_TO_INDEX["FIRE_SOURCE"]]],
        dtype=torch.long,
        device=device,
    )
    source_mask = torch.ones(1, 2, dtype=torch.bool, device=device)
    with_source = model.forward_step(
        previous, source_features, source_classes, source_mask, allowed
    )
    source_features[:, 1] = 1000.0
    moved_source = model.forward_step(
        previous, source_features, source_classes, source_mask, allowed
    )
    source_mask[:, 1] = False
    removed_source = model.forward_step(
        previous, source_features, source_classes, source_mask, allowed
    )
    for changed in (moved_source, removed_source):
        if not torch.equal(with_source["belief"], changed["belief"]):
            raise RuntimeError("FIRE_SOURCE changed the source-blind belief")

    print(
        f"recurrent parity PASS device={device} episodes={len(episode_indices)} "
        f"atol={tolerance:.0e} max_abs_diff={maximum_difference:.2e}",
        flush=True,
    )


def main():
    args = _parse_args()
    records = discover_belief_episodes(args.dataset)
    dataset = GroundedTrackBeliefDataset(records, augment_d4=False)
    _validate_streaming_features(dataset, records)
    indices = _selected_episode_indices(dataset, args.parity_episodes)
    if args.devices == "both":
        devices = (torch.device("cpu"), torch.device("cuda"))
    elif args.devices == "auto":
        devices = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )
    else:
        devices = (torch.device(args.devices),)
    for device in devices:
        _validate_model_contract(dataset, args.checkpoint, device, indices)
    print(
        f"PASS Stage 3B offline contract episodes={len(records)} devices="
        f"{','.join(str(device) for device in devices)}\n"
        "No RGB-D, belief, or teacher payload was written to disk.",
        flush=True,
    )


if __name__ == "__main__":
    main()
