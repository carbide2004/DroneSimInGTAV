"""Strict offline contracts for the Stage 3C explicit-belief policy."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import discover_belief_episodes  # noqa: E402
from learning.evaluate_spatial_belief import selected_records  # noqa: E402
from learning.policy_dataset import (  # noqa: E402
    ACTION_HISTORY_PAD,
    ACTION_NAMES,
    StructuredBeliefPolicyDataset,
    apply_policy_d4_transform,
    collate_policy_episodes,
)
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--maximum-episodes", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.maximum_episodes <= 0:
        parser.error("--maximum-episodes must be positive")
    return args


def _assert_close(name, left, right, tolerance):
    difference = float(torch.max(torch.abs(left - right)).item())
    if difference > tolerance:
        raise RuntimeError(
            f"{name} mismatch: max_abs={difference:.9g}, tolerance={tolerance:.9g}"
        )
    return difference


def _device_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model, batch):
    return model.forward_sequence(
        batch["track_features"],
        batch["track_classes"],
        batch["track_mask"],
        batch["sequence_mask"],
        batch["belief_update_mask"],
        batch["pose_features"],
        batch["action_history"],
        batch["local_geometry"],
        batch["source_features"],
        batch["source_mask"],
        batch["source_position_local"],
    )


def _validate_alignment(dataset, records):
    total_steps = 0
    for index, record in enumerate(records):
        item = dataset[index]
        rows = []
        with (record.episode_root / "agent" / "steps.jsonl").open(
            "r", encoding="utf-8"
        ) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != int(item["length"]):
            raise RuntimeError(f"Action alignment length mismatch in {record.episode_id}")
        names = [ACTION_NAMES[int(value)] for value in item["action_target"]]
        expected = [str(row["action"]["type"]) for row in rows]
        if names != expected:
            raise RuntimeError(f"Observation/action labels are shifted in {record.episode_id}")
        for step, target in enumerate(item["action_target"]):
            history = item["action_history"][step].tolist()
            expected_history = [ACTION_HISTORY_PAD] * 4
            prefix = item["action_target"][max(0, step - 4):step].tolist()
            expected_history[-len(prefix):] = prefix
            if not prefix:
                expected_history = [ACTION_HISTORY_PAD] * 4
            if history != expected_history:
                raise RuntimeError(
                    f"Action history leaks the current label in {record.episode_id} step {step + 1}"
                )
        geometry = item["local_geometry"]
        if geometry.shape[1:] != (6, 41, 41) or not bool(torch.isfinite(geometry).all()):
            raise RuntimeError("Local geometry shape/finite contract failed")
        occupied = torch.maximum(torch.maximum(geometry[:, 1], geometry[:, 2]), geometry[:, 3])
        if bool(((geometry[:, 0] > 0) & (occupied > 0)).any()):
            raise RuntimeError("A local cell cannot be both observed-free and occupied")
        expected_unknown = ((geometry[:, 0] == 0) & (occupied == 0)).to(geometry.dtype)
        if not torch.equal(geometry[:, 5], expected_unknown):
            raise RuntimeError("Local geometry unknown mask is inconsistent")
        total_steps += int(item["length"])
    return total_steps


@torch.no_grad()
def _validate_sequence_step(model, batch, tolerance):
    model.eval()
    sequence = _forward(model, batch)
    previous = model.belief_updater.initial_log_belief(
        batch["sequence_mask"].shape[0], torch.float32
    ).to(batch["track_features"].device)
    rows = {name: [] for name in (
        "belief", "log_belief", "action_logits", "remaining_value", "event_estimate_local"
    )}
    steps = batch["sequence_mask"].shape[1]
    for step in range(steps):
        output = model.forward_step(
            previous,
            batch["track_features"][:, step],
            batch["track_classes"][:, step],
            batch["track_mask"][:, step],
            batch["belief_update_mask"][:, step] & batch["sequence_mask"][:, step],
            batch["pose_features"][:, step],
            batch["action_history"][:, step],
            batch["local_geometry"][:, step],
            batch["source_features"][:, step],
            batch["source_mask"][:, step],
            batch["source_position_local"][:, step],
        )
        previous = output["log_belief"]
        for name in rows:
            rows[name].append(output[name])
    maxima = {}
    for name, values in rows.items():
        stepped = torch.stack(values, dim=1)
        maxima[name] = _assert_close(name, sequence[name], stepped, tolerance)
    return sequence, maxima


@torch.no_grad()
def _validate_track_permutation(model, batch, original, tolerance):
    permuted = dict(batch)
    count = batch["track_features"].shape[2]
    order = torch.arange(count - 1, -1, -1, device=batch["track_features"].device)
    for name in ("track_features", "track_classes", "track_mask"):
        permuted[name] = batch[name].index_select(2, order)
    output = _forward(model, permuted)
    belief_difference = _assert_close(
        "track-permuted belief", original["belief"], output["belief"], tolerance
    )
    action_difference = _assert_close(
        "track-permuted actions", original["action_logits"], output["action_logits"], tolerance
    )
    return max(belief_difference, action_difference)


@torch.no_grad()
def _validate_source_blindness(model, batch, original, tolerance):
    altered = dict(batch)
    source_class = 3
    source_rows = batch["track_classes"] == source_class
    altered_features = batch["track_features"].clone()
    altered_features[source_rows] = torch.randn_like(altered_features[source_rows]) * 20.0
    altered["track_features"] = altered_features
    output = _forward(model, altered)
    return _assert_close(
        "FIRE_SOURCE-insensitive belief", original["belief"], output["belief"], tolerance
    )


@torch.no_grad()
def _validate_bottleneck_intervention(model, batch):
    signature = inspect.signature(model.policy_from_belief)
    forbidden = {"track_features", "track_classes", "track_mask", "evidence_map", "event_truth"}
    leaked = forbidden.intersection(signature.parameters)
    if leaked:
        raise RuntimeError(f"Policy API exposes forbidden inputs: {sorted(leaked)}")
    first = 0
    belief = model.belief_updater.initial_log_belief(1, torch.float32).to(
        batch["track_features"].device
    )
    probability = belief.exp()
    baseline = model.policy_from_belief(
        probability,
        belief,
        batch["pose_features"][:1, first],
        batch["action_history"][:1, first],
        batch["local_geometry"][:1, first],
        batch["source_features"][:1, first],
        batch["source_mask"][:1, first],
        batch["source_position_local"][:1, first],
    )
    shifted = torch.roll(probability, shifts=6, dims=-1)
    valid = model.belief_updater.valid_mask[None]
    shifted = torch.where(valid, shifted, torch.zeros_like(shifted))
    shifted = shifted / shifted.sum(dim=(-2, -1), keepdim=True)
    shifted_log = torch.where(valid, shifted.clamp_min(1.0e-30).log(), belief)
    intervened = model.policy_from_belief(
        shifted,
        shifted_log,
        batch["pose_features"][:1, first],
        batch["action_history"][:1, first],
        batch["local_geometry"][:1, first],
        batch["source_features"][:1, first],
        batch["source_mask"][:1, first],
        batch["source_position_local"][:1, first],
    )
    difference = float(torch.max(torch.abs(
        baseline["action_logits"] - intervened["action_logits"]
    )).item())
    if difference <= 1.0e-8:
        raise RuntimeError("do(belief=b') did not change the policy action distribution")
    return difference


@torch.no_grad()
def _validate_spatial_belief_encoding(model, device):
    """Check the body warp and ensure the encoder retains left/right layout."""
    pool_size = int(model.policy_config.spatial_pool_size)
    belief_pool = next(
        (layer for layer in model.belief_encoder if isinstance(layer, torch.nn.AdaptiveAvgPool2d)),
        None,
    )
    geometry_pool = next(
        (layer for layer in model.geometry_encoder if isinstance(layer, torch.nn.AdaptiveAvgPool2d)),
        None,
    )
    expected_pool = (pool_size, pool_size)
    for name, layer in (("belief", belief_pool), ("geometry", geometry_pool)):
        if layer is None:
            raise RuntimeError(f"{name} encoder has no explicit spatial pooling contract")
        output_size = layer.output_size
        actual = (
            (int(output_size), int(output_size))
            if isinstance(output_size, int)
            else tuple(int(value) for value in output_size)
        )
        if actual != expected_pool:
            raise RuntimeError(
                f"{name} encoder spatial pool is {actual}, expected {expected_pool}"
            )

    forward = model.body_grid_forward.to(device)
    right = model.body_grid_right.to(device)
    valid = model.belief_updater.valid_mask.to(device)
    sigma = 8.0
    images = []
    peak_body_right = []
    for target_right in (-40.0, 40.0):
        probability = torch.exp(
            -((forward - 48.0).square() + (right - target_right).square())
            / (2.0 * sigma * sigma)
        )
        probability = torch.where(valid, probability, torch.zeros_like(probability))
        probability = probability / probability.sum()
        uniform_log = model.belief_updater.initial_log_belief(1, torch.float32).to(device)
        log_probability = torch.where(
            valid[None], probability.clamp_min(1.0e-30).log()[None], uniform_log
        )
        pose = torch.zeros((1, 6), dtype=torch.float32, device=device)
        pose[:, 4] = 1.0
        image = model._agent_centric_belief(
            probability[None], log_probability, pose
        )
        images.append(image)
        flat_index = int(image[0, 0].argmax().item())
        column = flat_index % image.shape[-1]
        peak_body_right.append(float(model.body_grid_right[0, column].item()))

    if not peak_body_right[0] < 0.0 < peak_body_right[1]:
        raise RuntimeError(
            "Agent-centric belief warp reversed or lost the left/right axis: "
            f"peaks={peak_body_right}"
        )
    embeddings = model.belief_encoder(torch.cat(images, dim=0))
    mirror_delta = float(torch.max(torch.abs(embeddings[0] - embeddings[1])).item())
    if mirror_delta <= 1.0e-5:
        raise RuntimeError(
            "Belief encoder erased a mirrored left/right hypothesis before fusion"
        )
    return mirror_delta, tuple(peak_body_right)


def _validate_d4(item):
    maxima = []
    for code in range(8):
        transformed = apply_policy_d4_transform(item, code)
        inverse = code if code >= 4 else (-code) % 4
        restored = apply_policy_d4_transform(transformed, inverse)
        for name in (
            "track_features", "pose_features", "teacher_belief", "event_xyz",
            "source_position_local", "source_features", "local_geometry",
        ):
            maxima.append(float(torch.max(torch.abs(restored[name] - item[name])).item()))
        if not torch.equal(restored["action_target"], item["action_target"]):
            raise RuntimeError(f"D4 action inverse failed for code={code}")
        if not torch.equal(restored["action_history"], item["action_history"]):
            raise RuntimeError(f"D4 history inverse failed for code={code}")
    maximum = max(maxima, default=0.0)
    if maximum > 1.0e-5:
        raise RuntimeError(f"D4 inverse numerical error={maximum}")
    return maximum


def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, model, _geometry = load_policy_checkpoint(args.checkpoint, device)
    records = discover_belief_episodes(args.dataset_root)
    records = selected_records(records, checkpoint, args.split)
    records = records[: args.maximum_episodes]
    dataset = StructuredBeliefPolicyDataset(records, augment_d4=False)
    total_steps = _validate_alignment(dataset, records)
    d4_error = _validate_d4(dataset[0])
    batch = _device_batch(collate_policy_episodes([dataset[i] for i in range(len(dataset))]), device)
    tolerance = 1.0e-5 if device.type == "cuda" else 1.0e-6
    sequence, parity = _validate_sequence_step(model, batch, tolerance)
    permutation = _validate_track_permutation(model, batch, sequence, tolerance)
    source_difference = _validate_source_blindness(model, batch, sequence, tolerance)
    intervention = _validate_bottleneck_intervention(model, batch)
    spatial_delta, spatial_peaks = _validate_spatial_belief_encoding(
        model, device
    )
    if not bool(torch.isfinite(sequence["action_logits"]).all()):
        raise RuntimeError("Policy produced non-finite logits")
    print(
        f"PASS Stage3C offline contract split={args.split} device={device} "
        f"episodes={len(records)} steps={total_steps} checkpoint_epoch={checkpoint.get('epoch')}"
    )
    print(
        f"parity_max={max(parity.values()):.3g} permutation_max={permutation:.3g} "
        f"source_belief_max={source_difference:.3g} d4_inverse_max={d4_error:.3g} "
        f"belief_intervention_action_delta={intervention:.6g}"
    )
    print(
        f"spatial_mirror_embedding_delta={spatial_delta:.6g} body_right_peaks={spatial_peaks}"
    )
    print("No RGB-D, teacher-belief, or policy prediction payload was written to disk.")


if __name__ == "__main__":
    main()
