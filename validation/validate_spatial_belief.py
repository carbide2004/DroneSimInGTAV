"""Offline contract validator for the Stage 3A Spatial RNN belief updater."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    GroundedTrackBeliefDataset,
    SEMANTIC_CLASS_TO_INDEX,
    TRACK_FEATURE_NAMES,
    apply_d4_transform,
    collate_belief_episodes,
    discover_belief_episodes,
    inverse_d4_code,
)
from learning.belief_objective import inference_belief_objective  # noqa: E402
from learning.spatial_belief_model import (  # noqa: E402
    SpatialRecurrentBeliefConfig,
    SpatialRecurrentBeliefUpdater,
)
from learning.spatial_belief_runtime import (  # noqa: E402
    SPATIAL_CHECKPOINT_FORMAT,
    SPATIAL_MODEL_NAME,
    load_spatial_checkpoint,
    spatial_forward,
    to_device,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overfit-steps", type=int, default=20)
    parser.add_argument(
        "--training-smoke-epochs",
        type=int,
        default=5,
        help="Set to zero for the fast structural-only validation",
    )
    args = parser.parse_args()
    if args.overfit_steps < 0 or args.training_smoke_epochs < 0:
        parser.error("Smoke-test iteration counts must be non-negative")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _small_model(dataset: GroundedTrackBeliefDataset, device: torch.device):
    grid = dataset.grid_spec
    config = SpatialRecurrentBeliefConfig(
        radius_m=grid.radius_m,
        cell_m=grid.cell_m,
        grid_size=grid.size,
        semantic_embedding_dim=4,
        pair_hidden_dim=16,
        evidence_channels=4,
        recurrent_hidden_dim=8,
        use_activation_checkpoint=False,
    )
    model = SpatialRecurrentBeliefUpdater(config).to(device)
    # The production zero initialization starts at a uniform prior. Give this
    # validation-only instance a deterministic nonzero candidate readout so
    # direction-reversal sensitivity can be checked before training.
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    with torch.no_grad():
        values = torch.randn(
            model.candidate_head.weight.shape, generator=generator, device=device
        ) * 0.3
        model.candidate_head.weight.copy_(values)
    model.eval()
    return model


def _mask_audit(dataset: GroundedTrackBeliefDataset) -> tuple[int, int, int]:
    shortest = 1 << 30
    longest = 0
    steps = 0
    source_class = SEMANTIC_CLASS_TO_INDEX["FIRE_SOURCE"]
    for item in dataset:
        source = item["source_visible_mask"]
        motion = item["motion_evidence_mask"]
        inference = item["inference_mask"]
        update = item["belief_update_mask"]
        source_indices = torch.nonzero(source, as_tuple=False).flatten()
        cue_indices = torch.nonzero(motion, as_tuple=False).flatten()
        if source_indices.numel() == 0 or cue_indices.numel() == 0:
            raise RuntimeError(f"Empty mask boundary in {item['episode_id']}")
        first_source = int(source_indices[0])
        first_cue = int(cue_indices[0])
        expected = torch.zeros_like(inference)
        expected[first_cue:first_source] = True
        if first_cue >= first_source or not torch.equal(inference, expected):
            raise RuntimeError(f"Incorrect inference boundary in {item['episode_id']}")
        if not torch.equal(update, inference):
            raise RuntimeError(f"belief_update_mask differs in {item['episode_id']}")
        actual_source = (
            (item["track_classes"] == source_class) & item["track_mask"]
        ).any(dim=1)
        if int(torch.nonzero(actual_source, as_tuple=False)[0]) != first_source:
            raise RuntimeError(f"FIRE_SOURCE boundary mismatch in {item['episode_id']}")
        count = int(inference.sum())
        shortest = min(shortest, count)
        longest = max(longest, count)
        steps += count
    return shortest, longest, steps


def _assert_close(left: torch.Tensor, right: torch.Tensor, message: str) -> None:
    if not torch.equal(left, right):
        maximum = float((left - right).abs().max().item())
        raise RuntimeError(f"{message}; maximum difference={maximum:g}")


@torch.no_grad()
def _invariance_audit(model, batch: dict) -> None:
    signature = tuple(inspect.signature(model.forward).parameters)
    if signature != (
        "track_features",
        "track_classes",
        "track_mask",
        "sequence_mask",
        "inference_mask",
    ):
        raise RuntimeError(f"Unexpected recurrent API: {signature}")
    baseline = spatial_forward(model, batch)
    if set(baseline) != {"belief", "log_belief", "evidence_map", "update_gate"}:
        raise RuntimeError(f"Unexpected model outputs: {sorted(baseline)}")
    if any("hidden" in name.lower() for name, _ in model.named_buffers()):
        raise RuntimeError("Model holds an unexpected hidden-state buffer")

    pose_changed = dict(batch)
    pose_changed["pose_features"] = torch.randn_like(batch["pose_features"]) * 100.0
    _assert_close(
        baseline["belief"],
        spatial_forward(model, pose_changed)["belief"],
        "Unused pose tensor changed Spatial RNN output",
    )

    source = SEMANTIC_CLASS_TO_INDEX["FIRE_SOURCE"]
    source_changed = dict(batch)
    features = batch["track_features"].clone()
    source_mask = batch["track_mask"] & (batch["track_classes"] == source)
    features[source_mask] = torch.randn_like(features[source_mask]) * 100.0
    source_changed["track_features"] = features
    _assert_close(
        baseline["belief"],
        spatial_forward(model, source_changed)["belief"],
        "FIRE_SOURCE content changed Spatial RNN belief",
    )
    extra_features = torch.cat(
        (batch["track_features"], torch.randn_like(batch["track_features"][..., :2, :])),
        dim=2,
    )
    extra_classes = torch.cat(
        (
            batch["track_classes"],
            torch.full_like(batch["track_classes"][..., :2], source),
        ),
        dim=2,
    )
    extra_mask = torch.cat(
        (batch["track_mask"], torch.ones_like(batch["track_mask"][..., :2])),
        dim=2,
    )
    extra = dict(batch)
    extra["track_features"] = extra_features
    extra["track_classes"] = extra_classes
    extra["track_mask"] = extra_mask
    _assert_close(
        baseline["belief"],
        spatial_forward(model, extra)["belief"],
        "FIRE_SOURCE quantity changed Spatial RNN belief",
    )

    permutation = torch.arange(batch["track_features"].shape[2] - 1, -1, -1)
    permuted = dict(batch)
    permuted["track_features"] = batch["track_features"][:, :, permutation]
    permuted["track_classes"] = batch["track_classes"][:, :, permutation]
    permuted["track_mask"] = batch["track_mask"][:, :, permutation]
    permuted_prediction = spatial_forward(model, permuted)
    if not torch.allclose(
        baseline["evidence_map"], permuted_prediction["evidence_map"], atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Track permutation changed evidence_map")
    if not torch.allclose(
        baseline["belief"], permuted_prediction["belief"], atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Track permutation changed belief")


@torch.no_grad()
def _belief_contract_audit(model, batch: dict) -> None:
    prediction = spatial_forward(model, batch)
    belief = prediction["belief"]
    if not bool(torch.isfinite(belief).all()) or bool((belief < 0.0).any()):
        raise RuntimeError("Belief contains non-finite or negative values")
    if not torch.allclose(belief.sum(dim=(-2, -1)), torch.ones_like(belief[:, :, 0, 0])):
        raise RuntimeError("Belief is not normalized")
    if bool((belief[..., ~model.valid_mask] != 0.0).any()):
        raise RuntimeError("Belief leaks outside the circular valid grid")

    initial = model.initial_log_belief(belief.shape[0], belief.dtype).exp()
    previous = initial
    for step in range(belief.shape[1]):
        valid_motion = model._valid_track_mask(
            batch["track_features"][:, step],
            batch["track_classes"][:, step],
            batch["track_mask"][:, step],
        ).any(dim=1)
        identity = ~batch["belief_update_mask"][:, step] | ~valid_motion
        if bool(identity.any()) and not torch.equal(
            belief[identity, step], previous[identity]
        ):
            raise RuntimeError(f"Belief drifted during identity step {step}")
        previous = belief[:, step]

    deleted = dict(batch)
    deleted["track_mask"] = torch.zeros_like(batch["track_mask"])
    deleted_result = spatial_forward(model, deleted)
    deleted_prediction = deleted_result["belief"]
    expected = initial[:, None].expand_as(deleted_prediction)
    _assert_close(deleted_prediction, expected, "Cue deletion did not retain prior")
    if bool((deleted_result["evidence_map"] != 0.0).any()):
        raise RuntimeError("No-evidence sequence returned a nonzero evidence map")
    if bool((deleted_result["update_gate"] != 0.0).any()):
        raise RuntimeError("No-evidence sequence returned a visible update gate")
    outside_update = ~batch["belief_update_mask"][:, :, None, None, None]
    if bool((prediction["evidence_map"].masked_select(outside_update) != 0.0).any()):
        raise RuntimeError("Evidence map is nonzero outside the inference interval")
    outside_gate = ~batch["belief_update_mask"][:, :, None, None]
    if bool((prediction["update_gate"].masked_select(outside_gate) != 0.0).any()):
        raise RuntimeError("Update gate is nonzero outside the inference interval")

    reversed_batch = dict(batch)
    reversed_features = batch["track_features"].clone()
    reversed_features[..., 3:5] *= -1.0
    reversed_batch["track_features"] = reversed_features
    reversed_prediction = spatial_forward(model, reversed_batch)["belief"]
    active = batch["inference_mask"][:, :, None, None]
    difference = ((belief - reversed_prediction).abs() * active).max()
    if not torch.isfinite(difference) or float(difference) <= 1.0e-10:
        raise RuntimeError("Motion direction reversal did not change belief")


def _objective_audit(model, batch: dict) -> None:
    prediction = spatial_forward(model, batch)
    objective = inference_belief_objective(
        prediction, batch["event_cell"], batch["inference_mask"]
    )
    rows = []
    for index in range(batch["sequence_mask"].shape[0]):
        probability = prediction["belief"][
            index,
            :,
            batch["event_cell"][index, 0],
            batch["event_cell"][index, 1],
        ]
        rows.append(
            (-probability[batch["inference_mask"][index]].clamp_min(1e-30).log()).mean()
        )
    manual = torch.stack(rows).mean()
    if not torch.allclose(objective["loss"], manual, atol=1e-7, rtol=1e-7):
        raise RuntimeError("Episode-normalized inference NLL does not match manual result")


def _d4_audit(item: dict) -> None:
    tensor_keys = (
        "track_features",
        "track_classes",
        "track_mask",
        "pose_features",
        "teacher_belief",
        "event_cell",
        "event_xy",
        "source_visible_mask",
        "motion_evidence_mask",
        "inference_mask",
        "belief_update_mask",
    )
    for code in range(8):
        restored = apply_d4_transform(
            apply_d4_transform(item, code), inverse_d4_code(code)
        )
        for key in tensor_keys:
            if not torch.allclose(restored[key], item[key], atol=1e-6, rtol=0.0):
                raise RuntimeError(f"D4 round trip failed code={code} tensor={key}")


def _backward_audit(model, batch: dict) -> None:
    model.train()
    prediction = spatial_forward(model, batch)
    loss = inference_belief_objective(
        prediction, batch["event_cell"], batch["inference_mask"]
    )["loss"]
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    if not gradients or not any(
        value is not None and bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise RuntimeError("Variable-length/long-sequence backward produced no gradients")
    model.zero_grad(set_to_none=True)
    model.eval()


def _training_smoke(
    dataset: GroundedTrackBeliefDataset,
    model,
    device: torch.device,
    overfit_steps: int,
    epochs: int,
) -> None:
    first = collate_belief_episodes((dataset[0],))
    first = to_device(first, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    if overfit_steps:
        model.train()
        losses = []
        for _ in range(overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = spatial_forward(model, first)
            loss = inference_belief_objective(
                prediction, first["event_cell"], first["inference_mask"]
            )["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if not all(torch.isfinite(torch.tensor(losses))) or losses[-1] >= losses[0]:
            raise RuntimeError(f"Tiny overfit did not reduce NLL: {losses[0]}->{losses[-1]}")
        print(f"tiny overfit PASS nll={losses[0]:.4f}->{losses[-1]:.4f}", flush=True)

    if epochs:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_belief_episodes,
        )
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            episodes = 0
            model.train()
            for raw in loader:
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = spatial_forward(model, batch)
                loss = inference_belief_objective(
                    prediction, batch["event_cell"], batch["inference_mask"]
                )["loss"]
                loss.backward()
                optimizer.step()
                count = int(batch["sequence_mask"].shape[0])
                epoch_loss += float(loss.item()) * count
                episodes += count
            if not torch.isfinite(torch.tensor(epoch_loss)):
                raise RuntimeError(f"Non-finite full-data smoke loss at epoch {epoch}")
            print(
                f"full-data smoke epoch={epoch}/{epochs} nll={epoch_loss / episodes:.4f}",
                flush=True,
            )

    payload = {
        "format_version": SPATIAL_CHECKPOINT_FORMAT,
        "model": SPATIAL_MODEL_NAME,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "semantic_classes": tuple(SEMANTIC_CLASS_TO_INDEX),
        "track_feature_names": TRACK_FEATURE_NAMES,
        "source_boundary": "first_grounded_FIRE_SOURCE_exclusive",
        "recurrent_state": "one_channel_log_belief_only",
        "supervision": "source_blind_inference_nll",
    }
    with tempfile.TemporaryDirectory(prefix="spatial-belief-") as directory:
        path = Path(directory) / "smoke.pt"
        torch.save(payload, path)
        _, restored = load_spatial_checkpoint(path, device)
        model.eval()
        restored.eval()
        original = spatial_forward(model, first)["belief"]
        reloaded = spatial_forward(restored, first)["belief"]
        _assert_close(original, reloaded, "Checkpoint round trip changed belief")
    print("checkpoint save/load PASS", flush=True)


def main() -> None:
    args = _arguments()
    torch.manual_seed(13)
    device = _device(args.device)
    records = discover_belief_episodes(args.dataset_root)
    dataset = GroundedTrackBeliefDataset(records)
    shortest, longest, supervised_steps = _mask_audit(dataset)
    _d4_audit(dataset[0])

    lengths = [int(dataset[index]["length"]) for index in range(len(dataset))]
    shortest_index = min(range(len(lengths)), key=lengths.__getitem__)
    longest_index = max(range(len(lengths)), key=lengths.__getitem__)
    batch = collate_belief_episodes(
        (dataset[shortest_index], dataset[longest_index])
    )
    batch = to_device(batch, device)
    model = _small_model(dataset, device)
    _invariance_audit(model, batch)
    _belief_contract_audit(model, batch)
    _objective_audit(model, batch)
    _backward_audit(model, batch)
    _training_smoke(
        dataset,
        model,
        device,
        args.overfit_steps,
        args.training_smoke_epochs,
    )
    print(
        f"PASS episodes={len(records)} supervised_steps={supervised_steps} "
        f"window={shortest}..{longest} longest_episode={max(lengths)} "
        f"device={device}",
        flush=True,
    )
    print("No RGB-D or teacher-belief payload was copied or saved.", flush=True)


if __name__ == "__main__":
    main()
