"""Train the Stage 3C explicit-belief bottleneck action policy."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import (  # noqa: E402
    discover_belief_episodes,
    split_records_by_anchor,
)
from learning.belief_features import SEMANTIC_CLASSES, TRACK_FEATURE_NAMES  # noqa: E402
from learning.explicit_belief_policy import (  # noqa: E402
    ExplicitBeliefActionPolicy,
    ExplicitBeliefPolicyConfig,
)
from learning.dagger_dataset import (  # noqa: E402
    DaggerPolicyDataset,
    discover_dagger_shards,
)
from learning.policy_dataset import (  # noqa: E402
    ACTION_NAMES,
    POLICY_MAX_TRACKS,
    SOURCE_FEATURE_NAMES,
    StructuredBeliefPolicyDataset,
    collate_policy_episodes,
)
from learning.policy_geometry import (  # noqa: E402
    LOCAL_GEOMETRY_CHANNELS,
    LocalGeometryConfig,
)
from learning.policy_objective import (  # noqa: E402
    action_confusion,
    class_weights_from_counts,
    explicit_policy_objective,
)
from learning.policy_runtime import (  # noqa: E402
    POLICY_CHECKPOINT_FORMAT,
    POLICY_BELIEF_BOTTLENECK,
    POLICY_OBSERVATION_FEATURE_CONTRACT,
    POLICY_D4_CONTRACT,
    POLICY_LOSS_WEIGHTS,
    POLICY_RECURRENT_STATE,
    POLICY_SPATIAL_ENCODER_CONTRACT,
    POLICY_SUPERVISION_CONTRACT,
    POLICY_MODEL_NAME,
    load_policy_checkpoint,
    resolve_device,
)
from learning.spatial_belief_runtime import load_spatial_checkpoint  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--belief-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-initialization", type=Path)
    parser.add_argument("--dagger-root", type=Path, action="append", default=[])
    parser.add_argument("--dagger-iteration", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning/checkpoints/stage3c_explicit_belief_policy_bc.pt"),
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=1e-4)
    parser.add_argument("--belief-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-anchors", nargs="+")
    parser.add_argument("--maximum-train-episodes", type=int)
    parser.add_argument("--maximum-validation-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = (
        args.warmup_epochs,
        args.joint_epochs,
        args.patience,
        args.batch_size,
    )
    if any(value <= 0 for value in positive):
        parser.error("Epoch, patience, and batch-size values must be positive")
    rates = (
        args.warmup_learning_rate,
        args.policy_learning_rate,
        args.belief_learning_rate,
        args.gradient_clip,
    )
    if any(value <= 0 for value in rates) or args.weight_decay < 0:
        parser.error("Learning rates/gradient clip must be positive")
    for field in ("maximum_train_episodes", "maximum_validation_episodes"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    args.dataset_root = args.dataset_root.resolve()
    args.belief_checkpoint = args.belief_checkpoint.resolve()
    args.policy_initialization = (
        None if args.policy_initialization is None else args.policy_initialization.resolve()
    )
    args.dagger_root = [path.resolve() for path in args.dagger_root]
    args.output = args.output.resolve()
    if not args.belief_checkpoint.is_file():
        parser.error(f"Belief checkpoint does not exist: {args.belief_checkpoint}")
    if args.policy_initialization is not None and not args.policy_initialization.is_file():
        parser.error(f"Policy initialization does not exist: {args.policy_initialization}")
    if args.dagger_iteration < 0:
        parser.error("--dagger-iteration must be non-negative")
    if args.dagger_root and args.dagger_iteration <= 0:
        parser.error("--dagger-root requires --dagger-iteration >= 1")
    if args.dagger_iteration > 0 and not args.dagger_root:
        parser.error("--dagger-iteration requires at least one --dagger-root")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_counts(records):
    counts = [0] * len(ACTION_NAMES)
    index = {name: i for i, name in enumerate(ACTION_NAMES)}
    for record in records:
        path = record.episode_root / "agent" / "steps.jsonl"
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                name = str(json.loads(line)["action"]["type"])
                if name not in index:
                    raise RuntimeError(f"Unknown action {name!r} in {path}")
                counts[index[name]] += 1
    return counts


def _dagger_action_counts(shards):
    counts = [0] * len(ACTION_NAMES)
    for shard in shards:
        with np.load(shard / "steps.npz") as archive:
            labels = np.asarray(archive["expert_action"], dtype=np.int64)
        for label in labels[labels >= 0]:
            if int(label) >= len(ACTION_NAMES):
                raise RuntimeError(f"Invalid DAgger action label in {shard}: {label}")
            counts[int(label)] += 1
    return counts


def _dataset_execution_contract(records):
    episode_keys = (
        "horizon_steps", "activity_radius_m", "activity_vertical_m",
        "forward_step_m", "vertical_step_m", "yaw_step_degrees",
        "simulation_step_ms",
    )
    observation_keys = (
        "fov_degrees", "near_clip", "far_clip",
        "oblique_pitch_degrees", "nadir_pitch_degrees",
    )
    contract = None
    resolutions = Counter()
    for record in records:
        path = record.episode_root / "agent" / "episode.json"
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        current = {
            "episode_spec": {
                key: payload["episode_spec"][key] for key in episode_keys
            },
            "observation_spec": {
                key: payload["observation_spec"][key] for key in observation_keys
            },
        }
        resolution = (
            int(payload["observation_spec"]["width"]),
            int(payload["observation_spec"]["height"]),
        )
        if resolution[0] <= 1 or resolution[1] <= 1:
            raise RuntimeError(f"Invalid Stage3C observation resolution in {path}")
        resolutions[resolution] += 1
        if contract is None:
            contract = current
        elif current != contract:
            raise RuntimeError(
                "Stage3C episodes differ in a non-resolution execution contract: "
                f"{path}"
            )
    if contract is None:
        raise RuntimeError("Cannot derive an episode contract from no records")
    entries = [
        {"width": width, "height": height, "episodes": count}
        for (width, height), count in sorted(resolutions.items())
    ]
    return contract, entries


def _canonical_episode_contract(shared_contract, training_resolutions):
    canonical = max(
        training_resolutions,
        key=lambda entry: (
            int(entry["episodes"]),
            int(entry["width"]) * int(entry["height"]),
            int(entry["width"]),
            int(entry["height"]),
        ),
    )
    return {
        "episode_spec": dict(shared_contract["episode_spec"]),
        "observation_spec": {
            "width": int(canonical["width"]),
            "height": int(canonical["height"]),
            **dict(shared_contract["observation_spec"]),
        },
    }


def _resolution_summary(entries):
    return ",".join(
        f'{entry["width"]}x{entry["height"]}:{entry["episodes"]}'
        for entry in entries
    )


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
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


@torch.no_grad()
def _evaluate(model, loader, device, class_weights):
    model.eval()
    totals = {name: 0.0 for name in ("loss", "action_loss", "belief_loss", "coordinate_loss", "value_loss")}
    episodes = 0
    confusion = torch.zeros(len(ACTION_NAMES), len(ACTION_NAMES), dtype=torch.long)
    coordinate_errors = []
    for raw in loader:
        batch = _to_device(raw, device)
        output = _forward(model, batch)
        objective = explicit_policy_objective(output, batch, class_weights)
        count = int(batch["sequence_mask"].shape[0])
        for name in totals:
            totals[name] += float(objective[name].item()) * count
        episodes += count
        confusion += action_confusion(output, batch).cpu()
        mask = batch["source_mask"] & batch["sequence_mask"]
        if bool(mask.any()):
            target = batch["event_xyz"][:, None, :].expand_as(output["event_estimate_local"])
            coordinate_errors.extend(
                torch.linalg.vector_norm(output["event_estimate_local"] - target, dim=-1)[mask]
                .detach()
                .cpu()
                .tolist()
            )
    metrics = {name: value / episodes for name, value in totals.items()}
    diagonal = confusion.diag().to(torch.float64)
    recall = diagonal / confusion.sum(dim=1).clamp_min(1)
    metrics.update(
        {
            "accuracy": float(diagonal.sum() / confusion.sum().clamp_min(1)),
            "macro_recall": float(recall.mean()),
            "coordinate_error_m": float(np.mean(coordinate_errors)) if coordinate_errors else float("nan"),
            "confusion": confusion.tolist(),
            "per_class_recall": {
                name: float(recall[i]) for i, name in enumerate(ACTION_NAMES)
            },
        }
    )
    return metrics


def _save_checkpoint(
    path,
    temporary,
    model,
    geometry_config,
    epoch,
    validation,
    args,
    class_counts,
    class_weights,
    train_anchors,
    validation_anchors,
    episode_contract,
    observation_resolution_contract,
):
    payload = {
        "format_version": POLICY_CHECKPOINT_FORMAT,
        "model": POLICY_MODEL_NAME,
        "model_state": model.state_dict(),
        "belief_config": model.belief_config.to_dict(),
        "policy_config": model.policy_config.to_dict(),
        "geometry_config": geometry_config.to_dict(),
        "semantic_classes": SEMANTIC_CLASSES,
        "track_feature_names": TRACK_FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "source_feature_names": SOURCE_FEATURE_NAMES,
        "geometry_channels": LOCAL_GEOMETRY_CHANNELS,
        "policy_max_tracks": POLICY_MAX_TRACKS,
        "belief_bottleneck": POLICY_BELIEF_BOTTLENECK,
        "recurrent_state": POLICY_RECURRENT_STATE,
        "spatial_encoder_contract": POLICY_SPATIAL_ENCODER_CONTRACT,
        "d4_contract": POLICY_D4_CONTRACT,
        "supervision_contract": POLICY_SUPERVISION_CONTRACT,
        "belief_initialization": {
            "path": str(args.belief_checkpoint),
            "sha256": _sha256(args.belief_checkpoint),
        },
        "dataset_root": str(args.dataset_root),
        "dagger_roots": [str(path) for path in args.dagger_root],
        "episode_contract": episode_contract,
        "observation_resolution_contract": observation_resolution_contract,
        "train_anchors": train_anchors,
        "validation_anchors": validation_anchors,
        "epoch": epoch,
        "stage": "joint_finetune",
        "validation_metrics": validation,
        "action_class_counts": dict(zip(ACTION_NAMES, class_counts, strict=True)),
        "action_class_weights": dict(zip(ACTION_NAMES, class_weights.tolist(), strict=True)),
        "loss_weights": POLICY_LOSS_WEIGHTS,
        "training_config": {
            "seed": args.seed,
            "warmup_epochs": args.warmup_epochs,
            "joint_epochs": args.joint_epochs,
            "batch_size": args.batch_size,
            "warmup_learning_rate": args.warmup_learning_rate,
            "policy_learning_rate": args.policy_learning_rate,
            "belief_learning_rate": args.belief_learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "d4_augmentation": True,
        },
        "dagger_iteration": int(args.dagger_iteration),
        "policy_initialization": None if args.policy_initialization is None else {
            "path": str(args.policy_initialization),
            "sha256": _sha256(args.policy_initialization),
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_epoch(model, loader, optimizer, device, class_weights, gradient_clip):
    model.train()
    totals = {name: 0.0 for name in ("loss", "action_loss", "belief_loss", "coordinate_loss", "value_loss")}
    episodes = 0
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch)
        objective = explicit_policy_objective(output, batch, class_weights)
        objective["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        count = int(batch["sequence_mask"].shape[0])
        for name in totals:
            totals[name] += float(objective[name].detach().item()) * count
        episodes += count
    return {name: value / episodes for name, value in totals.items()}


def main():
    args = _arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    belief_payload, initialized_belief = load_spatial_checkpoint(args.belief_checkpoint, device)
    policy_payload = None
    initialized_policy = None
    initialized_geometry = None
    if args.policy_initialization is not None:
        policy_payload, initialized_policy, initialized_geometry = load_policy_checkpoint(
            args.policy_initialization, device
        )
        if (
            initialized_policy.belief_config.to_dict()
            != initialized_belief.config.to_dict()
        ):
            raise RuntimeError(
                "Policy and Spatial RNN initialization belief configs differ"
            )
    records = discover_belief_episodes(args.dataset_root)
    validation_anchors = (
        args.validation_anchors
        or (None if policy_payload is None else policy_payload.get("validation_anchors"))
        or belief_payload.get("validation_anchors")
    )
    train_records, validation_records = split_records_by_anchor(records, validation_anchors)
    if args.maximum_train_episodes is not None:
        train_records = train_records[: args.maximum_train_episodes]
    if args.maximum_validation_episodes is not None:
        validation_records = validation_records[: args.maximum_validation_episodes]
    shared_contract, train_resolutions = _dataset_execution_contract(train_records)
    validation_contract, validation_resolutions = _dataset_execution_contract(
        validation_records
    )
    if validation_contract != shared_contract:
        raise RuntimeError("Train and validation non-resolution execution contracts differ")
    episode_contract = _canonical_episode_contract(shared_contract, train_resolutions)
    canonical_resolution = {
        key: episode_contract["observation_spec"][key]
        for key in ("width", "height")
    }
    observation_resolution_contract = {
        "canonical_online_resolution": canonical_resolution,
        "train": train_resolutions,
        "validation": validation_resolutions,
        "feature_contract": POLICY_OBSERVATION_FEATURE_CONTRACT,
    }
    print(
        "DATA_RESOLUTION_CONTRACT "
        f"train={_resolution_summary(train_resolutions)} "
        f"validation={_resolution_summary(validation_resolutions)} "
        f'online={canonical_resolution["width"]}x{canonical_resolution["height"]}',
        flush=True,
    )
    if policy_payload is not None:
        if policy_payload.get("episode_contract") != episode_contract:
            raise RuntimeError(
                "Policy initialization canonical online execution contract differs"
            )
        if (
            policy_payload.get("observation_resolution_contract")
            != observation_resolution_contract
        ):
            raise RuntimeError(
                "Policy initialization observation resolution contract differs"
            )
    geometry_config = (
        LocalGeometryConfig() if initialized_geometry is None else initialized_geometry
    )
    base_train_dataset = StructuredBeliefPolicyDataset(
        train_records, augment_d4=True, augmentation_seed=args.seed, geometry_config=geometry_config
    )
    dagger_shards = (
        () if not args.dagger_root else discover_dagger_shards(args.dagger_root)
    )
    train_anchors = sorted({record.anchor_name for record in train_records})
    if dagger_shards:
        dagger_anchors = {
            DaggerPolicyDataset.metadata(path)["anchor_name"] for path in dagger_shards
        }
        forbidden = sorted(dagger_anchors - set(train_anchors))
        if forbidden:
            raise RuntimeError(
                f"Held-out/unknown anchors entered DAgger training data: {forbidden}"
            )
        dagger_dataset = DaggerPolicyDataset(
            dagger_shards, augment_d4=True, augmentation_seed=args.seed + 100000
        )
        train_dataset = ConcatDataset((base_train_dataset, dagger_dataset))
    else:
        train_dataset = base_train_dataset
    validation_dataset = StructuredBeliefPolicyDataset(
        validation_records, geometry_config=geometry_config
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_policy_episodes,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_policy_episodes,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    if initialized_policy is None:
        model = ExplicitBeliefActionPolicy(
            initialized_belief.config, ExplicitBeliefPolicyConfig()
        ).to(device)
        model.belief_updater.load_state_dict(initialized_belief.state_dict(), strict=True)
    else:
        model = initialized_policy
    base_counts = _action_counts(train_records)
    dagger_counts = _dagger_action_counts(dagger_shards)
    class_counts = [
        left + right for left, right in zip(base_counts, dagger_counts, strict=True)
    ]
    class_weights = class_weights_from_counts(class_counts).to(device)
    output = args.output
    temporary = output.with_suffix(output.suffix + ".partial")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {output}")
    if temporary.exists():
        raise FileExistsError(f"Stale partial checkpoint exists: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    held_out = sorted({record.anchor_name for record in validation_records})
    print(
        f"Stage3C device={device} episodes={len(train_records)}+{len(dagger_shards)}/{len(validation_records)} "
        f"anchors={train_anchors}/{held_out} action_counts={dict(zip(ACTION_NAMES,class_counts))}",
        flush=True,
    )

    for parameter in model.belief_updater.parameters():
        parameter.requires_grad_(False)
    warmup_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        warmup_parameters,
        lr=args.warmup_learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.warmup_epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, device, class_weights, args.gradient_clip)
        validation = _evaluate(model, validation_loader, device, class_weights)
        print(
            f"warmup={epoch:03d} train={train['loss']:.4f} val={validation['loss']:.4f} "
            f"action={validation['action_loss']:.4f} belief={validation['belief_loss']:.4f} "
            f"acc={validation['accuracy']:.3f} macro_recall={validation['macro_recall']:.3f}",
            flush=True,
        )

    for parameter in model.belief_updater.parameters():
        parameter.requires_grad_(True)
    belief_parameters = list(model.belief_updater.parameters())
    belief_ids = {id(p) for p in belief_parameters}
    policy_parameters = [p for p in model.parameters() if id(p) not in belief_ids]
    optimizer = torch.optim.AdamW(
        (
            {"params": policy_parameters, "lr": args.policy_learning_rate},
            {"params": belief_parameters, "lr": args.belief_learning_rate},
        ),
        weight_decay=args.weight_decay,
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.joint_epochs + 1):
        train = _train_epoch(model, train_loader, optimizer, device, class_weights, args.gradient_clip)
        validation = _evaluate(model, validation_loader, device, class_weights)
        print(
            f"joint={epoch:03d} train={train['loss']:.4f} val={validation['loss']:.4f} "
            f"action={validation['action_loss']:.4f} belief={validation['belief_loss']:.4f} "
            f"coord={validation['coordinate_error_m']:.2f}m acc={validation['accuracy']:.3f} "
            f"macro_recall={validation['macro_recall']:.3f}",
            flush=True,
        )
        if validation["loss"] < best:
            best = validation["loss"]
            best_epoch = epoch
            stale = 0
            _save_checkpoint(
                output,
                temporary,
                model,
                geometry_config,
                epoch,
                validation,
                args,
                class_counts,
                class_weights.cpu(),
                train_anchors,
                held_out,
                episode_contract,
                observation_resolution_contract,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early stopping joint={epoch} patience={args.patience}", flush=True)
            break
    print(f"PASS Stage3C best_joint_epoch={best_epoch} val={best:.5f} checkpoint={output}")
    print("No RGB-D or teacher-belief payload was copied.")


if __name__ == "__main__":
    main()
