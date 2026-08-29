"""Offline contract test for compact Stage 3C DAgger shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import discover_belief_episodes  # noqa: E402
from learning.dagger_dataset import (  # noqa: E402
    DaggerPolicyDataset,
    DaggerShardRecorder,
    discover_dagger_shards,
)
from learning.policy_dataset import (  # noqa: E402
    ACTION_NAMES,
    StructuredBeliefPolicyDataset,
    collate_policy_episodes,
)
from learning.policy_objective import (  # noqa: E402
    class_weights_from_counts,
    explicit_policy_objective,
)
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 2 <= args.steps <= 16:
        parser.error("--steps must be in [2, 16]")
    return args


def _training_row(item, step):
    return {
        "track_features": item["track_features"][step].numpy(),
        "track_classes": item["track_classes"][step].numpy(),
        "track_mask": item["track_mask"][step].numpy(),
        "inference_mask": bool(item["belief_update_mask"][step]),
        "motion_evidence": bool(item["motion_evidence_mask"][step]),
        "pose_features": item["pose_features"][step].numpy(),
        "action_history": item["action_history"][step].numpy(),
        "local_geometry": item["local_geometry"][step].numpy(),
        "source_features": item["source_features"][step].numpy(),
        "source_mask": bool(item["source_mask"][step]),
        "source_position_local": item["source_position_local"][step].numpy(),
    }


@torch.no_grad()
def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, model, geometry = load_policy_checkpoint(args.checkpoint, device)
    records = discover_belief_episodes(args.dataset_root)
    record = records[0]
    base = StructuredBeliefPolicyDataset((record,), geometry_config=geometry)[0]
    steps = min(args.steps, int(base["length"]))
    event_cell = tuple(int(value) for value in base["event_cell"])
    event_xyz = base["event_xyz"].numpy()
    with tempfile.TemporaryDirectory(prefix="stage3c_dagger_") as temporary:
        root = Path(temporary)
        shard = root / "shard_000"
        recorder = DaggerShardRecorder(
            shard,
            {
                "episode_id": "offline-contract",
                "anchor_name": record.anchor_name,
                "anchor": [0.0, 0.0, 0.0],
                "scenario_seed": 1,
                "start_seed": 1,
                "pool_start_id": 1,
                "checkpoint": str(args.checkpoint.resolve()),
                "dagger_beta": 0.5,
                "horizon_steps": int(checkpoint["episode_contract"]["episode_spec"]["horizon_steps"]),
            },
        )
        for step in range(steps):
            label = ACTION_NAMES[int(base["action_target"][step])]
            recorder.record_step(
                step_index=step + 1,
                training_row=_training_row(base, step),
                learner_action="HOLD",
                expert_action=None if step == 1 else label,
                executed_by="LEARNER",
                event_local=event_xyz,
                event_cell=event_cell,
            )
        recorder.finish("TEST", result=SimpleNamespace(
            success=False, status="TEST", actions=steps, localization_error_m=None
        ))
        files = {path.relative_to(shard).as_posix() for path in shard.rglob("*") if path.is_file()}
        if files != {"metadata.json", "steps.npz"}:
            raise RuntimeError(f"DAgger shard wrote unexpected files: {sorted(files)}")
        loaded = DaggerPolicyDataset(discover_dagger_shards((root,)))[0]
        if bool(loaded["action_label_mask"][1]) or int(loaded["action_label_mask"].sum()) != steps - 1:
            raise RuntimeError("NO_EXPERT_LABEL action mask was not preserved")
        if bool(loaded["value_label_mask"].any()):
            raise RuntimeError("DAgger shard fabricated expert remaining-value labels")
        geometry_error = float(torch.max(torch.abs(
            loaded["local_geometry"] - base["local_geometry"][:steps]
        )))
        if geometry_error > 1.0 / 127.5 + 1.0e-6:
            raise RuntimeError(f"DAgger geometry quantization error is too large: {geometry_error}")
        batch = collate_policy_episodes((loaded,))
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        prediction = model.forward_sequence(
            batch["track_features"], batch["track_classes"], batch["track_mask"],
            batch["sequence_mask"], batch["belief_update_mask"],
            batch["pose_features"], batch["action_history"], batch["local_geometry"],
            batch["source_features"], batch["source_mask"],
            batch["source_position_local"],
        )
        objective = explicit_policy_objective(
            prediction, batch, class_weights_from_counts([1] * len(ACTION_NAMES)).to(device)
        )
        if not all(torch.isfinite(value) for value in objective.values()):
            raise RuntimeError("DAgger objective produced a non-finite value")
        mixed = collate_policy_episodes((base, loaded))
        if not bool(mixed["value_label_mask"][0, : int(base["length"])].all()):
            raise RuntimeError("Base expert episode lost remaining-action value labels")
        if bool(mixed["value_label_mask"][1].any()):
            raise RuntimeError("Mixed collate enabled value loss for DAgger rows")
        if bool(mixed["action_label_mask"][1, 1]):
            raise RuntimeError("Mixed collate lost the NO_EXPERT_LABEL action mask")
        mixed = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in mixed.items()
        }
        mixed_prediction = model.forward_sequence(
            mixed["track_features"], mixed["track_classes"], mixed["track_mask"],
            mixed["sequence_mask"], mixed["belief_update_mask"],
            mixed["pose_features"], mixed["action_history"], mixed["local_geometry"],
            mixed["source_features"], mixed["source_mask"],
            mixed["source_position_local"],
        )
        mixed_objective = explicit_policy_objective(
            mixed_prediction,
            mixed,
            class_weights_from_counts([1] * len(ACTION_NAMES)).to(device),
        )
        if not all(torch.isfinite(value) for value in mixed_objective.values()):
            raise RuntimeError("Mixed base/DAgger objective produced a non-finite value")
    print(
        f"PASS Stage3C DAgger steps={steps} labels={steps-1} "
        f"geometry_error={geometry_error:.6f} loss={float(objective['loss']):.5f}"
    )
    print("Atomic shard contains only metadata.json and compressed steps.npz; RGB/Depth files=0.")


if __name__ == "__main__":
    main()
