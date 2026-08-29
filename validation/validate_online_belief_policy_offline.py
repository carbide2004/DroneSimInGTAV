"""Offline parity test for the stateful Stage 3C online runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.expert_teacher import (  # noqa: E402
    GroundedFrame, GroundedTrackObservation,
)
from agent_control.task_starts import AgentOdometry  # noqa: E402
from learning.belief_dataset import discover_belief_episodes  # noqa: E402
from learning.belief_features import GroundedTrackFeatureConfig  # noqa: E402
from learning.evaluate_spatial_belief import selected_records  # noqa: E402
from learning.online_belief_policy import OnlineExplicitBeliefPolicyRuntime  # noqa: E402
from learning.policy_dataset import (  # noqa: E402
    ACTION_NAMES, StructuredBeliefPolicyDataset, collate_policy_episodes,
)
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402


class _Frame:
    def __init__(self, depth, projection):
        self._depth = depth
        self.projection_matrix = tuple(np.asarray(projection).reshape(-1))

    def depth_array(self):
        return self._depth


class _Pair:
    def __init__(self, oblique, nadir):
        self.oblique = oblique
        self.nadir = nadir


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("auto", "train", "validation"), default="auto")
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.maximum_steps is not None and args.maximum_steps <= 0:
        parser.error("--maximum-steps must be positive")
    return args


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _projection(spec):
    from learning.policy_geometry import projection_matrix_from_spec
    return projection_matrix_from_spec(spec)


def _grounded(step, rows):
    tracks = []
    for row in rows:
        tracks.append(GroundedTrackObservation(
            track_id=int(row["track_id"]),
            semantic_class=str(row["semantic_class"]),
            position_local=tuple(row["position_local"]),
            view_name=str(row["view_name"]),
            projected_bbox=tuple(row["projected_bbox"]),
            supporting_pixels=tuple(tuple(value) for value in row.get("supporting_pixels", ())),
        ))
    return GroundedFrame(step_index=step, tracks=tuple(tracks))


def _matching_record(records, checkpoint, requested_split):
    observation = checkpoint["episode_contract"]["observation_spec"]
    expected = (int(observation["width"]), int(observation["height"]))
    split_order = (
        ("validation", "train")
        if requested_split == "auto"
        else (requested_split,)
    )
    mismatches = {}
    for split in split_order:
        candidates = selected_records(records, checkpoint, split)
        mismatches[split] = []
        for record in candidates:
            metadata = _read_json(record.episode_root / "agent" / "episode.json")
            spec = metadata["observation_spec"]
            actual = (int(spec["width"]), int(spec["height"]))
            if actual == expected:
                return record, split
            mismatches[split].append(actual)
    summaries = {
        split: sorted(set(values)) for split, values in mismatches.items()
    }
    raise RuntimeError(
        "No episode in the requested parity split matches the checkpoint's "
        f"canonical online RGB-D resolution {expected}; available={summaries}"
    )


@torch.no_grad()
def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, model, geometry_config = load_policy_checkpoint(args.checkpoint, device)
    record, selected_split = _matching_record(
        discover_belief_episodes(args.dataset_root), checkpoint, args.split
    )
    dataset = StructuredBeliefPolicyDataset(
        (record,), augment_d4=False, geometry_config=geometry_config
    )
    item = dataset[0]
    batch = collate_policy_episodes((item,))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    root = record.episode_root
    metadata = _read_json(root / "agent" / "episode.json")
    awareness = _read_jsonl(root / "teacher" / "awareness.jsonl")
    agent_rows = _read_jsonl(root / "agent" / "steps.jsonl")
    episode_spec = metadata["episode_spec"]
    observation_spec = metadata["observation_spec"]
    runtime = OnlineExplicitBeliefPolicyRuntime(
        args.checkpoint,
        GroundedTrackFeatureConfig(
            radius_m=120.0,
            vertical_bound_m=float(episode_spec["activity_vertical_m"]),
            horizon_steps=int(episode_spec["horizon_steps"]),
            width=int(observation_spec["width"]),
            height=int(observation_spec["height"]),
        ),
        int(episode_spec["horizon_steps"]),
        float(episode_spec["activity_vertical_m"]),
        device=device,
    )
    expected = runtime.model.forward_sequence(
        batch["track_features"], batch["track_classes"], batch["track_mask"],
        batch["sequence_mask"], batch["belief_update_mask"], batch["pose_features"],
        batch["action_history"], batch["local_geometry"], batch["source_features"],
        batch["source_mask"], batch["source_position_local"],
    )
    projection = _projection(observation_spec)
    maximum_steps = len(agent_rows) if args.maximum_steps is None else min(
        len(agent_rows), args.maximum_steps
    )
    tolerance = 1.0e-5 if device.type == "cuda" else 1.0e-6
    maxima = {name: 0.0 for name in (
        "belief", "log_belief", "action_probability", "coordinate", "value",
        "geometry", "pose", "source", "history",
    )}
    source_started = False
    frozen_reference = None
    for index in range(maximum_steps):
        step = index + 1
        depths = []
        for view in ("oblique", "nadir"):
            with np.load(root / "agent" / "depth" / f"{step:03d}_{view}.npz") as archive:
                depths.append(np.asarray(archive["depth"], dtype=np.float32))
        pair = _Pair(_Frame(depths[0], projection), _Frame(depths[1], projection))
        odometry_payload = agent_rows[index]["odometry"]
        observation = SimpleNamespace(odometry=AgentOdometry(
            position_local=tuple(odometry_payload["position_local"]),
            yaw_from_start_degrees=float(odometry_payload["yaw_from_start_degrees"]),
        ))
        grounded = _grounded(step, awareness[index]["grounded_tracks"])
        snapshot = runtime.update(grounded, observation, pair, action_count=index)
        comparisons = {
            "belief": (snapshot.belief, expected["belief"][0, index].cpu().numpy()),
            "action_probability": (
                np.asarray(snapshot.raw_action_probabilities),
                torch.softmax(expected["action_logits"][0, index], dim=-1).cpu().numpy(),
            ),
            "coordinate": (
                np.asarray(snapshot.event_estimate_local),
                expected["event_estimate_local"][0, index].cpu().numpy(),
            ),
            "value": (
                np.asarray(snapshot.remaining_value),
                expected["remaining_value"][0, index].cpu().numpy(),
            ),
        }
        expected_log = expected["log_belief"][0, index].cpu().numpy()
        online_finite = np.isfinite(snapshot.log_belief)
        expected_finite = np.isfinite(expected_log)
        if not np.array_equal(online_finite, expected_finite):
            raise RuntimeError(f"Online/offline log-belief support mismatch at step={step}")
        finite = online_finite
        comparisons["log_belief"] = (
            snapshot.log_belief[finite],
            expected_log[finite],
        )
        row = runtime.last_training_row
        comparisons.update({
            "geometry": (row["local_geometry"], item["local_geometry"][index].numpy()),
            "pose": (row["pose_features"], item["pose_features"][index].numpy()),
            "source": (row["source_features"], item["source_features"][index].numpy()),
            "history": (row["action_history"], item["action_history"][index].numpy()),
        })
        for name, (left, right) in comparisons.items():
            difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            maxima[name] = max(maxima[name], difference)
            if difference > tolerance:
                raise RuntimeError(
                    f"Online/offline {name} mismatch at step={step}: {difference}"
                )
        if not snapshot.source_visible_now:
            if snapshot.legal_action_probabilities[ACTION_NAMES.index("STOP")] != 0.0:
                raise RuntimeError("STOP remained legal without a grounded source")
        else:
            if not source_started:
                source_started = True
                frozen_reference = snapshot.belief.copy()
            elif not np.array_equal(snapshot.belief, frozen_reference):
                raise RuntimeError("Belief changed after the source boundary")
        runtime.commit_action_name(ACTION_NAMES[int(item["action_target"][index])])
    runtime.reset()
    initial = runtime.model.belief_updater.initial_log_belief(1, torch.float32).to(device)
    finite = torch.isfinite(initial) & torch.isfinite(runtime.recurrent_log_belief)
    if not torch.equal(initial[finite], runtime.recurrent_log_belief[finite]):
        raise RuntimeError("Runtime reset did not restore the uniform prior")
    print(
        f"PASS Stage3C online/offline parity episode={record.episode_id} "
        f"split={selected_split} steps={maximum_steps} device={device} "
        f"max={max(maxima.values()):.3g} details={maxima}"
    )
    print("STOP legality, source freeze, executed-action history, and reset PASS.")
    print("No RGB-D, geometry, or policy prediction payload was written to disk.")


if __name__ == "__main__":
    main()
