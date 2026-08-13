"""Interactively replay Stage 2E dataset episodes offline."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.visualize_stage2e_trajectory import (  # noqa: E402
    Stage2EPlayer,
)


ENTITY_ROLES = {
    1: "FIRE_SOURCE_VEHICLE",
    2: "FIRE_TRUCK",
    3: "FIREFIGHTER_DRIVER",
    4: "FLEEING_PEDESTRIAN",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Play one episode or every episode in a Stage 2E dataset batch "
            "without "
            "connecting to GTA or loading its metric Depth payload."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help=(
            "One successful episode directory or a batch directory "
            "containing episode_* subdirectories"
        ),
    )
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"Blank JSONL record at {path}:{line_number}"
                )
            records.append(json.loads(line))
    return records


def _world_to_local(blueprint, world_position):
    pose = blueprint["absolute_pose"]
    yaw = math.radians(float(pose[5]))
    forward = np.asarray(
        (-math.sin(yaw), math.cos(yaw), 0.0),
        dtype=np.float64,
    )
    right = np.asarray(
        (math.cos(yaw), math.sin(yaw), 0.0),
        dtype=np.float64,
    )
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    delta = np.asarray(world_position, dtype=np.float64) - np.asarray(
        pose[:3],
        dtype=np.float64,
    )
    return [
        float(np.dot(delta, forward)),
        float(np.dot(delta, right)),
        float(np.dot(delta, up)),
    ]


def _entity_record(entity, blueprint):
    role_value = int(entity["role"])
    try:
        role = ENTITY_ROLES[role_value]
    except KeyError as error:
        raise RuntimeError(
            f"Unsupported scenario entity role {role_value}"
        ) from error
    result = dict(entity)
    result["role"] = role
    result["position_local"] = _world_to_local(
        blueprint,
        entity["position"],
    )
    return result


def _validate_step_alignment(index, agent, teacher, truth):
    steps = (
        int(agent["step_index"]),
        int(teacher["step_index"]),
        int(truth["step_index"]),
    )
    if len(set(steps)) != 1:
        raise RuntimeError(
            "Dataset JSONL step mismatch at record "
            f"{index}: agent={steps[0]}, teacher={steps[1]}, "
            f"truth={steps[2]}"
        )
    if steps[0] <= 0:
        raise RuntimeError(f"Dataset step index must be positive: {steps[0]}")
    return steps[0]


def load_dataset_episode(episode):
    root = Path(episode).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset episode directory not found: {root}")

    paths = {
        "agent_episode": root / "agent" / "episode.json",
        "agent_steps": root / "agent" / "steps.jsonl",
        "teacher_episode": root / "teacher" / "episode.json",
        "teacher_steps": root / "teacher" / "awareness.jsonl",
        "beliefs": root / "teacher" / "beliefs.npz",
        "truth_episode": root / "evaluation_truth" / "episode.json",
        "truth_steps": root / "evaluation_truth" / "steps.jsonl",
        "summary": root / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Stage 2E dataset episode is incomplete; missing: "
            + ", ".join(missing)
        )

    agent_metadata = _read_json(paths["agent_episode"])
    teacher_metadata = _read_json(paths["teacher_episode"])
    truth_metadata = _read_json(paths["truth_episode"])
    summary = _read_json(paths["summary"])
    agent_steps = _read_jsonl(paths["agent_steps"])
    teacher_steps = _read_jsonl(paths["teacher_steps"])
    truth_steps = _read_jsonl(paths["truth_steps"])
    with np.load(paths["beliefs"]) as archive:
        beliefs = np.asarray(archive["belief"], dtype=np.float32)

    counts = (
        len(agent_steps),
        len(teacher_steps),
        len(truth_steps),
        len(beliefs),
    )
    if counts[0] == 0 or len(set(counts)) != 1:
        raise RuntimeError(
            "Dataset step-count mismatch: "
            f"agent={counts[0]}, teacher={counts[1]}, "
            f"truth={counts[2]}, beliefs={counts[3]}"
        )

    blueprint = truth_metadata["start_blueprint"]
    observation_spec = agent_metadata["observation_spec"]
    frames = []
    previous_step = None
    for index, (agent, teacher, truth) in enumerate(
        zip(agent_steps, teacher_steps, truth_steps, strict=True)
    ):
        step_index = _validate_step_alignment(
            index,
            agent,
            teacher,
            truth,
        )
        if previous_step is not None and step_index <= previous_step:
            raise RuntimeError(
                "Dataset step indices are not strictly increasing: "
                f"{previous_step} -> {step_index}"
            )
        previous_step = step_index

        rgb = {}
        for view_name in ("oblique", "nadir"):
            relative = Path("agent") / "rgb" / (
                f"{step_index:03d}_{view_name}.jpg"
            )
            if not (root / relative).is_file():
                raise RuntimeError(f"Dataset RGB is missing: {root / relative}")
            rgb[view_name] = {
                "path": relative.as_posix(),
                "frame_id": int(agent["frame_ids"][view_name]),
                "width": int(observation_spec["width"]),
                "height": int(observation_spec["height"]),
            }

        event_position = truth["event_position"]
        frames.append(
            {
                "index": index,
                "observation_step": step_index,
                "action_index": index + 1,
                "action": agent["action"],
                "action_execution": "EXECUTED",
                "clock": agent["clock"],
                "rgb": rgb,
                "odometry": agent["odometry"],
                "grounded_tracks": teacher["grounded_tracks"],
                "awareness": teacher["awareness"],
                "evaluation_truth": {
                    "event_active": bool(truth["event_active"]),
                    "event_position": event_position,
                    "event_position_local": _world_to_local(
                        blueprint,
                        event_position,
                    ),
                    "entities": [
                        _entity_record(entity, blueprint)
                        for entity in truth["entities"]
                    ],
                    "valid_dynamic_cue_so_far": bool(
                        truth["valid_dynamic_cue_so_far"]
                    ),
                },
            }
        )

    result = summary.get("result", {})
    if not bool(result.get("success")):
        raise RuntimeError(
            "Dataset episode summary does not declare a successful rollout"
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "error": None,
        "result": result,
        "metadata": {
            "agent": agent_metadata,
            "teacher": teacher_metadata,
            "evaluation_truth": truth_metadata,
            "dataset_summary": summary,
        },
        "frames": frames,
    }
    return root, payload, beliefs


def discover_dataset_episodes(dataset):
    root = Path(dataset).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {root}")
    if (root / "agent" / "steps.jsonl").is_file():
        return [root]
    episodes = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.startswith("episode_")
            and not path.name.endswith(".partial")
        ),
        key=lambda path: path.name,
    )
    if not episodes:
        raise RuntimeError(
            f"No completed Stage 2E episode_* directories under {root}"
        )
    return episodes


class Stage2EDatasetPlayer(Stage2EPlayer):
    def __init__(self, episode_paths, args):
        self.episode_paths = tuple(episode_paths)
        if not self.episode_paths:
            raise ValueError("episode_paths cannot be empty")
        self.episode_index = 0
        root, payload, beliefs = load_dataset_episode(
            self.episode_paths[0]
        )
        super().__init__(root, payload, beliefs, args)

    def _select_episode(self, episode_index, frame_index):
        root, payload, beliefs = load_dataset_episode(
            self.episode_paths[episode_index]
        )
        self.episode_index = int(episode_index)
        self.root = root
        self.payload = payload
        self.frames = payload["frames"]
        self.beliefs = beliefs
        frame_index = int(frame_index)
        if frame_index < 0:
            frame_index += len(self.frames)
        if not 0 <= frame_index < len(self.frames):
            raise IndexError(
                f"Frame {frame_index} is outside episode with "
                f"{len(self.frames)} frames"
            )
        self.index = frame_index

    def _draw_info(self, frame):
        super()._draw_info(frame)
        episode_text = self.info_axis.text(
            0.99,
            0.02,
            (
                f"Episode {self.episode_index + 1}/"
                f"{len(self.episode_paths)}: {self.root.name}\n"
                "Up/Down: previous/next episode"
            ),
            transform=self.info_axis.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            family="monospace",
            clip_on=True,
        )
        episode_text.set_in_layout(False)

    def _render(self):
        super()._render()
        self.figure.suptitle(
            (
                f"Stage 2E dataset replay [{self.episode_index + 1}/"
                f"{len(self.episode_paths)}] {self.root.name} -- "
                "green: grounded cue, red: grounded goal"
            ),
            fontsize=14,
        )

    def _advance(self, delta):
        candidate = self.index + delta
        if 0 <= candidate < len(self.frames):
            self.index = candidate
            self._render()
            return

        if delta > 0:
            next_episode = self.episode_index + 1
            if next_episode < len(self.episode_paths):
                self._select_episode(next_episode, 0)
            elif self.loop:
                self._select_episode(0, 0)
            else:
                self.paused = True
                self.index = len(self.frames) - 1
        else:
            previous_episode = self.episode_index - 1
            if previous_episode >= 0:
                self._select_episode(previous_episode, -1)
            elif self.loop:
                last_episode = len(self.episode_paths) - 1
                self._select_episode(last_episode, -1)
            else:
                self.paused = True
                self.index = 0
        self._render()

    def _on_key(self, event):
        if event.key in ("up", "down"):
            self.paused = True
            delta = -1 if event.key == "up" else 1
            target = self.episode_index + delta
            if self.loop:
                target %= len(self.episode_paths)
            else:
                target = min(max(target, 0), len(self.episode_paths) - 1)
            if target != self.episode_index:
                self._select_episode(target, 0)
                self._render()
            return
        super()._on_key(event)


def main():
    args = _parse_args()
    episode_paths = discover_dataset_episodes(args.dataset)
    Stage2EDatasetPlayer(episode_paths, args)
    plt.show()


if __name__ == "__main__":
    main()
