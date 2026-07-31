import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


RESPONSE_ROLES = {
    "FIRE_TRUCK",
    "FLEEING_PEDESTRIAN",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively play a Stage 2D trajectory recording. "
            "This script does not connect to GTA."
        )
    )
    parser.add_argument(
        "recording",
        type=Path,
        help=(
            "Recording root, stratum directory, or trajectory.json"
        ),
    )
    parser.add_argument(
        "--stratum",
        choices=("CUE_VISIBLE", "CUE_HIDDEN", "both"),
        default="both",
    )
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_episode(path):
    path = Path(path).resolve()
    payload = _load_json(path)
    if payload.get("schema_version") != 2:
        raise RuntimeError(
            f"Unsupported trajectory schema in {path}"
        )
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"Trajectory has no frames: {path}")
    return {
        "root": path.parent,
        "payload": payload,
        "frames": frames,
    }


def _resolve_episodes(path, stratum):
    path = Path(path).resolve()
    if path.is_file():
        return [_load_episode(path)]
    direct = path / "trajectory.json"
    if direct.is_file():
        return [_load_episode(direct)]
    run_path = path / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(
            f"No run.json or trajectory.json under {path}"
        )
    run = _load_json(run_path)
    episodes = run.get("episodes")
    if isinstance(episodes, list):
        selected = [
            item
            for item in episodes
            if stratum == "both" or item.get("stratum") == stratum
        ]
        if not selected:
            raise RuntimeError(
                f"Recording contains no episodes for {stratum}"
            )
        return [
            _load_episode(path / item["path"])
            for item in selected
        ]
    available = {
        item["name"]: item["path"]
        for item in run.get("strata", [])
    }
    if stratum == "both":
        selected = tuple(
            name
            for name in ("CUE_VISIBLE", "CUE_HIDDEN")
            if name in available
        )
        if not selected:
            raise RuntimeError(
                "Recording manifest contains no playable strata"
            )
    else:
        selected = (stratum,)
    missing = [name for name in selected if name not in available]
    if missing:
        raise RuntimeError(
            f"Recording does not contain strata: {missing}"
        )
    return [
        _load_episode(path / available[name])
        for name in selected
    ]


def _action_text(action):
    action_type = action["type"]
    if action_type in {
        "FORWARD",
        "ASCEND",
        "DESCEND",
        "TURN_LEFT",
        "TURN_RIGHT",
    }:
        return action_type
    if action_type == "HOLD":
        return "HOLD"
    estimate = action["event_estimate_local"]
    return (
        "STOP(estimate_local="
        f"[{estimate[0]:.2f}, {estimate[1]:.2f}, "
        f"{estimate[2]:.2f}])"
    )


def _draw_box(axis, target, view_name):
    view = target[view_name]
    bbox = view["projected_bbox"]
    if bbox is None:
        return
    role = target["role"]
    observable = view["task_observable"]
    if role in RESPONSE_ROLES:
        color = "lime" if observable else "yellow"
        if not observable:
            return
        label = f"CUE {role} #{target['stable_id']}"
        linewidth = 2.5
    elif role == "FIRE_SOURCE_VEHICLE":
        color = "red"
        label = (
            f"GOAL #{target['stable_id']} "
            f"{'observable' if observable else 'partial'}"
        )
        linewidth = 2.5 if observable else 1.5
    elif role == "FIRE_ENVELOPE":
        color = "orange"
        label = "FIRE_ENVELOPE"
        linewidth = 1.5
    else:
        return
    x_min, y_min, x_max, y_max = bbox
    axis.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle="-" if observable else "--",
        )
    )
    axis.text(
        x_min,
        max(0.0, y_min - 4.0),
        label,
        color="black",
        fontsize=7,
        va="bottom",
        bbox={
            "facecolor": color,
            "alpha": 0.78,
            "edgecolor": "none",
            "pad": 1.5,
        },
    )


def _entity_style(role):
    if role == "FIRE_SOURCE_VEHICLE":
        return "red", "*", 100, "fire source"
    if role == "FIRE_TRUCK":
        return "darkorange", "s", 36, "fire truck"
    if role == "FLEEING_PEDESTRIAN":
        return "limegreen", "o", 18, "fleeing pedestrian"
    return "gray", ".", 10, None


class TrajectoryPlayer:
    def __init__(
        self,
        episodes,
        interval_ms,
        loop,
        start_paused,
    ):
        self.episodes = episodes
        self.timeline = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(episodes)
            for frame_index in range(len(episode["frames"]))
        ]
        self.timeline_index = 0
        self.loop = bool(loop)
        self.paused = bool(start_paused)
        self.figure = plt.figure(
            figsize=(16, 10),
            constrained_layout=True,
        )
        grid = self.figure.add_gridspec(2, 2)
        self.oblique_axis = self.figure.add_subplot(grid[0, 0])
        self.nadir_axis = self.figure.add_subplot(grid[0, 1])
        self.map_axis = self.figure.add_subplot(grid[1, 0])
        self.info_axis = self.figure.add_subplot(grid[1, 1])
        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )
        self.figure.canvas.mpl_connect(
            "close_event",
            self._on_close,
        )
        self.timer = self.figure.canvas.new_timer(
            interval=int(interval_ms)
        )
        self.timer.add_callback(self._tick)
        self._render()
        self.timer.start()

    def _current(self):
        episode_index, frame_index = self.timeline[
            self.timeline_index
        ]
        episode = self.episodes[episode_index]
        return episode_index, episode, frame_index, (
            episode["frames"][frame_index]
        )

    def _draw_view(self, axis, episode, frame, view_name):
        axis.clear()
        image_path = (
            episode["root"]
            / frame["rgb"][view_name]["path"]
        )
        image = plt.imread(image_path)
        axis.imshow(image)
        for target in frame["visibility"]["targets"]:
            _draw_box(axis, target, view_name)
        axis.set_xlim(0, image.shape[1] - 1)
        axis.set_ylim(image.shape[0] - 1, 0)
        axis.set_axis_off()
        axis.set_title(
            f"{view_name} "
            f"RGB frame={frame['rgb'][view_name]['frame_id']}"
        )

    def _draw_map(self, episode, frame_index, frame):
        axis = self.map_axis
        axis.clear()
        frames = episode["frames"]
        path = np.asarray(
            [
                item["camera_pose_world"][:3]
                for item in frames
            ],
            dtype=np.float64,
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="lightgray",
            linewidth=1.0,
            label="full camera path",
        )
        axis.plot(
            path[: frame_index + 1, 0],
            path[: frame_index + 1, 1],
            color="royalblue",
            linewidth=2.0,
            label="traversed path",
        )
        pose = frame["camera_pose_world"]
        yaw = math.radians(float(pose[5]))
        axis.quiver(
            pose[0],
            pose[1],
            -math.sin(yaw),
            math.cos(yaw),
            color="blue",
            angles="xy",
            scale_units="xy",
            scale=0.08,
            width=0.007,
            label="camera heading",
        )
        event = frame["scenario"]["event_position"]
        axis.scatter(
            event[0],
            event[1],
            marker="*",
            color="red",
            s=130,
            zorder=6,
            label="event truth",
        )
        used_labels = set()
        for entity in frame["scenario"]["entities"]:
            if not entity["exists"]:
                continue
            color, marker, size, label = _entity_style(
                entity["role"]
            )
            if label in used_labels:
                label = None
            elif label is not None:
                used_labels.add(label)
            position = entity["position"]
            axis.scatter(
                position[0],
                position[1],
                marker=marker,
                color=color,
                s=size,
                alpha=0.85,
                label=label,
            )
        witness = episode["payload"]["witness"]
        cue_pose = witness["cue"]["first_pose"]
        goal_pose = witness["goal"]["pose"]
        axis.scatter(
            cue_pose[0],
            cue_pose[1],
            marker="X",
            color="orange",
            s=70,
            label="witness cue view",
        )
        axis.scatter(
            goal_pose[0],
            goal_pose[1],
            marker="X",
            color="darkred",
            s=70,
            label="witness goal view",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("GTA world X")
        axis.set_ylabel("GTA world Y")
        axis.set_title("World-space trajectory and entities")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")

    def _draw_info(
        self,
        episode_index,
        episode,
        frame_index,
        frame,
    ):
        axis = self.info_axis
        axis.clear()
        axis.set_axis_off()
        pose = frame["camera_pose_world"]
        local = frame["camera_position_start_local"]
        clock = frame["clock"]
        action = _action_text(frame["action"])
        flags = []
        if frame["is_witness_cue_observation"]:
            flags.append("WITNESS CUE OBSERVATION")
        if frame["is_terminal_observation"]:
            flags.append("TERMINAL GOAL OBSERVATION")
        if frame["visibility"]["cue_task_observable"]:
            flags.append("CUE TASK-OBSERVABLE NOW")
        if frame["visibility"]["event_task_observable"]:
            flags.append("GOAL TASK-OBSERVABLE NOW")
        lines = [
            (
                f"Episode {episode_index + 1}/{len(self.episodes)}: "
                f"{episode['payload']['visibility_stratum']}"
            ),
            (
                f"Frame {frame_index + 1}/{len(episode['frames'])} | "
                f"observation step={frame['observation_step']} | "
                f"action index={frame['action_index']}"
            ),
            "",
            f"CURRENT ACTION: {action}",
            "",
            (
                "World pose: "
                f"x={pose[0]:.2f}, y={pose[1]:.2f}, z={pose[2]:.2f}, "
                f"pitch={pose[3]:.1f}, roll={pose[4]:.1f}, "
                f"yaw={pose[5]:.1f}"
            ),
            (
                "Start-local position: "
                f"forward={local[0]:.2f}, right={local[1]:.2f}, "
                f"up={local[2]:.2f}"
            ),
            (
                f"Lockstep: step={clock['step_index']} "
                f"game_timer={clock['game_timer_ms']}ms "
                f"elapsed={clock['actual_elapsed_ms']}ms"
            ),
            (
                f"Event active={frame['scenario']['event_active']} | "
                f"cue_reproduced="
                f"{episode['payload']['cue_reproduced']}"
            ),
            "",
            " | ".join(flags) if flags else "No task-observable cue/goal flag",
            "",
            "Controls: Space pause/play | Left/Right step | "
            "Home/End | Q close",
        ]
        axis.text(
            0.01,
            0.98,
            "\n".join(lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    def _render(self):
        (
            episode_index,
            episode,
            frame_index,
            frame,
        ) = self._current()
        self._draw_view(
            self.oblique_axis,
            episode,
            frame,
            "oblique",
        )
        self._draw_view(
            self.nadir_axis,
            episode,
            frame,
            "nadir",
        )
        self._draw_map(episode, frame_index, frame)
        self._draw_info(
            episode_index,
            episode,
            frame_index,
            frame,
        )
        self.figure.suptitle(
            "Stage 2D recorded replay — "
            "green: observable response cue, red: fire source",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def _advance(self, delta):
        candidate = self.timeline_index + delta
        if 0 <= candidate < len(self.timeline):
            self.timeline_index = candidate
        elif self.loop:
            self.timeline_index = candidate % len(self.timeline)
        else:
            self.paused = True
            self.timeline_index = min(
                max(candidate, 0),
                len(self.timeline) - 1,
            )
        self._render()

    def _tick(self):
        if not self.paused:
            self._advance(1)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "right":
            self.paused = True
            self._advance(1)
        elif event.key == "left":
            self.paused = True
            self._advance(-1)
        elif event.key == "home":
            self.paused = True
            self.timeline_index = 0
            self._render()
        elif event.key == "end":
            self.paused = True
            self.timeline_index = len(self.timeline) - 1
            self._render()
        elif event.key in ("q", "escape"):
            plt.close(self.figure)

    def _on_close(self, _event):
        self.timer.stop()


def main():
    args = _parse_args()
    episodes = _resolve_episodes(
        args.recording,
        args.stratum,
    )
    TrajectoryPlayer(
        episodes,
        args.interval_ms,
        args.loop,
        args.start_paused,
    )
    plt.show()


if __name__ == "__main__":
    main()
