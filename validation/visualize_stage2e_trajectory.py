"""Interactively replay one Stage 2E validation trajectory offline."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


RESPONSE_CLASSES = {"FIRE_TRUCK", "PEDESTRIAN"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Play a Stage 2E validation recording without connecting to GTA."
        )
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


def _load(recording):
    root = Path(recording).resolve()
    path = root / "trajectory.json" if root.is_dir() else root
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported Stage 2E recording: {path}")
    if not payload.get("frames"):
        raise RuntimeError(f"Stage 2E recording has no frames: {path}")
    beliefs_path = path.parent / "beliefs.npz"
    with np.load(beliefs_path) as archive:
        beliefs = np.asarray(archive["belief"], dtype=np.float32)
    if len(beliefs) != len(payload["frames"]):
        raise RuntimeError("Belief/frame count mismatch")
    return path.parent, payload, beliefs


def _action_text(action):
    if action["type"] != "STOP":
        return action["type"]
    estimate = action["event_estimate_local"]
    return (
        "STOP(["
        f"{estimate[0]:.1f}, {estimate[1]:.1f}, {estimate[2]:.1f}])"
    )


def _entity_style(role):
    if role == "FIRE_SOURCE_VEHICLE":
        return "red", "*", 110, "fire source truth"
    if role == "FIRE_TRUCK":
        return "darkorange", "s", 32, "fire truck truth"
    if role == "FLEEING_PEDESTRIAN":
        return "limegreen", "o", 14, "pedestrian truth"
    return "gray", ".", 8, None


class Stage2EPlayer:
    def __init__(self, root, payload, beliefs, args):
        self.root = root
        self.payload = payload
        self.frames = payload["frames"]
        self.beliefs = beliefs
        self.index = 0
        self.loop = bool(args.loop)
        self.paused = bool(args.start_paused)
        self.figure = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = self.figure.add_gridspec(2, 2)
        self.oblique_axis = self.figure.add_subplot(grid[0, 0])
        self.nadir_axis = self.figure.add_subplot(grid[0, 1])
        self.map_axis = self.figure.add_subplot(grid[1, 0])
        self.info_axis = self.figure.add_subplot(grid[1, 1])
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.timer = self.figure.canvas.new_timer(interval=args.interval_ms)
        self.timer.add_callback(self._tick)
        self._render()
        self.timer.start()

    def _draw_view(self, axis, frame, view_name):
        axis.clear()
        image = plt.imread(self.root / frame["rgb"][view_name]["path"])
        axis.imshow(image)
        for track in frame["grounded_tracks"]:
            if track["view_name"] != view_name:
                continue
            semantic_class = track["semantic_class"]
            if semantic_class == "FIRE_SOURCE":
                color = "red"
                label = f"GOAL track #{track['track_id']}"
            elif semantic_class in RESPONSE_CLASSES:
                color = "lime"
                label = f"CUE {semantic_class} #{track['track_id']}"
            else:
                continue
            x_min, y_min, x_max, y_max = track["projected_bbox"]
            axis.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                )
            )
            axis.text(
                x_min,
                max(0.0, y_min - 4.0),
                label,
                fontsize=7,
                color="black",
                bbox={
                    "facecolor": color,
                    "alpha": 0.78,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
        axis.set_xlim(0, image.shape[1] - 1)
        axis.set_ylim(image.shape[0] - 1, 0)
        axis.set_axis_off()
        axis.set_title(
            f"{view_name} RGB frame={frame['rgb'][view_name]['frame_id']}"
        )

    def _draw_map(self, frame):
        axis = self.map_axis
        axis.clear()
        belief = self.beliefs[self.index]
        if belief.size:
            radius = 120.0
            axis.imshow(
                belief.T,
                origin="lower",
                extent=(-radius, radius, -radius, radius),
                cmap="Blues",
                alpha=0.55,
                aspect="equal",
            )
        path = np.asarray(
            [item["odometry"]["position_local"] for item in self.frames],
            dtype=np.float64,
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="lightgray",
            linewidth=1.0,
            label="full expert path",
        )
        axis.plot(
            path[: self.index + 1, 0],
            path[: self.index + 1, 1],
            color="royalblue",
            linewidth=2.0,
            label="traversed path",
        )
        position = path[self.index]
        yaw = math.radians(
            float(frame["odometry"]["yaw_from_start_degrees"])
        )
        axis.quiver(
            position[0],
            position[1],
            math.cos(yaw),
            -math.sin(yaw),
            color="blue",
            angles="xy",
            scale_units="xy",
            scale=0.08,
            width=0.007,
            label="camera heading",
        )
        event = frame["evaluation_truth"]["event_position_local"]
        axis.scatter(
            event[0],
            event[1],
            marker="*",
            color="red",
            s=130,
            zorder=6,
            label="event truth",
        )
        labels = set()
        for entity in frame["evaluation_truth"]["entities"]:
            if not entity["exists"]:
                continue
            color, marker, size, label = _entity_style(entity["role"])
            if label in labels:
                label = None
            elif label is not None:
                labels.add(label)
            local = entity["position_local"]
            axis.scatter(
                local[0],
                local[1],
                color=color,
                marker=marker,
                s=size,
                alpha=0.85,
                label=label,
            )
        hypothesis = frame["awareness"]["primary_hypothesis_local"]
        if hypothesis is not None:
            axis.scatter(
                hypothesis[0],
                hypothesis[1],
                color="magenta",
                marker="X",
                s=70,
                label="belief mode",
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-120.0, 120.0)
        axis.set_ylim(-120.0, 120.0)
        axis.set_xlabel("start-local forward (m)")
        axis.set_ylabel("start-local right (m)")
        axis.set_title("Expert trajectory, belief, and evaluation truth")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")

    def _draw_info(self, frame):
        axis = self.info_axis
        axis.clear()
        axis.set_axis_off()
        awareness = frame["awareness"]
        odometry = frame["odometry"]
        clock = frame["clock"]
        evidence = awareness["motion_evidence"]
        visible = awareness["visible_tracks"]
        lines = [
            f"Stage 2E trajectory: {self.payload['status']}",
            (
                f"Frame {self.index + 1}/{len(self.frames)} | "
                f"step={frame['observation_step']} | "
                f"action={_action_text(frame['action'])}"
            ),
            "",
            (
                "Odometry: "
                f"forward={odometry['position_local'][0]:.2f}, "
                f"right={odometry['position_local'][1]:.2f}, "
                f"up={odometry['position_local'][2]:.2f}, "
                f"yaw={odometry['yaw_from_start_degrees']:.1f} deg"
            ),
            (
                f"Lockstep: step={clock['step_index']} "
                f"elapsed={clock['actual_elapsed_ms']} ms"
            ),
            "",
            f"Intent: {awareness['intent']}",
            (
                f"Belief: entropy={awareness['belief_entropy']:.3f}, "
                f"mode={awareness['primary_mode_id']}, "
                f"mass={awareness['primary_mode_mass']:.4f}, "
                f"ambiguous={awareness['belief_ambiguous']}"
            ),
            (
                f"Planner: replanned={awareness['planner_replanned']}, "
                f"remaining={awareness['planner_remaining_actions']}, "
                f"failure={awareness['planner_failure']}"
            ),
            (
                f"Tracks: visible={len(visible)}, "
                f"motion_evidence={len(evidence)}, "
                f"support={awareness['supporting_track_ids']}, "
                f"contradict={awareness['contradicting_track_ids']}"
            ),
            (
                "Valid dynamic cue so far="
                f"{frame['evaluation_truth']['valid_dynamic_cue_so_far']}"
            ),
            "",
            "Controls: Space pause/play | Left/Right step | Home/End | Q close",
        ]
        axis.text(
            0.01,
            0.98,
            "\n".join(lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            family="monospace",
        )

    def _render(self):
        frame = self.frames[self.index]
        self._draw_view(self.oblique_axis, frame, "oblique")
        self._draw_view(self.nadir_axis, frame, "nadir")
        self._draw_map(frame)
        self._draw_info(frame)
        self.figure.suptitle(
            "Stage 2E expert replay — green: grounded cue, red: grounded goal",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def _advance(self, delta):
        candidate = self.index + delta
        if 0 <= candidate < len(self.frames):
            self.index = candidate
        elif self.loop:
            self.index = candidate % len(self.frames)
        else:
            self.paused = True
            self.index = min(max(candidate, 0), len(self.frames) - 1)
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
            self.index = 0
            self._render()
        elif event.key == "end":
            self.paused = True
            self.index = len(self.frames) - 1
            self._render()
        elif event.key in ("q", "escape"):
            plt.close(self.figure)

    def _on_close(self, _event):
        self.timer.stop()


def main():
    args = _parse_args()
    root, payload, beliefs = _load(args.recording)
    Stage2EPlayer(root, payload, beliefs, args)
    plt.show()


if __name__ == "__main__":
    main()
