"""Replay one compact Stage 3C online trajectory without GTA."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.policy_dataset import ACTION_NAMES  # noqa: E402
from validation.visualize_stage2e_trajectory import (  # noqa: E402
    Stage2EPlayer, _action_text, _load, _wrap_info_lines,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


class OnlineBeliefPolicyPlayer(Stage2EPlayer):
    full_path_label = "full learned-policy path"
    map_title = "Stage 3C Spatial RNN belief, trajectory, and evaluation truth"

    def _draw_map(self, frame):
        display = deepcopy(frame)
        display["awareness"] = {"primary_hypothesis_local": None}
        super()._draw_map(display)
        map_xy = frame["awareness"]["map_local_xy"]
        self.map_axis.scatter(
            map_xy[0], map_xy[1], color="magenta", marker="X", s=80,
            zorder=8, label="Spatial RNN MAP",
        )
        self.map_axis.legend(fontsize=7, loc="best")

    def _draw_info(self, frame):
        axis = self.info_axis
        axis.clear()
        axis.set_axis_off()
        awareness = frame["awareness"]
        odometry = frame["odometry"]
        clock = frame["clock"]
        result = self.payload.get("result") or {}
        probabilities = awareness["legal_action_probabilities"]
        probability_text = " ".join(
            f"{name}={float(probabilities[index]):.2f}"
            for index, name in enumerate(ACTION_NAMES)
        )
        error = result.get("localization_error_m")
        error_text = "None" if error is None else f"{float(error):.3f}m"
        lines = [
            f"Stage 3C online trajectory: {self.payload['status']}",
            (
                f"Frame {self.index + 1}/{len(self.frames)} | step={frame['observation_step']} | "
                f"executed={_action_text(frame['action'])} | "
                f"execution={frame.get('action_execution', 'UNKNOWN')}"
            ),
            "",
            (
                f"Mode={awareness['mode']} | checkpoint={awareness['checkpoint_name']} "
                f"epoch={awareness['checkpoint_epoch']} dagger={awareness['dagger_iteration']}"
            ),
            (
                f"Odometry: forward={odometry['position_local'][0]:.2f}, "
                f"right={odometry['position_local'][1]:.2f}, up={odometry['position_local'][2]:.2f}, "
                f"yaw={odometry['yaw_from_start_degrees']:.1f} deg"
            ),
            f"Lockstep: step={clock['step_index']} elapsed={clock['actual_elapsed_ms']}ms",
            "",
            (
                f"Belief: updated={awareness['belief_updated']} "
                f"source_seen={awareness['source_seen']} source_now={awareness['source_visible_now']} "
                f"tracks={awareness['evidence_track_ids']}"
            ),
            (
                f"Belief: entropy={awareness['belief_entropy']:.3f} "
                f"MAP=({awareness['map_local_xy'][0]:.1f},{awareness['map_local_xy'][1]:.1f})m "
                f"credible50/80/90={awareness['credible_areas_m2']}m2"
            ),
            "",
            f"Policy probabilities: {probability_text}",
            (
                f"Policy proposed={awareness['proposed_action']} "
                f"executed={awareness['executed_action']} by={awareness['executed_by']}"
            ),
            (
                f"Expert label={awareness['expert_action']} available="
                f"{awareness['expert_label_available']} error={awareness['expert_error']}"
            ),
            f"Last four executed actions={awareness['action_history']}",
            (
                f"Source track={awareness['source_track_id']} age={awareness['source_age']} "
                f"coordinate estimate={awareness['event_estimate_local']}"
            ),
            f"Remaining-action value={awareness['remaining_value']:.3f}",
            (
                f"Valid dynamic cue so far="
                f"{frame['evaluation_truth']['valid_dynamic_cue_so_far']}"
            ),
            f"Final success={result.get('success')} localization_error={error_text}",
            (
                f"Source-blind final: NLL={result.get('last_source_blind_event_nll')} "
                f"MAP error={result.get('last_source_blind_map_error_m')}m"
            ),
            "",
            "Controls: Space pause/play | Left/Right step | Home/End | Q close",
        ]
        text = axis.text(
            0.01, 0.98, "\n".join(_wrap_info_lines(lines)),
            transform=axis.transAxes, va="top", ha="left", fontsize=9.3,
            family="monospace", clip_on=True,
        )
        text.set_in_layout(False)

    def _render(self):
        frame = self.frames[self.index]
        self._draw_view(self.oblique_axis, frame, "oblique")
        self._draw_view(self.nadir_axis, frame, "nadir")
        self._draw_map(frame)
        self._draw_info(frame)
        self.figure.suptitle(
            "Stage 3C explicit-belief policy replay -- green: grounded cue, red: source",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()


def show_recording(recording, interval_ms=250, loop=False, start_paused=False):
    root, payload, beliefs = _load(recording)
    teacher = payload.get("metadata", {}).get("teacher", {})
    if teacher.get("teacher") != "stage3c-explicit-belief-action-policy":
        raise RuntimeError("Recording is not a Stage 3C online policy trajectory")
    args = argparse.Namespace(
        interval_ms=int(interval_ms), loop=bool(loop), start_paused=bool(start_paused)
    )
    OnlineBeliefPolicyPlayer(root, payload, beliefs, args)
    plt.show()


def main():
    args = _arguments()
    show_recording(args.recording, args.interval_ms, args.loop, args.start_paused)


if __name__ == "__main__":
    main()
