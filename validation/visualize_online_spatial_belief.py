"""Replay an online Spatial RNN belief trajectory without connecting to GTA."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.visualize_stage2e_trajectory import (  # noqa: E402
    Stage2EPlayer,
    _action_text,
    _load,
    _wrap_info_lines,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Play one compact online Spatial RNN GTA recording."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    return args


class OnlineSpatialBeliefPlayer(Stage2EPlayer):
    full_path_label = "full online path"
    map_title = "Online Spatial RNN belief, trajectory, and evaluation truth"

    def _draw_map(self, frame):
        display_frame = deepcopy(frame)
        display_frame["awareness"] = dict(frame["awareness"]["navigation"])
        display_frame["awareness"]["primary_hypothesis_local"] = None
        super()._draw_map(display_frame)
        map_xy = frame["awareness"]["map_local_xy"]
        self.map_axis.scatter(
            map_xy[0],
            map_xy[1],
            color="magenta",
            marker="X",
            s=80,
            zorder=8,
            label="Spatial RNN MAP",
        )
        self.map_axis.legend(fontsize=7, loc="best")

    def _draw_info(self, frame):
        axis = self.info_axis
        axis.clear()
        axis.set_axis_off()
        awareness = frame["awareness"]
        navigation = awareness["navigation"]
        odometry = frame["odometry"]
        clock = frame["clock"]
        result = self.payload.get("result") or {}
        error = result.get("localization_error_m")
        error_text = "None" if error is None else f"{float(error):.3f}m"
        step_timing = frame.get("timing", {})
        action_timing = step_timing.get("action", {})
        advance_seconds = float(action_timing.get("advance_seconds", 0.0))
        capture_seconds = float(action_timing.get("capture_seconds", 0.0))
        lines = [
            f"Stage 3B online trajectory: {self.payload['status']}",
            (
                f"Frame {self.index + 1}/{len(self.frames)} | "
                f"step={frame['observation_step']} | "
                f"action={_action_text(frame['action'])} | "
                f"execution={frame.get('action_execution', 'UNKNOWN')}"
            ),
            "",
            (
                f"Mode={awareness['mode']} | checkpoint="
                f"{awareness['checkpoint_name']} epoch={awareness['checkpoint_epoch']}"
            ),
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
            (
                f"RNN: source_seen={awareness['source_seen']} "
                f"source_now={awareness['source_visible_now']} "
                f"inference_started={awareness['inference_started']}"
            ),
            (
                f"RNN: updated={awareness['belief_updated']} "
                f"evidence_tracks={awareness['evidence_track_ids']}"
            ),
            (
                "Navigation confidence: "
                f"ready={awareness.get('belief_navigation_ready')} "
                "reason="
                f"{awareness.get('belief_navigation_reason', 'UNRECORDED')}"
            ),
            (
                f"Belief: entropy={awareness['belief_entropy']:.3f} "
                f"MAP=({awareness['map_local_xy'][0]:.1f}, "
                f"{awareness['map_local_xy'][1]:.1f})m "
                f"cell={awareness['map_cell']}"
            ),
            f"Credible areas 50/80/90={awareness['credible_areas_m2']} m2",
            (
                f"Timing: ground={float(step_timing.get('grounding_seconds', 0.0)):.3f}s "
                f"model={awareness['model_seconds']:.3f}s "
                f"planner={awareness['planner_seconds']:.3f}s "
                f"advance={advance_seconds:.3f}s capture={capture_seconds:.3f}s"
            ),
            "",
            f"Intent: {navigation['intent']}",
            (
                f"Planner: replanned={navigation['planner_replanned']} "
                f"remaining={navigation['planner_remaining_actions']} "
                f"failure={navigation['planner_failure']}"
            ),
            (
                "Valid dynamic cue so far="
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
            0.01,
            0.98,
            "\n".join(_wrap_info_lines(lines)),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10.0,
            family="monospace",
            clip_on=True,
        )
        text.set_in_layout(False)

    def _render(self):
        frame = self.frames[self.index]
        self._draw_view(self.oblique_axis, frame, "oblique")
        self._draw_view(self.nadir_axis, frame, "nadir")
        self._draw_map(frame)
        self._draw_info(frame)
        self.figure.suptitle(
            "Stage 3B online Spatial RNN replay -- green: grounded cue, red: source",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()


def show_recording(recording, interval_ms=250, loop=False, start_paused=False):
    root, payload, beliefs = _load(recording)
    metadata = payload.get("metadata", {})
    teacher = metadata.get("teacher", {})
    if teacher.get("teacher") != "online-spatial-rnn-belief":
        raise RuntimeError("Recording is not an online Spatial RNN trajectory")
    args = argparse.Namespace(
        interval_ms=int(interval_ms),
        loop=bool(loop),
        start_paused=bool(start_paused),
    )
    OnlineSpatialBeliefPlayer(root, payload, beliefs, args)
    plt.show()


def main():
    args = _parse_args()
    show_recording(
        args.recording,
        interval_ms=args.interval_ms,
        loop=args.loop,
        start_paused=args.start_paused,
    )


if __name__ == "__main__":
    main()
