"""Audit rendered fire-particle coverage from controlled viewpoints in GTA.

The tool keeps one running fire scenario frozen in lockstep while moving the
camera over a deterministic viewpoint grid.  Geometry only selects the image
region to inspect.  A multi-frame temporal-activity diagnostic then measures
whether changing rendered pixels appear inside the projected fire envelope.

The activity values are diagnostics, not benchmark truth: no RGB threshold is
used to declare the fire visible.  Nothing is written to disk.
"""

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    LockstepSession,
    OBLIQUE_PITCH_DEGREES,
    VisibilityTargetRole,
)
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    assess_visibility,
    pair_view_matrices,
)


@dataclass(frozen=True)
class ViewDiagnostic:
    name: str
    frame_id: int
    thumbnail: np.ndarray
    activity_thumbnail: np.ndarray
    projected_bbox: tuple | None
    clear_samples: int
    total_samples: int
    projected_span_pixels: float
    roi_mean: float | None
    roi_p95: float | None
    ring_mean: float | None
    activity_excess: float | None


@dataclass(frozen=True)
class PoseDiagnostic:
    index: int
    radius: float
    height: float
    azimuth: float
    camera_position: tuple
    yaw: float
    clock_step: int
    game_timer_ms: int
    latency_ms: float
    oblique: ViewDiagnostic
    nadir: ViewDiagnostic


def _finite_positive_values(parser, label, values):
    result = tuple(float(value) for value in values)
    if not result or any(
        not math.isfinite(value) or value <= 0.0 for value in result
    ):
        parser.error(f"{label} must contain positive finite values")
    return result


def _finite_azimuths(parser, values):
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        parser.error("--azimuths must contain finite values")
    return result


def _yaw_toward(camera_position, target_position):
    delta_x = float(target_position[0]) - float(camera_position[0])
    delta_y = float(target_position[1]) - float(camera_position[1])
    if math.hypot(delta_x, delta_y) < 1.0e-6:
        raise ValueError("Camera and event have the same horizontal position")
    return math.degrees(math.atan2(-delta_x, delta_y))


def _camera_position(event_position, radius, height, azimuth_degrees):
    azimuth = math.radians(float(azimuth_degrees))
    return (
        float(event_position[0]) + float(radius) * math.cos(azimuth),
        float(event_position[1]) + float(radius) * math.sin(azimuth),
        float(event_position[2]) + float(height),
    )


def _clamped_box(box, width, height):
    if box is None:
        return None
    x_min, y_min, x_max, y_max = box
    x0 = max(0, min(width - 1, int(math.floor(x_min))))
    y0 = max(0, min(height - 1, int(math.floor(y_min))))
    x1 = max(0, min(width, int(math.ceil(x_max)) + 1))
    y1 = max(0, min(height, int(math.ceil(y_max)) + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _temporal_activity(rgb_frames):
    stack = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
    if stack.shape[0] < 2:
        raise ValueError("At least two RGB frames are required")
    differences = np.abs(np.diff(stack, axis=0))
    return np.mean(differences, axis=(0, 3))


def _activity_statistics(activity, projected_bbox):
    height, width = activity.shape
    box = _clamped_box(projected_bbox, width, height)
    if box is None:
        return None, None, None, None
    x0, y0, x1, y1 = box
    roi = activity[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None, None, None

    padding = max(8, int(round(max(x1 - x0, y1 - y0) * 0.35)))
    outer_x0 = max(0, x0 - padding)
    outer_y0 = max(0, y0 - padding)
    outer_x1 = min(width, x1 + padding)
    outer_y1 = min(height, y1 + padding)
    outer = activity[outer_y0:outer_y1, outer_x0:outer_x1]
    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[
        y0 - outer_y0 : y1 - outer_y0,
        x0 - outer_x0 : x1 - outer_x0,
    ] = False
    ring = outer[ring_mask]
    roi_mean = float(np.mean(roi))
    roi_p95 = float(np.percentile(roi, 95.0))
    ring_mean = float(np.mean(ring)) if ring.size else 0.0
    return roi_mean, roi_p95, ring_mean, roi_mean - ring_mean


def _thumbnail(array, maximum_width):
    height, width = array.shape[:2]
    stride = max(1, int(math.ceil(width / int(maximum_width))))
    return np.ascontiguousarray(array[::stride, ::stride])


def _scaled_bbox(box, source_width, thumbnail_width):
    if box is None:
        return None
    scale = float(thumbnail_width) / float(source_width)
    return tuple(float(value) * scale for value in box)


def _view_diagnostic(
    name,
    frames,
    target_view,
    maximum_thumbnail_width,
):
    activity = _temporal_activity([frame.rgb_array() for frame in frames])
    roi_mean, roi_p95, ring_mean, excess = _activity_statistics(
        activity,
        target_view.projected_bbox,
    )
    representative = frames[-1].rgb_array()
    thumbnail = _thumbnail(representative, maximum_thumbnail_width)
    activity_thumbnail = _thumbnail(activity, maximum_thumbnail_width)
    return ViewDiagnostic(
        name=name,
        frame_id=int(frames[-1].frame_id),
        thumbnail=thumbnail,
        activity_thumbnail=activity_thumbnail,
        projected_bbox=_scaled_bbox(
            target_view.projected_bbox,
            frames[-1].width,
            thumbnail.shape[1],
        ),
        clear_samples=int(target_view.clear_in_frustum_samples),
        total_samples=int(target_view.in_frustum_samples),
        projected_span_pixels=float(target_view.projected_span_pixels),
        roi_mean=roi_mean,
        roi_p95=roi_p95,
        ring_mean=ring_mean,
        activity_excess=excess,
    )


def _fire_envelope(assessment):
    matches = [
        target
        for target in assessment.targets
        if target.role == VisibilityTargetRole.FIRE_ENVELOPE
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one fire envelope assessment, found {len(matches)}"
        )
    return matches[0]


def _clock_key(clock):
    return (
        int(clock.session_id),
        int(clock.step_index),
        int(clock.game_timer_ms),
        int(clock.actual_elapsed_ms),
    )


def _capture_pose(
    client,
    session,
    scenario_id,
    observation_spec,
    event_position,
    radius,
    height,
    azimuth,
    repeats,
    capture_timeout_ms,
    maximum_thumbnail_width,
    index,
):
    position = _camera_position(event_position, radius, height, azimuth)
    yaw = _yaw_toward(position, event_position)
    client.set_camera_pose(*position, yaw, collision_check=False)
    client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
    before = session.refresh()

    started = time.perf_counter()
    pairs = [
        session.capture_rgbd_pair(capture_timeout_ms)
        for _ in range(repeats)
    ]
    latency_ms = (time.perf_counter() - started) * 1000.0
    after = session.refresh()
    expected_clock = _clock_key(before)
    if _clock_key(after) != expected_clock or any(
        _clock_key(pair.clock) != expected_clock for pair in pairs
    ):
        raise RuntimeError("Fire visual probe advanced simulation time")

    frame_ids = [
        frame.frame_id
        for pair in pairs
        for frame in (pair.oblique, pair.nadir)
    ]
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise RuntimeError("RGB-D frame IDs are not strictly increasing")

    visibility = client.query_visibility(
        scenario_id,
        session.session_id,
        position,
        timeout=30.0,
    )
    if (
        visibility.step_index != before.step_index
        or visibility.game_timer_ms != before.game_timer_ms
    ):
        raise RuntimeError("Visibility and RGB-D use different simulation times")
    assessment = assess_visibility(
        visibility,
        pair_view_matrices(pairs[-1]),
        observation_spec,
    )
    envelope = _fire_envelope(assessment)
    return PoseDiagnostic(
        index=index,
        radius=float(radius),
        height=float(height),
        azimuth=float(azimuth) % 360.0,
        camera_position=tuple(float(value) for value in position),
        yaw=float(yaw),
        clock_step=int(before.step_index),
        game_timer_ms=int(before.game_timer_ms),
        latency_ms=float(latency_ms),
        oblique=_view_diagnostic(
            "oblique",
            [pair.oblique for pair in pairs],
            envelope.oblique,
            maximum_thumbnail_width,
        ),
        nadir=_view_diagnostic(
            "nadir",
            [pair.nadir for pair in pairs],
            envelope.nadir,
            maximum_thumbnail_width,
        ),
    )


def _metric_text(value):
    return "n/a" if value is None else f"{value:.5f}"


def _update_digest(digest, result):
    digest.update(
        np.asarray(
            (
                result.radius,
                result.height,
                result.azimuth,
                result.yaw,
                result.oblique.activity_excess
                if result.oblique.activity_excess is not None
                else math.nan,
                result.nadir.activity_excess
                if result.nadir.activity_excess is not None
                else math.nan,
            ),
            dtype="<f8",
        ).tobytes()
    )


def _show_results(results):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as error:
        raise RuntimeError("--show requires matplotlib") from error

    figure = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    rgb_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    activity_axes = (
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    summary_axis = figure.add_subplot(grid[:, 2])
    state = {"index": 0}

    def draw():
        result = results[state["index"]]
        for axis, view in zip(rgb_axes, (result.oblique, result.nadir)):
            axis.clear()
            axis.imshow(view.thumbnail)
            axis.set_title(f"{view.name} RGB frame={view.frame_id}")
            axis.set_axis_off()
            if view.projected_bbox is not None:
                x0, y0, x1, y1 = view.projected_bbox
                axis.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2.0,
                    )
                )

        vmax = max(
            float(np.percentile(result.oblique.activity_thumbnail, 99.0)),
            float(np.percentile(result.nadir.activity_thumbnail, 99.0)),
            1.0e-6,
        )
        for axis, view in zip(
            activity_axes,
            (result.oblique, result.nadir),
        ):
            axis.clear()
            axis.imshow(
                view.activity_thumbnail,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            axis.set_title(f"{view.name} temporal RGB activity")
            axis.set_axis_off()
            if view.projected_bbox is not None:
                x0, y0, x1, y1 = view.projected_bbox
                axis.add_patch(
                    Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=2.0,
                    )
                )

        summary_axis.clear()
        summary_axis.set_axis_off()
        lines = [
            f"Viewpoint {state['index'] + 1}/{len(results)}",
            "",
            f"radius={result.radius:.1f} m",
            f"height above event={result.height:.1f} m",
            f"azimuth={result.azimuth:.1f} deg",
            f"yaw toward event={result.yaw:.1f} deg",
            f"camera=({result.camera_position[0]:.2f}, "
            f"{result.camera_position[1]:.2f}, "
            f"{result.camera_position[2]:.2f})",
            f"lockstep={result.clock_step} timer={result.game_timer_ms} ms",
            f"capture latency={result.latency_ms:.1f} ms",
            "",
        ]
        for view in (result.oblique, result.nadir):
            lines.extend(
                (
                    view.name.upper(),
                    f"  envelope clear={view.clear_samples}/"
                    f"{view.total_samples}",
                    f"  projected span={view.projected_span_pixels:.1f} px",
                    f"  ROI mean={_metric_text(view.roi_mean)}",
                    f"  ROI p95={_metric_text(view.roi_p95)}",
                    f"  ring mean={_metric_text(view.ring_mean)}",
                    f"  ROI-ring={_metric_text(view.activity_excess)}",
                    "",
                )
            )
        lines.extend(
            (
                "Orange/cyan box: projected fire envelope",
                "Activity is a diagnostic, not visibility truth.",
                "",
                "Controls: Left/Right step | Home/End | Q close",
            )
        )
        summary_axis.text(
            0.0,
            1.0,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )
        figure.suptitle(
            "Fire visual coverage — frozen simulation, changing render frames"
        )
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key in ("q", "escape"):
            plt.close(figure)
            return
        if event.key == "left":
            state["index"] = max(0, state["index"] - 1)
        elif event.key == "right":
            state["index"] = min(len(results) - 1, state["index"] + 1)
        elif event.key == "home":
            state["index"] = 0
        elif event.key == "end":
            state["index"] = len(results) - 1
        else:
            return
        draw()

    figure.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    plt.close(figure)


def _summary(values):
    finite = [value for value in values if value is not None]
    if not finite:
        return "n/a"
    return (
        f"min={min(finite):.5f} "
        f"median={statistics.median(finite):.5f} "
        f"max={max(finite):.5f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit actual rendered fire coverage over a deterministic "
            "viewpoint grid without saving RGB-D."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=(20.0, 40.0, 60.0),
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=(20.0, 40.0, 60.0),
    )
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="+",
        default=(0.0, 90.0, 180.0, 270.0),
    )
    parser.add_argument("--thumbnail-width", type=int, default=480)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    args.radii = _finite_positive_values(parser, "--radii", args.radii)
    args.heights = _finite_positive_values(parser, "--heights", args.heights)
    args.azimuths = _finite_azimuths(parser, args.azimuths)
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if args.warmup_steps <= 0:
        parser.error("--warmup-steps must be positive")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.capture_timeout_ms <= 0:
        parser.error("--capture-timeout-ms must be positive")
    if args.thumbnail_width < 64:
        parser.error("--thumbnail-width must be at least 64")

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    scenario_id = None
    scenario_reset = False
    session = None
    failure = None
    results = []
    digest = hashlib.blake2b(digest_size=16)
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            seed=args.seed,
            firetruck_count=0,
            pedestrian_count=0,
        )
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        event_position = ready.event_position
        initial_position = _camera_position(
            event_position,
            args.radii[0],
            args.heights[0],
            args.azimuths[0],
        )
        client.set_camera_pose(
            *initial_position,
            _yaw_toward(initial_position, event_position),
            collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)

        session = LockstepSession(client)
        session.__enter__()
        start = client.start_scenario(scenario_id)
        if start.game_timer_ms != session.snapshot.epoch_game_timer_ms:
            raise RuntimeError("Scenario start and lockstep epoch differ")
        for _ in range(args.warmup_steps):
            session.advance()
        scenario = client.get_scenario_state(scenario_id)
        if not scenario.event_active:
            raise RuntimeError("Fire is not active after warmup")

        calibration = session.capture_rgbd_pair(args.capture_timeout_ms)
        observation_spec = ObservationSpec.from_pair(calibration)
        viewpoints = [
            (radius, height, azimuth)
            for height in args.heights
            for radius in args.radii
            for azimuth in args.azimuths
        ]
        print(
            f"fire visual coverage start scenario={scenario_id} "
            f"event=({event_position[0]:.2f}, {event_position[1]:.2f}, "
            f"{event_position[2]:.2f}) warmup={session.snapshot.actual_elapsed_ms}ms "
            f"viewpoints={len(viewpoints)} repeats={args.repeats}"
        )
        for index, (radius, height, azimuth) in enumerate(viewpoints):
            result = _capture_pose(
                client,
                session,
                scenario_id,
                observation_spec,
                event_position,
                radius,
                height,
                azimuth,
                args.repeats,
                args.capture_timeout_ms,
                args.thumbnail_width,
                index,
            )
            results.append(result)
            _update_digest(digest, result)
            print(
                f"{index + 1}/{len(viewpoints)} "
                f"r={radius:.1f}m h={height:.1f}m az={azimuth:.1f}deg "
                f"oblique[clear={result.oblique.clear_samples}/"
                f"{result.oblique.total_samples} span="
                f"{result.oblique.projected_span_pixels:.1f}px excess="
                f"{_metric_text(result.oblique.activity_excess)}] "
                f"nadir[clear={result.nadir.clear_samples}/"
                f"{result.nadir.total_samples} span="
                f"{result.nadir.projected_span_pixels:.1f}px excess="
                f"{_metric_text(result.nadir.activity_excess)}] "
                f"latency={result.latency_ms:.1f}ms"
            )

        client.reset_scenario(scenario_id)
        scenario_reset = True
        session.close()
        session = None
    except BaseException as error:
        failure = error

    cleanup_errors = []
    if not scenario_reset and scenario_id is not None:
        try:
            client.reset_scenario(scenario_id)
            scenario_reset = True
        except Exception as error:
            cleanup_errors.append(f"scenario Reset failed: {error}")
    if session is not None and scenario_reset:
        try:
            session.close()
            session = None
        except Exception as error:
            cleanup_errors.append(f"lockstep Exit failed: {error}")
    try:
        if client.is_camera_active():
            client.set_camera_pose(
                original_pose[0],
                original_pose[1],
                original_pose[2],
                original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        client.restore_player()
    except Exception as error:
        cleanup_errors.append(f"camera/player restore failed: {error}")
    if cleanup_errors:
        message = "; ".join(cleanup_errors)
        if failure is None:
            raise RuntimeError(message)
        if hasattr(failure, "add_note"):
            failure.add_note(message + "; press F11 in GTA")
    if failure is not None:
        raise failure

    print(
        "COMPLETE "
        f"viewpoints={len(results)} "
        "oblique_excess["
        f"{_summary([item.oblique.activity_excess for item in results])}] "
        "nadir_excess["
        f"{_summary([item.nadir.activity_excess for item in results])}] "
        f"digest={digest.hexdigest()}"
    )
    print(
        "Temporal activity is diagnostic only; inspect --show before "
        "changing fire effects or task semantics."
    )
    print("No RGB-D, image, video, or fire-coverage payload was written to disk.")
    if args.show:
        _show_results(results)


if __name__ == "__main__":
    main()
