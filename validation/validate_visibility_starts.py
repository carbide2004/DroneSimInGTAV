"""Validate Stage 2C visibility truth and deterministic task starts in GTA."""

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import psutil

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
    StartVisibilityStratum,
    TASK_MAX_ALTITUDE_AGL_METERS,
    TASK_MAX_EVENT_DISTANCE_METERS,
    TASK_MIN_ALTITUDE_AGL_METERS,
    TASK_MIN_EVENT_DISTANCE_METERS,
    assess_visibility,
    generate_task_start,
    pair_view_matrices,
)


def _find_process(name):
    matches = [
        process
        for process in psutil.process_iter(("name", "pid"))
        if (process.info["name"] or "").lower() == name.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} process, "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_strata(value):
    if value == "both":
        return (
            StartVisibilityStratum.CUE_VISIBLE,
            StartVisibilityStratum.CUE_HIDDEN,
        )
    if value == "cue-visible":
        return (StartVisibilityStratum.CUE_VISIBLE,)
    if value == "cue-hidden":
        return (StartVisibilityStratum.CUE_HIDDEN,)
    raise ValueError(value)


def _assert_start(generated, scenario):
    blueprint = generated.blueprint
    assessment = generated.assessment
    if not assessment.event_initially_hidden:
        raise RuntimeError(
            "Generated start exposes the fire-source vehicle"
        )
    expected_cue = (
        blueprint.visibility_stratum
        == StartVisibilityStratum.CUE_VISIBLE
    )
    if assessment.cue_task_observable != expected_cue:
        raise RuntimeError(
            "Generated start does not match its cue visibility stratum"
        )
    if not (
        TASK_MIN_EVENT_DISTANCE_METERS
        <= blueprint.event_distance
        <= TASK_MAX_EVENT_DISTANCE_METERS
    ):
        raise RuntimeError(
            "Generated start event distance is outside the task band"
        )
    if not (
        TASK_MIN_ALTITUDE_AGL_METERS
        <= blueprint.altitude_agl
        <= TASK_MAX_ALTITUDE_AGL_METERS
    ):
        raise RuntimeError(
            "Generated start altitude AGL is outside the task band"
        )
    event_local = blueprint.world_to_local(
        scenario.event_position
    )
    roundtrip = blueprint.local_to_world(event_local)
    error = float(
        np.linalg.norm(
            np.asarray(roundtrip, dtype=np.float64)
            - np.asarray(
                scenario.event_position,
                dtype=np.float64,
            )
        )
    )
    if error > 1.0e-6:
        raise RuntimeError(
            "Start-local/world coordinate roundtrip error is "
            f"{error:.9f}m"
        )
    if (
        generated.visibility.step_index
        != generated.rgbd_pair.clock.step_index
        or generated.visibility.game_timer_ms
        != generated.rgbd_pair.clock.game_timer_ms
    ):
        raise RuntimeError(
            "Visibility and RGB-D do not belong to one lockstep instant"
        )


def _visibility_digest(digest, visibility):
    digest.update(visibility.scenario_id.to_bytes(8, "little"))
    digest.update(
        visibility.lockstep_session_id.to_bytes(8, "little")
    )
    digest.update(visibility.step_index.to_bytes(8, "little"))
    for target in visibility.targets:
        digest.update(target.stable_id.to_bytes(8, "little"))
        digest.update(int(target.role).to_bytes(4, "little"))
        for sample in target.samples:
            digest.update(
                b"\x01" if sample.clear_line_of_sight else b"\x00"
            )


def _show_start_views(generated):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as error:
        raise RuntimeError(
            "--show-starts requires matplotlib"
        ) from error

    response_roles = {
        VisibilityTargetRole.FIRE_TRUCK,
        VisibilityTargetRole.FLEEING_PEDESTRIAN,
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        constrained_layout=True,
    )
    views = (
        (
            "oblique",
            generated.rgbd_pair.oblique,
        ),
        (
            "nadir",
            generated.rgbd_pair.nadir,
        ),
    )
    for axis, (view_name, frame) in zip(axes, views):
        axis.imshow(frame.rgb_array())
        axis.set_title(
            f"{generated.blueprint.visibility_stratum.name} "
            f"{view_name} frame={frame.frame_id}"
        )
        axis.set_xlim(0, frame.width - 1)
        axis.set_ylim(frame.height - 1, 0)
        axis.set_axis_off()
        for target in generated.assessment.targets:
            view = getattr(target, view_name)
            if view.projected_bbox is None:
                continue
            x_min, y_min, x_max, y_max = view.projected_bbox
            is_response = target.role in response_roles
            if is_response and view.task_observable:
                color = "lime"
                linewidth = 2.5
            elif target.role == VisibilityTargetRole.FIRE_SOURCE_VEHICLE:
                color = "red"
                linewidth = 1.5
            elif target.role == VisibilityTargetRole.FIRE_ENVELOPE:
                color = "orange"
                linewidth = 1.5
            else:
                color = "yellow"
                linewidth = 1.0
            axis.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                )
            )
            center_x = (x_min + x_max) * 0.5
            center_y = (y_min + y_max) * 0.5
            axis.plot(
                center_x,
                center_y,
                marker="+",
                color=color,
                markersize=8,
                markeredgewidth=linewidth,
            )
            role_label = (
                "SMOKE_ENVELOPE"
                if target.role == VisibilityTargetRole.FIRE_ENVELOPE
                else target.role.name
            )
            label = (
                f"{role_label} id={target.stable_id} "
                f"clear={view.clear_in_frustum_samples}/"
                f"{view.in_frustum_samples} "
                f"span={view.projected_span_pixels:.1f}px "
                "margin="
                f"{'yes' if view.inside_image_margin else 'no'}"
            )
            if is_response and view.task_observable:
                label += " CUE_VISIBLE"
            x_offset = -12 if center_x > frame.width * 0.65 else 12
            y_offset = -22 if center_y < frame.height * 0.2 else 18
            axis.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="right" if x_offset < 0 else "left",
                va="top" if y_offset < 0 else "bottom",
                color="black",
                fontsize=7,
                bbox={
                    "facecolor": color,
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": linewidth,
                },
            )
    figure.suptitle(
        f"start={generated.blueprint.start_id} "
        f"event bearing="
        f"{generated.blueprint.event_bearing_body_degrees:.1f} deg | "
        "green=response target that satisfies CUE_VISIBLE"
    )
    plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate virtual visibility queries and Stage 2C task "
            "starts without saving RGB-D."
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
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=16)
    parser.add_argument(
        "--strata",
        choices=("both", "cue-visible", "cue-hidden"),
        default="both",
    )
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--camera-height", type=float, default=40.0)
    parser.add_argument("--process-name", default="GTA5.exe")
    parser.add_argument(
        "--max-memory-growth-mb",
        type=float,
        default=512.0,
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
    )
    parser.add_argument(
        "--show-starts",
        action="store_true",
        help=(
            "Show in-memory oblique/nadir RGB with projected "
            "visibility boxes; close each window to continue"
        ),
    )
    args = parser.parse_args()

    if args.queries <= 0:
        parser.error("--queries must be positive")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")

    gta_process = _find_process(args.process_name)
    python_process = psutil.Process()
    gta_initial_rss = gta_process.memory_info().rss
    python_initial_rss = python_process.memory_info().rss
    gta_peak_rss = gta_initial_rss
    python_peak_rss = python_initial_rss

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    scenario_id = None
    scenario_reset = False
    session = None
    failure = None
    latencies = []
    digest = hashlib.blake2b(digest_size=16)
    generated_starts = []
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            seed=args.seed,
            firetruck_count=args.firetrucks,
            pedestrian_count=args.pedestrians,
        )
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        overview = (
            ready.event_position[0],
            ready.event_position[1] - args.camera_height,
            ready.event_position[2] + args.camera_height,
        )
        client.set_camera_pose(
            *overview,
            original_pose[5],
            collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)

        session = LockstepSession(client)
        session.__enter__()
        start = client.start_scenario(scenario_id)
        if (
            start.game_timer_ms
            != session.snapshot.epoch_game_timer_ms
        ):
            raise RuntimeError(
                "Scenario start does not match the lockstep epoch"
            )
        session.advance()
        scenario = client.get_scenario_state(scenario_id)
        calibration_pair = session.capture_rgbd_pair()
        observation_spec = ObservationSpec.from_pair(
            calibration_pair
        )

        for stratum_index, stratum in enumerate(
            _parse_strata(args.strata)
        ):
            start_seed = args.start_seed + stratum_index
            generated = generate_task_start(
                client,
                session,
                scenario,
                observation_spec,
                stratum,
                start_seed,
                max_candidates=args.max_candidates,
            )
            _assert_start(generated, scenario)
            generated_starts.append(generated)
            _visibility_digest(digest, generated.visibility)
            print(
                f"{stratum.name} PASS "
                f"start={generated.blueprint.start_id} "
                f"candidate={generated.blueprint.candidate_index} "
                f"distance={generated.blueprint.event_distance:.2f}m "
                f"agl={generated.blueprint.altitude_agl:.2f}m "
                "bearing="
                f"{generated.blueprint.event_bearing_body_degrees:.1f}deg "
                f"frames={generated.rgbd_pair.oblique.frame_id}/"
                f"{generated.rgbd_pair.nadir.frame_id} "
                "envelope_los="
                f"{generated.assessment.fire_envelope_clear_fraction:.3f} "
                f"rejections={dict(generated.rejection_counts)}"
            )
            if args.show_starts:
                _show_start_views(generated)

            if args.verify_determinism:
                repeated = generate_task_start(
                    client,
                    session,
                    scenario,
                    observation_spec,
                    stratum,
                    start_seed,
                    max_candidates=args.max_candidates,
                )
                if (
                    repeated.blueprint.start_id
                    != generated.blueprint.start_id
                    or repeated.blueprint.candidate_index
                    != generated.blueprint.candidate_index
                    or repeated.blueprint.absolute_pose
                    != generated.blueprint.absolute_pose
                ):
                    raise RuntimeError(
                        f"{stratum.name} task-start generation "
                        "is not deterministic"
                    )

        probe_start = generated_starts[-1]
        center = probe_start.blueprint.absolute_pose[:3]
        for index in range(args.queries):
            before = session.refresh()
            started = time.perf_counter()
            visibility = client.query_visibility(
                scenario_id,
                session.session_id,
                center,
                timeout=30.0,
            )
            latency = (time.perf_counter() - started) * 1000.0
            after = session.refresh()
            if (
                before.step_index != after.step_index
                or before.game_timer_ms != after.game_timer_ms
                or visibility.step_index != before.step_index
                or visibility.game_timer_ms
                != before.game_timer_ms
            ):
                raise RuntimeError(
                    "Visibility query advanced simulation time"
                )
            assessment = assess_visibility(
                visibility,
                pair_view_matrices(
                    probe_start.rgbd_pair
                ),
                observation_spec,
            )
            if (
                assessment.event_initially_hidden
                != probe_start.assessment.event_initially_hidden
                or assessment.source_vehicle_has_line_of_sight
                != probe_start.assessment.source_vehicle_has_line_of_sight
                or assessment.cue_task_observable
                != probe_start.assessment.cue_task_observable
            ):
                raise RuntimeError(
                    "Repeated visibility query changed at a frozen "
                    "simulation instant"
                )
            latencies.append(latency)
            _visibility_digest(digest, visibility)
            gta_peak_rss = max(
                gta_peak_rss,
                gta_process.memory_info().rss,
            )
            python_peak_rss = max(
                python_peak_rss,
                python_process.memory_info().rss,
            )
            if (index + 1) % 10 == 0 or index + 1 == args.queries:
                print(
                    f"{index + 1}/{args.queries} visibility "
                    f"latency={latency:.1f}ms"
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
            cleanup_errors.append(
                f"scenario Reset failed: {error}"
            )
    if session is not None and scenario_reset:
        try:
            session.close()
            session = None
        except Exception as error:
            cleanup_errors.append(
                f"lockstep Exit failed: {error}"
            )
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
        cleanup_errors.append(
            f"camera/player restore failed: {error}"
        )
    if cleanup_errors:
        message = "; ".join(cleanup_errors)
        if failure is None:
            raise RuntimeError(message)
        if hasattr(failure, "add_note"):
            failure.add_note(message + "; press F11 in GTA")
    if failure is not None:
        raise failure

    gta_growth_mb = (
        gta_process.memory_info().rss - gta_initial_rss
    ) / (1024**2)
    python_growth_mb = (
        python_process.memory_info().rss - python_initial_rss
    ) / (1024**2)
    gta_peak_growth_mb = (
        gta_peak_rss - gta_initial_rss
    ) / (1024**2)
    python_peak_growth_mb = (
        python_peak_rss - python_initial_rss
    ) / (1024**2)
    for label, value in (
        ("GTA final", gta_growth_mb),
        ("GTA peak", gta_peak_growth_mb),
        ("Python final", python_growth_mb),
        ("Python peak", python_peak_growth_mb),
    ):
        if value > args.max_memory_growth_mb:
            raise RuntimeError(
                f"{label} memory grew by {value:.1f}MiB; "
                f"limit={args.max_memory_growth_mb:.1f}MiB"
            )

    print(
        "PASS "
        f"starts={len(generated_starts)} "
        f"queries={args.queries} "
        f"latency_p50={statistics.median(latencies):.1f}ms "
        f"latency_p95={np.percentile(latencies, 95):.1f}ms "
        f"gta_growth={gta_growth_mb:.1f}MiB "
        f"python_growth={python_growth_mb:.1f}MiB "
        f"digest={digest.hexdigest()}"
    )
    print("No RGB-D, visibility, or task-start payload was written to disk.")


if __name__ == "__main__":
    main()
