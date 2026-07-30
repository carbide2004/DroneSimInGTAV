"""Validate GTA lockstep timing online without saving observation payloads."""

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
    DroneSimCommandError,
    ScenarioEntityRole,
    ScenarioLifecycle,
)


def _find_process(name):
    matches = [
        process
        for process in psutil.process_iter(("name", "pid"))
        if (process.info["name"] or "").lower() == name.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} process, found {len(matches)}"
        )
    return matches[0]


def _expect_command_error(call, expected_status):
    try:
        call()
    except DroneSimCommandError as error:
        if error.status_name != expected_status:
            raise RuntimeError(
                f"Expected {expected_status}, received {error.status_name}"
            ) from error
        return
    raise RuntimeError(f"Expected {expected_status}, command succeeded")


def _digest_frame(frame):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(memoryview(frame.rgb))
    digest.update(memoryview(frame.depth))
    depth = frame.depth_array()
    return (
        digest.hexdigest(),
        float(np.min(depth)),
        float(np.max(depth)),
    )


def _frozen_entity_maps(before, after):
    if before.game_timer_ms != after.game_timer_ms:
        raise RuntimeError(
            "Scenario game timer advanced while lockstep was frozen"
        )
    if before.event_position != after.event_position:
        raise RuntimeError(
            "Fire-source position changed while lockstep was frozen"
        )
    before_entities = {
        entity.stable_id: entity for entity in before.entities
    }
    after_entities = {
        entity.stable_id: entity for entity in after.entities
    }
    if before_entities.keys() != after_entities.keys():
        raise RuntimeError(
            "Scenario entity registry changed while lockstep was frozen"
        )
    return before_entities, after_entities


def _check_frozen_positions(
    before,
    after,
    position_tolerance=1.0e-3,
):
    before_entities, after_entities = _frozen_entity_maps(
        before, after
    )
    maximum_position_drift = 0.0
    for stable_id, first in before_entities.items():
        second = after_entities[stable_id]
        if first.exists != second.exists:
            raise RuntimeError(
                f"Entity {stable_id} existence changed while frozen"
            )
        position_drift = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(first.position, second.position)
            )
        )
        maximum_position_drift = max(
            maximum_position_drift, position_drift
        )
        if position_drift > position_tolerance:
            raise RuntimeError(
                f"Entity {stable_id} position changed by "
                f"{position_drift:.6f}m while frozen"
            )
    return maximum_position_drift


def _check_frozen_entities(
    before,
    after,
    position_tolerance=1.0e-3,
    velocity_tolerance=5.0e-2,
    heading_tolerance=5.0e-2,
):
    before_entities, after_entities = _frozen_entity_maps(
        before, after
    )
    maximum_position_drift = _check_frozen_positions(
        before,
        after,
        position_tolerance=position_tolerance,
    )
    maximum_velocity_drift = 0.0
    maximum_heading_drift = 0.0
    for stable_id, first in before_entities.items():
        second = after_entities[stable_id]
        if first.exists != second.exists:
            raise RuntimeError(
                f"Entity {stable_id} existence changed while frozen"
            )
        velocity_drift = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(first.velocity, second.velocity)
            )
        )
        heading_drift = abs(
            (second.heading - first.heading + 180.0)
            % 360.0
            - 180.0
        )
        maximum_velocity_drift = max(
            maximum_velocity_drift, velocity_drift
        )
        maximum_heading_drift = max(
            maximum_heading_drift, heading_drift
        )
        if heading_drift > heading_tolerance:
            raise RuntimeError(
                f"Entity {stable_id} heading changed by "
                f"{heading_drift:.6f}deg while frozen"
            )
        if velocity_drift > velocity_tolerance:
            raise RuntimeError(
                f"Entity {stable_id} velocity changed by "
                f"{velocity_drift:.6f}m/s while frozen; "
                f"numeric-drift limit is {velocity_tolerance:.3f}m/s"
            )
        if (
            abs(first.speed - second.speed)
            > velocity_tolerance
        ):
            raise RuntimeError(
                f"Entity {stable_id} speed changed by "
                f"{abs(first.speed - second.speed):.6f}m/s "
                "while frozen"
            )
    return (
        maximum_position_drift,
        maximum_velocity_drift,
        maximum_heading_drift,
    )


def _measure_frozen_drift(before, after):
    before_entities, after_entities = _frozen_entity_maps(
        before, after
    )
    maxima = {
        "position": (0.0, 0),
        "velocity": (0.0, 0),
        "heading": (0.0, 0),
    }
    for stable_id, first in before_entities.items():
        second = after_entities[stable_id]
        if first.exists != second.exists:
            raise RuntimeError(
                f"Entity {stable_id} existence changed while frozen"
            )
        position_drift = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(first.position, second.position)
            )
        )
        velocity_drift = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(first.velocity, second.velocity)
            )
        )
        heading_drift = abs(
            (second.heading - first.heading + 180.0)
            % 360.0
            - 180.0
        )
        for label, value in (
            ("position", position_drift),
            ("velocity", velocity_drift),
            ("heading", heading_drift),
        ):
            if value > maxima[label][0]:
                maxima[label] = (value, stable_id)
    return maxima


def _distance(left, right):
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right))
    )


def _trace_role_metrics(ready, samples, role, toward_event):
    initial_entities = {
        entity.stable_id: entity
        for entity in ready.entities
        if entity.role == role and entity.exists
    }
    if not initial_entities:
        return None

    paths = []
    progress = []
    sampled_speeds = []
    for stable_id, initial in initial_entities.items():
        previous_position = initial.position
        path_length = 0.0
        final_entity = initial
        for snapshot in samples:
            current = next(
                (
                    entity
                    for entity in snapshot.entities
                    if entity.stable_id == stable_id
                ),
                None,
            )
            if current is None or not current.exists:
                raise RuntimeError(
                    f"{role.name} entity {stable_id} disappeared "
                    "during dynamics continuity validation"
                )
            path_length += _distance(
                previous_position, current.position
            )
            previous_position = current.position
            final_entity = current
            sampled_speeds.append(current.speed)

        initial_distance = _distance(
            initial.position, ready.event_position
        )
        final_distance = _distance(
            final_entity.position, ready.event_position
        )
        signed_progress = (
            initial_distance - final_distance
            if toward_event
            else final_distance - initial_distance
        )
        paths.append(path_length)
        progress.append(signed_progress)

    return {
        "count": len(initial_entities),
        "path_p50": statistics.median(paths),
        "progress_p50": statistics.median(progress),
        "speed_p50": statistics.median(sampled_speeds),
        "speed_max": max(sampled_speeds),
    }


def _run_realtime_dynamics(
    client,
    args,
    blueprint_id,
):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=args.seed,
        firetruck_count=args.firetrucks,
        pedestrian_count=args.pedestrians,
        blueprint_id=blueprint_id,
    )
    reset_complete = False
    try:
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        if ready.blueprint_id != blueprint_id:
            raise RuntimeError(
                "Realtime dynamics run did not reuse the requested "
                "blueprint"
            )
        start = client.start_scenario(scenario_id)
        target_elapsed_ms = 250
        final_target_ms = args.dynamics_steps * 250
        samples = []
        deadline = (
            time.monotonic()
            + max(30.0, final_target_ms / 1000.0 * 4.0)
        )
        while target_elapsed_ms <= final_target_ms:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Realtime dynamics reference did not reach its "
                    "simulation-time target"
                )
            snapshot = client.get_scenario_state(scenario_id)
            if snapshot.lifecycle != ScenarioLifecycle.RUNNING:
                raise RuntimeError(
                    "Realtime dynamics reference left RUNNING: "
                    f"{snapshot.lifecycle.name} "
                    f"{snapshot.failure_message}"
                )
            elapsed = (
                snapshot.game_timer_ms - start.game_timer_ms
            ) & 0xFFFFFFFF
            while (
                elapsed >= target_elapsed_ms
                and target_elapsed_ms <= final_target_ms
            ):
                samples.append(snapshot)
                target_elapsed_ms += 250

        client.reset_scenario(scenario_id)
        reset_complete = True
        return ready, samples
    except BaseException as error:
        if not reset_complete:
            try:
                client.reset_scenario(scenario_id)
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "Realtime dynamics Reset failed: "
                        f"{cleanup_error}"
                    )
        raise


def _run_lockstep_dynamics(
    client,
    args,
    blueprint_id,
):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=args.seed,
        firetruck_count=args.firetrucks,
        pedestrian_count=args.pedestrians,
        blueprint_id=blueprint_id,
    )
    reset_complete = False
    session_id = None
    try:
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        if ready.blueprint_id != blueprint_id:
            raise RuntimeError(
                "Lockstep dynamics run did not reuse the requested "
                "blueprint"
            )
        entered = client.enter_lockstep()
        session_id = entered.session_id
        start = client.start_scenario(scenario_id)
        if start.game_timer_ms != entered.epoch_game_timer_ms:
            raise RuntimeError(
                "Lockstep dynamics Start timer does not match its epoch"
            )

        samples = []
        for expected_step in range(1, args.dynamics_steps + 1):
            clock = client.advance_lockstep(session_id)
            if clock.step_index != expected_step:
                raise RuntimeError(
                    "Lockstep dynamics returned an unexpected step index"
                )
            snapshot = client.get_scenario_state(scenario_id)
            if snapshot.lifecycle != ScenarioLifecycle.RUNNING:
                raise RuntimeError(
                    "Lockstep dynamics run left RUNNING: "
                    f"{snapshot.lifecycle.name} "
                    f"{snapshot.failure_message}"
                )
            samples.append(snapshot)
            time.sleep(args.dynamics_freeze_seconds)

        client.reset_scenario(scenario_id)
        reset_complete = True
        client.exit_lockstep(session_id)
        session_id = None
        return ready, samples
    except BaseException as error:
        cleanup_errors = []
        if not reset_complete:
            try:
                client.reset_scenario(scenario_id)
                reset_complete = True
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"dynamics Reset failed: {cleanup_error}"
                )
        if reset_complete and session_id is not None:
            try:
                client.exit_lockstep(session_id)
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"dynamics lockstep Exit failed: {cleanup_error}"
                )
        if cleanup_errors and hasattr(error, "add_note"):
            error.add_note(
                "; ".join(cleanup_errors)
                + "; press F11 in GTA"
            )
        raise


def _validate_dynamics_continuity(
    client,
    args,
    blueprint_id,
):
    realtime_ready, realtime_samples = _run_realtime_dynamics(
        client, args, blueprint_id
    )
    lockstep_ready, lockstep_samples = _run_lockstep_dynamics(
        client, args, blueprint_id
    )
    if (
        realtime_ready.event_position
        != lockstep_ready.event_position
    ):
        raise RuntimeError(
            "Matched dynamics runs resolved different event positions"
        )

    role_specs = (
        (ScenarioEntityRole.FIRE_TRUCK, True),
        (ScenarioEntityRole.FLEEING_PEDESTRIAN, False),
    )
    compared_roles = 0
    for role, toward_event in role_specs:
        realtime = _trace_role_metrics(
            realtime_ready,
            realtime_samples,
            role,
            toward_event,
        )
        lockstep = _trace_role_metrics(
            lockstep_ready,
            lockstep_samples,
            role,
            toward_event,
        )
        if realtime is None and lockstep is None:
            continue
        if realtime is None or lockstep is None:
            raise RuntimeError(
                f"Matched dynamics runs disagree on {role.name} count"
            )
        compared_roles += 1
        if realtime["path_p50"] < args.min_reference_motion:
            raise RuntimeError(
                f"Realtime {role.name} reference moved only "
                f"{realtime['path_p50']:.3f}m; the comparison is "
                "not informative"
            )
        path_ratio = (
            lockstep["path_p50"] / realtime["path_p50"]
        )
        if realtime["progress_p50"] >= args.min_reference_motion:
            progress_ratio = (
                lockstep["progress_p50"]
                / realtime["progress_p50"]
            )
        else:
            progress_ratio = float("nan")
        if realtime["speed_p50"] >= args.min_reference_speed:
            speed_ratio = (
                lockstep["speed_p50"]
                / realtime["speed_p50"]
            )
        else:
            speed_ratio = float("nan")
        print(
            f"dynamics {role.name} count={realtime['count']} "
            "realtime["
            f"path={realtime['path_p50']:.3f}m "
            f"progress={realtime['progress_p50']:.3f}m "
            f"speed_p50={realtime['speed_p50']:.3f}m/s "
            f"speed_max={realtime['speed_max']:.3f}m/s] "
            "lockstep["
            f"path={lockstep['path_p50']:.3f}m "
            f"progress={lockstep['progress_p50']:.3f}m "
            f"speed_p50={lockstep['speed_p50']:.3f}m/s "
            f"speed_max={lockstep['speed_max']:.3f}m/s] "
            f"path_ratio={path_ratio:.3f} "
            f"progress_ratio={progress_ratio:.3f} "
            f"speed_ratio={speed_ratio:.3f}"
        )
        if not (
            args.min_dynamics_ratio
            <= path_ratio
            <= args.max_dynamics_ratio
        ):
            raise RuntimeError(
                f"{role.name} lockstep/realtime path ratio "
                f"{path_ratio:.3f} is outside "
                f"[{args.min_dynamics_ratio:.3f}, "
                f"{args.max_dynamics_ratio:.3f}]"
            )
        if (
            math.isfinite(speed_ratio)
            and not (
                args.min_dynamics_ratio
                <= speed_ratio
                <= args.max_dynamics_ratio
            )
        ):
            raise RuntimeError(
                f"{role.name} lockstep/realtime median speed ratio "
                f"{speed_ratio:.3f} is outside "
                f"[{args.min_dynamics_ratio:.3f}, "
                f"{args.max_dynamics_ratio:.3f}]"
            )
        if (
            math.isfinite(progress_ratio)
            and not (
                args.min_dynamics_ratio
                <= progress_ratio
                <= args.max_dynamics_ratio
            )
        ):
            raise RuntimeError(
                f"{role.name} lockstep/realtime directional progress "
                f"ratio {progress_ratio:.3f} is outside "
                f"[{args.min_dynamics_ratio:.3f}, "
                f"{args.max_dynamics_ratio:.3f}]"
            )

    if compared_roles == 0:
        raise RuntimeError(
            "Dynamics continuity validation requires at least one "
            "firetruck or fleeing pedestrian"
        )
    print(
        "dynamics continuity PASS "
        f"blueprint={blueprint_id} "
        f"sim={args.dynamics_steps * 250}ms "
        f"freeze_per_step={args.dynamics_freeze_seconds:.3f}s"
    )


def _validate_clock_only(client, args, gta_process):
    initial_rss = gta_process.memory_info().rss
    entered = client.enter_lockstep()
    session_id = entered.session_id
    try:
        _expect_command_error(
            client.enter_lockstep,
            "LOCKSTEP_ALREADY_ACTIVE",
        )
        wrong_session = (
            session_id + 1
            if session_id < 0xFFFFFFFFFFFFFFFF
            else session_id - 1
        )
        _expect_command_error(
            lambda: client.get_lockstep_state(wrong_session),
            "LOCKSTEP_SESSION_MISMATCH",
        )

        first_frame = client.capture(args.capture_timeout_ms)
        first_digest, first_min, first_max = _digest_frame(first_frame)
        soak_started = time.monotonic()
        while time.monotonic() - soak_started < args.freeze_seconds:
            state = client.get_lockstep_state(session_id)
            if state.actual_elapsed_ms != 0:
                raise RuntimeError(
                    "Simulation time advanced during the frozen soak"
                )
            time.sleep(
                min(
                    args.freeze_poll_interval,
                    max(
                        0.0,
                        args.freeze_seconds
                        - (time.monotonic() - soak_started),
                    ),
                )
            )
        second_frame = client.capture(args.capture_timeout_ms)
        second_digest, second_min, second_max = _digest_frame(second_frame)
        if second_frame.frame_id <= first_frame.frame_id:
            raise RuntimeError(
                "Capture frame_id did not increase while frozen"
            )
        state = client.get_lockstep_state(session_id)
        if state.actual_elapsed_ms != 0:
            raise RuntimeError(
                "Simulation time advanced during frozen captures"
            )
        if not np.allclose(
            first_frame.view_matrix,
            second_frame.view_matrix,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RuntimeError(
                "Capture view matrix changed while the camera was fixed"
            )
        print(
            "frozen soak "
            f"seconds={args.freeze_seconds:.1f} "
            f"frames={first_frame.frame_id}->{second_frame.frame_id} "
            f"depth=[{min(first_min, second_min):.3f}, "
            f"{max(first_max, second_max):.3f}]m "
            f"rgbd_digest={first_digest}/{second_digest}"
        )

        latencies_ms = []
        frames_per_step = []
        maximum_frame_times_ms = []
        maximum_overshoot_ms = 0
        peak_rss = gta_process.memory_info().rss
        for index in range(args.steps):
            started = time.perf_counter()
            state = client.advance_lockstep(session_id)
            latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            expected_step = index + 1
            expected_target = expected_step * 250
            if state.step_index != expected_step:
                raise RuntimeError(
                    f"step_index={state.step_index}; "
                    f"expected {expected_step}"
                )
            if state.target_elapsed_ms != expected_target:
                raise RuntimeError(
                    f"target={state.target_elapsed_ms}; "
                    f"expected {expected_target}"
                )
            overshoot = (
                state.actual_elapsed_ms - state.target_elapsed_ms
            )
            maximum_overshoot_ms = max(
                maximum_overshoot_ms, overshoot
            )
            if overshoot > args.max_overshoot_ms:
                raise RuntimeError(
                    f"Step {expected_step} overshot by {overshoot}ms; "
                    f"limit is {args.max_overshoot_ms}ms"
                )
            frames_per_step.append(state.render_frames)
            maximum_frame_times_ms.append(state.max_frame_time_ms)
            peak_rss = max(
                peak_rss, gta_process.memory_info().rss
            )
            if (
                expected_step % args.progress_interval == 0
                or expected_step == args.steps
            ):
                print(
                    f"{expected_step}/{args.steps} "
                    f"sim={state.actual_elapsed_ms}ms "
                    f"overshoot={overshoot}ms "
                    f"frames={state.render_frames} "
                    f"max_frame={state.max_frame_time_ms:.2f}ms "
                    f"latency={latencies_ms[-1]:.1f}ms"
                )

        final = client.get_lockstep_state(session_id)
        final_overshoot = (
            final.actual_elapsed_ms - args.steps * 250
        )
        if final_overshoot > args.max_overshoot_ms:
            raise RuntimeError(
                f"Final cumulative overshoot is {final_overshoot}ms"
            )
        peak_growth_mb = (
            peak_rss - initial_rss
        ) / (1024**2)
        if peak_growth_mb > args.max_memory_growth_mb:
            raise RuntimeError(
                f"GTA peak memory grew by {peak_growth_mb:.1f}MiB "
                f"during clock steps; limit is "
                f"{args.max_memory_growth_mb:.1f}MiB"
            )
        print(
            "clock steps PASS "
            f"count={args.steps} "
            f"latency_p50={np.percentile(latencies_ms, 50):.1f}ms "
            f"latency_p95={np.percentile(latencies_ms, 95):.1f}ms "
            f"frames_p50={statistics.median(frames_per_step):.1f} "
            f"frames_range={min(frames_per_step)}.."
            f"{max(frames_per_step)} "
            f"max_frame={max(maximum_frame_times_ms):.2f}ms "
            f"max_overshoot={maximum_overshoot_ms}ms "
            f"peak_rss={peak_rss / (1024**2):.1f}MiB"
        )
    except BaseException as error:
        try:
            client.exit_lockstep(session_id)
        except Exception as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "Clock-only lockstep cleanup failed: "
                    f"{cleanup_error}; press F11 in GTA"
                )
        raise
    else:
        client.exit_lockstep(session_id)
        _expect_command_error(
            lambda: client.get_lockstep_state(session_id),
            "LOCKSTEP_NOT_ACTIVE",
        )
        return


def _validate_reentry_cycles(client, args):
    for index in range(args.cycles):
        session_id = None
        try:
            entered = client.enter_lockstep()
            session_id = entered.session_id
            advanced = client.advance_lockstep(session_id)
            if advanced.step_index != 1:
                raise RuntimeError(
                    f"Cycle {index + 1} did not complete one step"
                )
            client.exit_lockstep(session_id)
            session_id = None
        except BaseException as error:
            if session_id is not None:
                try:
                    client.exit_lockstep(session_id)
                except Exception as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            "Re-entry cleanup failed: "
                            f"{cleanup_error}; press F11 in GTA"
                        )
            raise
    print(f"re-entry cycles PASS count={args.cycles}")


def _validate_scenario(client, args):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=args.seed,
        firetruck_count=args.firetrucks,
        pedestrian_count=args.pedestrians,
    )
    ready = None
    session_id = None
    reset_complete = False
    try:
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        entered = client.enter_lockstep()
        session_id = entered.session_id
        start = client.start_scenario(scenario_id)
        if start.game_timer_ms != entered.epoch_game_timer_ms:
            raise RuntimeError(
                "Scenario Start timer does not match lockstep epoch"
            )

        def advance_and_snapshot(step_number):
            clock = client.advance_lockstep(session_id)
            snapshot = client.get_scenario_state(scenario_id)
            if snapshot.lifecycle != ScenarioLifecycle.RUNNING:
                raise RuntimeError(
                    "Scenario left RUNNING during lockstep: "
                    f"{snapshot.lifecycle.name} "
                    f"{snapshot.failure_message}"
                )
            if snapshot.game_timer_ms != clock.game_timer_ms:
                raise RuntimeError(
                    "Scenario and lockstep timers disagree"
                )
            if not snapshot.event_active:
                raise RuntimeError(
                    f"Fire became inactive at step {step_number}"
                )
            overshoot = (
                clock.actual_elapsed_ms - clock.target_elapsed_ms
            )
            if overshoot > args.max_overshoot_ms:
                raise RuntimeError(
                    f"Scenario step {step_number} overshot by "
                    f"{overshoot}ms"
                )
            return clock, snapshot, overshoot

        _clock, _first_pre_capture, first_overshoot = (
            advance_and_snapshot(1)
        )
        client.capture(args.capture_timeout_ms)
        first_snapshot = client.get_scenario_state(scenario_id)
        first_frozen_after = first_snapshot
        previous_wait = 0.0
        for cumulative_wait in (
            args.scenario_freeze_seconds,
            args.scenario_freeze_seconds * 2.0,
            args.scenario_freeze_seconds * 4.0,
        ):
            time.sleep(cumulative_wait - previous_wait)
            previous_wait = cumulative_wait
            first_frozen_after = client.get_scenario_state(
                scenario_id
            )
            measured = _measure_frozen_drift(
                first_snapshot, first_frozen_after
            )
            print(
                "frozen drift probe "
                f"wall={cumulative_wait:.3f}s "
                f"position={measured['position'][0]:.6f}m"
                f"@entity{measured['position'][1]} "
                f"velocity={measured['velocity'][0]:.6f}m/s"
                f"@entity{measured['velocity'][1]} "
                f"heading={measured['heading'][0]:.6f}deg"
                f"@entity{measured['heading'][1]}"
            )
        initial_drift = _check_frozen_entities(
            first_snapshot, first_frozen_after
        )
        print(
            "scenario t=250ms frozen baseline "
            f"position_drift={initial_drift[0]:.6f}m "
            f"velocity_drift={initial_drift[1]:.6f}m/s "
            f"heading_drift={initial_drift[2]:.6f}deg"
        )

        maximum_overshoot_ms = first_overshoot
        maximum_frozen_position_drift = initial_drift[0]
        maximum_frozen_velocity_drift = initial_drift[1]
        maximum_frozen_heading_drift = initial_drift[2]
        for index in range(1, args.scenario_steps):
            _clock, snapshot, overshoot = advance_and_snapshot(
                index + 1
            )
            maximum_overshoot_ms = max(
                maximum_overshoot_ms, overshoot
            )
            if (
                (index + 1) % args.scenario_frozen_check_interval
                == 0
            ):
                client.capture(args.capture_timeout_ms)
                frozen_before = client.get_scenario_state(
                    scenario_id
                )
                time.sleep(args.scenario_freeze_seconds)
                frozen_after = client.get_scenario_state(
                    scenario_id
                )
                frozen_drift = _check_frozen_entities(
                    frozen_before, frozen_after
                )
                maximum_frozen_position_drift = max(
                    maximum_frozen_position_drift,
                    frozen_drift[0],
                )
                maximum_frozen_velocity_drift = max(
                    maximum_frozen_velocity_drift,
                    frozen_drift[1],
                )
                maximum_frozen_heading_drift = max(
                    maximum_frozen_heading_drift,
                    frozen_drift[2],
                )

        client.reset_scenario(scenario_id)
        reset_complete = True
        client.exit_lockstep(session_id)
        print(
            "scenario lockstep PASS "
            f"scenario={scenario_id} "
            f"blueprint={ready.blueprint_id} "
            f"steps={args.scenario_steps} "
            f"sim={args.scenario_steps * 250}ms "
            f"max_overshoot={maximum_overshoot_ms}ms "
            f"frozen_position_drift="
            f"{maximum_frozen_position_drift:.6f}m "
            f"frozen_velocity_drift="
            f"{maximum_frozen_velocity_drift:.6f}m/s "
            f"frozen_heading_drift="
            f"{maximum_frozen_heading_drift:.6f}deg"
        )
        return ready.blueprint_id
    except BaseException as error:
        cleanup_errors = []
        if not reset_complete:
            try:
                client.reset_scenario(scenario_id)
                reset_complete = True
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"scenario Reset failed: {cleanup_error}"
                )
        if reset_complete and session_id is not None:
            try:
                client.exit_lockstep(session_id)
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"lockstep Exit failed: {cleanup_error}"
                )
        if cleanup_errors and hasattr(error, "add_note"):
            error.add_note(
                "; ".join(cleanup_errors)
                + "; press F11 in GTA"
            )
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed 250ms GTA lockstep advances, frozen RGB-D "
            "capture, re-entry, and a controlled fire scenario."
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--freeze-seconds", type=float, default=30.0)
    parser.add_argument(
        "--freeze-poll-interval", type=float, default=0.5
    )
    parser.add_argument("--max-overshoot-ms", type=int, default=50)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--scenario-steps", type=int, default=40)
    parser.add_argument(
        "--scenario-freeze-seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--scenario-frozen-check-interval", type=int, default=10
    )
    parser.add_argument(
        "--dynamics-steps",
        type=int,
        default=40,
        help=(
            "250ms samples for the matched realtime/lockstep "
            "dynamics comparison"
        ),
    )
    parser.add_argument(
        "--dynamics-freeze-seconds",
        type=float,
        default=0.25,
        help=(
            "wall-clock inference delay inserted after every matched "
            "lockstep sample"
        ),
    )
    parser.add_argument(
        "--min-reference-motion",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-reference-speed",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-dynamics-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-dynamics-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument(
        "--pedestrians",
        type=int,
        default=8,
        help=(
            "Lockstep integration actors; use validate_fire_scenario.py "
            "for the separate 32-pedestrian placement stress test"
        ),
    )
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--process-name", default="GTA5.exe")
    parser.add_argument(
        "--max-memory-growth-mb", type=float, default=512.0
    )
    parser.add_argument("--camera-height", type=float, default=40.0)
    parser.add_argument("--camera-pitch", type=float, default=-70.0)
    args = parser.parse_args()

    positive_values = {
        "--steps": args.steps,
        "--cycles": args.cycles,
        "--progress-interval": args.progress_interval,
        "--scenario-steps": args.scenario_steps,
        "--scenario-frozen-check-interval":
            args.scenario_frozen_check_interval,
        "--dynamics-steps": args.dynamics_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if (
        not math.isfinite(args.freeze_seconds)
        or args.freeze_seconds <= 0
        or not math.isfinite(args.freeze_poll_interval)
        or args.freeze_poll_interval <= 0
        or not math.isfinite(args.scenario_freeze_seconds)
        or args.scenario_freeze_seconds <= 0
        or not math.isfinite(args.dynamics_freeze_seconds)
        or args.dynamics_freeze_seconds <= 0
    ):
        parser.error("freeze durations and poll intervals must be positive")
    if (
        not math.isfinite(args.min_reference_motion)
        or args.min_reference_motion <= 0
        or not math.isfinite(args.min_reference_speed)
        or args.min_reference_speed <= 0
    ):
        parser.error(
            "reference motion and speed thresholds must be positive"
        )
    if (
        not math.isfinite(args.min_dynamics_ratio)
        or args.min_dynamics_ratio <= 0
        or not math.isfinite(args.max_dynamics_ratio)
        or args.max_dynamics_ratio < args.min_dynamics_ratio
    ):
        parser.error(
            "dynamics ratio bounds must be positive and ordered"
        )
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor must contain finite values")
    if args.max_overshoot_ms < 0:
        parser.error("--max-overshoot-ms must not be negative")
    if (
        not math.isfinite(args.max_memory_growth_mb)
        or args.max_memory_growth_mb < 0
    ):
        parser.error("--max-memory-growth-mb must be finite and non-negative")
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be in [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be in [0, 32]")
    if (
        not math.isfinite(args.camera_height)
        or args.camera_height <= 0
    ):
        parser.error("--camera-height must be positive")
    if not -90.0 <= args.camera_pitch <= 90.0:
        parser.error("--camera-pitch must be in [-90, 90]")

    gta_process = _find_process(args.process_name)
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    initial_rss = gta_process.memory_info().rss
    failure = None
    try:
        client.teleport_player(*args.anchor)
        client.set_camera_pose(
            args.anchor[0],
            args.anchor[1],
            args.anchor[2] + args.camera_height,
            original_pose[5],
            collision_check=False,
        )
        client.set_camera_pitch(args.camera_pitch)
        _validate_clock_only(client, args, gta_process)
        _validate_reentry_cycles(client, args)
        blueprint_id = _validate_scenario(client, args)
        _validate_dynamics_continuity(
            client,
            args,
            blueprint_id,
        )
    except BaseException as error:
        failure = error

    cleanup_error = None
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
        cleanup_error = error

    if cleanup_error is not None:
        if failure is None:
            raise cleanup_error
        if hasattr(failure, "add_note"):
            failure.add_note(
                "Final camera/player cleanup failed: "
                f"{cleanup_error}"
            )
    if failure is not None:
        raise failure

    growth_mb = (
        gta_process.memory_info().rss - initial_rss
    ) / (1024**2)
    if growth_mb > args.max_memory_growth_mb:
        raise RuntimeError(
            f"GTA memory grew by {growth_mb:.1f}MiB; "
            f"limit is {args.max_memory_growth_mb:.1f}MiB"
        )
    print(
        "PASS "
        f"clock_steps={args.steps} "
        f"cycles={args.cycles} "
        f"scenario_steps={args.scenario_steps} "
        f"dynamics_steps={args.dynamics_steps} "
        f"memory_growth={growth_mb:.1f}MiB"
    )
    print("No RGB-D or lockstep payload was written to disk.")


if __name__ == "__main__":
    main()
