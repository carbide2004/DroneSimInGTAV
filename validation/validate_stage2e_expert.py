import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    LockstepSession,
    OBLIQUE_PITCH_DEGREES,
    ScenarioEntityRole,
    ScenarioTaskState,
)
from agent_control.expert_episode import run_expert_episode  # noqa: E402
from agent_control.expert_starts import (  # noqa: E402
    generate_audited_task_start,
)
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    TASK_HORIZON_STEPS,
)
from validation.stage2e_trajectory_recording import (  # noqa: E402
    Stage2EValidationRecorder,
)


PEDESTRIAN_BANDS = (
    (8.0, 20.0),
    (20.0, 35.0),
    (35.0, 50.0),
    (50.0, 65.0),
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate Stage 2E response timing, audited starts, and "
            "one cue-grounded expert episode. Payload recording is "
            "disabled unless --record-dir is supplied."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Requested event anchor. If omitted, use the current scripted "
            "camera position and snap horizontally to a road within 30 m."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--prepare-timeout", type=float, default=30.0)
    parser.add_argument(
        "--start-audit-timeout",
        type=float,
        default=120.0,
        help="Wall-clock limit for one lightweight start goal audit",
    )
    parser.add_argument("--start-attempts", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TASK_HORIZON_STEPS,
        help=(
            "Maximum Agent actions including STOP; canonical default is "
            f"{TASK_HORIZON_STEPS}"
        ),
    )
    parser.add_argument(
        "--response-steps",
        type=int,
        default=48,
        help="250ms steps used to audit the full 0..12s response wave",
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument(
        "--record-dir",
        type=Path,
        help=(
            "Write a compact replayable trajectory to this new directory. "
            "Stores JPEG RGB, structured state, belief, and compact truth; "
            "never stores Depth."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()
    if args.anchor is not None and not all(
        math.isfinite(value) for value in args.anchor
    ):
        parser.error("--anchor values must be finite")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    if not 1 <= args.response_steps <= 64:
        parser.error("--response-steps must be in [1, 64]")
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in [1, 95]")
    if args.record_dir is not None:
        args.record_dir = args.record_dir.resolve()
        partial = args.record_dir.with_name(
            args.record_dir.name + ".partial"
        )
        if args.record_dir.exists() or partial.exists():
            parser.error(
                "--record-dir and its .partial path must not already "
                f"exist: {args.record_dir}"
            )
        Stage2EValidationRecorder.require_dependencies()
    return args


def _peds(snapshot):
    return tuple(
        entity
        for entity in snapshot.entities
        if entity.role == ScenarioEntityRole.FLEEING_PEDESTRIAN
    )


def _band_index(distance):
    for index, (minimum, maximum) in enumerate(
        PEDESTRIAN_BANDS
    ):
        if minimum - 1.0e-3 <= distance <= maximum + 1.0e-3:
            return index
    raise RuntimeError(
        f"Pedestrian distance {distance:.3f}m is outside all bands"
    )


def _blueprint_signature(snapshot):
    return tuple(
        (
            entity.role,
            entity.model_hash,
            tuple(round(value, 4) for value in entity.position),
            entity.planned_activation_offset_ms,
        )
        for entity in snapshot.entities
    )


def _validate_ready(snapshot):
    pedestrians = _peds(snapshot)
    if len(pedestrians) != 32:
        raise RuntimeError(
            f"Expected 32 pedestrians, found {len(pedestrians)}"
        )
    counts = Counter()
    for entity in pedestrians:
        if entity.task_state != ScenarioTaskState.PENDING:
            raise RuntimeError(
                f"Pedestrian {entity.stable_id} is not PENDING in READY"
            )
        distance = math.dist(
            entity.position[:2],
            snapshot.event_position[:2],
        )
        counts[_band_index(distance)] += 1
        if (
            entity.planned_activation_offset_ms % 250 != 0
            or not 0
            <= entity.planned_activation_offset_ms
            <= 12000
        ):
            raise RuntimeError(
                f"Pedestrian {entity.stable_id} has invalid activation "
                f"offset {entity.planned_activation_offset_ms}ms"
            )
        expected = max(0.0, (distance - 20.0) / 10.0) * 1000.0
        if abs(entity.planned_activation_offset_ms - expected) > 625.0:
            raise RuntimeError(
                f"Pedestrian {entity.stable_id} activation offset does "
                "not match the distance-wave model"
            )
    if counts != Counter({0: 8, 1: 8, 2: 8, 3: 8}):
        raise RuntimeError(
            f"Pedestrian distance-band counts are invalid: {dict(counts)}"
        )
    return _blueprint_signature(snapshot)


def _audit_response_wave(client, session, scenario_id, steps):
    previous = client.get_scenario_state(scenario_id)
    previous_positions = {
        entity.stable_id: entity.position
        for entity in _peds(previous)
    }
    activation_counts = []
    for step in range(1, steps + 1):
        clock = session.advance()
        current = client.get_scenario_state(scenario_id)
        elapsed = clock.actual_elapsed_ms
        pending = 0
        activated = 0
        for entity in _peds(current):
            if entity.planned_activation_offset_ms <= elapsed:
                if entity.activation_game_timer_ms == 0:
                    raise RuntimeError(
                        f"Pedestrian {entity.stable_id} missed activation "
                        f"at elapsed={elapsed}ms"
                    )
                activated += 1
            else:
                pending += 1
                if entity.task_state != ScenarioTaskState.PENDING:
                    raise RuntimeError(
                        f"Pedestrian {entity.stable_id} activated early"
                    )
                if math.dist(
                    entity.position,
                    previous_positions[entity.stable_id],
                ) > 1.0e-3:
                    raise RuntimeError(
                        f"Pending pedestrian {entity.stable_id} moved"
                    )
            previous_positions[entity.stable_id] = entity.position
        activation_counts.append(activated)
        if step == 1 or step % 8 == 0 or step == steps:
            print(
                f"response step={step} elapsed={elapsed}ms "
                f"activated={activated} pending={pending}",
                flush=True,
            )
        previous = current
    if activation_counts != sorted(activation_counts):
        raise RuntimeError("Activated pedestrian count is not monotonic")


def _prepare_ready(client, args, blueprint_id=0):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=args.seed,
        firetruck_count=1,
        pedestrian_count=32,
        blueprint_id=blueprint_id,
    )
    ready = client.wait_scenario_ready(
        scenario_id,
        timeout=args.prepare_timeout,
    )
    return scenario_id, ready


def _reset_then_close(client, scenario_id, session):
    client.reset_scenario(scenario_id)
    session.close()


def main():
    args = _parse_args()
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    if args.anchor is None:
        args.anchor = tuple(float(value) for value in original_pose[:3])
        print(
            "using current camera position as requested anchor "
            f"({args.anchor[0]:.2f}, {args.anchor[1]:.2f}, "
            f"{args.anchor[2]:.2f}); scenario will snap horizontally "
            "to the nearest road node",
            flush=True,
        )
    started = time.perf_counter()
    scenario_id = None
    session = None
    recorder = None
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)

        scenario_id, ready = _prepare_ready(client, args)
        signature = _validate_ready(ready)
        session = LockstepSession(client)
        session.__enter__()
        client.start_scenario(scenario_id)
        _audit_response_wave(
            client,
            session,
            scenario_id,
            args.response_steps,
        )
        blueprint_id = ready.blueprint_id
        _reset_then_close(client, scenario_id, session)
        scenario_id = None
        session = None
        print("response field PASS", flush=True)

        scenario_id, repeated_ready = _prepare_ready(
            client,
            args,
            blueprint_id=blueprint_id,
        )
        repeated_signature = _validate_ready(repeated_ready)
        if signature != repeated_signature:
            raise RuntimeError(
                "Reused blueprint changed layout/model/activation plan"
            )
        client.set_camera_pose(
            repeated_ready.event_position[0],
            repeated_ready.event_position[1] - 40.0,
            repeated_ready.event_position[2] + 40.0,
            original_pose[5],
            collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
        session = LockstepSession(client)
        session.__enter__()
        client.start_scenario(scenario_id)
        session.advance()
        calibration = session.capture_rgbd_pair()
        scenario = client.get_scenario_state(scenario_id)
        audited_start = generate_audited_task_start(
            client,
            session,
            scenario,
            ObservationSpec.from_pair(calibration),
            args.start_seed,
            maximum_attempts=args.start_attempts,
            audit_timeout_seconds=args.start_audit_timeout,
            horizon_steps=args.max_steps,
            progress_callback=lambda message: print(
                message,
                flush=True,
            ),
        )
        if args.record_dir is not None:
            recorder = Stage2EValidationRecorder(
                args.record_dir,
                jpeg_quality=args.jpeg_quality,
            )
        try:
            result = run_expert_episode(
                client,
                session,
                scenario_id,
                audited_start,
                recorder=recorder,
            )
        except BaseException as error:
            if recorder is not None:
                path = recorder.finish(
                    "ERROR",
                    error=(
                        f"{type(error).__name__}: {error}"
                    ),
                )
                print(
                    f"Stage 2E partial trajectory recorded at {path}",
                    flush=True,
                )
                recorder = None
            raise
        if not result.success:
            if recorder is not None:
                path = recorder.finish("FAILED", result=result)
                print(
                    f"Stage 2E failed trajectory recorded at {path}",
                    flush=True,
                )
                recorder = None
            raise RuntimeError(
                f"Expert episode failed: {result.message}; "
                f"actions={result.actions}, "
                f"plans={result.planner_calls}, "
                f"valid_cue={result.valid_dynamic_cue_observed}, "
                f"sensitivity={result.cue_sensitivity}"
            )
        recording_path = None
        recording_size = None
        if recorder is not None:
            recording_path = recorder.finish("PASS", result=result)
            recording_size = recorder.size_bytes()
            recorder = None
        _reset_then_close(client, scenario_id, session)
        scenario_id = None
        session = None
        print(
            "PASS "
            "goal_lower_bound="
            f"{audited_start.goal_budget_audit.lower_bound_total_actions} "
            "goal_budget_remaining="
            f"{audited_start.goal_budget_audit.remaining_actions} "
            "initial_grounded_responses="
            f"{audited_start.initial_grounded_response_count} "
            f"actions={result.actions} "
            f"plans={result.planner_calls} "
            f"error={result.localization_error_m:.3f}m "
            "sensitivity="
            f"{result.cue_sensitivity.divergence_kind} "
            f"wall={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        start_timing = audited_start.timing
        rollout_timing = result.timing
        print(
            "TIMING_DETAIL start[attempts="
            f"{start_timing.attempts} generate="
            f"{start_timing.task_start_generation_seconds:.1f}s "
            f"ground={start_timing.rgbd_grounding_seconds:.1f}s "
            f"budget_audit="
            f"{start_timing.static_goal_budget_audit_seconds:.1f}s "
            f"total={start_timing.total_seconds:.1f}s] "
            "rollout[steps="
            f"{rollout_timing.observed_steps} visibility="
            f"{rollout_timing.visibility_seconds:.1f}s scenario="
            f"{rollout_timing.scenario_snapshot_seconds:.1f}s ground="
            f"{rollout_timing.grounding_seconds:.1f}s teacher="
            f"{rollout_timing.teacher_seconds:.1f}s record="
            f"{rollout_timing.recording_seconds:.1f}s pose="
            f"{rollout_timing.action_pose_seconds:.1f}s advance="
            f"{rollout_timing.action_advance_seconds:.1f}s capture="
            f"{rollout_timing.action_capture_seconds:.1f}s sensitivity="
            f"{rollout_timing.cue_sensitivity_seconds:.1f}s geometry_query="
            f"{rollout_timing.geometry_query_seconds:.1f}s "
            "geometry_segments="
            f"{rollout_timing.geometry_queried_segments}/"
            f"{rollout_timing.geometry_requested_segments} cache_hits="
            f"{rollout_timing.geometry_cache_hits} batches="
            f"{rollout_timing.geometry_batch_queries} total="
            f"{rollout_timing.total_seconds:.1f}s]",
            flush=True,
        )
        if recording_path is None:
            print(
                "No RGB-D, trajectory, or belief payload was written "
                "to disk.",
                flush=True,
            )
        else:
            print(
                f"Stage 2E trajectory path={recording_path} "
                f"size={recording_size / (1024 * 1024):.1f}MiB; "
                "no Depth payload was written.",
                flush=True,
            )
    finally:
        if scenario_id is not None:
            try:
                client.reset_scenario(scenario_id)
                scenario_id = None
            except Exception:
                print(
                    "Scenario Reset failed; lockstep remains frozen. "
                    "Use F11 to recover.",
                    flush=True,
                )
                raise
        if session is not None:
            session.close()
        try:
            client.set_camera_pose(
                original_pose[0],
                original_pose[1],
                original_pose[2],
                original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()


if __name__ == "__main__":
    main()
