import argparse
import math
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    LockstepSession,
    OBLIQUE_PITCH_DEGREES,
)
from agent_control.expert_episode import run_expert_episode  # noqa: E402
from agent_control.expert_recording import (  # noqa: E402
    ExpertEpisodeRecorder,
    append_attempt_timing,
    append_failure,
)
from agent_control.expert_starts import (  # noqa: E402
    generate_audited_task_start,
)
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    TASK_HORIZON_STEPS,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate strict Stage 2E cue-grounded expert episodes. "
            "Only successful episodes retain RGB-D payloads."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=32)
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
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--max-success-episodes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be in [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be in [0, 32]")
    if args.firetrucks + args.pedestrians == 0:
        parser.error("At least one response actor is required")
    if not 1 <= args.start_attempts <= 256:
        parser.error("--start-attempts must be in [1, 256]")
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.max_success_episodes <= 0:
        parser.error("--max-success-episodes must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in [1, 95]")
    return args


def _progress(message):
    print(message, flush=True)


def _timed(callable_, *args, **kwargs):
    started = time.perf_counter()
    result = callable_(*args, **kwargs)
    return result, time.perf_counter() - started


def _format_timing(timing):
    ordered = (
        "prepare",
        "lockstep_setup",
        "start_audit",
        "rollout",
        "write_finalize",
        "cleanup",
        "total",
    )
    return " ".join(
        f"{name}={timing.get(name, 0.0):.1f}s"
        for name in ordered
    )


def _format_detail(audited_start, result):
    fields = []
    if audited_start is not None:
        timing = audited_start.timing
        fields.append(
            "start[attempts="
            f"{timing.attempts} generate="
            f"{timing.task_start_generation_seconds:.1f}s ground="
            f"{timing.rgbd_grounding_seconds:.1f}s budget_audit="
            f"{timing.static_goal_budget_audit_seconds:.1f}s]"
        )
    if result is not None:
        timing = result.timing
        fields.append(
            "rollout[steps="
            f"{timing.observed_steps} visibility="
            f"{timing.visibility_seconds:.1f}s scenario="
            f"{timing.scenario_snapshot_seconds:.1f}s ground="
            f"{timing.grounding_seconds:.1f}s teacher="
            f"{timing.teacher_seconds:.1f}s record="
            f"{timing.recording_seconds:.1f}s pose="
            f"{timing.action_pose_seconds:.1f}s advance="
            f"{timing.action_advance_seconds:.1f}s capture="
            f"{timing.action_capture_seconds:.1f}s sensitivity="
            f"{timing.cue_sensitivity_seconds:.1f}s geometry_query="
            f"{timing.geometry_query_seconds:.1f}s geometry_segments="
            f"{timing.geometry_queried_segments}/"
            f"{timing.geometry_requested_segments} cache_hits="
            f"{timing.geometry_cache_hits} batches="
            f"{timing.geometry_batch_queries}]"
        )
    return " ".join(fields)


def _cleanup_frozen(client, session, scenario_id):
    if scenario_id is not None:
        client.reset_scenario(scenario_id)
        scenario_id = None
    if session is not None:
        session.close()
        session = None
    return session, scenario_id


def main():
    args = _parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    successes = 0
    attempts_run = 0
    started = time.perf_counter()
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)
        for attempt in range(args.max_attempts):
            if successes >= args.max_success_episodes:
                break
            attempts_run += 1
            scenario_id = None
            session = None
            recorder = None
            attempt_started = time.perf_counter()
            attempt_timing = {}
            audited_start = None
            failed_start_timing = None
            result = None
            outcome = "ERROR"
            phase = "prepare"
            phase_started = attempt_started
            attempt_seed = (
                args.seed + attempt
            ) & 0xFFFFFFFFFFFFFFFF
            start_seed = (
                args.start_seed + attempt
            ) & 0xFFFFFFFFFFFFFFFF
            try:
                print(
                    f"attempt {attempt + 1}/{args.max_attempts} "
                    f"scenario_seed={attempt_seed} "
                    f"start_seed={start_seed}",
                    flush=True,
                )
                prepare_started = time.perf_counter()
                scenario_id = client.prepare_fire_scenario(
                    args.anchor,
                    seed=attempt_seed,
                    firetruck_count=args.firetrucks,
                    pedestrian_count=args.pedestrians,
                )
                ready = client.wait_scenario_ready(
                    scenario_id,
                    timeout=args.prepare_timeout,
                )
                attempt_timing["prepare"] = (
                    time.perf_counter() - prepare_started
                )
                phase = "lockstep_setup"
                phase_started = time.perf_counter()
                client.set_camera_pose(
                    ready.event_position[0],
                    ready.event_position[1] - 40.0,
                    ready.event_position[2] + 40.0,
                    original_pose[5],
                    collision_check=False,
                )
                client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
                session = LockstepSession(client)
                session.__enter__()
                client.start_scenario(scenario_id)
                session.advance()
                calibration = session.capture_rgbd_pair()
                observation_spec = ObservationSpec.from_pair(
                    calibration
                )
                scenario = client.get_scenario_state(scenario_id)
                attempt_timing["lockstep_setup"] = (
                    time.perf_counter() - phase_started
                )
                phase = "start_audit"
                phase_started = time.perf_counter()
                audited_start, attempt_timing["start_audit"] = _timed(
                    generate_audited_task_start,
                    client,
                    session,
                    scenario,
                    observation_spec,
                    start_seed,
                    maximum_attempts=args.start_attempts,
                    audit_timeout_seconds=args.start_audit_timeout,
                    horizon_steps=args.max_steps,
                    progress_callback=_progress,
                )
                phase = "rollout"
                phase_started = time.perf_counter()
                episode_name = (
                    f"episode_{successes:06d}_"
                    f"scenario_{attempt_seed}_"
                    f"start_{audited_start.generated.blueprint.start_id}"
                )
                recorder = ExpertEpisodeRecorder(
                    output_root,
                    episode_name,
                    jpeg_quality=args.jpeg_quality,
                )
                result, attempt_timing["rollout"] = _timed(
                    run_expert_episode,
                    client,
                    session,
                    scenario_id,
                    audited_start,
                    recorder=recorder,
                )
                if not result.success:
                    phase = "write_finalize"
                    phase_started = time.perf_counter()
                    recorder.abort()
                    recorder = None
                    append_failure(
                        output_root,
                        {
                            "attempt": attempt,
                            "scenario_seed": attempt_seed,
                            "start_seed": start_seed,
                            "result": result,
                        },
                    )
                    attempt_timing["write_finalize"] = (
                        time.perf_counter() - phase_started
                    )
                    outcome = "FAILED"
                    print(
                        f"FAIL actions={result.actions} "
                        f"plans={result.planner_calls} "
                        f"message={result.message}",
                        flush=True,
                    )
                else:
                    phase = "write_finalize"
                    phase_started = time.perf_counter()
                    path = recorder.finish(
                        {
                            "result": result,
                            "scenario_seed": attempt_seed,
                            "start_seed": start_seed,
                            "goal_budget_audit": (
                                audited_start.goal_budget_audit
                            ),
                            "bearing_bin": audited_start.bearing_bin,
                            "altitude_bin": audited_start.altitude_bin,
                            "initial_grounded_response_count": (
                                audited_start.initial_grounded_response_count
                            ),
                            "timing": {
                                "audited_start": audited_start.timing,
                                "rollout": result.timing,
                            },
                        }
                    )
                    attempt_timing["write_finalize"] = (
                        time.perf_counter() - phase_started
                    )
                    recorder = None
                    successes += 1
                    outcome = "PASS"
                    print(
                        f"PASS {successes}/"
                        f"{args.max_success_episodes} "
                        f"actions={result.actions} "
                        "initial_grounded_responses="
                        f"{audited_start.initial_grounded_response_count} "
                        f"plans={result.planner_calls} "
                        f"error={result.localization_error_m:.3f}m "
                        "sensitivity="
                        f"{result.cue_sensitivity.divergence_kind} "
                        f"path={path}",
                        flush=True,
                    )
                phase_started = time.perf_counter()
                phase = "cleanup"
                session, scenario_id = _cleanup_frozen(
                    client,
                    session,
                    scenario_id,
                )
                phase = "complete"
                attempt_timing["cleanup"] = (
                    time.perf_counter() - phase_started
                )
            except Exception as error:
                failed_start_timing = getattr(error, "timing", None)
                if phase != "complete" and phase not in attempt_timing:
                    attempt_timing[phase] = (
                        time.perf_counter() - phase_started
                    )
                cleanup_started = time.perf_counter()
                if recorder is not None:
                    recorder.abort()
                append_failure(
                    output_root,
                    {
                        "attempt": attempt,
                        "scenario_seed": attempt_seed,
                        "start_seed": start_seed,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                )
                print(
                    f"FAIL {type(error).__name__}: {error}",
                    flush=True,
                )
                if scenario_id is not None:
                    try:
                        client.reset_scenario(scenario_id)
                        scenario_id = None
                    except Exception:
                        print(
                            "Scenario Reset failed; lockstep remains "
                            "frozen. Use F11 to recover.",
                            flush=True,
                        )
                        raise
                if session is not None:
                    session.close()
                attempt_timing["cleanup"] = (
                    time.perf_counter() - cleanup_started
                )
            finally:
                attempt_timing["total"] = (
                    time.perf_counter() - attempt_started
                )
                timing_record = {
                    "attempt": attempt,
                    "scenario_seed": attempt_seed,
                    "start_seed": start_seed,
                    "outcome": outcome,
                    "timing": attempt_timing,
                }
                if audited_start is not None:
                    timing_record["audited_start"] = audited_start.timing
                elif failed_start_timing is not None:
                    timing_record["audited_start"] = failed_start_timing
                if result is not None:
                    timing_record["rollout"] = result.timing
                append_attempt_timing(output_root, timing_record)
                print(
                    f"TIMING outcome={outcome} "
                    f"{_format_timing(attempt_timing)}",
                    flush=True,
                )
                detail = _format_detail(audited_start, result)
                if audited_start is None and failed_start_timing is not None:
                    detail = (
                        "start[attempts="
                        f"{failed_start_timing.attempts} generate="
                        f"{failed_start_timing.task_start_generation_seconds:.1f}s "
                        "ground="
                        f"{failed_start_timing.rgbd_grounding_seconds:.1f}s "
                        "budget_audit="
                        f"{failed_start_timing.static_goal_budget_audit_seconds:.1f}s]"
                    )
                if detail:
                    print(f"TIMING_DETAIL {detail}", flush=True)
        elapsed = time.perf_counter() - started
        rate = successes / attempts_run
        print(
            f"DONE successes={successes} "
            f"attempts_run={attempts_run} "
            f"attempt_budget={args.max_attempts} "
            f"rate={rate:.3f} wall={elapsed:.1f}s",
            flush=True,
        )
        if successes < args.max_success_episodes:
            raise RuntimeError(
                "The requested successful episode count was not reached"
            )
    finally:
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
