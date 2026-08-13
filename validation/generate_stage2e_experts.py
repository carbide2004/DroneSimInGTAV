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
    append_failure,
)
from agent_control.expert_starts import (  # noqa: E402
    generate_certified_task_start,
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
    parser.add_argument("--search-timeout", type=float, default=120.0)
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
    started = time.perf_counter()
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)
        for attempt in range(args.max_attempts):
            if successes >= args.max_success_episodes:
                break
            scenario_id = None
            session = None
            recorder = None
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
                certified = generate_certified_task_start(
                    client,
                    session,
                    scenario,
                    observation_spec,
                    start_seed,
                    maximum_attempts=args.start_attempts,
                    search_timeout_seconds=args.search_timeout,
                    horizon_steps=args.max_steps,
                    progress_callback=_progress,
                )
                episode_name = (
                    f"episode_{successes:06d}_"
                    f"scenario_{attempt_seed}_"
                    f"start_{certified.generated.blueprint.start_id}"
                )
                recorder = ExpertEpisodeRecorder(
                    output_root,
                    episode_name,
                    jpeg_quality=args.jpeg_quality,
                )
                result = run_expert_episode(
                    client,
                    session,
                    scenario_id,
                    certified,
                    recorder=recorder,
                )
                if not result.success:
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
                    print(
                        f"FAIL actions={result.actions} "
                        f"plans={result.planner_calls} "
                        f"message={result.message}",
                        flush=True,
                    )
                else:
                    path = recorder.finish(
                        {
                            "result": result,
                            "scenario_seed": attempt_seed,
                            "start_seed": start_seed,
                            "path_cost_bin": certified.path_cost_bin,
                            "bearing_bin": certified.bearing_bin,
                            "altitude_bin": certified.altitude_bin,
                        }
                    )
                    recorder = None
                    successes += 1
                    print(
                        f"PASS {successes}/"
                        f"{args.max_success_episodes} "
                        f"actions={result.actions} "
                        f"plans={result.planner_calls} "
                        f"error={result.localization_error_m:.3f}m "
                        "sensitivity="
                        f"{result.cue_sensitivity.divergence_kind} "
                        f"path={path}",
                        flush=True,
                    )
                session, scenario_id = _cleanup_frozen(
                    client,
                    session,
                    scenario_id,
                )
            except Exception as error:
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
        elapsed = time.perf_counter() - started
        rate = successes / args.max_attempts
        print(
            f"DONE successes={successes} "
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
