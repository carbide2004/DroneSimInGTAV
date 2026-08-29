"""Run the planner-free Stage 3C explicit-belief policy in GTA."""

from __future__ import annotations

import argparse
import math
import statistics
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
from agent_control.expert_starts import (  # noqa: E402
    certify_scene_start_catalog_rgbd,
    generate_audited_task_start,
)
from agent_control.online_belief_policy_episode import (  # noqa: E402
    run_online_belief_policy_episode,
)
from agent_control.scene_catalog import (  # noqa: E402
    build_scene_start_catalog,
    scene_start_catalog_subset,
)
from agent_control.start_pool import (  # noqa: E402
    build_static_start_pool,
    load_pool,
)
from agent_control.task_starts import ObservationSpec, TASK_HORIZON_STEPS  # noqa: E402
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402
from learning.policy_dataset import ACTION_NAMES  # noqa: E402
from learning.dagger_dataset import DaggerShardRecorder  # noqa: E402
from validation.stage2e_trajectory_recording import Stage2EValidationRecorder  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 3C learned action policy. control mode never constructs "
            "the fixed planner or GTA geometry action mask."
        )
    )
    parser.add_argument("--anchor", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "learning/checkpoints/stage3c_explicit_belief_policy_bc.pt",
    )
    parser.add_argument("--mode", choices=("shadow", "control", "dagger"), default="control")
    parser.add_argument("--dagger-beta", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scenario-seed", type=int, default=11)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=TASK_HORIZON_STEPS)
    parser.add_argument("--prepare-timeout", type=float, default=30.0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--show-belief", action="store_true")
    parser.add_argument("--dagger-output-dir", type=Path)
    parser.add_argument("--start-pool", type=Path)
    parser.add_argument(
        "--pool-start-id",
        type=int,
        help=("Require one exact start-pool entry; requires --start-pool and "
              "--episodes 1"),
    )
    parser.add_argument("--anchor-name")
    args = parser.parse_args()
    if args.anchor is not None and not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if not 1 <= args.episodes <= 100:
        parser.error("--episodes must be in [1, 100]")
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    if not 0.0 <= args.dagger_beta <= 1.0:
        parser.error("--dagger-beta must be in [0, 1]")
    if args.mode != "dagger" and args.dagger_beta != 0.0:
        parser.error("--dagger-beta is only valid in dagger mode")
    if args.capture_timeout_ms <= 0 or args.prepare_timeout <= 0.0:
        parser.error("Timeouts must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in [1, 95]")
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.start_pool is not None:
        pool_path = args.start_pool.resolve()
        if not pool_path.is_file():
            parser.error(f"start pool does not exist: {pool_path}")
        args.start_pool = load_pool(pool_path)
    else:
        args.start_pool = None
    if args.pool_start_id is not None:
        if not 0 < args.pool_start_id < (1 << 64):
            parser.error("--pool-start-id must be an unsigned non-zero 64-bit integer")
        if args.start_pool is None:
            parser.error("--pool-start-id requires --start-pool")
        if args.episodes != 1:
            parser.error("--pool-start-id requires --episodes 1")
        available_ids = {
            int(entry.pool_start_id) for entry in args.start_pool.entries
        }
        if args.pool_start_id not in available_ids:
            parser.error(
                "--pool-start-id is absent from the supplied --start-pool: "
                f"{args.pool_start_id}"
            )
    if args.record_dir is not None:
        args.record_dir = args.record_dir.resolve()
        if args.record_dir.exists() or args.record_dir.with_name(
            args.record_dir.name + ".partial"
        ).exists():
            parser.error(f"record path already exists: {args.record_dir}")
        Stage2EValidationRecorder.require_dependencies()
    if args.show_belief and (args.record_dir is None or args.episodes != 1):
        parser.error("--show-belief requires one episode and --record-dir")
    if args.dagger_output_dir is not None:
        if args.mode != "dagger":
            parser.error("--dagger-output-dir requires --mode dagger")
        if not args.anchor_name:
            parser.error("--dagger-output-dir requires --anchor-name")
        args.dagger_output_dir = args.dagger_output_dir.resolve()
        if args.dagger_output_dir.exists():
            parser.error(f"DAgger output directory already exists: {args.dagger_output_dir}")
    elif args.mode == "dagger":
        print("WARNING dagger mode is not saving a training shard", flush=True)
    return args


def _record_path(root, episode_index, count):
    if root is None:
        return None
    return root if count == 1 else root / f"episode_{episode_index:03d}"


def _prepare(client, anchor, seed, timeout):
    scenario_id = client.prepare_fire_scenario(
        anchor, seed=seed, firetruck_count=1, pedestrian_count=32
    )
    try:
        return scenario_id, client.wait_scenario_ready(scenario_id, timeout=timeout)
    except BaseException:
        client.reset_scenario(scenario_id)
        raise


def _run_one(client, args, anchor, episode_index):
    scenario_id = None
    session = None
    recorder = None
    dagger_recorder = None
    started = time.perf_counter()
    try:
        client.teleport_player(*anchor)
        scenario_id, ready = _prepare(
            client, anchor, args.scenario_seed + episode_index, args.prepare_timeout
        )
        client.set_camera_pose(
            ready.event_position[0], ready.event_position[1] - 40.0,
            ready.event_position[2] + 40.0, client.get_pose()[5], collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
        session = LockstepSession(client)
        session.__enter__()
        observation_spec = ObservationSpec.from_pair(session.capture_rgbd_pair())
        if args.start_pool is None:
            start_pool = build_static_start_pool(
                client, session, ready, minimum_entries=1,
                observation_spec=observation_spec, horizon_steps=args.max_steps,
                progress_callback=lambda message: print(message, flush=True),
            )
        else:
            start_pool = args.start_pool
            if tuple(round(value, 3) for value in start_pool.anchor) != tuple(
                round(value, 3) for value in anchor
            ):
                raise RuntimeError("ANCHOR_POOL_MISMATCH: requested anchor changed")
            if tuple(round(value, 3) for value in start_pool.event_position) != tuple(
                round(value, 3) for value in ready.event_position
            ):
                raise RuntimeError("ANCHOR_POOL_MISMATCH: event position changed")
            print(
                f"start pool REUSE entries={len(start_pool.entries)} digest={start_pool.digest}",
                flush=True,
            )
        client.start_scenario(scenario_id)
        session.advance()
        calibration = session.capture_rgbd_pair(args.capture_timeout_ms)
        scenario = client.get_scenario_state(scenario_id)
        observation_spec = ObservationSpec.from_pair(calibration)
        catalog = build_scene_start_catalog(
            client, session, scenario, observation_spec, start_pool,
            minimum_entries=1, horizon_steps=args.max_steps,
        )
        if args.pool_start_id is not None:
            catalog = scene_start_catalog_subset(
                catalog, (args.pool_start_id,)
            )
            print(
                f"scene catalog EXACT pool_start={args.pool_start_id}",
                flush=True,
            )
        catalog = certify_scene_start_catalog_rgbd(
            client, session, scenario, observation_spec, start_pool, catalog,
            minimum_entries=1, horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        audited_start = generate_audited_task_start(
            client, session, scenario, observation_spec,
            args.start_seed + episode_index,
            start_pool=start_pool, scene_catalog=catalog,
            attempted_pool_start_ids=(), horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        path = _record_path(args.record_dir, episode_index, args.episodes)
        if path is not None:
            recorder = Stage2EValidationRecorder(path, args.jpeg_quality)
        if args.dagger_output_dir is not None:
            shard_name = (
                f"episode_{episode_index:03d}_scenario_"
                f"{args.scenario_seed + episode_index}_start_"
                f"{args.start_seed + episode_index}_pool_{audited_start.pool_start_id}"
            )
            dagger_recorder = DaggerShardRecorder(
                args.dagger_output_dir / shard_name,
                {
                    "episode_id": shard_name,
                    "anchor_name": args.anchor_name,
                    "anchor": [float(value) for value in anchor],
                    "scenario_seed": args.scenario_seed + episode_index,
                    "start_seed": args.start_seed + episode_index,
                    "pool_start_id": int(audited_start.pool_start_id),
                    "checkpoint": str(args.checkpoint),
                    "dagger_beta": float(args.dagger_beta),
                    "horizon_steps": int(args.max_steps),
                },
            )
        try:
            result = run_online_belief_policy_episode(
                client, session, scenario_id, audited_start, args.checkpoint,
                mode=args.mode, device=args.device, recorder=recorder,
                dagger_recorder=dagger_recorder,
                dagger_beta=args.dagger_beta,
                dagger_seed=args.start_seed + episode_index,
                capture_timeout_ms=args.capture_timeout_ms,
            )
        except BaseException as error:
            if recorder is not None:
                recorded = recorder.finish(
                    "ERROR", error=f"{type(error).__name__}: {error}"
                )
                print(f"ERROR trajectory recorded at {recorded}", flush=True)
                recorder = None
            if dagger_recorder is not None:
                dagger_recorder.abort()
                dagger_recorder = None
            raise
        if dagger_recorder is not None:
            shard = dagger_recorder.finish(result.status, result=result)
            print(f"dagger_shard={shard} depth_files=0 rgb_files=0", flush=True)
            dagger_recorder = None
        if recorder is not None:
            recorded = recorder.finish(
                "PASS" if result.success else "FAILED", result=result
            )
            print(
                f"trajectory={recorded} size={recorder.size_bytes() / 1048576.0:.1f}MiB "
                "depth_files=0", flush=True,
            )
            recorder = None
        print(
            f"EPISODE {episode_index + 1}/{args.episodes} mode={args.mode} "
            f"status={result.status} actions={result.actions} updates={result.belief_updates} "
            f"first_update={result.first_update_step} first_source={result.first_source_step} "
            f"agreement={result.policy_expert_agreement} labels={result.expert_labels} "
            f"no_labels={result.no_expert_labels} last_nll={result.last_source_blind_event_nll} "
            f"last_map_error={result.last_source_blind_map_error_m} "
            f"stop_error={result.localization_error_m} "
            f"action_entropy={result.mean_policy_action_entropy} "
            f"coverage={result.inference_credible_coverage} "
            f"policy_counts={dict(zip(ACTION_NAMES, result.policy_action_counts, strict=True))} "
            f"timing[ground={result.timing.grounding_seconds:.2f}s "
            f"model={result.timing.model_seconds:.2f}s expert={result.timing.expert_seconds:.2f}s "
            f"action={result.timing.action_seconds:.2f}s] wall={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return result
    finally:
        if dagger_recorder is not None:
            dagger_recorder.abort()
        if scenario_id is not None:
            client.reset_scenario(scenario_id)
        if session is not None:
            session.close()


def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, _model, _geometry = load_policy_checkpoint(args.checkpoint, device)
    print(
        f"checkpoint PASS model={checkpoint['model']} epoch={checkpoint.get('epoch')} "
        f"dagger_iteration={checkpoint.get('dagger_iteration', 0)} device={device} "
        f"validation_anchors={checkpoint.get('validation_anchors')}", flush=True,
    )
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    anchor = tuple(args.anchor) if args.anchor is not None else tuple(original_pose[:3])
    results = []
    try:
        if args.dagger_output_dir is not None:
            args.dagger_output_dir.mkdir(parents=True)
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        for episode_index in range(args.episodes):
            results.append(_run_one(client, args, anchor, episode_index))
    finally:
        try:
            client.set_camera_pose(
                original_pose[0], original_pose[1], original_pose[2],
                original_pose[5], collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
    successes = sum(result.success for result in results)
    actions = [result.actions for result in results]
    errors = [
        result.localization_error_m for result in results
        if result.localization_error_m is not None
    ]
    policy_counts = tuple(
        sum(result.policy_action_counts[index] for result in results)
        for index in range(len(ACTION_NAMES))
    )
    executed_counts = tuple(
        sum(result.executed_action_counts[index] for result in results)
        for index in range(len(ACTION_NAMES))
    )
    entropy_values = [
        result.mean_policy_action_entropy for result in results
        if result.mean_policy_action_entropy is not None
    ]
    coverage_values = [
        result.inference_credible_coverage for result in results
        if result.inference_credible_coverage is not None
    ]
    mean_coverage = None if not coverage_values else tuple(
        statistics.fmean(value[index] for value in coverage_values)
        for index in range(3)
    )
    statuses = {}
    for result in results:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    print(
        f"DONE mode={args.mode} successes={successes}/{len(results)} "
        f"rate={successes / len(results):.3f} actions_p50={statistics.median(actions):.1f} "
        f"localization_mean={math.nan if not errors else statistics.fmean(errors):.3f}m "
        f"statuses={statuses}", flush=True,
    )
    print(
        "AUDIT "
        f"policy_action_entropy={math.nan if not entropy_values else statistics.fmean(entropy_values):.4f} "
        f"credible_coverage_50_80_90={mean_coverage} "
        f"policy_counts={dict(zip(ACTION_NAMES, policy_counts, strict=True))} "
        f"executed_counts={dict(zip(ACTION_NAMES, executed_counts, strict=True))} "
        f"belief_nll_mean={statistics.fmean(value for value in (result.inference_event_nll for result in results) if value is not None) if any(result.inference_event_nll is not None for result in results) else math.nan:.4f} "
        f"map_error_mean={statistics.fmean(value for value in (result.last_source_blind_map_error_m for result in results) if value is not None) if any(result.last_source_blind_map_error_m is not None for result in results) else math.nan:.3f}m",
        flush=True,
    )
    if args.show_belief:
        from validation.visualize_online_belief_policy import show_recording
        show_recording(args.record_dir, interval_ms=250, loop=False, start_paused=True)
    if not successes and args.mode == "control":
        raise RuntimeError("No planner-free Stage 3C control episode passed")


if __name__ == "__main__":
    main()
