"""Run the trained Spatial RNN belief updater in an online GTA episode."""

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
from agent_control.online_spatial_episode import (  # noqa: E402
    run_online_spatial_belief_episode,
)
from agent_control.scene_catalog import build_scene_start_catalog  # noqa: E402
from agent_control.start_pool import build_static_start_pool  # noqa: E402
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    TASK_HORIZON_STEPS,
)
from learning.online_spatial_belief import resolve_torch_device  # noqa: E402
from learning.spatial_belief_runtime import load_spatial_checkpoint  # noqa: E402
from validation.stage2e_trajectory_recording import (  # noqa: E402
    Stage2EValidationRecorder,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run source-blind Spatial RNN belief inference in GTA. RGB replay "
            "recording is disabled unless --record-dir is supplied."
        )
    )
    parser.add_argument("--anchor", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "learning/checkpoints/stage3_spatial_rnn.pt",
    )
    parser.add_argument("--mode", choices=("shadow", "control"), default="control")
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
    args = parser.parse_args()
    if args.anchor is not None and not all(math.isfinite(v) for v in args.anchor):
        parser.error("--anchor values must be finite")
    if not 1 <= args.episodes <= 100:
        parser.error("--episodes must be in [1, 100]")
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    if not 0 <= args.scenario_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--scenario-seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    if args.capture_timeout_ms <= 0:
        parser.error("--capture-timeout-ms must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in [1, 95]")
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.record_dir is not None:
        args.record_dir = args.record_dir.resolve()
        if args.record_dir.exists() or args.record_dir.with_name(
            args.record_dir.name + ".partial"
        ).exists():
            parser.error(f"record path already exists: {args.record_dir}")
        Stage2EValidationRecorder.require_dependencies()
    if args.show_belief and args.record_dir is None:
        parser.error("--show-belief requires --record-dir")
    if args.show_belief and args.episodes != 1:
        parser.error("--show-belief currently requires --episodes 1")
    return args


def _record_path(root, episode_index, count):
    if root is None:
        return None
    return root if count == 1 else root / f"episode_{episode_index:03d}"


def _prepare(client, anchor, seed, timeout):
    scenario_id = client.prepare_fire_scenario(
        anchor,
        seed=seed,
        firetruck_count=1,
        pedestrian_count=32,
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
    episode_started = time.perf_counter()
    seed = args.scenario_seed + episode_index
    start_seed = args.start_seed + episode_index
    try:
        client.teleport_player(*anchor)
        scenario_id, ready = _prepare(client, anchor, seed, args.prepare_timeout)
        client.set_camera_pose(
            ready.event_position[0],
            ready.event_position[1] - 40.0,
            ready.event_position[2] + 40.0,
            client.get_pose()[5],
            collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
        session = LockstepSession(client)
        session.__enter__()
        observation_spec = ObservationSpec.from_pair(session.capture_rgbd_pair())
        start_pool = build_static_start_pool(
            client,
            session,
            ready,
            minimum_entries=1,
            observation_spec=observation_spec,
            horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        client.start_scenario(scenario_id)
        session.advance()
        calibration = session.capture_rgbd_pair(args.capture_timeout_ms)
        scenario = client.get_scenario_state(scenario_id)
        observation_spec = ObservationSpec.from_pair(calibration)
        catalog = build_scene_start_catalog(
            client,
            session,
            scenario,
            observation_spec,
            start_pool,
            minimum_entries=1,
            horizon_steps=args.max_steps,
        )
        catalog = certify_scene_start_catalog_rgbd(
            client,
            session,
            scenario,
            observation_spec,
            start_pool,
            catalog,
            minimum_entries=1,
            horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        audited_start = generate_audited_task_start(
            client,
            session,
            scenario,
            observation_spec,
            start_seed,
            start_pool=start_pool,
            scene_catalog=catalog,
            attempted_pool_start_ids=(),
            horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        path = _record_path(args.record_dir, episode_index, args.episodes)
        if path is not None:
            recorder = Stage2EValidationRecorder(path, args.jpeg_quality)
        try:
            result = run_online_spatial_belief_episode(
                client,
                session,
                scenario_id,
                audited_start,
                args.checkpoint,
                mode=args.mode,
                device=args.device,
                recorder=recorder,
                capture_timeout_ms=args.capture_timeout_ms,
            )
        except BaseException as error:
            if recorder is not None:
                recorded = recorder.finish(
                    "ERROR", error=f"{type(error).__name__}: {error}"
                )
                print(f"ERROR trajectory recorded at {recorded}", flush=True)
                recorder = None
            raise
        if recorder is not None:
            recorded = recorder.finish(
                "PASS" if result.success else "FAILED", result=result
            )
            print(
                f"trajectory={recorded} size={recorder.size_bytes() / 1048576.0:.1f}MiB "
                "depth_files=0",
                flush=True,
            )
            recorder = None
        print(
            f"EPISODE {episode_index + 1}/{args.episodes} status={result.status} "
            f"actions={result.actions} updates={result.belief_updates} "
            f"first_update={result.first_update_step} first_source={result.first_source_step} "
            f"inference_nll={result.inference_event_nll} "
            f"last_nll={result.last_source_blind_event_nll} "
            f"last_map_error={result.last_source_blind_map_error_m} "
            f"stop_error={result.localization_error_m} "
            f"timing[ground={result.timing.grounding_seconds:.2f}s "
            f"model={result.timing.model_seconds:.2f}s "
            f"planner={result.timing.planner_seconds:.2f}s "
            f"action={result.timing.action_seconds:.2f}s] "
            f"wall={time.perf_counter() - episode_started:.1f}s",
            flush=True,
        )
        return result
    finally:
        if scenario_id is not None:
            client.reset_scenario(scenario_id)
        if session is not None:
            session.close()


def main():
    args = _parse_args()
    device = resolve_torch_device(args.device)
    checkpoint, _model = load_spatial_checkpoint(args.checkpoint, device)
    print(
        f"checkpoint PASS model={checkpoint['model']} epoch={checkpoint['epoch']} "
        f"device={device} validation_anchors={checkpoint['validation_anchors']}",
        flush=True,
    )
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    anchor = (
        tuple(float(value) for value in args.anchor)
        if args.anchor is not None
        else tuple(float(value) for value in original_pose[:3])
    )
    results = []
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        for episode_index in range(args.episodes):
            results.append(_run_one(client, args, anchor, episode_index))
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
    successes = sum(result.success for result in results)
    action_counts = [result.actions for result in results]
    source_steps = [
        result.first_source_step
        for result in results
        if result.first_source_step is not None
    ]
    map_errors = [
        result.last_source_blind_map_error_m
        for result in results
        if result.last_source_blind_map_error_m is not None
    ]
    nll_values = [
        result.last_source_blind_event_nll
        for result in results
        if result.last_source_blind_event_nll is not None
    ]
    print(
        f"DONE mode={args.mode} successes={successes}/{len(results)} "
        f"rate={successes / len(results):.3f} "
        f"actions_p50={statistics.median(action_counts):.1f} "
        f"source_step_p50={math.nan if not source_steps else statistics.median(source_steps)} "
        f"last_nll_mean={math.nan if not nll_values else statistics.fmean(nll_values):.3f} "
        f"last_map_error_mean={math.nan if not map_errors else statistics.fmean(map_errors):.3f}m",
        flush=True,
    )
    if args.show_belief:
        from validation.visualize_online_spatial_belief import show_recording

        show_recording(
            args.record_dir, interval_ms=250, loop=False, start_paused=True
        )
    if not successes:
        raise RuntimeError("No online Spatial RNN episode passed")


if __name__ == "__main__":
    main()
