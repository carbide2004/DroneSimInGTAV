import argparse
import json
import math
import os
import shutil
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


COLLECTION_SCHEMA_VERSION = 1
UINT64_MASK = 0xFFFFFFFFFFFFFFFF


class CollectionInvariantError(RuntimeError):
    pass


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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Attempt budget for legacy flat generation",
    )
    parser.add_argument("--max-success-episodes", type=int)
    parser.add_argument(
        "--scenario-count",
        type=int,
        help=(
            "Enable grouped collection with this many seed-distinct "
            "scenario blueprints at one anchor"
        ),
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        help="Successful episode quota for each grouped scenario",
    )
    parser.add_argument(
        "--max-attempts-per-scenario",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a grouped collection after validating its manifest, "
            "completed episodes, and rebuilt blueprint signature"
        ),
    )
    parser.add_argument(
        "--estimated-mib-per-episode",
        type=float,
        default=300.0,
        help="Conservative startup disk estimate for grouped collection",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=5.0,
        help="Free space retained beyond the grouped payload estimate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    args = parser.parse_args()

    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if not 0 <= args.seed <= UINT64_MASK:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= UINT64_MASK:
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
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in [1, 95]")

    grouped_values = (
        args.scenario_count,
        args.episodes_per_scenario,
    )
    if any(value is not None for value in grouped_values) and not all(
        value is not None for value in grouped_values
    ):
        parser.error(
            "--scenario-count and --episodes-per-scenario must be supplied "
            "together"
        )
    args.grouped = args.scenario_count is not None
    if args.grouped:
        if args.max_success_episodes is not None:
            parser.error(
                "--max-success-episodes cannot be combined with grouped "
                "collection"
            )
        if args.scenario_count <= 0:
            parser.error("--scenario-count must be positive")
        if args.episodes_per_scenario <= 0:
            parser.error("--episodes-per-scenario must be positive")
        if args.max_attempts_per_scenario < args.episodes_per_scenario:
            parser.error(
                "--max-attempts-per-scenario cannot be smaller than "
                "--episodes-per-scenario"
            )
        if (
            not math.isfinite(args.estimated_mib_per_episode)
            or args.estimated_mib_per_episode <= 0.0
        ):
            parser.error(
                "--estimated-mib-per-episode must be finite and positive"
            )
        if (
            not math.isfinite(args.minimum_free_gib)
            or args.minimum_free_gib < 0.0
        ):
            parser.error("--minimum-free-gib must be finite and non-negative")
    else:
        if args.max_success_episodes is None:
            parser.error(
                "legacy flat generation requires --max-success-episodes; "
                "or select grouped collection with --scenario-count and "
                "--episodes-per-scenario"
            )
        if args.max_success_episodes <= 0:
            parser.error("--max-success-episodes must be positive")
        if args.resume:
            parser.error("--resume is available only for grouped collection")
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
        f"{name}={timing.get(name, 0.0):.1f}s" for name in ordered
    )


def _format_detail(audited_start, result, failed_start_timing):
    fields = []
    timing = (
        audited_start.timing
        if audited_start is not None
        else failed_start_timing
    )
    if timing is not None:
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


def _rounded(values):
    return [round(float(value), 4) for value in values]


def _blueprint_signature(snapshot):
    # Only immutable blueprint-determined identity may appear here. A scene is
    # rebuilt or reused across attempts, and by the time a later attempt runs
    # its Prepare the responders of the previous attempt have already driven
    # and fled, so any live kinematic field (position, heading, velocity,
    # task_target, task_state) differs between two attempts that share the very
    # same blueprint. Including such a field makes the signature compare the
    # simulation state instead of the scene identity and turns every attempt
    # after the first into a spurious SCENARIO_BLUEPRINT_SIGNATURE_MISMATCH.
    # Spawn layout is already guarded plugin-side by reuse_blueprint(), which
    # verifies the cached seed, anchor and actor capacity.
    #
    # event_id is likewise excluded: fire_scenario.cpp derives it as
    # (scenario_id << 8) | 1, and scenario_id is a fresh per-Prepare runtime
    # instance id, so it increments on every attempt even when the very same
    # cached blueprint is reused. It identifies the run, not the scene.
    return {
        "event_position": _rounded(snapshot.event_position),
        "entities": [
            {
                "stable_id": int(entity.stable_id),
                "model_hash": int(entity.model_hash),
                "kind": int(entity.kind),
                "role": int(entity.role),
                "planned_activation_offset_ms": int(
                    entity.planned_activation_offset_ms
                ),
            }
            for entity in snapshot.entities
        ],
    }


def _signature_difference(expected, actual, max_reported=6):
    """Describe why two blueprint signatures differ, for diagnosis."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return f"expected type {type(expected).__name__}, got {type(actual).__name__}"
    notes = []
    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        added = sorted(actual_keys - expected_keys)
        notes.append(
            "top-level keys differ"
            + (f" missing={missing}" if missing else "")
            + (f" unexpected={added}" if added else "")
            + " (a manifest written by an older signature format cannot be "
            "compared; restart the collection or migrate the manifest)"
        )
        return "; ".join(notes)
    if expected.get("event_position") != actual.get("event_position"):
        notes.append(
            f"event_position expected={expected.get('event_position')} "
            f"actual={actual.get('event_position')}"
        )
    expected_entities = expected.get("entities") or []
    actual_entities = actual.get("entities") or []
    if len(expected_entities) != len(actual_entities):
        notes.append(
            f"entity count expected={len(expected_entities)} "
            f"actual={len(actual_entities)}"
        )
    field_totals = {}
    examples = []
    for index, (left, right) in enumerate(
        zip(expected_entities, actual_entities)
    ):
        for key in sorted(set(left) | set(right)):
            before = left.get(key)
            after = right.get(key)
            if before != after:
                field_totals[key] = field_totals.get(key, 0) + 1
                if len(examples) < max_reported:
                    examples.append(
                        f"entity[{index}].{key} expected={before} actual={after}"
                    )
    if field_totals:
        notes.append(
            "differing fields ["
            + ", ".join(
                f"{name}x{count}"
                for name, count in sorted(
                    field_totals.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            + "]"
        )
        notes.extend(examples)
    if not notes:
        notes.append("signatures compare unequal but no field difference found")
    return "; ".join(notes)


def _episode_complete(path):
    required = (
        "summary.json",
        "agent/episode.json",
        "agent/steps.jsonl",
        "teacher/episode.json",
        "teacher/awareness.jsonl",
        "teacher/beliefs.npz",
        "evaluation_truth/episode.json",
        "evaluation_truth/steps.jsonl",
    )
    return path.is_dir() and all((path / item).is_file() for item in required)


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manifest_config(args):
    return {
        "anchor": [float(value) for value in args.anchor],
        "scenario_seed_base": int(args.seed),
        "start_seed_base": int(args.start_seed),
        "scenario_count": int(args.scenario_count),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "max_attempts_per_scenario": int(args.max_attempts_per_scenario),
        "start_attempts": int(args.start_attempts),
        "firetrucks": int(args.firetrucks),
        "pedestrians": int(args.pedestrians),
        "max_steps": int(args.max_steps),
        "jpeg_quality": int(args.jpeg_quality),
    }


def _new_manifest(args):
    scenes = []
    for scene_index in range(args.scenario_count):
        scenario_seed = (int(args.seed) + scene_index) & UINT64_MASK
        scenes.append(
            {
                "scene_index": scene_index,
                "scenario_seed": scenario_seed,
                "directory": f"scene_{scene_index:03d}_seed_{scenario_seed}",
                "status": "PENDING",
                "attempts_completed": 0,
                "successes": 0,
                "episodes": [],
                "blueprint_signature": None,
            }
        )
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_type": "stage2e_grouped_scenarios",
        "config": _manifest_config(args),
        "status": "IN_PROGRESS",
        "scenes": scenes,
    }


def _load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise RuntimeError("Unsupported grouped collection manifest schema")
    if payload.get("collection_type") != "stage2e_grouped_scenarios":
        raise RuntimeError("Output manifest is not a Stage 2E grouped collection")
    return payload


def _validate_resume(output_root, manifest, args):
    expected_config = _manifest_config(args)
    if manifest.get("config") != expected_config:
        raise RuntimeError(
            "Resume configuration does not exactly match dataset_manifest.json"
        )
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != args.scenario_count:
        raise RuntimeError("Grouped manifest scene table is invalid")
    partials = sorted(output_root.rglob("*.partial"))
    if partials:
        raise RuntimeError(
            "Resume found incomplete payload directories; inspect or remove "
            "them explicitly before retrying: "
            + ", ".join(str(path) for path in partials)
        )
    for expected_index, scene in enumerate(scenes):
        if int(scene.get("scene_index", -1)) != expected_index:
            raise RuntimeError("Grouped manifest scene indices are invalid")
        scene_root = output_root / scene["directory"]
        episode_names = scene.get("episodes")
        if not isinstance(episode_names, list):
            raise RuntimeError("Grouped manifest episode list is invalid")
        if int(scene.get("successes", -1)) != len(episode_names):
            raise RuntimeError("Grouped manifest success count is inconsistent")
        if len(set(episode_names)) != len(episode_names):
            raise RuntimeError("Grouped manifest contains duplicate episodes")
        for name in episode_names:
            if not _episode_complete(scene_root / name):
                raise RuntimeError(
                    f"Resume found an incomplete recorded episode: "
                    f"{scene_root / name}"
                )
        actual_names = sorted(
            path.name
            for path in scene_root.glob("episode_*")
            if path.is_dir()
        )
        if actual_names != sorted(episode_names):
            raise RuntimeError(
                f"Scene directory and manifest disagree: {scene_root}"
            )
        attempts = int(scene.get("attempts_completed", -1))
        if not 0 <= attempts <= args.max_attempts_per_scenario:
            raise RuntimeError("Grouped manifest attempt count is invalid")


def _prepare_grouped_output(args):
    output_root = args.output_dir.resolve()
    manifest_path = output_root / "dataset_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume requires {manifest_path}"
            )
        manifest = _load_manifest(manifest_path)
        _validate_resume(output_root, manifest, args)
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                "Grouped --output-dir must be absent or empty unless "
                "--resume is supplied"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = _new_manifest(args)
        for scene in manifest["scenes"]:
            (output_root / scene["directory"]).mkdir()
        _atomic_json(manifest_path, manifest)
    return output_root, manifest_path, manifest


def _check_disk_budget(output_root, manifest, args):
    complete = sum(int(scene["successes"]) for scene in manifest["scenes"])
    requested = args.scenario_count * args.episodes_per_scenario
    remaining = requested - complete
    estimated_payload = remaining * args.estimated_mib_per_episode * 1024**2
    reserve = args.minimum_free_gib * 1024**3
    free = shutil.disk_usage(output_root).free
    required = estimated_payload + reserve
    print(
        "DISK_ESTIMATE "
        f"remaining_episodes={remaining} "
        f"payload={estimated_payload / 1024**3:.1f}GiB "
        f"reserve={args.minimum_free_gib:.1f}GiB "
        f"required={required / 1024**3:.1f}GiB "
        f"free={free / 1024**3:.1f}GiB",
        flush=True,
    )
    if free < required:
        raise RuntimeError(
            "Insufficient free space for the configured grouped collection "
            "estimate; change the destination or explicitly adjust "
            "--estimated-mib-per-episode/--minimum-free-gib"
        )


def _cleanup_frozen(client, session, scenario_id):
    if scenario_id is not None:
        client.reset_scenario(scenario_id)
        scenario_id = None
    if session is not None:
        session.close()
        session = None
    return session, scenario_id


def _run_attempt(
    client,
    args,
    original_pose,
    log_root,
    episode_root,
    scenario_seed,
    start_seed,
    attempt_index,
    success_index,
    runtime_blueprint_id=0,
    expected_signature=None,
    scene_index=None,
):
    scenario_id = None
    session = None
    recorder = None
    ready = None
    signature = None
    audited_start = None
    failed_start_timing = None
    result = None
    recorded_path = None
    attempt_started = time.perf_counter()
    attempt_timing = {}
    outcome = "ERROR"
    phase = "prepare"
    phase_started = attempt_started
    interrupted = False
    try:
        try:
            print(
                "ATTEMPT "
                + (
                    ""
                    if scene_index is None
                    else f"scene={scene_index + 1}/{args.scenario_count} "
                )
                + f"index={attempt_index + 1} scenario_seed={scenario_seed} "
                f"start_seed={start_seed} blueprint={runtime_blueprint_id}",
                flush=True,
            )
            ready, attempt_timing["prepare"] = _timed(
                _prepare_ready,
                client,
                args,
                scenario_seed,
                runtime_blueprint_id,
            )
            scenario_id = ready.scenario_id
            signature = _blueprint_signature(ready)
            if expected_signature is not None and signature != expected_signature:
                raise CollectionInvariantError(
                    "SCENARIO_BLUEPRINT_SIGNATURE_MISMATCH: rebuilt or reused "
                    "scene differs from dataset manifest; "
                    + _signature_difference(expected_signature, signature)
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
            observation_spec = ObservationSpec.from_pair(calibration)
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
            if scene_index is None:
                episode_name = (
                    f"episode_{success_index:06d}_scenario_{scenario_seed}_"
                    f"start_{audited_start.generated.blueprint.start_id}"
                )
            else:
                episode_name = (
                    f"episode_{success_index:04d}_attempt_{attempt_index:04d}_"
                    f"start_{audited_start.generated.blueprint.start_id}"
                )
            recorder = ExpertEpisodeRecorder(
                episode_root,
                episode_name,
                jpeg_quality=args.jpeg_quality,
            )

            phase = "rollout"
            phase_started = time.perf_counter()
            result, attempt_timing["rollout"] = _timed(
                run_expert_episode,
                client,
                session,
                scenario_id,
                audited_start,
                recorder=recorder,
            )
            phase = "write_finalize"
            phase_started = time.perf_counter()
            if result.success:
                recorded_path = recorder.finish(
                    {
                        "result": result,
                        "scene_index": scene_index,
                        "scenario_seed": scenario_seed,
                        "start_seed": start_seed,
                        "attempt_index": attempt_index,
                        "goal_budget_audit": audited_start.goal_budget_audit,
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
                recorder = None
                outcome = "PASS"
                print(
                    f"PASS actions={result.actions} "
                    f"initial_grounded_responses="
                    f"{audited_start.initial_grounded_response_count} "
                    f"plans={result.planner_calls} "
                    f"error={result.localization_error_m:.3f}m "
                    f"sensitivity={result.cue_sensitivity.divergence_kind} "
                    f"path={recorded_path}",
                    flush=True,
                )
            else:
                recorder.abort()
                recorder = None
                outcome = "FAILED"
                append_failure(
                    log_root,
                    {
                        "scene_index": scene_index,
                        "attempt": attempt_index,
                        "scenario_seed": scenario_seed,
                        "start_seed": start_seed,
                        "result": result,
                    },
                )
                print(
                    f"FAIL actions={result.actions} "
                    f"plans={result.planner_calls} message={result.message}",
                    flush=True,
                )
            attempt_timing["write_finalize"] = (
                time.perf_counter() - phase_started
            )
        except KeyboardInterrupt:
            interrupted = True
            outcome = "INTERRUPTED"
        except Exception as error:
            failed_start_timing = getattr(error, "timing", None)
            if phase not in attempt_timing:
                attempt_timing[phase] = time.perf_counter() - phase_started
            append_failure(
                log_root,
                {
                    "scene_index": scene_index,
                    "attempt": attempt_index,
                    "scenario_seed": scenario_seed,
                    "start_seed": start_seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            print(f"FAIL {type(error).__name__}: {error}", flush=True)
            if isinstance(error, CollectionInvariantError):
                raise
    finally:
        cleanup_started = time.perf_counter()
        if recorder is not None:
            recorder.abort()
        session, scenario_id = _cleanup_frozen(
            client,
            session,
            scenario_id,
        )
        attempt_timing["cleanup"] = time.perf_counter() - cleanup_started
        attempt_timing["total"] = time.perf_counter() - attempt_started
        timing_record = {
            "scene_index": scene_index,
            "attempt": attempt_index,
            "scenario_seed": scenario_seed,
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
        append_attempt_timing(log_root, timing_record)
        print(
            f"TIMING outcome={outcome} {_format_timing(attempt_timing)}",
            flush=True,
        )
        detail = _format_detail(audited_start, result, failed_start_timing)
        if detail:
            print(f"TIMING_DETAIL {detail}", flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return {
        "success": outcome == "PASS",
        "outcome": outcome,
        "runtime_blueprint_id": (
            None if ready is None else int(ready.blueprint_id)
        ),
        "blueprint_signature": signature,
        "episode_name": (
            None if recorded_path is None else recorded_path.name
        ),
    }


def _prepare_ready(client, args, scenario_seed, blueprint_id):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=scenario_seed,
        firetruck_count=args.firetrucks,
        pedestrian_count=args.pedestrians,
        blueprint_id=blueprint_id,
    )
    try:
        return client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
    except BaseException:
        client.reset_scenario(scenario_id)
        raise


def _start_seed_for_grouped_attempt(args, scene_index, attempt_index):
    # A fixed scene stride keeps start-seed windows disjoint without making
    # them depend on the configured attempt budget.
    window_index = (int(scene_index) << 32) + int(attempt_index)
    return (
        int(args.start_seed) + window_index * int(args.start_attempts)
    ) & UINT64_MASK


def _run_grouped(args):
    output_root, manifest_path, manifest = _prepare_grouped_output(args)
    _check_disk_budget(output_root, manifest, args)
    if all(
        int(scene["successes"]) >= args.episodes_per_scenario
        for scene in manifest["scenes"]
    ):
        print("DONE grouped collection is already complete", flush=True)
        return

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    collection_started = time.perf_counter()
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchor)
        for scene in manifest["scenes"]:
            scene_index = int(scene["scene_index"])
            if int(scene["successes"]) >= args.episodes_per_scenario:
                scene["status"] = "COMPLETE"
                continue
            print(
                "SCENE_START "
                f"scene={scene_index + 1}/{args.scenario_count} "
                f"seed={scene['scenario_seed']} "
                f"successes={scene['successes']}/"
                f"{args.episodes_per_scenario} "
                f"attempts={scene['attempts_completed']}/"
                f"{args.max_attempts_per_scenario}",
                flush=True,
            )
            scene["status"] = "IN_PROGRESS"
            _atomic_json(manifest_path, manifest)
            runtime_blueprint_id = 0
            expected_signature = scene["blueprint_signature"]
            while (
                int(scene["successes"]) < args.episodes_per_scenario
                and int(scene["attempts_completed"])
                < args.max_attempts_per_scenario
            ):
                attempt_index = int(scene["attempts_completed"])
                start_seed = _start_seed_for_grouped_attempt(
                    args,
                    scene_index,
                    attempt_index,
                )
                result = _run_attempt(
                    client,
                    args,
                    original_pose,
                    output_root,
                    output_root / scene["directory"],
                    int(scene["scenario_seed"]),
                    start_seed,
                    attempt_index,
                    int(scene["successes"]),
                    runtime_blueprint_id=runtime_blueprint_id,
                    expected_signature=expected_signature,
                    scene_index=scene_index,
                )
                if result["runtime_blueprint_id"] is not None:
                    runtime_blueprint_id = result["runtime_blueprint_id"]
                if expected_signature is None and result["blueprint_signature"]:
                    expected_signature = result["blueprint_signature"]
                    scene["blueprint_signature"] = expected_signature
                scene["attempts_completed"] = attempt_index + 1
                if result["success"]:
                    scene["successes"] = int(scene["successes"]) + 1
                    scene["episodes"].append(result["episode_name"])
                print(
                    "SCENE_PROGRESS "
                    f"scene={scene_index + 1}/{args.scenario_count} "
                    f"successes={scene['successes']}/"
                    f"{args.episodes_per_scenario} "
                    f"attempts={scene['attempts_completed']}/"
                    f"{args.max_attempts_per_scenario}",
                    flush=True,
                )
                _atomic_json(manifest_path, manifest)
            if int(scene["successes"]) >= args.episodes_per_scenario:
                scene["status"] = "COMPLETE"
            else:
                scene["status"] = "EXHAUSTED"
            _atomic_json(manifest_path, manifest)

        incomplete = [
            scene
            for scene in manifest["scenes"]
            if int(scene["successes"]) < args.episodes_per_scenario
        ]
        manifest["status"] = "COMPLETE" if not incomplete else "INCOMPLETE"
        _atomic_json(manifest_path, manifest)
        total_successes = sum(
            int(scene["successes"]) for scene in manifest["scenes"]
        )
        print(
            "DONE_GROUPED "
            f"scenes={args.scenario_count - len(incomplete)}/"
            f"{args.scenario_count} episodes={total_successes}/"
            f"{args.scenario_count * args.episodes_per_scenario} "
            f"wall={time.perf_counter() - collection_started:.1f}s",
            flush=True,
        )
        if incomplete:
            summary = ", ".join(
                f"scene {scene['scene_index']}: "
                f"{scene['successes']}/{args.episodes_per_scenario}"
                for scene in incomplete
            )
            raise RuntimeError(
                "Grouped collection did not reach every per-scene quota; "
                + summary
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


def _run_flat(args):
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
            scenario_seed = (args.seed + attempt) & UINT64_MASK
            start_seed = (args.start_seed + attempt) & UINT64_MASK
            result = _run_attempt(
                client,
                args,
                original_pose,
                output_root,
                output_root,
                scenario_seed,
                start_seed,
                attempt,
                successes,
            )
            if result["success"]:
                successes += 1
                print(
                    f"FLAT_PROGRESS successes={successes}/"
                    f"{args.max_success_episodes}",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        rate = successes / attempts_run
        print(
            f"DONE successes={successes} attempts_run={attempts_run} "
            f"attempt_budget={args.max_attempts} rate={rate:.3f} "
            f"wall={elapsed:.1f}s",
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


def main():
    args = _parse_args()
    if args.grouped:
        _run_grouped(args)
    else:
        _run_flat(args)


if __name__ == "__main__":
    main()
