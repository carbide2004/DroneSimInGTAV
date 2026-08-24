import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    CaptureError,
    DroneSimClient,
    DroneSimCommandError,
    DroneSimProtocolError,
    LockstepSession,
    OBLIQUE_PITCH_DEGREES,
)
from agent_control.expert_episode import run_expert_episode  # noqa: E402
from agent_control.expert_teacher import (  # noqa: E402
    ExpertGenerationError,
    SOURCE_STOP_MAX_HORIZONTAL_RANGE_METERS,
    SOURCE_STOP_MIN_PROJECTED_SPAN_PIXELS,
)
from agent_control.expert_recording import (  # noqa: E402
    ExpertEpisodeRecorder,
    append_attempt_timing,
    append_failure,
)
from agent_control.expert_starts import (  # noqa: E402
    certify_scene_start_catalog_rgbd,
    generate_audited_task_start,
)
from agent_control.scene_catalog import (  # noqa: E402
    build_scene_start_catalog,
    scene_catalog_from_json,
    scene_catalog_to_json,
)
from agent_control.start_pool import (  # noqa: E402
    START_POOL_MAX_ENTRIES,
    build_static_start_pool,
    load_pool,
    revalidate_static_start_pool,
    write_pool,
)
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    TASK_FORWARD_STEP_METERS,
    TASK_HORIZON_STEPS,
    TASK_VERTICAL_STEP_METERS,
    TASK_YAW_STEP_DEGREES,
    TaskStartGenerationError,
)


COLLECTION_SCHEMA_VERSION = 4
UINT64_MASK = 0xFFFFFFFFFFFFFFFF


class CollectionInvariantError(RuntimeError):
    pass


def _load_anchor_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Anchor file does not exist: {path}")
    anchors = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if isinstance(record, dict) and all(
                name in record for name in ("x", "y", "z")
            ):
                values = (record["x"], record["y"], record["z"])
            elif isinstance(record, dict) and "position" in record:
                values = record["position"]
            elif isinstance(record, list):
                values = record
            else:
                raise ValueError(
                    f"Anchor at {path}:{line_number} must contain x/y/z"
                )
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in values
                )
            ):
                raise ValueError(
                    f"Anchor at {path}:{line_number} must contain three "
                    "JSON numbers"
                )
            try:
                anchor = tuple(float(value) for value in values)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"Anchor at {path}:{line_number} is not numeric"
                ) from error
            if not all(math.isfinite(value) for value in anchor):
                raise ValueError(
                    f"Anchor at {path}:{line_number} must contain three "
                    "finite coordinates"
                )
            if anchor in anchors:
                raise ValueError(
                    f"Duplicate anchor at {path}:{line_number}: {anchor}"
                )
            anchors.append(anchor)
    if not anchors:
        raise ValueError(f"Anchor file contains no coordinates: {path}")
    return tuple(anchors)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate strict Stage 2E cue-grounded expert episodes. "
            "Only successful episodes retain RGB-D payloads."
        )
    )
    anchor_group = parser.add_mutually_exclusive_group(required=True)
    anchor_group.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="One anchor coordinate (legacy/single-anchor mode)",
    )
    anchor_group.add_argument(
        "--anchor-file",
        type=Path,
        help=(
            "JSONL anchor list written by the ASI F8 hotkey; each row is "
            "{\"x\":...,\"y\":...,\"z\":...}"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=32)
    parser.add_argument("--prepare-timeout", type=float, default=30.0)
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
        "--scenes-per-anchor",
        dest="scenario_count",
        type=int,
        help=(
            "Seed-distinct scenario blueprints created for every anchor "
            "(--scenes-per-anchor is the clearer alias)"
        ),
    )
    parser.add_argument(
        "--episodes-per-scenario",
        "--starts-per-scene",
        dest="episodes_per_scenario",
        type=int,
        help=(
            "Successful start/episode quota for every scene "
            "(--starts-per-scene is the clearer alias)"
        ),
    )
    parser.add_argument(
        "--max-attempts-per-scenario",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--scene-catalog-reserve",
        type=int,
        default=4,
        help=(
            "Extra real-RGB-D starts the generator tries to certify beyond "
            "the hard scene quota; shortfall does not reject the scene"
        ),
    )
    parser.add_argument(
        "--max-scene-seed-candidates",
        type=int,
        default=20,
        help=(
            "Maximum seed-distinct blueprints tried to fill each requested "
            "scene slot"
        ),
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

    if args.anchor is not None:
        if not all(math.isfinite(value) for value in args.anchor):
            parser.error("--anchor values must be finite")
        args.anchors = (tuple(float(value) for value in args.anchor),)
    else:
        try:
            args.anchors = _load_anchor_file(args.anchor_file)
            args.anchor_line_numbers = tuple(
                line_number
                for line_number, raw in enumerate(
                    args.anchor_file.read_text(encoding="utf-8").splitlines(),
                    1,
                )
                if raw.strip()
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
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
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if not 0 <= args.scene_catalog_reserve <= 64:
        parser.error("--scene-catalog-reserve must be in [0, 64]")
    if not 1 <= args.max_scene_seed_candidates <= 256:
        parser.error("--max-scene-seed-candidates must be in [1, 256]")
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
    if len(args.anchors) > 1 and not args.grouped:
        parser.error(
            "--anchor-file with multiple coordinates requires grouped "
            "--scenario-count/--episodes-per-scenario"
        )
    if args.grouped:
        if args.max_success_episodes is not None:
            parser.error(
                "--max-success-episodes cannot be combined with grouped "
                "collection"
            )
        if args.scenario_count <= 0:
            parser.error("--scenario-count must be positive")
        if not 1 <= args.episodes_per_scenario <= 128:
            parser.error("--episodes-per-scenario must be in [1, 128]")
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
        if not 1 <= args.max_success_episodes <= 128:
            parser.error("--max-success-episodes must be in [1, 128]")
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
            f"{timing.attempts} dynamic_response_query="
            f"{timing.dynamic_response_query_seconds:.1f}s yaw_selection="
            f"{timing.yaw_selection_seconds:.1f}s real_camera_verify="
            f"{timing.real_camera_verify_seconds:.1f}s generate="
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
        "anchors": [
            [float(value) for value in anchor]
            for anchor in args.anchors
        ],
        "scenes_per_anchor": int(args.scenario_count),
        "scenario_seed_base": int(args.seed),
        "start_seed_base": int(args.start_seed),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "max_attempts_per_scenario": int(args.max_attempts_per_scenario),
        "firetrucks": int(args.firetrucks),
        "pedestrians": int(args.pedestrians),
        "max_steps": int(args.max_steps),
        "action_spec": {
            "forward_step_m": TASK_FORWARD_STEP_METERS,
            "vertical_step_m": TASK_VERTICAL_STEP_METERS,
            "yaw_step_degrees": TASK_YAW_STEP_DEGREES,
        },
        "source_stop_policy": {
            "maximum_horizontal_range_m": (
                SOURCE_STOP_MAX_HORIZONTAL_RANGE_METERS
            ),
            "minimum_projected_span_pixels": (
                SOURCE_STOP_MIN_PROJECTED_SPAN_PIXELS
            ),
            "consecutive_grounded_observations": 2,
        },
        "scene_catalog_reserve": int(args.scene_catalog_reserve),
        "max_scene_seed_candidates": int(
            args.max_scene_seed_candidates
        ),
        "jpeg_quality": int(args.jpeg_quality),
    }


def _new_manifest(args):
    scenes = []
    for anchor_index, anchor in enumerate(args.anchors):
        for scene_in_anchor in range(args.scenario_count):
            scene_index = anchor_index * args.scenario_count + scene_in_anchor
            scenario_seed = (int(args.seed) + scene_index) & UINT64_MASK
            scenes.append(
                {
                    "scene_index": scene_index,
                    "anchor_index": anchor_index,
                    "scene_in_anchor": scene_in_anchor,
                    "anchor": [float(value) for value in anchor],
                    "scenario_seed": scenario_seed,
                    "scenario_seed_candidate_index": 0,
                    "rejected_scenario_seeds": [],
                    "directory": (
                        f"anchor_{anchor_index:03d}/"
                        f"scene_{scene_in_anchor:03d}_seed_{scenario_seed}"
                    ),
                    "status": "PENDING",
                    "attempts_completed": 0,
                    "successes": 0,
                    "episodes": [],
                    # This id addresses the plugin's single-slot in-memory
                    # blueprint cache. Persist it so restarting this Python
                    # process can reuse the same cached blueprint instead of
                    # accidentally rebuilding a different set of GTA safe
                    # coordinates with blueprint_id=0.
                    "runtime_blueprint_id": None,
                    "blueprint_signature": None,
                    "used_pool_start_ids": [],
                    "attempted_pool_start_ids": [],
                    "scene_start_catalog": None,
                }
            )
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_type": "stage2e_grouped_scenarios",
        "config": _manifest_config(args),
        "status": "IN_PROGRESS",
        "anchor_pools": [],
        "scenes": scenes,
    }


def _load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    schema_version = payload.get("schema_version")
    if schema_version != COLLECTION_SCHEMA_VERSION:
        raise RuntimeError(
            "Grouped manifest schema 1/2/3 cannot be resumed with the "
            "source-shadow start-pool generator; create a new output directory"
        )
    if payload.get("collection_type") != "stage2e_grouped_scenarios":
        raise RuntimeError("Output manifest is not a Stage 2E grouped collection")
    return payload


def _recover_runtime_blueprint_id(scene_root, scene):
    """Recover a pre-fix manifest's cache id from its newest episode."""
    episode_names = scene.get("episodes") or []
    if not episode_names:
        return None
    truth_path = (
        Path(scene_root)
        / episode_names[-1]
        / "evaluation_truth"
        / "episode.json"
    )
    try:
        with truth_path.open("r", encoding="utf-8") as stream:
            truth = json.load(stream)
        blueprint_id = int(
            truth["start_blueprint"]["scenario_blueprint_id"]
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Resume cannot recover runtime_blueprint_id from "
            f"{truth_path}"
        ) from error
    if not 1 <= blueprint_id <= UINT64_MASK:
        raise RuntimeError(
            f"Resume found invalid runtime blueprint id {blueprint_id} "
            f"in {truth_path}"
        )
    return blueprint_id


def _validate_resume(output_root, manifest, args):
    expected_config = _manifest_config(args)
    if manifest.get("config") != expected_config:
        raise RuntimeError(
            "Resume configuration does not exactly match dataset_manifest.json"
        )
    scenes = manifest.get("scenes")
    expected_scene_count = len(args.anchors) * args.scenario_count
    if not isinstance(scenes, list) or len(scenes) != expected_scene_count:
        raise RuntimeError(
            "Grouped manifest scene table does not match "
            "anchors * scenes_per_anchor"
        )
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
        anchor_index = expected_index // args.scenario_count
        scene_in_anchor = expected_index % args.scenario_count
        expected_anchor = [
            float(value) for value in args.anchors[anchor_index]
        ]
        if (
            int(scene.get("anchor_index", -1)) != anchor_index
            or int(scene.get("scene_in_anchor", -1)) != scene_in_anchor
            or scene.get("anchor") != expected_anchor
        ):
            raise RuntimeError(
                "Grouped manifest scene-to-anchor mapping is invalid"
            )
        seed_candidate_index = int(
            scene.get("scenario_seed_candidate_index", -1)
        )
        rejected_seeds = scene.get("rejected_scenario_seeds")
        if (
            not 0 <= seed_candidate_index < args.max_scene_seed_candidates
            or not isinstance(rejected_seeds, list)
            or len(rejected_seeds) != seed_candidate_index
        ):
            raise RuntimeError(
                "Grouped manifest scene-seed candidate history is invalid"
            )
        expected_seed = (
            int(args.seed)
            + expected_index
            + seed_candidate_index * expected_scene_count
        ) & UINT64_MASK
        if int(scene.get("scenario_seed", -1)) != expected_seed:
            raise RuntimeError(
                "Grouped manifest active scenario seed is invalid"
            )
        for rejected_index, rejected in enumerate(rejected_seeds):
            rejected_expected = (
                int(args.seed)
                + expected_index
                + rejected_index * expected_scene_count
            ) & UINT64_MASK
            if (
                int(rejected.get("candidate_index", -1)) != rejected_index
                or int(rejected.get("scenario_seed", -1))
                != rejected_expected
                or not isinstance(rejected.get("reason"), str)
            ):
                raise RuntimeError(
                    "Grouped manifest rejected scenario-seed history is invalid"
                )
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
        used_pool_start_ids = scene.get("used_pool_start_ids")
        if (not isinstance(used_pool_start_ids, list) or
                len(used_pool_start_ids) != int(scene.get("successes", -1)) or
                len(set(int(value) for value in used_pool_start_ids)) !=
                len(used_pool_start_ids)):
            raise RuntimeError("Grouped manifest successful start IDs are invalid")
        attempted_pool_start_ids = scene.get("attempted_pool_start_ids")
        if (not isinstance(attempted_pool_start_ids, list) or
                len(set(int(value) for value in attempted_pool_start_ids)) !=
                len(attempted_pool_start_ids) or
                not set(int(value) for value in used_pool_start_ids).issubset(
                    int(value) for value in attempted_pool_start_ids
                )):
            raise RuntimeError("Grouped manifest attempted start IDs are invalid")
        catalog_payload = scene.get("scene_start_catalog")
        if catalog_payload is not None:
            scene_catalog_from_json(catalog_payload)
        attempts = int(scene.get("attempts_completed", -1))
        if not 0 <= attempts <= args.max_attempts_per_scenario:
            raise RuntimeError("Grouped manifest attempt count is invalid")
        runtime_blueprint_id = scene.get("runtime_blueprint_id")
        if runtime_blueprint_id is None:
            # Backward compatibility for manifests created before this field
            # was persisted. Successful episode truth already contains the
            # exact plugin cache id that created the scene.
            runtime_blueprint_id = _recover_runtime_blueprint_id(
                scene_root,
                scene,
            )
            scene["runtime_blueprint_id"] = runtime_blueprint_id
        elif not 1 <= int(runtime_blueprint_id) <= UINT64_MASK:
            raise RuntimeError(
                "Grouped manifest runtime blueprint id is invalid"
            )


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
        # _validate_resume may migrate an older manifest by recovering the
        # runtime blueprint cache id from its newest completed episode.
        _atomic_json(manifest_path, manifest)
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                "Grouped --output-dir must be absent or empty unless "
                "--resume is supplied"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = _new_manifest(args)
        for scene in manifest["scenes"]:
            (output_root / scene["directory"]).mkdir(
                parents=True,
                exist_ok=False,
            )
        # Fresh grouped collection writes the first manifest only after all
        # per-anchor pool files have been installed by _run_grouped().
    return output_root, manifest_path, manifest


def _check_disk_budget(output_root, manifest, args):
    complete = sum(int(scene["successes"]) for scene in manifest["scenes"])
    requested = len(manifest["scenes"]) * args.episodes_per_scenario
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
    anchor=None,
    runtime_blueprint_id=0,
    expected_signature=None,
    scene_index=None,
    start_pool=None,
    scene_catalog=None,
    attempted_pool_start_ids=(),
    catalog_minimum_entries=1,
    catalog_target_entries=None,
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
    error_phase = None
    error_message = None
    error_type_name = None
    semantic_prepare_failure = False
    semantic_catalog_failure = False
    fatal_error = False
    active_scene_catalog = scene_catalog
    catalog_minimum_entries = int(catalog_minimum_entries)
    catalog_target_entries = (
        catalog_minimum_entries
        if catalog_target_entries is None
        else int(catalog_target_entries)
    )
    if catalog_target_entries < catalog_minimum_entries:
        raise ValueError(
            "catalog_target_entries cannot be smaller than the hard quota"
        )
    candidate_pool_start_id = None
    candidate_consumed = False
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
                    else (
                        f"scene={scene_index + 1}/"
                        f"{len(args.anchors) * args.scenario_count} "
                    )
                )
                + f"index={attempt_index + 1} scenario_seed={scenario_seed} "
                f"start_seed={start_seed} blueprint={runtime_blueprint_id}",
                flush=True,
            )
            ready, attempt_timing["prepare"] = _timed(
                _prepare_ready,
                client,
                args,
                args.anchors[0] if anchor is None else anchor,
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

            if active_scene_catalog is None:
                phase = "scene_catalog"
                phase_started = time.perf_counter()
                (
                    active_scene_catalog,
                    attempt_timing["scene_catalog_projection"],
                ) = _timed(
                    build_scene_start_catalog,
                    client,
                    session,
                    scenario,
                    observation_spec,
                    start_pool,
                    minimum_entries=int(catalog_minimum_entries),
                    horizon_steps=args.max_steps,
                )
                print(
                    "SCENE_CATALOG_PROJECTED "
                    f"scenario_seed={scenario_seed} "
                    f"candidates={len(active_scene_catalog.candidates)} "
                    f"time={attempt_timing['scene_catalog_projection']:.1f}s",
                    flush=True,
                )
            if not active_scene_catalog.real_rgbd_certified:
                phase = "scene_catalog"
                phase_started = time.perf_counter()
                (
                    active_scene_catalog,
                    attempt_timing["scene_catalog_rgbd"],
                ) = _timed(
                    certify_scene_start_catalog_rgbd,
                    client,
                    session,
                    scenario,
                    observation_spec,
                    start_pool,
                    active_scene_catalog,
                    minimum_entries=catalog_target_entries,
                    required_entries=catalog_minimum_entries,
                    horizon_steps=args.max_steps,
                    progress_callback=_progress,
                )
                print(
                    "SCENE_CATALOG_CERTIFIED "
                    f"scenario_seed={scenario_seed} "
                    f"candidates={len(active_scene_catalog.candidates)} "
                    f"digest={active_scene_catalog.digest} "
                    f"time={attempt_timing['scene_catalog_rgbd']:.1f}s",
                    flush=True,
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
                start_pool=start_pool,
                scene_catalog=active_scene_catalog,
                attempted_pool_start_ids=attempted_pool_start_ids,
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
            candidate_pool_start_id = int(audited_start.pool_start_id)
            candidate_consumed = True
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
                        "pool_start_id": audited_start.pool_start_id,
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
            error_phase = phase
            error_message = str(error)
            error_type_name = type(error).__name__
            semantic_prepare_failure = (
                phase == "prepare"
                and ready is None
                and isinstance(error, RuntimeError)
                and error_message.startswith("Fire scenario preparation failed:")
            )
            semantic_catalog_failure = (
                phase == "scene_catalog"
                and isinstance(error, TaskStartGenerationError)
                and error_message.startswith(
                    "SCENE_START_CATALOG_INSUFFICIENT:"
                )
            )
            semantic_start_failure = (
                phase == "start_audit"
                and isinstance(error, TaskStartGenerationError)
            )
            semantic_rollout_failure = (
                phase == "rollout"
                and isinstance(error, ExpertGenerationError)
            )
            fatal_error = not any((
                semantic_prepare_failure,
                semantic_catalog_failure,
                semantic_start_failure,
                semantic_rollout_failure,
            ))
            failed_start_timing = getattr(error, "timing", None)
            failed_catalog = getattr(error, "scene_start_catalog", None)
            if active_scene_catalog is None and failed_catalog is not None:
                active_scene_catalog = failed_catalog
            if audited_start is not None:
                candidate_pool_start_id = int(audited_start.pool_start_id)
                candidate_consumed = not isinstance(
                    error, (CaptureError, DroneSimProtocolError)
                )
            elif hasattr(error, "pool_start_id"):
                candidate_pool_start_id = int(error.pool_start_id)
                candidate_consumed = bool(
                    getattr(error, "consume_candidate", False)
                )
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
        "scene_prepare_failed": semantic_prepare_failure,
        "scene_catalog_failed": semantic_catalog_failure,
        "fatal_error": fatal_error,
        "error_type": error_type_name,
        "error_phase": error_phase,
        "error_message": error_message,
        "outcome": outcome,
        "runtime_blueprint_id": (
            None if ready is None else int(ready.blueprint_id)
        ),
        "blueprint_signature": signature,
        "episode_name": (
            None if recorded_path is None else recorded_path.name
        ),
        "pool_start_id": candidate_pool_start_id,
        "candidate_consumed": candidate_consumed,

        "catalog_exhausted": (
            error_message is not None
            and error_message.startswith("SCENE_START_CATALOG_EXHAUSTED:")
        ),
        "scene_start_catalog": (
            None
            if active_scene_catalog is None
            else scene_catalog_to_json(active_scene_catalog)
        ),
    }


def _prepare_ready(
    client,
    args,
    anchor,
    scenario_seed,
    blueprint_id,
    firetruck_count=None,
    pedestrian_count=None,
):
    firetrucks = args.firetrucks if firetruck_count is None else int(firetruck_count)
    pedestrians = args.pedestrians if pedestrian_count is None else int(pedestrian_count)

    def prepare(requested_blueprint_id):
        scenario_id = client.prepare_fire_scenario(
            anchor,
            seed=scenario_seed,
            firetruck_count=firetrucks,
            pedestrian_count=pedestrians,
            blueprint_id=requested_blueprint_id,
        )
        try:
            return client.wait_scenario_ready(
                scenario_id,
                timeout=args.prepare_timeout,
            )
        except BaseException:
            client.reset_scenario(scenario_id)
            raise

    blueprint_id = int(blueprint_id)
    if blueprint_id == 0:
        return prepare(0)
    try:
        return prepare(blueprint_id)
    except DroneSimCommandError as error:
        if error.status_name != "SCENARIO_PREPARE_FAILED":
            raise
        # The plugin cache is deliberately single-slot and does not survive a
        # GTA/plugin restart. A persisted runtime id therefore accelerates a
        # Python-only resume but cannot be trusted as durable storage. Rebuild
        # from the same anchor/seed when the slot is gone; the caller compares
        # the rebuilt immutable signature before any episode is appended.
        print(
            "BLUEPRINT_CACHE_MISS "
            f"requested={blueprint_id}; rebuilding from anchor and seed",
            flush=True,
        )
        return prepare(0)


def _start_seed_for_grouped_attempt(args, scene_index, attempt_index):
    # A fixed scene stride keeps start-seed windows disjoint without making
    # them depend on the configured attempt budget.
    window_index = (int(scene_index) << 32) + int(attempt_index)
    return (int(args.start_seed) + window_index) & UINT64_MASK



def _rewrite_anchor_file_without_indices(path, rejected_indices):
    rejected_indices = {int(value) for value in rejected_indices}
    path = Path(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    anchor_index = 0
    for raw in raw_lines:
        if raw.strip():
            if anchor_index not in rejected_indices:
                kept.append(raw)
            anchor_index += 1
        else:
            kept.append(raw)
    temporary = path.with_name(path.name + ".tmp")
    if not any(raw.strip() for raw in kept):
        kept = []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.writelines(kept)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _preflight_anchor_pools(client, args, original_pose):
    accepted = []
    pools = []
    rejected = []
    for original_index, anchor in enumerate(args.anchors):
        scenario_id = None
        session = None
        started = time.perf_counter()
        print(
            f"ANCHOR_AUDIT_START anchor={original_index + 1}/{len(args.anchors)} "
            f"position={anchor}",
            flush=True,
        )
        try:
            client.teleport_player(*anchor)
            ready = _prepare_ready(
                client,
                args,
                anchor,
                (int(args.seed) + original_index * args.scenario_count) & UINT64_MASK,
                0,
                firetruck_count=0,
                pedestrian_count=0,
            )
            scenario_id = ready.scenario_id
            client.set_camera_pose(
                ready.event_position[0], ready.event_position[1] - 40.0,
                ready.event_position[2] + 40.0, original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
            session = LockstepSession(client)
            session.__enter__()
            calibration = session.capture_rgbd_pair()
            pool = build_static_start_pool(
                client,
                session,
                ready,
                minimum_entries=min(
                    START_POOL_MAX_ENTRIES, args.episodes_per_scenario,
                ),
                observation_spec=ObservationSpec.from_pair(calibration),
                horizon_steps=args.max_steps,
                progress_callback=_progress,
            )
            pool = dataclass_replace(
                pool,
                timing=dataclass_replace(
                    pool.timing,
                    anchor_prepare=time.perf_counter() - started - pool.timing.total,
                ),
            )
            accepted.append(anchor)
            pools.append(pool)
            print(
                f"ANCHOR_AUDIT_PASS original_anchor={original_index} "
                f"pool={len(pool.entries)} digest={pool.digest} "
                f"bearing_histogram={pool.bearing_histogram} "
                f"wall={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        except (TaskStartGenerationError, RuntimeError) as error:
            message = str(error)
            static_unsuitable = message.startswith("ANCHOR_UNSUITABLE:")
            response_unsuitable = message.startswith(
                "Fire scenario preparation failed: Could not resolve enough "
            ) and (
                "firetruck road nodes" in message or
                "pedestrian safe coordinates" in message
            )
            if not static_unsuitable and not response_unsuitable:
                raise
            pool = getattr(error, "start_pool", None)
            rejected.append(original_index)
            line_number = (
                args.anchor_line_numbers[original_index]
                if args.anchor_file is not None
                else None
            )
            print(
                f"ANCHOR_REMOVED original_anchor={original_index} "
                f"line={line_number} position={anchor} "
                f"pool={0 if pool is None else len(pool.entries)} "
                f"rejections={None if pool is None else dict(pool.rejection_counts)} "
                f"reason={error}",
                flush=True,
            )
        finally:
            session, scenario_id = _cleanup_frozen(client, session, scenario_id)
    if args.anchor_file is not None and rejected:
        _rewrite_anchor_file_without_indices(args.anchor_file, rejected)
        print(
            f"ANCHOR_FILE_REWRITTEN path={args.anchor_file} removed={len(rejected)} "
            f"remaining={len(accepted)}",
            flush=True,
        )
    if not accepted:
        raise RuntimeError("No suitable anchors remain after collection preflight")
    return tuple(accepted), tuple(pools)


def _install_pool_manifest(output_root, manifest, pools):
    manifest["anchor_pools"] = []
    for anchor_index, pool in enumerate(pools):
        relative = Path(f"anchor_{anchor_index:03d}") / "start_pool.json"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_pool(destination, pool)
        manifest["anchor_pools"].append({
            "anchor_index": anchor_index,
            "anchor": list(pool.anchor),
            "event_position": list(pool.event_position),
            "path": relative.as_posix(),
            "digest": pool.digest,
            "count": len(pool.entries),
            "timing": {
                "anchor_prepare": pool.timing.anchor_prepare,
                "shadow_rays": pool.timing.shadow_rays,
                "ground_clearance": pool.timing.ground_clearance,
                "fire_occlusion": pool.timing.fire_occlusion,
                "goal_audit": pool.timing.goal_audit,
                "total": pool.timing.anchor_prepare + pool.timing.total,
            },
        })


def _load_manifest_pools(output_root, manifest):
    records = manifest.get("anchor_pools")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Schema 4 manifest has no anchor pools")
    pools = []
    for expected_index, record in enumerate(records):
        if int(record.get("anchor_index", -1)) != expected_index:
            raise RuntimeError("Manifest anchor-pool indices are invalid")
        pool = load_pool(output_root / record["path"])
        if pool.digest != record.get("digest") or len(pool.entries) != int(record.get("count", -1)):
            raise RuntimeError("ANCHOR_POOL_MISMATCH: manifest metadata differs")
        pools.append(pool)
    return tuple(pools)



def _validate_manifest_pool_usage(manifest, pools):
    for scene in manifest["scenes"]:
        anchor_index = int(scene["anchor_index"])
        valid_ids = {
            int(entry.pool_start_id)
            for entry in pools[anchor_index].entries
        }
        used_ids = {
            int(value) for value in scene["used_pool_start_ids"]
        }
        attempted_ids = {
            int(value) for value in scene["attempted_pool_start_ids"]
        }
        if (
            not used_ids.issubset(attempted_ids)
            or not attempted_ids.issubset(valid_ids)
        ):
            raise RuntimeError(
                "ANCHOR_POOL_MISMATCH: manifest references an unknown "
                "or unattempted pool start"
            )
        payload = scene.get("scene_start_catalog")
        if payload is not None:
            catalog = scene_catalog_from_json(payload)
            if not catalog.real_rgbd_certified:
                raise RuntimeError(
                    "SCENE_START_CATALOG_MISMATCH: manifest catalog is not "
                    "real-RGB-D certified"
                )
            catalog_ids = {
                candidate.pool_start_id
                for candidate in catalog.candidates
            }
            if (
                catalog.pool_digest != pools[anchor_index].digest
                or not catalog_ids.issubset(valid_ids)
                or not attempted_ids.issubset(catalog_ids)
            ):
                raise RuntimeError(
                    "SCENE_START_CATALOG_MISMATCH: catalog and manifest "
                    "candidate IDs disagree"
                )

def _revalidate_manifest_pools(client, args, original_pose, pools):
    for anchor_index, pool in enumerate(pools):
        scenario_id = None
        session = None
        try:
            client.teleport_player(*pool.anchor)
            ready = _prepare_ready(
                client,
                args,
                pool.anchor,
                (int(args.seed) + anchor_index * args.scenario_count) & UINT64_MASK,
                0,
                firetruck_count=0,
                pedestrian_count=0,
            )
            scenario_id = ready.scenario_id
            client.set_camera_pose(
                ready.event_position[0], ready.event_position[1] - 40.0,
                ready.event_position[2] + 40.0, original_pose[5],
                collision_check=False,
            )
            session = LockstepSession(client)
            session.__enter__()
            revalidate_static_start_pool(client, session, ready, pool)
            print(
                f"ANCHOR_POOL_REVALIDATED anchor={anchor_index} "
                f"count={len(pool.entries)} digest={pool.digest}",
                flush=True,
            )
        finally:
            session, scenario_id = _cleanup_frozen(client, session, scenario_id)


def _replace_empty_scene_seed(scene, args, total_scene_count, reason):
    """Replace a rejected empty scene slot with the next deterministic seed."""
    if int(scene["successes"]) != 0:
        return False
    current_index = int(scene.get("scenario_seed_candidate_index", 0))
    next_index = current_index + 1
    if next_index >= int(args.max_scene_seed_candidates):
        return False
    scene.setdefault("rejected_scenario_seeds", []).append({
        "candidate_index": current_index,
        "scenario_seed": int(scene["scenario_seed"]),
        "reason": str(reason),
    })
    scene_index = int(scene["scene_index"])
    scenario_seed = (
        int(args.seed)
        + scene_index
        + next_index * int(total_scene_count)
    ) & UINT64_MASK
    scene_in_anchor = int(scene["scene_in_anchor"])
    anchor_index = int(scene["anchor_index"])
    scene.update({
        "scenario_seed": scenario_seed,
        "scenario_seed_candidate_index": next_index,
        "directory": (
            f"anchor_{anchor_index:03d}/"
            f"scene_{scene_in_anchor:03d}_candidate_{next_index:03d}_"
            f"seed_{scenario_seed}"
        ),
        "status": "PENDING",
        "attempts_completed": 0,
        "successes": 0,
        "episodes": [],
        "runtime_blueprint_id": None,
        "blueprint_signature": None,
        "used_pool_start_ids": [],
        "attempted_pool_start_ids": [],
        "scene_start_catalog": None,
    })
    for key in (
        "prepare_failure",
        "catalog_failure",
        "rejected_scene_start_catalog",
    ):
        scene.pop(key, None)
    return True

def _run_grouped(args):
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Grouped --output-dir must be absent or empty unless --resume is supplied"
        )
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    try:
        if args.resume:
            output_root, manifest_path, manifest = _prepare_grouped_output(args)
            pools = _load_manifest_pools(output_root, manifest)
            _validate_manifest_pool_usage(manifest, pools)
            _revalidate_manifest_pools(client, args, original_pose, pools)
        else:
            accepted, pools = _preflight_anchor_pools(client, args, original_pose)
            args.anchors = accepted
            output_root, manifest_path, manifest = _prepare_grouped_output(args)
            _install_pool_manifest(output_root, manifest, pools)
            _atomic_json(manifest_path, manifest)
    except BaseException:
        try:
            client.set_camera_pose(
                original_pose[0], original_pose[1], original_pose[2],
                original_pose[5], collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
        raise
    try:
        _check_disk_budget(output_root, manifest, args)
    except BaseException:
        try:
            client.set_camera_pose(
                original_pose[0], original_pose[1], original_pose[2],
                original_pose[5], collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
        raise
    if all(
        int(scene["successes"]) >= args.episodes_per_scenario
        for scene in manifest["scenes"]
    ):
        try:
            client.set_camera_pose(
                original_pose[0], original_pose[1], original_pose[2],
                original_pose[5], collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
        print("DONE grouped collection is already complete", flush=True)
        return
    collection_started = time.perf_counter()
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        current_anchor_index = None
        total_scene_count = len(manifest["scenes"])
        for scene in manifest["scenes"]:
            scene_index = int(scene["scene_index"])
            anchor_index = int(scene["anchor_index"])
            scene_in_anchor = int(scene["scene_in_anchor"])
            anchor = tuple(float(value) for value in scene["anchor"])
            if anchor_index != current_anchor_index:
                client.teleport_player(*anchor)
                current_anchor_index = anchor_index
                print(
                    "ANCHOR_START "
                    f"anchor={anchor_index + 1}/{len(args.anchors)} "
                    f"position={anchor}",
                    flush=True,
                )
            if int(scene["successes"]) >= args.episodes_per_scenario:
                scene["status"] = "COMPLETE"
                continue
            print(
                "SCENE_START "
                f"anchor={anchor_index + 1}/{len(args.anchors)} "
                f"scene={scene_in_anchor + 1}/{args.scenario_count} "
                f"global_scene={scene_index + 1}/{total_scene_count} "
                f"seed={scene['scenario_seed']} "
                f"successes={scene['successes']}/"
                f"{args.episodes_per_scenario} "
                f"attempts={scene['attempts_completed']}/"
                f"{args.max_attempts_per_scenario}",
                flush=True,
            )
            scene["status"] = "IN_PROGRESS"
            _atomic_json(manifest_path, manifest)
            runtime_blueprint_id = int(
                scene.get("runtime_blueprint_id") or 0
            )
            expected_signature = scene["blueprint_signature"]
            catalog_payload = scene.get("scene_start_catalog")
            active_scene_catalog = (
                None
                if catalog_payload is None
                else scene_catalog_from_json(catalog_payload)
            )
            scene_prepare_failed = False
            scene_catalog_invalid = False
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
                    anchor=anchor,
                    runtime_blueprint_id=runtime_blueprint_id,
                    expected_signature=expected_signature,
                    scene_index=scene_index,
                    start_pool=pools[anchor_index],
                    scene_catalog=active_scene_catalog,
                    attempted_pool_start_ids=(
                        scene["attempted_pool_start_ids"]
                    ),
                    catalog_minimum_entries=min(
                        START_POOL_MAX_ENTRIES, args.episodes_per_scenario,
                    ),
                    catalog_target_entries=min(
                        START_POOL_MAX_ENTRIES,
                        args.episodes_per_scenario
                        + args.scene_catalog_reserve,
                    ),
                )
                if result["fatal_error"]:
                    _atomic_json(manifest_path, manifest)
                    raise RuntimeError(
                        "FATAL_COLLECTION_ERROR: "
                        f"phase={result['error_phase']} "
                        f"error={result['error_type']} "
                        f"message={result['error_message']}"
                    )
                if result["runtime_blueprint_id"] is not None:
                    runtime_blueprint_id = result["runtime_blueprint_id"]
                    scene["runtime_blueprint_id"] = runtime_blueprint_id
                if expected_signature is None and result["blueprint_signature"]:
                    expected_signature = result["blueprint_signature"]
                    scene["blueprint_signature"] = expected_signature

                returned_catalog = result["scene_start_catalog"]
                if returned_catalog is not None and not result[
                    "scene_catalog_failed"
                ]:
                    parsed = scene_catalog_from_json(returned_catalog)
                    if (
                        active_scene_catalog is not None
                        and parsed.digest != active_scene_catalog.digest
                    ):
                        raise CollectionInvariantError(
                            "SCENE_START_CATALOG_MISMATCH: rebuilt catalog "
                            "differs from the manifest"
                        )
                    active_scene_catalog = parsed
                    scene["scene_start_catalog"] = returned_catalog

                scene["attempts_completed"] = attempt_index + 1
                if result["candidate_consumed"]:
                    pool_start_id = int(result["pool_start_id"])
                    attempted = scene["attempted_pool_start_ids"]
                    if pool_start_id in attempted:
                        raise CollectionInvariantError(
                            "A consumed scene-start candidate was attempted twice"
                        )
                    attempted.append(pool_start_id)
                if result["success"]:
                    scene["successes"] = int(scene["successes"]) + 1
                    scene["episodes"].append(result["episode_name"])
                    scene["used_pool_start_ids"].append(
                        int(result["pool_start_id"])
                    )

                if result["scene_prepare_failed"]:
                    scene_prepare_failed = True
                    scene["prepare_failure"] = result["error_message"]
                    print(
                        "SCENE_PREPARE_FAILED "
                        f"scene={scene_index + 1}/{total_scene_count}; "
                        "this scene seed cannot be used",
                        flush=True,
                    )
                if result["scene_catalog_failed"] or result[
                    "catalog_exhausted"
                ]:
                    scene_catalog_invalid = True
                    scene["catalog_failure"] = result["error_message"]
                    if returned_catalog is not None:
                        scene["rejected_scene_start_catalog"] = (
                            returned_catalog
                        )
                    print(
                        "SCENE_CATALOG_REJECTED "
                        f"scene={scene_index + 1}/{total_scene_count} "
                        f"reason={result['error_message']}",
                        flush=True,
                    )

                print(
                    "SCENE_PROGRESS "
                    f"anchor={anchor_index + 1}/{len(args.anchors)} "
                    f"scene={scene_in_anchor + 1}/{args.scenario_count} "
                    f"successes={scene['successes']}/"
                    f"{args.episodes_per_scenario} "
                    f"attempts={scene['attempts_completed']}/"
                    f"{args.max_attempts_per_scenario} "
                    f"candidates={len(scene['attempted_pool_start_ids'])}",
                    flush=True,
                )
                _atomic_json(manifest_path, manifest)
                if scene_prepare_failed or scene_catalog_invalid:
                    replacement_reason = (
                        scene.get("prepare_failure")
                        or scene.get("catalog_failure")
                        or "scene rejected"
                    )
                    if _replace_empty_scene_seed(
                        scene,
                        args,
                        total_scene_count,
                        replacement_reason,
                    ):
                        (output_root / scene["directory"]).mkdir(
                            parents=True,
                            exist_ok=False,
                        )
                        runtime_blueprint_id = 0
                        expected_signature = None
                        active_scene_catalog = None
                        scene_prepare_failed = False
                        scene_catalog_invalid = False
                        print(
                            "SCENE_SEED_REPLACED "
                            f"scene={scene_index + 1}/{total_scene_count} "
                            f"candidate={scene['scenario_seed_candidate_index'] + 1}/"
                            f"{args.max_scene_seed_candidates} "
                            f"seed={scene['scenario_seed']} "
                            f"reason={replacement_reason}",
                            flush=True,
                        )
                        _atomic_json(manifest_path, manifest)
                        continue
                    break
            if int(scene["successes"]) >= args.episodes_per_scenario:
                scene["status"] = "COMPLETE"
            elif scene_prepare_failed:
                scene["status"] = "PREPARE_FAILED"
            elif scene_catalog_invalid:
                scene["status"] = "CATALOG_REJECTED"
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
        total_scene_count = len(manifest["scenes"])
        print(
            "DONE_GROUPED "
            f"anchors={len(args.anchors)} "
            f"scenes={total_scene_count - len(incomplete)}/"
            f"{total_scene_count} episodes={total_successes}/"
            f"{total_scene_count * args.episodes_per_scenario} "
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
    start_pool = None
    started = time.perf_counter()
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        client.teleport_player(*args.anchors[0])
        preflight_ready = _prepare_ready(
            client, args, args.anchors[0], args.seed, 0,
            firetruck_count=0, pedestrian_count=0,
        )
        preflight_session = LockstepSession(client)
        preflight_session.__enter__()
        try:
            calibration = preflight_session.capture_rgbd_pair()
            start_pool = build_static_start_pool(
                client, preflight_session, preflight_ready,
                minimum_entries=args.max_success_episodes,
                observation_spec=ObservationSpec.from_pair(calibration),
                horizon_steps=args.max_steps,
                progress_callback=_progress,
            )
        finally:
            client.reset_scenario(preflight_ready.scenario_id)
            preflight_session.close()
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
                start_pool=start_pool,
                scene_catalog=None,
                attempted_pool_start_ids=(),
                catalog_minimum_entries=1,
                catalog_target_entries=1,
            )
            if result["fatal_error"]:
                raise RuntimeError(
                    "FATAL_COLLECTION_ERROR: "
                    f"phase={result['error_phase']} "
                    f"error={result['error_type']} "
                    f"message={result['error_message']}"
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
