import argparse
import json
import math
import sys
import tempfile
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
from agent_control.start_pool import (  # noqa: E402
    START_POOL_MAX_ENTRIES,
    CANDIDATE_ALTITUDES_AGL_METERS,
    CANDIDATE_RADII_METERS,
    build_static_start_pool,
    load_pool,
    revalidate_static_start_pool,
    write_pool,
)
from agent_control.task_starts import ObservationSpec  # noqa: E402
from validation.generate_stage2e_experts import (  # noqa: E402
    _load_anchor_file,
    _rewrite_anchor_file_without_indices,
)


def _arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the evaluation-only source-shadow start-pool APIs. "
            "No RGB-D or trajectory payload is written."
        )
    )
    parser.add_argument("--anchor", type=float, nargs=3, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minimum-entries", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--prepare-timeout", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--verify-seed-isolation", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor must contain finite coordinates")
    if not 1 <= args.minimum_entries <= START_POOL_MAX_ENTRIES:
        parser.error(
            "--minimum-entries must be in [1, "
            f"{START_POOL_MAX_ENTRIES}]"
        )
    if not 21 <= args.max_steps <= 256:
        parser.error("--max-steps must be in [21, 256]")
    return args


def _prepare(client, args, seed, blueprint_id=0):
    scenario_id = client.prepare_fire_scenario(
        args.anchor,
        seed=seed,
        firetruck_count=0,
        pedestrian_count=0,
        blueprint_id=blueprint_id,
    )
    try:
        return client.wait_scenario_ready(
            scenario_id, timeout=args.prepare_timeout
        )
    except BaseException:
        client.reset_scenario(scenario_id)
        raise


def _enter_at_source(client, ready, yaw):
    client.set_camera_pose(
        ready.event_position[0],
        ready.event_position[1] - 40.0,
        ready.event_position[2] + 40.0,
        yaw,
        collision_check=False,
    )
    client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
    session = LockstepSession(client)
    session.__enter__()
    return session


def _cleanup(client, session, scenario_id):
    if scenario_id is not None:
        client.reset_scenario(scenario_id)
    if session is not None:
        session.close()


def _assert_clock(clock, snapshot, label):
    if (
        snapshot.step_index != clock.step_index
        or snapshot.game_timer_ms != clock.game_timer_ms
    ):
        raise RuntimeError(f"{label} advanced the lockstep instant")


def _validate_batch_contracts(client, session, ready):
    clock = session.refresh()
    # Upward is a deliberate no-hit probe for an ordinary outdoor road
    # anchor. An anchor under static overhead geometry fails explicitly.
    shadow = client.probe_fire_shadow_batch(
        ready.scenario_id,
        session.session_id,
        [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)],
        timeout=30.0,
    )
    _assert_clock(clock, shadow, "fire-shadow batch")
    for ray in shadow.rays:
        if not 0.0 <= ray.distance <= 120.001:
            raise RuntimeError("Fire-shadow distance is outside [0, 120]m")
        if ray.hit and math.dist(shadow.origin, ray.position) > 120.001:
            raise RuntimeError("Fire-shadow hit lies beyond the cast range")
    if shadow.rays[0].hit:
        raise RuntimeError(
            "The upward 120m no-hit control encountered static geometry"
        )
    if abs(shadow.rays[0].distance - 120.0) > 1.0e-3:
        raise RuntimeError("No-hit fire-shadow ray did not report 120m")

    starts = client.probe_camera_start_batch(
        session.session_id,
        [
            (ready.event_position[0] + 42.0, ready.event_position[1], 25.0),
            (ready.event_position[0] + 48.0, ready.event_position[1], 35.0),
        ],
        timeout=30.0,
    )
    _assert_clock(clock, starts, "camera-start batch")
    centers = (
        (ready.event_position[0], ready.event_position[1],
         ready.event_position[2] + 40.0),
    )
    fire = client.query_fire_occlusion_batch(
        ready.scenario_id,
        session.session_id,
        centers,
        timeout=30.0,
    )
    _assert_clock(clock, fire, "fire-occlusion batch")
    return shadow


def _validate_pool(pool, minimum_entries, max_steps):
    if len(pool.entries) < minimum_entries:
        raise RuntimeError("Pool contains fewer entries than requested")
    valid_radii = set(CANDIDATE_RADII_METERS)
    valid_altitudes = set(CANDIDATE_ALTITUDES_AGL_METERS)
    identifiers = set()
    for entry in pool.entries:
        if entry.pool_start_id in identifiers:
            raise RuntimeError("Pool contains a duplicate start ID")
        identifiers.add(entry.pool_start_id)
        if entry.radius not in valid_radii or entry.altitude_agl not in valid_altitudes:
            raise RuntimeError("Pool entry is outside the fixed radius/AGL grid")
        measured_radius = math.dist(entry.position[:2], pool.event_position[:2])
        if abs(measured_radius - entry.radius) > 0.05:
            raise RuntimeError("Pool entry horizontal radius is inconsistent")
        measured_agl = entry.position[2] - entry.ground_z
        if abs(measured_agl - entry.altitude_agl) > 0.05:
            raise RuntimeError("Pool entry AGL is inconsistent")
        if any(sample.clear_line_of_sight for sample in entry.source_vehicle.samples):
            raise RuntimeError("Pool entry has a clear source-vehicle sample")
        if entry.optimistic_goal_actions > int(max_steps) - 15:
            raise RuntimeError("Pool entry violates max_steps - 15")
    if not pool.goal_views:
        raise RuntimeError("Pool has no task-observable goal view")


def _validate_file_and_digest_semantics(pool):
    with tempfile.TemporaryDirectory(prefix="dronesim_pool_validation_") as root:
        root = Path(root)
        pool_path = root / "start_pool.json"
        write_pool(pool_path, pool)
        loaded = load_pool(pool_path)
        if loaded.digest != pool.digest:
            raise RuntimeError("Saved pool digest changed on reload")
        payload = json.loads(pool_path.read_text(encoding="utf-8"))
        payload["entries"][0]["position"][0] += 0.25
        pool_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_pool(pool_path)
        except RuntimeError as error:
            if "ANCHOR_POOL_MISMATCH" not in str(error):
                raise
        else:
            raise RuntimeError("Tampered start pool was accepted")

        anchor_path = root / "anchors.jsonl"
        rows = (
            '{"x":1,"y":2,"z":3}\n',
            '{"x":4,"y":5,"z":6}\n',
            '{"x":7,"y":8,"z":9}\n',
        )
        anchor_path.write_text("".join(rows), encoding="utf-8")
        _rewrite_anchor_file_without_indices(anchor_path, {1})
        if anchor_path.read_text(encoding="utf-8") != rows[0] + rows[2]:
            raise RuntimeError("Anchor-file rewrite changed retained rows")
        if _load_anchor_file(anchor_path) != ((1.0, 2.0, 3.0), (7.0, 8.0, 9.0)):
            raise RuntimeError("Anchor-file rewrite produced invalid JSONL")


def main():
    args = _arguments()
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    scenario_id = None
    session = None
    started = time.perf_counter()
    try:
        client.teleport_player(*args.anchor)
        ready = _prepare(client, args, args.seed)
        scenario_id = ready.scenario_id
        session = _enter_at_source(client, ready, original_pose[5])
        baseline = _validate_batch_contracts(client, session, ready)
        calibration = ObservationSpec.from_pair(session.capture_rgbd_pair())
        pool = build_static_start_pool(
            client,
            session,
            ready,
            minimum_entries=args.minimum_entries,
            observation_spec=calibration,
            horizon_steps=args.max_steps,
            progress_callback=lambda message: print(message, flush=True),
        )
        _validate_pool(pool, args.minimum_entries, args.max_steps)
        revalidate_static_start_pool(client, session, ready, pool)
        print(
            f"pool PASS count={len(pool.entries)} digest={pool.digest} "
            f"bearing_histogram={pool.bearing_histogram}",
            flush=True,
        )
        _cleanup(client, session, scenario_id)
        scenario_id = None
        session = None

        if args.verify_seed_isolation:
            ready = _prepare(client, args, (args.seed + 1) & 0xFFFFFFFFFFFFFFFF)
            scenario_id = ready.scenario_id
            session = _enter_at_source(client, ready, original_pose[5])
            calibration = ObservationSpec.from_pair(session.capture_rgbd_pair())
            second = build_static_start_pool(
                client,
                session,
                ready,
                minimum_entries=args.minimum_entries,
                observation_spec=calibration,
                horizon_steps=args.max_steps,
            )
            if second.digest != pool.digest:
                raise RuntimeError(
                    f"Scenario seed changed pool digest: {pool.digest} != {second.digest}"
                )
            print("seed isolation PASS", flush=True)

        _validate_file_and_digest_semantics(pool)
        print(
            f"PASS wall={time.perf_counter() - started:.1f}s "
            f"shadow_control_hits={[ray.hit for ray in baseline.rays]}",
            flush=True,
        )
        print("No RGB-D, visibility, or trajectory payload was written to disk.")
    finally:
        try:
            _cleanup(client, session, scenario_id)
        finally:
            try:
                client.set_camera_pose(
                    original_pose[0], original_pose[1], original_pose[2],
                    original_pose[5], collision_check=False,
                )
                client.set_camera_pitch(original_pose[3])
            finally:
                client.restore_player()


if __name__ == "__main__":
    main()
