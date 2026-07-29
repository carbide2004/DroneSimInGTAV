"""Validate the Stage 1 fire lifecycle and response truth online in GTA V.

All snapshots stay in memory. The script writes no RGB-D, trajectory, image,
video, or point-cloud payload.
"""

import argparse
import math
from pathlib import Path
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
    ScenarioTaskState,
)


def _distance(position, event_position):
    return float(
        np.linalg.norm(
            np.asarray(position, dtype=np.float64)
            - np.asarray(event_position, dtype=np.float64)
        )
    )


def _direction_cosine(entity, event_position, toward):
    velocity = np.asarray(entity.velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed < 0.1:
        return None
    radial = (
        np.asarray(event_position, dtype=np.float64)
        - np.asarray(entity.position, dtype=np.float64)
    )
    if not toward:
        radial = -radial
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-6:
        return None
    return float(np.dot(velocity, radial) / (speed * radial_norm))


def _entities_by_role(snapshot, role):
    return tuple(
        entity for entity in snapshot.entities if entity.role == role
    )


def _validate_ready(
    snapshot,
    firetruck_count,
    pedestrian_count,
    require_clean_area,
):
    if snapshot.lifecycle != ScenarioLifecycle.READY:
        raise RuntimeError(
            f"Expected READY, received {snapshot.lifecycle.name}"
        )
    if snapshot.event_active:
        raise RuntimeError("Fire was active before Start")
    expected_counts = {
        ScenarioEntityRole.FIRE_SOURCE_VEHICLE: 1,
        ScenarioEntityRole.FIRE_TRUCK: firetruck_count,
        ScenarioEntityRole.FIREFIGHTER_DRIVER: firetruck_count,
        ScenarioEntityRole.FLEEING_PEDESTRIAN: pedestrian_count,
    }
    ids = set()
    event_ids = set()
    for role, expected in expected_counts.items():
        actual = _entities_by_role(snapshot, role)
        if len(actual) != expected:
            raise RuntimeError(
                f"{role.name}: expected {expected}, received {len(actual)}"
            )
        for entity in actual:
            if not entity.exists:
                raise RuntimeError(
                    f"{role.name} entity {entity.stable_id} does not exist"
                )
            if entity.stable_id in ids:
                raise RuntimeError(
                    f"Duplicate stable ID {entity.stable_id}"
                )
            ids.add(entity.stable_id)
            event_ids.add(entity.event_id)
            if entity.speed > 0.25:
                raise RuntimeError(
                    f"{role.name} moved in READY at {entity.speed:.3f} m/s"
                )
            if (
                role != ScenarioEntityRole.FIRE_SOURCE_VEHICLE
                and entity.task_state != ScenarioTaskState.PENDING
            ):
                raise RuntimeError(
                    f"{role.name} task was {entity.task_state.name} in READY"
                )
    if len(event_ids) != 1:
        raise RuntimeError(
            f"Entities have inconsistent event affiliations: {event_ids}"
        )
    if snapshot.ambient_pedestrians or snapshot.ambient_vehicles:
        raise RuntimeError(
            "Controlled area contains ambient entities in READY: "
            f"peds={snapshot.ambient_pedestrians}, "
            f"vehicles={snapshot.ambient_vehicles}"
        )
    existing_protected = tuple(
        entity
        for entity in snapshot.protected_entities
        if entity.exists
    )
    if existing_protected:
        details = ", ".join(
            f"{entity.kind.name.lower()}"
            f"(handle={entity.gta_handle}, "
            f"model=0x{entity.model_hash:08x}, "
            f"pos=({entity.position[0]:.1f},"
            f"{entity.position[1]:.1f},"
            f"{entity.position[2]:.1f}))"
            for entity in existing_protected
        )
        if require_clean_area:
            raise RuntimeError(
                "Controlled area contains protected mission entities: "
                + details
            )
        print(
            "WARNING protected mission entities were preserved: "
            + details
        )


def _find_gta_process(name):
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


def _sample_running(
    client,
    scenario_id,
    duration,
    poll_interval,
    rgbd_captures,
    capture_timeout_ms,
    gta_process,
    max_memory_growth_mb,
    require_clean_area,
):
    deadline = time.monotonic() + duration
    snapshots = []
    previous_capture_frame = None
    capture_count = 0
    consecutive_ambient_samples = 0
    known_protected_handles = set()
    initial_rss = gta_process.memory_info().rss if gta_process else 0
    peak_rss = initial_rss
    while True:
        snapshot = client.get_scenario_state(scenario_id)
        if snapshot.lifecycle == ScenarioLifecycle.FAILED:
            raise RuntimeError(
                "Scenario failed while running: "
                f"{snapshot.failure_message or 'unknown failure'}"
            )
        if snapshot.lifecycle != ScenarioLifecycle.RUNNING:
            raise RuntimeError(
                f"Unexpected running state {snapshot.lifecycle.name}"
            )
        if snapshots:
            previous = snapshots[-1]
            if snapshot.frame_count <= previous.frame_count:
                raise RuntimeError("Scenario frame_count was not increasing")
            if snapshot.game_timer_ms < previous.game_timer_ms:
                raise RuntimeError("Scenario game timer moved backwards")
            if {
                entity.stable_id for entity in snapshot.entities
            } != {
                entity.stable_id for entity in previous.entities
            }:
                raise RuntimeError(
                    "Scenario stable entity IDs changed while running"
                )
        for entity in snapshot.entities:
            if not entity.exists:
                raise RuntimeError(
                    f"Scenario entity {entity.stable_id} disappeared"
                )
            if entity.task_state in (
                ScenarioTaskState.FAILED,
                ScenarioTaskState.LOST,
            ):
                raise RuntimeError(
                    f"{entity.role.name} task became "
                    f"{entity.task_state.name}"
                )
        current_protected = {
            entity.gta_handle
            for entity in snapshot.protected_entities
            if entity.exists
        }
        new_protected = current_protected - known_protected_handles
        if new_protected:
            if require_clean_area:
                raise RuntimeError(
                    "Protected mission entities entered the controlled "
                    f"area while running: {sorted(new_protected)}"
                )
            print(
                "WARNING protected mission entities entered while "
                f"running: {sorted(new_protected)}"
            )
        known_protected_handles.update(current_protected)
        if snapshot.ambient_pedestrians or snapshot.ambient_vehicles:
            consecutive_ambient_samples += 1
            print(
                "WARNING transient ambient entity in controlled area: "
                f"peds={snapshot.ambient_pedestrians}, "
                f"vehicles={snapshot.ambient_vehicles}"
            )
            if consecutive_ambient_samples >= 3:
                raise RuntimeError(
                    "Ambient entities persisted for three consecutive "
                    "scenario snapshots"
                )
        else:
            consecutive_ambient_samples = 0
        snapshots.append(snapshot)
        if capture_count < rgbd_captures:
            frame = client.capture(capture_timeout_ms)
            if (
                previous_capture_frame is not None
                and frame.frame_id <= previous_capture_frame
            ):
                raise RuntimeError(
                    "Capture frame_id was not strictly increasing"
                )
            previous_capture_frame = frame.frame_id
            rgb = frame.rgb_array()
            depth = frame.depth_array()
            if rgb.shape[:2] != depth.shape:
                raise RuntimeError(
                    "RGB and metric depth dimensions differ"
                )
            capture_count += 1
            peak_rss = max(peak_rss, gta_process.memory_info().rss)
            if capture_count % 50 == 0:
                print(
                    f"rgbd={capture_count}/{rgbd_captures} "
                    f"frame={frame.frame_id} "
                    f"depth=[{float(depth.min()):.3f}, "
                    f"{float(depth.max()):.3f}]m"
                )
            continue
        if time.monotonic() >= deadline:
            final_rss = (
                gta_process.memory_info().rss if gta_process else 0
            )
            growth_mb = (
                (final_rss - initial_rss) / (1024**2)
                if gta_process
                else 0.0
            )
            if growth_mb > max_memory_growth_mb:
                raise RuntimeError(
                    f"GTA memory grew by {growth_mb:.1f} MiB; "
                    f"limit is {max_memory_growth_mb:.1f} MiB"
                )
            return snapshots, growth_mb, (
                (peak_rss - initial_rss) / (1024**2)
                if gta_process
                else 0.0
            )
        time.sleep(poll_interval)


def _summarize_responses(snapshots):
    initial = snapshots[0]
    final = snapshots[-1]
    if not any(snapshot.event_active for snapshot in snapshots):
        raise RuntimeError("Fire never became active")

    initial_by_id = {
        entity.stable_id: entity for entity in initial.entities
    }
    toward_cosines = []
    away_cosines = []
    response_delays = []
    truck_distance_changes = []
    truck_diagnostics = []
    pedestrian_distance_changes = []
    for snapshot in snapshots:
        for entity in snapshot.entities:
            if entity.role == ScenarioEntityRole.FIRE_TRUCK:
                cosine = _direction_cosine(
                    entity, snapshot.event_position, toward=True
                )
                if cosine is not None:
                    toward_cosines.append(cosine)
            elif entity.role == ScenarioEntityRole.FLEEING_PEDESTRIAN:
                cosine = _direction_cosine(
                    entity, snapshot.event_position, toward=False
                )
                if cosine is not None:
                    away_cosines.append(cosine)

    for entity in final.entities:
        start_entity = initial_by_id[entity.stable_id]
        if entity.response_start_game_timer_ms:
            response_delays.append(
                entity.response_start_game_timer_ms
                - final.start_game_timer_ms
            )
        start_distance = _distance(
            start_entity.position, initial.event_position
        )
        end_distance = _distance(entity.position, final.event_position)
        if entity.role == ScenarioEntityRole.FIRE_TRUCK:
            progress = start_distance - end_distance
            truck_distance_changes.append(progress)
            entity_samples = [
                snapshot_entity
                for snapshot in snapshots
                for snapshot_entity in snapshot.entities
                if snapshot_entity.stable_id == entity.stable_id
            ]
            maximum_speed = max(
                sample.speed for sample in entity_samples
            )
            response_delay = (
                entity.response_start_game_timer_ms
                - final.start_game_timer_ms
                if entity.response_start_game_timer_ms
                else None
            )
            truck_diagnostics.append(
                "id="
                f"{entity.stable_id} state={entity.task_state.name} "
                f"distance={start_distance:.2f}->{end_distance:.2f}m "
                f"progress={progress:.2f}m max_speed={maximum_speed:.2f}m/s "
                f"response_delay_ms={response_delay}"
            )
        elif entity.role == ScenarioEntityRole.FLEEING_PEDESTRIAN:
            pedestrian_distance_changes.append(end_distance - start_distance)

    if truck_distance_changes and max(truck_distance_changes) <= 0.5:
        raise RuntimeError(
            "No firetruck made measurable progress toward fire: "
            + "; ".join(truck_diagnostics)
        )
    if (
        pedestrian_distance_changes
        and max(pedestrian_distance_changes) <= 0.5
    ):
        raise RuntimeError("No pedestrian made measurable progress away")

    toward_rate = (
        sum(value > 0.0 for value in toward_cosines)
        / len(toward_cosines)
        if toward_cosines
        else float("nan")
    )
    away_rate = (
        sum(value > 0.0 for value in away_cosines)
        / len(away_cosines)
        if away_cosines
        else float("nan")
    )
    delay_text = (
        f"{min(response_delays)}..{max(response_delays)}ms"
        if response_delays
        else "none"
    )
    print(
        "responses "
        f"truck_progress={truck_distance_changes} "
        f"ped_progress={pedestrian_distance_changes} "
        f"toward_rate={toward_rate:.3f} "
        f"away_rate={away_rate:.3f} "
        f"response_delay={delay_text}"
    )


def _expect_command_error(action, expected_status_name):
    try:
        action()
    except DroneSimCommandError as error:
        if error.status_name != expected_status_name:
            raise RuntimeError(
                f"Expected {expected_status_name}, got {error.status_name}"
            ) from error
        return
    raise RuntimeError(f"Expected {expected_status_name}, command succeeded")


def _place_observer_camera(client, center, args, label):
    if not client.is_camera_active():
        return
    previous_pose = client.get_pose()
    client.set_camera_pose(
        center[0],
        center[1],
        center[2] + args.camera_height,
        previous_pose[5],
        collision_check=False,
    )
    actual_pose = client.set_camera_pitch(args.camera_pitch)
    print(
        f"observer camera centered on {label} at "
        f"({actual_pose[0]:.2f}, {actual_pose[1]:.2f}, "
        f"{actual_pose[2]:.2f}), pitch={actual_pose[3]:.2f}, "
        f"yaw={actual_pose[5]:.2f}"
    )


def _prepare_seed_probe(
    client,
    args,
    firetruck_count,
    blueprint_id=0,
):
    scenario_id = None
    try:
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            args.seed,
            firetruck_count,
            args.pedestrians,
            blueprint_id=blueprint_id,
        )
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
            poll_interval=args.poll_interval,
        )
        _validate_ready(
            ready,
            firetruck_count,
            args.pedestrians,
            args.require_clean_area,
        )
        return ready
    finally:
        if scenario_id is not None:
            client.reset_scenario(scenario_id)


def _validate_seed_isolation(client, args):
    with_firetruck = _prepare_seed_probe(client, args, 1)
    without_firetruck = _prepare_seed_probe(
        client,
        args,
        0,
        blueprint_id=with_firetruck.blueprint_id,
    )
    if without_firetruck.blueprint_id != with_firetruck.blueprint_id:
        raise RuntimeError(
            "Matched instances did not reuse the same blueprint_id"
        )
    if not np.allclose(
        without_firetruck.event_position,
        with_firetruck.event_position,
        rtol=0.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Seed isolation changed the resolved event position"
        )

    left = _entities_by_role(
        without_firetruck,
        ScenarioEntityRole.FLEEING_PEDESTRIAN,
    )
    right = _entities_by_role(
        with_firetruck,
        ScenarioEntityRole.FLEEING_PEDESTRIAN,
    )
    if len(left) != len(right):
        raise RuntimeError(
            "Seed isolation produced different pedestrian counts"
        )
    for index, (left_entity, right_entity) in enumerate(
        zip(left, right)
    ):
        if left_entity.model_hash != right_entity.model_hash:
            raise RuntimeError(
                "Seed isolation changed pedestrian "
                f"{index} model: {left_entity.model_hash} != "
                f"{right_entity.model_hash}"
            )
        if not np.allclose(
            left_entity.position,
            right_entity.position,
            rtol=0.0,
            atol=1e-3,
        ):
            raise RuntimeError(
                "Seed isolation changed pedestrian "
                f"{index} position: {left_entity.position} != "
                f"{right_entity.position}"
            )
        heading_error = abs(
            (
                left_entity.heading
                - right_entity.heading
                + 180.0
            )
            % 360.0
            - 180.0
        )
        if heading_error > 1e-3:
            raise RuntimeError(
                "Seed isolation changed pedestrian "
                f"{index} heading by {heading_error:.6f} degrees"
            )
    print(
        "PASS seed isolation "
        f"blueprint={with_firetruck.blueprint_id} "
        f"seed={args.seed} pedestrians={len(left)} "
        "firetruck_count=0/1"
    )


def _run_cycle(client, args, cycle_index):
    scenario_id = None
    try:
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            args.seed + cycle_index,
            args.firetrucks,
            args.pedestrians,
        )
        _expect_command_error(
            lambda: client.prepare_fire_scenario(
                args.anchor,
                args.seed + cycle_index,
                args.firetrucks,
                args.pedestrians,
            ),
            "SCENARIO_ALREADY_ACTIVE",
        )
        _expect_command_error(
            lambda: client.get_scenario_state(scenario_id + 1),
            "SCENARIO_NOT_FOUND",
        )

        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
            poll_interval=args.poll_interval,
        )
        _validate_ready(
            ready,
            args.firetrucks,
            args.pedestrians,
            args.require_clean_area,
        )
        _place_observer_camera(
            client,
            ready.event_position,
            args,
            "resolved event",
        )
        start = client.start_scenario(scenario_id)
        first = client.get_scenario_state(scenario_id)
        if (
            first.start_game_timer_ms != start.game_timer_ms
            or first.start_frame_count != start.frame_count
        ):
            raise RuntimeError(
                "Start response and scenario snapshot have different zero times"
            )
        snapshots, memory_growth, peak_growth = _sample_running(
            client,
            scenario_id,
            args.observe_seconds,
            args.poll_interval,
            args.rgbd_captures,
            args.capture_timeout_ms,
            args.gta_process,
            args.max_memory_growth_mb,
            args.require_clean_area,
        )
        _summarize_responses(snapshots)
        print(
            f"cycle={cycle_index + 1}/{args.cycles} "
            f"scenario={scenario_id} "
            f"event=({ready.event_position[0]:.2f}, "
            f"{ready.event_position[1]:.2f}, "
            f"{ready.event_position[2]:.2f}) "
            f"removed={ready.removed_pedestrians}p/"
            f"{ready.removed_vehicles}v "
            f"protected={sum(entity.exists for entity in ready.protected_entities)} "
            f"frames={snapshots[0].frame_count}.."
            f"{snapshots[-1].frame_count} "
            f"rgbd={args.rgbd_captures} "
            f"memory_growth={memory_growth:.1f}MiB "
            f"peak_growth={peak_growth:.1f}MiB"
        )
    finally:
        if scenario_id is not None:
            try:
                client.reset_scenario(scenario_id)
            except DroneSimCommandError as error:
                if error.status_name != "SCENARIO_NOT_FOUND":
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Loaded GTA world coordinate near the desired road event",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--observe-seconds", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--rgbd-captures", type=int, default=0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--process-name", default="GTA5.exe")
    parser.add_argument("--max-memory-growth-mb", type=float, default=512.0)
    parser.add_argument(
        "--verify-seed-isolation",
        action="store_true",
        help=(
            "Prepare matched 0/1-firetruck blueprints and require identical "
            "pedestrian models, positions, and headings"
        ),
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=40.0,
        help=(
            "When the scripted camera is active, place it this many meters "
            "above the observer target"
        ),
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=-70.0,
        help="Observer camera pitch in degrees; -90 looks straight down",
    )
    parser.add_argument(
        "--require-clean-area",
        action="store_true",
        help="Fail if preserved mission entities remain in the area",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be within [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be within [0, 32]")
    if args.verify_seed_isolation and args.pedestrians == 0:
        parser.error(
            "--verify-seed-isolation requires at least one pedestrian"
        )
    if args.observe_seconds <= 0:
        parser.error("--observe-seconds must be positive")
    if args.rgbd_captures < 0:
        parser.error("--rgbd-captures must not be negative")
    if args.max_memory_growth_mb < 0:
        parser.error("--max-memory-growth-mb must not be negative")
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor must contain finite values")
    if not math.isfinite(args.camera_height) or args.camera_height <= 0:
        parser.error("--camera-height must be a positive finite value")
    if (
        not math.isfinite(args.camera_pitch)
        or not -90.0 <= args.camera_pitch <= 90.0
    ):
        parser.error("--camera-pitch must be within [-90, 90]")

    client = DroneSimClient()
    args.gta_process = _find_gta_process(args.process_name)
    if args.rgbd_captures:
        client.require_camera_active()
    initial_rss = args.gta_process.memory_info().rss
    try:
        client.teleport_player(*args.anchor)
        if args.verify_seed_isolation:
            _validate_seed_isolation(client, args)
        for cycle_index in range(args.cycles):
            _run_cycle(client, args, cycle_index)
        total_growth_mb = (
            args.gta_process.memory_info().rss - initial_rss
        ) / (1024**2)
        if total_growth_mb > args.max_memory_growth_mb:
            raise RuntimeError(
                f"GTA memory grew by {total_growth_mb:.1f} MiB across all "
                f"cycles; limit is {args.max_memory_growth_mb:.1f} MiB"
            )
    finally:
        client.restore_player()

    print(
        f"PASS cycles={args.cycles} "
        f"firetrucks={args.firetrucks} "
        f"pedestrians={args.pedestrians} "
        f"total_memory_growth={total_growth_mb:.1f}MiB"
    )
    print("No RGB-D or scenario payload was written to disk.")


if __name__ == "__main__":
    main()
