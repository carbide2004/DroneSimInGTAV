"""Evaluation-only source-shadow start pools for Stage 2E collection."""

import hashlib
import json
import math
import os
from pathlib import Path
import time
from dataclasses import asdict, dataclass

import numpy as np

from .dronesim_client import (
    CameraStartBatchStatus,
    LockstepSession,
    ScenarioLifecycle,
    VisibilitySample,
    VisibilitySnapshot,
    VisibilityTarget,
    VisibilityTargetRole,
)
from .feasibility import StaticGoalBudgetAudit
from .task_starts import (
    ObservationSpec,
    TASK_ACTIVITY_RADIUS_METERS,
    TASK_ACTIVITY_VERTICAL_METERS,
    TASK_FORWARD_STEP_METERS,
    TASK_GOAL_VIEW_HEIGHTS_METERS,
    TASK_HORIZON_STEPS,
    TASK_VERTICAL_STEP_METERS,
    TASK_YAW_STEP_DEGREES,
    TaskStartGenerationError,
    assess_visibility,
    virtual_view_matrices,
)


START_POOL_SCHEMA_VERSION = 3
START_POOL_ALGORITHM = "source-shadow-vehicle-only-v6"
SHADOW_DISTANCE_METERS = 120.0
SHADOW_AZIMUTH_STEP_DEGREES = 15.0
SHADOW_REFINEMENT_DEGREES = 7.5
SHADOW_ELEVATIONS_DEGREES = (20.0, 30.0, 40.0, 50.0, 60.0)
SHADOW_MAX_BLOCKER_HORIZONTAL_METERS = 57.0
CANDIDATE_RADII_METERS = (
    42.0, 44.0, 46.0, 48.0, 50.0,
    52.0, 54.0, 56.0, 58.0, 60.0,
)
CANDIDATE_AZIMUTH_OFFSETS_DEGREES = (
    -7.5, -6.0, -4.5, -3.0, -1.5, 0.0,
    1.5, 3.0, 4.5, 6.0, 7.5,
)
CANDIDATE_ALTITUDES_AGL_METERS = (25.0, 35.0, 45.0, 55.0, 60.0)
START_POOL_MAX_ENTRIES = 160
START_POOL_TARGET_ENTRIES = START_POOL_MAX_ENTRIES
START_POOL_REQUIRED_RESERVE_ACTIONS = 15


@dataclass(frozen=True)
class StartPoolTiming:
    shadow_rays: float
    ground_clearance: float
    fire_occlusion: float
    goal_audit: float
    total: float
    anchor_prepare: float = 0.0


@dataclass(frozen=True)
class StartPoolEntry:
    pool_start_id: int
    position: tuple
    ground_z: float
    altitude_agl: float
    radius: float
    bearing_degrees: float
    source_vehicle: object
    fire_envelope: object
    optimistic_goal_actions: int


@dataclass(frozen=True)
class StaticStartPool:
    anchor: tuple
    event_position: tuple
    entries: tuple
    goal_views: tuple
    goal_candidate_count: int
    goal_clear_count: int
    digest: str
    rejection_counts: tuple
    bearing_histogram: tuple
    timing: StartPoolTiming


def _direction(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    horizontal = math.cos(elevation)
    return (
        math.cos(azimuth) * horizontal,
        math.sin(azimuth) * horizontal,
        math.sin(elevation),
    )


def _ray_specs(azimuths):
    return tuple(
        (float(azimuth) % 360.0, elevation, _direction(azimuth, elevation))
        for azimuth in azimuths
        for elevation in SHADOW_ELEVATIONS_DEGREES
    )


def _shadow_hit(origin, ray):
    if not ray.hit:
        return False
    horizontal = math.hypot(
        float(ray.position[0]) - float(origin[0]),
        float(ray.position[1]) - float(origin[1]),
    )
    return horizontal <= SHADOW_MAX_BLOCKER_HORIZONTAL_METERS


def _supported_azimuths(coarse_specs, coarse_snapshot, refined_specs, refined_snapshot):
    by_key = {}
    for spec, ray in zip(coarse_specs, coarse_snapshot.rays):
        by_key[(spec[0], spec[1])] = _shadow_hit(coarse_snapshot.origin, ray)
    coarse_azimuths = tuple(index * SHADOW_AZIMUTH_STEP_DEGREES for index in range(24))
    seeds = set()
    for index, azimuth in enumerate(coarse_azimuths):
        previous = coarse_azimuths[(index - 1) % len(coarse_azimuths)]
        following = coarse_azimuths[(index + 1) % len(coarse_azimuths)]
        if any(
            by_key.get((azimuth, elevation), False)
            and (
                by_key.get((previous, elevation), False)
                or by_key.get((following, elevation), False)
            )
            for elevation in SHADOW_ELEVATIONS_DEGREES
        ):
            seeds.add(azimuth)
    refined_hits = {}
    for spec, ray in zip(refined_specs, refined_snapshot.rays):
        refined_hits[(spec[0], spec[1])] = _shadow_hit(refined_snapshot.origin, ray)
    supported = set(seeds)
    for azimuth in seeds:
        for refined in (
            (azimuth - SHADOW_REFINEMENT_DEGREES) % 360.0,
            (azimuth + SHADOW_REFINEMENT_DEGREES) % 360.0,
        ):
            if any(
                refined_hits.get((refined, elevation), False)
                for elevation in SHADOW_ELEVATIONS_DEGREES
            ):
                supported.add(refined)
    return tuple(sorted(supported))


def _goal_view_yaw(position, event_position):
    dx = float(event_position[0]) - float(position[0])
    dy = float(event_position[1]) - float(position[1])
    if math.hypot(dx, dy) < 1.0e-6:
        return 0.0
    return math.degrees(math.atan2(-dx, dy))


def _goal_lower_bound(position, start_yaw, goal_view):
    goal_position, goal_yaw = goal_view
    horizontal = math.hypot(
        float(position[0]) - float(goal_position[0]),
        float(position[1]) - float(goal_position[1]),
    )
    vertical = abs(float(position[2]) - float(goal_position[2]))
    turns = math.ceil(
        abs((float(goal_yaw) - float(start_yaw) + 180.0) % 360.0 - 180.0)
        / TASK_YAW_STEP_DEGREES
        - 1.0e-12
    )
    return (
        math.ceil(horizontal / TASK_FORWARD_STEP_METERS - 1.0e-12)
        + math.ceil(vertical / TASK_VERTICAL_STEP_METERS - 1.0e-12)
        + turns
        + 1
    )


def _best_goal_audit(entry, start_yaw, pool, horizon_steps):
    candidates = []
    for goal_view in pool.goal_views:
        goal_position, goal_yaw = goal_view
        horizontal = math.hypot(
            float(entry.position[0]) - float(goal_position[0]),
            float(entry.position[1]) - float(goal_position[1]),
        )
        vertical = abs(float(entry.position[2]) - float(goal_position[2]))
        if horizontal > TASK_ACTIVITY_RADIUS_METERS + 1.0e-6:
            continue
        if vertical > TASK_ACTIVITY_VERTICAL_METERS + 1.0e-6:
            continue
        candidates.append((_goal_lower_bound(entry.position, start_yaw, goal_view), goal_view))
    if not candidates:
        return None
    total, goal_view = min(candidates, key=lambda item: (item[0], item[1]))
    remaining = int(horizon_steps) - int(total)
    if remaining < START_POOL_REQUIRED_RESERVE_ACTIONS:
        return None
    return StaticGoalBudgetAudit(
        goal_pose=tuple(goal_view[0]) + (float(goal_view[1]),),
        lower_bound_nonterminal_actions=int(total) - 1,
        stop_actions=1,
        lower_bound_total_actions=int(total),
        required_reserve_actions=START_POOL_REQUIRED_RESERVE_ACTIONS,
        remaining_actions=remaining,
        candidate_ideals=int(pool.goal_candidate_count),
        clear_ideals=int(pool.goal_clear_count),
        observable_ideals=len(pool.goal_views),
    )


def _optimistic_goal_actions(position, goal_views, horizon_steps):
    costs = []
    for goal_view in goal_views:
        goal_position, _goal_yaw = goal_view
        horizontal = math.hypot(
            float(position[0]) - float(goal_position[0]),
            float(position[1]) - float(goal_position[1]),
        )
        vertical = abs(float(position[2]) - float(goal_position[2]))
        if horizontal > TASK_ACTIVITY_RADIUS_METERS + 1.0e-6:
            continue
        if vertical > TASK_ACTIVITY_VERTICAL_METERS + 1.0e-6:
            continue
        costs.append(
            math.ceil(horizontal / TASK_FORWARD_STEP_METERS - 1.0e-12)
            + math.ceil(vertical / TASK_VERTICAL_STEP_METERS - 1.0e-12)
            + 1
        )
    if not costs:
        return None
    best = min(costs)
    if best > int(horizon_steps) - START_POOL_REQUIRED_RESERVE_ACTIONS:
        return None
    return int(best)


def _pool_id(anchor, event_position, position):
    payload = np.asarray((*anchor, *event_position, *position), dtype="<f8").tobytes()
    value = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=b"StartPool").digest(),
        "little",
    )
    return value or 1


def _entry_key(entry):
    return (entry.bearing_degrees, entry.radius, entry.altitude_agl, entry.pool_start_id)


def _farthest_entries(entries, maximum):
    entries = tuple(sorted(entries, key=_entry_key))
    if len(entries) <= maximum:
        return entries
    points = np.asarray([entry.position for entry in entries], dtype=np.float64)
    center = np.mean(points, axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    minimum = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < maximum:
        minimum[selected] = -1.0
        next_index = int(np.argmax(minimum))
        selected.append(next_index)
        minimum = np.minimum(
            minimum,
            np.linalg.norm(points - points[next_index], axis=1),
        )
    return tuple(entries[index] for index in selected)


def _pool_digest(
    anchor,
    event_position,
    entries,
    goal_views=(),
    goal_candidate_count=0,
    goal_clear_count=0,
):
    digest = hashlib.blake2b(digest_size=16, person=b"PoolDigest")
    digest.update(START_POOL_ALGORITHM.encode("ascii"))
    digest.update(np.asarray((*anchor, *event_position), dtype="<f8").tobytes())
    digest.update(np.asarray(
        (goal_candidate_count, goal_clear_count), dtype="<u4"
    ).tobytes())
    for goal_position, goal_yaw in goal_views:
        digest.update(np.asarray((*goal_position, goal_yaw), dtype="<f8").tobytes())
    for entry in sorted(entries, key=lambda item: item.pool_start_id):
        digest.update(int(entry.pool_start_id).to_bytes(8, "little"))
        digest.update(np.asarray(
            (
                *entry.position,
                entry.ground_z,
                entry.altitude_agl,
                entry.radius,
                entry.bearing_degrees,
            ),
            dtype="<f8",
        ).tobytes())
        digest.update(int(entry.optimistic_goal_actions).to_bytes(4, "little"))
        target = entry.source_vehicle
        digest.update(int(target.stable_id).to_bytes(8, "little"))
        digest.update(int(target.role).to_bytes(4, "little"))
        for sample in target.samples:
            digest.update(np.asarray(sample.position, dtype="<f8").tobytes())
            digest.update(bytes((int(sample.clear_line_of_sight),)))
    return digest.hexdigest()


def build_static_start_pool(
    client,
    session,
    scenario,
    minimum_entries,
    observation_spec,
    horizon_steps=TASK_HORIZON_STEPS,
    progress_callback=None,
):
    if not isinstance(session, LockstepSession):
        raise TypeError("session must be a LockstepSession")
    if scenario.lifecycle not in (ScenarioLifecycle.READY, ScenarioLifecycle.RUNNING):
        raise ValueError("Static start pools require READY or RUNNING scenarios")
    minimum_entries = int(minimum_entries)
    if minimum_entries <= 0 or minimum_entries > START_POOL_MAX_ENTRIES:
        raise ValueError(
            "minimum_entries must be in [1, "
            f"{START_POOL_MAX_ENTRIES}]"
        )
    if not isinstance(observation_spec, ObservationSpec):
        raise TypeError("observation_spec must be an ObservationSpec")
    clock = session.refresh()
    started = time.perf_counter()
    phase = time.perf_counter()
    coarse_azimuths = tuple(index * SHADOW_AZIMUTH_STEP_DEGREES for index in range(24))
    coarse_specs = _ray_specs(coarse_azimuths)
    coarse = client.probe_fire_shadow_batch(
        scenario.scenario_id,
        session.session_id,
        [item[2] for item in coarse_specs],
        timeout=30.0,
    )
    if (coarse.step_index != clock.step_index or
            coarse.game_timer_ms != clock.game_timer_ms):
        raise TaskStartGenerationError("Fire-shadow batch changed lockstep instant")
    seed_azimuths = sorted({
        spec[0]
        for spec, ray in zip(coarse_specs, coarse.rays)
        if _shadow_hit(coarse.origin, ray)
    })
    refined_azimuths = sorted({
        value
        for azimuth in seed_azimuths
        for value in (
            (azimuth - SHADOW_REFINEMENT_DEGREES) % 360.0,
            (azimuth + SHADOW_REFINEMENT_DEGREES) % 360.0,
        )
    })
    refined_specs = _ray_specs(refined_azimuths)
    if refined_specs:
        refined = client.probe_fire_shadow_batch(
            scenario.scenario_id,
            session.session_id,
            [item[2] for item in refined_specs],
            timeout=30.0,
        )
    else:
        refined = coarse.__class__(
            coarse.scenario_id, coarse.lockstep_session_id, coarse.step_index,
            coarse.game_timer_ms, coarse.frame_count, coarse.origin, (),
        )
    supported = _supported_azimuths(coarse_specs, coarse, refined_specs, refined)
    shadow_seconds = time.perf_counter() - phase
    if progress_callback:
        progress_callback(
            f"anchor shadow rays coarse={len(coarse_specs)} refined={len(refined_specs)} "
            f"supported_azimuths={len(supported)} time={shadow_seconds:.1f}s"
        )

    raw_candidates = []
    event = tuple(float(value) for value in scenario.event_position)
    candidate_azimuths = tuple(sorted({
        round((azimuth + offset) % 360.0, 6)
        for azimuth in supported
        for offset in CANDIDATE_AZIMUTH_OFFSETS_DEGREES
    }))
    for radius in CANDIDATE_RADII_METERS:
        for altitude in CANDIDATE_ALTITUDES_AGL_METERS:
            for azimuth in candidate_azimuths:
                radians = math.radians(azimuth)
                x = event[0] + math.cos(radians) * radius
                y = event[1] + math.sin(radians) * radius
                raw_candidates.append((azimuth, radius, altitude, x, y))
    if not raw_candidates:
        raise TaskStartGenerationError(
            "ANCHOR_UNSUITABLE: no consistent source-shadow sector"
        )
    phase = time.perf_counter()
    probed = []
    for offset in range(0, len(raw_candidates), 256):
        chunk = raw_candidates[offset : offset + 256]
        snapshot = client.probe_camera_start_batch(
            session.session_id,
            [(item[3], item[4], item[2]) for item in chunk],
            timeout=30.0,
        )
        if snapshot.step_index != clock.step_index or snapshot.game_timer_ms != clock.game_timer_ms:
            raise TaskStartGenerationError("Camera-start batch changed lockstep instant")
        probed.extend(zip(chunk, snapshot.items))
    ground_seconds = time.perf_counter() - phase
    if progress_callback:
        progress_callback(
            f"anchor ground clearance candidates={len(raw_candidates)} "
            f"time={ground_seconds:.1f}s"
        )
    rejection = {
        "ground_not_found": 0,
        "space_blocked": 0,
        "source_vehicle_visible": 0,
        "goal_budget": 0,
        "not_audited": 0,
    }
    clear = []
    for candidate, item in probed:
        if item.status == CameraStartBatchStatus.GROUND_NOT_FOUND:
            rejection["ground_not_found"] += 1
        elif item.status == CameraStartBatchStatus.SPACE_BLOCKED:
            rejection["space_blocked"] += 1
        else:
            clear.append((candidate, item))

    phase = time.perf_counter()
    goal_probe = client.probe_camera_start_batch(
        session.session_id,
        [
            (event[0], event[1], height)
            for height in TASK_GOAL_VIEW_HEIGHTS_METERS
        ],
        timeout=30.0,
    )
    if (goal_probe.step_index != clock.step_index or
            goal_probe.game_timer_ms != clock.game_timer_ms):
        raise TaskStartGenerationError("Goal-view probe changed lockstep instant")
    clear_goal_items = tuple(
        item for item in goal_probe.items
        if item.status == CameraStartBatchStatus.OK
    )
    goal_views = []
    if clear_goal_items:
        goal_occlusion = client.query_fire_occlusion_batch(
            scenario.scenario_id,
            session.session_id,
            [item.position for item in clear_goal_items],
            timeout=30.0,
        )
        if (goal_occlusion.step_index != clock.step_index or
                goal_occlusion.game_timer_ms != clock.game_timer_ms):
            raise TaskStartGenerationError("Goal visibility changed lockstep instant")
        for item, case in zip(clear_goal_items, goal_occlusion.cases):
            yaw = _goal_view_yaw(item.position, event)
            visibility = VisibilitySnapshot(
                scenario_id=scenario.scenario_id,
                lockstep_session_id=clock.session_id,
                step_index=clock.step_index,
                game_timer_ms=clock.game_timer_ms,
                frame_count=goal_occlusion.frame_count,
                camera_center=item.position,
                targets=(case.source_vehicle, case.fire_envelope),
            )
            assessment = assess_visibility(
                visibility,
                virtual_view_matrices(item.position, yaw, observation_spec),
                observation_spec,
            )
            if assessment.event_task_observable:
                goal_views.append((tuple(item.position), float(yaw)))
    if not goal_views:
        raise TaskStartGenerationError(
            "ANCHOR_UNSUITABLE: no clear task-observable fire-source goal view"
        )

    eligible = []
    for candidate, item in clear:
        goal_actions = _optimistic_goal_actions(
            item.position, goal_views, horizon_steps
        )
        if goal_actions is None:
            rejection["goal_budget"] += 1
            continue
        eligible.append((candidate, item, goal_actions))
    goal_seconds = time.perf_counter() - phase
    if progress_callback:
        progress_callback(
            f"anchor goal audit clear={len(clear)} eligible={len(eligible)} "
            f"goal_views={len(goal_views)} time={goal_seconds:.1f}s"
        )

    anchor = tuple(float(value) for value in scenario.requested_anchor)
    target_entries = min(
        START_POOL_MAX_ENTRIES,
        max(
            START_POOL_TARGET_ENTRIES,
            minimum_entries + 8,
            minimum_entries * 2,
        ),
    )
    entries = []
    audited = 0
    phase = time.perf_counter()
    batch_size = 16
    total_batches = math.ceil(len(eligible) / batch_size) if eligible else 0
    for batch_index, offset in enumerate(
            range(0, len(eligible), batch_size), start=1):
        chunk = eligible[offset : offset + batch_size]
        snapshot = client.query_fire_occlusion_batch(
            scenario.scenario_id,
            session.session_id,
            [item.position for _candidate, item, _actions in chunk],
            timeout=30.0,
        )
        if (snapshot.step_index != clock.step_index or
                snapshot.game_timer_ms != clock.game_timer_ms):
            raise TaskStartGenerationError(
                "Fire-occlusion batch changed lockstep instant"
            )
        for ((candidate, item, goal_actions), occlusion) in zip(
                chunk, snapshot.cases):
            audited += 1
            if any(
                    sample.clear_line_of_sight
                    for sample in occlusion.source_vehicle.samples):
                rejection["source_vehicle_visible"] += 1
                continue
            azimuth, radius, altitude, _x, _y = candidate
            entries.append(StartPoolEntry(
                pool_start_id=_pool_id(anchor, event, item.position),
                position=tuple(float(value) for value in item.position),
                ground_z=float(item.ground_z),
                altitude_agl=float(altitude),
                radius=float(radius),
                bearing_degrees=float(azimuth),
                source_vehicle=occlusion.source_vehicle,
                fire_envelope=occlusion.fire_envelope,
                optimistic_goal_actions=goal_actions,
            ))
        if progress_callback:
            progress_callback(
                f"anchor fire occlusion batch={batch_index}/{total_batches} "
                f"audited={audited}/{len(eligible)} valid={len(entries)}/"
                f"{target_entries} elapsed={time.perf_counter() - phase:.1f}s"
            )
        if len(entries) >= target_entries:
            break
    rejection["not_audited"] = len(eligible) - audited
    occlusion_seconds = time.perf_counter() - phase
    entries = _farthest_entries(entries, START_POOL_MAX_ENTRIES)
    histogram = [0] * 8
    for entry in entries:
        histogram[int((entry.bearing_degrees % 360.0) // 45.0)] += 1
    digest = _pool_digest(
        anchor,
        event,
        entries,
        goal_views,
        len(TASK_GOAL_VIEW_HEIGHTS_METERS),
        len(clear_goal_items),
    )
    timing = StartPoolTiming(
        shadow_seconds,
        ground_seconds,
        occlusion_seconds,
        goal_seconds,
        time.perf_counter() - started,
    )
    pool = StaticStartPool(
        anchor,
        event,
        tuple(entries),
        tuple(goal_views),
        len(TASK_GOAL_VIEW_HEIGHTS_METERS),
        len(clear_goal_items),
        digest,
        tuple(sorted(rejection.items())),
        tuple(histogram),
        timing,
    )
    if len(entries) < minimum_entries:
        error = TaskStartGenerationError(
            f"ANCHOR_UNSUITABLE: static hidden-start pool has {len(entries)} "
            f"entries; requires {minimum_entries}"
        )
        error.start_pool = pool
        raise error
    return pool


def _merged_visibility(entry, batch_cases, scenario, clock):
    return VisibilitySnapshot(
        scenario_id=scenario.scenario_id,
        lockstep_session_id=clock.session_id,
        step_index=clock.step_index,
        game_timer_ms=clock.game_timer_ms,
        frame_count=clock.frame_count,
        camera_center=entry.position,
        targets=(entry.source_vehicle, entry.fire_envelope) + tuple(
            case.target for case in batch_cases
        ),
    )


def _target_to_json(target):
    return {
        "stable_id": int(target.stable_id),
        "gta_handle": 0,
        "role": int(target.role),
        "samples": [
            {
                "position": [float(value) for value in sample.position],
                "clear": bool(sample.clear_line_of_sight),
            }
            for sample in target.samples
        ],
    }


def pool_to_json(pool):
    payload = {
        "schema_version": START_POOL_SCHEMA_VERSION,
        "algorithm": START_POOL_ALGORITHM,
        "anchor": list(pool.anchor),
        "event_position": list(pool.event_position),
        "digest": pool.digest,
        "rejection_counts": dict(pool.rejection_counts),
        "bearing_histogram": list(pool.bearing_histogram),
        "timing": asdict(pool.timing),
        "goal_views": [
            {"position": list(position), "yaw": yaw}
            for position, yaw in pool.goal_views
        ],
        "goal_candidate_count": pool.goal_candidate_count,
        "goal_clear_count": pool.goal_clear_count,
        "entries": [
            {
                "pool_start_id": entry.pool_start_id,
                "position": list(entry.position),
                "ground_z": entry.ground_z,
                "altitude_agl": entry.altitude_agl,
                "radius": entry.radius,
                "bearing_degrees": entry.bearing_degrees,
                "optimistic_goal_actions": entry.optimistic_goal_actions,
                "source_vehicle": _target_to_json(entry.source_vehicle),
                "fire_envelope": _target_to_json(entry.fire_envelope),
            }
            for entry in pool.entries
        ],
    }
    return payload


def write_pool(path, pool):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(pool_to_json(pool), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)



def _target_from_json(payload):
    role = VisibilityTargetRole(int(payload["role"]))
    return VisibilityTarget(
        stable_id=int(payload["stable_id"]),
        gta_handle=0,
        role=role,
        samples=tuple(
            VisibilitySample(
                tuple(float(value) for value in sample["position"]),
                bool(sample["clear"]),
                0,
            )
            for sample in payload["samples"]
        ),
    )


def load_pool(path):
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("schema_version") != START_POOL_SCHEMA_VERSION
        or payload.get("algorithm") != START_POOL_ALGORITHM
    ):
        raise RuntimeError("Unsupported start-pool schema or algorithm")
    entries = tuple(
        StartPoolEntry(
            pool_start_id=int(item["pool_start_id"]),
            position=tuple(float(value) for value in item["position"]),
            ground_z=float(item["ground_z"]),
            altitude_agl=float(item["altitude_agl"]),
            radius=float(item["radius"]),
            bearing_degrees=float(item["bearing_degrees"]),
            source_vehicle=_target_from_json(item["source_vehicle"]),
            fire_envelope=_target_from_json(item["fire_envelope"]),
            optimistic_goal_actions=int(item["optimistic_goal_actions"]),
        )
        for item in payload["entries"]
    )
    timing = StartPoolTiming(**{
        name: float(value) for name, value in payload["timing"].items()
    })
    pool = StaticStartPool(
        anchor=tuple(float(value) for value in payload["anchor"]),
        event_position=tuple(float(value) for value in payload["event_position"]),
        entries=entries,
        goal_views=tuple(
            (tuple(float(value) for value in item["position"]), float(item["yaw"]))
            for item in payload["goal_views"]
        ),
        goal_candidate_count=int(payload["goal_candidate_count"]),
        goal_clear_count=int(payload["goal_clear_count"]),
        digest=str(payload["digest"]),
        rejection_counts=tuple(sorted(
            (str(name), int(value))
            for name, value in payload["rejection_counts"].items()
        )),
        bearing_histogram=tuple(int(value) for value in payload["bearing_histogram"]),
        timing=timing,
    )
    actual = _pool_digest(
        pool.anchor,
        pool.event_position,
        pool.entries,
        pool.goal_views,
        pool.goal_candidate_count,
        pool.goal_clear_count,
    )
    if actual != pool.digest:
        raise RuntimeError(
            f"ANCHOR_POOL_MISMATCH: expected digest {pool.digest}, got {actual}"
        )
    return pool


def revalidate_static_start_pool(client, session, scenario, pool):
    if tuple(round(value, 3) for value in scenario.event_position) != tuple(
        round(value, 3) for value in pool.event_position
    ):
        raise RuntimeError("ANCHOR_POOL_MISMATCH: event position changed")
    clock = session.refresh()
    returned = []
    for offset in range(0, len(pool.entries), 32):
        entries = pool.entries[offset : offset + 32]
        snapshot = client.query_fire_occlusion_batch(
            scenario.scenario_id,
            session.session_id,
            [entry.position for entry in entries],
            timeout=30.0,
        )
        if snapshot.step_index != clock.step_index or snapshot.game_timer_ms != clock.game_timer_ms:
            raise RuntimeError("ANCHOR_POOL_MISMATCH: lockstep instant changed")
        returned.extend(snapshot.cases)
    for entry, actual in zip(pool.entries, returned):
        source_mask = tuple(
            sample.clear_line_of_sight for sample in actual.source_vehicle.samples
        )
        expected_source = tuple(
            sample.clear_line_of_sight for sample in entry.source_vehicle.samples
        )
        if source_mask != expected_source:
            raise RuntimeError(
                "ANCHOR_POOL_MISMATCH: source-vehicle occlusion changed for "
                f"pool_start_id={entry.pool_start_id}"
            )
    return True
