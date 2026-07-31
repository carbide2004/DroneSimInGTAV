"""Evaluation-only Stage 2D joint-witness search.

The search uses privileged scenario and visibility truth. It operates directly
on the seven fixed research actions and must never be exposed as an agent
observation.
"""

import hashlib
import heapq
import math
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .dronesim_client import (
    ScenarioEntityRole,
    ScenarioTaskState,
    TargetVisibilityCase,
    VisibilityTargetRole,
)
from .research_actions import (
    AscendAction,
    DescendAction,
    ForwardAction,
    HoldAction,
    StopAction,
    TurnLeftAction,
    TurnRightAction,
)
from .task_starts import (
    TASK_FORWARD_STEP_METERS,
    TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS,
    TASK_VERTICAL_STEP_METERS,
    TASK_YAW_STEP_DEGREES,
    GeneratedTaskStart,
    _assess_target_view,
    pair_view_matrices,
    virtual_view_matrices,
)


SEARCH_FRONTIER_BATCH = 64
GEOMETRY_MOTIONS_PER_BATCH = 128
SEARCH_MAX_EXPANDED_GOAL = 8192
SEARCH_MAX_EXPANDED_CUE_PER_TARGET = 256
SEARCH_RESULT_LIMIT = 4
SEARCH_RESULTS_PER_CUE_TARGET = 2
VIEW_QUERY_MAX_HEURISTIC_ACTIONS = 3
CUE_RESPONDERS_PER_SEARCH = 8
CUE_SEARCH_STEP_STRIDE = 4
SEARCH_FORWARD_DETOUR_DEGREES = 60.0
REQUIRED_ACTION_MARGIN = 5
_POSITION_KEY_METERS = 0.5
_VIEW_MIN_DISTANCE_METERS = 6.0
_VIEW_MAX_DISTANCE_METERS = 80.0
_IDEAL_HEIGHTS_METERS = (20.0, 30.0, 40.0)


class FeasibilityStatus(IntEnum):
    JOINT_WITNESS_WITH_MARGIN = 1
    JOINT_WITNESS_TIGHT = 2
    NO_JOINT_WITNESS_IN_SEARCH = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class CueWitness:
    stable_id: int
    role: ScenarioEntityRole
    first_step: int
    second_step: int
    first_pose: tuple
    second_pose: tuple
    transition_action: object
    horizontal_displacement_m: float
    direction_cosine: float


@dataclass(frozen=True)
class GoalWitness:
    stable_id: int
    pose: tuple


@dataclass(frozen=True)
class JointPathWitness:
    cue: CueWitness
    goal: GoalWitness
    actions: tuple
    forward_actions: int
    ascend_actions: int
    descend_actions: int
    turn_left_actions: int
    turn_right_actions: int
    hold_actions: int
    stop_actions: int
    total_actions: int
    remaining_actions: int


@dataclass(frozen=True)
class SpatiotemporalFeasibilityReport:
    start_id: int
    visibility_stratum: object
    status: FeasibilityStatus
    cue_path_found: bool
    goal_view_path_found: bool
    cue_then_goal_path_found: bool
    search_digest: str
    searched_states: int
    checked_motion_edges: int
    queried_steps: int
    minimum_ordered_actions: int | None
    cue_diagnostics: tuple
    phase_seconds: tuple
    witness: JointPathWitness | None
    message: str


@dataclass(frozen=True)
class _SearchNode:
    position: tuple
    yaw: float
    cost: int
    parent: int
    action: object | None


@dataclass(frozen=True)
class _ObservationPath:
    stable_id: int
    role: ScenarioEntityRole
    actions: tuple
    position: tuple
    yaw: float


def _angle_delta(target, source):
    return (float(target) - float(source) + 180.0) % 360.0 - 180.0


def _yaw_toward(origin, target):
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return math.degrees(math.atan2(-dx, dy))


def _nearest_step_count(value, step):
    magnitude = int(
        math.floor(abs(float(value)) / float(step) + 0.5)
    )
    return magnitude if value >= 0.0 else -magnitude


def _apply_action_pose(position, yaw_degrees, action):
    position = np.asarray(position, dtype=np.float64).copy()
    yaw_degrees = float(yaw_degrees)
    if isinstance(action, ForwardAction):
        yaw = math.radians(yaw_degrees)
        position += np.asarray(
            (
                -math.sin(yaw) * TASK_FORWARD_STEP_METERS,
                math.cos(yaw) * TASK_FORWARD_STEP_METERS,
                0.0,
            )
        )
    elif isinstance(action, AscendAction):
        position[2] += TASK_VERTICAL_STEP_METERS
    elif isinstance(action, DescendAction):
        position[2] -= TASK_VERTICAL_STEP_METERS
    elif isinstance(action, TurnLeftAction):
        yaw_degrees += TASK_YAW_STEP_DEGREES
    elif isinstance(action, TurnRightAction):
        yaw_degrees -= TASK_YAW_STEP_DEGREES
    elif not isinstance(action, (HoldAction, StopAction)):
        raise TypeError(f"Unsupported research action {action!r}")
    yaw_degrees = (yaw_degrees + 180.0) % 360.0 - 180.0
    return tuple(float(value) for value in position), yaw_degrees


def _target_observable(target, center, yaw, observation_spec):
    matrices = virtual_view_matrices(
        center,
        yaw,
        observation_spec,
    )
    return any(
        _assess_target_view(
            target,
            matrices[name],
            observation_spec,
        ).task_observable
        for name in ("oblique", "nadir")
    )


def _scenario_entity_map(snapshot):
    return {
        entity.stable_id: entity
        for entity in snapshot.entities
        if entity.exists
    }


def _response_entities(snapshot):
    return tuple(
        entity
        for entity in snapshot.entities
        if entity.exists
        and entity.role
        in (
            ScenarioEntityRole.FIRE_TRUCK,
            ScenarioEntityRole.FLEEING_PEDESTRIAN,
        )
    )


def _selected_response_entities(snapshot, task_step):
    entities = sorted(
        _response_entities(snapshot),
        key=lambda entity: (
            entity.role != ScenarioEntityRole.FIRE_TRUCK,
            entity.stable_id,
        ),
    )
    trucks = [
        entity
        for entity in entities
        if entity.role == ScenarioEntityRole.FIRE_TRUCK
    ]
    pedestrians = [
        entity
        for entity in entities
        if entity.role == ScenarioEntityRole.FLEEING_PEDESTRIAN
    ]
    selected = trucks[:CUE_RESPONDERS_PER_SEARCH]
    remaining = CUE_RESPONDERS_PER_SEARCH - len(selected)
    if pedestrians and remaining > 0:
        offset = (
            (task_step // CUE_SEARCH_STEP_STRIDE) * remaining
        ) % len(pedestrians)
        rotated = pedestrians[offset:] + pedestrians[:offset]
        selected.extend(rotated[:remaining])
    return tuple(selected)


def _source_entity(snapshot):
    sources = [
        entity
        for entity in snapshot.entities
        if entity.exists
        and entity.role == ScenarioEntityRole.FIRE_SOURCE_VEHICLE
    ]
    if len(sources) != 1:
        raise RuntimeError(
            f"Expected one live fire source, found {len(sources)}"
        )
    return sources[0]


def _direction_cosine(previous, current, event_position):
    displacement = np.asarray(current.position[:2], dtype=np.float64) - (
        np.asarray(previous.position[:2], dtype=np.float64)
    )
    length = float(np.linalg.norm(displacement))
    if length <= 1.0e-9:
        return -1.0
    event = np.asarray(event_position[:2], dtype=np.float64)
    if previous.role == ScenarioEntityRole.FIRE_TRUCK:
        expected = event - np.asarray(
            previous.position[:2],
            dtype=np.float64,
        )
    elif previous.role == ScenarioEntityRole.FLEEING_PEDESTRIAN:
        expected = np.asarray(
            previous.position[:2],
            dtype=np.float64,
        ) - event
    else:
        return -1.0
    expected_length = float(np.linalg.norm(expected))
    if expected_length <= 1.0e-9:
        return -1.0
    return float(
        np.dot(displacement, expected)
        / (length * expected_length)
    )


class SpatiotemporalFeasibilityAuditor:
    def __init__(
        self,
        client,
        lockstep,
        scenario_id,
        generated_start,
        search_timeout_seconds=120.0,
        progress_callback=None,
    ):
        if not isinstance(generated_start, GeneratedTaskStart):
            raise TypeError(
                "generated_start must be a GeneratedTaskStart"
            )
        self.client = client
        self.lockstep = lockstep
        self.scenario_id = int(scenario_id)
        self.generated_start = generated_start
        self.blueprint = generated_start.blueprint
        self.spec = self.blueprint.observation_spec
        self.horizon_steps = int(
            self.blueprint.action_spec.horizon_steps
        )
        self.search_timeout_seconds = float(search_timeout_seconds)
        if (
            not math.isfinite(self.search_timeout_seconds)
            or self.search_timeout_seconds <= 0.0
        ):
            raise ValueError(
                "search_timeout_seconds must be finite and positive"
            )
        self.progress_callback = progress_callback
        self._search_started = None
        self._last_progress = None
        self.search_positions = []
        self._searched_keys = set()
        self._motion_cache = {}
        self._checked_motion_edges = 0
        self._cue_diagnostics = {
            "first_view_proposed": 0,
            "first_view_observable": 0,
            "transition_cases": 0,
            "entity_missing": 0,
            "task_inactive": 0,
            "displacement_below_minimum": 0,
            "direction_cosine_below_0_5": 0,
            "second_view_not_observable": 0,
            "valid_transition": 0,
        }
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"fixed-action-lattice-v1")
        digest.update(
            np.asarray(
                self.blueprint.absolute_pose[:3],
                dtype="<f4",
            ).tobytes()
        )
        digest.update(
            np.asarray(
                (
                    self.blueprint.absolute_pose[5],
                    TASK_FORWARD_STEP_METERS,
                    TASK_VERTICAL_STEP_METERS,
                    TASK_YAW_STEP_DEGREES,
                ),
                dtype="<f4",
            ).tobytes()
        )
        self._search_digest = digest.hexdigest()

    def _check_search_time(self, phase, expanded=0, queued=0):
        now = time.perf_counter()
        if self._search_started is None:
            self._search_started = now
            self._last_progress = now
        elapsed = now - self._search_started
        if elapsed >= self.search_timeout_seconds:
            raise TimeoutError(
                "ACTION_SEARCH_TIMEOUT: fixed-action search exceeded "
                f"{self.search_timeout_seconds:.1f}s during {phase}; "
                f"expanded={expanded}, queued={queued}, "
                f"states={len(self._searched_keys)}, "
                f"motion_edges={self._checked_motion_edges}"
            )
        if (
            self.progress_callback is not None
            and now - self._last_progress >= 10.0
        ):
            self.progress_callback(
                "action search "
                f"phase={phase} elapsed={elapsed:.1f}s "
                f"expanded={expanded} queued={queued} "
                f"states={len(self._searched_keys)} "
                f"motion_edges={self._checked_motion_edges}"
            )
            self._last_progress = now

    def _require_instant(self, snapshot, reference):
        if (
            snapshot.lockstep_session_id != reference.session_id
            or snapshot.step_index != reference.step_index
            or snapshot.game_timer_ms != reference.game_timer_ms
        ):
            raise RuntimeError(
                "Evaluation batch does not belong to the frozen instant"
            )

    def _query_cases(self, cases, clock):
        if not cases:
            return ()
        result = self.client.query_target_visibility_batch(
            self.scenario_id,
            self.lockstep.session_id,
            cases,
            timeout=30.0,
        )
        self._require_instant(result, clock)
        return result.cases

    def _state_key(self, position, yaw):
        origin = self.blueprint.absolute_pose
        return (
            round(
                (float(position[0]) - origin[0])
                / _POSITION_KEY_METERS
            ),
            round(
                (float(position[1]) - origin[1])
                / _POSITION_KEY_METERS
            ),
            round(
                (float(position[2]) - origin[2])
                / TASK_VERTICAL_STEP_METERS
            ),
            _nearest_step_count(
                _angle_delta(yaw, origin[5]),
                TASK_YAW_STEP_DEGREES,
            )
            % round(360.0 / TASK_YAW_STEP_DEGREES),
        )

    @staticmethod
    def _motion_key(position, yaw, action):
        return (
            *(round(float(value), 4) for value in position),
            round(float(yaw), 4),
            type(action).__name__,
        )

    def _ideal_poses(self, targets):
        base_yaw = float(self.blueprint.absolute_pose[5])
        ideals = []
        yaw_bins = round(360.0 / TASK_YAW_STEP_DEGREES)
        for target in targets:
            point = np.asarray(target.position, dtype=np.float64)
            for yaw_index in range(yaw_bins):
                yaw = (
                    base_yaw
                    + yaw_index * TASK_YAW_STEP_DEGREES
                )
                radians = math.radians(yaw)
                forward = np.asarray(
                    (-math.sin(radians), math.cos(radians))
                )
                for height in _IDEAL_HEIGHTS_METERS:
                    center = (
                        point[0] - forward[0] * height,
                        point[1] - forward[1] * height,
                        point[2] + height,
                    )
                    if self.blueprint.contains_world_position(center):
                        ideals.append(
                            (
                                target.stable_id,
                                target.role,
                                center,
                                (
                                    (yaw + 180.0) % 360.0
                                    - 180.0
                                ),
                            )
                        )
            for height in _IDEAL_HEIGHTS_METERS:
                center = (
                    point[0],
                    point[1],
                    point[2] + height,
                )
                if self.blueprint.contains_world_position(center):
                    ideals.append(
                        (
                            target.stable_id,
                            target.role,
                            center,
                            base_yaw,
                        )
                    )
        return tuple(ideals)

    @staticmethod
    def _heuristic(position, yaw, ideals):
        best = None
        position = np.asarray(position, dtype=np.float64)
        for stable_id, role, center, ideal_yaw in ideals:
            delta = np.asarray(center, dtype=np.float64) - position
            horizontal_distance = float(np.linalg.norm(delta[:2]))
            movement = math.ceil(
                horizontal_distance / TASK_FORWARD_STEP_METERS
                - 1.0e-12
            )
            vertical = math.ceil(
                abs(float(delta[2]))
                / TASK_VERTICAL_STEP_METERS
                - 1.0e-12
            )
            if horizontal_distance > TASK_FORWARD_STEP_METERS * 0.5:
                travel_yaw = _yaw_toward(position, center)
                turns = math.ceil(
                    abs(_angle_delta(travel_yaw, yaw))
                    / TASK_YAW_STEP_DEGREES
                    - 1.0e-12
                ) + math.ceil(
                    abs(_angle_delta(ideal_yaw, travel_yaw))
                    / TASK_YAW_STEP_DEGREES
                    - 1.0e-12
                )
            else:
                travel_yaw = float(ideal_yaw)
                turns = math.ceil(
                    abs(_angle_delta(ideal_yaw, yaw))
                    / TASK_YAW_STEP_DEGREES
                    - 1.0e-12
                )
            rank = (
                int(movement + vertical + turns),
                stable_id,
                int(role),
            )
            if best is None or rank < best[0]:
                best = (
                    rank,
                    stable_id,
                    role,
                    tuple(float(value) for value in center),
                    float(ideal_yaw),
                    float(travel_yaw),
                )
        return best

    @staticmethod
    def _potentially_visible(position, yaw, target):
        delta = np.asarray(target.position, dtype=np.float64) - np.asarray(
            position,
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(delta))
        if (
            distance < _VIEW_MIN_DISTANCE_METERS
            or distance > _VIEW_MAX_DISTANCE_METERS
            or delta[2] >= 0.0
        ):
            return False
        horizontal = float(np.linalg.norm(delta[:2]))
        vertical = -float(delta[2])
        nadir_candidate = horizontal <= max(10.0, vertical * 0.6)
        if horizontal <= 1.0e-9:
            return nadir_candidate
        yaw_error = abs(
            _angle_delta(
                _yaw_toward(position, target.position),
                yaw,
            )
        )
        oblique_candidate = (
            yaw_error <= 55.0
            and 0.3 <= vertical / horizontal <= 3.0
        )
        return nadir_candidate or oblique_candidate

    @staticmethod
    def _reconstruct(nodes, index):
        actions = []
        while nodes[index].parent >= 0:
            actions.append(nodes[index].action)
            index = nodes[index].parent
        actions.reverse()
        return tuple(actions)

    def _search_observation_paths(
        self,
        targets,
        start_position,
        start_yaw,
        max_actions,
        clock,
        max_expanded,
        phase,
    ):
        targets = tuple(
            sorted(targets, key=lambda item: (item.role, item.stable_id))
        )
        if not targets or max_actions < 0:
            return ()
        ideals = self._ideal_poses(targets)
        if not ideals:
            return ()
        target_map = {
            target.stable_id: target
            for target in targets
        }
        nodes = [
            _SearchNode(
                tuple(float(value) for value in start_position),
                float(start_yaw),
                0,
                -1,
                None,
            )
        ]
        best_cost = {
            self._state_key(start_position, start_yaw): 0
        }
        heuristic = self._heuristic(start_position, start_yaw, ideals)
        queue = [(heuristic[0][0], 0, 0)]
        expanded = 0
        results = []

        def enqueue(position, yaw, cost, parent, action):
            if cost > max_actions:
                return
            key = self._state_key(position, yaw)
            if cost >= best_cost.get(key, math.inf):
                return
            estimate = self._heuristic(position, yaw, ideals)
            if estimate is None:
                return
            best_cost[key] = cost
            node_index = len(nodes)
            nodes.append(
                _SearchNode(
                    tuple(float(value) for value in position),
                    float(yaw),
                    int(cost),
                    int(parent),
                    action,
                )
            )
            heapq.heappush(
                queue,
                (
                    cost + estimate[0][0],
                    cost,
                    node_index,
                ),
            )
            if key not in self._searched_keys:
                self._searched_keys.add(key)
                self.search_positions.append(
                    tuple(float(value) for value in position)
                )

        while queue and expanded < max_expanded:
            self._check_search_time(
                phase,
                expanded=expanded,
                queued=len(queue),
            )
            frontier = []
            while queue and len(frontier) < SEARCH_FRONTIER_BATCH:
                _score, cost, node_index = heapq.heappop(queue)
                node = nodes[node_index]
                if cost != node.cost:
                    continue
                if best_cost.get(
                    self._state_key(node.position, node.yaw)
                ) != cost:
                    continue
                frontier.append(node_index)
            if not frontier:
                break
            expanded += len(frontier)

            visibility_meta = []
            for node_index in frontier:
                node = nodes[node_index]
                estimate = self._heuristic(
                    node.position,
                    node.yaw,
                    ideals,
                )
                if (
                    estimate[0][0]
                    > VIEW_QUERY_MAX_HEURISTIC_ACTIONS
                ):
                    continue
                target = target_map[estimate[1]]
                if self._potentially_visible(
                    node.position,
                    node.yaw,
                    target,
                ):
                    visibility_meta.append((node_index, target))
            visibility_meta = visibility_meta[:64]
            snapshots = self._query_cases(
                [
                    TargetVisibilityCase(
                        target.stable_id,
                        nodes[node_index].position,
                    )
                    for node_index, target in visibility_meta
                ],
                clock,
            )
            for (node_index, target), snapshot in zip(
                visibility_meta,
                snapshots,
            ):
                node = nodes[node_index]
                if not _target_observable(
                    snapshot.target,
                    node.position,
                    node.yaw,
                    self.spec,
                ):
                    continue
                results.append(
                    _ObservationPath(
                        stable_id=target.stable_id,
                        role=target.role,
                        actions=self._reconstruct(nodes, node_index),
                        position=node.position,
                        yaw=node.yaw,
                    )
                )
            if results:
                results.sort(
                    key=lambda item: (
                        len(item.actions),
                        item.stable_id,
                        item.position,
                        item.yaw,
                    )
                )
                return tuple(results[:SEARCH_RESULT_LIMIT])

            movement_meta = []
            for node_index in frontier:
                node = nodes[node_index]
                if node.cost >= max_actions:
                    continue
                estimate = self._heuristic(
                    node.position,
                    node.yaw,
                    ideals,
                )
                for action in (TurnLeftAction(), TurnRightAction()):
                    position, yaw = _apply_action_pose(
                        node.position,
                        node.yaw,
                        action,
                    )
                    enqueue(
                        position,
                        yaw,
                        node.cost + 1,
                        node_index,
                        action,
                    )
                movement_actions = []
                ideal_center = estimate[3]
                travel_yaw = _yaw_toward(
                    node.position,
                    ideal_center,
                )
                if (
                    abs(_angle_delta(travel_yaw, node.yaw))
                    <= SEARCH_FORWARD_DETOUR_DEGREES
                ):
                    movement_actions.append(ForwardAction())
                vertical_delta = (
                    ideal_center[2] - node.position[2]
                )
                if vertical_delta >= TASK_VERTICAL_STEP_METERS * 0.5:
                    movement_actions.append(AscendAction())
                elif (
                    vertical_delta
                    <= -TASK_VERTICAL_STEP_METERS * 0.5
                ):
                    movement_actions.append(DescendAction())
                for action in movement_actions:
                    position, yaw = _apply_action_pose(
                        node.position,
                        node.yaw,
                        action,
                    )
                    if not self.blueprint.contains_world_position(position):
                        continue
                    motion_key = self._motion_key(
                        node.position,
                        node.yaw,
                        action,
                    )
                    cached_clear = self._motion_cache.get(motion_key)
                    if cached_clear is not None:
                        if cached_clear:
                            enqueue(
                                position,
                                yaw,
                                node.cost + 1,
                                node_index,
                                action,
                            )
                        continue
                    movement_meta.append(
                        (
                            node_index,
                            action,
                            position,
                            yaw,
                            motion_key,
                        )
                    )
            if movement_meta:
                for offset in range(
                    0,
                    len(movement_meta),
                    GEOMETRY_MOTIONS_PER_BATCH,
                ):
                    batch = movement_meta[
                        offset : offset
                        + GEOMETRY_MOTIONS_PER_BATCH
                    ]
                    geometry = (
                        self.client.probe_camera_geometry_batch(
                            self.lockstep.session_id,
                            points=[item[2] for item in batch],
                            segments=[
                                (
                                    nodes[item[0]].position,
                                    item[2],
                                )
                                for item in batch
                            ],
                            timeout=30.0,
                        )
                    )
                    self._require_instant(geometry, clock)
                    self._check_search_time(
                        phase,
                        expanded=expanded,
                        queued=len(queue),
                    )
                    self._checked_motion_edges += len(batch)
                    for item, point_clear, segment_clear in zip(
                        batch,
                        geometry.point_clear,
                        geometry.segment_clear,
                    ):
                        clear = bool(point_clear and segment_clear)
                        self._motion_cache[item[4]] = clear
                        if not clear:
                            continue
                        node_index, action, position, yaw, _key = item
                        enqueue(
                            position,
                            yaw,
                            nodes[node_index].cost + 1,
                            node_index,
                            action,
                        )
        return ()

    def _initial_observation_paths(self, scenario):
        entity_map = _scenario_entity_map(scenario)
        matrices = pair_view_matrices(
            self.generated_start.rgbd_pair
        )
        paths = []
        for target in self.generated_start.visibility.targets:
            if target.role not in (
                VisibilityTargetRole.FIRE_TRUCK,
                VisibilityTargetRole.FLEEING_PEDESTRIAN,
            ):
                continue
            entity = entity_map.get(target.stable_id)
            if entity is None:
                continue
            if not any(
                _assess_target_view(
                    target,
                    matrices[name],
                    self.spec,
                ).task_observable
                for name in ("oblique", "nadir")
            ):
                continue
            paths.append(
                _ObservationPath(
                    stable_id=entity.stable_id,
                    role=entity.role,
                    actions=(),
                    position=tuple(
                        self.blueprint.absolute_pose[:3]
                    ),
                    yaw=float(self.blueprint.absolute_pose[5]),
                )
            )
        paths.sort(key=lambda item: (item.role, item.stable_id))
        self._cue_diagnostics["first_view_proposed"] += len(paths)
        self._cue_diagnostics["first_view_observable"] += len(paths)
        return tuple(paths[:SEARCH_RESULT_LIMIT])

    def _start_observation_paths(self, scenario, clock):
        responders = tuple(
            sorted(
                _response_entities(scenario),
                key=lambda entity: (
                    int(entity.role),
                    entity.stable_id,
                ),
            )
        )
        if not responders:
            return ()
        center = tuple(self.blueprint.absolute_pose[:3])
        yaw = float(self.blueprint.absolute_pose[5])
        snapshots = self._query_cases(
            [
                TargetVisibilityCase(entity.stable_id, center)
                for entity in responders
            ],
            clock,
        )
        self._cue_diagnostics["first_view_proposed"] += len(
            responders
        )
        paths = []
        for entity, snapshot in zip(responders, snapshots):
            if not _target_observable(
                snapshot.target,
                center,
                yaw,
                self.spec,
            ):
                continue
            paths.append(
                _ObservationPath(
                    stable_id=entity.stable_id,
                    role=entity.role,
                    actions=(),
                    position=center,
                    yaw=yaw,
                )
            )
        return tuple(paths)

    @staticmethod
    def _merge_observation_paths(*groups):
        merged = {}
        for group in groups:
            for path in group:
                key = (
                    path.stable_id,
                    tuple(type(action) for action in path.actions),
                    path.position,
                    path.yaw,
                )
                previous = merged.get(key)
                if (
                    previous is None
                    or len(path.actions) < len(previous.actions)
                ):
                    merged[key] = path
        return tuple(
            sorted(
                merged.values(),
                key=lambda item: (
                    len(item.actions),
                    int(item.role),
                    item.stable_id,
                    item.position,
                    item.yaw,
                ),
            )
        )

    def _search_response_observation_paths(
        self,
        responders,
        start_position,
        start_yaw,
        max_actions,
        clock,
        phase,
    ):
        paths = []
        for responder in responders:
            target_paths = self._search_observation_paths(
                (responder,),
                start_position,
                start_yaw,
                max_actions,
                clock,
                SEARCH_MAX_EXPANDED_CUE_PER_TARGET,
                f"{phase}-entity-{responder.stable_id}",
            )
            paths.extend(
                target_paths[:SEARCH_RESULTS_PER_CUE_TARGET]
            )
        paths.sort(
            key=lambda item: (
                len(item.actions),
                int(item.role),
                item.stable_id,
                item.position,
                item.yaw,
            )
        )
        return tuple(paths)

    def _motion_clear(self, start_position, start_yaw, action, clock):
        position, yaw = _apply_action_pose(
            start_position,
            start_yaw,
            action,
        )
        if not self.blueprint.contains_world_position(position):
            return None
        motion_key = self._motion_key(
            start_position,
            start_yaw,
            action,
        )
        cached_clear = self._motion_cache.get(motion_key)
        if cached_clear is not None:
            return (position, yaw) if cached_clear else None
        geometry = self.client.probe_camera_geometry_batch(
            self.lockstep.session_id,
            points=[position],
            segments=[(start_position, position)],
            timeout=30.0,
        )
        self._require_instant(geometry, clock)
        self._checked_motion_edges += 1
        clear = bool(
            geometry.point_clear[0] and geometry.segment_clear[0]
        )
        self._motion_cache[motion_key] = clear
        if not clear:
            return None
        return position, yaw

    def _transition_candidates(
        self,
        previous_paths,
        previous_scenario,
        current_scenario,
        clock,
    ):
        previous_entities = _scenario_entity_map(previous_scenario)
        current_entities = _scenario_entity_map(current_scenario)
        results = []
        self._cue_diagnostics["transition_cases"] += len(previous_paths)
        for path in previous_paths:
            previous = previous_entities.get(path.stable_id)
            current = current_entities.get(path.stable_id)
            if previous is None or current is None:
                self._cue_diagnostics["entity_missing"] += 1
                continue
            if (
                previous.task_state != ScenarioTaskState.ACTIVE
                or current.task_state != ScenarioTaskState.ACTIVE
            ):
                self._cue_diagnostics["task_inactive"] += 1
                continue
            displacement = math.dist(
                previous.position[:2],
                current.position[:2],
            )
            if (
                displacement
                < TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS
            ):
                self._cue_diagnostics[
                    "displacement_below_minimum"
                ] += 1
                continue
            cosine = _direction_cosine(
                previous,
                current,
                current_scenario.event_position,
            )
            if cosine < 0.5:
                self._cue_diagnostics[
                    "direction_cosine_below_0_5"
                ] += 1
                continue

            variants = [
                (
                    HoldAction(),
                    path.position,
                    path.yaw,
                    False,
                ),
                (
                    TurnLeftAction(),
                    path.position,
                    _apply_action_pose(
                        path.position,
                        path.yaw,
                        TurnLeftAction(),
                    )[1],
                    False,
                ),
                (
                    TurnRightAction(),
                    path.position,
                    _apply_action_pose(
                        path.position,
                        path.yaw,
                        TurnRightAction(),
                    )[1],
                    False,
                ),
            ]
            for action in (
                ForwardAction(),
                AscendAction(),
                DescendAction(),
            ):
                moved = self._motion_clear(
                    path.position,
                    path.yaw,
                    action,
                    clock,
                )
                if moved is not None:
                    variants.append((action, *moved, True))

            unique_centers = []
            center_to_index = {}
            for _action, center, _yaw, _moved in variants:
                key = tuple(round(value, 6) for value in center)
                if key not in center_to_index:
                    center_to_index[key] = len(unique_centers)
                    unique_centers.append(center)
            snapshots = self._query_cases(
                [
                    TargetVisibilityCase(path.stable_id, center)
                    for center in unique_centers
                ],
                clock,
            )
            snapshot_map = {
                tuple(round(value, 6) for value in center): snapshot
                for center, snapshot in zip(unique_centers, snapshots)
            }
            selected = None
            for action, center, yaw, _moved in variants:
                snapshot = snapshot_map[
                    tuple(round(value, 6) for value in center)
                ]
                if _target_observable(
                    snapshot.target,
                    center,
                    yaw,
                    self.spec,
                ):
                    selected = (action, center, yaw)
                    break
            if selected is None:
                self._cue_diagnostics[
                    "second_view_not_observable"
                ] += 1
                continue
            self._cue_diagnostics["valid_transition"] += 1
            action, center, yaw = selected
            results.append(
                (
                    path,
                    action,
                    center,
                    yaw,
                    displacement,
                    cosine,
                )
            )
        return tuple(results)

    @staticmethod
    def _count_actions(actions):
        action_types = (
            ForwardAction,
            AscendAction,
            DescendAction,
            TurnLeftAction,
            TurnRightAction,
            HoldAction,
            StopAction,
        )
        counts = {action_type: 0 for action_type in action_types}
        for action in actions:
            counts[type(action)] += 1
        return counts

    def _joint_witness(
        self,
        transition,
        task_step,
        current_scenario,
        clock,
    ):
        (
            path,
            transition_action,
            second_position,
            second_yaw,
            displacement,
            cosine,
        ) = transition
        padding = task_step - len(path.actions)
        if padding < 0:
            return None
        prefix = (
            [HoldAction() for _ in range(padding)]
            + list(path.actions)
            + [transition_action]
        )
        remaining_for_goal = (
            self.horizon_steps - len(prefix) - 1
        )
        if remaining_for_goal < 0:
            return None
        source = _source_entity(current_scenario)
        goal_paths = self._search_observation_paths(
            (source,),
            second_position,
            second_yaw,
            remaining_for_goal,
            clock,
            SEARCH_MAX_EXPANDED_GOAL,
            f"cue-to-goal-step-{task_step}",
        )
        if not goal_paths:
            return None
        goal_path = goal_paths[0]
        final_actions = (
            prefix
            + list(goal_path.actions)
            + [
                StopAction(
                    self.blueprint.world_to_local(
                        current_scenario.event_position
                    )
                )
            ]
        )
        counts = self._count_actions(final_actions)
        cue = CueWitness(
            stable_id=path.stable_id,
            role=path.role,
            first_step=task_step,
            second_step=task_step + 1,
            first_pose=(*path.position, path.yaw),
            second_pose=(*second_position, second_yaw),
            transition_action=transition_action,
            horizontal_displacement_m=displacement,
            direction_cosine=cosine,
        )
        goal = GoalWitness(
            stable_id=goal_path.stable_id,
            pose=(*goal_path.position, goal_path.yaw),
        )
        return JointPathWitness(
            cue=cue,
            goal=goal,
            actions=tuple(final_actions),
            forward_actions=counts[ForwardAction],
            ascend_actions=counts[AscendAction],
            descend_actions=counts[DescendAction],
            turn_left_actions=counts[TurnLeftAction],
            turn_right_actions=counts[TurnRightAction],
            hold_actions=counts[HoldAction],
            stop_actions=counts[StopAction],
            total_actions=len(final_actions),
            remaining_actions=self.horizon_steps - len(final_actions),
        )

    def run(self):
        run_started = time.perf_counter()
        graph_seconds = 0.0
        goal_seconds = 0.0
        temporal_seconds = 0.0
        queried_steps = 0
        cue_path_found = False
        goal_path_found = False
        best_witness = None
        best_ordered_candidate = None
        try:
            initial_clock = self.lockstep.refresh()
            if initial_clock.step_index != 1:
                raise RuntimeError(
                    "Feasibility audit must begin at initial t=250ms"
                )
            previous_scenario = self.client.get_scenario_state(
                self.scenario_id
            )
            phase_started = time.perf_counter()
            source = _source_entity(previous_scenario)
            goal_paths = self._search_observation_paths(
                (source,),
                self.blueprint.absolute_pose[:3],
                self.blueprint.absolute_pose[5],
                self.horizon_steps - 1,
                initial_clock,
                SEARCH_MAX_EXPANDED_GOAL,
                "initial-goal",
            )
            goal_seconds = time.perf_counter() - phase_started
            goal_path_found = bool(goal_paths)

            phase_started = time.perf_counter()
            for task_step in range(self.horizon_steps - 1):
                if task_step == 0:
                    previous_paths = self._initial_observation_paths(
                        previous_scenario
                    )
                else:
                    start_paths = self._start_observation_paths(
                        previous_scenario,
                        self.lockstep.snapshot,
                    )
                    responders = _selected_response_entities(
                        previous_scenario,
                        task_step,
                    )
                    if task_step % CUE_SEARCH_STEP_STRIDE == 0:
                        self._cue_diagnostics[
                            "first_view_proposed"
                        ] += len(responders)
                        searched_paths = (
                            self._search_response_observation_paths(
                                responders,
                                self.blueprint.absolute_pose[:3],
                                self.blueprint.absolute_pose[5],
                                task_step,
                                self.lockstep.snapshot,
                                f"cue-step-{task_step}",
                            )
                        )
                    else:
                        searched_paths = ()
                    previous_paths = self._merge_observation_paths(
                        start_paths,
                        searched_paths,
                    )
                    self._cue_diagnostics[
                        "first_view_observable"
                    ] += len(previous_paths)
                current_clock = self.lockstep.advance()
                current_scenario = self.client.get_scenario_state(
                    self.scenario_id
                )
                transitions = self._transition_candidates(
                    previous_paths,
                    previous_scenario,
                    current_scenario,
                    current_clock,
                )
                queried_steps = task_step + 1
                if transitions:
                    cue_path_found = True
                for transition in transitions:
                    witness = self._joint_witness(
                        transition,
                        task_step,
                        current_scenario,
                        current_clock,
                    )
                    if witness is None:
                        continue
                    if (
                        best_ordered_candidate is None
                        or witness.total_actions
                        < best_ordered_candidate.total_actions
                    ):
                        best_ordered_candidate = witness
                    if (
                        witness.total_actions <= self.horizon_steps
                        and (
                            best_witness is None
                            or witness.total_actions
                            < best_witness.total_actions
                        )
                    ):
                        best_witness = witness
                if (
                    best_witness is not None
                    and best_witness.remaining_actions
                    >= REQUIRED_ACTION_MARGIN
                ):
                    break
                previous_scenario = current_scenario
            temporal_seconds = time.perf_counter() - phase_started
            graph_seconds = goal_seconds + temporal_seconds

            if best_witness is None:
                status = FeasibilityStatus.NO_JOINT_WITNESS_IN_SEARCH
                message = (
                    "Finite fixed-action search found no ordered "
                    "cue-to-goal witness"
                )
            elif (
                best_witness.remaining_actions
                >= REQUIRED_ACTION_MARGIN
            ):
                status = FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN
                message = "Joint witness retains at least five actions"
            else:
                status = FeasibilityStatus.JOINT_WITNESS_TIGHT
                message = (
                    "Joint witness exists but retains fewer than "
                    "five actions"
                )
            return SpatiotemporalFeasibilityReport(
                start_id=self.blueprint.start_id,
                visibility_stratum=self.blueprint.visibility_stratum,
                status=status,
                cue_path_found=cue_path_found,
                goal_view_path_found=goal_path_found,
                cue_then_goal_path_found=best_witness is not None,
                search_digest=self._search_digest,
                searched_states=len(self._searched_keys),
                checked_motion_edges=self._checked_motion_edges,
                queried_steps=queried_steps,
                minimum_ordered_actions=(
                    None
                    if best_ordered_candidate is None
                    else best_ordered_candidate.total_actions
                ),
                cue_diagnostics=tuple(
                    sorted(self._cue_diagnostics.items())
                ),
                phase_seconds=(
                    ("action_search", graph_seconds),
                    ("goal_visibility", goal_seconds),
                    ("temporal_cue_search", temporal_seconds),
                    ("auditor_total", time.perf_counter() - run_started),
                ),
                witness=best_witness,
                message=message,
            )
        except Exception as error:
            graph_seconds = time.perf_counter() - run_started
            return SpatiotemporalFeasibilityReport(
                start_id=self.blueprint.start_id,
                visibility_stratum=self.blueprint.visibility_stratum,
                status=FeasibilityStatus.UNKNOWN,
                cue_path_found=cue_path_found,
                goal_view_path_found=goal_path_found,
                cue_then_goal_path_found=False,
                search_digest=self._search_digest,
                searched_states=len(self._searched_keys),
                checked_motion_edges=self._checked_motion_edges,
                queried_steps=queried_steps,
                minimum_ordered_actions=None,
                cue_diagnostics=tuple(
                    sorted(self._cue_diagnostics.items())
                ),
                phase_seconds=(
                    ("action_search", graph_seconds),
                    ("goal_visibility", goal_seconds),
                    ("temporal_cue_search", temporal_seconds),
                    ("auditor_total", time.perf_counter() - run_started),
                ),
                witness=None,
                message=str(error),
            )
