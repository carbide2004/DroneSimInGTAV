"""Evaluation-only Stage 2D joint-path witness search.

This module uses privileged scenario and visibility truth. Its reports must
never be exposed as an agent observation.
"""

import hashlib
import heapq
import math
import random
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
    HoldAction,
    RotateAction,
    StopAction,
    TranslateAction,
)
from .task_starts import (
    TASK_ACTIVITY_RADIUS_METERS,
    TASK_ACTIVITY_VERTICAL_METERS,
    TASK_HORIZON_STEPS,
    TASK_MAX_TRANSLATION_METERS,
    TASK_MAX_YAW_DEGREES,
    TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS,
    GeneratedTaskStart,
    _assess_target_view,
    pair_view_matrices,
    virtual_view_matrices,
)


PRM_NODE_COUNT = 1024
PRM_MAX_PROPOSALS = 8192
PRM_NEIGHBORS = 16
PRM_MAX_EDGE_METERS = 24.0
GEOMETRY_BATCH_SIZE = 256
VISIBILITY_BATCH_SIZE = 64
CUE_RESPONDERS_PER_STEP = 16
CUE_FIRST_VIEW_CASES_PER_STEP = 32
CUE_VIEW_CASES_PER_RESPONDER = 2
GOAL_CANDIDATE_LIMIT = 64
GOAL_ANGULAR_BINS = 16
GOAL_CASES_PER_ANGULAR_BIN = 4
REQUIRED_ACTION_MARGIN = 5


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
    node_index: int


@dataclass(frozen=True)
class JointPathWitness:
    cue: CueWitness
    goal: GoalWitness
    actions: tuple
    translate_actions: int
    rotate_actions: int
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
    roadmap_digest: str
    roadmap_nodes: int
    roadmap_edges: int
    queried_steps: int
    minimum_ordered_actions: int | None
    cue_diagnostics: tuple
    phase_seconds: tuple
    witness: JointPathWitness | None
    message: str


@dataclass(frozen=True)
class _Candidate:
    stable_id: int
    role: ScenarioEntityRole
    step: int
    node_index: int
    center: tuple
    yaw: float
    entity_position: tuple
    entity_task_state: ScenarioTaskState


@dataclass(frozen=True)
class _Goal:
    stable_id: int
    node_index: int
    center: tuple
    yaw: float


def _angle_delta(target, source):
    return (float(target) - float(source) + 180.0) % 360.0 - 180.0


def _yaw_toward(origin, target):
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return math.degrees(math.atan2(-dx, dy))


def _rotation_count(source_yaw, target_yaw):
    return int(
        math.ceil(
            abs(_angle_delta(target_yaw, source_yaw))
            / TASK_MAX_YAW_DEGREES
            - 1.0e-12
        )
    )


def _rotation_actions(source_yaw, target_yaw):
    remaining = _angle_delta(target_yaw, source_yaw)
    actions = []
    while abs(remaining) > 1.0e-9:
        delta = math.copysign(
            min(abs(remaining), TASK_MAX_YAW_DEGREES),
            remaining,
        )
        actions.append(RotateAction(delta))
        remaining -= delta
    return actions


def _translation_actions(points, yaw_degrees):
    actions = []
    yaw = math.radians(float(yaw_degrees))
    forward = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
    right = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
    for start, end in zip(points, points[1:]):
        delta = np.asarray(end, dtype=np.float64) - np.asarray(
            start,
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(delta))
        count = max(
            1,
            int(
                math.ceil(
                    distance / TASK_MAX_TRANSLATION_METERS
                    - 1.0e-12
                )
            ),
        )
        piece = delta / count
        for _ in range(count):
            actions.append(
                TranslateAction(
                    float(np.dot(piece, forward)),
                    float(np.dot(piece, right)),
                    float(piece[2]),
                )
            )
    return actions


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


class _Roadmap:
    def __init__(self, nodes, adjacency, start_distances, start_parents):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.adjacency = tuple(tuple(edges) for edges in adjacency)
        self.start_distances = tuple(start_distances)
        self.start_parents = tuple(start_parents)
        self._shortest_cache = {}

    @property
    def edge_count(self):
        return sum(len(edges) for edges in self.adjacency) // 2

    def start_path(self, target):
        return self._reconstruct(self.start_parents, 0, target)

    def shortest(self, source, target):
        source = int(source)
        target = int(target)
        if source not in self._shortest_cache:
            self._shortest_cache[source] = self._dijkstra(source)
        distances, parents = self._shortest_cache[source]
        return distances[target], self._reconstruct(
            parents,
            source,
            target,
        )

    def _dijkstra(self, source):
        distances = [math.inf] * len(self.nodes)
        parents = [-1] * len(self.nodes)
        distances[source] = 0
        queue = [(0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for neighbor, cost in self.adjacency[node]:
                candidate = distance + cost
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        return distances, parents

    @staticmethod
    def _reconstruct(parents, source, target):
        if source == target:
            return (source,)
        if parents[target] < 0:
            return ()
        path = [target]
        while path[-1] != source:
            parent = parents[path[-1]]
            if parent < 0:
                return ()
            path.append(parent)
        path.reverse()
        return tuple(path)


def _dijkstra(adjacency, source):
    distances = [math.inf] * len(adjacency)
    parents = [-1] * len(adjacency)
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, cost in adjacency[node]:
            candidate = distance + cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, parents


class SpatiotemporalFeasibilityAuditor:
    def __init__(
        self,
        client,
        lockstep,
        scenario_id,
        generated_start,
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
        self.roadmap = None
        self._roadmap_digest = ""
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

    def _require_instant(self, snapshot, reference):
        if (
            snapshot.lockstep_session_id != reference.session_id
            or snapshot.step_index != reference.step_index
            or snapshot.game_timer_ms != reference.game_timer_ms
        ):
            raise RuntimeError(
                "Evaluation batch does not belong to the frozen instant"
            )

    def build_roadmap(self):
        if self.roadmap is not None:
            return self.roadmap
        clock = self.lockstep.refresh()
        roadmap_seed = (
            int(self.blueprint.start_seed)
            ^ (
                int(self.blueprint.visibility_stratum)
                * 0x9E3779B97F4A7C15
            )
            ^ (
                int(self.blueprint.candidate_index)
                * 0xBF58476D1CE4E5B9
            )
            ^ 0x50524D5F32445F31
        ) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(
            roadmap_seed
        )
        start = tuple(self.blueprint.absolute_pose[:3])
        proposals = [start]
        seen = {
            tuple(round(float(value), 3) for value in start)
        }
        while len(proposals) < PRM_MAX_PROPOSALS:
            radius = math.sqrt(rng.random()) * (
                TASK_ACTIVITY_RADIUS_METERS
            )
            angle = rng.uniform(-math.pi, math.pi)
            position = (
                start[0] + math.cos(angle) * radius,
                start[1] + math.sin(angle) * radius,
                start[2]
                + rng.uniform(
                    -TASK_ACTIVITY_VERTICAL_METERS,
                    TASK_ACTIVITY_VERTICAL_METERS,
                ),
            )
            if not self.blueprint.contains_world_position(position):
                continue
            key = tuple(round(float(value), 3) for value in position)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(position)

        accepted = []
        for offset in range(0, len(proposals), GEOMETRY_BATCH_SIZE):
            batch = proposals[offset : offset + GEOMETRY_BATCH_SIZE]
            result = self.client.probe_camera_geometry_batch(
                self.lockstep.session_id,
                points=batch,
                timeout=30.0,
            )
            self._require_instant(result, clock)
            for point, clear in zip(batch, result.point_clear):
                if clear:
                    accepted.append(point)
                    if len(accepted) == PRM_NODE_COUNT:
                        break
            if len(accepted) == PRM_NODE_COUNT:
                break
        if not accepted or accepted[0] != start:
            raise RuntimeError(
                "PROFILE_START_CLEARANCE_FAILED: start is not 2 m clear"
            )
        if len(accepted) != PRM_NODE_COUNT:
            raise RuntimeError(
                "PRM accepted "
                f"{len(accepted)}/{PRM_NODE_COUNT} clear nodes"
            )

        nodes = np.asarray(accepted, dtype=np.float64)
        delta = nodes[:, None, :] - nodes[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        edge_candidates = set()
        for node_index in range(len(nodes)):
            order = np.argsort(
                distances[node_index],
                kind="stable",
            )
            neighbors = [
                int(index)
                for index in order
                if index != node_index
                and distances[node_index, index]
                <= PRM_MAX_EDGE_METERS
            ][:PRM_NEIGHBORS]
            for neighbor in neighbors:
                edge_candidates.add(
                    tuple(sorted((node_index, neighbor)))
                )
        ordered_edges = sorted(edge_candidates)
        clear_edges = []
        for offset in range(
            0,
            len(ordered_edges),
            GEOMETRY_BATCH_SIZE,
        ):
            indices = ordered_edges[
                offset : offset + GEOMETRY_BATCH_SIZE
            ]
            segments = [
                (
                    tuple(nodes[left]),
                    tuple(nodes[right]),
                )
                for left, right in indices
            ]
            result = self.client.probe_camera_geometry_batch(
                self.lockstep.session_id,
                segments=segments,
                timeout=30.0,
            )
            self._require_instant(result, clock)
            clear_edges.extend(
                edge
                for edge, clear in zip(indices, result.segment_clear)
                if clear
            )

        adjacency = [[] for _ in range(len(nodes))]
        for left, right in clear_edges:
            length = float(distances[left, right])
            cost = int(
                math.ceil(
                    length / TASK_MAX_TRANSLATION_METERS
                    - 1.0e-12
                )
            )
            adjacency[left].append((right, cost))
            adjacency[right].append((left, cost))
        for edges in adjacency:
            edges.sort()
        start_distances, start_parents = _dijkstra(adjacency, 0)
        self.roadmap = _Roadmap(
            nodes,
            adjacency,
            start_distances,
            start_parents,
        )
        digest = hashlib.blake2b(digest_size=16)
        digest.update(nodes.astype("<f4").tobytes())
        for left, right in clear_edges:
            digest.update(int(left).to_bytes(4, "little"))
            digest.update(int(right).to_bytes(4, "little"))
        self._roadmap_digest = digest.hexdigest()
        return self.roadmap

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

    def _goal_candidates(self, scenario, clock):
        source = _source_entity(scenario)
        start_yaw = self.blueprint.absolute_pose[5]
        candidates = []
        for node_index, center in enumerate(self.roadmap.nodes):
            start_cost = self.roadmap.start_distances[node_index]
            if not math.isfinite(start_cost):
                continue
            distance = math.dist(center, source.position)
            if distance < 8.0 or distance > 80.0:
                continue
            yaw = _yaw_toward(center, source.position)
            lower_bound = (
                start_cost
                + _rotation_count(start_yaw, yaw)
                + 1
            )
            if lower_bound > TASK_HORIZON_STEPS:
                continue
            angle = (
                math.atan2(
                    float(center[1] - source.position[1]),
                    float(center[0] - source.position[0]),
                )
                + 2.0 * math.pi
            ) % (2.0 * math.pi)
            angular_bin = min(
                GOAL_ANGULAR_BINS - 1,
                int(
                    angle
                    / (2.0 * math.pi)
                    * GOAL_ANGULAR_BINS
                ),
            )
            rank = (
                abs(distance - 25.0),
                lower_bound,
                node_index,
            )
            candidates.append(
                (
                    angular_bin,
                    rank,
                    _Goal(
                        stable_id=source.stable_id,
                        node_index=node_index,
                        center=tuple(float(value) for value in center),
                        yaw=yaw,
                    ),
                )
            )
        selected_entries = []
        selected_nodes = set()
        for angular_bin in range(GOAL_ANGULAR_BINS):
            in_bin = sorted(
                (
                    (rank, candidate)
                    for candidate_bin, rank, candidate in candidates
                    if candidate_bin == angular_bin
                ),
                key=lambda item: item[0],
            )
            for _rank, candidate in in_bin[
                :GOAL_CASES_PER_ANGULAR_BIN
            ]:
                selected_entries.append(candidate)
                selected_nodes.add(candidate.node_index)
        if len(selected_entries) < GOAL_CANDIDATE_LIMIT:
            remaining = sorted(
                (
                    (rank, candidate)
                    for _angular_bin, rank, candidate in candidates
                    if candidate.node_index not in selected_nodes
                ),
                key=lambda item: item[0],
            )
            selected_entries.extend(
                candidate
                for _rank, candidate in remaining[
                    : GOAL_CANDIDATE_LIMIT - len(selected_entries)
                ]
            )
        selected = selected_entries[:GOAL_CANDIDATE_LIMIT]
        observable = []
        for offset in range(0, len(selected), VISIBILITY_BATCH_SIZE):
            batch = selected[offset : offset + VISIBILITY_BATCH_SIZE]
            snapshots = self._query_cases(
                [
                    TargetVisibilityCase(
                        item.stable_id,
                        item.center,
                    )
                    for item in batch
                ],
                clock,
            )
            for item, snapshot in zip(batch, snapshots):
                if _target_observable(
                    snapshot.target,
                    item.center,
                    item.yaw,
                    self.spec,
                ):
                    observable.append(item)
        return tuple(observable)

    def _select_new_candidates(self, scenario, task_step):
        entities = sorted(
            _response_entities(scenario),
            key=lambda entity: (
                entity.role
                != ScenarioEntityRole.FIRE_TRUCK,
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
        selected = trucks[:CUE_RESPONDERS_PER_STEP]
        remaining = CUE_RESPONDERS_PER_STEP - len(selected)
        if pedestrians and remaining > 0:
            offset = (
                task_step * remaining
            ) % len(pedestrians)
            rotated = pedestrians[offset:] + pedestrians[:offset]
            selected.extend(rotated[:remaining])

        per_entity_candidates = []
        start_yaw = self.blueprint.absolute_pose[5]
        for entity in selected:
            candidates_for_entity = []
            for node_index, center in enumerate(self.roadmap.nodes):
                translation_cost = self.roadmap.start_distances[node_index]
                if not math.isfinite(translation_cost):
                    continue
                distance = math.dist(center, entity.position)
                if distance < 8.0 or distance > 60.0:
                    continue
                yaw = _yaw_toward(center, entity.position)
                arrival_cost = (
                    translation_cost
                    + _rotation_count(start_yaw, yaw)
                )
                if arrival_cost > task_step:
                    continue
                candidates_for_entity.append(
                    {
                        "arrival_cost": arrival_cost,
                        "distance_rank": abs(distance - 25.0),
                        "node_index": node_index,
                        "yaw": yaw,
                    }
                )
            if not candidates_for_entity:
                continue
            fastest = min(
                candidates_for_entity,
                key=lambda item: (
                    item["arrival_cost"],
                    item["node_index"],
                    item["yaw"],
                ),
            )
            selected_for_entity = [fastest]
            observation_ranked = sorted(
                candidates_for_entity,
                key=lambda item: (
                    item["distance_rank"],
                    item["arrival_cost"],
                    item["node_index"],
                    item["yaw"],
                ),
            )
            for item in observation_ranked:
                if item["node_index"] == fastest["node_index"]:
                    continue
                selected_for_entity.append(item)
                if (
                    len(selected_for_entity)
                    == CUE_VIEW_CASES_PER_RESPONDER
                ):
                    break
            per_entity_candidates.extend(
                (
                    item["arrival_cost"],
                    entity.stable_id,
                    item["node_index"],
                    item["yaw"],
                    entity,
                )
                for item in selected_for_entity
            )
        per_entity_candidates.sort(key=lambda item: item[:4])
        candidates = []
        for (
            _arrival_cost,
            _stable_id,
            node_index,
            yaw,
            entity,
        ) in per_entity_candidates[:CUE_FIRST_VIEW_CASES_PER_STEP]:
            candidates.append(
                _Candidate(
                    stable_id=entity.stable_id,
                    role=entity.role,
                    step=task_step,
                    node_index=node_index,
                    center=tuple(
                        float(value)
                        for value in self.roadmap.nodes[node_index]
                    ),
                    yaw=yaw,
                    entity_position=entity.position,
                    entity_task_state=entity.task_state,
                )
            )
        return tuple(candidates)

    def _initial_candidates(self, scenario):
        entity_map = _scenario_entity_map(scenario)
        matrices = pair_view_matrices(
            self.generated_start.rgbd_pair
        )
        candidates = []
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
            candidates.append(
                _Candidate(
                    stable_id=entity.stable_id,
                    role=entity.role,
                    step=0,
                    node_index=0,
                    center=tuple(
                        self.blueprint.absolute_pose[:3]
                    ),
                    yaw=self.blueprint.absolute_pose[5],
                    entity_position=entity.position,
                    entity_task_state=entity.task_state,
                )
            )
        selected = tuple(candidates[:CUE_RESPONDERS_PER_STEP])
        self._cue_diagnostics["first_view_proposed"] += len(selected)
        self._cue_diagnostics["first_view_observable"] += len(selected)
        return selected

    def _observable_new_candidates(self, candidates, clock):
        self._cue_diagnostics["first_view_proposed"] += len(candidates)
        snapshots = self._query_cases(
            [
                TargetVisibilityCase(
                    candidate.stable_id,
                    candidate.center,
                )
                for candidate in candidates
            ],
            clock,
        )
        observable = tuple(
            candidate
            for candidate, snapshot in zip(candidates, snapshots)
            if _target_observable(
                snapshot.target,
                candidate.center,
                candidate.yaw,
                self.spec,
            )
        )
        self._cue_diagnostics["first_view_observable"] += len(
            observable
        )
        return observable

    def _transition_candidates(
        self,
        previous_candidates,
        previous_scenario,
        current_scenario,
        clock,
    ):
        previous_entities = _scenario_entity_map(previous_scenario)
        current_entities = _scenario_entity_map(current_scenario)
        base_cases = []
        translation_geometry = []
        translation_meta = []
        self._cue_diagnostics["transition_cases"] += len(
            previous_candidates
        )
        for candidate in previous_candidates:
            current = current_entities.get(candidate.stable_id)
            previous = previous_entities.get(candidate.stable_id)
            if current is None or previous is None:
                self._cue_diagnostics["entity_missing"] += 1
                continue
            base_cases.append(
                (
                    candidate,
                    previous,
                    current,
                    TargetVisibilityCase(
                        candidate.stable_id,
                        candidate.center,
                    ),
                )
            )
            delta = np.asarray(current.position, dtype=np.float64) - (
                np.asarray(previous.position, dtype=np.float64)
            )
            length = float(np.linalg.norm(delta))
            if length > TASK_MAX_TRANSLATION_METERS:
                delta *= TASK_MAX_TRANSLATION_METERS / length
            translated = tuple(
                float(value)
                for value in (
                    np.asarray(candidate.center, dtype=np.float64)
                    + delta
                )
            )
            if (
                length > 1.0e-6
                and self.blueprint.contains_world_position(translated)
            ):
                translation_geometry.append(
                    (candidate.center, translated)
                )
                translation_meta.append(
                    (candidate, previous, current, translated)
                )

        clear_translation = []
        if translation_geometry:
            geometry = self.client.probe_camera_geometry_batch(
                self.lockstep.session_id,
                points=[
                    translated
                    for _candidate, _previous, _current, translated
                    in translation_meta
                ],
                segments=translation_geometry,
                timeout=30.0,
            )
            self._require_instant(geometry, clock)
            for item, point_clear, segment_clear in zip(
                translation_meta,
                geometry.point_clear,
                geometry.segment_clear,
            ):
                if point_clear and segment_clear:
                    clear_translation.append(item)

        query_meta = [
            ("base", candidate, previous, current, candidate.center)
            for candidate, previous, current, _case in base_cases
        ]
        query_cases = [case for *_rest, case in base_cases]
        remaining = VISIBILITY_BATCH_SIZE - len(query_cases)
        for candidate, previous, current, translated in (
            clear_translation[:remaining]
        ):
            query_meta.append(
                (
                    "translate",
                    candidate,
                    previous,
                    current,
                    translated,
                )
            )
            query_cases.append(
                TargetVisibilityCase(
                    candidate.stable_id,
                    translated,
                )
            )
        snapshots = self._query_cases(query_cases, clock)
        grouped = {}
        for meta, snapshot in zip(query_meta, snapshots):
            grouped.setdefault(meta[1].stable_id, {})[meta[0]] = (
                meta,
                snapshot,
            )

        results = []
        for candidate in previous_candidates:
            variants = grouped.get(candidate.stable_id, {})
            if "base" not in variants:
                continue
            _meta, base_snapshot = variants["base"]
            previous = previous_entities[candidate.stable_id]
            current = current_entities[candidate.stable_id]
            displacement = math.dist(
                previous.position[:2],
                current.position[:2],
            )
            cosine = _direction_cosine(
                previous,
                current,
                current_scenario.event_position,
            )
            if (
                previous.task_state != ScenarioTaskState.ACTIVE
                or current.task_state != ScenarioTaskState.ACTIVE
            ):
                self._cue_diagnostics["task_inactive"] += 1
                continue
            if (
                displacement
                < TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS
            ):
                self._cue_diagnostics[
                    "displacement_below_minimum"
                ] += 1
                continue
            if cosine < 0.5:
                self._cue_diagnostics[
                    "direction_cosine_below_0_5"
                ] += 1
                continue

            transition = None
            second_center = candidate.center
            second_yaw = candidate.yaw
            if _target_observable(
                base_snapshot.target,
                candidate.center,
                candidate.yaw,
                self.spec,
            ):
                transition = HoldAction()
            else:
                rotate_yaw = _yaw_toward(
                    candidate.center,
                    current.position,
                )
                rotate_delta = _angle_delta(
                    rotate_yaw,
                    candidate.yaw,
                )
                if (
                    abs(rotate_delta)
                    <= TASK_MAX_YAW_DEGREES + 1.0e-9
                    and abs(rotate_delta) > 1.0e-9
                    and _target_observable(
                        base_snapshot.target,
                        candidate.center,
                        rotate_yaw,
                        self.spec,
                    )
                ):
                    transition = RotateAction(rotate_delta)
                    second_yaw = rotate_yaw
            if transition is None and "translate" in variants:
                meta, translated_snapshot = variants["translate"]
                translated = meta[4]
                if _target_observable(
                    translated_snapshot.target,
                    translated,
                    candidate.yaw,
                    self.spec,
                ):
                    world_delta = (
                        np.asarray(translated, dtype=np.float64)
                        - np.asarray(candidate.center, dtype=np.float64)
                    )
                    yaw = math.radians(candidate.yaw)
                    forward = np.asarray(
                        (-math.sin(yaw), math.cos(yaw), 0.0)
                    )
                    right = np.asarray(
                        (math.cos(yaw), math.sin(yaw), 0.0)
                    )
                    transition = TranslateAction(
                        float(np.dot(world_delta, forward)),
                        float(np.dot(world_delta, right)),
                        float(world_delta[2]),
                    )
                    second_center = translated
            if transition is not None:
                self._cue_diagnostics["valid_transition"] += 1
                results.append(
                    (
                        candidate,
                        transition,
                        second_center,
                        second_yaw,
                        displacement,
                        cosine,
                    )
                )
            else:
                self._cue_diagnostics[
                    "second_view_not_observable"
                ] += 1
        return tuple(results)

    def _joint_witness(self, cue_transition, goals, event_position):
        (
            candidate,
            transition,
            second_center,
            second_yaw,
            displacement,
            cosine,
        ) = cue_transition
        start_yaw = self.blueprint.absolute_pose[5]
        start_path_indices = self.roadmap.start_path(
            candidate.node_index
        )
        if not start_path_indices:
            return None
        start_points = [
            tuple(self.roadmap.nodes[index])
            for index in start_path_indices
        ]
        travel_to_cue = _translation_actions(
            start_points,
            start_yaw,
        )
        rotate_to_cue = _rotation_actions(
            start_yaw,
            candidate.yaw,
        )
        arrival_actions = travel_to_cue + rotate_to_cue
        if len(arrival_actions) > candidate.step:
            return None
        actions = [
            HoldAction()
            for _ in range(candidate.step - len(arrival_actions))
        ]
        actions.extend(arrival_actions)
        actions.append(transition)

        best = None
        for goal in goals:
            route_cost, path_indices = self.roadmap.shortest(
                candidate.node_index,
                goal.node_index,
            )
            if not math.isfinite(route_cost) or not path_indices:
                continue
            post_actions = []
            route_yaw = second_yaw
            if isinstance(transition, TranslateAction):
                delta = (
                    np.asarray(candidate.center, dtype=np.float64)
                    - np.asarray(second_center, dtype=np.float64)
                )
                yaw = math.radians(route_yaw)
                forward = np.asarray(
                    (-math.sin(yaw), math.cos(yaw), 0.0)
                )
                right = np.asarray(
                    (math.cos(yaw), math.sin(yaw), 0.0)
                )
                post_actions.append(
                    TranslateAction(
                        float(np.dot(delta, forward)),
                        float(np.dot(delta, right)),
                        float(delta[2]),
                    )
                )
            route_points = [
                tuple(self.roadmap.nodes[index])
                for index in path_indices
            ]
            post_actions.extend(
                _translation_actions(route_points, route_yaw)
            )
            post_actions.extend(
                _rotation_actions(route_yaw, goal.yaw)
            )
            final_actions = (
                actions
                + post_actions
                + [
                    StopAction(
                        self.blueprint.world_to_local(
                            event_position
                        )
                    )
                ]
            )
            cue = CueWitness(
                stable_id=candidate.stable_id,
                role=candidate.role,
                first_step=candidate.step,
                second_step=candidate.step + 1,
                first_pose=(
                    *candidate.center,
                    candidate.yaw,
                ),
                second_pose=(
                    *second_center,
                    second_yaw,
                ),
                transition_action=transition,
                horizontal_displacement_m=displacement,
                direction_cosine=cosine,
            )
            goal_witness = GoalWitness(
                stable_id=goal.stable_id,
                pose=(*goal.center, goal.yaw),
                node_index=goal.node_index,
            )
            counts = {
                TranslateAction: 0,
                RotateAction: 0,
                HoldAction: 0,
                StopAction: 0,
            }
            for action in final_actions:
                counts[type(action)] += 1
            witness = JointPathWitness(
                cue=cue,
                goal=goal_witness,
                actions=tuple(final_actions),
                translate_actions=counts[TranslateAction],
                rotate_actions=counts[RotateAction],
                hold_actions=counts[HoldAction],
                stop_actions=counts[StopAction],
                total_actions=len(final_actions),
                remaining_actions=(
                    TASK_HORIZON_STEPS - len(final_actions)
                ),
            )
            rank = (
                witness.total_actions,
                goal.node_index,
            )
            if best is None or rank < best[0]:
                best = (rank, witness)
        return None if best is None else best[1]

    def run(self):
        run_started = time.perf_counter()
        roadmap_seconds = 0.0
        goal_seconds = 0.0
        temporal_seconds = 0.0
        try:
            phase_started = time.perf_counter()
            self.build_roadmap()
            roadmap_seconds = time.perf_counter() - phase_started
            initial_clock = self.lockstep.refresh()
            if initial_clock.step_index != 1:
                raise RuntimeError(
                    "Feasibility audit must begin at initial t=250ms"
                )
            previous_scenario = self.client.get_scenario_state(
                self.scenario_id
            )
            phase_started = time.perf_counter()
            goals = self._goal_candidates(
                previous_scenario,
                initial_clock,
            )
            goal_seconds = time.perf_counter() - phase_started
            goal_path_found = bool(goals)
            previous_candidates = self._initial_candidates(
                previous_scenario
            )
            cue_path_found = False
            best_witness = None
            best_ordered_candidate = None
            queried_steps = 0

            phase_started = time.perf_counter()
            for task_step in range(TASK_HORIZON_STEPS - 1):
                if task_step > 0:
                    new_candidates = self._select_new_candidates(
                        previous_scenario,
                        task_step,
                    )
                    previous_candidates = (
                        self._observable_new_candidates(
                            new_candidates,
                            self.lockstep.snapshot,
                        )
                    )
                current_clock = self.lockstep.advance()
                current_scenario = self.client.get_scenario_state(
                    self.scenario_id
                )
                transitions = self._transition_candidates(
                    previous_candidates,
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
                        goals,
                        current_scenario.event_position,
                    )
                    if witness is not None:
                        if (
                            best_ordered_candidate is None
                            or witness.total_actions
                            < best_ordered_candidate.total_actions
                        ):
                            best_ordered_candidate = witness
                        if (
                            witness.total_actions
                            <= TASK_HORIZON_STEPS
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

            if best_witness is None:
                status = FeasibilityStatus.NO_JOINT_WITNESS_IN_SEARCH
                if best_ordered_candidate is None:
                    message = (
                        "Finite deterministic search found no ordered "
                        "cue-to-goal candidate"
                    )
                else:
                    message = (
                        "Shortest ordered cue-to-goal candidate requires "
                        f"{best_ordered_candidate.total_actions} actions, "
                        f"exceeding the {TASK_HORIZON_STEPS}-action horizon"
                    )
            elif (
                best_witness.remaining_actions
                >= REQUIRED_ACTION_MARGIN
            ):
                status = FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN
                message = "Joint witness retains at least five actions"
            else:
                status = FeasibilityStatus.JOINT_WITNESS_TIGHT
                message = "Joint witness exists but retains fewer than five actions"
            return SpatiotemporalFeasibilityReport(
                start_id=self.blueprint.start_id,
                visibility_stratum=(
                    self.blueprint.visibility_stratum
                ),
                status=status,
                cue_path_found=cue_path_found,
                goal_view_path_found=goal_path_found,
                cue_then_goal_path_found=best_witness is not None,
                roadmap_digest=self._roadmap_digest,
                roadmap_nodes=len(self.roadmap.nodes),
                roadmap_edges=self.roadmap.edge_count,
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
                    ("roadmap", roadmap_seconds),
                    ("goal_visibility", goal_seconds),
                    ("temporal_cue_search", temporal_seconds),
                    ("auditor_total", time.perf_counter() - run_started),
                ),
                witness=best_witness,
                message=message,
            )
        except Exception as error:
            return SpatiotemporalFeasibilityReport(
                start_id=self.blueprint.start_id,
                visibility_stratum=(
                    self.blueprint.visibility_stratum
                ),
                status=FeasibilityStatus.UNKNOWN,
                cue_path_found=False,
                goal_view_path_found=False,
                cue_then_goal_path_found=False,
                roadmap_digest=self._roadmap_digest,
                roadmap_nodes=(
                    0 if self.roadmap is None else len(self.roadmap.nodes)
                ),
                roadmap_edges=(
                    0 if self.roadmap is None
                    else self.roadmap.edge_count
                ),
                queried_steps=0,
                minimum_ordered_actions=None,
                cue_diagnostics=tuple(
                    sorted(self._cue_diagnostics.items())
                ),
                phase_seconds=(
                    ("roadmap", roadmap_seconds),
                    ("goal_visibility", goal_seconds),
                    ("temporal_cue_search", temporal_seconds),
                    ("auditor_total", time.perf_counter() - run_started),
                ),
                witness=None,
                message=str(error),
            )
