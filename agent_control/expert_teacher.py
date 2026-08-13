"""Cue-grounded Stage 2E expert without hidden-event access.

Scenario and visibility truth enter only through ``VisibleTrackGrounder``.
The expert itself consumes RGB-D-derived start-local tracks, odometry, and a
collision-only local geometry facade.
"""

import heapq
import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .dronesim_client import VisibilityTargetRole
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
    _assess_target_view,
    pair_view_matrices,
)


BELIEF_RADIUS_METERS = 120.0
BELIEF_CELL_METERS = 4.0
LOCAL_SUBGOAL_MAX_METERS = 20.0
LOCAL_PLAN_MAX_ACTIONS = 32
LOCAL_PLAN_MAX_EXPANDED_STATES = 12000
LOCAL_PLAN_TIMEOUT_SECONDS = 15.0
BELIEF_LIKELIHOOD_FLOOR = 0.05
FIRETRUCK_SIGMA_DEGREES = 35.0
PEDESTRIAN_SIGMA_DEGREES = 55.0
PEDESTRIAN_GROUP_RADIUS_METERS = 10.0
PEDESTRIAN_GROUP_HEADING_DEGREES = 20.0
REACQUIRE_AFTER_STEPS = 4
STATIC_CONFIRM_STEPS = 8
SCAN_MAX_ACTIONS = 6
SOURCE_CONFIRMATION_MAX_FAILURES = 2


class ExpertIntent(str, Enum):
    SEARCH_CUE = "SEARCH_CUE"
    CONFIRM_MOTION = "CONFIRM_MOTION"
    FOLLOW_BELIEF = "FOLLOW_BELIEF"
    REACQUIRE_CUE = "REACQUIRE_CUE"
    AVOID_COLLISION = "AVOID_COLLISION"
    VERIFY_SOURCE = "VERIFY_SOURCE"
    STOP = "STOP"


class ExpertGenerationError(RuntimeError):
    pass


class LocalPlanNotFound(ExpertGenerationError):
    pass


@dataclass(frozen=True)
class AgentEpisodeSpec:
    start_id: int
    horizon_steps: int
    activity_radius_m: float
    activity_vertical_m: float
    forward_step_m: float
    vertical_step_m: float
    yaw_step_degrees: float
    simulation_step_ms: int

    @classmethod
    def from_blueprint(cls, blueprint):
        return cls(
            start_id=int(blueprint.start_id),
            horizon_steps=int(blueprint.action_spec.horizon_steps),
            activity_radius_m=float(
                blueprint.activity_bounds.horizontal_radius_m
            ),
            activity_vertical_m=float(
                blueprint.activity_bounds.vertical_delta_m
            ),
            forward_step_m=float(
                blueprint.action_spec.forward_step_m
            ),
            vertical_step_m=float(
                blueprint.action_spec.vertical_step_m
            ),
            yaw_step_degrees=float(
                blueprint.action_spec.yaw_step_degrees
            ),
            simulation_step_ms=int(
                blueprint.action_spec.simulation_step_ms
            ),
        )


@dataclass(frozen=True)
class GroundedTrackObservation:
    track_id: int
    semantic_class: str
    position_local: tuple
    view_name: str
    projected_bbox: tuple
    supporting_pixels: tuple


@dataclass(frozen=True)
class GroundedFrame:
    step_index: int
    tracks: tuple


@dataclass(frozen=True)
class MotionEvidence:
    track_id: int
    semantic_class: str
    position_local: tuple
    displacement_local: tuple
    displacement_m: float
    inferred_event_direction: tuple
    update_weight: float


@dataclass(frozen=True)
class StructuredAwareness:
    step_index: int
    visible_tracks: tuple
    motion_evidence: tuple
    belief_entropy: float
    primary_mode_id: int | None
    primary_mode_mass: float
    belief_ambiguous: bool
    primary_hypothesis_local: tuple | None
    supporting_track_ids: tuple
    contradicting_track_ids: tuple
    intent: ExpertIntent
    temporary_subgoal_local: tuple | None
    scan_center_yaw: float | None
    scan_step: int | None
    scan_exit_reason: str | None
    planner_replanned: bool
    planner_remaining_actions: int
    planner_failure: str | None
    source_confirmation_failures: int
    action_name: str


@dataclass(frozen=True)
class ExpertDecision:
    action: object
    awareness: StructuredAwareness
    belief: np.ndarray


def _angle_delta_degrees(target, source):
    return (float(target) - float(source) + 180.0) % 360.0 - 180.0


def _yaw_toward_local(origin, target):
    delta_forward = float(target[0]) - float(origin[0])
    delta_right = float(target[1]) - float(origin[1])
    # GTA positive yaw turns left. In the start-local frame, whose positive
    # Y axis points right, a yaw delta therefore has heading
    # (cos(yaw), -sin(yaw)).
    return math.degrees(math.atan2(-delta_right, delta_forward))


def _action_pose_local(position, yaw_degrees, action):
    position = np.asarray(position, dtype=np.float64).copy()
    yaw = float(yaw_degrees)
    if isinstance(action, ForwardAction):
        radians = math.radians(yaw)
        position[0] += math.cos(radians) * TASK_FORWARD_STEP_METERS
        position[1] -= math.sin(radians) * TASK_FORWARD_STEP_METERS
    elif isinstance(action, AscendAction):
        position[2] += TASK_VERTICAL_STEP_METERS
    elif isinstance(action, DescendAction):
        position[2] -= TASK_VERTICAL_STEP_METERS
    elif isinstance(action, TurnLeftAction):
        yaw += TASK_YAW_STEP_DEGREES
    elif isinstance(action, TurnRightAction):
        yaw -= TASK_YAW_STEP_DEGREES
    else:
        raise TypeError(f"Unsupported local-planner action {action!r}")
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return tuple(float(value) for value in position), yaw


class LocalGeometryFacade:
    """Convert start-local geometry requests without exposing world truth."""

    def __init__(self, client, lockstep, blueprint):
        self._client = client
        self._lockstep = lockstep
        self._blueprint = blueprint
        self._cache_instant = None
        self._segment_cache = {}
        self.requested_segments = 0
        self.queried_segments = 0
        self.cache_hits = 0
        self.batch_queries = 0
        self.query_seconds = 0.0

    @staticmethod
    def _segment_key(start, end):
        return tuple(
            round(float(value), 3)
            for point in (start, end)
            for value in point
        )

    def segments_clear(self, local_segments):
        local_segments = tuple(local_segments)
        if not local_segments:
            return ()
        clock = self._lockstep.snapshot
        instant = (
            int(clock.session_id),
            int(clock.step_index),
            int(clock.game_timer_ms),
        )
        if instant != self._cache_instant:
            self._cache_instant = instant
            self._segment_cache.clear()

        keys = []
        missing = {}
        for start, end in local_segments:
            key = self._segment_key(start, end)
            keys.append(key)
            if key not in self._segment_cache and key not in missing:
                missing[key] = (
                    self._blueprint.local_to_world(start),
                    self._blueprint.local_to_world(end),
                )
        self.requested_segments += len(keys)
        self.queried_segments += len(missing)
        self.cache_hits += len(keys) - len(missing)
        if missing:
            missing_keys = tuple(missing)
            world_segments = tuple(missing.values())
            query_started = time.perf_counter()
            result = self._client.probe_camera_geometry_batch(
                self._lockstep.session_id,
                points=[segment[1] for segment in world_segments],
                segments=world_segments,
                timeout=30.0,
            )
            self.query_seconds += time.perf_counter() - query_started
            self.batch_queries += 1
            if (
                result.lockstep_session_id != clock.session_id
                or result.step_index != clock.step_index
                or result.game_timer_ms != clock.game_timer_ms
            ):
                raise ExpertGenerationError(
                    "Collision geometry does not belong to the frozen instant"
                )
            for key, point_clear, segment_clear in zip(
                missing_keys,
                result.point_clear,
                result.segment_clear,
            ):
                self._segment_cache[key] = bool(
                    point_clear and segment_clear
                )
        return tuple(self._segment_cache[key] for key in keys)

    def action_clear(self, position, yaw, action):
        if isinstance(action, (TurnLeftAction, TurnRightAction)):
            return True
        target, _ = _action_pose_local(position, yaw, action)
        return bool(self.segments_clear(((position, target),))[0])


class StrictLocalAStar:
    """Bounded strict-action A* to an event-independent local subgoal."""

    def __init__(
        self,
        maximum_actions=LOCAL_PLAN_MAX_ACTIONS,
        maximum_expanded_states=LOCAL_PLAN_MAX_EXPANDED_STATES,
        timeout_seconds=LOCAL_PLAN_TIMEOUT_SECONDS,
    ):
        self.maximum_actions = int(maximum_actions)
        self.maximum_expanded_states = int(maximum_expanded_states)
        self.timeout_seconds = float(timeout_seconds)
        if self.maximum_actions <= 0:
            raise ValueError("maximum_actions must be positive")
        if self.maximum_expanded_states <= 0:
            raise ValueError("maximum_expanded_states must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    @staticmethod
    def _key(position, yaw):
        return (
            *(int(round(float(value) * 4.0)) for value in position),
            int(round(float(yaw) / TASK_YAW_STEP_DEGREES)) % 24,
        )

    @staticmethod
    def _heuristic(position, target):
        horizontal = math.dist(position[:2], target[:2])
        vertical = abs(float(position[2]) - float(target[2]))
        return int(math.ceil(horizontal / TASK_FORWARD_STEP_METERS)) + int(
            math.ceil(vertical / TASK_VERTICAL_STEP_METERS)
        )

    @staticmethod
    def _heading_tiebreak(position, yaw, target):
        if math.dist(position[:2], target[:2]) <= 1.0e-9:
            return 0.0
        target_yaw = _yaw_toward_local(position, target)
        return abs(_angle_delta_degrees(target_yaw, yaw))

    @staticmethod
    def _goal(position, target):
        return (
            math.dist(position[:2], target[:2])
            <= TASK_FORWARD_STEP_METERS
            and abs(float(position[2]) - float(target[2]))
            <= TASK_VERTICAL_STEP_METERS * 0.5
        )

    @staticmethod
    def _inside_activity(position, activity_radius_m, activity_vertical_m):
        return (
            math.hypot(float(position[0]), float(position[1]))
            <= float(activity_radius_m)
            and abs(float(position[2])) <= float(activity_vertical_m)
        )

    def plan(
        self,
        position,
        yaw,
        target,
        geometry,
        activity_radius_m,
        activity_vertical_m,
    ):
        position = tuple(float(value) for value in position)
        target = tuple(float(value) for value in target)
        if not self._inside_activity(
            position,
            activity_radius_m,
            activity_vertical_m,
        ):
            raise LocalPlanNotFound(
                "LOCAL_PLAN_NOT_FOUND: current pose is outside the activity "
                "bounds"
            )
        started = time.perf_counter()
        expanded = 0
        collision_rejections = 0
        bounds_rejections = 0
        nodes = [(position, float(yaw), -1, None, 0)]
        queue = [
            (
                self._heuristic(position, target),
                self._heading_tiebreak(position, yaw, target),
                0,
                0,
            )
        ]
        best_cost = {self._key(position, yaw): 0}
        serial = 1

        def reconstruct(index):
            actions = []
            while index >= 0:
                _position, _yaw, parent, action, _cost = nodes[index]
                if action is not None:
                    actions.append(action)
                index = parent
            return tuple(reversed(actions))

        while queue:
            elapsed = time.perf_counter() - started
            if (
                expanded >= self.maximum_expanded_states
                or elapsed >= self.timeout_seconds
            ):
                raise LocalPlanNotFound(
                    "LOCAL_PLAN_NOT_FOUND: bounded A* stopped before finding "
                    f"a route; actions<={self.maximum_actions}, "
                    f"expanded={expanded}/{self.maximum_expanded_states}, "
                    f"elapsed={elapsed:.2f}/{self.timeout_seconds:.2f}s, "
                    f"collision_rejections={collision_rejections}, "
                    f"bounds_rejections={bounds_rejections}"
                )
            expansion_batch = []
            batch_limit = min(
                42,
                self.maximum_expanded_states - expanded,
            )
            while queue and len(expansion_batch) < batch_limit:
                (
                    _estimate,
                    _heading_error,
                    _order,
                    node_index,
                ) = heapq.heappop(queue)
                node_position, node_yaw, _parent, _action, cost = nodes[
                    node_index
                ]
                if best_cost.get(
                    self._key(node_position, node_yaw)
                ) != cost:
                    continue
                if self._goal(node_position, target):
                    return reconstruct(node_index)
                if cost < self.maximum_actions:
                    expansion_batch.append(node_index)
            expanded += len(expansion_batch)

            candidates = []
            for node_index in expansion_batch:
                node_position, node_yaw, _parent, _action, cost = nodes[
                    node_index
                ]
                for action in (
                    TurnLeftAction(),
                    TurnRightAction(),
                    ForwardAction(),
                    AscendAction(),
                    DescendAction(),
                ):
                    child_position, child_yaw = _action_pose_local(
                        node_position,
                        node_yaw,
                        action,
                    )
                    child_cost = cost + 1
                    if not self._inside_activity(
                        child_position,
                        activity_radius_m,
                        activity_vertical_m,
                    ):
                        bounds_rejections += 1
                        continue
                    if (
                        child_cost
                        + self._heuristic(child_position, target)
                        > self.maximum_actions
                    ):
                        continue
                    key = self._key(child_position, child_yaw)
                    if (
                        best_cost.get(
                            key,
                            self.maximum_actions + 1,
                        )
                        <= child_cost
                    ):
                        continue
                    candidates.append(
                        (
                            node_index,
                            node_position,
                            action,
                            child_position,
                            child_yaw,
                            child_cost,
                            key,
                        )
                    )

            motion = [
                item
                for item in candidates
                if not isinstance(
                    item[2],
                    (TurnLeftAction, TurnRightAction),
                )
            ]
            clear_motion = ()
            if motion:
                clear_motion = geometry.segments_clear(
                    (
                        (item[1], item[3])
                        for item in motion
                    )
                )
            motion_clear = {
                (item[0], item[6]): bool(clear)
                for item, clear in zip(motion, clear_motion)
            }
            for (
                node_index,
                _node_position,
                action,
                child_position,
                child_yaw,
                child_cost,
                key,
            ) in candidates:
                if (
                    not isinstance(
                        action,
                        (TurnLeftAction, TurnRightAction),
                    )
                    and not motion_clear.get(
                        (node_index, key),
                        False,
                    )
                ):
                    collision_rejections += 1
                    continue
                if best_cost.get(
                    key,
                    self.maximum_actions + 1,
                ) <= child_cost:
                    continue
                best_cost[key] = child_cost
                child_index = len(nodes)
                nodes.append(
                    (
                        child_position,
                        child_yaw,
                        node_index,
                        action,
                        child_cost,
                    )
                )
                estimate = child_cost + self._heuristic(
                    child_position,
                    target,
                )
                heapq.heappush(
                    queue,
                    (
                        estimate,
                        self._heading_tiebreak(
                            child_position,
                            child_yaw,
                            target,
                        ),
                        serial,
                        child_index,
                    ),
                )
                serial += 1
        raise LocalPlanNotFound(
            "LOCAL_PLAN_NOT_FOUND: no strict collision-free plan reaches "
            f"the temporary subgoal within {self.maximum_actions} actions; "
            f"expanded={expanded}, collision_rejections="
            f"{collision_rejections}, bounds_rejections={bounds_rejections}"
        )


class VisibleTrackGrounder:
    """Use truth only for association/pixels; recover geometry from Depth."""

    def __init__(self, blueprint, observation_spec):
        self._blueprint = blueprint
        self._spec = observation_spec
        self._track_ids = {}
        self._stable_ids = {}
        self._next_track_id = 1

    def _track_id(self, stable_id):
        stable_id = int(stable_id)
        if stable_id not in self._track_ids:
            self._track_ids[stable_id] = self._next_track_id
            self._stable_ids[self._next_track_id] = stable_id
            self._next_track_id += 1
        return self._track_ids[stable_id]

    def evaluation_stable_id(self, track_id):
        return self._stable_ids[int(track_id)]

    @staticmethod
    def _semantic_class(role):
        if role == VisibilityTargetRole.FIRE_TRUCK:
            return "FIRE_TRUCK"
        if role == VisibilityTargetRole.FLEEING_PEDESTRIAN:
            return "PEDESTRIAN"
        if role == VisibilityTargetRole.FIRE_SOURCE_VEHICLE:
            return "FIRE_SOURCE"
        return None

    @staticmethod
    def _project_samples(target, projection, view, width, height):
        positions = np.asarray(
            [sample.position for sample in target.samples],
            dtype=np.float64,
        )
        world = np.column_stack(
            (positions, np.ones(len(positions), dtype=np.float64))
        )
        view_points = (view @ world.T).T
        clip = (projection @ view_points.T).T
        valid = np.abs(clip[:, 3]) > 1.0e-12
        ndc = np.full((len(positions), 3), np.nan)
        ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
        pixels = np.column_stack(
            (
                (ndc[:, 0] + 1.0) * 0.5 * (width - 1),
                (1.0 - ndc[:, 1]) * 0.5 * (height - 1),
            )
        )
        expected_depth = -view_points[:, 2]
        return pixels, expected_depth

    @staticmethod
    def _backproject_pixel(frame, view, pixel, depth):
        projection = np.asarray(
            frame.projection_matrix,
            dtype=np.float64,
        ).reshape(4, 4)
        x_ndc = 2.0 * float(pixel[0]) / (frame.width - 1) - 1.0
        y_ndc = 1.0 - 2.0 * float(pixel[1]) / (frame.height - 1)
        view_point = np.asarray(
            (
                x_ndc * depth / projection[0, 0],
                y_ndc * depth / projection[1, 1],
                -depth,
                1.0,
            ),
            dtype=np.float64,
        )
        world = np.linalg.inv(view) @ view_point
        world = world[:3] / world[3]
        return tuple(float(value) for value in world)

    def ground(self, pair, visibility):
        matrices = pair_view_matrices(pair)
        frames = {
            "oblique": pair.oblique,
            "nadir": pair.nadir,
        }
        grounded = []
        for target in visibility.targets:
            semantic_class = self._semantic_class(target.role)
            if semantic_class is None:
                continue
            best = None
            for view_name in ("oblique", "nadir"):
                projection, view = matrices[view_name]
                assessed = _assess_target_view(
                    target,
                    (projection, view),
                    self._spec,
                )
                if not assessed.task_observable:
                    continue
                frame = frames[view_name]
                depth_array = frame.depth_array()
                pixels, expected_depths = self._project_samples(
                    target,
                    projection,
                    view,
                    frame.width,
                    frame.height,
                )
                recovered = []
                supporting_pixels = []
                for sample, pixel, expected_depth in zip(
                    target.samples,
                    pixels,
                    expected_depths,
                ):
                    if (
                        not sample.clear_line_of_sight
                        or not np.isfinite(pixel).all()
                        or expected_depth <= 0.0
                    ):
                        continue
                    x = int(round(float(pixel[0])))
                    y = int(round(float(pixel[1])))
                    if not (
                        0 <= x < frame.width
                        and 0 <= y < frame.height
                    ):
                        continue
                    measured_depth = float(depth_array[y, x])
                    tolerance = max(1.0, expected_depth * 0.05)
                    if abs(measured_depth - expected_depth) > tolerance:
                        continue
                    recovered.append(
                        self._backproject_pixel(
                            frame,
                            view,
                            (x, y),
                            measured_depth,
                        )
                    )
                    supporting_pixels.append((x, y))
                if len(recovered) < self._spec.min_clear_samples:
                    continue
                world_position = np.median(
                    np.asarray(recovered, dtype=np.float64),
                    axis=0,
                )
                local_position = self._blueprint.world_to_local(
                    world_position
                )
                candidate = (
                    len(recovered),
                    GroundedTrackObservation(
                        track_id=self._track_id(target.stable_id),
                        semantic_class=semantic_class,
                        position_local=tuple(
                            float(value) for value in local_position
                        ),
                        view_name=view_name,
                        projected_bbox=tuple(
                            float(value)
                            for value in assessed.projected_bbox
                        ),
                        supporting_pixels=tuple(supporting_pixels),
                    ),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                grounded.append(best[1])
        return GroundedFrame(
            step_index=int(pair.clock.step_index),
            tracks=tuple(
                sorted(grounded, key=lambda item: item.track_id)
            ),
        )


class SpatialBelief:
    def __init__(self):
        coordinates = np.arange(
            -BELIEF_RADIUS_METERS,
            BELIEF_RADIUS_METERS + BELIEF_CELL_METERS,
            BELIEF_CELL_METERS,
            dtype=np.float64,
        )
        forward, right = np.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        self.forward = forward
        self.right = right
        valid = (
            forward * forward + right * right
            <= BELIEF_RADIUS_METERS * BELIEF_RADIUS_METERS
        )
        self.valid = valid
        self.probability = valid.astype(np.float64)
        self.probability /= np.sum(self.probability)
        self.ambiguous = False

    def update(self, evidence):
        if not evidence:
            return
        for item in evidence:
            origin = np.asarray(item.position_local[:2], dtype=np.float64)
            direction = np.asarray(
                item.inferred_event_direction,
                dtype=np.float64,
            )
            direction_length = float(np.linalg.norm(direction))
            if direction_length <= 1.0e-9:
                continue
            direction /= direction_length
            delta_forward = self.forward - origin[0]
            delta_right = self.right - origin[1]
            lengths = np.hypot(delta_forward, delta_right)
            valid_length = lengths > 1.0e-6
            cosine = np.zeros_like(lengths)
            cosine[valid_length] = (
                delta_forward[valid_length] * direction[0]
                + delta_right[valid_length] * direction[1]
            ) / lengths[valid_length]
            angles = np.arccos(np.clip(cosine, -1.0, 1.0))
            sigma = math.radians(
                FIRETRUCK_SIGMA_DEGREES
                if item.semantic_class == "FIRE_TRUCK"
                else PEDESTRIAN_SIGMA_DEGREES
            )
            likelihood = np.exp(-0.5 * (angles / sigma) ** 2)
            likelihood = np.maximum(
                likelihood,
                BELIEF_LIKELIHOOD_FLOOR,
            )
            weighted = likelihood ** float(item.update_weight)
            self.probability *= np.where(self.valid, weighted, 0.0)
            total = float(np.sum(self.probability))
            if not math.isfinite(total) or total <= 0.0:
                raise ExpertGenerationError(
                    "BELIEF_UPDATE_FAILED: posterior is invalid"
                )
            self.probability /= total

    @property
    def entropy(self):
        values = self.probability[self.probability > 0.0]
        return float(-np.sum(values * np.log(values)))

    def primary_mode(self):
        flat_indices = np.argsort(
            self.probability.ravel()
        )[::-1]
        selected = np.zeros_like(self.valid, dtype=bool)
        mass = 0.0
        for flat_index in flat_indices:
            index = np.unravel_index(
                flat_index,
                self.probability.shape,
            )
            if not self.valid[index]:
                continue
            selected[index] = True
            mass += float(self.probability[index])
            if mass >= 0.20:
                break

        components = []
        visited = np.zeros_like(selected, dtype=bool)
        for start in zip(*np.nonzero(selected)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            cells = []
            while stack:
                cell = stack.pop()
                cells.append(cell)
                for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (
                        cell[0] + delta[0],
                        cell[1] + delta[1],
                    )
                    if (
                        0 <= neighbor[0] < selected.shape[0]
                        and 0 <= neighbor[1] < selected.shape[1]
                        and selected[neighbor]
                        and not visited[neighbor]
                    ):
                        visited[neighbor] = True
                        stack.append(neighbor)
            component_mass = sum(
                float(self.probability[cell]) for cell in cells
            )
            components.append((component_mass, cells))
        if not components:
            self.ambiguous = False
            return None, 0.0, None
        components.sort(
            key=lambda item: (
                -item[0],
                min(item[1]),
            )
        )
        component_mass, cells = components[0]
        self.ambiguous = (
            len(components) > 1
            and components[1][0] >= component_mass * 0.8
        )
        weights = np.asarray(
            [self.probability[cell] for cell in cells],
            dtype=np.float64,
        )
        forwards = np.asarray(
            [self.forward[cell] for cell in cells],
            dtype=np.float64,
        )
        rights = np.asarray(
            [self.right[cell] for cell in cells],
            dtype=np.float64,
        )
        center = (
            float(np.average(forwards, weights=weights)),
            float(np.average(rights, weights=weights)),
        )
        coarse_forward = int(round(center[0] / 12.0))
        coarse_right = int(round(center[1] / 12.0))
        mode_id = (
            (coarse_forward + 32) * 128
            + coarse_right
            + 32
        )
        return mode_id, float(component_mass), center


class CueGroundedExpert:
    def __init__(self, episode_spec, geometry):
        if not isinstance(episode_spec, AgentEpisodeSpec):
            raise TypeError("episode_spec must be AgentEpisodeSpec")
        self.spec = episode_spec
        self.geometry = geometry
        self.belief = SpatialBelief()
        self.planner = StrictLocalAStar()
        self._previous_tracks = {}
        self._track_update_counts = {}
        self._cached_actions = []
        self._cached_subgoal = None
        self._cached_intent = None
        self._cached_mode_id = None
        self._last_evidence_step = None
        self._seen_evidence_ids = set()
        self._scan_intent = None
        self._scan_mode_id = None
        self._scan_center_yaw = None
        self._scan_step = 0
        self._scan_aligned = False
        self._completed_scan = None
        self._consecutive_scan_turns = 0
        self._scan_transition_used = set()
        self._search_altitude_change_used = False
        self._planner_retry_signature = None
        self._source_confirmation_track = None
        self._source_confirmation_failures = 0

    @staticmethod
    def _action_name(action):
        return type(action).__name__.removesuffix("Action").upper()

    def _motion_evidence(self, grounded):
        raw = []
        for track in grounded.tracks:
            if track.semantic_class == "FIRE_SOURCE":
                continue
            previous = self._previous_tracks.get(track.track_id)
            if previous is None:
                continue
            displacement = np.asarray(
                track.position_local[:2],
                dtype=np.float64,
            ) - np.asarray(
                previous.position_local[:2],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(displacement))
            if distance < TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS:
                continue
            inferred = (
                displacement
                if track.semantic_class == "FIRE_TRUCK"
                else -displacement
            )
            raw.append((track, displacement, distance, inferred))

        weights = {}
        pedestrian_indices = [
            index
            for index, item in enumerate(raw)
            if item[0].semantic_class == "PEDESTRIAN"
        ]
        unassigned = set(pedestrian_indices)
        while unassigned:
            seed = min(unassigned)
            group = [seed]
            unassigned.remove(seed)
            seed_item = raw[seed]
            seed_heading = math.degrees(
                math.atan2(
                    seed_item[1][1],
                    seed_item[1][0],
                )
            )
            for candidate in sorted(tuple(unassigned)):
                item = raw[candidate]
                heading = math.degrees(
                    math.atan2(item[1][1], item[1][0])
                )
                if (
                    math.dist(
                        seed_item[0].position_local[:2],
                        item[0].position_local[:2],
                    )
                    <= PEDESTRIAN_GROUP_RADIUS_METERS
                    and abs(
                        _angle_delta_degrees(
                            heading,
                            seed_heading,
                        )
                    )
                    <= PEDESTRIAN_GROUP_HEADING_DEGREES
                ):
                    group.append(candidate)
                    unassigned.remove(candidate)
            for index in group:
                weights[index] = 0.6 / len(group)

        evidence = []
        for index, (track, displacement, distance, inferred) in enumerate(raw):
            count = self._track_update_counts.get(track.track_id, 0) + 1
            self._track_update_counts[track.track_id] = count
            base = 1.0 if track.semantic_class == "FIRE_TRUCK" else weights[
                index
            ]
            evidence.append(
                MotionEvidence(
                    track_id=track.track_id,
                    semantic_class=track.semantic_class,
                    position_local=track.position_local,
                    displacement_local=(
                        float(displacement[0]),
                        float(displacement[1]),
                    ),
                    displacement_m=distance,
                    inferred_event_direction=(
                        float(inferred[0]),
                        float(inferred[1]),
                    ),
                    update_weight=base / math.sqrt(count),
                )
            )
        return tuple(evidence)

    @staticmethod
    def _source_track(grounded):
        sources = [
            track
            for track in grounded.tracks
            if track.semantic_class == "FIRE_SOURCE"
        ]
        if len(sources) > 1:
            raise ExpertGenerationError(
                "Multiple fire-source tracks are grounded"
            )
        return sources[0] if sources else None

    @staticmethod
    def _estimated_agl(observation):
        view = observation.nadir
        depth = np.frombuffer(view.depth, dtype="<f4").reshape(
            view.height,
            view.width,
        )
        center_y = view.height // 2
        center_x = view.width // 2
        patch = depth[
            max(0, center_y - 2) : center_y + 3,
            max(0, center_x - 2) : center_x + 3,
        ]
        finite = patch[np.isfinite(patch) & (patch > 0.0)]
        return (
            float(np.median(finite))
            if finite.size
            else math.inf
        )

    @staticmethod
    def _temporary_subgoal(position, hypothesis):
        delta = np.asarray(
            (
                hypothesis[0] - position[0],
                hypothesis[1] - position[1],
            ),
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(delta))
        if distance > LOCAL_SUBGOAL_MAX_METERS:
            delta *= LOCAL_SUBGOAL_MAX_METERS / distance
        return (
            float(position[0] + delta[0]),
            float(position[1] + delta[1]),
            float(position[2]),
        )

    def _reset_scan(self, intent, mode_id, position, yaw, hypothesis):
        self._scan_intent = intent
        self._scan_mode_id = mode_id
        self._scan_center_yaw = (
            float(yaw)
            if hypothesis is None
            else _yaw_toward_local(position, hypothesis)
        )
        self._scan_step = 0
        self._scan_aligned = False
        self._completed_scan = None
        self._consecutive_scan_turns = 0

    def _scan_action(self, intent, mode_id, position, yaw, hypothesis):
        signature = (intent, mode_id)
        if (
            self._scan_intent != intent
            or self._scan_mode_id != mode_id
            or self._scan_center_yaw is None
        ):
            self._reset_scan(
                intent,
                mode_id,
                position,
                yaw,
                hypothesis,
            )
        center_error = _angle_delta_degrees(
            self._scan_center_yaw,
            yaw,
        )
        if not self._scan_aligned:
            if abs(center_error) > TASK_YAW_STEP_DEGREES * 0.5:
                action = (
                    TurnLeftAction()
                    if center_error > 0.0
                    else TurnRightAction()
                )
                self._scan_step += 1
                self._consecutive_scan_turns += 1
                if self._scan_step == SCAN_MAX_ACTIONS:
                    self._completed_scan = signature
                return action
            self._scan_aligned = True

        scan_actions = (
            TurnLeftAction(),
            TurnLeftAction(),
            TurnRightAction(),
            TurnRightAction(),
            TurnRightAction(),
            TurnLeftAction(),
        )
        if self._scan_step < SCAN_MAX_ACTIONS:
            action = scan_actions[self._scan_step]
            self._scan_step += 1
            self._consecutive_scan_turns += 1
            if self._scan_step == SCAN_MAX_ACTIONS:
                self._completed_scan = signature
            return action
        raise ExpertGenerationError(
            "Finite cue scan was requested after completion"
        )

    def _inside_activity(self, position):
        return (
            math.hypot(float(position[0]), float(position[1]))
            <= self.spec.activity_radius_m
            and abs(float(position[2])) <= self.spec.activity_vertical_m
        )

    def decide(self, observation, grounded):
        step_index = int(grounded.step_index)
        position = tuple(
            float(value)
            for value in observation.odometry.position_local
        )
        yaw = float(observation.odometry.yaw_from_start_degrees)
        evidence = self._motion_evidence(grounded)
        self.belief.update(evidence)
        if evidence:
            self._last_evidence_step = step_index
        evidence_ids = frozenset(item.track_id for item in evidence)
        new_evidence_ids = evidence_ids - self._seen_evidence_ids
        new_evidence = bool(new_evidence_ids)
        if new_evidence:
            self._scan_intent = None
            self._completed_scan = None
            self._scan_transition_used.clear()

        source = self._source_track(grounded)
        if source is None and self._source_confirmation_track is not None:
            self._source_confirmation_track = None
            self._source_confirmation_failures += 1
            if (
                self._source_confirmation_failures
                >= SOURCE_CONFIRMATION_MAX_FAILURES
            ):
                raise ExpertGenerationError(
                    "SOURCE_CONFIRMATION_UNSTABLE: the grounded fire "
                    "source disappeared after two confirmation HOLDs"
                )
        mode_id, mode_mass, hypothesis = self.belief.primary_mode()
        supporting_track_ids = []
        contradicting_track_ids = []
        if hypothesis is not None:
            hypothesis_xy = np.asarray(
                hypothesis,
                dtype=np.float64,
            )
            for item in evidence:
                origin = np.asarray(
                    item.position_local[:2],
                    dtype=np.float64,
                )
                expected = np.asarray(
                    item.inferred_event_direction,
                    dtype=np.float64,
                )
                candidate = hypothesis_xy - origin
                if float(np.dot(expected, candidate)) >= 0.0:
                    supporting_track_ids.append(item.track_id)
                else:
                    contradicting_track_ids.append(item.track_id)
        replanned = False
        subgoal = None
        scan_exit_reason = None
        planner_failure = None
        if source is not None:
            if self._source_confirmation_track == source.track_id:
                intent = ExpertIntent.STOP
                action = StopAction(source.position_local)
            else:
                intent = ExpertIntent.VERIFY_SOURCE
                action = HoldAction()
                self._source_confirmation_track = source.track_id
            self._cached_actions.clear()
        elif evidence and self.belief.ambiguous:
            intent = ExpertIntent.REACQUIRE_CUE
        elif evidence:
            intent = ExpertIntent.FOLLOW_BELIEF
        elif any(
            track.semantic_class in ("FIRE_TRUCK", "PEDESTRIAN")
            for track in grounded.tracks
        ) and step_index <= STATIC_CONFIRM_STEPS:
            intent = ExpertIntent.CONFIRM_MOTION
        elif self._last_evidence_step is None:
            intent = ExpertIntent.SEARCH_CUE
        elif step_index - self._last_evidence_step >= REACQUIRE_AFTER_STEPS:
            intent = ExpertIntent.REACQUIRE_CUE
        else:
            intent = ExpertIntent.FOLLOW_BELIEF

        if source is None:
            if intent == ExpertIntent.CONFIRM_MOTION:
                action = HoldAction()
                self._cached_actions.clear()
            elif intent in (
                ExpertIntent.SEARCH_CUE,
                ExpertIntent.REACQUIRE_CUE,
            ):
                signature = (intent, mode_id)
                if (
                    self._completed_scan == signature
                    or self._consecutive_scan_turns
                    >= SCAN_MAX_ACTIONS
                ):
                    if (
                        intent == ExpertIntent.REACQUIRE_CUE
                        and hypothesis is not None
                    ):
                        intent = ExpertIntent.FOLLOW_BELIEF
                        scan_exit_reason = "FINITE_SCAN_COMPLETE_FOLLOW_BELIEF"
                    else:
                        if signature in self._scan_transition_used:
                            raise ExpertGenerationError(
                                "CUE_SEARCH_EXHAUSTED: no RGB-D-grounded "
                                "response track was found after two finite "
                                f"scans for {intent.value} mode={mode_id}"
                            )
                        candidate, _candidate_yaw = _action_pose_local(
                            position,
                            yaw,
                            AscendAction(),
                        )
                        if (
                            not self._search_altitude_change_used
                            and self._estimated_agl(observation) < 45.0
                            and self._inside_activity(candidate)
                            and self.geometry.action_clear(
                                position,
                                yaw,
                                AscendAction(),
                            )
                        ):
                            action = AscendAction()
                            scan_exit_reason = "FINITE_SCAN_COMPLETE_ASCEND"
                            self._search_altitude_change_used = True
                        else:
                            action = HoldAction()
                            scan_exit_reason = "FINITE_SCAN_COMPLETE_HOLD"
                        self._scan_transition_used.add(signature)
                        self._scan_intent = None
                        self._completed_scan = None
                        self._consecutive_scan_turns = 0
                if intent in (
                    ExpertIntent.SEARCH_CUE,
                    ExpertIntent.REACQUIRE_CUE,
                ) and scan_exit_reason is None:
                    action = self._scan_action(
                        intent,
                        mode_id,
                        position,
                        yaw,
                        hypothesis if self._last_evidence_step is not None else None,
                    )
                self._cached_actions.clear()
            if intent == ExpertIntent.FOLLOW_BELIEF:
                if hypothesis is None:
                    raise ExpertGenerationError(
                        "No belief hypothesis is available"
                    )
                subgoal = self._temporary_subgoal(
                    position,
                    hypothesis,
                )
                needs_replan = (
                    not self._cached_actions
                    or self._cached_intent != intent
                    or self._cached_subgoal is None
                    or math.dist(
                        self._cached_subgoal,
                        subgoal,
                    )
                    > BELIEF_CELL_METERS
                )
                if (
                    not needs_replan
                    and not self.geometry.action_clear(
                        position,
                        yaw,
                        self._cached_actions[0],
                    )
                ):
                    needs_replan = True
                if needs_replan:
                    retry_signature = (
                        mode_id,
                        tuple(round(value, 3) for value in subgoal),
                    )
                    try:
                        self._cached_actions = list(self.planner.plan(
                            position,
                            yaw,
                            subgoal,
                            self.geometry,
                            self.spec.activity_radius_m,
                            self.spec.activity_vertical_m,
                        ))
                    except LocalPlanNotFound as error:
                        if self._planner_retry_signature == retry_signature:
                            raise ExpertGenerationError(
                                "LOCAL_PLAN_RETRY_FAILED: the same belief "
                                f"subgoal failed after one observation; {error}"
                            ) from error
                        self._planner_retry_signature = retry_signature
                        self._cached_actions.clear()
                        intent = ExpertIntent.AVOID_COLLISION
                        action = HoldAction()
                        planner_failure = str(error)
                        replanned = True
                    else:
                        self._planner_retry_signature = None
                        self._cached_subgoal = subgoal
                        self._cached_intent = intent
                        self._cached_mode_id = mode_id
                        replanned = True
                if (
                    intent == ExpertIntent.FOLLOW_BELIEF
                    and not self._cached_actions
                ):
                    raise ExpertGenerationError(
                        "Local planner returned an empty non-terminal plan"
                    )
                if intent == ExpertIntent.FOLLOW_BELIEF:
                    action = self._cached_actions.pop(0)

        self._previous_tracks = {
            track.track_id: track for track in grounded.tracks
        }
        self._seen_evidence_ids.update(evidence_ids)
        if intent not in (
            ExpertIntent.SEARCH_CUE,
            ExpertIntent.REACQUIRE_CUE,
        ):
            self._consecutive_scan_turns = 0
        awareness = StructuredAwareness(
            step_index=step_index,
            visible_tracks=tuple(
                (
                    track.track_id,
                    track.semantic_class,
                    track.view_name,
                    track.projected_bbox,
                    track.position_local,
                )
                for track in grounded.tracks
            ),
            motion_evidence=evidence,
            belief_entropy=self.belief.entropy,
            primary_mode_id=mode_id,
            primary_mode_mass=mode_mass,
            belief_ambiguous=self.belief.ambiguous,
            primary_hypothesis_local=(
                None
                if hypothesis is None
                else (
                    float(hypothesis[0]),
                    float(hypothesis[1]),
                )
            ),
            supporting_track_ids=tuple(supporting_track_ids),
            contradicting_track_ids=tuple(
                contradicting_track_ids
            ),
            intent=intent,
            temporary_subgoal_local=subgoal,
            scan_center_yaw=self._scan_center_yaw,
            scan_step=(
                self._scan_step
                if intent in (
                    ExpertIntent.SEARCH_CUE,
                    ExpertIntent.REACQUIRE_CUE,
                )
                else None
            ),
            scan_exit_reason=scan_exit_reason,
            planner_replanned=replanned,
            planner_remaining_actions=len(self._cached_actions),
            planner_failure=planner_failure,
            source_confirmation_failures=(
                self._source_confirmation_failures
            ),
            action_name=self._action_name(action),
        )
        return ExpertDecision(
            action=action,
            awareness=awareness,
            belief=self.belief.probability.copy(),
        )
