"""Evaluation-only visibility and task-start construction.

Absolute GTA coordinates and visibility truth in this module must not be
included in an agent observation.
"""

import hashlib
import math
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .dronesim_client import (
    DroneSimCommandError,
    LockstepRgbdPair,
    LockstepSession,
    NADIR_PITCH_DEGREES,
    OBLIQUE_PITCH_DEGREES,
    ScenarioEntityRole,
    ScenarioLifecycle,
    VisibilitySnapshot,
    VisibilityTargetRole,
)


TASK_FORWARD_STEP_METERS = 2.0
TASK_VERTICAL_STEP_METERS = 2.0
TASK_YAW_STEP_DEGREES = 15.0
TASK_STEP_MILLISECONDS = 250
TASK_HORIZON_STEPS = 65
TASK_ACTIVITY_RADIUS_METERS = 120.0
TASK_ACTIVITY_VERTICAL_METERS = 40.0
TASK_GOAL_VIEW_HEIGHTS_METERS = (
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
)
TASK_MIN_EVENT_DISTANCE_METERS = 40.0
TASK_MAX_EVENT_DISTANCE_METERS = 60.0
TASK_MIN_ALTITUDE_AGL_METERS = 25.0
TASK_MAX_ALTITUDE_AGL_METERS = 60.0
TASK_MIN_PROJECTED_SPAN_PIXELS = 24.0
TASK_MIN_CLEAR_SAMPLES = 4
TASK_IMAGE_BORDER_MARGIN_PIXELS = 12.0
TASK_LOCALIZATION_TOLERANCE_METERS = 5.0
TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS = 0.4
TASK_MAX_START_CANDIDATES = 256


class StartVisibilityStratum(IntEnum):
    POTENTIAL_CUE_VISIBLE = 1
    # Historical Stage 2C/2D spelling. New code must use the explicit name.
    CUE_VISIBLE = POTENTIAL_CUE_VISIBLE
    CUE_HIDDEN = 2


@dataclass(frozen=True)
class ObservationSpec:
    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    oblique_pitch_degrees: float = OBLIQUE_PITCH_DEGREES
    nadir_pitch_degrees: float = NADIR_PITCH_DEGREES
    min_projected_span_pixels: float = (
        TASK_MIN_PROJECTED_SPAN_PIXELS
    )
    min_clear_samples: int = TASK_MIN_CLEAR_SAMPLES
    image_border_margin_pixels: float = (
        TASK_IMAGE_BORDER_MARGIN_PIXELS
    )
    weather: str = "EXTRASUNNY"
    hour: int = 12
    minute: int = 0
    second: int = 0

    @classmethod
    def from_pair(cls, pair):
        if not isinstance(pair, LockstepRgbdPair):
            raise TypeError("pair must be a LockstepRgbdPair")
        left = pair.oblique
        right = pair.nadir
        if (
            left.width != right.width
            or left.height != right.height
            or left.fov_degrees != right.fov_degrees
            or left.near_clip != right.near_clip
            or left.far_clip != right.far_clip
        ):
            raise ValueError(
                "Dual-view pair does not have one observation calibration"
            )
        return cls(
            width=left.width,
            height=left.height,
            fov_degrees=left.fov_degrees,
            near_clip=left.near_clip,
            far_clip=left.far_clip,
        )

    def __post_init__(self):
        values = (
            self.fov_degrees,
            self.near_clip,
            self.far_clip,
            self.oblique_pitch_degrees,
            self.nadir_pitch_degrees,
            self.min_projected_span_pixels,
            self.image_border_margin_pixels,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("ObservationSpec contains non-finite values")
        if self.width < 2 or self.height < 2:
            raise ValueError("ObservationSpec dimensions are invalid")
        if not 0 < self.fov_degrees < 180:
            raise ValueError("ObservationSpec FOV is invalid")
        if not 0 < self.near_clip < self.far_clip:
            raise ValueError("ObservationSpec clip planes are invalid")
        if self.min_projected_span_pixels <= 0:
            raise ValueError("Projected-span threshold must be positive")
        if self.min_clear_samples <= 0:
            raise ValueError("Clear-sample threshold must be positive")
        if self.image_border_margin_pixels < 0:
            raise ValueError("Image-border margin cannot be negative")
        if (
            self.image_border_margin_pixels * 2 >= self.width
            or self.image_border_margin_pixels * 2 >= self.height
        ):
            raise ValueError(
                "Image-border margin leaves no observable image area"
            )


@dataclass(frozen=True)
class TaskActionSpec:
    forward_step_m: float = TASK_FORWARD_STEP_METERS
    vertical_step_m: float = TASK_VERTICAL_STEP_METERS
    yaw_step_degrees: float = TASK_YAW_STEP_DEGREES
    simulation_step_ms: int = TASK_STEP_MILLISECONDS
    horizon_steps: int = TASK_HORIZON_STEPS
    strictly_discrete: bool = True
    hold_advances_simulation: bool = True
    stop_consumes_action: bool = True


@dataclass(frozen=True)
class ActivityBounds:
    horizontal_radius_m: float = TASK_ACTIVITY_RADIUS_METERS
    vertical_delta_m: float = TASK_ACTIVITY_VERTICAL_METERS


@dataclass(frozen=True)
class ViewTargetVisibility:
    in_frustum_samples: int
    clear_in_frustum_samples: int
    projected_bbox: tuple | None
    projected_span_pixels: float
    inside_image_margin: bool
    task_observable: bool


@dataclass(frozen=True)
class TargetVisibility:
    stable_id: int
    role: VisibilityTargetRole
    oblique: ViewTargetVisibility
    nadir: ViewTargetVisibility


@dataclass(frozen=True)
class VisibilityAssessment:
    source_vehicle_has_line_of_sight: bool
    fire_envelope_has_line_of_sight: bool
    fire_envelope_clear_fraction: float
    fire_envelope_task_observable: bool
    event_initially_hidden: bool
    event_task_observable: bool
    cue_task_observable: bool
    targets: tuple


@dataclass(frozen=True)
class TaskStartBlueprint:
    start_id: int
    scenario_blueprint_id: int
    start_seed: int
    candidate_index: int
    absolute_pose: tuple
    altitude_agl: float
    event_distance: float
    event_bearing_body_degrees: float
    visibility_stratum: StartVisibilityStratum
    observation_spec: ObservationSpec
    activity_bounds: ActivityBounds
    action_spec: TaskActionSpec

    def local_to_world(self, local_position):
        forward, right, up = _local_axes(self.absolute_pose[5])
        local = _finite_vector3("local_position", local_position)
        origin = np.asarray(self.absolute_pose[:3], dtype=np.float64)
        world = (
            origin
            + forward * local[0]
            + right * local[1]
            + up * local[2]
        )
        return tuple(float(value) for value in world)

    def world_to_local(self, world_position):
        forward, right, up = _local_axes(self.absolute_pose[5])
        world = _finite_vector3("world_position", world_position)
        origin = np.asarray(self.absolute_pose[:3], dtype=np.float64)
        delta = world - origin
        return (
            float(np.dot(delta, forward)),
            float(np.dot(delta, right)),
            float(np.dot(delta, up)),
        )

    def contains_world_position(self, world_position):
        local = self.world_to_local(world_position)
        horizontal = math.hypot(local[0], local[1])
        return (
            horizontal <= self.activity_bounds.horizontal_radius_m
            and abs(local[2])
            <= self.activity_bounds.vertical_delta_m
        )


@dataclass(frozen=True)
class GeneratedTaskStart:
    blueprint: TaskStartBlueprint
    rgbd_pair: LockstepRgbdPair
    visibility: VisibilitySnapshot
    assessment: VisibilityAssessment
    rejection_counts: tuple


class TaskStartGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentOdometry:
    position_local: tuple
    yaw_from_start_degrees: float


@dataclass(frozen=True)
class AgentRgbdView:
    name: str
    frame_id: int
    width: int
    height: int
    rgb: bytes
    depth: bytes
    fov_degrees: float
    near_clip: float
    far_clip: float
    projection_matrix: tuple


@dataclass(frozen=True)
class AgentDualViewObservation:
    oblique: AgentRgbdView
    nadir: AgentRgbdView
    odometry: AgentOdometry


def make_agent_observation(pair, odometry):
    if not isinstance(pair, LockstepRgbdPair):
        raise TypeError("pair must be a LockstepRgbdPair")
    if not isinstance(odometry, AgentOdometry):
        raise TypeError("odometry must be AgentOdometry")

    def strip(name, frame):
        return AgentRgbdView(
            name=name,
            frame_id=frame.frame_id,
            width=frame.width,
            height=frame.height,
            rgb=frame.rgb,
            depth=frame.depth,
            fov_degrees=frame.fov_degrees,
            near_clip=frame.near_clip,
            far_clip=frame.far_clip,
            projection_matrix=frame.projection_matrix,
        )

    return AgentDualViewObservation(
        oblique=strip("oblique", pair.oblique),
        nadir=strip("nadir", pair.nadir),
        odometry=odometry,
    )


class TaskRelativePoseController:
    """Research action boundary without exposing the absolute GTA pose."""

    def __init__(self, client, blueprint):
        if not isinstance(blueprint, TaskStartBlueprint):
            raise TypeError("blueprint must be a TaskStartBlueprint")
        self._client = client
        self._blueprint = blueprint
        self._absolute_pose = None

    def reset(self):
        pose = self._blueprint.absolute_pose
        self._absolute_pose = self._client.set_camera_pose(
            pose[0],
            pose[1],
            pose[2],
            pose[5],
            collision_check=False,
        )
        self._client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
        return self.odometry

    def synchronize(self):
        pose = self._client.get_pose()
        if not self._blueprint.contains_world_position(pose[:3]):
            raise RuntimeError(
                "Current camera pose lies outside the task activity bounds"
            )
        self._absolute_pose = pose
        return self.odometry

    @property
    def odometry(self):
        if self._absolute_pose is None:
            raise RuntimeError(
                "TaskRelativePoseController is not reset"
            )
        local = self._blueprint.world_to_local(
            self._absolute_pose[:3]
        )
        yaw_delta = (
            self._absolute_pose[5]
            - self._blueprint.absolute_pose[5]
            + 180.0
        ) % 360.0 - 180.0
        return AgentOdometry(
            position_local=local,
            yaw_from_start_degrees=float(yaw_delta),
        )

    def step_relative(
        self,
        dx_body,
        dy_body,
        dz_world,
        dyaw,
    ):
        if self._absolute_pose is None:
            raise RuntimeError(
                "TaskRelativePoseController is not reset"
            )
        values = tuple(
            float(value)
            for value in (
                dx_body,
                dy_body,
                dz_world,
                dyaw,
            )
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Task action contains a non-finite value")
        action_spec = self._blueprint.action_spec
        tolerance = 1.0e-9
        is_hold = all(abs(value) <= tolerance for value in values)
        is_forward = (
            abs(values[0] - action_spec.forward_step_m) <= tolerance
            and abs(values[1]) <= tolerance
            and abs(values[2]) <= tolerance
            and abs(values[3]) <= tolerance
        )
        is_vertical = (
            abs(values[0]) <= tolerance
            and abs(values[1]) <= tolerance
            and abs(abs(values[2]) - action_spec.vertical_step_m)
            <= tolerance
            and abs(values[3]) <= tolerance
        )
        is_turn = (
            abs(values[0]) <= tolerance
            and abs(values[1]) <= tolerance
            and abs(values[2]) <= tolerance
            and abs(abs(values[3]) - action_spec.yaw_step_degrees)
            <= tolerance
        )
        if is_hold:
            return self.odometry
        if not (is_forward or is_vertical or is_turn):
            raise ValueError(
                "INVALID_TASK_ACTION: expected fixed FORWARD, "
                "ASCEND, DESCEND, TURN_LEFT, TURN_RIGHT, or HOLD"
            )

        x, y, z, _pitch, _roll, yaw = self._absolute_pose
        yaw_radians = math.radians(yaw)
        target = (
            x
            - math.sin(yaw_radians) * values[0]
            + math.cos(yaw_radians) * values[1],
            y
            + math.cos(yaw_radians) * values[0]
            + math.sin(yaw_radians) * values[1],
            z + values[2],
        )
        if not self._blueprint.contains_world_position(target):
            raise ValueError(
                "Task action leaves the start-centered activity bounds"
            )
        target_yaw = (
            yaw + values[3] + 180.0
        ) % 360.0 - 180.0
        self._absolute_pose = self._client.set_camera_pose(
            target[0],
            target[1],
            target[2],
            target_yaw,
            collision_check=True,
        )
        return self.odometry


def _finite_vector3(name, values):
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    return array


def _local_axes(yaw_degrees):
    yaw = math.radians(float(yaw_degrees))
    return (
        np.asarray((-math.sin(yaw), math.cos(yaw), 0.0)),
        np.asarray((math.cos(yaw), math.sin(yaw), 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )


def _view_matrix(center, pitch_degrees, yaw_degrees):
    center = _finite_vector3("camera center", center)
    pitch = math.radians(float(pitch_degrees))
    yaw = math.radians(float(yaw_degrees))
    cos_pitch = math.cos(pitch)
    forward = np.asarray(
        (
            -math.sin(yaw) * abs(cos_pitch),
            math.cos(yaw) * abs(cos_pitch),
            math.sin(pitch),
        ),
        dtype=np.float64,
    )
    right = np.asarray(
        (math.cos(yaw), math.sin(yaw), 0.0),
        dtype=np.float64,
    )
    up = np.cross(right, forward)
    backward = -forward
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = backward
    view[:3, 3] = -view[:3, :3] @ center
    return view


def _projection_matrix(spec):
    tangent = math.tan(math.radians(spec.fov_degrees) * 0.5)
    near_minus_far = spec.near_clip - spec.far_clip
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = (spec.height / spec.width) / tangent
    projection[1, 1] = 1.0 / tangent
    projection[2, 2] = -spec.near_clip / near_minus_far
    projection[2, 3] = (
        -spec.near_clip * spec.far_clip
    ) / near_minus_far
    projection[3, 2] = -1.0
    return projection


def virtual_view_matrices(center, yaw_degrees, spec):
    projection = _projection_matrix(spec)
    return {
        "oblique": (
            projection,
            _view_matrix(
                center,
                spec.oblique_pitch_degrees,
                yaw_degrees,
            ),
        ),
        "nadir": (
            projection,
            _view_matrix(
                center,
                spec.nadir_pitch_degrees,
                yaw_degrees,
            ),
        ),
    }


def pair_view_matrices(pair):
    return {
        "oblique": (
            np.asarray(
                pair.oblique.projection_matrix,
                dtype=np.float64,
            ).reshape(4, 4),
            np.asarray(
                pair.oblique.view_matrix,
                dtype=np.float64,
            ).reshape(4, 4),
        ),
        "nadir": (
            np.asarray(
                pair.nadir.projection_matrix,
                dtype=np.float64,
            ).reshape(4, 4),
            np.asarray(
                pair.nadir.view_matrix,
                dtype=np.float64,
            ).reshape(4, 4),
        ),
    }


def _assess_target_view(target, matrices, spec):
    projection, view = matrices
    positions = np.asarray(
        [sample.position for sample in target.samples],
        dtype=np.float64,
    )
    world = np.column_stack(
        (positions, np.ones(len(positions), dtype=np.float64))
    )
    view_points = (view @ world.T).T
    clip = (projection @ view_points.T).T
    valid_w = np.abs(clip[:, 3]) > 1.0e-12
    ndc = np.full((len(positions), 3), np.nan)
    ndc[valid_w] = (
        clip[valid_w, :3] / clip[valid_w, 3:4]
    )
    depth = -view_points[:, 2]
    in_frustum = (
        valid_w
        & np.isfinite(ndc).all(axis=1)
        & (depth >= spec.near_clip)
        & (depth <= spec.far_clip)
        & (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
    )
    clear = np.asarray(
        [
            sample.clear_line_of_sight
            for sample in target.samples
        ],
        dtype=bool,
    )
    clear_in_frustum = in_frustum & clear
    projected_bbox = None
    projected_span = 0.0
    if np.any(clear_in_frustum):
        x = (
            (ndc[clear_in_frustum, 0] + 1.0)
            * 0.5
            * (spec.width - 1)
        )
        y = (
            (1.0 - ndc[clear_in_frustum, 1])
            * 0.5
            * (spec.height - 1)
        )
        projected_bbox = (
            float(np.min(x)),
            float(np.min(y)),
            float(np.max(x)),
            float(np.max(y)),
        )
        projected_span = max(
            projected_bbox[2] - projected_bbox[0],
            projected_bbox[3] - projected_bbox[1],
        )
    clear_count = int(np.count_nonzero(clear_in_frustum))
    inside_image_margin = (
        projected_bbox is not None
        and projected_bbox[0]
        >= spec.image_border_margin_pixels
        and projected_bbox[1]
        >= spec.image_border_margin_pixels
        and projected_bbox[2]
        <= spec.width - 1 - spec.image_border_margin_pixels
        and projected_bbox[3]
        <= spec.height - 1 - spec.image_border_margin_pixels
    )
    return ViewTargetVisibility(
        in_frustum_samples=int(np.count_nonzero(in_frustum)),
        clear_in_frustum_samples=clear_count,
        projected_bbox=projected_bbox,
        projected_span_pixels=float(projected_span),
        inside_image_margin=inside_image_margin,
        task_observable=(
            clear_count >= spec.min_clear_samples
            and projected_span
            >= spec.min_projected_span_pixels
            and inside_image_margin
        ),
    )


def assess_visibility(snapshot, view_matrices, spec):
    if not isinstance(snapshot, VisibilitySnapshot):
        raise TypeError("snapshot must be a VisibilitySnapshot")
    results = []
    source_clear = False
    envelope_clear = False
    envelope_clear_count = 0
    envelope_sample_count = 0
    envelope_task_observable = False
    event_task_observable = False
    cue_task_observable = False
    for target in snapshot.targets:
        oblique = _assess_target_view(
            target,
            view_matrices["oblique"],
            spec,
        )
        nadir = _assess_target_view(
            target,
            view_matrices["nadir"],
            spec,
        )
        observable = (
            oblique.task_observable
            or nadir.task_observable
        )
        any_clear = any(
            sample.clear_line_of_sight
            for sample in target.samples
        )
        if target.role == VisibilityTargetRole.FIRE_SOURCE_VEHICLE:
            source_clear = any_clear
            event_task_observable = observable
        elif target.role == VisibilityTargetRole.FIRE_ENVELOPE:
            envelope_clear = any_clear
            envelope_clear_count = sum(
                sample.clear_line_of_sight
                for sample in target.samples
            )
            envelope_sample_count = len(target.samples)
            envelope_task_observable = observable
        elif target.role in (
            VisibilityTargetRole.FIRE_TRUCK,
            VisibilityTargetRole.FLEEING_PEDESTRIAN,
        ):
            cue_task_observable = (
                cue_task_observable or observable
            )
        results.append(
            TargetVisibility(
                stable_id=target.stable_id,
                role=target.role,
                oblique=oblique,
                nadir=nadir,
            )
        )
    if envelope_sample_count == 0:
        raise ValueError(
            "Visibility snapshot has no fire-envelope samples"
        )
    return VisibilityAssessment(
        source_vehicle_has_line_of_sight=source_clear,
        fire_envelope_has_line_of_sight=envelope_clear,
        fire_envelope_clear_fraction=(
            envelope_clear_count / envelope_sample_count
        ),
        fire_envelope_task_observable=envelope_task_observable,
        event_initially_hidden=(
            not source_clear and not envelope_task_observable
        ),
        event_task_observable=event_task_observable,
        cue_task_observable=cue_task_observable,
        targets=tuple(results),
    )


def _require_visibility_instant(visibility, clock):
    if (
        visibility.lockstep_session_id != clock.session_id
        or visibility.step_index != clock.step_index
        or visibility.game_timer_ms != clock.game_timer_ms
    ):
        raise TaskStartGenerationError(
            "Visibility snapshot does not belong to the current "
            "lockstep instant"
        )


def _require_observation_spec(pair, spec):
    measured = ObservationSpec.from_pair(pair)
    if measured != spec:
        raise TaskStartGenerationError(
            "Actual RGB-D calibration does not match the task-start "
            "ObservationSpec"
        )


def _stratum_matches(stratum, assessment):
    if not assessment.event_initially_hidden:
        return False
    if stratum == StartVisibilityStratum.POTENTIAL_CUE_VISIBLE:
        return assessment.cue_task_observable
    if stratum == StartVisibilityStratum.CUE_HIDDEN:
        return not assessment.cue_task_observable
    raise ValueError(f"Unknown visibility stratum {stratum!r}")


def potential_cue_visible(
    assessment,
    scenario,
    maximum_activation_offset_ms=2000,
):
    maximum_activation_offset_ms = int(
        maximum_activation_offset_ms
    )
    if maximum_activation_offset_ms < 0:
        raise ValueError(
            "maximum_activation_offset_ms cannot be negative"
        )
    entities = {
        entity.stable_id: entity
        for entity in scenario.entities
        if entity.exists
    }
    for target in assessment.targets:
        if target.role not in (
            VisibilityTargetRole.FIRE_TRUCK,
            VisibilityTargetRole.FLEEING_PEDESTRIAN,
        ):
            continue
        if not (
            target.oblique.task_observable
            or target.nadir.task_observable
        ):
            continue
        entity = entities.get(target.stable_id)
        if entity is None:
            continue
        if (
            entity.planned_activation_offset_ms
            <= maximum_activation_offset_ms
        ):
            return True
    return False


def _start_id(blueprint_id, start_seed, stratum, candidate_index):
    digest = hashlib.blake2b(
        (
            f"{int(blueprint_id)}:{int(start_seed)}:"
            f"{int(stratum)}:{int(candidate_index)}"
        ).encode("ascii"),
        digest_size=8,
        person=b"DroneStart",
    ).digest()
    value = int.from_bytes(digest, "little")
    return value or 1


def _event_bearing_body(start_position, yaw, event_position):
    forward, right, _up = _local_axes(yaw)
    delta = (
        _finite_vector3("event_position", event_position)
        - _finite_vector3("start_position", start_position)
    )
    return math.degrees(
        math.atan2(
            float(np.dot(delta, right)),
            float(np.dot(delta, forward)),
        )
    )


def _world_yaw_toward(origin, target):
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return math.degrees(math.atan2(-dx, dy))


def generate_task_start(
    client,
    session,
    scenario,
    observation_spec,
    visibility_stratum,
    start_seed,
    max_candidates=TASK_MAX_START_CANDIDATES,
    horizon_steps=TASK_HORIZON_STEPS,
):
    if not isinstance(session, LockstepSession):
        raise TypeError("session must be a LockstepSession")
    if not isinstance(observation_spec, ObservationSpec):
        raise TypeError("observation_spec must be an ObservationSpec")
    visibility_stratum = StartVisibilityStratum(
        visibility_stratum
    )
    if scenario.lifecycle != ScenarioLifecycle.RUNNING:
        raise ValueError("Task starts require a RUNNING scenario")
    if not scenario.event_active:
        raise ValueError("Task starts require an active fire")
    clock = session.refresh()
    if clock.step_index != 1:
        raise ValueError(
            "Task starts must be generated at the first t=250ms "
            "frozen observation"
        )
    start_seed = int(start_seed)
    if not 0 <= start_seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("start_seed must fit uint64")
    max_candidates = int(max_candidates)
    if max_candidates <= 0 or max_candidates > 4096:
        raise ValueError("max_candidates must be in [1, 4096]")
    horizon_steps = int(horizon_steps)
    if not 8 <= horizon_steps <= 256:
        raise ValueError("horizon_steps must be in [8, 256]")

    rng = random.Random(
        start_seed ^ 0x53544152545F504F
    )
    rejection_counts = {
        "ground_not_found": 0,
        "space_blocked": 0,
        "goal_view_outside_activity": 0,
        "source_vehicle_visible": 0,
        "fire_envelope_observable": 0,
        "stratum_mismatch": 0,
        "real_camera_mismatch": 0,
    }
    selected = None
    event = np.asarray(
        scenario.event_position,
        dtype=np.float64,
    )
    potential_responders = tuple(
        entity
        for entity in scenario.entities
        if entity.exists
        and entity.role
        in (
            ScenarioEntityRole.FIRE_TRUCK,
            ScenarioEntityRole.FLEEING_PEDESTRIAN,
        )
        and entity.planned_activation_offset_ms <= 2000
    )
    if (
        visibility_stratum
        == StartVisibilityStratum.POTENTIAL_CUE_VISIBLE
        and not potential_responders
    ):
        raise TaskStartGenerationError(
            "No responder is scheduled inside the two-second "
            "potential-cue window"
        )
    for candidate_index in range(max_candidates):
        radius_squared = rng.uniform(
            TASK_MIN_EVENT_DISTANCE_METERS**2,
            TASK_MAX_EVENT_DISTANCE_METERS**2,
        )
        radius = math.sqrt(radius_squared)
        angle = rng.uniform(-math.pi, math.pi)
        x = float(event[0] + math.cos(angle) * radius)
        y = float(event[1] + math.sin(angle) * radius)
        altitude_agl = rng.uniform(
            TASK_MIN_ALTITUDE_AGL_METERS,
            TASK_MAX_ALTITUDE_AGL_METERS,
        )
        yaw = rng.uniform(-180.0, 180.0)
        try:
            position, ground_z = client.probe_camera_start(
                x,
                y,
                altitude_agl,
            )
        except DroneSimCommandError as error:
            if error.status_name == "START_GROUND_NOT_FOUND":
                rejection_counts["ground_not_found"] += 1
                continue
            if error.status_name == "START_SPACE_BLOCKED":
                rejection_counts["space_blocked"] += 1
                continue
            raise
        event_horizontal_distance = math.hypot(
            float(position[0] - event[0]),
            float(position[1] - event[1]),
        )
        has_in_bounds_goal_height = (
            event_horizontal_distance
            <= TASK_ACTIVITY_RADIUS_METERS
            and any(
                abs(
                    float(event[2])
                    + height
                    - float(position[2])
                )
                <= TASK_ACTIVITY_VERTICAL_METERS
                for height in TASK_GOAL_VIEW_HEIGHTS_METERS
            )
        )
        if not has_in_bounds_goal_height:
            rejection_counts[
                "goal_view_outside_activity"
            ] += 1
            continue
        if (
            visibility_stratum
            == StartVisibilityStratum.POTENTIAL_CUE_VISIBLE
        ):
            responder = potential_responders[
                rng.randrange(len(potential_responders))
            ]
            yaw = (
                _world_yaw_toward(
                    position,
                    responder.position,
                )
                + rng.uniform(-15.0, 15.0)
                + 180.0
            ) % 360.0 - 180.0

        visibility = client.query_visibility(
            scenario.scenario_id,
            session.session_id,
            position,
            timeout=30.0,
        )
        _require_visibility_instant(visibility, clock)
        assessment = assess_visibility(
            visibility,
            virtual_view_matrices(
                position,
                yaw,
                observation_spec,
            ),
            observation_spec,
        )
        if assessment.source_vehicle_has_line_of_sight:
            rejection_counts["source_vehicle_visible"] += 1
            continue
        if assessment.fire_envelope_task_observable:
            rejection_counts["fire_envelope_observable"] += 1
            continue
        if not _stratum_matches(
            visibility_stratum,
            assessment,
        ):
            rejection_counts["stratum_mismatch"] += 1
            continue
        if (
            visibility_stratum
            == StartVisibilityStratum.POTENTIAL_CUE_VISIBLE
            and not potential_cue_visible(
                assessment,
                scenario,
            )
        ):
            rejection_counts["stratum_mismatch"] += 1
            continue
        with client._operation_lock:
            client.set_camera_pose(
                position[0],
                position[1],
                position[2],
                yaw,
                collision_check=False,
            )
            client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
            pair = session.capture_rgbd_pair()
            _require_observation_spec(pair, observation_spec)
            actual_pose = client.get_pose()
            actual_visibility = client.query_visibility(
                scenario.scenario_id,
                session.session_id,
                actual_pose[:3],
                timeout=30.0,
            )
        _require_visibility_instant(
            actual_visibility,
            pair.clock,
        )
        actual_assessment = assess_visibility(
            actual_visibility,
            pair_view_matrices(pair),
            observation_spec,
        )
        if not _stratum_matches(
            visibility_stratum,
            actual_assessment,
        ):
            rejection_counts["real_camera_mismatch"] += 1
            continue
        if (
            visibility_stratum
            == StartVisibilityStratum.POTENTIAL_CUE_VISIBLE
            and not potential_cue_visible(
                actual_assessment,
                scenario,
            )
        ):
            rejection_counts["real_camera_mismatch"] += 1
            continue
        selected = (
            candidate_index,
            position,
            ground_z,
            altitude_agl,
            yaw,
            pair,
            actual_pose,
            actual_visibility,
            actual_assessment,
        )
        break

    if selected is None:
        summary = ", ".join(
            f"{name}={count}"
            for name, count in rejection_counts.items()
        )
        raise TaskStartGenerationError(
            f"TASK_START_NOT_FOUND after {max_candidates} "
            f"candidates: {summary}"
        )

    (
        candidate_index,
        position,
        ground_z,
        altitude_agl,
        yaw,
        pair,
        actual_pose,
        actual_visibility,
        actual_assessment,
    ) = selected

    horizontal_distance = math.hypot(
        float(position[0] - event[0]),
        float(position[1] - event[1]),
    )
    measured_agl = float(actual_pose[2] - ground_z)
    blueprint = TaskStartBlueprint(
        start_id=_start_id(
            scenario.blueprint_id,
            start_seed,
            visibility_stratum,
            candidate_index,
        ),
        scenario_blueprint_id=scenario.blueprint_id,
        start_seed=start_seed,
        candidate_index=candidate_index,
        absolute_pose=tuple(float(value) for value in actual_pose),
        altitude_agl=measured_agl,
        event_distance=horizontal_distance,
        event_bearing_body_degrees=_event_bearing_body(
            actual_pose[:3],
            actual_pose[5],
            scenario.event_position,
        ),
        visibility_stratum=visibility_stratum,
        observation_spec=observation_spec,
        activity_bounds=ActivityBounds(),
        action_spec=TaskActionSpec(horizon_steps=horizon_steps),
    )
    if horizontal_distance > (
        blueprint.activity_bounds.horizontal_radius_m
    ):
        raise TaskStartGenerationError(
            "Generated event lies outside the start-centered "
            "horizontal activity radius"
        )
    if abs(measured_agl - altitude_agl) > 1.0e-3:
        raise TaskStartGenerationError(
            "Resolved task-start altitude changed unexpectedly"
        )
    return GeneratedTaskStart(
        blueprint=blueprint,
        rgbd_pair=pair,
        visibility=actual_visibility,
        assessment=actual_assessment,
        rejection_counts=tuple(
            sorted(rejection_counts.items())
        ),
    )
