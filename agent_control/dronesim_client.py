import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from itertools import count


MAGIC = b"DSV3"
VERSION = 3

TYPE_CREATE_CAMERA = 1
TYPE_SET_FOV = 4
TYPE_CAPTURE = 5
TYPE_PING = 6
TYPE_GET_POSE = 7
TYPE_SET_TIME = 8
TYPE_SET_WEATHER = 9
TYPE_STOP_CAMERA = 10
TYPE_TELEPORT_PLAYER = 17
TYPE_RESTORE_PLAYER = 18
TYPE_GET_CAMERA_STATE = 20
TYPE_SET_CAMERA_POSE = 21
TYPE_PREPARE_FIRE_SCENARIO = 22
TYPE_GET_SCENARIO_STATE = 23
TYPE_START_SCENARIO = 24
TYPE_RESET_SCENARIO = 25
TYPE_SET_CAMERA_PITCH = 26
TYPE_ENTER_LOCKSTEP = 27
TYPE_GET_LOCKSTEP_STATE = 28
TYPE_ADVANCE_LOCKSTEP = 29
TYPE_EXIT_LOCKSTEP = 30
TYPE_QUERY_VISIBILITY = 31
TYPE_PROBE_CAMERA_START = 32
TYPE_PROBE_CAMERA_GEOMETRY_BATCH = 33
TYPE_QUERY_TARGET_VISIBILITY_BATCH = 34

COMMAND_STATUS = {
    0: "OK",
    1: "CAMERA_INACTIVE",
    2: "INVALID_POSE",
    3: "COLLISION_BLOCKED",
    4: "COMMAND_TIMEOUT",
    5: "POSE_APPLY_FAILED",
    6: "POSE_MISMATCH",
    7: "INVALID_REQUEST",
    8: "INTERNAL_ERROR",
    9: "SCENARIO_ALREADY_ACTIVE",
    10: "SCENARIO_NOT_FOUND",
    11: "SCENARIO_NOT_READY",
    12: "SCENARIO_AREA_NOT_READY",
    13: "SCENARIO_PREPARE_FAILED",
    14: "SCENARIO_START_FAILED",
    15: "WORLD_AREA_NOT_READY",
    16: "LOCKSTEP_ALREADY_ACTIVE",
    17: "LOCKSTEP_NOT_ACTIVE",
    18: "LOCKSTEP_SESSION_MISMATCH",
    19: "LOCKSTEP_ADVANCE_TIMEOUT",
    20: "LOCKSTEP_INTERRUPTED",
    21: "LOCKSTEP_CLOCK_INVARIANT_FAILED",
    22: "VISIBILITY_GEOMETRY_INVALID",
    23: "VISIBILITY_RAYCAST_FAILED",
    24: "VISIBILITY_INTERRUPTED",
    25: "START_GROUND_NOT_FOUND",
    26: "START_SPACE_BLOCKED",
    27: "START_PROBE_FAILED",
}

CAPTURE_STATUS = {
    0: "OK",
    1: "CAPTURE_BUSY",
    2: "CAPTURE_TIMEOUT",
    3: "RGB_FORMAT_MISMATCH",
    4: "DEPTH_FORMAT_MISMATCH",
    5: "DEPTH_TARGET_NOT_FOUND",
    6: "DEPTH_TARGET_AMBIGUOUS",
    7: "RESOURCE_GENERATION_CHANGED",
    8: "INVALID_CAMERA_PARAMETERS",
    9: "DEPTH_CONVERSION_FAILED",
    10: "GPU_READBACK_FAILED",
    11: "INTERNAL_ERROR",
}

CAPTURE_METADATA_FORMAT = "<IQIIIIfff32f"
CAPTURE_METADATA_SIZE = struct.calcsize(CAPTURE_METADATA_FORMAT)
MAX_FIRETRUCK_COUNT = 4
MAX_PEDESTRIAN_COUNT = 32
MAX_SCENARIO_OWNED_ENTITY_COUNT = (
    1 + 2 * MAX_FIRETRUCK_COUNT + MAX_PEDESTRIAN_COUNT
)
LOCKSTEP_STEP_MS = 250
OBLIQUE_PITCH_DEGREES = -45.0
NADIR_PITCH_DEGREES = -90.0
LOCKSTEP_SNAPSHOT_FORMAT = "<QQIIIQQIIf"
LOCKSTEP_SNAPSHOT_SIZE = struct.calcsize(
    LOCKSTEP_SNAPSHOT_FORMAT
)


class DroneSimProtocolError(RuntimeError):
    pass


class DroneSimCommandError(RuntimeError):
    def __init__(self, status, message):
        self.status = int(status)
        self.status_name = COMMAND_STATUS.get(
            self.status, "UNKNOWN_COMMAND_STATUS"
        )
        super().__init__(f"{self.status_name}: {message}")


class CaptureError(RuntimeError):
    def __init__(self, status, message):
        self.status = int(status)
        self.status_name = CAPTURE_STATUS.get(
            self.status, "UNKNOWN_CAPTURE_STATUS"
        )
        super().__init__(f"{self.status_name}: {message}")


class ScenarioLifecycle(IntEnum):
    EMPTY = 0
    PREPARING = 1
    READY = 2
    RUNNING = 3
    FAILED = 4


class ScenarioEntityKind(IntEnum):
    VEHICLE = 1
    PEDESTRIAN = 2


class ScenarioEntityRole(IntEnum):
    FIRE_SOURCE_VEHICLE = 1
    FIRE_TRUCK = 2
    FIREFIGHTER_DRIVER = 3
    FLEEING_PEDESTRIAN = 4


class ScenarioTaskState(IntEnum):
    NONE = 0
    PENDING = 1
    ACTIVE = 2
    SUCCEEDED = 3
    FAILED = 4
    LOST = 5


class VisibilityTargetRole(IntEnum):
    FIRE_SOURCE_VEHICLE = 1
    FIRE_ENVELOPE = 2
    FIRE_TRUCK = 3
    FLEEING_PEDESTRIAN = 4


@dataclass(frozen=True)
class ScenarioEntitySnapshot:
    stable_id: int
    gta_handle: int
    model_hash: int
    kind: ScenarioEntityKind
    role: ScenarioEntityRole
    event_id: int
    task_state: ScenarioTaskState
    exists: bool
    position: tuple
    velocity: tuple
    speed: float
    heading: float
    spawn_game_timer_ms: int
    planned_activation_offset_ms: int
    activation_game_timer_ms: int
    task_start_game_timer_ms: int
    response_start_game_timer_ms: int
    task_target: tuple


@dataclass(frozen=True)
class ScenarioProtectedEntitySnapshot:
    gta_handle: int
    model_hash: int
    kind: ScenarioEntityKind
    exists: bool
    position: tuple


@dataclass(frozen=True)
class ScenarioSnapshot:
    scenario_id: int
    blueprint_id: int
    seed: int
    lifecycle: ScenarioLifecycle
    game_timer_ms: int
    frame_count: int
    start_game_timer_ms: int
    start_frame_count: int
    requested_anchor: tuple
    event_position: tuple
    event_active: bool
    removed_pedestrians: int
    removed_vehicles: int
    ambient_pedestrians: int
    ambient_vehicles: int
    failure_message: str
    protected_entities: tuple
    entities: tuple


@dataclass(frozen=True)
class ScenarioStartInfo:
    scenario_id: int
    game_timer_ms: int
    frame_count: int


@dataclass(frozen=True)
class LockstepSnapshot:
    session_id: int
    step_index: int
    epoch_game_timer_ms: int
    game_timer_ms: int
    frame_count: int
    target_elapsed_ms: int
    actual_elapsed_ms: int
    last_advance_ms: int
    render_frames: int
    max_frame_time_ms: float


@dataclass(frozen=True)
class VisibilitySample:
    position: tuple
    clear_line_of_sight: bool
    hit_entity: int


@dataclass(frozen=True)
class VisibilityTarget:
    stable_id: int
    gta_handle: int
    role: VisibilityTargetRole
    samples: tuple


@dataclass(frozen=True)
class VisibilitySnapshot:
    scenario_id: int
    lockstep_session_id: int
    step_index: int
    game_timer_ms: int
    frame_count: int
    camera_center: tuple
    targets: tuple


@dataclass(frozen=True)
class GeometryBatchSnapshot:
    lockstep_session_id: int
    step_index: int
    game_timer_ms: int
    frame_count: int
    point_clear: tuple
    segment_clear: tuple


@dataclass(frozen=True)
class TargetVisibilityCase:
    stable_id: int
    camera_center: tuple


@dataclass(frozen=True)
class TargetVisibilityCaseSnapshot:
    stable_id: int
    camera_center: tuple
    target: VisibilityTarget


@dataclass(frozen=True)
class TargetVisibilityBatchSnapshot:
    scenario_id: int
    lockstep_session_id: int
    step_index: int
    game_timer_ms: int
    frame_count: int
    cases: tuple


@dataclass(frozen=True)
class RgbdFrame:
    request_id: int
    frame_id: int
    width: int
    height: int
    rgb: bytes
    depth: bytes
    fov_degrees: float
    near_clip: float
    far_clip: float
    projection_matrix: tuple
    view_matrix: tuple

    def __iter__(self):
        yield self.width
        yield self.height
        yield self.rgb
        yield self.depth

    def rgb_array(self):
        import numpy as np

        expected = self.width * self.height * 3
        if len(self.rgb) != expected:
            raise DroneSimProtocolError(
                f"RGB payload has {len(self.rgb)} bytes; expected {expected}"
            )
        return np.frombuffer(self.rgb, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )

    def depth_array(self):
        import numpy as np

        expected = self.width * self.height * 4
        if len(self.depth) != expected:
            raise DroneSimProtocolError(
                f"Depth payload has {len(self.depth)} bytes; expected {expected}"
            )
        depth = np.frombuffer(self.depth, dtype="<f4").reshape(
            self.height, self.width
        )
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise DroneSimProtocolError(
                "Depth payload contains a non-finite or negative metric value"
            )
        return depth


@dataclass(frozen=True)
class LockstepRgbdPair:
    clock: LockstepSnapshot
    oblique: RgbdFrame
    nadir: RgbdFrame


def _pack_header(message_type, request_id, length):
    return struct.pack(
        "<4sBBBBQI",
        MAGIC,
        VERSION,
        message_type,
        0,
        0,
        request_id,
        length,
    )


def _recv_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count_received = sock.recv_into(
            view[received:], size - received
        )
        if count_received == 0:
            return None
        received += count_received
    return bytes(data)


def _require_finite(*values):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("All pose values must be finite")


class _PayloadReader:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0

    def unpack(self, format_string):
        size = struct.calcsize(format_string)
        if len(self.payload) - self.offset < size:
            raise DroneSimProtocolError(
                "Scenario response ended before all declared fields"
            )
        values = struct.unpack_from(
            format_string, self.payload, self.offset
        )
        self.offset += size
        return values

    def text(self):
        (size,) = self.unpack("<I")
        if len(self.payload) - self.offset < size:
            raise DroneSimProtocolError(
                "Scenario failure message exceeds its payload"
            )
        value = self.payload[
            self.offset : self.offset + size
        ].decode("utf-8", errors="strict")
        self.offset += size
        return value

    def finish(self):
        if self.offset != len(self.payload):
            raise DroneSimProtocolError(
                f"Scenario response has "
                f"{len(self.payload) - self.offset} trailing bytes"
            )


def _finite_tuple(name, values):
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise DroneSimProtocolError(
            f"{name} contains a non-finite value"
        )
    return result


def _decode_scenario_snapshot(payload):
    reader = _PayloadReader(payload)
    scenario_id, blueprint_id, seed = reader.unpack("<QQQ")
    if blueprint_id == 0:
        raise DroneSimProtocolError(
            "Scenario snapshot contains a zero blueprint_id"
        )
    (
        lifecycle_value,
        game_timer_ms,
        frame_count,
        start_game_timer_ms,
        start_frame_count,
    ) = reader.unpack("<5I")
    requested_anchor = _finite_tuple(
        "requested_anchor", reader.unpack("<3f")
    )
    event_position = _finite_tuple(
        "event_position", reader.unpack("<3f")
    )
    (event_active_value,) = reader.unpack("<B")
    if event_active_value not in (0, 1):
        raise DroneSimProtocolError(
            "Scenario event_active is not a boolean"
        )
    (
        removed_pedestrians,
        removed_vehicles,
        ambient_pedestrians,
        ambient_vehicles,
    ) = reader.unpack("<4I")
    failure_message = reader.text()
    (protected_entity_count,) = reader.unpack("<I")
    if protected_entity_count > 2048:
        raise DroneSimProtocolError(
            "Scenario declares "
            f"{protected_entity_count} protected entities; maximum is 2048"
        )
    protected_entities = []
    protected_handles = set()
    for _ in range(protected_entity_count):
        (
            gta_handle,
            model_hash,
            kind_value,
            exists_value,
            x,
            y,
            z,
        ) = reader.unpack("<iIIB3f")
        if exists_value not in (0, 1):
            raise DroneSimProtocolError(
                "Protected entity exists field is not a boolean"
            )
        if gta_handle == 0 or gta_handle in protected_handles:
            raise DroneSimProtocolError(
                f"Invalid or duplicate protected handle {gta_handle}"
            )
        protected_handles.add(gta_handle)
        try:
            kind = ScenarioEntityKind(kind_value)
        except ValueError as error:
            raise DroneSimProtocolError(
                f"Unknown protected entity kind {kind_value}"
            ) from error
        protected_entities.append(
            ScenarioProtectedEntitySnapshot(
                gta_handle=gta_handle,
                model_hash=model_hash,
                kind=kind,
                exists=bool(exists_value),
                position=_finite_tuple(
                    "protected entity position", (x, y, z)
                ),
            )
        )
    (entity_count,) = reader.unpack("<I")
    if entity_count > MAX_SCENARIO_OWNED_ENTITY_COUNT:
        raise DroneSimProtocolError(
            "Scenario declares "
            f"{entity_count} owned entities; maximum is "
            f"{MAX_SCENARIO_OWNED_ENTITY_COUNT}"
        )

    try:
        lifecycle = ScenarioLifecycle(lifecycle_value)
    except ValueError as error:
        raise DroneSimProtocolError(
            f"Unknown scenario lifecycle {lifecycle_value}"
        ) from error

    entities = []
    stable_ids = set()
    for _ in range(entity_count):
        (
            stable_id,
            gta_handle,
            model_hash,
            kind_value,
            role_value,
            event_id,
            task_state_value,
            exists_value,
        ) = reader.unpack("<QiIIIQIB")
        if exists_value not in (0, 1):
            raise DroneSimProtocolError(
                "Scenario entity exists field is not a boolean"
            )
        if stable_id == 0 or stable_id in stable_ids:
            raise DroneSimProtocolError(
                f"Invalid or duplicate stable entity ID {stable_id}"
            )
        stable_ids.add(stable_id)
        try:
            kind = ScenarioEntityKind(kind_value)
            role = ScenarioEntityRole(role_value)
            task_state = ScenarioTaskState(task_state_value)
        except ValueError as error:
            raise DroneSimProtocolError(
                "Scenario entity contains an unknown enum value"
            ) from error
        position = _finite_tuple(
            "entity position", reader.unpack("<3f")
        )
        velocity = _finite_tuple(
            "entity velocity", reader.unpack("<3f")
        )
        speed, heading = reader.unpack("<2f")
        if not math.isfinite(speed) or not math.isfinite(heading):
            raise DroneSimProtocolError(
                "Scenario entity speed/heading is not finite"
            )
        (
            spawn_game_timer_ms,
            planned_activation_offset_ms,
            activation_game_timer_ms,
            task_start_game_timer_ms,
            response_start_game_timer_ms,
        ) = reader.unpack("<5I")
        task_target = _finite_tuple(
            "entity task target", reader.unpack("<3f")
        )
        entities.append(
            ScenarioEntitySnapshot(
                stable_id=stable_id,
                gta_handle=gta_handle,
                model_hash=model_hash,
                kind=kind,
                role=role,
                event_id=event_id,
                task_state=task_state,
                exists=bool(exists_value),
                position=position,
                velocity=velocity,
                speed=float(speed),
                heading=float(heading),
                spawn_game_timer_ms=spawn_game_timer_ms,
                planned_activation_offset_ms=(
                    planned_activation_offset_ms
                ),
                activation_game_timer_ms=activation_game_timer_ms,
                task_start_game_timer_ms=task_start_game_timer_ms,
                response_start_game_timer_ms=response_start_game_timer_ms,
                task_target=task_target,
            )
        )
    reader.finish()
    return ScenarioSnapshot(
        scenario_id=scenario_id,
        blueprint_id=blueprint_id,
        seed=seed,
        lifecycle=lifecycle,
        game_timer_ms=game_timer_ms,
        frame_count=frame_count,
        start_game_timer_ms=start_game_timer_ms,
        start_frame_count=start_frame_count,
        requested_anchor=requested_anchor,
        event_position=event_position,
        event_active=bool(event_active_value),
        removed_pedestrians=removed_pedestrians,
        removed_vehicles=removed_vehicles,
        ambient_pedestrians=ambient_pedestrians,
        ambient_vehicles=ambient_vehicles,
        failure_message=failure_message,
        protected_entities=tuple(protected_entities),
        entities=tuple(entities),
    )


def _decode_lockstep_snapshot(payload):
    if len(payload) != LOCKSTEP_SNAPSHOT_SIZE:
        raise DroneSimProtocolError(
            f"Lockstep snapshot has {len(payload)} bytes; "
            f"expected {LOCKSTEP_SNAPSHOT_SIZE}"
        )
    (
        session_id,
        step_index,
        epoch_game_timer_ms,
        game_timer_ms,
        frame_count,
        target_elapsed_ms,
        actual_elapsed_ms,
        last_advance_ms,
        render_frames,
        max_frame_time_ms,
    ) = struct.unpack(LOCKSTEP_SNAPSHOT_FORMAT, payload)
    if session_id == 0:
        raise DroneSimProtocolError(
            "Lockstep snapshot contains a zero session_id"
        )
    expected_target = step_index * LOCKSTEP_STEP_MS
    if target_elapsed_ms != expected_target:
        raise DroneSimProtocolError(
            "Lockstep target does not match step_index: "
            f"{target_elapsed_ms}/{expected_target}"
        )
    if actual_elapsed_ms < target_elapsed_ms:
        raise DroneSimProtocolError(
            "Lockstep actual elapsed time is earlier than its target"
        )
    timer_elapsed = (
        game_timer_ms - epoch_game_timer_ms
    ) & 0xFFFFFFFF
    if timer_elapsed != actual_elapsed_ms:
        raise DroneSimProtocolError(
            "Lockstep GTA timer does not match actual_elapsed_ms: "
            f"{timer_elapsed}/{actual_elapsed_ms}"
        )
    if (
        not math.isfinite(max_frame_time_ms)
        or max_frame_time_ms < 0
    ):
        raise DroneSimProtocolError(
            "Lockstep max_frame_time_ms is invalid"
        )
    if step_index == 0:
        if (
            last_advance_ms != 0
            or render_frames != 0
            or max_frame_time_ms != 0
        ):
            raise DroneSimProtocolError(
                "Initial lockstep snapshot contains step diagnostics"
            )
    elif last_advance_ms == 0 or render_frames == 0:
        raise DroneSimProtocolError(
            "Advanced lockstep snapshot contains zero step diagnostics"
        )
    return LockstepSnapshot(
        session_id=session_id,
        step_index=step_index,
        epoch_game_timer_ms=epoch_game_timer_ms,
        game_timer_ms=game_timer_ms,
        frame_count=frame_count,
        target_elapsed_ms=target_elapsed_ms,
        actual_elapsed_ms=actual_elapsed_ms,
        last_advance_ms=last_advance_ms,
        render_frames=render_frames,
        max_frame_time_ms=float(max_frame_time_ms),
    )


def _decode_visibility_snapshot(payload):
    reader = _PayloadReader(payload)
    (
        scenario_id,
        lockstep_session_id,
        step_index,
    ) = reader.unpack("<QQQ")
    game_timer_ms, frame_count = reader.unpack("<II")
    camera_center = _finite_tuple(
        "visibility camera center",
        reader.unpack("<3f"),
    )
    (target_count,) = reader.unpack("<I")
    if target_count == 0 or target_count > (
        2 + MAX_FIRETRUCK_COUNT + MAX_PEDESTRIAN_COUNT
    ):
        raise DroneSimProtocolError(
            f"Visibility declares invalid target count {target_count}"
        )

    targets = []
    stable_ids = set()
    envelope_count = 0
    for _ in range(target_count):
        (
            stable_id,
            gta_handle,
            role_value,
            sample_count,
        ) = reader.unpack("<QiII")
        try:
            role = VisibilityTargetRole(role_value)
        except ValueError as error:
            raise DroneSimProtocolError(
                f"Unknown visibility target role {role_value}"
            ) from error
        if role == VisibilityTargetRole.FIRE_ENVELOPE:
            envelope_count += 1
            if stable_id != 0 or gta_handle != 0:
                raise DroneSimProtocolError(
                    "Fire envelope must not declare an entity identity"
                )
        else:
            if stable_id == 0 or stable_id in stable_ids:
                raise DroneSimProtocolError(
                    f"Invalid visibility stable ID {stable_id}"
                )
            if gta_handle == 0:
                raise DroneSimProtocolError(
                    "Entity visibility target has a zero GTA handle"
                )
            stable_ids.add(stable_id)
        if sample_count == 0 or sample_count > 64:
            raise DroneSimProtocolError(
                f"Visibility target declares {sample_count} samples"
            )
        samples = []
        for _ in range(sample_count):
            x, y, z, clear_value, hit_entity = reader.unpack(
                "<3fBi"
            )
            if clear_value not in (0, 1):
                raise DroneSimProtocolError(
                    "Visibility clear_line_of_sight is not boolean"
                )
            samples.append(
                VisibilitySample(
                    position=_finite_tuple(
                        "visibility sample position",
                        (x, y, z),
                    ),
                    clear_line_of_sight=bool(clear_value),
                    hit_entity=hit_entity,
                )
            )
        targets.append(
            VisibilityTarget(
                stable_id=stable_id,
                gta_handle=gta_handle,
                role=role,
                samples=tuple(samples),
            )
        )
    reader.finish()
    if envelope_count != 1:
        raise DroneSimProtocolError(
            f"Visibility declares {envelope_count} fire envelopes"
        )
    if scenario_id == 0 or lockstep_session_id == 0:
        raise DroneSimProtocolError(
            "Visibility snapshot has a zero scenario/session ID"
        )
    return VisibilitySnapshot(
        scenario_id=scenario_id,
        lockstep_session_id=lockstep_session_id,
        step_index=step_index,
        game_timer_ms=game_timer_ms,
        frame_count=frame_count,
        camera_center=camera_center,
        targets=tuple(targets),
    )


def _decode_geometry_batch_snapshot(payload):
    reader = _PayloadReader(payload)
    lockstep_session_id, step_index = reader.unpack("<QQ")
    game_timer_ms, frame_count, point_count, segment_count = (
        reader.unpack("<4I")
    )
    if (
        lockstep_session_id == 0
        or point_count + segment_count == 0
        or point_count + segment_count > 256
    ):
        raise DroneSimProtocolError(
            "Camera-geometry batch declares invalid identity or counts"
        )
    point_values = reader.unpack(f"<{point_count}B") if point_count else ()
    segment_values = (
        reader.unpack(f"<{segment_count}B") if segment_count else ()
    )
    if any(value not in (0, 1) for value in point_values + segment_values):
        raise DroneSimProtocolError(
            "Camera-geometry batch contains a non-boolean result"
        )
    reader.finish()
    return GeometryBatchSnapshot(
        lockstep_session_id=lockstep_session_id,
        step_index=step_index,
        game_timer_ms=game_timer_ms,
        frame_count=frame_count,
        point_clear=tuple(bool(value) for value in point_values),
        segment_clear=tuple(bool(value) for value in segment_values),
    )


def _decode_target_visibility_batch_snapshot(payload):
    reader = _PayloadReader(payload)
    scenario_id, lockstep_session_id, step_index = reader.unpack("<QQQ")
    game_timer_ms, frame_count, case_count = reader.unpack("<3I")
    if (
        scenario_id == 0
        or lockstep_session_id == 0
        or case_count == 0
        or case_count > 64
    ):
        raise DroneSimProtocolError(
            "Target-visibility batch declares invalid identity or count"
        )
    cases = []
    for _ in range(case_count):
        stable_id = reader.unpack("<Q")[0]
        camera_center = _finite_tuple(
            "target-visibility camera center",
            reader.unpack("<3f"),
        )
        gta_handle, role_value, sample_count = reader.unpack("<iII")
        if stable_id == 0 or gta_handle == 0:
            raise DroneSimProtocolError(
                "Target-visibility case contains a zero entity identity"
            )
        try:
            role = VisibilityTargetRole(role_value)
        except ValueError as error:
            raise DroneSimProtocolError(
                f"Unknown target-visibility role {role_value}"
            ) from error
        if role == VisibilityTargetRole.FIRE_ENVELOPE:
            raise DroneSimProtocolError(
                "Target-visibility batch cannot contain the fire envelope"
            )
        if sample_count == 0 or sample_count > 64:
            raise DroneSimProtocolError(
                "Target-visibility case declares an invalid sample count"
            )
        samples = []
        for _ in range(sample_count):
            x, y, z, clear_value, hit_entity = reader.unpack("<3fBi")
            if clear_value not in (0, 1):
                raise DroneSimProtocolError(
                    "Target-visibility line-of-sight is not boolean"
                )
            samples.append(
                VisibilitySample(
                    position=_finite_tuple(
                        "target-visibility sample position",
                        (x, y, z),
                    ),
                    clear_line_of_sight=bool(clear_value),
                    hit_entity=hit_entity,
                )
            )
        cases.append(
            TargetVisibilityCaseSnapshot(
                stable_id=stable_id,
                camera_center=camera_center,
                target=VisibilityTarget(
                    stable_id=stable_id,
                    gta_handle=gta_handle,
                    role=role,
                    samples=tuple(samples),
                ),
            )
        )
    reader.finish()
    return TargetVisibilityBatchSnapshot(
        scenario_id=scenario_id,
        lockstep_session_id=lockstep_session_id,
        step_index=step_index,
        game_timer_ms=game_timer_ms,
        frame_count=frame_count,
        cases=tuple(cases),
    )


def _angle_error_degrees(actual, expected):
    return abs(
        (float(actual) - float(expected) + 180.0)
        % 360.0
        - 180.0
    )


def _require_dual_view_pose(
    reference,
    actual,
    expected_pitch,
):
    position_error = math.sqrt(
        sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(reference[:3], actual[:3])
        )
    )
    pitch_error = abs(float(actual[3]) - expected_pitch)
    raw_roll_error = _angle_error_degrees(actual[4], 0.0)
    raw_yaw_error = _angle_error_degrees(actual[5], reference[5])
    if expected_pitch <= -90.0 + 1.0e-2:
        orientation_error = _angle_error_degrees(
            float(actual[5]) - float(actual[4]),
            reference[5],
        )
    elif expected_pitch >= 90.0 - 1.0e-2:
        orientation_error = _angle_error_degrees(
            float(actual[5]) + float(actual[4]),
            reference[5],
        )
    else:
        orientation_error = max(
            raw_roll_error,
            raw_yaw_error,
        )
    if (
        position_error > 1.0e-3
        or pitch_error > 1.0e-2
        or orientation_error > 1.0e-2
    ):
        raise DroneSimProtocolError(
            "Dual-view camera pose invariant failed: "
            f"position={position_error:.6f}m "
            f"pitch={pitch_error:.6f}deg "
            f"raw_roll={raw_roll_error:.6f}deg "
            f"raw_yaw={raw_yaw_error:.6f}deg "
            f"physical_orientation={orientation_error:.6f}deg"
        )


def _require_same_lockstep_instant(before, after):
    fields = (
        "session_id",
        "step_index",
        "epoch_game_timer_ms",
        "game_timer_ms",
        "target_elapsed_ms",
        "actual_elapsed_ms",
        "last_advance_ms",
        "render_frames",
        "max_frame_time_ms",
    )
    changed = [
        name
        for name in fields
        if getattr(before, name) != getattr(after, name)
    ]
    if changed:
        raise DroneSimProtocolError(
            "Simulation advanced during dual-view capture: "
            + ", ".join(changed)
        )


def _require_matching_capture_calibration(oblique, nadir):
    if oblique.frame_id >= nadir.frame_id:
        raise DroneSimProtocolError(
            "Dual-view frame IDs are not strictly increasing: "
            f"{oblique.frame_id}/{nadir.frame_id}"
        )
    scalar_fields = (
        "width",
        "height",
        "fov_degrees",
        "near_clip",
        "far_clip",
    )
    changed = [
        name
        for name in scalar_fields
        if getattr(oblique, name) != getattr(nadir, name)
    ]
    if changed:
        raise DroneSimProtocolError(
            "Dual-view capture calibration changed: "
            + ", ".join(changed)
        )
    if len(oblique.projection_matrix) != 16 or len(
        nadir.projection_matrix
    ) != 16:
        raise DroneSimProtocolError(
            "Dual-view projection matrix does not contain 16 values"
        )
    if any(
        not math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        for left, right in zip(
            oblique.projection_matrix,
            nadir.projection_matrix,
        )
    ):
        raise DroneSimProtocolError(
            "Dual-view projection matrices do not match"
        )


class DroneSimClient:
    def __init__(
        self,
        host="127.0.0.5",
        port=23456,
        command_timeout=7.0,
    ):
        self.host = host
        self.port = int(port)
        self.command_timeout = float(command_timeout)
        if (
            not math.isfinite(self.command_timeout)
            or self.command_timeout <= 0
        ):
            raise ValueError("command_timeout must be a positive finite value")
        self._request_ids = count(1000)
        self._operation_lock = threading.RLock()

    def _exchange(
        self,
        message_type,
        payload=b"",
        timeout=None,
    ):
        with self._operation_lock:
            return self._exchange_unlocked(
                message_type,
                payload,
                timeout,
            )

    def _exchange_unlocked(
        self,
        message_type,
        payload=b"",
        timeout=None,
    ):
        request_id = next(self._request_ids)
        effective_timeout = (
            self.command_timeout if timeout is None else float(timeout)
        )
        with socket.create_connection(
            (self.host, self.port),
            timeout=effective_timeout,
        ) as sock:
            sock.settimeout(effective_timeout)
            sock.sendall(
                _pack_header(
                    message_type,
                    request_id,
                    len(payload),
                )
                + payload
            )
            header = _recv_exact(sock, 20)
            if header is None:
                raise DroneSimProtocolError(
                    "Connection closed before the response header"
                )
            (
                magic,
                version,
                response_type,
                _flags,
                _reserved,
                response_id,
                length,
            ) = struct.unpack("<4sBBBBQI", header)
            if magic != MAGIC or version != VERSION:
                raise DroneSimProtocolError(
                    f"Expected {MAGIC!r}/v{VERSION}, "
                    f"received {magic!r}/v{version}"
                )
            if (
                response_type != message_type
                or response_id != request_id
            ):
                raise DroneSimProtocolError(
                    "Response does not match request: "
                    f"type={response_type}/{message_type}, "
                    f"id={response_id}/{request_id}"
                )
            response_payload = (
                _recv_exact(sock, length) if length else b""
            )
            if response_payload is None:
                raise DroneSimProtocolError(
                    f"Connection closed while reading {length} response bytes"
                )
            return request_id, response_payload

    @staticmethod
    def _command_data(payload):
        if len(payload) < 8:
            raise DroneSimProtocolError(
                "Command response has no status/message header"
            )
        status, message_size = struct.unpack_from("<II", payload, 0)
        if len(payload) < 8 + message_size:
            raise DroneSimProtocolError(
                "Command response message exceeds payload"
            )
        message = payload[8 : 8 + message_size].decode(
            "utf-8", errors="strict"
        )
        if status != 0:
            raise DroneSimCommandError(status, message)
        return payload[8 + message_size :]

    def _command(
        self,
        message_type,
        payload=b"",
        timeout=None,
    ):
        _request_id, response = self._exchange(
            message_type,
            payload,
            timeout,
        )
        return self._command_data(response)

    def ping(self, payload=b""):
        if not isinstance(payload, bytes):
            raise TypeError("ping payload must be bytes")
        return self._command(TYPE_PING, payload)

    def create_camera(self):
        data = self._command(TYPE_CREATE_CAMERA)
        if len(data) != 8:
            raise DroneSimProtocolError(
                f"CREATE_CAMERA returned {len(data)} data bytes; expected 8"
            )
        return struct.unpack("<Q", data)[0]

    def stop_camera(self):
        data = self._command(TYPE_STOP_CAMERA)
        if data:
            raise DroneSimProtocolError(
                "STOP_CAMERA returned unexpected data"
            )

    def is_camera_active(self):
        data = self._command(TYPE_GET_CAMERA_STATE)
        if len(data) != 1 or data[0] not in (0, 1):
            raise DroneSimProtocolError(
                "GET_CAMERA_STATE returned an invalid boolean"
            )
        return data[0] == 1

    def require_camera_active(self):
        if not self.is_camera_active():
            raise RuntimeError(
                "DroneSim camera mode is inactive. Press F10 in GTA V or call "
                "DroneSimClient.create_camera() before validation."
            )

    def get_pose(self):
        data = self._command(TYPE_GET_POSE)
        if len(data) != 24:
            raise DroneSimProtocolError(
                f"GET_POSE returned {len(data)} data bytes; expected 24"
            )
        return struct.unpack("<6f", data)

    def set_camera_pose(
        self,
        x_world,
        y_world,
        z_world,
        yaw_degrees,
        collision_check=True,
    ):
        _require_finite(x_world, y_world, z_world, yaw_degrees)
        payload = struct.pack(
            "<4fB",
            float(x_world),
            float(y_world),
            float(z_world),
            float(yaw_degrees),
            1 if collision_check else 0,
        )
        data = self._command(TYPE_SET_CAMERA_POSE, payload)
        if len(data) != 24:
            raise DroneSimProtocolError(
                "SET_CAMERA_POSE did not return the six-component actual pose"
            )
        return struct.unpack("<6f", data)

    def set_camera_pitch(self, pitch_degrees):
        _require_finite(pitch_degrees)
        if not -90.0 <= float(pitch_degrees) <= 90.0:
            raise ValueError("pitch_degrees must be within [-90, 90]")
        data = self._command(
            TYPE_SET_CAMERA_PITCH,
            struct.pack("<f", float(pitch_degrees)),
        )
        if len(data) != 24:
            raise DroneSimProtocolError(
                "SET_CAMERA_PITCH did not return the six-component actual pose"
            )
        return struct.unpack("<6f", data)

    def set_fov(self, fov_degrees):
        _require_finite(fov_degrees)
        data = self._command(
            TYPE_SET_FOV,
            struct.pack("<f", float(fov_degrees)),
        )
        if data:
            raise DroneSimProtocolError(
                "SET_FOV returned unexpected data"
            )

    def set_time(self, hour, minute, second):
        values = (int(hour), int(minute), int(second))
        if not (
            0 <= values[0] <= 23
            and 0 <= values[1] <= 59
            and 0 <= values[2] <= 59
        ):
            raise ValueError(
                "Time must satisfy hour 0..23 and minute/second 0..59"
            )
        data = self._command(
            TYPE_SET_TIME,
            struct.pack("<3i", *values),
        )
        if data:
            raise DroneSimProtocolError(
                "SET_TIME returned unexpected data"
            )

    def set_weather(self, name):
        if not isinstance(name, str):
            raise TypeError("Weather name must be a string")
        encoded = name.encode("ascii", errors="strict")
        if (
            not encoded
            or len(encoded) > 32
            or any(
                not (
                    65 <= value <= 90
                    or 48 <= value <= 57
                    or value == 95
                )
                for value in encoded
            )
        ):
            raise ValueError(
                "Weather must contain 1..32 uppercase ASCII letters, "
                "digits, or underscores"
            )
        data = self._command(TYPE_SET_WEATHER, encoded)
        if data:
            raise DroneSimProtocolError(
                "SET_WEATHER returned unexpected data"
            )

    def teleport_player(self, x_world, y_world, z_world):
        _require_finite(x_world, y_world, z_world)
        data = self._command(
            TYPE_TELEPORT_PLAYER,
            struct.pack(
                "<3f",
                float(x_world),
                float(y_world),
                float(z_world),
            ),
        )
        if data:
            raise DroneSimProtocolError(
                "TELEPORT_PLAYER returned unexpected data"
            )

    def restore_player(self):
        data = self._command(TYPE_RESTORE_PLAYER)
        if data:
            raise DroneSimProtocolError(
                "RESTORE_PLAYER returned unexpected data"
            )

    def prepare_fire_scenario(
        self,
        anchor_world_xyz,
        seed,
        firetruck_count=1,
        pedestrian_count=32,
        blueprint_id=0,
    ):
        if len(anchor_world_xyz) != 3:
            raise ValueError("anchor_world_xyz must contain three values")
        _require_finite(*anchor_world_xyz)
        seed = int(seed)
        firetruck_count = int(firetruck_count)
        pedestrian_count = int(pedestrian_count)
        blueprint_id = int(blueprint_id)
        if not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must fit uint64")
        if not 0 <= firetruck_count <= MAX_FIRETRUCK_COUNT:
            raise ValueError(
                "firetruck_count must be in "
                f"[0, {MAX_FIRETRUCK_COUNT}]"
            )
        if not 0 <= pedestrian_count <= MAX_PEDESTRIAN_COUNT:
            raise ValueError(
                "pedestrian_count must be in "
                f"[0, {MAX_PEDESTRIAN_COUNT}]"
            )
        if not 0 <= blueprint_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("blueprint_id must fit uint64")
        data = self._command(
            TYPE_PREPARE_FIRE_SCENARIO,
            struct.pack(
                "<3fQHHQ",
                *(float(value) for value in anchor_world_xyz),
                seed,
                firetruck_count,
                pedestrian_count,
                blueprint_id,
            ),
        )
        if len(data) != 8:
            raise DroneSimProtocolError(
                "PREPARE_FIRE_SCENARIO did not return a uint64 scenario_id"
            )
        return struct.unpack("<Q", data)[0]

    def get_scenario_state(self, scenario_id):
        data = self._command(
            TYPE_GET_SCENARIO_STATE,
            struct.pack("<Q", int(scenario_id)),
        )
        snapshot = _decode_scenario_snapshot(data)
        if snapshot.scenario_id != int(scenario_id):
            raise DroneSimProtocolError(
                "Scenario snapshot ID does not match the request"
            )
        return snapshot

    def start_scenario(self, scenario_id):
        data = self._command(
            TYPE_START_SCENARIO,
            struct.pack("<Q", int(scenario_id)),
        )
        if len(data) != 16:
            raise DroneSimProtocolError(
                "START_SCENARIO did not return uint64+uint32+uint32"
            )
        returned_id, game_timer_ms, frame_count = struct.unpack(
            "<QII", data
        )
        if returned_id != int(scenario_id):
            raise DroneSimProtocolError(
                "Scenario start ID does not match the request"
            )
        return ScenarioStartInfo(
            scenario_id=returned_id,
            game_timer_ms=game_timer_ms,
            frame_count=frame_count,
        )

    def reset_scenario(self, scenario_id):
        data = self._command(
            TYPE_RESET_SCENARIO,
            struct.pack("<Q", int(scenario_id)),
        )
        if data:
            raise DroneSimProtocolError(
                "RESET_SCENARIO returned unexpected data"
            )

    def wait_scenario_ready(
        self,
        scenario_id,
        timeout=15.0,
        poll_interval=0.05,
    ):
        timeout = float(timeout)
        poll_interval = float(poll_interval)
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError(
                "timeout and poll_interval must be positive finite values"
            )
        deadline = time.monotonic() + timeout
        while True:
            snapshot = self.get_scenario_state(scenario_id)
            if snapshot.lifecycle == ScenarioLifecycle.READY:
                return snapshot
            if snapshot.lifecycle == ScenarioLifecycle.FAILED:
                raise RuntimeError(
                    "Fire scenario preparation failed: "
                    f"{snapshot.failure_message or 'unknown failure'}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Scenario {scenario_id} did not become READY "
                    f"within {timeout:.3f} seconds; "
                    f"last state={snapshot.lifecycle.name}"
                )
            time.sleep(min(poll_interval, remaining))

    def enter_lockstep(self):
        return _decode_lockstep_snapshot(
            self._command(TYPE_ENTER_LOCKSTEP)
        )

    def get_lockstep_state(self, session_id):
        session_id = int(session_id)
        if not 0 < session_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("session_id must be a non-zero uint64")
        snapshot = _decode_lockstep_snapshot(
            self._command(
                TYPE_GET_LOCKSTEP_STATE,
                struct.pack("<Q", session_id),
            )
        )
        if snapshot.session_id != session_id:
            raise DroneSimProtocolError(
                "Lockstep snapshot session does not match the request"
            )
        return snapshot

    def advance_lockstep(self, session_id):
        session_id = int(session_id)
        if not 0 < session_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("session_id must be a non-zero uint64")
        snapshot = _decode_lockstep_snapshot(
            self._command(
                TYPE_ADVANCE_LOCKSTEP,
                struct.pack("<Q", session_id),
            )
        )
        if snapshot.session_id != session_id:
            raise DroneSimProtocolError(
                "Lockstep snapshot session does not match the request"
            )
        return snapshot

    def exit_lockstep(self, session_id):
        session_id = int(session_id)
        if not 0 < session_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("session_id must be a non-zero uint64")
        data = self._command(
            TYPE_EXIT_LOCKSTEP,
            struct.pack("<Q", session_id),
        )
        if data:
            raise DroneSimProtocolError(
                "EXIT_LOCKSTEP returned unexpected data"
            )

    def query_visibility(
        self,
        scenario_id,
        lockstep_session_id,
        camera_center,
        timeout=None,
    ):
        scenario_id = int(scenario_id)
        lockstep_session_id = int(lockstep_session_id)
        if scenario_id <= 0 or lockstep_session_id <= 0:
            raise ValueError(
                "scenario_id and lockstep_session_id must be positive"
            )
        try:
            x, y, z = camera_center
        except (TypeError, ValueError) as error:
            raise ValueError(
                "camera_center must contain exactly three values"
            ) from error
        _require_finite(x, y, z)
        payload = struct.pack(
            "<QQ3f",
            scenario_id,
            lockstep_session_id,
            float(x),
            float(y),
            float(z),
        )
        data = self._command(
            TYPE_QUERY_VISIBILITY,
            payload,
            timeout=timeout,
        )
        snapshot = _decode_visibility_snapshot(data)
        if (
            snapshot.scenario_id != scenario_id
            or snapshot.lockstep_session_id
            != lockstep_session_id
        ):
            raise DroneSimProtocolError(
                "Visibility response identity does not match request"
            )
        return snapshot

    def probe_camera_start(self, x, y, altitude_agl):
        _require_finite(x, y, altitude_agl)
        altitude_agl = float(altitude_agl)
        if altitude_agl <= 0:
            raise ValueError("altitude_agl must be positive")
        data = self._command(
            TYPE_PROBE_CAMERA_START,
            struct.pack(
                "<3f",
                float(x),
                float(y),
                altitude_agl,
            ),
        )
        if len(data) != struct.calcsize("<4f"):
            raise DroneSimProtocolError(
                "PROBE_CAMERA_START returned an invalid payload size"
            )
        px, py, pz, ground_z = struct.unpack("<4f", data)
        return (
            _finite_tuple(
                "camera-start position",
                (px, py, pz),
            ),
            float(ground_z),
        )

    def probe_camera_geometry_batch(
        self,
        lockstep_session_id,
        points=(),
        segments=(),
        timeout=None,
    ):
        lockstep_session_id = int(lockstep_session_id)
        if not 0 < lockstep_session_id <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(
                "lockstep_session_id must be a non-zero uint64"
            )
        normalized_points = []
        for point in points:
            try:
                x, y, z = point
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Each geometry point must contain exactly three values"
                ) from error
            _require_finite(x, y, z)
            normalized_points.append((float(x), float(y), float(z)))
        normalized_segments = []
        for segment in segments:
            try:
                start, end = segment
                x0, y0, z0 = start
                x1, y1, z1 = end
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Each geometry segment must contain two 3D endpoints"
                ) from error
            _require_finite(x0, y0, z0, x1, y1, z1)
            normalized_segments.append(
                (
                    float(x0),
                    float(y0),
                    float(z0),
                    float(x1),
                    float(y1),
                    float(z1),
                )
            )
        total = len(normalized_points) + len(normalized_segments)
        if not 1 <= total <= 256:
            raise ValueError(
                "Geometry batch must contain 1..256 total items"
            )
        payload = bytearray(
            struct.pack(
                "<QII",
                lockstep_session_id,
                len(normalized_points),
                len(normalized_segments),
            )
        )
        for point in normalized_points:
            payload.extend(struct.pack("<3f", *point))
        for segment in normalized_segments:
            payload.extend(struct.pack("<6f", *segment))
        snapshot = _decode_geometry_batch_snapshot(
            self._command(
                TYPE_PROBE_CAMERA_GEOMETRY_BATCH,
                bytes(payload),
                timeout=timeout,
            )
        )
        if snapshot.lockstep_session_id != lockstep_session_id:
            raise DroneSimProtocolError(
                "Camera-geometry response session does not match request"
            )
        if (
            len(snapshot.point_clear) != len(normalized_points)
            or len(snapshot.segment_clear) != len(normalized_segments)
        ):
            raise DroneSimProtocolError(
                "Camera-geometry response counts do not match request"
            )
        return snapshot

    def query_target_visibility_batch(
        self,
        scenario_id,
        lockstep_session_id,
        cases,
        timeout=None,
    ):
        scenario_id = int(scenario_id)
        lockstep_session_id = int(lockstep_session_id)
        if (
            not 0 < scenario_id <= 0xFFFFFFFFFFFFFFFF
            or not 0 < lockstep_session_id <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError(
                "scenario_id and lockstep_session_id must be "
                "non-zero uint64 values"
            )
        normalized_cases = []
        for item in cases:
            if isinstance(item, TargetVisibilityCase):
                stable_id = int(item.stable_id)
                camera_center = item.camera_center
            else:
                try:
                    stable_id, camera_center = item
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Each target-visibility case must contain "
                        "stable_id and camera_center"
                    ) from error
                stable_id = int(stable_id)
            if not 0 < stable_id <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(
                    "Target stable_id must be a non-zero uint64"
                )
            try:
                x, y, z = camera_center
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Target camera_center must contain three values"
                ) from error
            _require_finite(x, y, z)
            normalized_cases.append(
                TargetVisibilityCase(
                    stable_id=stable_id,
                    camera_center=(
                        float(x),
                        float(y),
                        float(z),
                    ),
                )
            )
        if not 1 <= len(normalized_cases) <= 64:
            raise ValueError(
                "Target-visibility batch must contain 1..64 cases"
            )
        payload = bytearray(
            struct.pack(
                "<QQI",
                scenario_id,
                lockstep_session_id,
                len(normalized_cases),
            )
        )
        for item in normalized_cases:
            payload.extend(
                struct.pack(
                    "<Q3f",
                    item.stable_id,
                    *item.camera_center,
                )
            )
        snapshot = _decode_target_visibility_batch_snapshot(
            self._command(
                TYPE_QUERY_TARGET_VISIBILITY_BATCH,
                bytes(payload),
                timeout=timeout,
            )
        )
        if (
            snapshot.scenario_id != scenario_id
            or snapshot.lockstep_session_id != lockstep_session_id
            or len(snapshot.cases) != len(normalized_cases)
        ):
            raise DroneSimProtocolError(
                "Target-visibility response identity/count does not "
                "match request"
            )
        for requested, returned in zip(
            normalized_cases,
            snapshot.cases,
        ):
            if (
                requested.stable_id != returned.stable_id
                or any(
                    abs(left - right) > 1.0e-3
                    for left, right in zip(
                        requested.camera_center,
                        returned.camera_center,
                    )
                )
            ):
                raise DroneSimProtocolError(
                    "Target-visibility response order does not match request"
                )
        return snapshot

    def capture(self, timeout_ms=5000):
        timeout_ms = int(timeout_ms)
        if timeout_ms <= 0 or timeout_ms > 60000:
            raise ValueError(
                "timeout_ms must be in the range [1, 60000]"
            )
        request_id, payload = self._exchange(
            TYPE_CAPTURE,
            struct.pack("<I", timeout_ms),
            timeout=timeout_ms / 1000.0 + 2.0,
        )
        if len(payload) < 4:
            raise DroneSimProtocolError(
                "Capture response has no status field"
            )

        status = struct.unpack_from("<I", payload, 0)[0]
        if status != 0:
            if len(payload) < 8:
                raise DroneSimProtocolError(
                    "Failed capture response has no error length"
                )
            error_size = struct.unpack_from("<I", payload, 4)[0]
            if len(payload) != 8 + error_size:
                raise DroneSimProtocolError(
                    "Failed capture response length does not match error length"
                )
            message = payload[8:].decode("utf-8", errors="strict")
            raise CaptureError(status, message)

        if len(payload) < CAPTURE_METADATA_SIZE:
            raise DroneSimProtocolError(
                f"Capture metadata has {len(payload)} bytes; "
                f"expected at least {CAPTURE_METADATA_SIZE}"
            )
        fields = struct.unpack_from(
            CAPTURE_METADATA_FORMAT, payload, 0
        )
        (
            _status,
            frame_id,
            width,
            height,
            rgb_size,
            depth_size,
            fov,
            near_clip,
            far_clip,
            *matrices,
        ) = fields
        expected_rgb = width * height * 3
        expected_depth = width * height * 4
        if (
            rgb_size != expected_rgb
            or depth_size != expected_depth
        ):
            raise DroneSimProtocolError(
                "Capture sizes do not match declared dimensions: "
                f"rgb={rgb_size}/{expected_rgb}, "
                f"depth={depth_size}/{expected_depth}"
            )
        expected_total = (
            CAPTURE_METADATA_SIZE + rgb_size + depth_size
        )
        if len(payload) != expected_total:
            raise DroneSimProtocolError(
                f"Capture response has {len(payload)} bytes; "
                f"expected {expected_total}"
            )
        rgb_offset = CAPTURE_METADATA_SIZE
        depth_offset = rgb_offset + rgb_size
        return RgbdFrame(
            request_id=request_id,
            frame_id=frame_id,
            width=width,
            height=height,
            rgb=payload[rgb_offset:depth_offset],
            depth=payload[depth_offset:],
            fov_degrees=fov,
            near_clip=near_clip,
            far_clip=far_clip,
            projection_matrix=tuple(matrices[:16]),
            view_matrix=tuple(matrices[16:]),
        )

    def _capture_lockstep_rgbd_pair(
        self,
        session_id,
        timeout_ms,
    ):
        with self._operation_lock:
            before_clock = self.get_lockstep_state(session_id)
            reference_pose = self.get_pose()
            if _angle_error_degrees(reference_pose[4], 0.0) > 1.0e-2:
                raise DroneSimProtocolError(
                    "Dual-view capture requires camera roll to be zero; "
                    f"actual roll={reference_pose[4]:.6f}deg"
                )

            operation_error = None
            oblique = None
            nadir = None
            try:
                oblique_pose = self.set_camera_pitch(
                    OBLIQUE_PITCH_DEGREES
                )
                _require_dual_view_pose(
                    reference_pose,
                    oblique_pose,
                    OBLIQUE_PITCH_DEGREES,
                )
                oblique = self.capture(timeout_ms)

                nadir_pose = self.set_camera_pitch(
                    NADIR_PITCH_DEGREES
                )
                _require_dual_view_pose(
                    reference_pose,
                    nadir_pose,
                    NADIR_PITCH_DEGREES,
                )
                nadir = self.capture(timeout_ms)
            except BaseException as error:
                operation_error = error

            restore_error = None
            restored_pose = None
            try:
                restored_pose = self.set_camera_pitch(
                    OBLIQUE_PITCH_DEGREES
                )
            except BaseException as error:
                restore_error = error

            if operation_error is not None:
                if restore_error is not None:
                    message = (
                        "Restoring the canonical -45 degree pitch also "
                        f"failed: {restore_error}"
                    )
                    if hasattr(operation_error, "add_note"):
                        operation_error.add_note(message)
                    else:
                        raise restore_error from operation_error
                raise operation_error
            if restore_error is not None:
                raise restore_error

            _require_dual_view_pose(
                reference_pose,
                restored_pose,
                OBLIQUE_PITCH_DEGREES,
            )
            after_clock = self.get_lockstep_state(session_id)
            _require_same_lockstep_instant(
                before_clock,
                after_clock,
            )
            _require_matching_capture_calibration(
                oblique,
                nadir,
            )
            return LockstepRgbdPair(
                clock=after_clock,
                oblique=oblique,
                nadir=nadir,
            )


class LockstepSession:
    """Strict context manager for a single plugin lockstep session.

    An active scenario must be Reset explicitly before leaving the context.
    Cleanup errors remain visible; this class never resumes a running scene.
    """

    def __init__(self, client):
        if not isinstance(client, DroneSimClient):
            raise TypeError("client must be a DroneSimClient")
        self.client = client
        self._snapshot = None

    @property
    def snapshot(self):
        if self._snapshot is None:
            raise RuntimeError("LockstepSession is not active")
        return self._snapshot

    @property
    def session_id(self):
        return self.snapshot.session_id

    def __enter__(self):
        if self._snapshot is not None:
            raise RuntimeError("LockstepSession is already active")
        self._snapshot = self.client.enter_lockstep()
        return self

    def refresh(self):
        self._snapshot = self.client.get_lockstep_state(
            self.session_id
        )
        return self._snapshot

    def advance(self):
        self._snapshot = self.client.advance_lockstep(
            self.session_id
        )
        return self._snapshot

    def capture_rgbd_pair(self, timeout_ms=5000):
        pair = self.client._capture_lockstep_rgbd_pair(
            self.session_id,
            timeout_ms,
        )
        self._snapshot = pair.clock
        return pair

    def close(self):
        session_id = self.session_id
        self.client.exit_lockstep(session_id)
        self._snapshot = None

    def __exit__(self, exception_type, exception, traceback):
        if self._snapshot is None:
            return False
        try:
            self.close()
        except Exception as cleanup_error:
            if exception is None:
                raise
            raise cleanup_error from exception
        return False


class RelativePoseController:
    """Agent-facing continuous relative-pose wrapper.

    dx_body is forward, dy_body is right, dz_world is vertical, and dyaw is
    added in degrees. The plugin receives only the resulting absolute pose.
    """

    def __init__(self, client, collision_check=True):
        if not isinstance(client, DroneSimClient):
            raise TypeError("client must be a DroneSimClient")
        self.client = client
        self.collision_check = bool(collision_check)
        self._pose = None

    @property
    def pose(self):
        if self._pose is None:
            raise RuntimeError(
                "RelativePoseController is not synchronized"
            )
        return self._pose

    def synchronize(self):
        self._pose = self.client.get_pose()
        return self._pose

    def step_relative(
        self,
        dx_body,
        dy_body,
        dz_world,
        dyaw,
    ):
        _require_finite(dx_body, dy_body, dz_world, dyaw)
        if self._pose is None:
            self.synchronize()

        x, y, z, _pitch, _roll, yaw = self._pose
        yaw_radians = math.radians(yaw)
        forward_x = -math.sin(yaw_radians)
        forward_y = math.cos(yaw_radians)
        right_x = math.cos(yaw_radians)
        right_y = math.sin(yaw_radians)
        target_x = (
            x
            + forward_x * float(dx_body)
            + right_x * float(dy_body)
        )
        target_y = (
            y
            + forward_y * float(dx_body)
            + right_y * float(dy_body)
        )
        target_z = z + float(dz_world)
        target_yaw = (
            yaw + float(dyaw) + 180.0
        ) % 360.0 - 180.0

        actual_pose = self.client.set_camera_pose(
            target_x,
            target_y,
            target_z,
            target_yaw,
            collision_check=self.collision_check,
        )
        self._pose = actual_pose
        return actual_pose
