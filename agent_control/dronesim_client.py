import math
import socket
import struct
from dataclasses import dataclass
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

    def _exchange(
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
