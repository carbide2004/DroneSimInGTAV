import socket
import struct
import time
from dataclasses import dataclass
from itertools import count

MAGIC = b"DSV3"
VERSION = 3

TYPE_CREATE_CAMERA = 1
TYPE_MOVE = 2
TYPE_ROTATE = 3
TYPE_SET_FOV = 4
TYPE_CAPTURE = 5
TYPE_PING = 6
TYPE_GET_POSE = 7
TYPE_SET_TIME = 8
TYPE_SET_WEATHER = 9
TYPE_STOP_CAMERA = 10
TYPE_CREATE_ACCIDENT = 11
TYPE_GET_RECORDING_INFO = 12
TYPE_SET_RECORDING_SESSION = 13
TYPE_CREATE_FIRE = 14
TYPE_CREATE_ARREST = 15
TYPE_SET_POSTURE = 16
TYPE_TELEPORT_PLAYER = 17
TYPE_RESTORE_PLAYER = 18
TYPE_CLEAR_SCENE = 19
TYPE_GET_CAMERA_STATE = 20


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


class CaptureError(RuntimeError):
    def __init__(self, status, message):
        self.status = int(status)
        self.status_name = CAPTURE_STATUS.get(self.status, "UNKNOWN_CAPTURE_STATUS")
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
        # Existing call sites that unpack w, h, rgb, depth keep working while
        # V3 metadata remains available as named fields.
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


def _pack_header(t, req_id, length):
    return struct.pack("<4sBBBBQI", MAGIC, VERSION, t, 0, 0, req_id, length)

def _recv_exact(sock, n):
    data = bytearray(n)
    view = memoryview(data)
    received = 0
    while received < n:
        r = sock.recv_into(view[received:], n - received)
        if not r:
            return None
        received += r
    return bytes(data)

class DroneSimClient:
    def __init__(self, host="127.0.0.5", port=23456):
        self.host = host
        self.port = port
        self._request_ids = count(1000)

    def _send(self, t, req_id, payload=b"", timeout=None):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if timeout is not None:
            s.settimeout(timeout)
        s.connect((self.host, self.port))
        s.sendall(_pack_header(t, req_id, len(payload)) + payload)
        return s

    def _recv(self, s):
        h = _recv_exact(s, 20)
        if not h:
            s.close()
            raise DroneSimProtocolError(
                "Connection closed before the response header was received"
            )
        magic, ver, t, flags, reserved, req_id, length = struct.unpack(
            "<4sBBBBQI", h
        )
        if magic != MAGIC or ver != VERSION:
            s.close()
            raise DroneSimProtocolError(
                f"Expected {MAGIC!r}/v{VERSION}, received {magic!r}/v{ver}"
            )
        p = _recv_exact(s, length) if length else b""
        s.close()
        if p is None:
            raise DroneSimProtocolError(
                f"Connection closed while reading {length} response bytes"
            )
        return t, req_id, p

    def create_camera(self):
        s = self._send(TYPE_CREATE_CAMERA, 1)
        t, rid, p = self._recv(s)
        cam_id = struct.unpack("Q", p)[0] if p else 0
        time.sleep(2.5) # 服务器端启动相机需要等待2秒
        return cam_id

    def move(self, dx, dy, dz):
        payload = struct.pack("fff", dx, dy, dz)
        s = self._send(TYPE_MOVE, 2, payload)
        self._recv(s)
        time.sleep(0.1)

    def move_to(self, x, y, z):
        """将相机移动到绝对位置，保持当前旋转。"""
        current_pose = self.get_pose()
        if current_pose is None:
            return
        _, _, _, rx, ry, rz = current_pose
        self.set_posture(x, y, z, rx, ry, rz)

    def rotate(self, rx, ry, rz):
        payload = struct.pack("fff", rx, ry, rz)
        s = self._send(TYPE_ROTATE, 3, payload)
        self._recv(s)
        time.sleep(0.1)

    def set_rotation(self, rx, ry, rz):
        """设置相机绝对旋转，保持当前位置。"""
        current_pose = self.get_pose()
        if current_pose is None:
            return
        x, y, z = current_pose[:3]
        self.set_posture(x, y, z, rx, ry, rz)

    def set_posture(self, x, y, z, rx, ry, rz):
        """设置相机绝对位置和旋转。"""
        payload = struct.pack("ffffff", x, y, z, rx, ry, rz)
        s = self._send(TYPE_SET_POSTURE, 16, payload)
        self._recv(s)
        time.sleep(0.1)

    def set_fov(self, fov):
        payload = struct.pack("f", fov)
        s = self._send(TYPE_SET_FOV, 4, payload)
        self._recv(s)
        time.sleep(0.1)

    def capture(self, timeout_ms=5000):
        timeout_ms = int(timeout_ms)
        if timeout_ms <= 0 or timeout_ms > 60000:
            raise ValueError("timeout_ms must be in the range [1, 60000]")
        request_id = next(self._request_ids)
        s = self._send(
            TYPE_CAPTURE,
            request_id,
            struct.pack("<I", timeout_ms),
            timeout=timeout_ms / 1000.0 + 2.0,
        )
        t, rid, p = self._recv(s)
        if t != TYPE_CAPTURE or rid != request_id:
            raise DroneSimProtocolError(
                f"Capture response mismatch: type={t}, request_id={rid}"
            )
        if len(p) < 4:
            raise DroneSimProtocolError("Capture response has no status field")

        status = struct.unpack_from("<I", p, 0)[0]
        if status != 0:
            if len(p) < 8:
                raise DroneSimProtocolError(
                    "Failed capture response has no error length"
                )
            error_size = struct.unpack_from("<I", p, 4)[0]
            if len(p) != 8 + error_size:
                raise DroneSimProtocolError(
                    "Failed capture response length does not match error length"
                )
            message = p[8:].decode("utf-8", errors="strict")
            raise CaptureError(status, message)

        if len(p) < CAPTURE_METADATA_SIZE:
            raise DroneSimProtocolError(
                f"Capture metadata has {len(p)} bytes; "
                f"expected at least {CAPTURE_METADATA_SIZE}"
            )
        fields = struct.unpack_from(CAPTURE_METADATA_FORMAT, p, 0)
        (
            _,
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
        if rgb_size != expected_rgb or depth_size != expected_depth:
            raise DroneSimProtocolError(
                "Capture payload sizes do not match declared dimensions: "
                f"rgb={rgb_size}/{expected_rgb}, "
                f"depth={depth_size}/{expected_depth}"
            )
        expected_total = CAPTURE_METADATA_SIZE + rgb_size + depth_size
        if len(p) != expected_total:
            raise DroneSimProtocolError(
                f"Capture response has {len(p)} bytes; expected {expected_total}"
            )
        rgb_offset = CAPTURE_METADATA_SIZE
        depth_offset = rgb_offset + rgb_size
        return RgbdFrame(
            request_id=rid,
            frame_id=frame_id,
            width=width,
            height=height,
            rgb=p[rgb_offset:depth_offset],
            depth=p[depth_offset:],
            fov_degrees=fov,
            near_clip=near_clip,
            far_clip=far_clip,
            projection_matrix=tuple(matrices[:16]),
            view_matrix=tuple(matrices[16:]),
        )

    def get_pose(self):
        s = self._send(TYPE_GET_POSE, 6)
        t, rid, p = self._recv(s)
        if not p or len(p) < 24:
            return None
        x,y,z,rx,ry,rz = struct.unpack("ffffff", p)
        time.sleep(0.1)
        return x,y,z,rx,ry,rz

    def is_camera_active(self):
        request_id = next(self._request_ids)
        s = self._send(TYPE_GET_CAMERA_STATE, request_id)
        t, rid, payload = self._recv(s)
        if t != TYPE_GET_CAMERA_STATE or rid != request_id:
            raise DroneSimProtocolError(
                f"Camera-state response mismatch: type={t}, request_id={rid}"
            )
        if len(payload) != 1:
            raise DroneSimProtocolError(
                "Camera-state query timed out or returned an invalid payload"
            )
        if payload[0] not in (0, 1):
            raise DroneSimProtocolError(
                f"Camera-state response has invalid value {payload[0]}"
            )
        return payload[0] == 1

    def require_camera_active(self):
        if not self.is_camera_active():
            raise RuntimeError(
                "DroneSim camera mode is inactive. Press F10 in GTA V or call "
                "DroneSimClient.create_camera() before running validation."
            )

    def set_time(self, hour, minute, second):
        payload = struct.pack("iii", int(hour), int(minute), int(second))
        s = self._send(TYPE_SET_TIME, 7, payload)
        self._recv(s)
        time.sleep(0.1)

    def set_weather(self, name):
        data = name.encode('ascii')
        s = self._send(TYPE_SET_WEATHER, 8, data)
        self._recv(s)
        time.sleep(0.1)

    def stop_camera(self):
        s = self._send(TYPE_STOP_CAMERA, 9)
        self._recv(s)
        time.sleep(0.1)

    def create_accident(self):
        s = self._send(TYPE_CREATE_ACCIDENT, 11)
        t, rid, p = self._recv(s)
        if not p or len(p) < 12:
            return None
        x, y, z = struct.unpack("fff", p[:12])
        return x, y, z

    def get_recording_info(self):
        s = self._send(TYPE_GET_RECORDING_INFO, 12)
        t, rid, p = self._recv(s)
        if not p or len(p) < 7:
            return None
        enabled = struct.unpack("B", p[:1])[0]
        step = struct.unpack("i", p[1:5])[0]
        path_len = struct.unpack("H", p[5:7])[0]
        path = p[7:7+path_len].decode("utf-8", errors="replace") if path_len else ""
        return {"enabled": bool(enabled), "step": int(step), "session_dir": path}

    def set_recording_session(self, session_name, task=None):
        if task is None:
            payload = str(session_name).encode("utf-8")
        else:
            payload = (str(session_name) + "\n" + str(task)).encode("utf-8")
        s = self._send(TYPE_SET_RECORDING_SESSION, 13, payload)
        self._recv(s)
        time.sleep(0.1)

    def create_fire(self):
        s = self._send(TYPE_CREATE_FIRE, 14)
        t, rid, p = self._recv(s)
        if not p or len(p) < 16:
            return None
        x, y, z = struct.unpack("fff", p[:12])
        fire_id = struct.unpack("i", p[12:16])[0]
        return x, y, z, fire_id

    def create_arrest(self):
        s = self._send(TYPE_CREATE_ARREST, 15)
        t, rid, p = self._recv(s)
        if not p or len(p) < 12:
            return None
        x, y, z = struct.unpack("fff", p[:12])
        return x, y, z
    
    def teleport_player(self, x, y, z):
        """将玩家角色传送到异常中心，并设置为无敌和不可见。"""
        payload = struct.pack("fff", x, y, z)
        s = self._send(TYPE_TELEPORT_PLAYER, 17, payload)
        self._recv(s)
        time.sleep(1.5)  # Longer wait for view switching and teleportation

    def restore_player(self):
        """验证结束后恢复玩家正常状态。"""
        s = self._send(TYPE_RESTORE_PLAYER, 18)
        self._recv(s)
        time.sleep(1.5)  # Longer wait for view switching and restoration

    def clear_scene(self):
        """清理插件创建的异常车辆、行人和火焰。"""
        s = self._send(TYPE_CLEAR_SCENE, 19)
        self._recv(s)
        time.sleep(0.5)

def visualize(rgb_bytes, depth_bytes, w, h):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    expected_rgb = int(w) * int(h) * 3
    expected_depth = int(w) * int(h) * 4
    if len(rgb_bytes) != expected_rgb or len(depth_bytes) != expected_depth:
        raise DroneSimProtocolError(
            "RGB-D byte counts do not match the declared dimensions"
        )
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 3))
    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))
    vmin = float(np.percentile(depth, 5))
    vmax = float(np.percentile(depth, 95))
    if vmin == vmax:
        vmin, vmax = float(depth.min()), float(depth.max())
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.title("RGB"); plt.imshow(rgb); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Depth"); im = plt.imshow(depth, cmap='magma', vmin=vmin, vmax=vmax); plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()
    return True
