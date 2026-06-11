import socket
import struct
import time

MAGIC = b"DSV2"
VERSION = 1

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

def _pack_header(t, req_id, length):
    return struct.pack("4sBBBBQI", MAGIC, VERSION, t, 0, 0, req_id, length)

def _recv_exact(sock, n):
    b = b""
    while len(b) < n:
        r = sock.recv(n - len(b))
        if not r:
            return None
        b += r
    return b

class DroneSimClient:
    def __init__(self, host="127.0.0.5", port=23456):
        self.host = host
        self.port = port

    def _send(self, t, req_id, payload=b""):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        s.sendall(_pack_header(t, req_id, len(payload)) + payload)
        return s

    def _recv(self, s):
        h = _recv_exact(s, 20)
        if not h:
            s.close()
            return None
        magic, ver, t, flags, reserved, req_id, length = struct.unpack("4sBBBBQI", h)
        p = _recv_exact(s, length) if length else b""
        s.close()
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

    def capture(self):
        s = self._send(TYPE_CAPTURE, 5)
        t, rid, p = self._recv(s)
        if not p or len(p) < 16:
            return None
        rgb_size, depth_size, w, h = struct.unpack("IIII", p[:16])
        rgb = p[16:16+rgb_size]
        depth = p[16+rgb_size:16+rgb_size+depth_size]
        time.sleep(0.5)
        return w, h, rgb, depth

    def get_pose(self):
        s = self._send(TYPE_GET_POSE, 6)
        t, rid, p = self._recv(s)
        if not p or len(p) < 24:
            return None
        x,y,z,rx,ry,rz = struct.unpack("ffffff", p)
        time.sleep(0.1)
        return x,y,z,rx,ry,rz

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
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
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
