import socket
import struct

MAGIC = b"DSV2"
VERSION = 1

TYPE_CREATE_CAMERA = 1
TYPE_MOVE = 2
TYPE_ROTATE = 3
TYPE_SET_FOV = 4
TYPE_CAPTURE = 5
TYPE_PING = 6

def send(sock, t, req_id, payload=b""):
    hdr = struct.pack("4sBBBBQI", MAGIC, VERSION, t, 0, 0, req_id, len(payload))
    sock.sendall(hdr + payload)

def recv(sock):
    def recvn(n):
        b = b""
        while len(b) < n:
            r = sock.recv(n - len(b))
            if not r:
                return None
            b += r
        return b
    h = recvn(20)
    if not h:
        return None
    magic, ver, t, flags, reserved, req_id, length = struct.unpack("4sBBBBQI", h)
    p = recvn(length) if length else b""
    return t, req_id, p

def create_camera(host="127.0.0.1", port=23456):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    send(s, TYPE_CREATE_CAMERA, 1)
    t, rid, p = recv(s)
    s.close()
    cam_id = struct.unpack("Q", p)[0] if p else 0
    return cam_id

def move(dx, dy, dz, host="127.0.0.1", port=23456):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    payload = struct.pack("fff", dx, dy, dz)
    send(s, TYPE_MOVE, 2, payload)
    recv(s)
    s.close()

def rotate(rx, ry, rz, host="127.0.0.1", port=23456):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    payload = struct.pack("fff", rx, ry, rz)
    send(s, TYPE_ROTATE, 3, payload)
    recv(s)
    s.close()

def set_fov(fov, host="127.0.0.1", port=23456):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    payload = struct.pack("f", fov)
    send(s, TYPE_SET_FOV, 4, payload)
    recv(s)
    s.close()

def capture(host="127.0.0.1", port=23456):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    send(s, TYPE_CAPTURE, 5)
    t, rid, p = recv(s)
    s.close()
    if not p:
        return None
    rgb_size, depth_size, w, h = struct.unpack("IIII", p[:16])
    rgb = p[16:16+rgb_size]
    depth = p[16+rgb_size:16+rgb_size+depth_size]
    return w, h, rgb, depth

if __name__ == "__main__":
    cam = create_camera()
    set_fov(60.0)
    move(5.0, 0.0, 0.0)
    rotate(0.0, 0.0, 45.0)
    w, h, rgb, depth = capture()
    with open("rgb.bin", "wb") as f:
        f.write(rgb)
    with open("depth.bin", "wb") as f:
        f.write(depth)
