from dronesim_client import DroneSimClient
import os
import json
import time
from datetime import datetime

ACTIONS = {
    1: {"move": (5.0, 0.0, 0.0), "rotate": (0.0, 0.0, 0.0)},
    2: {"move": (0.0, 0.0, 0.0), "rotate": (0.0, 0.0, 45.0)},
    3: {"move": (0.0, 0.0, 0.0), "rotate": (0.0, 0.0, -45.0)},
    4: {"move": (0.0, 0.0, 5.0), "rotate": (0.0, 0.0, 0.0)},
    5: {"move": (0.0, 0.0, -5.0), "rotate": (0.0, 0.0, 0.0)},
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_bin(path, data):
    with open(path, "wb") as f:
        f.write(data)

def capture_five_directions(cli, base_dir, frame_idx):
    rgb_dir = os.path.join(base_dir, "RGB")
    depth_dir = os.path.join(base_dir, "Depth")
    ensure_dir(rgb_dir); ensure_dir(depth_dir)

    result = {}

    # front
    w,h,rgb,depth = cli.capture()
    fr = f"frame_{frame_idx:03d}__front.bin"
    save_bin(os.path.join(rgb_dir, fr), rgb)
    save_bin(os.path.join(depth_dir, fr), depth)
    result["front"] = {"rgb_path": os.path.join("RGB", fr), "depth_path": os.path.join("Depth", fr), "width": int(w), "height": int(h)}

    # back
    cli.rotate(0.0, 0.0, 180.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    fr = f"frame_{frame_idx:03d}__back.bin"
    save_bin(os.path.join(rgb_dir, fr), rgb)
    save_bin(os.path.join(depth_dir, fr), depth)
    result["back"] = {"rgb_path": os.path.join("RGB", fr), "depth_path": os.path.join("Depth", fr), "width": int(w), "height": int(h)}
    cli.rotate(0.0, 0.0, -180.0); time.sleep(0.05)

    # left
    cli.rotate(0.0, 0.0, -90.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    fr = f"frame_{frame_idx:03d}__left.bin"
    save_bin(os.path.join(rgb_dir, fr), rgb)
    save_bin(os.path.join(depth_dir, fr), depth)
    result["left"] = {"rgb_path": os.path.join("RGB", fr), "depth_path": os.path.join("Depth", fr), "width": int(w), "height": int(h)}
    cli.rotate(0.0, 0.0, 90.0); time.sleep(0.05)

    # right
    cli.rotate(0.0, 0.0, 90.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    fr = f"frame_{frame_idx:03d}__right.bin"
    save_bin(os.path.join(rgb_dir, fr), rgb)
    save_bin(os.path.join(depth_dir, fr), depth)
    result["right"] = {"rgb_path": os.path.join("RGB", fr), "depth_path": os.path.join("Depth", fr), "width": int(w), "height": int(h)}
    cli.rotate(0.0, 0.0, -90.0); time.sleep(0.05)

    # down
    cli.rotate(-90.0, 0.0, 0.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    fr = f"frame_{frame_idx:03d}__down.bin"
    save_bin(os.path.join(rgb_dir, fr), rgb)
    save_bin(os.path.join(depth_dir, fr), depth)
    result["down"] = {"rgb_path": os.path.join("RGB", fr), "depth_path": os.path.join("Depth", fr), "width": int(w), "height": int(h)}
    cli.rotate(90.0, 0.0, 0.0); time.sleep(0.05)

    return result

def main():
    # 1. 等待5秒
    print("Waiting 5 seconds...")
    time.sleep(5)

    # 2. 启动相机模式
    cli = DroneSimClient()
    cli.create_camera()
    cli.set_fov(60.0)

    # 3. 设置天气为 RAIN
    cli.set_time(12, 0, 0)
    cli.set_weather("RAIN")

    # 目录与序列
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('data', ts)
    ensure_dir(base_dir)
    frames = []
    sequence = [1,2,3,4,5]

    # 4. 采集循环：采集五方向 + 记录动作与位姿 → 执行动作
    for idx, code in enumerate(sequence):
        # 位姿
        pose = cli.get_pose()
        if pose is None:
            # 简单重试一次
            time.sleep(0.5)
            pose = cli.get_pose()
        # 采集五方向
        captures = capture_five_directions(cli, base_dir, idx)

        frames.append({
            'index': idx,
            'action_code': code,
            'pose': {
                'x': float(pose[0]) if pose else None,
                'y': float(pose[1]) if pose else None,
                'z': float(pose[2]) if pose else None,
                'rx': float(pose[3]) if pose else None,
                'ry': float(pose[4]) if pose else None,
                'rz': float(pose[5]) if pose else None,
            },
            'captures': captures
        })

        # 执行动作
        act = ACTIONS[code]
        dx,dy,dz = act['move']
        rx,ry,rz = act['rotate']
        if dx or dy or dz:
            cli.move(dx,dy,dz)
        if rx or ry or rz:
            cli.rotate(rx,ry,rz)

    # 5. 结束相机模式
    cli.stop_camera()

    # 保存 JSON
    with open(os.path.join(base_dir, 'serial.json'), 'w', encoding='utf-8') as f:
        json.dump({'timestamp': ts, 'frames': frames}, f, ensure_ascii=False, indent=2)
    print('Saved session to', base_dir)

if __name__ == "__main__":
    main()
