import os
import time
import json
import numpy as np
import msvcrt
from datetime import datetime
from dronesim_client import DroneSimClient

ACTIONS = {
    'w': {'code': 1, 'move': (5.0, 0.0, 0.0), 'rotate': (0.0, 0.0, 0.0)},
    'q': {'code': 2, 'move': (0.0, 0.0, 0.0), 'rotate': (0.0, 0.0, 45.0)},
    'e': {'code': 3, 'move': (0.0, 0.0, 0.0), 'rotate': (0.0, 0.0, -45.0)},
    'j': {'code': 4, 'move': (0.0, 0.0, 5.0), 'rotate': (0.0, 0.0, 0.0)},
    'k': {'code': 5, 'move': (0.0, 0.0, -5.0), 'rotate': (0.0, 0.0, 0.0)},
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_frame(base_dir, idx, captures, action_code):
    frame_dir = os.path.join(base_dir, f"frame_{idx:03d}")
    ensure_dir(frame_dir)
    meta = { 'action_code': action_code, 'captures': {} }
    for direction, (w,h,rgb,depth) in captures.items():
        rgb_path = os.path.join(frame_dir, f"{direction}_rgb.npy")
        depth_path = os.path.join(frame_dir, f"{direction}_depth.npy")
        np.save(rgb_path, np.frombuffer(rgb, dtype=np.uint8))
        np.save(depth_path, np.frombuffer(depth, dtype=np.float32))
        meta['captures'][direction] = {
            'rgb_path': os.path.relpath(rgb_path, base_dir),
            'depth_path': os.path.relpath(depth_path, base_dir),
            'width': int(w),
            'height': int(h)
        }
    return meta

def capture_directions(cli):
    # front
    w,h,rgb,depth = cli.capture()
    cap_front = (w,h,rgb,depth)
    # back
    cli.rotate(0.0, 0.0, 180.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    cap_back = (w,h,rgb,depth)
    cli.rotate(0.0, 0.0, -180.0); time.sleep(0.05)
    # left
    cli.rotate(0.0, 0.0, -90.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    cap_left = (w,h,rgb,depth)
    cli.rotate(0.0, 0.0, 90.0); time.sleep(0.05)
    # right
    cli.rotate(0.0, 0.0, 90.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    cap_right = (w,h,rgb,depth)
    cli.rotate(0.0, 0.0, -90.0); time.sleep(0.05)
    # down
    cli.rotate(-90.0, 0.0, 0.0); time.sleep(0.2)
    w,h,rgb,depth = cli.capture()
    cap_down = (w,h,rgb,depth)
    cli.rotate(90.0, 0.0, 0.0); time.sleep(0.05)
    return {
        'front': cap_front,
        'back': cap_back,
        'left': cap_left,
        'right': cap_right,
        'down': cap_down
    }

def main():
    cli = DroneSimClient()
    session_dir = os.path.join('sessions', datetime.now().strftime('%Y%m%d_%H%M%S'))
    ensure_dir(session_dir)
    print('Press y to start control; n to end; w/q/e/j/k to act.')
    started = False
    frames = []
    idx = 0
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode('utf-8').lower()
            if ch == 'y' and not started:
                cli.create_camera()
                started = True
                print('Control started')
            elif ch == 'n':
                print('Ending session and saving...')
                break
            elif started and ch in ACTIONS:
                captures = capture_directions(cli)
                action = ACTIONS[ch]
                meta = save_frame(session_dir, idx, captures, action['code'])
                frames.append(meta)
                idx += 1
                dx,dy,dz = action['move']
                rx,ry,rz = action['rotate']
                if dx or dy or dz:
                    cli.move(dx,dy,dz)
                if rx or ry or rz:
                    cli.rotate(rx,ry,rz)
                print(f"Captured frame {idx} and executed action {ch}")
        time.sleep(0.01)
    with open(os.path.join(session_dir, 'session.json'), 'w', encoding='utf-8') as f:
        json.dump({ 'frames': frames }, f, ensure_ascii=False, indent=2)
    print('Saved session to', session_dir)

if __name__ == '__main__':
    main()
