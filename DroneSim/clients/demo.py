from dronesim_client import DroneSimClient
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_rgb_png(path, w, h, rgb_bytes):
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    Image.fromarray(rgb).save(path)

def save_depth_png(path, w, h, depth_bytes):
    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))
    vmin = float(np.percentile(depth, 5))
    vmax = float(np.percentile(depth, 95))
    if vmin == vmax:
        vmin, vmax = float(depth.min()), float(depth.max())
    plt.imsave(path, depth, cmap="magma", vmin=vmin, vmax=vmax)

def main():
    print("等待5秒...")
    time.sleep(5)

    cli = DroneSimClient()
    cli.create_camera()

    times = [
        (6, 0, 0),
        (12, 0, 0),
        (18, 0, 0),
        (23, 0, 0),
    ]
    weathers = [
        # "CLEAR",
        # "RAIN",
        # "FOGGY",
        "THUNDER",
    ]

    out_dir = Path("data") / "demo"
    ensure_dir(out_dir)

    for weather in weathers:
        cli.set_weather(weather)
        for h, m, s in times:
            cli.set_time(h, m, s)
            cap = cli.capture()
            w, h_img, rgb, depth = cap
            tstr = f"{h:02d}-{m:02d}-{s:02d}"
            rgb_path = out_dir / f"RGB_{weather}_{tstr}.png"
            depth_path = out_dir / f"Depth_{weather}_{tstr}.png"
            save_rgb_png(str(rgb_path), int(w), int(h_img), rgb)
            save_depth_png(str(depth_path), int(w), int(h_img), depth)
            print(f"已保存: {rgb_path.name}, {depth_path.name}")
            time.sleep(10)

    cli.stop_camera()
    print("演示完成")

if __name__ == "__main__":
    main()
