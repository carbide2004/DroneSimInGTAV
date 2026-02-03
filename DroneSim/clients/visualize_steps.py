import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def _load_steps_jsonl(session_dir):
    path = os.path.join(session_dir, "steps.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    steps = []
    for i, ln in enumerate(lines):
        try:
            steps.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON at line {i+1} in {path}: {e}") from e
    return steps


def _safe_join(session_dir, rel_path):
    rel_path = rel_path.replace("/", os.sep).replace("\\", os.sep)
    return os.path.join(session_dir, rel_path)


def _load_rgb_image(step, session_dir):
    rgb_meta = step.get("rgb") or {}
    w = int(rgb_meta.get("width") or 0)
    h = int(rgb_meta.get("height") or 0)
    p = rgb_meta.get("path") or ""
    if not p or w <= 0 or h <= 0:
        return None
    full = _safe_join(session_dir, p)
    with open(full, "rb") as f:
        b = f.read()
    arr = np.frombuffer(b, dtype=np.uint8)
    expected = w * h * 4
    if arr.size < expected:
        raise RuntimeError(f"RGB buffer too small: got {arr.size} bytes, expected {expected} ({os.path.basename(full)})")
    arr = arr[:expected].reshape((h, w, 4))
    rgb = arr[:, :, :3]
    return Image.fromarray(rgb)


def _format_action(action_obj):
    if not action_obj:
        return "N/A"
    name = action_obj.get("name", "N/A")
    dx = action_obj.get("dx", 0.0)
    dy = action_obj.get("dy", 0.0)
    dz = action_obj.get("dz", 0.0)
    drx = action_obj.get("drx", 0.0)
    dry = action_obj.get("dry", 0.0)
    drz = action_obj.get("drz", 0.0)
    return f"{name}  d=({dx:.2f},{dy:.2f},{dz:.2f})  dr=({drx:.1f},{dry:.1f},{drz:.1f})"


def _overlay_text(img, step_idx, total, next_action_text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    lines = [
        f"step: {step_idx}/{max(total - 1, 0)}",
        f"next: {next_action_text}",
    ]
    pad = 16
    x0, y0 = pad, pad
    box_w = 0
    box_h = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        box_w = max(box_w, bbox[2] - bbox[0])
        box_h += (bbox[3] - bbox[1]) + 2
    box_w += pad * 2
    box_h += pad * 2
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 160))
    y = y0 + pad
    for ln in lines:
        draw.text((x0 + pad, y), ln, fill=(255, 255, 255), font=font)
        bbox = draw.textbbox((0, 0), ln, font=font)
        y += (bbox[3] - bbox[1]) + 2
    return img


def main():
    session_dir = r"D:\SteamLibrary\steamapps\common\Grand Theft Auto V\data\manual\20260129_203926"
    fps = 1.0
    target_w = 1920
    target_h = 1080
    session_dir = os.path.abspath(session_dir)
    steps = _load_steps_jsonl(session_dir)
    if not steps:
        raise RuntimeError("No steps found")
    print(f"Loaded {len(steps)} steps from {session_dir}", flush=True)

    plt.ion()
    fig, ax = plt.subplots(figsize=(target_w / 100.0, target_h / 100.0), dpi=100)
    ax.axis("off")
    im_artist = None

    delay = 1.0 / max(fps, 1e-6)
    for i in range(len(steps)):
        cur = steps[i]
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        next_action_text = _format_action((cur or {}).get("action"))

        rgb_meta = cur.get("rgb") or {}
        rgb_rel = rgb_meta.get("path") or ""
        w = int(rgb_meta.get("width") or 0)
        h = int(rgb_meta.get("height") or 0)
        if rgb_rel:
            rgb_full = _safe_join(session_dir, rgb_rel)
            size_on_disk = os.path.getsize(rgb_full) if os.path.exists(rgb_full) else -1
            expected_bytes = w * h * 4
            print(f"[step {i}] file={rgb_full} size={size_on_disk} expected={expected_bytes} (w={w},h={h})", flush=True)
        else:
            print(f"[step {i}] missing rgb path in meta", flush=True)

        img = _load_rgb_image(cur, session_dir)
        if img is None:
            print(f"[step {i}] _load_rgb_image returned None, skipping", flush=True)
            continue
        img = img.convert("RGBA")
        img = _overlay_text(img, int(cur.get("step", i)), len(steps), next_action_text)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        frame = np.asarray(img.convert("RGB"))

        if im_artist is None:
            im_artist = ax.imshow(frame)
        else:
            im_artist.set_data(frame)
        print(f"[step {i}] frame shape={frame.shape} dtype={frame.dtype}", flush=True)
        fig.canvas.draw_idle()
        plt.pause(delay)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
