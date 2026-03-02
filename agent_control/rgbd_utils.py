import numpy as np
from PIL import Image


def rgb_bytes_to_pil(w, h, rgb_bytes):
    w = int(w)
    h = int(h)
    raw = np.frombuffer(rgb_bytes, dtype=np.uint8)
    if raw.size == w * h * 4:
        arr = raw.reshape((h, w, 4))[:, :, :3]
    elif raw.size == w * h * 3:
        arr = raw.reshape((h, w, 3))
    else:
        raise ValueError(f"Unexpected rgb bytes size: {raw.size} for {w}x{h}")
    return Image.fromarray(arr, mode="RGB")


def depth_bytes_to_pil(w, h, depth_bytes):
    w = int(w)
    h = int(h)
    raw = np.frombuffer(depth_bytes, dtype=np.float32)
    if raw.size != w * h:
        raw_u8 = np.frombuffer(depth_bytes, dtype=np.uint8)
        if raw_u8.size == w * h:
            depth = raw_u8.reshape((h, w)).astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected depth bytes size: {len(depth_bytes)} for {w}x{h}"
            )
    else:
        depth = raw.reshape((h, w))

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    try:
        import cv2

        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray(depth_color, mode="RGB")
    except Exception:
        dmin = float(np.min(depth))
        dmax = float(np.max(depth))
        if dmax <= dmin:
            gray = np.zeros((h, w), dtype=np.uint8)
        else:
            gray = ((depth - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(
                np.uint8
            )
        rgb = np.stack([gray, gray, gray], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

