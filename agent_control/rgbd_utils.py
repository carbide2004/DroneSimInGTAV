import numpy as np
from PIL import Image


def rgb_bytes_to_pil(w, h, rgb_bytes):
    w = int(w)
    h = int(h)
    raw = np.frombuffer(rgb_bytes, dtype=np.uint8)
    expected_rgba = w * h * 4
    expected_rgb = w * h * 3
    
    if raw.size == expected_rgba:
        # 完整匹配 RGBA 格式
        arr = raw.reshape((h, w, 4))[:, :, :3]
    elif raw.size == expected_rgb:
        # 完整匹配 RGB 格式
        arr = raw.reshape((h, w, 3))
    elif abs(raw.size - expected_rgba) <= 4:
        # 接近 RGBA 格式（允许最多 4 字节对齐差异）
        if raw.size < expected_rgba:
            # 缺少字节时用零填充
            padded = np.pad(raw, (0, expected_rgba - raw.size), mode='constant', constant_values=0)
            arr = padded.reshape((h, w, 4))[:, :, :3]
        else:
            # 字节过多时截断
            truncated = raw[:expected_rgba]
            arr = truncated.reshape((h, w, 4))[:, :, :3]
    elif abs(raw.size - expected_rgb) <= 4:
        # 接近 RGB 格式（允许最多 4 字节对齐差异）
        if raw.size < expected_rgb:
            # 缺少字节时用零填充
            padded = np.pad(raw, (0, expected_rgb - raw.size), mode='constant', constant_values=0)
            arr = padded.reshape((h, w, 3))
        else:
            # 字节过多时截断
            truncated = raw[:expected_rgb]
            arr = truncated.reshape((h, w, 3))
    else:
        raise ValueError(f"Unexpected rgb bytes size: {raw.size} for {w}x{h} (expected {expected_rgb} or {expected_rgba})")
    
    return Image.fromarray(arr, mode="RGB")


def depth_bytes_to_pil(w, h, depth_bytes):
    w = int(w)
    h = int(h)
    expected_float32 = w * h * 4  # float32 = 4 bytes per pixel
    expected_uint8 = w * h        # uint8 = 1 byte per pixel
    
    # 优先尝试 float32 格式
    if len(depth_bytes) >= expected_float32 - 4:  # Allow small tolerance
        try:
            # 填充或截断到精确大小
            if len(depth_bytes) < expected_float32:
                padded_bytes = depth_bytes + b'\x00' * (expected_float32 - len(depth_bytes))
            else:
                padded_bytes = depth_bytes[:expected_float32]
            
            raw = np.frombuffer(padded_bytes, dtype=np.float32)
            depth = raw.reshape((h, w))
        except Exception:
            # float32 失败时回退到 uint8
            raw_u8 = np.frombuffer(depth_bytes[:expected_uint8], dtype=np.uint8)
            depth = raw_u8.reshape((h, w)).astype(np.float32)
    else:
        # 尝试 uint8 格式
        if len(depth_bytes) >= expected_uint8:
            raw_u8 = np.frombuffer(depth_bytes[:expected_uint8], dtype=np.uint8)
            depth = raw_u8.reshape((h, w)).astype(np.float32)
        else:
            raise ValueError(f"Unexpected depth bytes size: {len(depth_bytes)} for {w}x{h} (expected {expected_uint8} or {expected_float32})")

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
