"""Run repeated live RGB-D captures without saving any image or depth payload."""

import argparse
import hashlib
import time

import numpy as np
import psutil

from dronesim_client import DroneSimClient


def _find_process(name):
    matches = []
    for process in psutil.process_iter(("name", "pid")):
        if (process.info["name"] or "").lower() == name.lower():
            matches.append(process)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} process, found {len(matches)}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Capture RGB-D repeatedly in memory and check frame, "
        "payload, depth, latency, and GTA process memory invariants."
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--process-name", default="GTA5.exe")
    parser.add_argument("--max-memory-growth-mb", type=float, default=512.0)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.max_memory_growth_mb < 0:
        raise ValueError("--max-memory-growth-mb must not be negative")

    gta_process = _find_process(args.process_name)
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    initial_rss = gta_process.memory_info().rss
    peak_rss = initial_rss
    previous_frame_id = None
    latencies = []
    digest = hashlib.blake2b(digest_size=16)
    global_depth_min = float("inf")
    global_depth_max = 0.0

    for index in range(args.count):
        started = time.perf_counter()
        frame = client.capture(args.capture_timeout_ms)
        latency = time.perf_counter() - started
        if previous_frame_id is not None and frame.frame_id <= previous_frame_id:
            raise RuntimeError(
                f"frame_id did not increase: {frame.frame_id} "
                f"after {previous_frame_id}"
            )
        previous_frame_id = frame.frame_id

        rgb = frame.rgb_array()
        depth = frame.depth_array()
        if rgb.shape[:2] != depth.shape:
            raise RuntimeError(
                f"RGB shape {rgb.shape} and depth shape {depth.shape} differ"
            )
        depth_min = float(np.min(depth))
        depth_max = float(np.max(depth))
        global_depth_min = min(global_depth_min, depth_min)
        global_depth_max = max(global_depth_max, depth_max)

        digest.update(frame.frame_id.to_bytes(8, "little"))
        digest.update(memoryview(frame.rgb))
        digest.update(memoryview(frame.depth))
        latencies.append(latency)
        current_rss = gta_process.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

        if (index + 1) % 50 == 0 or index + 1 == args.count:
            print(
                f"{index + 1}/{args.count} frame={frame.frame_id} "
                f"latency={latency * 1000.0:.1f} ms "
                f"depth=[{depth_min:.3f}, {depth_max:.3f}] m "
                f"gta_rss={current_rss / (1024 ** 2):.1f} MiB"
            )

    final_rss = gta_process.memory_info().rss
    growth_mb = (final_rss - initial_rss) / (1024 ** 2)
    if growth_mb > args.max_memory_growth_mb:
        raise RuntimeError(
            f"GTA process memory grew by {growth_mb:.1f} MiB; "
            f"limit is {args.max_memory_growth_mb:.1f} MiB"
        )

    latency_ms = np.asarray(latencies, dtype=np.float64) * 1000.0
    print(
        "PASS "
        f"count={args.count} "
        f"latency_p50={np.percentile(latency_ms, 50):.1f}ms "
        f"latency_p95={np.percentile(latency_ms, 95):.1f}ms "
        f"latency_p99={np.percentile(latency_ms, 99):.1f}ms "
        f"depth=[{global_depth_min:.3f}, {global_depth_max:.3f}]m "
        f"memory_growth={growth_mb:.1f}MiB "
        f"peak_growth={(peak_rss - initial_rss) / (1024 ** 2):.1f}MiB "
        f"digest={digest.hexdigest()}"
    )
    print("No RGB-D payload was written to disk.")


if __name__ == "__main__":
    main()
