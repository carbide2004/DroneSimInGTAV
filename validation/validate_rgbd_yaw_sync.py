"""Detect RGB/Depth view lag by alternating between two distant camera yaws.

The test keeps all payloads in memory and writes no RGB, depth, or point-cloud
files.
"""

import argparse
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import DroneSimClient
from validation.rgbd_sync_metrics import (  # noqa: E402
    aggregate_reference,
    classify,
    cosine_similarity,
    edge_alignment,
    frame_summary,
)


def _set_pose(client, target):
    client.set_camera_pose(
        target[0],
        target[1],
        target[2],
        target[5],
        collision_check=False,
    )
def _build_reference(
    client,
    target,
    capture_timeout_ms,
    max_view_depth,
    edge_stride,
    reference_captures,
):
    _set_pose(client, target)
    summaries = []
    for _ in range(reference_captures):
        frame = client.capture(capture_timeout_ms)
        summaries.append(
            frame_summary(frame, max_view_depth, edge_stride)
        )
    return aggregate_reference(summaries)


def main():
    parser = argparse.ArgumentParser(
        description="Alternate between two camera yaws and detect RGB/Depth "
        "one-frame lag without saving any payload."
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--yaw-delta", type=float, default=180.0)
    parser.add_argument("--reference-captures", type=int, default=3)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-view-depth", type=float, default=200.0)
    parser.add_argument("--edge-stride", type=int, default=4)
    parser.add_argument("--min-classification-margin", type=float, default=0.10)
    args = parser.parse_args()

    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")
    if args.reference_captures < 2:
        raise ValueError("--reference-captures must be at least 2")
    if (
        not math.isfinite(args.yaw_delta)
        or not 90.0 <= abs(args.yaw_delta) <= 270.0
    ):
        raise ValueError("--yaw-delta magnitude must be in [90, 270] degrees")
    if not math.isfinite(args.max_view_depth) or args.max_view_depth <= 0.0:
        raise ValueError("--max-view-depth must be positive and finite")
    if args.edge_stride <= 0:
        raise ValueError("--edge-stride must be positive")
    if args.min_classification_margin < 0.0:
        raise ValueError("--min-classification-margin must not be negative")

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    if original_pose is None:
        raise RuntimeError("No active GTA V rendering camera")

    x, y, z, pitch, roll, initial_yaw = original_pose
    targets = {
        "A": (x, y, z, pitch, roll, initial_yaw),
        "B": (x, y, z, pitch, roll, initial_yaw + args.yaw_delta),
    }

    digest = hashlib.blake2b(digest_size=16)
    tested_frames = []
    minimum_rgb_margin = float("inf")
    minimum_depth_margin = float("inf")
    minimum_edge_margin = float("inf")
    edge_margins = []

    try:
        reference_a = _build_reference(
            client,
            targets["A"],
            args.capture_timeout_ms,
            args.max_view_depth,
            args.edge_stride,
            args.reference_captures,
        )
        reference_b = _build_reference(
            client,
            targets["B"],
            args.capture_timeout_ms,
            args.max_view_depth,
            args.edge_stride,
            args.reference_captures,
        )
        references = {
            "rgb": {
                "A": reference_a["rgb_descriptor"],
                "B": reference_b["rgb_descriptor"],
            },
            "depth": {
                "A": reference_a["depth_descriptor"],
                "B": reference_b["depth_descriptor"],
            },
        }

        rgb_reference_separation = 1.0 - cosine_similarity(
            references["rgb"]["A"],
            references["rgb"]["B"],
        )
        depth_reference_separation = 1.0 - cosine_similarity(
            references["depth"]["A"],
            references["depth"]["B"],
        )
        if rgb_reference_separation < args.min_classification_margin:
            raise RuntimeError(
                "The two yaws are not visually distinguishable enough in RGB: "
                f"separation={rgb_reference_separation:.3f}"
            )
        if depth_reference_separation < args.min_classification_margin:
            raise RuntimeError(
                "The two yaws are not geometrically distinguishable enough: "
                f"separation={depth_reference_separation:.3f}"
            )

        previous = reference_b
        previous_frame_id = reference_b["frame_id"]
        total = args.cycles * 2
        for index in range(total):
            expected = "A" if index % 2 == 0 else "B"
            _set_pose(client, targets[expected])
            frame = client.capture(args.capture_timeout_ms)
            if frame.frame_id <= previous_frame_id:
                raise RuntimeError(
                    f"frame_id did not increase: {frame.frame_id} "
                    f"after {previous_frame_id}"
                )
            current = frame_summary(
                frame,
                args.max_view_depth,
                args.edge_stride,
            )

            rgb_label, rgb_margin, rgb_similarities = classify(
                current["rgb_descriptor"],
                references["rgb"],
            )
            depth_label, depth_margin, depth_similarities = classify(
                current["depth_descriptor"],
                references["depth"],
            )
            rgb_a = rgb_similarities["A"]
            rgb_b = rgb_similarities["B"]
            depth_a = depth_similarities["A"]
            depth_b = depth_similarities["B"]
            if (
                rgb_label != expected
                or rgb_margin < args.min_classification_margin
            ):
                raise RuntimeError(
                    f"frame {frame.frame_id}: RGB classified as {rgb_label}, "
                    f"expected {expected}, margin={rgb_margin:.3f}, "
                    f"similarity(A/B)={rgb_a:.3f}/{rgb_b:.3f}"
                )
            if (
                depth_label != expected
                or depth_margin < args.min_classification_margin
            ):
                raise RuntimeError(
                    f"frame {frame.frame_id}: Depth classified as {depth_label}, "
                    f"expected {expected}, margin={depth_margin:.3f}, "
                    f"similarity(A/B)={depth_a:.3f}/{depth_b:.3f}"
                )

            same_alignment = edge_alignment(
                current["rgb_edges"],
                current["depth_edges"],
            )
            rgb_lag_alignment = edge_alignment(
                previous["rgb_edges"],
                current["depth_edges"],
            )
            depth_lag_alignment = edge_alignment(
                current["rgb_edges"],
                previous["depth_edges"],
            )
            edge_margin = same_alignment - max(
                rgb_lag_alignment,
                depth_lag_alignment,
            )

            minimum_rgb_margin = min(minimum_rgb_margin, rgb_margin)
            minimum_depth_margin = min(minimum_depth_margin, depth_margin)
            minimum_edge_margin = min(minimum_edge_margin, edge_margin)
            edge_margins.append(edge_margin)
            tested_frames.append(frame.frame_id)
            digest.update(frame.frame_id.to_bytes(8, "little"))
            digest.update(expected.encode("ascii"))
            digest.update(
                np.asarray(
                    [
                        rgb_a,
                        rgb_b,
                        depth_a,
                        depth_b,
                        same_alignment,
                        rgb_lag_alignment,
                        depth_lag_alignment,
                    ],
                    dtype="<f8",
                ).tobytes()
            )
            print(
                f"{index + 1}/{total} frame={frame.frame_id} yaw={expected} "
                f"rgb={rgb_label}:{rgb_margin:.3f} "
                f"depth={depth_label}:{depth_margin:.3f} "
                f"edges={same_alignment:.3f}/"
                f"{rgb_lag_alignment:.3f}/"
                f"{depth_lag_alignment:.3f}"
            )
            previous = current
            previous_frame_id = frame.frame_id
        median_edge_margin = float(np.median(edge_margins))
    finally:
        client.set_camera_pose(
            original_pose[0],
            original_pose[1],
            original_pose[2],
            original_pose[5],
            collision_check=False,
        )

    print(
        "PASS "
        f"captures={len(tested_frames)} "
        f"frames={tested_frames[0]}..{tested_frames[-1]} "
        f"rgb_margin_min={minimum_rgb_margin:.3f} "
        f"depth_margin_min={minimum_depth_margin:.3f} "
        f"edge_diagnostic_min={minimum_edge_margin:.3f} "
        f"edge_diagnostic_median={median_edge_margin:.3f} "
        f"digest={digest.hexdigest()}"
    )
    print("No RGB-D payload was written to disk.")


if __name__ == "__main__":
    main()
