"""Detect RGB/Depth view lag by alternating between two distant camera yaws.

The test keeps all payloads in memory and writes no RGB, depth, or point-cloud
files.
"""

import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import DroneSimClient


def _angle_error_degrees(actual, expected):
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def _wait_for_posture(client, target, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pose = client.get_pose()
        if pose is None:
            raise RuntimeError("GET_POSE returned no camera pose")
        position_error = np.linalg.norm(
            np.asarray(pose[:3], dtype=np.float64)
            - np.asarray(target[:3], dtype=np.float64)
        )
        rotation_error = max(
            _angle_error_degrees(pose[3], target[3]),
            _angle_error_degrees(pose[4], target[4]),
            _angle_error_degrees(pose[5], target[5]),
        )
        if position_error <= 1.0e-3 and rotation_error <= 1.0e-2:
            return pose
        time.sleep(0.01)
    raise TimeoutError(f"Camera did not reach requested posture {target}")


def _sample_grid(array, rows=24, columns=40):
    y = np.linspace(0, array.shape[0] - 1, rows).astype(np.int32)
    x = np.linspace(0, array.shape[1] - 1, columns).astype(np.int32)
    return array[np.ix_(y, x)]


def _standardize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    standard_deviation = float(np.std(values))
    if not math.isfinite(standard_deviation) or standard_deviation < 1.0e-8:
        raise RuntimeError("View descriptor has no usable spatial variation")
    result = (values - float(np.mean(values))) / standard_deviation
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise RuntimeError("View descriptor normalization failed")
    return result / norm


def _rgb_descriptor(rgb):
    luminance = (
        0.2126 * rgb[:, :, 0].astype(np.float64)
        + 0.7152 * rgb[:, :, 1].astype(np.float64)
        + 0.0722 * rgb[:, :, 2].astype(np.float64)
    )
    return _standardize(_sample_grid(luminance))


def _depth_descriptor(depth, max_view_depth):
    clipped = np.minimum(depth.astype(np.float64), max_view_depth)
    inverse_depth = 1.0 / np.maximum(clipped, 1.0e-3)
    return _standardize(_sample_grid(inverse_depth))


def _cosine_similarity(left, right):
    return float(np.dot(left, right))


def _classify(descriptor, references):
    similarity_a = _cosine_similarity(descriptor, references["A"])
    similarity_b = _cosine_similarity(descriptor, references["B"])
    label = "A" if similarity_a > similarity_b else "B"
    margin = abs(similarity_a - similarity_b)
    return label, margin, similarity_a, similarity_b


def _gradient_magnitude(values):
    values = np.asarray(values, dtype=np.float64)
    gradient_y, gradient_x = np.gradient(values)
    return np.hypot(gradient_x, gradient_y)


def _top_edge_mask(magnitude, percentile):
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        raise RuntimeError("Edge map contains no finite values")
    threshold = float(np.percentile(finite, percentile))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError("Edge map has no usable gradients")
    return magnitude >= threshold


def _dilate(mask, radius=1):
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            output |= padded[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
    return output


def _edge_masks(rgb, depth, max_view_depth, stride):
    rgb_sample = rgb[::stride, ::stride]
    depth_sample = depth[::stride, ::stride].astype(np.float64)
    luminance = (
        0.2126 * rgb_sample[:, :, 0].astype(np.float64)
        + 0.7152 * rgb_sample[:, :, 1].astype(np.float64)
        + 0.0722 * rgb_sample[:, :, 2].astype(np.float64)
    )
    clipped_depth = np.minimum(depth_sample, max_view_depth)
    rgb_edges = _top_edge_mask(_gradient_magnitude(luminance), 85.0)
    depth_edges = _top_edge_mask(
        _gradient_magnitude(np.log1p(clipped_depth)),
        90.0,
    )
    return _dilate(rgb_edges), depth_edges


def _edge_alignment(rgb_edges, depth_edges):
    count = int(np.count_nonzero(depth_edges))
    if count == 0:
        raise RuntimeError("Depth edge map is empty")
    return float(np.count_nonzero(rgb_edges & depth_edges)) / count


def _frame_summary(frame, max_view_depth, edge_stride):
    rgb = frame.rgb_array()
    depth = frame.depth_array()
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(
            f"RGB shape {rgb.shape} and depth shape {depth.shape} differ"
        )
    rgb_edges, depth_edges = _edge_masks(
        rgb,
        depth,
        max_view_depth,
        edge_stride,
    )
    return {
        "frame_id": frame.frame_id,
        "rgb_descriptor": _rgb_descriptor(rgb),
        "depth_descriptor": _depth_descriptor(depth, max_view_depth),
        "rgb_edges": rgb_edges,
        "depth_edges": depth_edges,
    }


def _set_posture(client, target, timeout_seconds):
    client.set_posture(*target)
    _wait_for_posture(client, target, timeout_seconds)


def _build_reference(
    client,
    target,
    capture_timeout_ms,
    posture_timeout,
    max_view_depth,
    edge_stride,
    reference_captures,
):
    _set_posture(client, target, posture_timeout)
    summaries = []
    for _ in range(reference_captures):
        frame = client.capture(capture_timeout_ms)
        summaries.append(
            _frame_summary(frame, max_view_depth, edge_stride)
        )
    return {
        "frame_id": summaries[-1]["frame_id"],
        "rgb_descriptor": _standardize(
            np.mean(
                [summary["rgb_descriptor"] for summary in summaries],
                axis=0,
            )
        ),
        "depth_descriptor": _standardize(
            np.mean(
                [summary["depth_descriptor"] for summary in summaries],
                axis=0,
            )
        ),
        "rgb_edges": summaries[-1]["rgb_edges"],
        "depth_edges": summaries[-1]["depth_edges"],
    }


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
    parser.add_argument("--posture-timeout", type=float, default=3.0)
    parser.add_argument("--max-view-depth", type=float, default=200.0)
    parser.add_argument("--edge-stride", type=int, default=4)
    parser.add_argument("--min-classification-margin", type=float, default=0.10)
    parser.add_argument("--min-alignment-margin", type=float, default=0.03)
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
    if args.min_alignment_margin < 0.0:
        raise ValueError("--min-alignment-margin must not be negative")

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

    try:
        reference_a = _build_reference(
            client,
            targets["A"],
            args.capture_timeout_ms,
            args.posture_timeout,
            args.max_view_depth,
            args.edge_stride,
            args.reference_captures,
        )
        reference_b = _build_reference(
            client,
            targets["B"],
            args.capture_timeout_ms,
            args.posture_timeout,
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

        rgb_reference_separation = 1.0 - _cosine_similarity(
            references["rgb"]["A"],
            references["rgb"]["B"],
        )
        depth_reference_separation = 1.0 - _cosine_similarity(
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
            _set_posture(client, targets[expected], args.posture_timeout)
            frame = client.capture(args.capture_timeout_ms)
            if frame.frame_id <= previous_frame_id:
                raise RuntimeError(
                    f"frame_id did not increase: {frame.frame_id} "
                    f"after {previous_frame_id}"
                )
            current = _frame_summary(
                frame,
                args.max_view_depth,
                args.edge_stride,
            )

            rgb_label, rgb_margin, rgb_a, rgb_b = _classify(
                current["rgb_descriptor"],
                references["rgb"],
            )
            depth_label, depth_margin, depth_a, depth_b = _classify(
                current["depth_descriptor"],
                references["depth"],
            )
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

            same_alignment = _edge_alignment(
                current["rgb_edges"],
                current["depth_edges"],
            )
            rgb_lag_alignment = _edge_alignment(
                previous["rgb_edges"],
                current["depth_edges"],
            )
            depth_lag_alignment = _edge_alignment(
                current["rgb_edges"],
                previous["depth_edges"],
            )
            edge_margin = same_alignment - max(
                rgb_lag_alignment,
                depth_lag_alignment,
            )
            if edge_margin < args.min_alignment_margin:
                raise RuntimeError(
                    f"frame {frame.frame_id}: RGB/Depth edge alignment is not "
                    f"better than one-view-lag alternatives by the required "
                    f"margin: same={same_alignment:.3f}, "
                    f"rgb_lag={rgb_lag_alignment:.3f}, "
                    f"depth_lag={depth_lag_alignment:.3f}, "
                    f"margin={edge_margin:.3f}"
                )

            minimum_rgb_margin = min(minimum_rgb_margin, rgb_margin)
            minimum_depth_margin = min(minimum_depth_margin, depth_margin)
            minimum_edge_margin = min(minimum_edge_margin, edge_margin)
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
    finally:
        client.set_posture(*original_pose)
        _wait_for_posture(client, original_pose, args.posture_timeout)

    print(
        "PASS "
        f"captures={len(tested_frames)} "
        f"frames={tested_frames[0]}..{tested_frames[-1]} "
        f"rgb_margin_min={minimum_rgb_margin:.3f} "
        f"depth_margin_min={minimum_depth_margin:.3f} "
        f"edge_margin_min={minimum_edge_margin:.3f} "
        f"digest={digest.hexdigest()}"
    )
    print("No RGB-D payload was written to disk.")


if __name__ == "__main__":
    main()
