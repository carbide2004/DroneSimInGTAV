"""Pure NumPy metrics for detecting RGB-D view and one-frame lag."""

import math

import numpy as np


def _sample_grid(array, rows=24, columns=40):
    y = np.linspace(0, array.shape[0] - 1, rows).astype(np.int32)
    x = np.linspace(0, array.shape[1] - 1, columns).astype(np.int32)
    return array[np.ix_(y, x)]


def standardize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    standard_deviation = float(np.std(values))
    if (
        not math.isfinite(standard_deviation)
        or standard_deviation < 1.0e-8
    ):
        raise RuntimeError(
            "View descriptor has no usable spatial variation"
        )
    result = (
        values - float(np.mean(values))
    ) / standard_deviation
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
    return standardize(_sample_grid(luminance))


def _depth_descriptor(depth, max_view_depth):
    clipped = np.minimum(
        depth.astype(np.float64),
        max_view_depth,
    )
    inverse_depth = 1.0 / np.maximum(clipped, 1.0e-3)
    return standardize(_sample_grid(inverse_depth))


def cosine_similarity(left, right):
    return float(np.dot(left, right))


def classify(descriptor, references):
    if len(references) != 2:
        raise ValueError(
            "View classification requires exactly two references"
        )
    similarities = {
        label: cosine_similarity(descriptor, reference)
        for label, reference in references.items()
    }
    ordered = sorted(
        similarities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    margin = ordered[0][1] - ordered[1][1]
    return ordered[0][0], margin, similarities


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
    padded = np.pad(
        mask,
        radius,
        mode="constant",
        constant_values=False,
    )
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
    depth_sample = depth[::stride, ::stride].astype(
        np.float64
    )
    luminance = (
        0.2126 * rgb_sample[:, :, 0].astype(np.float64)
        + 0.7152 * rgb_sample[:, :, 1].astype(np.float64)
        + 0.0722 * rgb_sample[:, :, 2].astype(np.float64)
    )
    clipped_depth = np.minimum(
        depth_sample,
        max_view_depth,
    )
    rgb_edges = _top_edge_mask(
        _gradient_magnitude(luminance),
        85.0,
    )
    depth_edges = _top_edge_mask(
        _gradient_magnitude(np.log1p(clipped_depth)),
        90.0,
    )
    return _dilate(rgb_edges), depth_edges


def edge_alignment(rgb_edges, depth_edges):
    count = int(np.count_nonzero(depth_edges))
    if count == 0:
        raise RuntimeError("Depth edge map is empty")
    return (
        float(np.count_nonzero(rgb_edges & depth_edges))
        / count
    )


def frame_summary(frame, max_view_depth, edge_stride):
    rgb = frame.rgb_array()
    depth = frame.depth_array()
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(
            f"RGB shape {rgb.shape} and depth shape "
            f"{depth.shape} differ"
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
        "depth_descriptor": _depth_descriptor(
            depth,
            max_view_depth,
        ),
        "rgb_edges": rgb_edges,
        "depth_edges": depth_edges,
    }


def aggregate_reference(summaries):
    if not summaries:
        raise ValueError(
            "Reference aggregation requires at least one frame"
        )
    return {
        "frame_id": summaries[-1]["frame_id"],
        "rgb_descriptor": standardize(
            np.mean(
                [
                    summary["rgb_descriptor"]
                    for summary in summaries
                ],
                axis=0,
            )
        ),
        "depth_descriptor": standardize(
            np.mean(
                [
                    summary["depth_descriptor"]
                    for summary in summaries
                ],
                axis=0,
            )
        ),
        "rgb_edges": summaries[-1]["rgb_edges"],
        "depth_edges": summaries[-1]["depth_edges"],
    }
