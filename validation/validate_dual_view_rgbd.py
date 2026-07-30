"""Validate lockstep oblique+nadir RGB-D pairs without saving payloads."""

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    LockstepSession,
    NADIR_PITCH_DEGREES,
    OBLIQUE_PITCH_DEGREES,
    ScenarioLifecycle,
)
from validation.rgbd_geometry import (  # noqa: E402
    frame_to_world_points,
)
from validation.rgbd_sync_metrics import (  # noqa: E402
    aggregate_reference,
    classify,
    cosine_similarity,
    edge_alignment,
    frame_summary,
)


def _find_process(name):
    matches = [
        process
        for process in psutil.process_iter(("name", "pid"))
        if (process.info["name"] or "").lower() == name.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} process, "
            f"found {len(matches)}"
        )
    return matches[0]


def _angle_error(actual, expected):
    return abs(
        (float(actual) - float(expected) + 180.0)
        % 360.0
        - 180.0
    )


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError("View matrix contains a zero-length axis")
    return vector / norm


def _view_geometry(frame):
    view = np.asarray(
        frame.view_matrix,
        dtype=np.float64,
    ).reshape(4, 4)
    if not np.isfinite(view).all():
        raise RuntimeError(
            f"Frame {frame.frame_id} view matrix is not finite"
        )
    inverse_view = np.linalg.inv(view)
    if not np.isfinite(inverse_view).all():
        raise RuntimeError(
            f"Frame {frame.frame_id} inverse view is not finite"
        )
    return {
        "center": inverse_view[:3, 3],
        "right": _normalize(view[0, :3]),
        "up": _normalize(view[1, :3]),
        "forward": _normalize(-view[2, :3]),
    }


def _assert_pair_geometry(pair, canonical_pose):
    oblique = _view_geometry(pair.oblique)
    nadir = _view_geometry(pair.nadir)
    expected_center = np.asarray(
        canonical_pose[:3],
        dtype=np.float64,
    )
    for label, geometry in (
        ("oblique", oblique),
        ("nadir", nadir),
    ):
        center_error = float(
            np.linalg.norm(
                geometry["center"] - expected_center
            )
        )
        if center_error > 1.0e-3:
            raise RuntimeError(
                f"{label} camera center error is "
                f"{center_error:.6f}m"
            )
    pair_center_error = float(
        np.linalg.norm(
            oblique["center"] - nadir["center"]
        )
    )
    if pair_center_error > 1.0e-3:
        raise RuntimeError(
            "Dual-view camera centers differ by "
            f"{pair_center_error:.6f}m"
        )

    yaw = math.radians(float(canonical_pose[5]))
    body_forward = np.asarray(
        (-math.sin(yaw), math.cos(yaw), 0.0),
        dtype=np.float64,
    )
    body_right = np.asarray(
        (math.cos(yaw), math.sin(yaw), 0.0),
        dtype=np.float64,
    )
    expected_oblique_forward = _normalize(
        body_forward / math.sqrt(2.0)
        + np.asarray((0.0, 0.0, -1.0))
        / math.sqrt(2.0)
    )
    expected_nadir_forward = np.asarray(
        (0.0, 0.0, -1.0),
        dtype=np.float64,
    )
    checks = (
        (
            "oblique forward",
            oblique["forward"],
            expected_oblique_forward,
        ),
        (
            "nadir forward",
            nadir["forward"],
            expected_nadir_forward,
        ),
        ("nadir up", nadir["up"], body_forward),
        ("nadir right", nadir["right"], body_right),
    )
    for label, actual, expected in checks:
        error = float(np.linalg.norm(actual - expected))
        if error > 1.0e-5:
            raise RuntimeError(
                f"{label} direction error is {error:.6e}"
            )


def _clock_identity(clock):
    return (
        clock.session_id,
        clock.step_index,
        clock.epoch_game_timer_ms,
        clock.game_timer_ms,
        clock.target_elapsed_ms,
        clock.actual_elapsed_ms,
        clock.last_advance_ms,
        clock.render_frames,
        clock.max_frame_time_ms,
    )


def _assert_scenario_frozen(before, after):
    if before.scenario_id != after.scenario_id:
        raise RuntimeError(
            "Scenario ID changed during dual-view capture"
        )
    scalar_fields = (
        "blueprint_id",
        "seed",
        "lifecycle",
        "game_timer_ms",
        "start_game_timer_ms",
        "start_frame_count",
        "event_position",
        "event_active",
        "failure_message",
    )
    changed = [
        name
        for name in scalar_fields
        if getattr(before, name) != getattr(after, name)
    ]
    if changed:
        raise RuntimeError(
            "Scenario state changed during dual-view capture: "
            + ", ".join(changed)
        )
    before_entities = {
        entity.stable_id: entity
        for entity in before.entities
    }
    after_entities = {
        entity.stable_id: entity
        for entity in after.entities
    }
    if before_entities.keys() != after_entities.keys():
        raise RuntimeError(
            "Scenario registry changed during dual-view capture"
        )
    entity_fields = (
        "gta_handle",
        "model_hash",
        "kind",
        "role",
        "event_id",
        "task_state",
        "exists",
        "position",
        "velocity",
        "speed",
        "heading",
        "task_start_game_timer_ms",
        "response_start_game_timer_ms",
        "task_target",
    )
    for stable_id, first in before_entities.items():
        second = after_entities[stable_id]
        changed = [
            name
            for name in entity_fields
            if getattr(first, name) != getattr(second, name)
        ]
        if changed:
            raise RuntimeError(
                f"Entity {stable_id} changed during dual-view "
                "capture: "
                + ", ".join(changed)
            )


def _capture_reference(
    client,
    pitch,
    args,
):
    client.set_camera_pitch(pitch)
    summaries = [
        frame_summary(
            client.capture(args.capture_timeout_ms),
            args.max_view_depth,
            args.edge_stride,
        )
        for _ in range(args.reference_captures)
    ]
    return aggregate_reference(summaries)


def _validate_pitch_sync(client, session, args):
    before_clock = session.refresh()
    try:
        reference_oblique = _capture_reference(
            client,
            OBLIQUE_PITCH_DEGREES,
            args,
        )
        reference_nadir = _capture_reference(
            client,
            NADIR_PITCH_DEGREES,
            args,
        )
        references = {
            "rgb": {
                "oblique": reference_oblique[
                    "rgb_descriptor"
                ],
                "nadir": reference_nadir["rgb_descriptor"],
            },
            "depth": {
                "oblique": reference_oblique[
                    "depth_descriptor"
                ],
                "nadir": reference_nadir[
                    "depth_descriptor"
                ],
            },
        }
        rgb_separation = 1.0 - cosine_similarity(
            references["rgb"]["oblique"],
            references["rgb"]["nadir"],
        )
        depth_separation = 1.0 - cosine_similarity(
            references["depth"]["oblique"],
            references["depth"]["nadir"],
        )
        if rgb_separation < args.min_classification_margin:
            raise RuntimeError(
                "Oblique and nadir RGB references are not "
                "distinguishable enough: "
                f"{rgb_separation:.3f}"
            )
        if depth_separation < args.min_classification_margin:
            raise RuntimeError(
                "Oblique and nadir Depth references are not "
                "distinguishable enough: "
                f"{depth_separation:.3f}"
            )

        previous = reference_nadir
        previous_frame_id = reference_nadir["frame_id"]
        minimum_rgb_margin = float("inf")
        minimum_depth_margin = float("inf")
        minimum_edge_margin = float("inf")
        edge_margins = []
        total = args.sync_cycles * 2
        for index in range(total):
            expected = (
                "oblique" if index % 2 == 0 else "nadir"
            )
            pitch = (
                OBLIQUE_PITCH_DEGREES
                if expected == "oblique"
                else NADIR_PITCH_DEGREES
            )
            client.set_camera_pitch(pitch)
            frame = client.capture(args.capture_timeout_ms)
            if frame.frame_id <= previous_frame_id:
                raise RuntimeError(
                    "Pitch-sync frame ID did not increase: "
                    f"{frame.frame_id}/{previous_frame_id}"
                )
            current = frame_summary(
                frame,
                args.max_view_depth,
                args.edge_stride,
            )
            rgb_label, rgb_margin, rgb_scores = classify(
                current["rgb_descriptor"],
                references["rgb"],
            )
            depth_label, depth_margin, depth_scores = classify(
                current["depth_descriptor"],
                references["depth"],
            )
            if (
                rgb_label != expected
                or rgb_margin < args.min_classification_margin
            ):
                raise RuntimeError(
                    f"Frame {frame.frame_id} RGB classified as "
                    f"{rgb_label}, expected {expected}, "
                    f"margin={rgb_margin:.3f}, "
                    f"scores={rgb_scores}"
                )
            if (
                depth_label != expected
                or depth_margin < args.min_classification_margin
            ):
                raise RuntimeError(
                    f"Frame {frame.frame_id} Depth classified as "
                    f"{depth_label}, expected {expected}, "
                    f"margin={depth_margin:.3f}, "
                    f"scores={depth_scores}"
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
            minimum_rgb_margin = min(
                minimum_rgb_margin,
                rgb_margin,
            )
            minimum_depth_margin = min(
                minimum_depth_margin,
                depth_margin,
            )
            minimum_edge_margin = min(
                minimum_edge_margin,
                edge_margin,
            )
            edge_margins.append(edge_margin)
            previous = current
            previous_frame_id = frame.frame_id
        median_edge_margin = float(
            np.median(edge_margins)
        )
    finally:
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)

    after_clock = session.refresh()
    if _clock_identity(before_clock) != _clock_identity(after_clock):
        raise RuntimeError(
            "Simulation advanced during pitch-sync validation"
        )
    print(
        "pitch sync PASS "
        f"captures={args.sync_cycles * 2} "
        f"rgb_margin_min={minimum_rgb_margin:.3f} "
        f"depth_margin_min={minimum_depth_margin:.3f} "
        f"edge_diagnostic_min={minimum_edge_margin:.3f} "
        f"edge_diagnostic_median={median_edge_margin:.3f}"
    )


def _validate_pair(
    client,
    session,
    scenario_id,
    args,
    previous_frame_id,
):
    before_clock = session.refresh()
    before_scenario = client.get_scenario_state(scenario_id)
    if before_scenario.lifecycle != ScenarioLifecycle.RUNNING:
        raise RuntimeError(
            "Scenario is not RUNNING before dual-view capture: "
            f"{before_scenario.lifecycle.name} "
            f"{before_scenario.failure_message}"
        )
    if not before_scenario.event_active:
        raise RuntimeError(
            "Fire is inactive before dual-view capture"
        )
    canonical_pose = client.get_pose()
    started = time.perf_counter()
    pair = session.capture_rgbd_pair(args.capture_timeout_ms)
    latency_ms = (time.perf_counter() - started) * 1000.0
    after_scenario = client.get_scenario_state(scenario_id)
    after_clock = session.refresh()

    if _clock_identity(before_clock) != _clock_identity(pair.clock):
        raise RuntimeError(
            "Pair clock does not match its pre-capture instant"
        )
    if _clock_identity(pair.clock) != _clock_identity(after_clock):
        raise RuntimeError(
            "Pair clock does not match its post-capture instant"
        )
    _assert_scenario_frozen(
        before_scenario,
        after_scenario,
    )
    _assert_pair_geometry(pair, canonical_pose)

    actual_pose = client.get_pose()
    position_error = float(
        np.linalg.norm(
            np.asarray(actual_pose[:3], dtype=np.float64)
            - np.asarray(canonical_pose[:3], dtype=np.float64)
        )
    )
    if (
        position_error > 1.0e-3
        or abs(
            float(actual_pose[3])
            - OBLIQUE_PITCH_DEGREES
        )
        > 1.0e-2
        or _angle_error(actual_pose[4], 0.0) > 1.0e-2
        or _angle_error(
            actual_pose[5],
            canonical_pose[5],
        )
        > 1.0e-2
    ):
        raise RuntimeError(
            "Camera did not return to the canonical oblique pose"
        )
    if pair.oblique.frame_id <= previous_frame_id:
        raise RuntimeError(
            "Oblique frame ID did not increase globally"
        )
    if pair.nadir.frame_id <= pair.oblique.frame_id:
        raise RuntimeError(
            "Nadir frame ID did not follow oblique"
        )

    depth_minimum = float("inf")
    depth_maximum = 0.0
    for frame in (pair.oblique, pair.nadir):
        depth = frame.depth_array()
        depth_minimum = min(
            depth_minimum,
            float(np.min(depth)),
        )
        depth_maximum = max(
            depth_maximum,
            float(np.max(depth)),
        )
    return (
        pair,
        latency_ms,
        depth_minimum,
        depth_maximum,
    )


def _show_pair_pointcloud(pair, pixel_stride, max_view_depth):
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for --show-pointcloud"
        ) from error

    point_batches = []
    color_batches = []
    for frame in (pair.oblique, pair.nadir):
        points, colors = frame_to_world_points(
            frame,
            pixel_stride,
            max_view_depth,
        )
        point_batches.append(points)
        color_batches.append(colors)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.concatenate(point_batches, axis=0)
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.concatenate(color_batches, axis=0)
    )
    center = _view_geometry(pair.oblique)["center"]
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0,
        origin=center,
    )
    print(
        "displaying "
        f"{len(cloud.points)} in-memory points from frames "
        f"{pair.oblique.frame_id}/{pair.nadir.frame_id}; "
        "nothing will be saved"
    )
    o3d.visualization.draw_geometries(
        [cloud, axes],
        window_name="DroneSim lockstep oblique+nadir RGB-D",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate same-simulation-time oblique and nadir RGB-D "
            "pairs without saving payloads."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--sync-cycles", type=int, default=20)
    parser.add_argument("--reference-captures", type=int, default=3)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-view-depth", type=float, default=200.0)
    parser.add_argument("--edge-stride", type=int, default=4)
    parser.add_argument(
        "--min-classification-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=8)
    parser.add_argument("--prepare-timeout", type=float, default=15.0)
    parser.add_argument("--camera-height", type=float, default=40.0)
    parser.add_argument("--process-name", default="GTA5.exe")
    parser.add_argument(
        "--max-memory-growth-mb",
        type=float,
        default=512.0,
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--show-pointcloud", action="store_true")
    parser.add_argument("--pixel-stride", type=int, default=4)
    args = parser.parse_args()

    positive_integers = {
        "--pairs": args.pairs,
        "--sync-cycles": args.sync_cycles,
        "--reference-captures": args.reference_captures,
        "--edge-stride": args.edge_stride,
        "--progress-interval": args.progress_interval,
        "--pixel-stride": args.pixel_stride,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.reference_captures < 2:
        parser.error("--reference-captures must be at least 2")
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor must contain finite values")
    finite_positive = {
        "--max-view-depth": args.max_view_depth,
        "--prepare-timeout": args.prepare_timeout,
        "--camera-height": args.camera_height,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    if args.min_classification_margin < 0:
        parser.error(
            "--min-classification-margin must not be negative"
        )
    if (
        not math.isfinite(args.max_memory_growth_mb)
        or args.max_memory_growth_mb < 0
    ):
        parser.error(
            "--max-memory-growth-mb must be finite and non-negative"
        )
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be in [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be in [0, 32]")

    gta_process = _find_process(args.process_name)
    python_process = psutil.Process()
    gta_initial_rss = gta_process.memory_info().rss
    python_initial_rss = python_process.memory_info().rss
    gta_peak_rss = gta_initial_rss
    python_peak_rss = python_initial_rss

    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    scenario_id = None
    scenario_reset = False
    session = None
    failure = None
    last_pair = None
    try:
        client.teleport_player(*args.anchor)
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            seed=args.seed,
            firetruck_count=args.firetrucks,
            pedestrian_count=args.pedestrians,
        )
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        yaw = float(original_pose[5])
        yaw_radians = math.radians(yaw)
        observer_x = (
            ready.event_position[0]
            + math.sin(yaw_radians) * args.camera_height
        )
        observer_y = (
            ready.event_position[1]
            - math.cos(yaw_radians) * args.camera_height
        )
        observer_z = (
            ready.event_position[2] + args.camera_height
        )
        client.set_camera_pose(
            observer_x,
            observer_y,
            observer_z,
            yaw,
            collision_check=False,
        )
        canonical_pose = client.set_camera_pitch(
            OBLIQUE_PITCH_DEGREES
        )
        if _angle_error(canonical_pose[4], 0.0) > 1.0e-2:
            raise RuntimeError(
                "Dual-view validation requires zero camera roll"
            )

        session = LockstepSession(client)
        session.__enter__()
        start = client.start_scenario(scenario_id)
        if (
            start.game_timer_ms
            != session.snapshot.epoch_game_timer_ms
        ):
            raise RuntimeError(
                "Scenario Start does not match lockstep epoch"
            )
        session.advance()
        _validate_pitch_sync(client, session, args)

        latencies = []
        minimum_depth = float("inf")
        maximum_depth = 0.0
        digest = hashlib.blake2b(digest_size=16)
        previous_frame_id = 0
        for index in range(args.pairs):
            if index > 0:
                session.advance()
            (
                pair,
                latency_ms,
                pair_depth_minimum,
                pair_depth_maximum,
            ) = _validate_pair(
                client,
                session,
                scenario_id,
                args,
                previous_frame_id,
            )
            previous_frame_id = pair.nadir.frame_id
            last_pair = pair
            latencies.append(latency_ms)
            minimum_depth = min(
                minimum_depth,
                pair_depth_minimum,
            )
            maximum_depth = max(
                maximum_depth,
                pair_depth_maximum,
            )
            digest.update(
                pair.oblique.frame_id.to_bytes(8, "little")
            )
            digest.update(
                pair.nadir.frame_id.to_bytes(8, "little")
            )
            digest.update(pair.oblique.rgb[:64])
            digest.update(pair.nadir.rgb[:64])
            gta_peak_rss = max(
                gta_peak_rss,
                gta_process.memory_info().rss,
            )
            python_peak_rss = max(
                python_peak_rss,
                python_process.memory_info().rss,
            )
            if (
                (index + 1) % args.progress_interval == 0
                or index + 1 == args.pairs
            ):
                print(
                    f"{index + 1}/{args.pairs} "
                    f"step={pair.clock.step_index} "
                    "frames="
                    f"{pair.oblique.frame_id}/"
                    f"{pair.nadir.frame_id} "
                    f"latency={latency_ms:.1f}ms "
                    f"depth=[{pair_depth_minimum:.3f}, "
                    f"{pair_depth_maximum:.3f}]m"
                )

        client.reset_scenario(scenario_id)
        scenario_reset = True
        session.close()
        session = None
    except BaseException as error:
        failure = error

    cleanup_errors = []
    if not scenario_reset and scenario_id is not None:
        try:
            client.reset_scenario(scenario_id)
            scenario_reset = True
        except Exception as error:
            cleanup_errors.append(
                f"scenario Reset failed: {error}"
            )
    if session is not None and scenario_reset:
        try:
            session.close()
            session = None
        except Exception as error:
            cleanup_errors.append(
                f"lockstep Exit failed: {error}"
            )
    try:
        if client.is_camera_active():
            client.set_camera_pose(
                original_pose[0],
                original_pose[1],
                original_pose[2],
                original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        client.restore_player()
    except Exception as error:
        cleanup_errors.append(
            f"camera/player restore failed: {error}"
        )
    if cleanup_errors:
        cleanup_message = "; ".join(cleanup_errors)
        if failure is None:
            raise RuntimeError(cleanup_message)
        if hasattr(failure, "add_note"):
            failure.add_note(
                cleanup_message + "; press F11 in GTA"
            )
    if failure is not None:
        raise failure

    gta_growth_mb = (
        gta_process.memory_info().rss - gta_initial_rss
    ) / (1024**2)
    python_growth_mb = (
        python_process.memory_info().rss - python_initial_rss
    ) / (1024**2)
    gta_peak_growth_mb = (
        gta_peak_rss - gta_initial_rss
    ) / (1024**2)
    python_peak_growth_mb = (
        python_peak_rss - python_initial_rss
    ) / (1024**2)
    for label, value in (
        ("GTA final", gta_growth_mb),
        ("GTA peak", gta_peak_growth_mb),
        ("Python final", python_growth_mb),
        ("Python peak", python_peak_growth_mb),
    ):
        if value > args.max_memory_growth_mb:
            raise RuntimeError(
                f"{label} memory grew by {value:.1f}MiB; "
                f"limit={args.max_memory_growth_mb:.1f}MiB"
            )

    print(
        "PASS "
        f"pairs={args.pairs} "
        f"latency_p50={statistics.median(latencies):.1f}ms "
        f"latency_p95="
        f"{np.percentile(latencies, 95):.1f}ms "
        f"depth=[{minimum_depth:.3f}, {maximum_depth:.3f}]m "
        f"gta_growth={gta_growth_mb:.1f}MiB "
        f"python_growth={python_growth_mb:.1f}MiB "
        f"digest={digest.hexdigest()}"
    )
    print("No RGB-D or point-cloud payload was written to disk.")

    if args.show_pointcloud:
        _show_pair_pointcloud(
            last_pair,
            args.pixel_stride,
            args.max_view_depth,
        )


if __name__ == "__main__":
    main()
