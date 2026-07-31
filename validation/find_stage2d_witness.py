"""Repeated exploratory search for one strict Stage 2D witness.

This is not the formal two-stratum Stage 2D acceptance test. It repeatedly
samples one requested stratum and stops at the first strict joint witness.
"""

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import DroneSimClient  # noqa: E402
from agent_control.feasibility import (  # noqa: E402
    FeasibilityStatus,
)
from agent_control.task_starts import (  # noqa: E402
    StartVisibilityStratum,
    TASK_HORIZON_STEPS,
)
from validation.trajectory_recording import (  # noqa: E402
    TrajectoryRecorder,
    write_json,
)
from validation.validate_spatiotemporal_feasibility import (  # noqa: E402
    _replay,
    _search_once,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Try multiple episodes from one Stage 2D visibility stratum "
            "and stop after the first joint cue-to-goal witness with "
            "margin."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--all-attempts",
        action="store_true",
        help=(
            "Run every requested attempt and report witness success "
            "rate instead of stopping at the first success"
        ),
    )
    parser.add_argument(
        "--stratum",
        choices=("CUE_VISIBLE", "CUE_HIDDEN"),
        default="CUE_VISIBLE",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument(
        "--start-seed-step",
        type=int,
        default=1,
        help=(
            "Increment applied between attempts; use 0 to retry the "
            "same start candidate with new GTA AI evolution"
        ),
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=TASK_HORIZON_STEPS,
    )
    parser.add_argument("--search-timeout", type=float, default=120.0)
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=32)
    parser.add_argument("--prepare-timeout", type=float, default=20.0)
    parser.add_argument("--max-start-candidates", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--show-witness", action="store_true")
    parser.add_argument(
        "--record-dir",
        type=Path,
        help=(
            "Record successful replay visualization data under this "
            "new directory; by default only the first success is stored"
        ),
    )
    parser.add_argument(
        "--record-all-successes",
        action="store_true",
        help=(
            "With --all-attempts and --record-dir, store every "
            "successful replay instead of only the first"
        ),
    )
    parser.add_argument("--record-jpeg-quality", type=int, default=85)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    final_start_seed = (
        args.start_seed
        + (args.attempts - 1) * args.start_seed_step
    )
    if not 0 <= final_start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("Attempt start seeds must fit uint64")
    if not 8 <= args.horizon_steps <= 256:
        parser.error("--horizon-steps must be in [8, 256]")
    if (
        not math.isfinite(args.search_timeout)
        or args.search_timeout <= 0.0
    ):
        parser.error("--search-timeout must be finite and positive")
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be in [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be in [0, 32]")
    if args.firetrucks + args.pedestrians == 0:
        parser.error("At least one response actor is required")
    if not 1 <= args.record_jpeg_quality <= 95:
        parser.error("--record-jpeg-quality must be in [1, 95]")
    if args.record_all_successes and (
        not args.all_attempts or args.record_dir is None
    ):
        parser.error(
            "--record-all-successes requires both --all-attempts "
            "and --record-dir"
        )
    if args.record_dir is not None:
        args.record_dir = args.record_dir.resolve()
        if args.record_dir.exists():
            parser.error(f"--record-dir already exists: {args.record_dir}")
        TrajectoryRecorder.require_dependencies()
    args.verify_determinism = False
    return args


def _new_recording_manifest(args):
    return {
        "schema_version": 2,
        "status": "PASS",
        "error": None,
        "config": {
            "anchor": list(args.anchor),
            "seed": args.seed,
            "start_seed": args.start_seed,
            "start_seed_step": args.start_seed_step,
            "attempts": args.attempts,
            "stratum": args.stratum,
            "horizon_steps": args.horizon_steps,
            "search_timeout": args.search_timeout,
            "firetrucks": args.firetrucks,
            "pedestrians": args.pedestrians,
            "jpeg_quality": args.record_jpeg_quality,
            "record_all_successes": args.record_all_successes,
        },
        "episodes": [],
    }


def _append_recording_manifest(
    root,
    manifest,
    args,
    episode_root,
    attempt,
    start_seed,
    blueprint_id,
    report,
):
    if root is None:
        return
    trajectory = (
        Path(episode_root)
        / args.stratum
        / "trajectory.json"
    ).relative_to(root)
    manifest["episodes"].append(
        {
            "attempt": attempt,
            "start_seed": start_seed,
            "blueprint_id": blueprint_id,
            "start_id": report.start_id,
            "stratum": args.stratum,
            "status": report.status.name,
            "actions": report.witness.total_actions,
            "remaining_actions": report.witness.remaining_actions,
            "path": trajectory.as_posix(),
        }
    )
    write_json(
        root / "run.json",
        manifest,
    )


def main():
    args = _parse_args()
    started = time.perf_counter()
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    client.set_time(12, 0, 0)
    client.set_weather("EXTRASUNNY")
    client.teleport_player(*args.anchor)
    blueprint_id = 0
    outcomes = Counter()
    stratum = StartVisibilityStratum[args.stratum]
    first_success = None
    recording_written = False
    recording_manifest = (
        None
        if args.record_dir is None
        else _new_recording_manifest(args)
    )
    try:
        print(
            f"{stratum.name} repeated search "
            f"attempts={args.attempts} "
            f"horizon={args.horizon_steps} "
            f"search_timeout={args.search_timeout:.1f}s "
            f"start_seed={args.start_seed} "
            f"start_seed_step={args.start_seed_step}"
        )
        for attempt_index in range(args.attempts):
            attempt = attempt_index + 1
            start_seed = (
                args.start_seed
                + attempt_index * args.start_seed_step
            )
            print(
                f"attempt {attempt}/{args.attempts} "
                f"start_seed={start_seed} blueprint={blueprint_id}"
            )
            (
                blueprint_id,
                generated,
                report,
                digest,
            ) = _search_once(
                client,
                args,
                blueprint_id,
                stratum,
                start_seed,
            )
            outcomes[report.status.name] += 1
            if (
                report.status
                != FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN
            ):
                continue
            if args.record_dir is None:
                recording_root = None
            elif args.record_all_successes:
                recording_root = (
                    args.record_dir
                    / f"attempt_{attempt:03d}_seed_{start_seed}"
                )
            elif not recording_written:
                recording_root = args.record_dir
            else:
                recording_root = None
            _replay(
                client,
                args,
                blueprint_id,
                generated,
                report,
                recording_root=recording_root,
            )
            if recording_root is not None:
                _append_recording_manifest(
                    args.record_dir,
                    recording_manifest,
                    args,
                    recording_root,
                    attempt,
                    start_seed,
                    blueprint_id,
                    report,
                )
                recording_written = True
            if first_success is None:
                first_success = (
                    attempt,
                    start_seed,
                    blueprint_id,
                    report,
                    digest,
                )
            print(
                f"PASS {stratum.name} witness found "
                f"attempt={attempt} start_seed={start_seed} "
                f"start_id={report.start_id} "
                f"actions={report.witness.total_actions} "
                f"slack={report.witness.remaining_actions} "
                f"digest={digest} "
                f"outcomes={dict(outcomes)} "
                f"wall={time.perf_counter() - started:.1f}s"
            )
            if not args.all_attempts:
                if args.record_dir is None:
                    print(
                        "No RGB-D or trajectory payload was written "
                        "to disk."
                    )
                else:
                    print(
                        "Compressed RGB replay written to "
                        f"{args.record_dir}; no Depth payload was written."
                    )
                return
        if first_success is not None:
            success_count = outcomes[
                FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN.name
            ]
            (
                attempt,
                start_seed,
                _blueprint_id,
                report,
                digest,
            ) = first_success
            print(
                f"PASS {stratum.name} repeated audit "
                f"success={success_count}/{args.attempts} "
                f"rate={success_count / args.attempts:.3f} "
                f"outcomes={dict(outcomes)} "
                f"first_success_attempt={attempt} "
                f"first_success_start_seed={start_seed} "
                f"first_success_actions={report.witness.total_actions} "
                f"first_success_digest={digest}"
            )
            if args.record_dir is None:
                print("No RGB-D or trajectory payload was written to disk.")
            else:
                recorded_count = len(
                    recording_manifest["episodes"]
                )
                print(
                    f"Recorded {recorded_count} successful compressed "
                    f"RGB replay(s) under {args.record_dir}; no Depth "
                    "payload was written."
                )
            return
        raise RuntimeError(
            f"No {stratum.name} joint witness with margin was found after "
            f"{args.attempts} attempts; outcomes={dict(outcomes)}"
        )
    finally:
        try:
            client.set_camera_pose(
                *original_pose[:3],
                original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
            print(
                f"repeated-search wall time="
                f"{time.perf_counter() - started:.1f}s"
            )


if __name__ == "__main__":
    main()
