"""Collect one compact Stage 3C DAgger round on checkpoint train anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import DroneSimClient  # noqa: E402
from agent_control.start_pool import load_pool  # noqa: E402
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402
from validation.validate_online_belief_policy import _run_one  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--episodes-per-anchor", type=int, default=5)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--scenario-seed-base", type=int, default=10000)
    parser.add_argument("--start-seed-base", type=int, default=20000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--prepare-timeout", type=float, default=30.0)
    parser.add_argument("--capture-timeout-ms", type=int, default=5000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_anchor <= 100:
        parser.error("--episodes-per-anchor must be in [1, 100]")
    if args.prepare_timeout <= 0.0 or args.capture_timeout_ms <= 0:
        parser.error("Timeouts must be positive")
    default_beta = {1: 0.50, 2: 0.25, 3: 0.00}[args.round]
    args.beta = default_beta if args.beta is None else args.beta
    if not 0.0 <= args.beta <= 1.0:
        parser.error("--beta must be in [0, 1]")
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def _manifest_anchors(dataset_root):
    path = dataset_root / "dataset_manifest.json"
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read schema-4 manifest {path}: {error}") from error
    if manifest.get("schema_version") != 4:
        raise RuntimeError("Stage 3C DAgger requires a schema-4 source dataset")
    anchors = manifest.get("config", {}).get("anchors")
    pool_records = manifest.get("anchor_pools")
    if not isinstance(anchors, list):
        raise RuntimeError("Manifest is missing config.anchors")
    if not isinstance(pool_records, list) or len(pool_records) != len(anchors):
        raise RuntimeError("Schema-4 manifest has no aligned anchor_pools")
    result = {}
    for index, (anchor_payload, pool_record) in enumerate(
        zip(anchors, pool_records, strict=True)
    ):
        if int(pool_record.get("anchor_index", -1)) != index:
            raise RuntimeError("Manifest anchor-pool indices are invalid")
        anchor = tuple(float(value) for value in anchor_payload)
        pool_path = dataset_root / str(pool_record["path"])
        pool = load_pool(pool_path)
        if (
            pool.digest != str(pool_record.get("digest"))
            or len(pool.entries) != int(pool_record.get("count", -1))
            or tuple(round(value, 3) for value in pool.anchor)
            != tuple(round(value, 3) for value in anchor)
        ):
            raise RuntimeError(
                f"ANCHOR_POOL_MISMATCH for anchor_{index:03d}: {pool_path}"
            )
        result[f"anchor_{index:03d}"] = (anchor, pool)
    return result


def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, _model, _geometry = load_policy_checkpoint(args.checkpoint, device)
    train_anchors = tuple(checkpoint.get("train_anchors", ()))
    validation_anchors = set(checkpoint.get("validation_anchors", ()))
    if not train_anchors or set(train_anchors) & validation_anchors:
        raise RuntimeError("Checkpoint train/validation anchor contract is invalid")
    manifest = _manifest_anchors(args.dataset_root)
    unknown = sorted(set(train_anchors) - set(manifest))
    if unknown:
        raise RuntimeError(f"Checkpoint train anchors are absent from manifest: {unknown}")
    contract_horizon = int(checkpoint["episode_contract"]["episode_spec"]["horizon_steps"])
    horizon = contract_horizon if args.max_steps is None else int(args.max_steps)
    if horizon != contract_horizon:
        raise RuntimeError(
            f"--max-steps must match checkpoint horizon {contract_horizon}, received {horizon}"
        )
    args.output_dir.mkdir(parents=True)
    collection = SimpleNamespace(
        checkpoint=args.checkpoint,
        mode="dagger",
        dagger_beta=float(args.beta),
        episodes=len(train_anchors) * args.episodes_per_anchor,
        scenario_seed=int(args.scenario_seed_base),
        start_seed=int(args.start_seed_base),
        max_steps=horizon,
        prepare_timeout=float(args.prepare_timeout),
        capture_timeout_ms=int(args.capture_timeout_ms),
        device=str(args.device),
        record_dir=None,
        jpeg_quality=85,
        dagger_output_dir=args.output_dir,
        anchor_name=None,
        start_pool=None,
    )
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    results = []
    started = time.perf_counter()
    global_index = 0
    try:
        client.set_time(12, 0, 0)
        client.set_weather("EXTRASUNNY")
        for anchor_name in train_anchors:
            collection.anchor_name = anchor_name
            anchor, collection.start_pool = manifest[anchor_name]
            print(f"DAgger start pool REUSE digest={collection.start_pool.digest}", flush=True)
            print(
                f"DAGGER_ANCHOR {anchor_name} beta={collection.dagger_beta:.2f} "
                f"position={anchor}", flush=True,
            )
            for _ in range(args.episodes_per_anchor):
                result = _run_one(client, collection, anchor, global_index)
                results.append(result)
                global_index += 1
    finally:
        try:
            client.set_camera_pose(
                original_pose[0], original_pose[1], original_pose[2],
                original_pose[5], collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
    labels = sum(result.expert_labels for result in results)
    missing = sum(result.no_expert_labels for result in results)
    print(
        f"PASS DAgger round={args.round} beta={args.beta:.2f} "
        f"episodes={len(results)} labels={labels} no_labels={missing} "
        f"successes={sum(result.success for result in results)} wall={time.perf_counter()-started:.1f}s"
    )
    print(f"shards={args.output_dir} RGB/Depth payload files=0")


if __name__ == "__main__":
    main()
