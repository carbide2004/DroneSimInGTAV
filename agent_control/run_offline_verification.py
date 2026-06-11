import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from action_mapping import parse_action
from offline_replay_db import OfflineReplayDB, ReplayState
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from verification_runtime import calculate_distance, calculate_summary_stats, print_summary, read_jsonl, write_json


def _repo_root():
    env_path = os.getenv("DRONESIM_ROOT")
    if env_path:
        return Path(env_path)
    return Path(r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V")


def _pose_from_state(state: ReplayState) -> Dict[str, float]:
    return {
        "x": float(state.x),
        "y": float(state.y),
        "z": float(state.z),
        "rx": float(state.rx),
        "ry": float(state.ry),
        "rz": float(state.rz),
    }


def _target_pose_after_action(state: ReplayState, action: str, movement_params: Dict) -> Dict[str, float]:
    pose = _pose_from_state(state)
    action = str(action).upper()
    if action == "AUTO_FORWARD":
        rz_rad = math.radians(float(pose["rz"]))
        pose["x"] += math.cos(rz_rad) * float(movement_params["forward_step"])
        pose["y"] += math.sin(rz_rad) * float(movement_params["forward_step"])
    elif action == "AUTO_UP":
        pose["z"] += float(movement_params["up_step"])
    elif action == "AUTO_DOWN":
        pose["z"] -= float(movement_params["down_step"])
    elif action == "AUTO_YAW_LEFT":
        pose["rz"] = (pose["rz"] + float(movement_params["yaw_step"])) % 360.0
    elif action == "AUTO_YAW_RIGHT":
        pose["rz"] = (pose["rz"] - float(movement_params["yaw_step"])) % 360.0
    return pose


def _miss_record(
    sample: Dict,
    step: int,
    reason: str,
    wanted_pose: Dict,
    nearest,
    action: Optional[str] = None,
    current_state: Optional[ReplayState] = None,
) -> Dict:
    return {
        "scenario_id": sample.get("scenario_id") or sample.get("session_id") or sample.get("sample_id"),
        "step": int(step),
        "reason": reason,
        "action": action,
        "current_state": {
            "state_id": current_state.state_id,
            "sample_id": current_state.sample_id,
            "trajectory_id": current_state.trajectory_id,
            "step_index": current_state.step_index,
            "pose": _pose_from_state(current_state),
        } if current_state is not None else None,
        "wanted_pose": {
            "x": float(wanted_pose.get("x", 0.0)),
            "y": float(wanted_pose.get("y", 0.0)),
            "z": float(wanted_pose.get("z", 0.0)),
            "rx": float(wanted_pose.get("rx", 0.0)),
            "ry": float(wanted_pose.get("ry", 0.0)),
            "rz": float(wanted_pose.get("rz", 0.0)),
        },
        "nearest_state": {
            "state_id": nearest.state.state_id,
            "sample_id": nearest.state.sample_id,
            "trajectory_id": nearest.state.trajectory_id,
            "step_index": nearest.state.step_index,
            "pose": _pose_from_state(nearest.state),
        },
        "distance_xyz": float(nearest.distance_xyz),
        "distance_yaw": float(nearest.distance_yaw),
        "nearest_score": float(nearest.score),
        "repeat_count": 1,
    }


def _quantize_float(value: float, decimals: int = 3) -> float:
    return round(float(value), int(decimals))


def _miss_key(record: Dict) -> Tuple:
    wanted = record.get("wanted_pose") or {}
    nearest_state = record.get("nearest_state") or {}
    return (
        record.get("reason"),
        record.get("action"),
        _quantize_float(wanted.get("x", 0.0)),
        _quantize_float(wanted.get("y", 0.0)),
        _quantize_float(wanted.get("z", 0.0)),
        _quantize_float(wanted.get("rz", 0.0)),
        nearest_state.get("state_id"),
    )


def _append_deduped_miss(misses: List[Dict], miss_index: Dict[Tuple, Dict], record: Dict):
    key = _miss_key(record)
    existing = miss_index.get(key)
    if existing is None:
        record["repeat_count"] = 1
        record["first_step"] = int(record.get("step", 0))
        record["last_step"] = int(record.get("step", 0))
        miss_index[key] = record
        misses.append(record)
        return

    existing["repeat_count"] = int(existing.get("repeat_count", 1)) + 1
    existing["last_step"] = int(record.get("step", existing.get("last_step", existing.get("step", 0))))


def _dedupe_misses(rows: List[Dict]) -> List[Dict]:
    deduped = []
    index: Dict[Tuple, Dict] = {}
    for row in rows:
        key = _miss_key(row)
        existing = index.get(key)
        scenario_id = row.get("scenario_id")
        if existing is None:
            merged = dict(row)
            merged["repeat_count"] = int(merged.get("repeat_count", 1))
            merged["scenario_ids"] = [scenario_id] if scenario_id else []
            index[key] = merged
            deduped.append(merged)
            continue

        existing["repeat_count"] = int(existing.get("repeat_count", 1)) + int(row.get("repeat_count", 1))
        if scenario_id and scenario_id not in existing.setdefault("scenario_ids", []):
            existing["scenario_ids"].append(scenario_id)
    return deduped


def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return ordered[idx]


def run_single_offline_verification(
    replay_db: OfflineReplayDB,
    model,
    sample: Dict,
    max_steps: int,
    movement_params: Dict,
    strict_on_miss: bool = False,
) -> Dict:
    scenario_id = sample.get("scenario_id") or sample.get("session_id") or sample.get("sample_id") or "unknown"
    anomaly_type = sample.get("anomaly_type", "unknown")
    anomaly_pos = sample["anomaly_position"]
    start_pose = sample["start_pose"]
    expected_steps = int(sample.get("expected_steps", max_steps))
    task_desc = sample.get("task_description") or sample.get("task") or "find the closest burning car"

    print(f"\n=== Offline testing {scenario_id} ===")
    print(f"Start: ({start_pose['x']:.1f}, {start_pose['y']:.1f}, {start_pose['z']:.1f})")

    misses: List[Dict] = []
    miss_index: Dict[Tuple, Dict] = {}
    fallback_hits = 0
    transition_hits = 0
    distance_xyz_values: List[float] = []
    distance_yaw_values: List[float] = []

    start_match = replay_db.nearest(start_pose)
    current = start_match.state
    distance_xyz_values.append(start_match.distance_xyz)
    distance_yaw_values.append(start_match.distance_yaw)
    if not start_match.within_threshold:
        _append_deduped_miss(misses, miss_index, _miss_record(sample, 0, "start_pose_miss", start_pose, start_match))
        if strict_on_miss:
            return {
                "scenario_id": scenario_id,
                "success": False,
                "strict_success": False,
                "error": "start_pose_miss",
                "actual_steps": 0,
                "expected_steps": expected_steps,
                "final_distance": float("inf"),
                "path_efficiency": 0.0,
                "anomaly_type": anomaly_type,
                "task_description": task_desc,
                "coverage": {"miss_count": len(misses), "miss_occurrence_count": sum(int(m.get("repeat_count", 1)) for m in misses), "fallback_hits": 0, "transition_hits": 0},
                "misses": misses,
            }

    steps = 0
    stopped_by_model = False
    final_state = current
    step_trace = []

    while steps < max_steps:
        final_state = current
        try:
            rgb_pil, depth_pil = replay_db.load_images(current)
        except Exception as e:
            _append_deduped_miss(
                misses,
                miss_index,
                _miss_record(
                    sample,
                    steps,
                    f"image_load_failed: {e}",
                    _pose_from_state(current),
                    replay_db.nearest(_pose_from_state(current)),
                    current_state=current,
                ),
            )
            break

        prompt = build_prompt(current.x, current.y, current.z, current.rz, task=task_desc)
        raw = model.generate_action(prompt, rgb_pil, depth_pil)
        action = parse_action(raw) or "AUTO_FORWARD"

        print(f"  [{steps}] state={current.sample_id} action={action}")
        step_trace.append({
            "step": int(steps),
            "state_id": int(current.state_id),
            "sample_id": current.sample_id,
            "pose": _pose_from_state(current),
            "action": action,
        })

        if action == "AUTO_STOP_REACHED":
            stopped_by_model = True
            break

        next_state = replay_db.transition(current.state_id, action)
        if next_state is not None:
            transition_hits += 1
            current = next_state
            steps += 1
            continue

        wanted_pose = _target_pose_after_action(current, action, movement_params)
        nearest = replay_db.nearest(wanted_pose)
        distance_xyz_values.append(nearest.distance_xyz)
        distance_yaw_values.append(nearest.distance_yaw)
        fallback_hits += 1

        if not nearest.within_threshold:
            _append_deduped_miss(
                misses,
                miss_index,
                _miss_record(sample, steps + 1, "transition_miss", wanted_pose, nearest, action=action, current_state=current),
            )
            if strict_on_miss:
                break

        current = nearest.state
        steps += 1

    final_pose = _pose_from_state(final_state) if final_state is not None else None
    if final_pose is None:
        final_distance = float("inf")
    else:
        final_distance = calculate_distance(
            (final_pose["x"], final_pose["y"], final_pose["z"]),
            (float(anomaly_pos["x"]), float(anomaly_pos["y"]), float(anomaly_pos["z"])),
        )

    success = stopped_by_model and final_distance <= 20.0
    strict_success = success and not misses
    path_efficiency = min(1.0, expected_steps / max(1, steps)) if steps > 0 else 0.0
    miss_occurrence_count = sum(int(miss.get("repeat_count", 1)) for miss in misses)

    result = {
        "scenario_id": scenario_id,
        "success": success,
        "strict_success": strict_success,
        "stopped_by_model": stopped_by_model,
        "actual_steps": steps,
        "expected_steps": expected_steps,
        "final_distance": final_distance,
        "path_efficiency": path_efficiency,
        "anomaly_type": anomaly_type,
        "task_description": task_desc,
        "final_position": {
            "x": final_pose["x"],
            "y": final_pose["y"],
            "z": final_pose["z"],
        } if final_pose else None,
        "coverage": {
            "miss_count": len(misses),
            "miss_occurrence_count": int(miss_occurrence_count),
            "fallback_hits": int(fallback_hits),
            "transition_hits": int(transition_hits),
            "avg_nearest_distance_xyz": sum(distance_xyz_values) / len(distance_xyz_values) if distance_xyz_values else 0.0,
            "p95_nearest_distance_xyz": _percentile(distance_xyz_values, 0.95),
            "avg_nearest_distance_yaw": sum(distance_yaw_values) / len(distance_yaw_values) if distance_yaw_values else 0.0,
            "p95_nearest_distance_yaw": _percentile(distance_yaw_values, 0.95),
        },
        "misses": misses,
        "step_trace": step_trace,
    }

    print(f"  Result: {'SUCCESS' if success else 'FAILED'}, strict={'YES' if strict_success else 'NO'}, misses={len(misses)}, miss_occurrences={miss_occurrence_count}")
    print(f"  Steps: {steps}/{expected_steps}, Distance: {final_distance:.1f}m, Efficiency: {path_efficiency:.2f}")
    return result


def _offline_summary(results: List[Dict]) -> Dict:
    total = len(results)
    misses = [miss for result in results for miss in result.get("misses", [])]
    deduped_misses = _dedupe_misses(misses)
    strict_success = sum(1 for result in results if result.get("strict_success"))
    relaxed_success = sum(1 for result in results if result.get("success"))
    fallback_hits = sum(int(result.get("coverage", {}).get("fallback_hits", 0)) for result in results)
    transition_hits = sum(int(result.get("coverage", {}).get("transition_hits", 0)) for result in results)
    miss_occurrences = sum(int(miss.get("repeat_count", 1)) for miss in deduped_misses)
    xyz = [float(miss.get("distance_xyz", 0.0)) for miss in deduped_misses]
    yaw = [float(miss.get("distance_yaw", 0.0)) for miss in deduped_misses]
    return {
        "strict_successful_samples": strict_success,
        "strict_success_rate": strict_success / total if total else 0.0,
        "relaxed_successful_samples": relaxed_success,
        "relaxed_success_rate": relaxed_success / total if total else 0.0,
        "coverage_miss_count": len(deduped_misses),
        "coverage_miss_occurrence_count": int(miss_occurrences),
        "transition_hits": int(transition_hits),
        "fallback_hits": int(fallback_hits),
        "miss_p95_distance_xyz": _percentile(xyz, 0.95),
        "miss_p95_distance_yaw": _percentile(yaw, 0.95),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline replay verification from SQLite DB")
    parser.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_merged"),
        help="Model directory path",
    )
    parser.add_argument("--root_path", default=None, help="Root path for verification file defaults")
    parser.add_argument("--verification_file", default=None, help="默认 root_path/data/verification/samples.jsonl")
    parser.add_argument("--output_file", default=None, help="默认 root_path/data/verification/results_offline.json")
    parser.add_argument("--misses_file", default=None, help="默认 output_file 同目录 coverage_misses.jsonl")
    parser.add_argument("--db_path", required=True, help="offline replay SQLite DB")
    parser.add_argument("--dataset_root", default=None, help="DB 未内嵌图片时使用的图片根目录")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--forward_step", type=float, default=5.0)
    parser.add_argument("--up_step", type=float, default=5.0)
    parser.add_argument("--down_step", type=float, default=5.0)
    parser.add_argument("--yaw_step", type=float, default=15.0)
    parser.add_argument("--xyz_threshold", type=float, default=5.0)
    parser.add_argument("--yaw_threshold", type=float, default=15.0)
    parser.add_argument("--sample_limit", type=int, default=-1)
    parser.add_argument("--strict_on_miss", action="store_true", help="遇到 coverage miss 立刻结束该样本")
    return parser


def main():
    args = build_arg_parser().parse_args()
    root_path = Path(args.root_path) if args.root_path else _repo_root()
    if args.verification_file is None:
        args.verification_file = str(root_path / "data" / "verification" / "samples.jsonl")
    if args.output_file is None:
        args.output_file = str(root_path / "data" / "verification" / "results_offline.json")
    if args.misses_file is None:
        args.misses_file = str(Path(args.output_file).with_name("coverage_misses.jsonl"))

    samples = read_jsonl(Path(args.verification_file))
    if args.sample_limit > 0:
        samples = samples[: int(args.sample_limit)]
    if not samples:
        print(f"Error: No samples found in {args.verification_file}")
        return 1

    print(f"Loaded {len(samples)} verification samples")
    print("Loading model...")
    model = Qwen3VLWrapper(args.model_dir).load()

    replay_db = OfflineReplayDB(
        db_path=Path(args.db_path),
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        xyz_threshold=float(args.xyz_threshold),
        yaw_threshold=float(args.yaw_threshold),
    )

    movement_params = {
        "forward_step": float(args.forward_step),
        "up_step": float(args.up_step),
        "down_step": float(args.down_step),
        "yaw_step": float(args.yaw_step),
    }
    results = []
    all_misses = []
    try:
        for i, sample in enumerate(samples):
            print(f"\nProgress: {i + 1}/{len(samples)}")
            result = run_single_offline_verification(
                replay_db=replay_db,
                model=model,
                sample=sample,
                max_steps=int(args.max_steps),
                movement_params=movement_params,
                strict_on_miss=bool(args.strict_on_miss),
            )
            results.append(result)
            all_misses.extend(result.get("misses", []))
    finally:
        replay_db.close()

    all_misses = _dedupe_misses(all_misses)
    summary_stats = calculate_summary_stats(results)
    result_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "offline_replay",
        "total_samples": len(samples),
        "completed_samples": len(results),
        "parameters": {
            "db_path": str(Path(args.db_path)),
            "dataset_root": args.dataset_root,
            "max_steps": args.max_steps,
            "forward_step": args.forward_step,
            "up_step": args.up_step,
            "down_step": args.down_step,
            "yaw_step": args.yaw_step,
            "xyz_threshold": args.xyz_threshold,
            "yaw_threshold": args.yaw_threshold,
            "strict_on_miss": bool(args.strict_on_miss),
        },
        "summary_statistics": summary_stats,
        "offline_replay_statistics": _offline_summary(results),
        "results": results,
    }

    write_json(Path(args.output_file), result_data)
    _write_jsonl(Path(args.misses_file), all_misses)
    print(f"\nResults saved to: {args.output_file}")
    print(f"Coverage misses saved to: {args.misses_file}")
    print_summary(results)
    print("\nOFFLINE REPLAY SUMMARY")
    print(json.dumps(result_data["offline_replay_statistics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
