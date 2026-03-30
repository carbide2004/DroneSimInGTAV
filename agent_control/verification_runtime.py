import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from action_mapping import dispatch_action, parse_action
from prompting import build_prompt
from rgbd_utils import depth_bytes_to_pil, rgb_bytes_to_pil


def read_jsonl(path: Path) -> List[Dict]:
    samples = []
    if not path.exists():
        return samples

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
    return samples


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def calculate_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 +
        (pos1[1] - pos2[1]) ** 2 +
        (pos1[2] - pos2[2]) ** 2
    )


def create_anomaly_at_position(cli, anomaly_type: str, position: Tuple[float, float, float]) -> Optional[Tuple]:
    x, y, z = position

    cli.stop_camera()
    time.sleep(1.0)

    cli.teleport_player(x, y, z)
    time.sleep(1.0)

    cli.create_camera()
    time.sleep(2.0)

    original_pose = cli.get_pose()
    if original_pose is None:
        return None

    cli.set_posture(x, y, z + 2.0, 0.0, 0.0, 0.0)
    time.sleep(0.5)

    if anomaly_type == "fire":
        result = cli.create_fire()
    elif anomaly_type == "fight":
        result = cli.create_fight()
    else:
        print(f"Unknown anomaly type: {anomaly_type}")
        return None

    ox, oy, oz, orx, ory, orz = original_pose
    cli.set_posture(ox, oy, oz, orx, ory, orz)
    return result


def build_movement_params(args) -> Dict:
    return {
        "forward_step": float(args.forward_step),
        "up_step": float(args.down_step),
        "down_step": float(args.down_step),
        "yaw_step": float(args.yaw_step),
    }


def run_single_verification(cli, model, sample: Dict, max_steps: int, movement_params: Dict) -> Dict:
    scenario_id = sample["scenario_id"]
    anomaly_type = sample["anomaly_type"]
    anomaly_pos = sample["anomaly_position"]
    start_pose = sample["start_pose"]
    expected_steps = sample["expected_steps"]
    task_desc = sample["task_description"]

    print(f"\n=== Testing {scenario_id} ===")
    print(f"Anomaly: {anomaly_type} at ({anomaly_pos['x']:.1f}, {anomaly_pos['y']:.1f}, {anomaly_pos['z']:.1f})")
    print(f"Start: ({start_pose['x']:.1f}, {start_pose['y']:.1f}, {start_pose['z']:.1f})")
    print(f"Expected steps: {expected_steps}")

    anomaly_result = create_anomaly_at_position(
        cli,
        anomaly_type,
        (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"]),
    )
    if anomaly_result is None:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": "Failed to create anomaly",
            "actual_steps": 0,
            "final_distance": float("inf"),
            "path_efficiency": 0.0,
        }

    cli.set_posture(
        start_pose["x"],
        start_pose["y"],
        start_pose["z"],
        start_pose.get("rx", 0.0),
        start_pose.get("ry", 0.0),
        start_pose.get("rz", 0.0),
    )
    time.sleep(1.0)

    steps = 0
    stopped_by_model = False
    final_pose = None

    try:
        while steps < max_steps:
            pose = cli.get_pose()
            if pose is None:
                print(f"  [{steps}] Failed to get pose after retries, skipping step")
                continue

            final_pose = pose
            cap = cli.capture()
            if cap is None:
                print(f"  [{steps}] Failed to capture images after retries, skipping step")
                continue

            w, h, rgb_bytes, depth_bytes = cap
            try:
                rgb_pil = rgb_bytes_to_pil(w, h, rgb_bytes)
                depth_pil = depth_bytes_to_pil(w, h, depth_bytes)
            except Exception as e:
                print(f"  [{steps}] Failed to process captured images: {e}, skipping step")
                continue

            x, y, z, rx, ry, rz = pose
            prompt = build_prompt(x, y, z, rz, task=task_desc)
            raw = model.generate_action(prompt, rgb_pil, depth_pil)
            action = parse_action(raw) or "AUTO_FORWARD"

            print(f"  [{steps}] {action}")

            if action == "AUTO_STOP_REACHED":
                stopped_by_model = True
                break

            dispatch_action(cli, action, **movement_params)
            steps += 1
    except Exception as e:
        print(f"  Error during exploration: {e}")
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": str(e),
            "actual_steps": steps,
            "final_distance": float("inf"),
            "path_efficiency": 0.0,
        }

    if final_pose is None:
        final_distance = float("inf")
    else:
        fx, fy, fz = final_pose[:3]
        final_distance = calculate_distance(
            (fx, fy, fz),
            (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"]),
        )

    success = stopped_by_model and final_distance <= 20.0
    path_efficiency = min(1.0, expected_steps / max(1, steps)) if steps > 0 else 0.0

    result = {
        "scenario_id": scenario_id,
        "success": success,
        "stopped_by_model": stopped_by_model,
        "actual_steps": steps,
        "expected_steps": expected_steps,
        "final_distance": final_distance,
        "path_efficiency": path_efficiency,
        "anomaly_type": anomaly_type,
        "task_description": task_desc,
    }

    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Steps: {steps}/{expected_steps}, Distance: {final_distance:.1f}m, Efficiency: {path_efficiency:.2f}")
    return result


def calculate_summary_stats(results: List[Dict]) -> Dict:
    if not results:
        return {}

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    successful_results = [r for r in results if r["success"]]

    if successful_results:
        avg_steps = sum(r["actual_steps"] for r in successful_results) / len(successful_results)
        avg_efficiency = sum(r["path_efficiency"] for r in successful_results) / len(successful_results)
        avg_distance = sum(r["final_distance"] for r in successful_results) / len(successful_results)
    else:
        avg_steps = 0.0
        avg_efficiency = 0.0
        avg_distance = 0.0

    overall_avg_steps = sum(r["actual_steps"] for r in results) / total
    overall_avg_efficiency = sum(r["path_efficiency"] for r in results) / total

    by_type = {}
    for result in results:
        anomaly_type = result.get("anomaly_type", "unknown")
        if anomaly_type not in by_type:
            by_type[anomaly_type] = {"total": 0, "success": 0}
        by_type[anomaly_type]["total"] += 1
        if result["success"]:
            by_type[anomaly_type]["success"] += 1

    success_rates_by_type = {}
    for anomaly_type, stats in by_type.items():
        success_rates_by_type[anomaly_type] = {
            "total": stats["total"],
            "successful": stats["success"],
            "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0.0,
        }

    return {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": total - successful,
        "overall_success_rate": successful / total if total > 0 else 0.0,
        "successful_runs_stats": {
            "avg_steps": avg_steps,
            "avg_path_efficiency": avg_efficiency,
            "avg_final_distance": avg_distance,
        } if successful_results else None,
        "overall_averages": {
            "avg_steps": overall_avg_steps,
            "avg_path_efficiency": overall_avg_efficiency,
        },
        "by_anomaly_type": success_rates_by_type,
    }


def print_summary(results: List[Dict]):
    if not results:
        print("\nNo results to summarize.")
        return

    stats = calculate_summary_stats(results)

    print(f"\n{'=' * 50}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful_samples']} ({stats['overall_success_rate'] * 100:.1f}%)")
    print(f"Failed: {stats['failed_samples']} ({(1 - stats['overall_success_rate']) * 100:.1f}%)")

    if stats["successful_runs_stats"]:
        print("\nSuccessful runs averages:")
        print(f"  Steps: {stats['successful_runs_stats']['avg_steps']:.1f}")
        print(f"  Path efficiency: {stats['successful_runs_stats']['avg_path_efficiency']:.2f}")
        print(f"  Final distance: {stats['successful_runs_stats']['avg_final_distance']:.1f}m")

    print("\nOverall averages (all runs):")
    print(f"  Steps: {stats['overall_averages']['avg_steps']:.1f}")
    print(f"  Path efficiency: {stats['overall_averages']['avg_path_efficiency']:.2f}")

    print("\nBy anomaly type:")
    for anomaly_type, type_stats in stats["by_anomaly_type"].items():
        success_rate = type_stats["success_rate"] * 100
        print(f"  {anomaly_type}: {type_stats['successful']}/{type_stats['total']} ({success_rate:.1f}%)")
    print(f"{'=' * 50}")
