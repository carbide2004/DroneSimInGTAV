import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dronesim_client import DroneSimClient
from action_mapping import dispatch_action, parse_action
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from rgbd_utils import depth_bytes_to_pil, rgb_bytes_to_pil


def _repo_root():
    # Try environment variable first, then fallback to hardcoded path
    env_path = os.getenv('DRONESIM_ROOT')
    if env_path:
        return Path(env_path)
    return Path(r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V")


def _read_jsonl(path: Path) -> List[Dict]:
    """Read verification samples from JSONL file"""
    samples = []
    if not path.exists():
        return samples
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
    
    return samples


def _write_json(path: Path, obj):
    """Write JSON object to file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _calculate_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Calculate 3D distance between two positions"""
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 +
        (pos1[1] - pos2[1]) ** 2 +
        (pos1[2] - pos2[2]) ** 2
    )


def create_anomaly_at_position(cli: DroneSimClient, anomaly_type: str, position: Tuple[float, float, float]) -> Optional[Tuple]:
    """Create anomaly at specified position with explicit view management"""
    x, y, z = position
    
    # Step 1: Stop camera to switch to player view
    cli.stop_camera()
    time.sleep(1.0)  # Wait for view switch
    
    # Step 2: Teleport player to anomaly center (sets invincible + invisible)
    cli.teleport_player(x, y, z)
    time.sleep(1.0)  # Wait for teleportation
    
    # Step 3: Switch back to camera mode
    cam_id = cli.create_camera()
    time.sleep(2.0)  # Wait for camera initialization
    
    # Step 4: Save current camera posture
    original_pose = cli.get_pose()
    if original_pose is None:
        return None
    
    # Step 5: Move camera to anomaly position temporarily to create the anomaly there
    cli.set_posture(x, y, z + 2.0, 0.0, 0.0, 0.0)  # 2m above, no rotation change
    time.sleep(0.5)  # Wait for camera to move
    
    # Step 6: Create anomaly
    if anomaly_type == "fire":
        result = cli.create_fire()
    elif anomaly_type == "fight":
        result = cli.create_fight()
    else:
        print(f"Unknown anomaly type: {anomaly_type}")
        return None
    
    # Step 7: Restore original camera posture
    if original_pose:
        ox, oy, oz, orx, ory, orz = original_pose
        cli.set_posture(ox, oy, oz, orx, ory, orz)
    
    return result


def run_single_verification(
    cli: DroneSimClient,
    model: Qwen3VLWrapper,
    sample: Dict,
    max_steps: int,
    movement_params: Dict
) -> Dict:
    """Run verification for a single sample"""
    
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
    
    # Create anomaly at specified position
    anomaly_result = create_anomaly_at_position(
        cli, anomaly_type, (anomaly_pos['x'], anomaly_pos['y'], anomaly_pos['z'])
    )
    
    if anomaly_result is None:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": "Failed to create anomaly",
            "actual_steps": 0,
            "final_distance": float('inf'),
            "path_efficiency": 0.0
        }
    
    # Set camera to start position
    cli.set_posture(
        start_pose['x'], start_pose['y'], start_pose['z'],
        start_pose.get('rx', 0.0), start_pose.get('ry', 0.0), start_pose.get('rz', 0.0)
    )
    
    time.sleep(1.0)  # Wait for camera to stabilize
    
    # Run AI exploration
    steps = 0
    stopped_by_model = False
    final_pose = None
    
    try:
        while steps < max_steps:
            # Get current pose with retry
            pose = cli.get_pose()
            if pose is None:
                print(f"  [{steps}] Failed to get pose after retries, skipping step")
                continue
            
            final_pose = pose
            
            # Capture images with retry
            cap = cli.capture()
            if cap is None:
                print(f"  [{steps}] Failed to capture images after retries, skipping step")
                continue
            
            w, h, rgb_bytes, depth_bytes = cap
            
            # Validate captured data before processing
            try:
                rgb_pil = rgb_bytes_to_pil(w, h, rgb_bytes)
                depth_pil = depth_bytes_to_pil(w, h, depth_bytes)
            except Exception as e:
                print(f"  [{steps}] Failed to process captured images: {e}, skipping step")
                continue
            
            # Generate action
            x, y, z, rx, ry, rz = pose
            prompt = build_prompt(x, y, z, rz, task=task_desc)
            raw = model.generate_action(prompt, rgb_pil, depth_pil)
            action = parse_action(raw)
            if action is None:
                action = "AUTO_FORWARD"
            
            print(f"  [{steps}] {action}")
            
            if action == "AUTO_STOP_REACHED":
                stopped_by_model = True
                break
            
            # Execute action
            dispatch_action(cli, action, **movement_params)
            steps += 1
    
    except Exception as e:
        print(f"  Error during exploration: {e}")
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": str(e),
            "actual_steps": steps,
            "final_distance": float('inf'),
            "path_efficiency": 0.0
        }
    
    # Calculate results
    if final_pose is None:
        final_distance = float('inf')
    else:
        fx, fy, fz = final_pose[:3]
        final_distance = _calculate_distance(
            (fx, fy, fz),
            (anomaly_pos['x'], anomaly_pos['y'], anomaly_pos['z'])
        )
    
    # Success criteria: stopped by model AND within 10m of anomaly
    success = stopped_by_model and final_distance <= 10.0
    
    # Path efficiency: expected_steps / actual_steps (capped at 1.0)
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
        "task_description": task_desc
    }
    
    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Steps: {steps}/{expected_steps}, Distance: {final_distance:.1f}m, Efficiency: {path_efficiency:.2f}")
    
    return result


def calculate_summary_stats(results: List[Dict]) -> Dict:
    """Calculate summary statistics from results"""
    if not results:
        return {}
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    
    # Calculate averages for successful runs
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_steps = sum(r["actual_steps"] for r in successful_results) / len(successful_results)
        avg_efficiency = sum(r["path_efficiency"] for r in successful_results) / len(successful_results)
        avg_distance = sum(r["final_distance"] for r in successful_results) / len(successful_results)
    else:
        avg_steps = avg_efficiency = avg_distance = 0.0
    
    # Calculate overall averages (including failed runs)
    overall_avg_steps = sum(r["actual_steps"] for r in results) / total
    overall_avg_efficiency = sum(r["path_efficiency"] for r in results) / total
    
    # Group by anomaly type
    by_type = {}
    for r in results:
        anomaly_type = r.get("anomaly_type", "unknown")
        if anomaly_type not in by_type:
            by_type[anomaly_type] = {"total": 0, "success": 0}
        by_type[anomaly_type]["total"] += 1
        if r["success"]:
            by_type[anomaly_type]["success"] += 1
    
    # Calculate success rates by type
    success_rates_by_type = {}
    for anomaly_type, stats in by_type.items():
        success_rates_by_type[anomaly_type] = {
            "total": stats["total"],
            "successful": stats["success"],
            "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        }
    
    return {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": total - successful,
        "overall_success_rate": successful / total if total > 0 else 0.0,
        "successful_runs_stats": {
            "avg_steps": avg_steps,
            "avg_path_efficiency": avg_efficiency,
            "avg_final_distance": avg_distance
        } if successful_results else None,
        "overall_averages": {
            "avg_steps": overall_avg_steps,
            "avg_path_efficiency": overall_avg_efficiency
        },
        "by_anomaly_type": success_rates_by_type
    }


def print_summary(results: List[Dict]):
    """Print verification summary"""
    if not results:
        print("\nNo results to summarize.")
        return
    
    stats = calculate_summary_stats(results)
    
    print(f"\n{'='*50}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful_samples']} ({stats['overall_success_rate']*100:.1f}%)")
    print(f"Failed: {stats['failed_samples']} ({(1-stats['overall_success_rate'])*100:.1f}%)")
    
    if stats['successful_runs_stats']:
        print(f"\nSuccessful runs averages:")
        print(f"  Steps: {stats['successful_runs_stats']['avg_steps']:.1f}")
        print(f"  Path efficiency: {stats['successful_runs_stats']['avg_path_efficiency']:.2f}")
        print(f"  Final distance: {stats['successful_runs_stats']['avg_final_distance']:.1f}m")
    
    print(f"\nOverall averages (all runs):")
    print(f"  Steps: {stats['overall_averages']['avg_steps']:.1f}")
    print(f"  Path efficiency: {stats['overall_averages']['avg_path_efficiency']:.2f}")
    
    print(f"\nBy anomaly type:")
    for anomaly_type, type_stats in stats['by_anomaly_type'].items():
        success_rate = type_stats['success_rate'] * 100
        print(f"  {anomaly_type}: {type_stats['successful']}/{type_stats['total']} ({success_rate:.1f}%)")
    
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Run verification tests on collected samples")
    parser.add_argument("--host", default="127.0.0.5", help="DroneSim host")
    parser.add_argument("--port", type=int, default=23456, help="DroneSim port")
    parser.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_merged"),
        help="Model directory path"
    )
    parser.add_argument(
        "--root_path",
        default=None,
        help="Root path for data files (overrides default and environment variable)"
    )
    parser.add_argument(
        "--verification_file",
        default=None,
        help="Verification samples file (if not specified, uses root_path/data/verification/samples.jsonl)"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Results output file (if not specified, uses root_path/data/verification/results.json)"
    )
    parser.add_argument("--max_steps", type=int, default=150, help="Maximum steps per test")
    parser.add_argument("--sleep_s", type=float, default=3.0, help="Initial sleep time")
    parser.add_argument("--fov", type=float, default=None, help="Camera FOV")
    parser.add_argument("--forward_step", type=float, default=5.0, help="Forward step size")
    parser.add_argument("--down_step", type=float, default=5.0, help="Vertical step size")
    parser.add_argument("--yaw_step", type=float, default=15.0, help="Yaw step size")
    parser.add_argument("--sample_limit", type=int, default=-1, help="Limit number of samples to test (-1 for all)")
    
    args = parser.parse_args()
    
    # Determine root path
    if args.root_path:
        root_path = Path(args.root_path)
    else:
        root_path = _repo_root()
    
    # Set default file paths if not specified
    if args.verification_file is None:
        args.verification_file = str(root_path / "data" / "verification" / "samples.jsonl")
    if args.output_file is None:
        args.output_file = str(root_path / "data" / "verification" / "results.json")
    
    # Load verification samples
    verification_file = Path(args.verification_file)
    if not verification_file.exists():
        print(f"Error: Verification file not found: {verification_file}")
        return 1
    
    samples = _read_jsonl(verification_file)
    if not samples:
        print(f"Error: No samples found in {verification_file}")
        return 1
    
    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]
    
    print(f"Loaded {len(samples)} verification samples")
    
    # Load model
    print("Loading model...")
    model = Qwen3VLWrapper(args.model_dir).load()
    
    # Connect to DroneSim
    print("Connecting to DroneSim...")
    cli = DroneSimClient(host=args.host, port=int(args.port))
    
    print(f"Waiting {args.sleep_s} seconds...")
    time.sleep(float(args.sleep_s))
    
    # Movement parameters
    movement_params = {
        "forward_step": float(args.forward_step),
        "up_step": float(args.down_step),
        "down_step": float(args.down_step),
        "yaw_step": float(args.yaw_step),
    }
    
    results = []
    
    try:
        # Initialize camera
        cli.set_time(12, 0, 0)
        cam_id = cli.create_camera()
        if args.fov is not None:
            cli.set_fov(float(args.fov))
        
        print(f"Camera initialized: cam_id={cam_id}")
        
        # Run verification for each sample
        for i, sample in enumerate(samples):
            print(f"\nProgress: {i+1}/{len(samples)}")
            result = run_single_verification(cli, model, sample, args.max_steps, movement_params)
            results.append(result)
            
            # Small delay between tests
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
    except Exception as e:
        print(f"\nError during verification: {e}")
    finally:
        try:
            # Stop camera to switch to player view for restoration
            cli.stop_camera()
            time.sleep(1.0)
            
            # Restore player to normal state
            cli.restore_player()
            time.sleep(1.0)
            
            print("Player restored to normal state")
        except Exception:
            pass
    
    # Save results
    if results:
        output_file = Path(args.output_file)
        
        # Calculate summary statistics
        summary_stats = calculate_summary_stats(results)
        
        result_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(samples),
            "completed_samples": len(results),
            "parameters": {
                "max_steps": args.max_steps,
                "forward_step": args.forward_step,
                "down_step": args.down_step,
                "yaw_step": args.yaw_step,
            },
            "summary_statistics": summary_stats,
            "results": results
        }
        
        _write_json(output_file, result_data)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print_summary(results)
    else:
        print("\nNo results to save.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())