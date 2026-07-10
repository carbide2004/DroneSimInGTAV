import argparse
import os
import time
from pathlib import Path

from dronesim_client import DroneSimClient
from qwen3vl_wrapper import Qwen3VLWrapper
from verification_runtime import (
    build_movement_params,
    calculate_summary_stats,
    print_summary,
    read_jsonl,
    run_single_verification,
    write_json,
)


def _repo_root():
    # 优先尝试环境变量，然后回退到硬编码路径
    env_path = os.getenv('DRONESIM_ROOT')
    if env_path:
        return Path(env_path)
    return Path(r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V")


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
    parser.add_argument(
        "--up_step",
        type=float,
        default=None,
        help="Upward step size; defaults to --down_step for backward compatibility",
    )
    parser.add_argument("--down_step", type=float, default=5.0, help="Downward step size")
    parser.add_argument("--yaw_step", type=float, default=15.0, help="Yaw step size")
    parser.add_argument("--sample_limit", type=int, default=-1, help="Limit number of samples to test (-1 for all)")
    
    args = parser.parse_args()
    
    # 确定根路径
    if args.root_path:
        root_path = Path(args.root_path)
    else:
        root_path = _repo_root()
    
    # 未指定时设置默认文件路径
    if args.verification_file is None:
        args.verification_file = str(root_path / "data" / "verification" / "samples.jsonl")
    if args.output_file is None:
        args.output_file = str(root_path / "data" / "verification" / "results.json")
    
    # 加载验证样本
    verification_file = Path(args.verification_file)
    if not verification_file.exists():
        print(f"Error: Verification file not found: {verification_file}")
        return 1
    
    samples = read_jsonl(verification_file)
    if not samples:
        print(f"Error: No samples found in {verification_file}")
        return 1
    
    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]
    
    print(f"Loaded {len(samples)} verification samples")
    
    # 加载模型
    print("Loading model...")
    model = Qwen3VLWrapper(args.model_dir).load()
    
    # 连接 DroneSim
    print("Connecting to DroneSim...")
    cli = DroneSimClient(host=args.host, port=int(args.port))
    
    print(f"Waiting {args.sleep_s} seconds...")
    time.sleep(float(args.sleep_s))
    
    movement_params = build_movement_params(args)
    results = []
    
    try:
        cli.set_time(12, 0, 0)
        cam_id = cli.create_camera()
        if args.fov is not None:
            cli.set_fov(float(args.fov))
        
        print(f"Camera initialized: cam_id={cam_id}")
        
        for i, sample in enumerate(samples):
            print(f"\nProgress: {i+1}/{len(samples)}")
            result = run_single_verification(cli, model, sample, args.max_steps, movement_params)
            results.append(result)
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
    except Exception as e:
        print(f"\nError during verification: {e}")
    finally:
        try:
            cli.stop_camera()
            time.sleep(1.0)
            cli.restore_player()
            time.sleep(1.0)
            print("Player restored to normal state")
        except Exception:
            pass
    
    if results:
        output_file = Path(args.output_file)
        summary_stats = calculate_summary_stats(results)
        result_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(samples),
            "completed_samples": len(results),
            "parameters": {
                "max_steps": args.max_steps,
                "forward_step": movement_params["forward_step"],
                "up_step": movement_params["up_step"],
                "down_step": movement_params["down_step"],
                "yaw_step": movement_params["yaw_step"],
            },
            "summary_statistics": summary_stats,
            "results": results
        }

        write_json(output_file, result_data)
        print(f"\nResults saved to: {output_file}")
        print_summary(results)
    else:
        print("\nNo results to save.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
