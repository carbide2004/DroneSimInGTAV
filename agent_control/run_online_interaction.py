import argparse
import json
import math
import time
from pathlib import Path

from dronesim_client import DroneSimClient

from action_mapping import dispatch_action, parse_action
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from rgbd_utils import depth_bytes_to_pil, rgb_bytes_to_pil


def _ensure_parent(path):
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument(
        "--model_dir",
        default=str(Path(__file__).resolve().parent / "qwen3_vl_sft_merged"),
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--sleep_s", type=float, default=5.0)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--forward_step", type=float, default=1.0)
    parser.add_argument("--down_step", type=float, default=1.0)
    parser.add_argument("--yaw_step", type=float, default=15.0)
    parser.add_argument("--log_jsonl", default=None)
    args = parser.parse_args()

    if args.log_jsonl:
        log_path = _ensure_parent(args.log_jsonl)
        if log_path.exists():
            log_path.unlink()
    else:
        log_path = None

    model = Qwen3VLWrapper(args.model_dir).load()
    cli = DroneSimClient(host=args.host, port=int(args.port))

    print(f"模型加载完成，等待{args.sleep_s}秒...")
    time.sleep(float(args.sleep_s))

    fire = None
    last_pose = None
    stopped_by_model = False
    steps = 0

    try:
        cli.set_time(12, 0, 0)
        cam_id = cli.create_camera()
        if args.fov is not None:
            cli.set_fov(float(args.fov))

        print(f"相机已进入模式 cam_id={cam_id}")

        fire = cli.create_fire()
        if fire is None:
            time.sleep(0.5)
            fire = cli.create_fire()
        if fire is None:
            raise RuntimeError("create_fire() 失败")
        fire_x, fire_y, fire_z, fire_id = fire
        print(
            f"已创建火灾 fire_id={fire_id} pos=({fire_x:.2f},{fire_y:.2f},{fire_z:.2f})"
        )

        while steps < int(args.max_steps):
            pose = cli.get_pose()
            if pose is None:
                time.sleep(0.2)
                pose = cli.get_pose()
            if pose is None:
                raise RuntimeError("get_pose() 失败")
            last_pose = pose

            cap = cli.capture()
            if cap is None:
                time.sleep(0.2)
                cap = cli.capture()
            if cap is None:
                raise RuntimeError("capture() 失败")

            w, h, rgb_bytes, depth_bytes = cap
            rgb_pil = rgb_bytes_to_pil(w, h, rgb_bytes)
            depth_pil = depth_bytes_to_pil(w, h, depth_bytes)

            x, y, z, rx, ry, rz = pose
            prompt = build_prompt(x, y, z, rz)
            raw = model.generate_action(prompt, rgb_pil, depth_pil)
            action = parse_action(raw)
            if action is None:
                action = "AUTO_FORWARD"

            if log_path:
                dist = math.sqrt(
                    (float(x) - float(fire_x)) ** 2
                    + (float(y) - float(fire_y)) ** 2
                    + (float(z) - float(fire_z)) ** 2
                )
                _append_jsonl(
                    log_path,
                    {
                        "step": int(steps),
                        "fire": {
                            "x": float(fire_x),
                            "y": float(fire_y),
                            "z": float(fire_z),
                            "id": int(fire_id),
                        },
                        "pose": {
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "rx": float(rx),
                            "ry": float(ry),
                            "rz": float(rz),
                        },
                        "model_raw": str(raw),
                        "action": str(action),
                        "distance_to_fire": float(dist),
                        "w": int(w),
                        "h": int(h),
                    },
                )

            print(f"[{steps}] model_raw={raw!r} -> action={action}")
            if action == "AUTO_STOP_REACHED":
                stopped_by_model = True
                break

            dispatch_action(
                cli,
                action,
                forward_step=float(args.forward_step),
                up_step=float(args.down_step),
                down_step=float(args.down_step),
                yaw_step=float(args.yaw_step),
            )
            steps += 1

    finally:
        try:
            cli.stop_camera()
        except Exception:
            pass

    if fire is None:
        print("探索失败：未能创建火灾")
        return 2

    if last_pose is None:
        last_pose = cli.get_pose()

    if last_pose is None:
        print("探索失败：无法获取最终pose")
        return 2

    fire_x, fire_y, fire_z, fire_id = fire
    x, y, z, rx, ry, rz = last_pose
    dist = math.sqrt(
        (float(x) - float(fire_x)) ** 2
        + (float(y) - float(fire_y)) ** 2
        + (float(z) - float(fire_z)) ** 2
    )

    if dist < 20.0:
        print(f"探索成功：distance_to_fire={dist:.2f} < 20")
        return 0

    if stopped_by_model:
        print(f"探索失败：已停止但distance_to_fire={dist:.2f} >= 20")
        return 2

    print(f"探索失败：超出max_steps且distance_to_fire={dist:.2f} >= 20")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
