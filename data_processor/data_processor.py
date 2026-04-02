import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ACTION_NAMES = (
    "AUTO_DOWN",
    "AUTO_UP",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
)

ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}

def bin_to_image(bin_path, width, height, is_depth=False):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing bin file: {bin_path}")

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid width/height: {width}x{height} for {bin_path}")

    raw = bin_path.read_bytes()
    if is_depth:
        expected_f32 = width * height * 4
        expected_u8 = width * height
        if len(raw) == expected_f32:
            img_array = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
        elif len(raw) == expected_u8:
            img_array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width)).astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected depth bin size: {len(raw)} bytes, expected {expected_f32} (f32) or {expected_u8} (u8). file={bin_path}"
            )
    else:
        expected_rgba = width * height * 4
        expected_rgb = width * height * 3
        if len(raw) == expected_rgba:
            img_array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        elif len(raw) == expected_rgb:
            img_array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError(
                f"Unexpected rgb bin size: {len(raw)} bytes, expected {expected_rgba} (rgba) or {expected_rgb} (rgb). file={bin_path}"
            )
    
    if is_depth:
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        img_array = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_array)


def build_prompt(task_desc, pose):
    return (
        f"Task: You are an outdoor exploration drone. Analyze the RGB and Depth observations to decide the next best move. Your current task is to {task_desc}.\n"
        f"Observations: <image><image>\n"
        f"Current Pose: x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}, rz={pose['rz']}°.\n"
        f"Action Set: [AUTO_DOWN, AUTO_UP, AUTO_FORWARD, AUTO_YAW_LEFT, AUTO_YAW_RIGHT, AUTO_STOP_REACHED].\n"
        f"Requirement: You must output only one string from the action set.\n"
        f"Decision:"
    )


def build_pose_entry(pose):
    return {
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "z": float(pose.get("z", 0.0)),
        "rx": float(pose.get("rx", 0.0)),
        "ry": float(pose.get("ry", 0.0)),
        "rz": float(pose.get("rz", 0.0)),
    }


def build_dataset_entry(item, task_desc, trajectory_id, trajectory_length, rgb_rel_path, depth_rel_path):
    step_index = int(item["step"])
    pose = build_pose_entry(item["pose"])
    action_name = str(item["action"]["name"]).strip()
    prompt = build_prompt(task_desc, pose)

    return {
        "schema_version": 2,
        "sample_id": f"{trajectory_id}:{step_index:06d}",
        "trajectory_id": trajectory_id,
        "trajectory_length": int(trajectory_length),
        "step_index": step_index,
        "task": task_desc,
        "pose": pose,
        "action": {
            "name": action_name,
        },
        "action_id": ACTION_TO_ID.get(action_name),
        "observations": {
            "rgb": {
                "path": rgb_rel_path,
                "width": int(item["rgb"]["width"]),
                "height": int(item["rgb"]["height"]),
                "source_bin_path": str(item["rgb"]["path"]).replace("\\", "/"),
            },
            "depth": {
                "path": depth_rel_path,
                "width": int(item["depth"]["width"]),
                "height": int(item["depth"]["height"]),
                "source_bin_path": str(item["depth"]["path"]).replace("\\", "/"),
            },
        },
        "images": [
            rgb_rel_path,
            depth_rel_path,
        ],
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": action_name,
            },
        ],
    }

def process_all_datasets(base_dir):
    base_path = Path(base_dir)
    all_conversations = []
    skipped = 0
    out_root = Path(__file__).resolve().parent.parent
    dataset_dir = out_root / "dataset"
    imgs_dir = dataset_dir / "imgs"
    dataset_dir.mkdir(exist_ok=True)
    imgs_dir.mkdir(exist_ok=True)
    
    for dt_dir in base_path.iterdir():
        if not dt_dir.is_dir():
            continue

        jsonl_path = dt_dir / "steps.jsonl"
        if not jsonl_path.exists():
            continue

        task_desc = "find the closest burning car"
        meta_path = dt_dir / "metadata.jsonl"
        if meta_path.exists():
            try:
                first = meta_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if first:
                    meta = json.loads(first[0])
                    t = meta.get("task_description")
                    if isinstance(t, str) and t.strip():
                        task_desc = t.strip().rstrip(".")
            except Exception:
                pass

        print(f"正在处理目录: {dt_dir.name}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            trajectory_items = [json.loads(line) for line in f if line.strip()]

        trajectory_length = len(trajectory_items)

        for item in trajectory_items:
            step_idx = int(item["step"])
            rgb_bin = dt_dir / item["rgb"]["path"]
            depth_bin = dt_dir / item["depth"]["path"]

            rgb_name = f"{dt_dir.name}_step_{step_idx:06d}_rgb.jpg"
            depth_name = f"{dt_dir.name}_step_{step_idx:06d}_depth.jpg"
            rgb_jpg_path = imgs_dir / rgb_name
            depth_jpg_path = imgs_dir / depth_name

            try:
                bin_to_image(rgb_bin, item["rgb"]["width"], item["rgb"]["height"]).save(rgb_jpg_path)
                bin_to_image(depth_bin, item["depth"]["width"], item["depth"]["height"], is_depth=True).save(depth_jpg_path)
            except Exception as e:
                skipped += 1
                print(f"跳过 step={step_idx}，原因: {e}")
                continue

            rgb_rel_path = (Path("imgs") / rgb_name).as_posix()
            depth_rel_path = (Path("imgs") / depth_name).as_posix()
            entry = build_dataset_entry(
                item=item,
                task_desc=task_desc,
                trajectory_id=dt_dir.name,
                trajectory_length=trajectory_length,
                rgb_rel_path=rgb_rel_path,
                depth_rel_path=depth_rel_path,
            )
            all_conversations.append(entry)

    with open(dataset_dir / "train_data_all.json", "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    print(f"转换完成，共计 {len(all_conversations)} 条数据，跳过 {skipped} 条。")

if __name__ == "__main__":
    process_all_datasets(r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V\data\manual\checked")
