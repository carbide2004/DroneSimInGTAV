import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_traj_step(entry):
    if "trajectory_id" not in entry:
        raise KeyError("Missing required field: trajectory_id")
    if "step_index" not in entry:
        raise KeyError("Missing required field: step_index")
    traj = str(entry["trajectory_id"]).strip()
    if not traj:
        raise ValueError("trajectory_id must be non-empty")
    return traj, int(entry["step_index"])


def _get_action(entry):
    action = entry.get("action")
    if not isinstance(action, dict):
        raise KeyError("Missing required field: action")
    name = str(action.get("name", "")).strip()
    if not name:
        raise ValueError("action.name must be non-empty")
    return name


def _pose_text(entry):
    pose = entry.get("pose")
    if not isinstance(pose, dict):
        raise KeyError("Missing required field: pose")
    return (
        f"x={float(pose['x']):.2f}, "
        f"y={float(pose['y']):.2f}, "
        f"z={float(pose['z']):.2f}, "
        f"rz={float(pose['rz'])}°."
    )


def _summarize_action_seq(actions, max_items):
    actions = [a for a in actions if isinstance(a, str) and a]
    if not actions:
        return "None"
    tail = actions[-max_items:]
    return ", ".join(tail)


def _build_context_for_step(traj_entries, idx, history_k):
    total = len(traj_entries)
    entry = traj_entries[idx]
    prev_actions = [_get_action(e) for e in traj_entries[:idx]]
    prev_actions = [a for a in prev_actions if a]
    recent = _summarize_action_seq(prev_actions, history_k)
    explored = idx
    pose_line = _pose_text(entry)
    context = (
        f"Trajectory: You are at step {idx} out of {max(total - 1, 0)} (0-indexed). "
        f"You have executed {explored} actions so far.\n"
        f"Recent actions: {recent}.\n"
    )
    if pose_line:
        context += f"Pose: {pose_line}\n"
    return context


def _extract_task_desc(entry):
    t = entry.get("task")
    if not isinstance(t, str) or not t.strip():
        raise KeyError("Missing required field: task")
    return t.strip().rstrip(".")


def _resolve_rgb_path(root, entry):
    observations = entry.get("observations")
    if not isinstance(observations, dict):
        raise KeyError("Missing required field: observations")
    rgb_info = observations.get("rgb")
    if not isinstance(rgb_info, dict):
        raise KeyError("Missing required field: observations.rgb")
    rgb_path = str(rgb_info.get("path", "")).strip()
    if not rgb_path:
        raise ValueError("observations.rgb.path must be non-empty")
    return root / "dataset" / rgb_path


def _awareness_prompt(task_line, context_text, current_action):
    return (
        "You are an autonomous exploration drone operating in a city environment.\n"
        "You will be given the current RGB image and a short trajectory context.\n"
        "Write an awareness note that reflects your internal state and reasoning, taking the whole exploration trajectory into account.\n"
        "\n"
        f"Current Action: You are executing '{current_action}' in this frame.\n"
        "\n"
        "Output must be in English and exactly four lines, each starting with the given prefix:\n"
        f"{task_line}\n"
        "History: ...\n"
        "Observation & Inference: ...\n"
        "Plan: ...\n"
        "\n"
        "Constraints:\n"
        "- Keep each line concise (no more than ~25 words).\n"
        "- Do not include bullet points, numbering, or extra lines.\n"
        "- Do not mention model, prompt, or formatting instructions.\n"
        "- The Task line must be exactly the one provided.\n"
        "- The Plan should be consistent with or explain the current action being executed.\n"
        "\n"
        f"{context_text}"
    )


def _extract_prefixed_line(text, prefix):
    if text is None:
        return None
    for ln in str(text).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith(prefix.lower()):
            return ln
    return None


def _coerce_awareness(task_line, model_text, current_action):
    history = _extract_prefixed_line(model_text, "History:")
    obs = _extract_prefixed_line(model_text, "Observation & Inference:")
    plan = _extract_prefixed_line(model_text, "Plan:")
    if history is None:
        history = "History: I have been exploring and have not found the target yet."
    if obs is None:
        obs = "Observation & Inference: I see urban structures and road-like regions; the target is likely near roads."
    if plan is None:
        plan = f"Plan: Executing {current_action} to continue systematic exploration of the area."
    return "\n".join([task_line, history, obs, plan])


def main():
    parser = argparse.ArgumentParser(description="Add awareness annotations to train_data_all.json")
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output_json",
        default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model_dir",
        default=str(_repo_root() / "agent_control" / "models" / "qwen3_vl_sft_merged"),
        help="Model directory path"
    )
    parser.add_argument("--history_k", type=int, default=12, help="Number of recent actions to include in context")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Maximum new tokens for generation")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Sleep time before starting")
    parser.add_argument("--no_skip_existing", action="store_true", help="Process entries that already have awareness field (default: skip existing)")
    parser.add_argument("--no_extract_vectors", action="store_true", help="Disable extraction of representation vectors (default: extract vectors)")
    parser.add_argument("--no_skip_vector_existing", action="store_true", help="Process vector extraction for entries that already have representation_vector field (default: skip existing vectors)")
    
    args = parser.parse_args()
    
    # Set default values to True, but allow override with --no_* flags
    args.skip_existing = not args.no_skip_existing
    args.extract_vectors = not args.no_extract_vectors
    args.skip_vector_existing = not args.no_skip_vector_existing

    if float(args.sleep_s) > 0:
        time.sleep(float(args.sleep_s))

    # Import model wrapper
    root = _repo_root()
    agent_control_dir = root / "agent_control"
    sys.path.insert(0, str(agent_control_dir))
    from qwen3vl_wrapper import Qwen3VLWrapper

    # Load model
    print("Loading model...")
    model = Qwen3VLWrapper(args.model_dir).load()

    # Read input data
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Reading data from: {input_path}")
    data = _read_json(input_path)
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"第 {i} 条样本不是对象")
        schema_version = entry.get("schema_version")
        if int(schema_version) != 2:
            raise RuntimeError(f"第 {i} 条样本 schema_version 不是 2")
        _parse_traj_step(entry)
        _get_action(entry)
        _pose_text(entry)
        _extract_task_desc(entry)
        _resolve_rgb_path(root, entry)

    # Group entries by trajectory
    groups = {}
    for i, entry in enumerate(data):
        traj, step = _parse_traj_step(entry)
        groups.setdefault(traj, []).append((step, i))

    # Sort trajectories and steps
    traj_order = sorted(groups.items(), key=lambda kv: kv[0])
    
    # Count total entries and already processed
    total_entries = len(data)
    if args.skip_existing:
        already_processed = sum(1 for entry in data if "awareness" in entry)
        print(f"Found {already_processed} entries already processed")
    else:
        already_processed = 0

    print(f"Processing {total_entries} entries...")
    
    with tqdm(total=total_entries, initial=already_processed, desc="Processing entries", unit="entry") as pbar:
        for traj, step_pairs in traj_order:
            step_pairs.sort(key=lambda x: x[0])
            ordered_indices = [idx for _, idx in step_pairs]
            traj_entries = [data[idx] for idx in ordered_indices]
            
            # Extract task description from first entry
            task_desc = _extract_task_desc(traj_entries[0]) if traj_entries else "find the closest burning car"
            task_line = f"Task: I need to {task_desc} (Task)."

            for local_idx, global_idx in enumerate(ordered_indices):
                entry = data[global_idx]
                
                # Skip if already has awareness and skip_existing is True
                if args.skip_existing and "awareness" in entry:
                    pbar.update(1)
                    continue
                
                # Skip if already has vector and skip_vector_existing is True
                if args.skip_vector_existing and "representation_vector" in entry:
                    pbar.update(1)
                    continue

                rgb_path = _resolve_rgb_path(root, entry)
                if not rgb_path.exists():
                    raise FileNotFoundError(f"RGB image not found: {rgb_path}")

                rgb_img = Image.open(rgb_path).convert("RGB")

                current_action = _get_action(entry)

                context = _build_context_for_step(traj_entries, local_idx, int(args.history_k))
                prompt = _awareness_prompt(task_line, context, current_action)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    }
                ]

                if args.extract_vectors:
                    awareness, representation_vector = model.generate_chat_with_representation(
                        messages=messages,
                        images=[rgb_img],
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        normalize_vector=True,
                    )
                    awareness = _coerce_awareness(task_line, awareness, current_action)

                    entry["awareness"] = awareness
                    entry["representation_vector"] = representation_vector
                    entry["vector_dim"] = len(representation_vector)
                else:
                    awareness = model.generate_chat(
                        messages=messages,
                        images=[rgb_img],
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                    )
                    awareness = _coerce_awareness(task_line, awareness, current_action)

                    entry["awareness"] = awareness
                pbar.update(1)

    # Write output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing results to: {output_path}")
    _write_json(output_path, data)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
