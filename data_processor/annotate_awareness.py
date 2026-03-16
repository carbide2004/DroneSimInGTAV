import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm


_IMG_RE = re.compile(r"^(?P<traj>.+?)_step_(?P<step>\d+)_rgb\.(jpg|jpeg|png)$", re.I)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_traj_step(entry):
    images = entry.get("images") or []
    if not images:
        return None, None
    p0 = Path(str(images[0]).replace("\\", "/"))
    name = p0.name
    m = _IMG_RE.match(name)
    if not m:
        return None, None
    traj = m.group("traj")
    step = int(m.group("step"))
    return traj, step


def _get_action(entry):
    msgs = entry.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            s = str(m.get("content", "")).strip()
            if s:
                return s
    return None


def _pose_text(entry):
    msgs = entry.get("messages") or []
    if not msgs:
        return ""
    user = None
    for m in msgs:
        if m.get("role") == "user":
            user = m
            break
    if user is None:
        return ""
    content = user.get("content", "")
    if not isinstance(content, str):
        return ""
    m = re.search(r"Current Pose:\s*(.+)", content)
    if not m:
        return ""
    return m.group(1).strip()


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
    if isinstance(t, str) and t.strip():
        return t.strip().rstrip(".")

    msgs = entry.get("messages") or []
    for m in msgs:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        mm = re.search(r"Your current task is to\s+(.+?)\.\s*$", content, re.MULTILINE)
        if mm:
            return mm.group(1).strip().rstrip(".")
    return "find the closest burning car"


def _awareness_prompt(task_line, context_text, current_action):
    return (
        "You are an autonomous exploration drone operating in a city environment.\n"
        "You will be given the current RGB image and a depth visualization, plus a short trajectory context.\n"
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
    parser.add_argument("--skip_existing", action="store_true", help="Skip entries that already have awareness field")
    
    args = parser.parse_args()

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

    # Group entries by trajectory
    groups = {}
    for i, entry in enumerate(data):
        traj, step = _parse_traj_step(entry)
        if traj is None:
            traj = f"__unknown_{i}__"
            step = 0
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

                images = entry.get("images") or []
                if len(images) < 2:
                    pbar.update(1)
                    continue

                rgb_path = root / "dataset" / str(images[0])
                depth_path = root / "dataset" / str(images[1])
                if not rgb_path.exists() or not depth_path.exists():
                    pbar.update(1)
                    continue

                # Load images
                rgb_img = Image.open(rgb_path).convert("RGB")
                depth_img = Image.open(depth_path).convert("RGB")

                # Get current action
                current_action = _get_action(entry)
                if not current_action:
                    current_action = "Unknown action"

                # Build context
                context = _build_context_for_step(traj_entries, local_idx, int(args.history_k))
                prompt = _awareness_prompt(task_line, context, current_action)

                # Generate awareness
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                            {"type": "image"},
                        ],
                    }
                ]

                awareness = model.generate_chat(
                    messages=messages,
                    images=[rgb_img, depth_img],
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                )
                awareness = _coerce_awareness(task_line, awareness, current_action)

                # Add awareness to entry
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
