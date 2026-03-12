import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image


_IMG_RE = re.compile(r"^(?P<traj>.+?)_step_(?P<step>\d+)_rgb\.(jpg|jpeg|png)$", re.I)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _ensure_parent(p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_done_keys(jsonl_path):
    done = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            if isinstance(key, str) and key:
                done.add(key)
    return done


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


def _awareness_prompt(task_line, context_text):
    return (
        "You are an autonomous exploration drone operating in a city environment.\n"
        "You will be given the current RGB image and a depth visualization, plus a short trajectory context.\n"
        "Write an awareness note that reflects your internal state and reasoning, taking the whole exploration trajectory into account.\n"
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
        "\n"
        f"{context_text}"
    )


def _parse_awareness_struct(text):
    s = str(text or "").strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) < 4:
        return None
    out = {}
    for ln in lines:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in ("task", "history", "observation & inference", "plan"):
            out[k] = v
    if len(out) != 4:
        return None
    return out


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


def _coerce_awareness(task_line, model_text):
    history = _extract_prefixed_line(model_text, "History:")
    obs = _extract_prefixed_line(model_text, "Observation & Inference:")
    plan = _extract_prefixed_line(model_text, "Plan:")
    if history is None:
        history = "History: I have been exploring and have not found the target yet (History)."
    if obs is None:
        obs = "Observation & Inference: I see urban structures and road-like regions; the target is likely near roads (Observation & Inference)."
    if plan is None:
        plan = "Plan: Continue exploring along roads and scan intersections for cues (Plan)."
    return "\n".join([task_line, history, obs, plan])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
    )
    parser.add_argument(
        "--output_json",
        default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"),
    )
    parser.add_argument(
        "--output_jsonl",
        default=str(_repo_root() / "dataset" / "awareness_labels.jsonl"),
    )
    parser.add_argument(
        "--model_dir",
        default=str(_repo_root() / "agent_control" / "qwen3_vl_sft_merged"),
    )
    parser.add_argument("--history_k", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    args = parser.parse_args()

    if float(args.sleep_s) > 0:
        time.sleep(float(args.sleep_s))

    root = _repo_root()
    agent_control_dir = root / "agent_control"
    sys.path.insert(0, str(agent_control_dir))
    from qwen3vl_wrapper import Qwen3VLWrapper

    model = Qwen3VLWrapper(args.model_dir).load()

    input_json = Path(args.input_json)
    output_json = _ensure_parent(args.output_json)
    output_jsonl = _ensure_parent(args.output_jsonl)

    data = _read_json(input_json)
    if not isinstance(data, list):
        raise RuntimeError("input_json 不是 list")

    done = _load_done_keys(output_jsonl)

    groups = {}
    for i, entry in enumerate(data):
        traj, step = _parse_traj_step(entry)
        if traj is None:
            traj = "__unknown__"
            step = i
        groups.setdefault(traj, []).append((step, i))

    traj_order = sorted(groups.items(), key=lambda kv: kv[0])

    for traj, step_pairs in traj_order:
        step_pairs.sort(key=lambda x: x[0])
        ordered_indices = [idx for _, idx in step_pairs]
        traj_entries = [data[idx] for idx in ordered_indices]
        task_desc = _extract_task_desc(traj_entries[0]) if traj_entries else "find the closest burning car"
        task_line = f"Task: I need to {task_desc} (Task)."

        for local_idx, global_idx in enumerate(ordered_indices):
            entry = data[global_idx]
            key = f"{traj}::{local_idx}"
            if key in done:
                continue

            images = entry.get("images") or []
            if len(images) < 2:
                continue

            rgb_path = root / "dataset" / str(images[0])
            depth_path = root / "dataset" / str(images[1])
            if not rgb_path.exists() or not depth_path.exists():
                continue

            rgb_img = Image.open(rgb_path).convert("RGB")
            depth_img = Image.open(depth_path).convert("RGB")

            context = _build_context_for_step(traj_entries, local_idx, int(args.history_k))
            prompt = _awareness_prompt(task_line, context)

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
            awareness = _coerce_awareness(task_line, awareness)

            entry["awareness"] = awareness
            aw_struct = _parse_awareness_struct(awareness)
            if aw_struct is not None:
                entry["awareness_struct"] = aw_struct

            _append_jsonl(
                output_jsonl,
                {
                    "key": key,
                    "traj": traj,
                    "task": task_desc,
                    "local_step": int(local_idx),
                    "global_index": int(global_idx),
                    "images": [str(images[0]), str(images[1])],
                    "awareness": awareness,
                },
            )
            done.add(key)

    _write_json(output_json, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
