import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

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
        f"rz={float(pose['rz']):.2f} deg."
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


def _resolve_rgbd_paths(root, entry):
    observations = entry.get("observations")
    if not isinstance(observations, dict):
        raise KeyError("Missing required field: observations")
    rgb_info = observations.get("rgb")
    depth_info = observations.get("depth")
    if not isinstance(rgb_info, dict) or not isinstance(depth_info, dict):
        raise KeyError("Missing required field: observations.rgb/depth")
    rgb_path = str(rgb_info.get("path", "")).strip()
    depth_path = str(depth_info.get("path", "")).strip()
    if not rgb_path or not depth_path:
        raise ValueError("observations.rgb.path/depth.path must be non-empty")
    return root / "dataset" / rgb_path, root / "dataset" / depth_path


def _awareness_prompt(task_line, context_text, current_action):
    return (
        "You are an autonomous exploration drone operating in a city environment.\n"
        "You will be given the current RGB image, a depth visualization, and a short trajectory context.\n"
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


def _validate_entries(data, root):
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Entry at index {i} must be a JSON object")
        schema_version = entry.get("schema_version")
        if int(schema_version) != 2:
            raise RuntimeError(f"Entry at index {i} must have schema_version == 2")
        _parse_traj_step(entry)
        _get_action(entry)
        _pose_text(entry)
        _extract_task_desc(entry)
        _resolve_rgbd_paths(root, entry)


def _should_generate(entry, skip_existing, extract_vectors, skip_vector_existing):
    has_awareness = "awareness" in entry
    has_vector = "representation_vector" in entry
    need_awareness = not (skip_existing and has_awareness)
    need_vector = bool(extract_vectors) and not (skip_vector_existing and has_vector)
    return need_awareness or need_vector, need_awareness, need_vector


def _group_entries_by_trajectory(data):
    groups = {}
    for i, entry in enumerate(data):
        traj, step = _parse_traj_step(entry)
        groups.setdefault(traj, []).append((step, i))
    return groups


def _build_generation_tasks(data, root, history_k, skip_existing, extract_vectors, skip_vector_existing):
    tasks = []
    groups = _group_entries_by_trajectory(data)
    traj_order = sorted(groups.items(), key=lambda kv: kv[0])

    for traj, step_pairs in traj_order:
        step_pairs.sort(key=lambda x: x[0])
        ordered_indices = [idx for _, idx in step_pairs]
        traj_entries = [data[idx] for idx in ordered_indices]
        task_desc = _extract_task_desc(traj_entries[0]) if traj_entries else "find the closest burning car"
        task_line = f"Task: I need to {task_desc} (Task)."

        trajectory_tasks = []
        for local_idx, global_idx in enumerate(ordered_indices):
            entry = data[global_idx]
            should_generate, need_awareness, need_vector = _should_generate(
                entry=entry,
                skip_existing=skip_existing,
                extract_vectors=extract_vectors,
                skip_vector_existing=skip_vector_existing,
            )
            if not should_generate:
                continue

            rgb_path, depth_path = _resolve_rgbd_paths(root, entry)
            context = _build_context_for_step(traj_entries, local_idx, int(history_k))
            current_action = _get_action(entry)
            trajectory_tasks.append({
                "global_idx": global_idx,
                "task_line": task_line,
                "current_action": current_action,
                "prompt": _awareness_prompt(task_line, context, current_action),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "need_awareness": bool(need_awareness),
                "need_vector": bool(need_vector),
            })

        if trajectory_tasks:
            tasks.append({
                "trajectory_id": traj,
                "num_steps": len(trajectory_tasks),
                "items": trajectory_tasks,
            })
    return tasks


def _parse_gpu_ids(gpu_ids_text):
    gpu_ids = []
    for part in str(gpu_ids_text).split(","):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id")
    return gpu_ids


def _assign_trajectories_to_workers(trajectory_tasks, worker_count):
    buckets = [{"total_steps": 0, "trajectories": []} for _ in range(int(worker_count))]
    for task in sorted(trajectory_tasks, key=lambda item: item["num_steps"], reverse=True):
        target = min(buckets, key=lambda item: item["total_steps"])
        target["trajectories"].append(task)
        target["total_steps"] += int(task["num_steps"])
    return [bucket["trajectories"] for bucket in buckets if bucket["trajectories"]]


def _load_rgbd_images(rgb_path, depth_path):
    rgb_file = Path(rgb_path)
    depth_file = Path(depth_path)
    if not rgb_file.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_file}")
    if not depth_file.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_file}")
    with Image.open(rgb_file) as rgb_src:
        rgb_img = rgb_src.convert("RGB")
    with Image.open(depth_file) as depth_src:
        depth_img = depth_src.convert("RGB")
    return rgb_img, depth_img


def _load_model(model_dir, gpu_id):
    root = _repo_root()
    agent_control_dir = root / "agent_control"
    if str(agent_control_dir) not in sys.path:
        sys.path.insert(0, str(agent_control_dir))
    from qwen3vl_wrapper import Qwen3VLWrapper

    device_map = {"": f"cuda:{int(gpu_id)}"}
    return Qwen3VLWrapper(model_dir, torch_dtype="auto", device_map=device_map).load()


def _run_model_for_item(model, item, max_new_tokens):
    rgb_img, depth_img = _load_rgbd_images(item["rgb_path"], item["depth_path"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": item["prompt"]},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]

    if item["need_vector"]:
        awareness, representation_vector = model.generate_chat_with_representation(
            messages=messages,
            images=[rgb_img, depth_img],
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            normalize_vector=True,
        )
        return {
            "awareness": _coerce_awareness(item["task_line"], awareness, item["current_action"]),
            "representation_vector": representation_vector,
            "vector_dim": len(representation_vector),
        }

    awareness = model.generate_chat(
        messages=messages,
        images=[rgb_img, depth_img],
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    return {
        "awareness": _coerce_awareness(item["task_line"], awareness, item["current_action"]),
    }


def _process_worker(worker_id, gpu_id, model_dir, assigned_trajectories, max_new_tokens, progress_queue=None):
    model = _load_model(model_dir=model_dir, gpu_id=gpu_id)
    updates = []
    processed = 0
    if progress_queue is not None:
        progress_queue.put({
            "type": "worker_loaded",
            "worker_id": int(worker_id),
            "gpu_id": int(gpu_id),
        })

    for trajectory in assigned_trajectories:
        if progress_queue is not None:
            progress_queue.put({
                "type": "trajectory_started",
                "worker_id": int(worker_id),
                "gpu_id": int(gpu_id),
                "trajectory_id": trajectory["trajectory_id"],
                "num_steps": int(trajectory["num_steps"]),
            })
        for item in trajectory["items"]:
            result = _run_model_for_item(model=model, item=item, max_new_tokens=max_new_tokens)
            update = {"global_idx": int(item["global_idx"])}
            if item["need_awareness"]:
                update["awareness"] = result["awareness"]
            if item["need_vector"]:
                update["representation_vector"] = result["representation_vector"]
                update["vector_dim"] = int(result["vector_dim"])
            updates.append(update)
            processed += 1
            if progress_queue is not None:
                progress_queue.put({
                    "type": "step_done",
                    "worker_id": int(worker_id),
                    "gpu_id": int(gpu_id),
                    "trajectory_id": trajectory["trajectory_id"],
                    "processed": int(processed),
                })

    return {
        "worker_id": int(worker_id),
        "gpu_id": int(gpu_id),
        "processed": int(processed),
        "updates": updates,
    }


def _apply_updates(data, worker_result):
    for update in worker_result["updates"]:
        entry = data[int(update["global_idx"])]
        if "awareness" in update:
            entry["awareness"] = update["awareness"]
        if "representation_vector" in update:
            entry["representation_vector"] = update["representation_vector"]
            entry["vector_dim"] = int(update["vector_dim"])


def _run_parallel_generation(trajectory_tasks, args):
    worker_count = max(1, len(args.gpu_ids) * int(args.workers_per_gpu))
    assignments = _assign_trajectories_to_workers(trajectory_tasks, worker_count)
    futures = []
    results = []
    total_steps = sum(task["num_steps"] for task in trajectory_tasks)
    worker_states = {}
    mp_context = get_context("spawn")
    with mp_context.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=len(assignments), mp_context=mp_context) as executor:
            for worker_id, assigned_trajectories in enumerate(assignments):
                gpu_id = args.gpu_ids[worker_id % len(args.gpu_ids)]
                futures.append(
                    executor.submit(
                        _process_worker,
                        worker_id,
                        gpu_id,
                        args.model_dir,
                        assigned_trajectories,
                        args.max_new_tokens,
                        progress_queue,
                    )
                )
            pending = set(futures)
            with tqdm(total=total_steps, desc="Generating awareness", unit="entry") as pbar:
                while pending:
                    try:
                        message = progress_queue.get(timeout=0.5)
                        _handle_progress_message(message, worker_states, pbar)
                    except Empty:
                        pass

                    finished = [future for future in pending if future.done()]
                    for future in finished:
                        result = future.result()
                        results.append(result)
                        pending.remove(future)

                _drain_progress_queue(progress_queue, worker_states, pbar)

    return results


def _run_single_worker_generation(trajectory_tasks, args):
    return _run_parallel_generation(trajectory_tasks, args)


def _format_worker_state(worker_states):
    if not worker_states:
        return "waiting for workers"
    parts = []
    for worker_id in sorted(worker_states.keys()):
        state = worker_states[worker_id]
        gpu_id = state.get("gpu_id", "?")
        processed = state.get("processed", 0)
        status = state.get("status", "starting")
        trajectory_id = state.get("trajectory_id")
        if trajectory_id:
            parts.append(f"w{worker_id}/g{gpu_id}:{processed} {status} {trajectory_id}")
        else:
            parts.append(f"w{worker_id}/g{gpu_id}:{processed} {status}")
    return " | ".join(parts)


def _handle_progress_message(message, worker_states, pbar):
    msg_type = message.get("type")
    worker_id = int(message.get("worker_id", -1))
    state = worker_states.setdefault(worker_id, {"processed": 0})
    if "gpu_id" in message:
        state["gpu_id"] = int(message["gpu_id"])

    if msg_type == "worker_loaded":
        state["status"] = "loaded"
    elif msg_type == "trajectory_started":
        state["status"] = "running"
        state["trajectory_id"] = str(message.get("trajectory_id", ""))
        state["trajectory_steps"] = int(message.get("num_steps", 0))
    elif msg_type == "step_done":
        state["status"] = "running"
        state["trajectory_id"] = str(message.get("trajectory_id", ""))
        new_processed = int(message.get("processed", state.get("processed", 0)))
        delta = max(0, new_processed - int(state.get("processed", 0)))
        state["processed"] = new_processed
        if delta > 0:
            pbar.update(delta)

    pbar.set_postfix_str(_format_worker_state(worker_states), refresh=False)


def _drain_progress_queue(progress_queue, worker_states, pbar):
    while True:
        try:
            message = progress_queue.get_nowait()
            _handle_progress_message(message, worker_states, pbar)
        except Empty:
            break


def main():
    parser = argparse.ArgumentParser(description="Add awareness annotations to train_data_all.json")
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output_json",
        default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model_dir",
        default=str(_repo_root() / "agent_control" / "models" / "qwen3_vl_sft_merged"),
        help="Model directory path",
    )
    parser.add_argument("--history_k", type=int, default=12, help="Number of recent actions to include in context")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Maximum new tokens for generation")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Sleep time before starting")
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU ids, for example: 0,1,2,3")
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of independent worker processes to launch per GPU",
    )
    parser.add_argument("--no_skip_existing", action="store_true", help="Process entries that already have awareness field (default: skip existing)")
    parser.add_argument("--no_extract_vectors", action="store_true", help="Disable extraction of representation vectors (default: extract vectors)")
    parser.add_argument("--no_skip_vector_existing", action="store_true", help="Process vector extraction for entries that already have representation_vector field (default: skip existing vectors)")

    args = parser.parse_args()

    args.skip_existing = not args.no_skip_existing
    args.extract_vectors = not args.no_extract_vectors
    args.skip_vector_existing = not args.no_skip_vector_existing
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)
    args.workers_per_gpu = max(1, int(args.workers_per_gpu))

    if float(args.sleep_s) > 0:
        time.sleep(float(args.sleep_s))

    root = _repo_root()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading data from: {input_path}")
    data = _read_json(input_path)
    _validate_entries(data, root)

    trajectory_tasks = _build_generation_tasks(
        data=data,
        root=root,
        history_k=args.history_k,
        skip_existing=args.skip_existing,
        extract_vectors=args.extract_vectors,
        skip_vector_existing=args.skip_vector_existing,
    )

    total_entries = len(data)
    total_to_generate = sum(task["num_steps"] for task in trajectory_tasks)
    skipped_entries = total_entries - total_to_generate

    print(f"Total entries: {total_entries}")
    print(f"Entries to generate: {total_to_generate}")
    print(f"Entries skipped: {skipped_entries}")
    print(f"Using GPUs: {args.gpu_ids}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Total worker processes: {len(args.gpu_ids) * args.workers_per_gpu}")

    if total_to_generate > 0:
        worker_results = _run_parallel_generation(trajectory_tasks, args)
        for result in worker_results:
            _apply_updates(data, result)
    else:
        print("Nothing to generate.")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing results to: {output_path}")
    _write_json(output_path, data)

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
