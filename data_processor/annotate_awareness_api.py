import argparse
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from annotate_awareness import (
    _extract_prefixed_line,
    _read_json,
    _repo_root,
    _write_json,
)


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max-latest"
PROMPT_VERSION = "api_belief_v1"
LEGACY_IMAGE_RE = re.compile(r"(?P<traj>\d{8}_\d{6})_step_(?P<step>\d+)_")
POSE_RE = re.compile(
    r"Current Pose:\s*x=(?P<x>[-+0-9.]+),\s*y=(?P<y>[-+0-9.]+),\s*z=(?P<z>[-+0-9.]+),\s*rz=(?P<rz>[-+0-9.]+)",
    re.IGNORECASE,
)


def _safe_custom_id(global_idx):
    return f"entry-{int(global_idx)}"


def _parse_custom_id(custom_id):
    match = re.match(r"^entry-(\d+)$", str(custom_id))
    if not match:
        raise ValueError(f"Unsupported custom_id: {custom_id}")
    return int(match.group(1))


def _entry_task_desc(entry):
    task = entry.get("task")
    if not isinstance(task, str) or not task.strip():
        raise KeyError("Missing required field: task")
    return task.strip().rstrip(".")


def _legacy_user_text(entry):
    for message in entry.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _entry_action(entry):
    action = entry.get("action")
    if isinstance(action, dict):
        name = str(action.get("name", "")).strip()
        if name:
            return name

    for message in entry.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                return content.splitlines()[0].strip()
    raise KeyError("Missing required field: action.name or assistant message")


def _entry_pose_text(entry):
    pose = entry.get("pose")
    if isinstance(pose, dict):
        return (
            f"x={float(pose['x']):.2f}, "
            f"y={float(pose['y']):.2f}, "
            f"z={float(pose['z']):.2f}, "
            f"rz={float(pose['rz']):.2f} deg."
        )

    match = POSE_RE.search(_legacy_user_text(entry))
    if match:
        return (
            f"x={float(match.group('x')):.2f}, "
            f"y={float(match.group('y')):.2f}, "
            f"z={float(match.group('z')):.2f}, "
            f"rz={float(match.group('rz')):.2f} deg."
        )
    return "unknown"


def _entry_rgbd_paths(root, entry):
    observations = entry.get("observations")
    if isinstance(observations, dict):
        rgb_info = observations.get("rgb")
        depth_info = observations.get("depth")
        if isinstance(rgb_info, dict) and isinstance(depth_info, dict):
            rgb_path = str(rgb_info.get("path", "")).strip()
            depth_path = str(depth_info.get("path", "")).strip()
            if rgb_path and depth_path:
                return root / "dataset" / rgb_path, root / "dataset" / depth_path

    images = entry.get("images")
    if isinstance(images, list) and len(images) >= 2:
        rgb_path = str(images[0]).strip()
        depth_path = str(images[1]).strip()
        if rgb_path and depth_path:
            return root / "dataset" / rgb_path, root / "dataset" / depth_path

    raise KeyError("Missing required image paths: observations.rgb/depth or images[0:2]")


def _entry_traj_step(entry, fallback_index):
    if "trajectory_id" in entry and "step_index" in entry:
        traj = str(entry["trajectory_id"]).strip()
        if traj:
            return traj, int(entry["step_index"])

    images = entry.get("images")
    if isinstance(images, list) and images:
        match = LEGACY_IMAGE_RE.search(str(images[0]).replace("\\", "/"))
        if match:
            return match.group("traj"), int(match.group("step"))

    return "legacy_trajectory", int(fallback_index)


def _group_entries(data):
    groups = {}
    for i, entry in enumerate(data):
        traj, step = _entry_traj_step(entry, i)
        groups.setdefault(traj, []).append((step, i))
    return groups


def _summarize_action_seq(actions, max_items):
    actions = [a for a in actions if isinstance(a, str) and a]
    if not actions:
        return "None"
    return ", ".join(actions[-int(max_items) :])


def _context_for_step(traj_entries, idx, history_k):
    total = len(traj_entries)
    entry = traj_entries[idx]
    prev_actions = [_entry_action(item) for item in traj_entries[:idx]]
    return (
        f"Trajectory: You are at step {idx} out of {max(total - 1, 0)} (0-indexed). "
        f"You have executed {idx} actions so far.\n"
        f"Recent actions: {_summarize_action_seq(prev_actions, history_k)}.\n"
        f"Pose: {_entry_pose_text(entry)}\n"
    )


def _validate_api_entries(data, root):
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Entry at index {i} must be a JSON object")
        _entry_task_desc(entry)
        _entry_action(entry)
        _entry_pose_text(entry)
        rgb_path, depth_path = _entry_rgbd_paths(root, entry)
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found at entry {i}: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found at entry {i}: {depth_path}")


def _load_image_data_url(path, max_image_size, image_quality):
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as src:
        image = src.convert("RGB")
        max_side = int(max_image_size)
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(image_quality), optimize=True)

    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _image_reference(rel_path, abs_path, args):
    if args.image_base_url:
        base = str(args.image_base_url).rstrip("/")
        rel = str(rel_path).replace("\\", "/").lstrip("/")
        return f"{base}/{rel}"
    return _load_image_data_url(
        abs_path,
        max_image_size=args.max_image_size,
        image_quality=args.image_quality,
    )


def _api_system_prompt():
    return (
        "You annotate belief-state awareness for an autonomous exploration drone.\n"
        "The annotation is supervision for trajectory memory, not a user-facing caption.\n"
        "Infer task-relevant beliefs from the current RGB image, depth visualization, "
        "pose, current action, and action history.\n"
        "Do not merely justify the provided action after the fact. First infer what the "
        "drone should believe about the target or incident, including uncertainty.\n"
        "Use direct evidence when visible and indirect contextual evidence when useful. "
        "Do not follow a fixed checklist of cues; reason from what is actually visible.\n"
        "If evidence is weak, say so. Do not invent a target that is not supported.\n"
        "Return one JSON object only."
    )


def _few_shot_text():
    return (
        "Examples of reasoning style, not rules:\n"
        "1) If the target is not visible but the scene contains a distant abnormal plume "
        "and the recent path has not checked that block, the belief can point toward "
        "that unexplored area with uncertainty.\n"
        "2) If people or emergency vehicles appear to cluster near an intersection, "
        "the belief can treat that area as task-relevant context rather than a confirmed target.\n"
        "3) If the image only shows ordinary road and buildings after several forward moves, "
        "the belief should emphasize weak evidence and justify scanning or changing heading.\n"
        "4) If the target is visible, the belief should cite the direct visual evidence and "
        "explain why the current action approaches or confirms it."
    )


def _api_user_text(task_line, context_text, current_action):
    schema = {
        "task_line": "copy the provided Task line exactly",
        "history": "one concise sentence about relevant explored history or lack of evidence",
        "observation_inference": "one concise sentence linking visual/depth evidence to a belief state",
        "plan": "one concise sentence explaining why the current action is reasonable or cautious",
        "uncertainty": "low, medium, or high, based on evidence strength",
        "visual_evidence": ["short evidence phrase 1", "short evidence phrase 2"],
        "confidence": 0.0,
    }
    return (
        "Write an awareness annotation for this frame.\n\n"
        f"{_few_shot_text()}\n\n"
        f"{context_text}"
        f"Current Action: {current_action}\n"
        f"Required Task line: {task_line}\n\n"
        "Return JSON matching this schema. Keep text fields in English and concise. "
        "Do not include markdown fences or extra text.\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _build_messages(task_line, context_text, current_action, rgb_ref, depth_ref):
    return [
        {
            "role": "system",
            "content": _api_system_prompt(),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _api_user_text(task_line, context_text, current_action)},
                {"type": "image_url", "image_url": {"url": rgb_ref}},
                {"type": "image_url", "image_url": {"url": depth_ref}},
            ],
        },
    ]


def _extract_json_object(text):
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw.strip()).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError("Model response is not a JSON object")


def _shorten_line(text, max_words=32):
    words = str(text or "").replace("\n", " ").split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def _normalize_confidence(value):
    try:
        score = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, score))


def _normalize_api_payload(payload, task_line, current_action):
    if not isinstance(payload, dict):
        raise ValueError("API payload must be a JSON object")

    history = _shorten_line(payload.get("history"))
    obs = _shorten_line(payload.get("observation_inference"))
    plan = _shorten_line(payload.get("plan"))

    if not history:
        history = "The recent trajectory provides limited confirmed evidence, so I maintain a cautious search belief."
    if not obs:
        obs = "The current view offers weak or ambiguous task evidence, so the target location remains uncertain."
    if not plan:
        plan = f"The current action {current_action} is used to gather more evidence while preserving the search."

    normalized = {
        "task_line": task_line,
        "history": history,
        "observation_inference": obs,
        "plan": plan,
        "uncertainty": str(payload.get("uncertainty", "")).strip().lower() or None,
        "visual_evidence": payload.get("visual_evidence") if isinstance(payload.get("visual_evidence"), list) else [],
        "confidence": _normalize_confidence(payload.get("confidence")),
    }
    normalized["awareness"] = "\n".join(
        [
            task_line,
            f"History: {history}",
            f"Observation & Inference: {obs}",
            f"Plan: {plan}",
        ]
    )
    return normalized


def _coerce_text_response(text, task_line, current_action):
    history = _extract_prefixed_line(text, "History:")
    obs = _extract_prefixed_line(text, "Observation & Inference:")
    plan = _extract_prefixed_line(text, "Plan:")
    payload = {
        "history": history[len("History:") :].strip() if history else "",
        "observation_inference": obs[len("Observation & Inference:") :].strip() if obs else "",
        "plan": plan[len("Plan:") :].strip() if plan else "",
        "uncertainty": None,
        "visual_evidence": [],
        "confidence": None,
    }
    return _normalize_api_payload(payload, task_line=task_line, current_action=current_action)


def _parse_model_content(content, task_line, current_action):
    try:
        payload = _extract_json_object(content)
        normalized = _normalize_api_payload(payload, task_line=task_line, current_action=current_action)
        normalized["parse_status"] = "json"
        return normalized
    except Exception:
        normalized = _coerce_text_response(content, task_line=task_line, current_action=current_action)
        normalized["parse_status"] = "coerced_text"
        normalized["raw_response"] = str(content or "")
        return normalized


def _task_rel_paths(root, entry):
    rgb_abs, depth_abs = _entry_rgbd_paths(root, entry)
    dataset_root = root / "dataset"
    try:
        rgb_rel = rgb_abs.relative_to(dataset_root).as_posix()
    except ValueError:
        rgb_rel = rgb_abs.as_posix()
    try:
        depth_rel = depth_abs.relative_to(dataset_root).as_posix()
    except ValueError:
        depth_rel = depth_abs.as_posix()
    return rgb_abs, depth_abs, rgb_rel, depth_rel


def _build_api_tasks(data, root, args):
    groups = _group_entries(data)
    trajectory_filter = None
    if args.trajectory_ids:
        trajectory_filter = {part.strip() for part in str(args.trajectory_ids).split(",") if part.strip()}

    tasks = []
    for traj, step_pairs in sorted(groups.items(), key=lambda kv: kv[0]):
        if trajectory_filter is not None and traj not in trajectory_filter:
            continue
        step_pairs.sort(key=lambda x: x[0])
        ordered_indices = [idx for _, idx in step_pairs]
        traj_entries = [data[idx] for idx in ordered_indices]
        if not traj_entries:
            continue

        task_desc = _entry_task_desc(traj_entries[0])
        task_line = f"Task: I need to {task_desc} (Task)."

        for local_idx, global_idx in enumerate(ordered_indices):
            entry = data[global_idx]
            if args.skip_existing and "awareness" in entry:
                continue
            rgb_abs, depth_abs, rgb_rel, depth_rel = _task_rel_paths(root, entry)
            context = _context_for_step(traj_entries, local_idx, int(args.history_k))
            current_action = _entry_action(entry)
            _, step_index = _entry_traj_step(entry, global_idx)
            tasks.append(
                {
                    "custom_id": _safe_custom_id(global_idx),
                    "global_idx": int(global_idx),
                    "trajectory_id": traj,
                    "step_index": int(step_index),
                    "task_line": task_line,
                    "context": context,
                    "current_action": current_action,
                    "rgb_abs": str(rgb_abs),
                    "depth_abs": str(depth_abs),
                    "rgb_rel": rgb_rel,
                    "depth_rel": depth_rel,
                }
            )
            if args.limit and len(tasks) >= int(args.limit):
                return tasks
    return tasks


def _request_body_for_task(task, args):
    rgb_ref = _image_reference(task["rgb_rel"], task["rgb_abs"], args)
    depth_ref = _image_reference(task["depth_rel"], task["depth_abs"], args)
    body = {
        "model": args.model,
        "messages": _build_messages(
            task_line=task["task_line"],
            context_text=task["context"],
            current_action=task["current_action"],
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
        ),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
    }
    if args.response_format != "none":
        body["response_format"] = {"type": args.response_format}
    return body


def _get_api_key(args):
    if args.api_key:
        return args.api_key
    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY or pass --api_key.")


def _load_openai_client(args):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    return OpenAI(api_key=_get_api_key(args), base_url=args.base_url)


def _extract_chat_content(response):
    choice = response.choices[0]
    return choice.message.content


def _call_chat_completion(client, task, args):
    body = _request_body_for_task(task, args)
    try:
        response = client.chat.completions.create(**body)
    except Exception:
        if "response_format" not in body or not args.retry_without_response_format:
            raise
        body.pop("response_format", None)
        response = client.chat.completions.create(**body)
    return _extract_chat_content(response)


def _apply_annotation(data, task, parsed, sidecar_rows):
    entry = data[int(task["global_idx"])]
    entry["awareness"] = parsed["awareness"]
    sidecar_rows.append(
        {
            "custom_id": task["custom_id"],
            "global_idx": int(task["global_idx"]),
            "trajectory_id": task["trajectory_id"],
            "step_index": int(task["step_index"]),
            "prompt_version": PROMPT_VERSION,
            "task_line": task["task_line"],
            "current_action": task["current_action"],
            "awareness": parsed["awareness"],
            "structured": {
                "history": parsed.get("history"),
                "observation_inference": parsed.get("observation_inference"),
                "plan": parsed.get("plan"),
                "uncertainty": parsed.get("uncertainty"),
                "visual_evidence": parsed.get("visual_evidence", []),
                "confidence": parsed.get("confidence"),
            },
            "parse_status": parsed.get("parse_status"),
            "raw_response": parsed.get("raw_response"),
        }
    )


def _run_sync(data, tasks, args):
    client = _load_openai_client(args)
    sidecar_rows = []
    for task in tqdm(tasks, desc="API awareness", unit="entry"):
        content = _call_chat_completion(client, task, args)
        parsed = _parse_model_content(content, task_line=task["task_line"], current_action=task["current_action"])
        _apply_annotation(data, task, parsed, sidecar_rows)
        if float(args.sleep_s) > 0:
            time.sleep(float(args.sleep_s))
    return sidecar_rows


def _write_batch_jsonl(tasks, args):
    output_path = Path(args.batch_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for task in tqdm(tasks, desc="Writing batch jsonl", unit="entry"):
            row = {
                "custom_id": task["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _request_body_for_task(task, args),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = Path(args.batch_meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        meta_path,
        {
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "input_json": args.input_json,
            "output_json": args.output_json,
            "batch_jsonl": str(output_path),
            "tasks": [
                {
                    key: task[key]
                    for key in (
                        "custom_id",
                        "global_idx",
                        "trajectory_id",
                        "step_index",
                        "task_line",
                        "current_action",
                        "rgb_rel",
                        "depth_rel",
                    )
                }
                for task in tasks
            ],
        },
    )
    return output_path, meta_path


def _extract_batch_content(row):
    response = row.get("response") if isinstance(row, dict) else None
    if not isinstance(response, dict):
        raise ValueError("Batch row missing response object")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("Batch row missing response.body")
    choices = body.get("choices")
    if not choices:
        raise ValueError("Batch row missing choices")
    return choices[0]["message"]["content"]


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {path}") from exc
    return rows


def _run_apply_batch(data, args):
    meta = _read_json(args.batch_meta_json)
    task_map = {item["custom_id"]: item for item in meta.get("tasks", [])}
    sidecar_rows = []
    failed_rows = []

    for row in tqdm(_load_jsonl(args.batch_result_jsonl), desc="Applying batch", unit="entry"):
        custom_id = row.get("custom_id")
        task = task_map.get(custom_id)
        if task is None:
            try:
                global_idx = _parse_custom_id(custom_id)
                entry = data[global_idx]
                task = {
                    "custom_id": custom_id,
                    "global_idx": global_idx,
                    "trajectory_id": entry.get("trajectory_id"),
                    "step_index": entry.get("step_index"),
                    "task_line": f"Task: I need to {_entry_task_desc(entry)} (Task).",
                    "current_action": _entry_action(entry),
                }
            except Exception as exc:
                failed_rows.append({"custom_id": custom_id, "error": str(exc)})
                continue
        try:
            content = _extract_batch_content(row)
            parsed = _parse_model_content(content, task_line=task["task_line"], current_action=task["current_action"])
            _apply_annotation(data, task, parsed, sidecar_rows)
        except Exception as exc:
            failed_rows.append({"custom_id": custom_id, "error": str(exc)})

    return sidecar_rows, failed_rows


def _default_sidecar_path(output_json):
    output_path = Path(output_json)
    return output_path.with_name(output_path.stem + "_api_sidecar.json")


def _default_batch_jsonl_path():
    return _repo_root() / "dataset" / "awareness_api_batch.jsonl"


def _default_batch_meta_path():
    return _repo_root() / "dataset" / "awareness_api_batch_meta.json"


def _write_json_with_parent(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, obj)


def main():
    parser = argparse.ArgumentParser(description="Annotate awareness with an OpenAI-compatible vision API.")
    parser.add_argument("--mode", choices=("sync", "batch_jsonl", "apply_batch"), default="sync")
    parser.add_argument("--input_json", default=str(_repo_root() / "dataset" / "train_data_all.json"))
    parser.add_argument("--output_json", default=str(_repo_root() / "dataset" / "train_data_all_with_awareness_api.json"))
    parser.add_argument("--sidecar_json", default=None)
    parser.add_argument("--batch_jsonl", default=str(_default_batch_jsonl_path()))
    parser.add_argument("--batch_meta_json", default=str(_default_batch_meta_path()))
    parser.add_argument("--batch_result_jsonl", default=None)

    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--response_format", choices=("json_object", "none"), default="json_object")
    parser.add_argument(
        "--no_retry_without_response_format",
        dest="retry_without_response_format",
        action="store_false",
        help="Disable automatic retry when the provider rejects response_format.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sleep_s", type=float, default=0.0)

    parser.add_argument("--history_k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trajectory_ids", default=None)
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--image_base_url", default=None, help="Use hosted image URLs instead of base64 data URLs.")
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--image_quality", type=int, default=85)

    parser.set_defaults(retry_without_response_format=True)
    args = parser.parse_args()
    args.skip_existing = not args.no_skip_existing

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    root = _repo_root()
    data = _read_json(input_path)
    _validate_api_entries(data, root)

    sidecar_path = Path(args.sidecar_json) if args.sidecar_json else _default_sidecar_path(args.output_json)

    if args.mode == "apply_batch":
        if not args.batch_result_jsonl:
            raise RuntimeError("--batch_result_jsonl is required in apply_batch mode")
        sidecar_rows, failed_rows = _run_apply_batch(data, args)
        _write_json_with_parent(args.output_json, data)
        _write_json_with_parent(sidecar_path, {"rows": sidecar_rows, "failed_rows": failed_rows})
        print(f"Applied rows: {len(sidecar_rows)}")
        print(f"Failed rows: {len(failed_rows)}")
        print(f"Wrote output: {args.output_json}")
        print(f"Wrote sidecar: {sidecar_path}")
        return 0

    tasks = _build_api_tasks(data, root, args)
    total_entries = len(data)
    print(f"Total entries: {total_entries}")
    print(f"Entries to annotate: {len(tasks)}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Model: {args.model}")

    if args.mode == "batch_jsonl":
        batch_path, meta_path = _write_batch_jsonl(tasks, args)
        print(f"Wrote batch JSONL: {batch_path}")
        print(f"Wrote batch metadata: {meta_path}")
        return 0

    sidecar_rows = _run_sync(data, tasks, args) if tasks else []
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_with_parent(output_path, data)
    _write_json_with_parent(
        sidecar_path,
        {
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "input_json": args.input_json,
            "output_json": args.output_json,
            "rows": sidecar_rows,
        },
    )
    print(f"Wrote output: {output_path}")
    print(f"Wrote sidecar: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
