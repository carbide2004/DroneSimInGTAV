import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from hf_auth import load_hf_token_from_env_file


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sample_id(entry, fallback_index):
    sample_id = entry.get("sample_id")
    if sample_id:
        return str(sample_id)
    trajectory_id = str(entry.get("trajectory_id", f"__unknown_{fallback_index}__"))
    step_index = int(entry.get("step_index", 0))
    return f"{trajectory_id}:{step_index:06d}"


def _rgbd_paths(entry):
    observations = entry.get("observations")
    if not isinstance(observations, dict):
        raise KeyError("Missing required field: observations")
    rgb = observations.get("rgb")
    depth = observations.get("depth")
    if not isinstance(rgb, dict) or not isinstance(depth, dict):
        raise KeyError("Missing required field: observations.rgb/depth")
    rgb_path = str(rgb.get("path", "")).strip()
    depth_path = str(depth.get("path", "")).strip()
    if not rgb_path or not depth_path:
        raise ValueError("observations.rgb.path/depth.path must be non-empty")
    return rgb_path, depth_path


def _task_text(entry):
    task = entry.get("task")
    if not isinstance(task, str):
        raise KeyError("Missing required field: task")
    text = task.strip()
    if not text:
        raise ValueError("task must be non-empty")
    return text


def _tile_coords(width: int, height: int, window_size: int, stride: int):
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if window_size > width or window_size > height:
        raise ValueError(f"window_size={window_size} is larger than image size {width}x{height}")
    y_range = list(range(0, max(height - window_size, 1), stride))
    x_range = list(range(0, max(width - window_size, 1), stride))
    if not y_range or y_range[-1] != height - window_size:
        y_range.append(height - window_size)
    if not x_range or x_range[-1] != width - window_size:
        x_range.append(width - window_size)
    return [(x, y) for y in y_range for x in x_range]


def _resize_float_map(array_2d: np.ndarray, out_size: int):
    img = Image.fromarray(array_2d.astype(np.float32), mode="F")
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def _process_heatmap_fixed_scale(array_2d: np.ndarray):
    # logit 达到 20 是极值
    scale = 20.0
    return np.clip(array_2d / scale, 0.0, 1.0)


def _parse_gpu_ids(gpu_ids_text):
    gpu_ids = []
    for part in str(gpu_ids_text).split(","):
        part = part.strip()
        if part:
            gpu_ids.append(int(part))
    return gpu_ids


def _cache_meta(args_like, out_rel):
    return {
        "heatdepth": out_rel,
        "feature_dim": int(args_like["heatmap_size"]) * int(args_like["heatmap_size"]) * 3,
        "heatmap_size": int(args_like["heatmap_size"]),
        "feature_components": ["clip_heatmap", "depth", "rgb_gray"],
        "backbone": "clip",
        "mode": "heatmap_depth",
        "window_size": int(args_like["window_size"]),
        "stride": int(args_like["stride"]),
        "use_null_text_baseline": bool(args_like["use_null_text_baseline"]),
    }


def _assign_items_to_workers(items, worker_count):
    buckets = [[] for _ in range(max(1, int(worker_count)))]
    for i, item in enumerate(items):
        buckets[i % len(buckets)].append(item)
    return [bucket for bucket in buckets if bucket]


def _compute_clip_heatmap(
    clip_model,
    clip_image_processor,
    clip_tokenizer,
    rgb_image: Image.Image,
    task_text: str,
    device,
    window_size: int,
    stride: int,
    tile_batch_size: int,
    use_null_text: bool,
):
    width, height = rgb_image.size
    coords = _tile_coords(width, height, window_size, stride)
    score_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    text_list = [task_text, ""] if use_null_text else [task_text]
    text_inputs = clip_tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.inference_mode():
        text_outputs = clip_model.text_model(**text_inputs)
        text_proj = clip_model.text_projection(text_outputs.pooler_output)
        text_proj = torch.nn.functional.normalize(text_proj, p=2, dim=-1)
        logit_scale = clip_model.logit_scale.exp()

    for start in range(0, len(coords), tile_batch_size):
        batch_coords = coords[start:start + tile_batch_size]
        tiles = [
            rgb_image.crop((x, y, x + window_size, y + window_size))
            for x, y in batch_coords
        ]
        inputs = clip_image_processor(images=tiles, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            vision_outputs = clip_model.vision_model(**inputs)
            image_proj = clip_model.visual_projection(vision_outputs.pooler_output)
            image_proj = torch.nn.functional.normalize(image_proj, p=2, dim=-1)
            logits = torch.matmul(image_proj, text_proj.t()) * logit_scale
            if use_null_text:
                rel = logits[:, 0] - logits[:, 1]
            else:
                rel = logits[:, 0]
            rel_np = rel.detach().cpu().numpy().astype(np.float32)

        for i, (x, y) in enumerate(batch_coords):
            score_map[y:y + window_size, x:x + window_size] += rel_np[i]
            count_map[y:y + window_size, x:x + window_size] += 1.0

    return score_map / (count_map + 1e-8)


def _process_cache_item(
    item,
    dataset_root: Path,
    cache_dir: Path,
    clip_model,
    clip_image_processor,
    clip_tokenizer,
    device,
    args_like,
):
    sid, rgb_rel, depth_rel, task_text, out_rel = item
    rgb_path = dataset_root / rgb_rel
    depth_path = dataset_root / depth_rel
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    with Image.open(rgb_path) as rgb_src:
        rgb_img = rgb_src.convert("RGB")
    with Image.open(depth_path) as depth_src:
        depth_img = depth_src.convert("RGB")

    heatmap = _compute_clip_heatmap(
        clip_model=clip_model,
        clip_image_processor=clip_image_processor,
        clip_tokenizer=clip_tokenizer,
        rgb_image=rgb_img,
        task_text=task_text,
        device=device,
        window_size=int(args_like["window_size"]),
        stride=int(args_like["stride"]),
        tile_batch_size=int(args_like["tile_batch_size"]),
        use_null_text=bool(args_like["use_null_text_baseline"]),
    )
    heatmap_resized = _resize_float_map(heatmap, int(args_like["heatmap_size"]))
    heatmap_norm = _process_heatmap_fixed_scale(heatmap_resized)

    depth_gray = np.array(depth_img.convert("L"), dtype=np.float32) / 255.0
    depth_resized = _resize_float_map(depth_gray, int(args_like["heatmap_size"]))
    depth_norm = np.clip(depth_resized, 0.0, 1.0).astype(np.float32)

    rgb_gray = np.array(rgb_img.convert("L"), dtype=np.float32) / 255.0
    rgb_resized = _resize_float_map(rgb_gray, int(args_like["heatmap_size"]))
    rgb_norm = np.clip(rgb_resized, 0.0, 1.0).astype(np.float32)

    feature = np.concatenate(
        [heatmap_norm.reshape(-1), depth_norm.reshape(-1), rgb_norm.reshape(-1)],
        axis=0,
    ).astype(np.float32)
    np.save(cache_dir / out_rel, feature)
    return sid, _cache_meta(args_like, out_rel)


def _process_worker(worker_id, gpu_id, items, args_like, hf_token):
    if gpu_id is None:
        device_name = str(args_like["device"])
    else:
        device_name = f"cuda:{int(gpu_id)}"
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    clip_model = CLIPModel.from_pretrained(str(args_like["clip_model_name"]), token=hf_token).to(device)
    clip_model.eval()
    clip_image_processor = CLIPImageProcessor.from_pretrained(str(args_like["clip_model_name"]), token=hf_token)
    clip_tokenizer = AutoTokenizer.from_pretrained(str(args_like["clip_model_name"]), token=hf_token)

    dataset_root = Path(args_like["dataset_root"])
    cache_dir = Path(args_like["cache_dir"])
    updates = {}
    processed = 0
    for item in items:
        sid, meta = _process_cache_item(
            item=item,
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            clip_model=clip_model,
            clip_image_processor=clip_image_processor,
            clip_tokenizer=clip_tokenizer,
            device=device,
            args_like=args_like,
        )
        updates[str(sid)] = meta
        processed += 1
    return {
        "worker_id": int(worker_id),
        "gpu_id": None if gpu_id is None else int(gpu_id),
        "device": str(device),
        "processed": int(processed),
        "updates": updates,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CLIP tiled heatmap + depth + rgb cache")
    parser.add_argument(
        "--dataset_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Dataset JSON path",
    )
    parser.add_argument(
        "--dataset_root",
        default=str(_repo_root() / "dataset"),
        help="Dataset root path for image files",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(_repo_root() / "dataset" / "clip_cache"),
        help="Output cache directory",
    )
    parser.add_argument(
        "--clip_model_name",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name",
    )
    parser.add_argument("--heatmap_size", type=int, default=24, help="Heatmap/depth/rgb resize size")
    parser.add_argument("--window_size", type=int, default=144, help="Tiled CLIP window size")
    parser.add_argument("--stride", type=int, default=48, help="Tiled CLIP stride")
    parser.add_argument("--tile_batch_size", type=int, default=128, help="Tiles per batch for CLIP heatmap")
    parser.add_argument(
        "--use_null_text_baseline",
        action="store_true",
        help="Use target-null relative similarity when building heatmap",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Sample batch size")
    parser.add_argument("--device", default="cuda", help="Device name")
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids, for example: 0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Worker processes to launch per GPU")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    args = parser.parse_args()

    repo_root = _repo_root()
    hf_token = load_hf_token_from_env_file(repo_root)

    dataset_path = Path(args.dataset_json)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset_root = Path(args.dataset_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    data = _read_json(dataset_path)
    if not isinstance(data, list):
        raise RuntimeError("Dataset JSON must be a list.")

    pairs = []
    for i, entry in enumerate(data):
        sid = _sample_id(entry, i)
        rgb_rel, depth_rel = _rgbd_paths(entry)
        task_text = _task_text(entry)
        pairs.append((sid, rgb_rel, depth_rel, task_text))

    existing = {}
    if manifest_path.exists() and not args.overwrite:
        old = _read_json(manifest_path)
        if isinstance(old, dict):
            existing = old

    pending = []
    args_like = {
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "clip_model_name": str(args.clip_model_name),
        "heatmap_size": int(args.heatmap_size),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "tile_batch_size": int(args.tile_batch_size),
        "use_null_text_baseline": bool(args.use_null_text_baseline),
        "device": str(args.device),
    }
    for sid, rgb_rel, depth_rel, task_text in pairs:
        out_rel = f"{sid.replace(':', '_')}_heatdepth.npy"
        out_path = cache_dir / out_rel
        if out_path.exists() and not args.overwrite:
            existing[sid] = _cache_meta(args_like, out_rel)
            continue
        pending.append((sid, rgb_rel, depth_rel, task_text, out_rel))

    total = len(pending)
    print(f"Total entries: {len(pairs)}, pending cache: {total}")

    if pending:
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        workers_per_gpu = max(1, int(args.workers_per_gpu))
        if gpu_ids:
            worker_count = len(gpu_ids) * workers_per_gpu
            assignments = _assign_items_to_workers(pending, worker_count)
            print(f"Using GPUs: {gpu_ids}")
            print(f"Workers per GPU: {workers_per_gpu}")
        else:
            assignments = _assign_items_to_workers(pending, 1)
            print(f"Using device: {args.device}")

        done = 0
        mp_context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(assignments), mp_context=mp_context) as executor:
            futures = []
            for worker_id, assigned_items in enumerate(assignments):
                gpu_id = gpu_ids[worker_id % len(gpu_ids)] if gpu_ids else None
                futures.append(
                    executor.submit(
                        _process_worker,
                        worker_id,
                        gpu_id,
                        assigned_items,
                        args_like,
                        hf_token,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                existing.update(result["updates"])
                done += int(result["processed"])
                print(
                    f"Worker {result['worker_id']} device={result['device']} "
                    f"processed={result['processed']} total_done={done}/{total}"
                )

    _write_json(manifest_path, existing)
    print(f"Cache manifest saved: {manifest_path}")
    print(f"Cached features: {len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
