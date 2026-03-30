import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

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


def main():
    parser = argparse.ArgumentParser(description="Prepare DINOv2 RGB+Depth feature cache")
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
        default=str(_repo_root() / "dataset" / "dino_cache"),
        help="Output cache directory",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/dinov2-base",
        help="DINO model name",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Image batch size")
    parser.add_argument("--device", default="cuda", help="Device name")
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True, token=hf_token)
    model = AutoModel.from_pretrained(args.model_name, token=hf_token).to(device)
    model.eval()

    pairs = []
    for i, entry in enumerate(data):
        sid = _sample_id(entry, i)
        rgb_rel, depth_rel = _rgbd_paths(entry)
        pairs.append((sid, rgb_rel, depth_rel))

    existing = {}
    if manifest_path.exists() and not args.overwrite:
        old = _read_json(manifest_path)
        if isinstance(old, dict):
            existing = old

    pending = []
    for sid, rgb_rel, depth_rel in pairs:
        rgb_out_rel = f"{sid.replace(':', '_')}_rgb.npy"
        depth_out_rel = f"{sid.replace(':', '_')}_depth.npy"
        rgb_out_path = cache_dir / rgb_out_rel
        depth_out_path = cache_dir / depth_out_rel
        if rgb_out_path.exists() and depth_out_path.exists() and not args.overwrite:
            existing[sid] = {
                "rgb": rgb_out_rel,
                "depth": depth_out_rel,
                "dim": 768,
            }
            continue
        pending.append((sid, rgb_rel, depth_rel, rgb_out_rel, depth_out_rel))

    total = len(pending)
    print(f"Total entries: {len(pairs)}, pending cache: {total}")

    for start in range(0, total, int(args.batch_size)):
        chunk = pending[start:start + int(args.batch_size)]
        rgb_images = []
        depth_images = []
        valid = []
        for sid, rgb_rel, depth_rel, rgb_out_rel, depth_out_rel in chunk:
            rgb_path = dataset_root / rgb_rel
            depth_path = dataset_root / depth_rel
            if not rgb_path.exists() or not depth_path.exists():
                continue
            try:
                rgb_image = Image.open(rgb_path).convert("RGB")
                depth_image = Image.open(depth_path).convert("RGB")
            except Exception:
                continue
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            valid.append((sid, rgb_out_rel, depth_out_rel))
        if not rgb_images:
            continue

        rgb_inputs = processor(images=rgb_images, return_tensors="pt")
        rgb_inputs = {k: v.to(device) for k, v in rgb_inputs.items()}
        depth_inputs = processor(images=depth_images, return_tensors="pt")
        depth_inputs = {k: v.to(device) for k, v in depth_inputs.items()}
        with torch.inference_mode():
            rgb_outputs = model(**rgb_inputs)
            depth_outputs = model(**depth_inputs)
            rgb_cls_feat = rgb_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            depth_cls_feat = depth_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

        for idx, (sid, rgb_out_rel, depth_out_rel) in enumerate(valid):
            rgb_out_path = cache_dir / rgb_out_rel
            depth_out_path = cache_dir / depth_out_rel
            np.save(rgb_out_path, rgb_cls_feat[idx])
            np.save(depth_out_path, depth_cls_feat[idx])
            existing[sid] = {
                "rgb": rgb_out_rel,
                "depth": depth_out_rel,
                "dim": 768,
            }

        end = min(start + int(args.batch_size), total)
        print(f"Processed {end}/{total}")

    _write_json(manifest_path, existing)
    print(f"Cache manifest saved: {manifest_path}")
    print(f"Cached features: {len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
