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


def _rgb_path(entry):
    observations = entry.get("observations")
    if isinstance(observations, dict):
        rgb = observations.get("rgb") or {}
        path = rgb.get("path")
        if path:
            return str(path)
    images = entry.get("images") or []
    if images:
        return str(images[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare DINOv2 RGB feature cache")
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
        rgb_rel = _rgb_path(entry)
        if not rgb_rel:
            continue
        pairs.append((sid, rgb_rel))

    existing = {}
    if manifest_path.exists() and not args.overwrite:
        old = _read_json(manifest_path)
        if isinstance(old, dict):
            existing = old

    pending = []
    for sid, rgb_rel in pairs:
        out_rel = f"{sid.replace(':', '_')}.npy"
        out_path = cache_dir / out_rel
        if out_path.exists() and not args.overwrite:
            existing[sid] = out_rel
            continue
        pending.append((sid, rgb_rel, out_rel))

    total = len(pending)
    print(f"Total entries: {len(pairs)}, pending cache: {total}")

    for start in range(0, total, int(args.batch_size)):
        chunk = pending[start:start + int(args.batch_size)]
        images = []
        valid = []
        for sid, rgb_rel, out_rel in chunk:
            img_path = dataset_root / rgb_rel
            if not img_path.exists():
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            images.append(image)
            valid.append((sid, out_rel))
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            cls_feat = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

        for idx, (sid, out_rel) in enumerate(valid):
            out_path = cache_dir / out_rel
            np.save(out_path, cls_feat[idx])
            existing[sid] = out_rel

        end = min(start + int(args.batch_size), total)
        print(f"Processed {end}/{total}")

    _write_json(manifest_path, existing)
    print(f"Cache manifest saved: {manifest_path}")
    print(f"Cached features: {len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
