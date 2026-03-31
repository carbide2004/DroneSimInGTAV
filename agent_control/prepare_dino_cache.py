import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModel, CLIPModel, AutoTokenizer

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


def _init_backbone(args, hf_token, device):
    backbone = str(args.vision_backbone).lower().strip()
    if backbone == "dino":
        model_name = str(args.model_name)
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, token=hf_token)
        model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
    elif backbone == "clip":
        model_name = str(args.clip_model_name)
        processor = CLIPImageProcessor.from_pretrained(model_name, token=hf_token)
        model = CLIPVisionModel.from_pretrained(model_name, token=hf_token).to(device)
    else:
        raise ValueError(f"Unsupported vision_backbone: {args.vision_backbone}")
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    return backbone, model_name, processor, model, hidden_size


def _infer_grid_size(patch_count: int):
    side = int(round(float(patch_count) ** 0.5))
    if side * side != patch_count:
        raise RuntimeError(f"Patch count {patch_count} is not a square number")
    return side


def _goal_cross_features(clip_model, processor, tokenizer, rgb_images, depth_images, task_texts, device, temperature: float):
    rgb_inputs = processor(images=rgb_images, return_tensors="pt")
    rgb_inputs = {k: v.to(device) for k, v in rgb_inputs.items()}
    depth_inputs = processor(images=depth_images, return_tensors="pt")
    depth_inputs = {k: v.to(device) for k, v in depth_inputs.items()}
    text_inputs = tokenizer(
        task_texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.inference_mode():
        text_outputs = clip_model.text_model(**text_inputs)
        text_pooled = text_outputs.pooler_output
        text_proj = clip_model.text_projection(text_pooled)
        text_proj = torch.nn.functional.normalize(text_proj, p=2, dim=-1)

        rgb_outputs = clip_model.vision_model(**rgb_inputs)
        depth_outputs = clip_model.vision_model(**depth_inputs)

        rgb_patches = rgb_outputs.last_hidden_state[:, 1:, :]
        depth_patches = depth_outputs.last_hidden_state[:, 1:, :]
        rgb_proj = clip_model.visual_projection(rgb_patches)
        depth_proj = clip_model.visual_projection(depth_patches)
        rgb_proj = torch.nn.functional.normalize(rgb_proj, p=2, dim=-1)
        depth_proj = torch.nn.functional.normalize(depth_proj, p=2, dim=-1)

        text_q = text_proj.unsqueeze(-1)
        rgb_sim = torch.matmul(rgb_proj, text_q).squeeze(-1) / float(temperature)
        depth_sim = torch.matmul(depth_proj, text_q).squeeze(-1) / float(temperature)
        sim = 0.5 * (rgb_sim + depth_sim)
        attn = torch.softmax(sim, dim=-1)

        rgb_goal = torch.sum(attn.unsqueeze(-1) * rgb_patches, dim=1)
        depth_goal = torch.sum(attn.unsqueeze(-1) * depth_patches, dim=1)
        f_goal = torch.cat([rgb_goal, depth_goal], dim=-1)

        p_exist = torch.sigmoid(torch.max(sim, dim=-1).values).unsqueeze(-1)

        patch_count = int(attn.shape[-1])
        side = _infer_grid_size(patch_count)
        coords = torch.arange(side, device=attn.device, dtype=attn.dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        grid_x = (xx.reshape(-1) + 0.5) / float(side)
        grid_y = (yy.reshape(-1) + 0.5) / float(side)
        u = torch.sum(attn * grid_x.unsqueeze(0), dim=-1, keepdim=True)
        v = torch.sum(attn * grid_y.unsqueeze(0), dim=-1, keepdim=True)

        goal_vec = torch.cat([f_goal, p_exist, u, v], dim=-1)
        goal_vec = goal_vec.detach().cpu().numpy().astype(np.float32)

    return goal_vec


def main():
    parser = argparse.ArgumentParser(description="Prepare RGB+Depth feature cache")
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
    parser.add_argument(
        "--clip_model_name",
        default="openai/clip-vit-base-patch32",
        help="CLIP vision model name",
    )
    parser.add_argument(
        "--vision_backbone",
        choices=["dino", "clip"],
        default="dino",
        help="Vision backbone for feature extraction",
    )
    parser.add_argument(
        "--cache_mode",
        choices=["cls", "goal_cross"],
        default="goal_cross",
        help="Feature cache mode",
    )
    parser.add_argument("--cross_temperature", type=float, default=0.07, help="Goal cross temperature")
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
    backbone, model_name, processor, model, hidden_size = _init_backbone(args, hf_token, device)
    print(f"Backbone: {backbone}, model: {model_name}, hidden_size: {hidden_size}")
    clip_cross_model = None
    clip_tokenizer = None
    if str(args.cache_mode) == "goal_cross":
        if backbone != "clip":
            raise RuntimeError("cache_mode=goal_cross requires --vision_backbone clip")
        clip_cross_model = CLIPModel.from_pretrained(str(args.clip_model_name), token=hf_token).to(device)
        clip_cross_model.eval()
        clip_tokenizer = AutoTokenizer.from_pretrained(str(args.clip_model_name), token=hf_token)

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
    for sid, rgb_rel, depth_rel, task_text in pairs:
        rgb_out_rel = f"{sid.replace(':', '_')}_rgb.npy"
        depth_out_rel = f"{sid.replace(':', '_')}_depth.npy"
        goal_out_rel = f"{sid.replace(':', '_')}_goal.npy"
        rgb_out_path = cache_dir / rgb_out_rel
        depth_out_path = cache_dir / depth_out_rel
        goal_out_path = cache_dir / goal_out_rel
        if str(args.cache_mode) == "goal_cross":
            if goal_out_path.exists() and not args.overwrite:
                existing[sid] = {
                    "goal": goal_out_rel,
                    "feature_dim": hidden_size * 2 + 3,
                    "patch_dim": hidden_size,
                    "backbone": backbone,
                    "mode": "goal_cross",
                }
                continue
            pending.append((sid, rgb_rel, depth_rel, task_text, goal_out_rel))
        else:
            if rgb_out_path.exists() and depth_out_path.exists() and not args.overwrite:
                existing[sid] = {
                    "rgb": rgb_out_rel,
                    "depth": depth_out_rel,
                    "dim": hidden_size,
                    "backbone": backbone,
                    "mode": "cls",
                }
                continue
            pending.append((sid, rgb_rel, depth_rel, task_text, rgb_out_rel, depth_out_rel))

    total = len(pending)
    print(f"Total entries: {len(pairs)}, pending cache: {total}")

    for start in range(0, total, int(args.batch_size)):
        chunk = pending[start:start + int(args.batch_size)]
        rgb_images = []
        depth_images = []
        task_texts = []
        valid = []
        for item in chunk:
            if str(args.cache_mode) == "goal_cross":
                sid, rgb_rel, depth_rel, task_text, goal_out_rel = item
            else:
                sid, rgb_rel, depth_rel, task_text, rgb_out_rel, depth_out_rel = item
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
            if str(args.cache_mode) == "goal_cross":
                task_texts.append(task_text)
                valid.append((sid, goal_out_rel))
            else:
                valid.append((sid, rgb_out_rel, depth_out_rel))
        if not rgb_images:
            continue
        if str(args.cache_mode) == "goal_cross":
            goal_vec = _goal_cross_features(
                clip_model=clip_cross_model,
                processor=processor,
                tokenizer=clip_tokenizer,
                rgb_images=rgb_images,
                depth_images=depth_images,
                task_texts=task_texts,
                device=device,
                temperature=float(args.cross_temperature),
            )
            for idx, (sid, goal_out_rel) in enumerate(valid):
                goal_out_path = cache_dir / goal_out_rel
                np.save(goal_out_path, goal_vec[idx])
                existing[sid] = {
                    "goal": goal_out_rel,
                    "feature_dim": hidden_size * 2 + 3,
                    "patch_dim": hidden_size,
                    "backbone": backbone,
                    "mode": "goal_cross",
                }
        else:
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
                    "dim": hidden_size,
                    "backbone": backbone,
                    "mode": "cls",
                }

        end = min(start + int(args.batch_size), total)
        print(f"Processed {end}/{total}")

    _write_json(manifest_path, existing)
    print(f"Cache manifest saved: {manifest_path}")
    print(f"Cached features: {len(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
