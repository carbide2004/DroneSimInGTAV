import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from action_mapping import ACTIONS, parse_action
from hf_auth import load_hf_token_from_env_file
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from stage1_model import Stage1Config, Stage1GRUModel
from stage2_bridge import Stage2BridgeConfig, Stage2BridgeModel
from stage2_softprompt import forward_action_ce_with_soft_prompt, generate_action_with_soft_prompt
from train_stage1 import FeatureCacheStore
from trajectory_dataset import (
    build_stage1_dataloaders,
    build_stage1_dataloaders_from_manifest,
    build_stage1_dataloaders_from_split_json,
)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_stage1_encoder(stage1_ckpt: Path, device: torch.device):
    payload = torch.load(stage1_ckpt, map_location=device)
    config_data = payload.get("config", {}).get("model")
    if not isinstance(config_data, dict):
        raise RuntimeError("Invalid stage1 checkpoint: missing config.model")
    stage1_cfg = Stage1Config(**config_data)
    model = Stage1GRUModel(stage1_cfg).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Invalid stage1 checkpoint: missing model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_stage2_bridge(stage2_ckpt: Path, device: torch.device):
    payload = torch.load(stage2_ckpt, map_location=device)
    config = payload.get("config", {})
    stage1_cfg = config.get("stage1", {})
    stage2_cfg = config.get("stage2", {})
    bridge_cfg = Stage2BridgeConfig(
        hidden_dim=int(stage1_cfg.get("hidden_dim", 512)),
        llm_dim=int(stage2_cfg.get("llm_dim", 3584)),
        num_soft_tokens=int(stage2_cfg.get("num_soft_tokens", 16)),
    )
    bridge = Stage2BridgeModel(bridge_cfg).to(device)
    state = payload.get("bridge_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Invalid stage2 checkpoint: missing bridge_state_dict")
    bridge.load_state_dict(state, strict=True)
    bridge.eval()
    for p in bridge.parameters():
        p.requires_grad = False
    return bridge


def _load_step_images(dataset_root: Path, step: dict):
    rgb_rel = step.get("rgb_path")
    depth_rel = step.get("depth_path")
    if not rgb_rel or not depth_rel:
        raise RuntimeError(f"Missing rgb/depth path in step: {step.get('sample_id')}")
    rgb_path = dataset_root / str(rgb_rel)
    depth_path = dataset_root / str(depth_rel)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    with Image.open(rgb_path) as rgb_raw:
        rgb_img = rgb_raw.convert("RGB")
    with Image.open(depth_path) as depth_raw:
        depth_img = depth_raw.convert("RGB")
    return rgb_img, depth_img


def _prepare_features(raw_batch, feature_store: FeatureCacheStore, max_len: int, device: torch.device):
    batch_size = len(raw_batch)
    features = torch.zeros((batch_size, max_len, feature_store.feature_dim), dtype=torch.float32, device=device)
    valid_steps = []
    for i, item in enumerate(raw_batch):
        steps = item["steps"][:max_len]
        for j, step in enumerate(steps):
            sample_id = step["sample_id"]
            feat = feature_store.get(sample_id)
            if feat is None:
                raise RuntimeError(f"Missing feature cache for sample_id={sample_id}")
            features[i, j] = torch.from_numpy(feat).to(device=device)
            valid_steps.append((i, j, item, step))
    return features, valid_steps


def _select_steps(valid_steps, steps_per_batch: int):
    if steps_per_batch <= 0 or len(valid_steps) <= steps_per_batch:
        return valid_steps
    return random.sample(valid_steps, k=int(steps_per_batch))


def _safe_action_index(action_name: str):
    try:
        return ACTIONS.index(action_name)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for stage2 soft prompt model")
    parser.add_argument("--dataset_json", default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"))
    parser.add_argument("--cache_dir", default=str(_repo_root() / "dataset" / "clip_cache"))
    parser.add_argument("--dataset_root", default=str(_repo_root() / "dataset"))
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_GTAV_20260403"))
    parser.add_argument("--stage2_lora_dir", default=None, help="Optional LoRA dir, default sibling lora_best")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample_limit", type=int, default=-1, help="Limit trajectory batches")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split_manifest_json", default=None, help="Fixed split manifest JSON path")
    parser.add_argument("--train_json", default=None, help="Fixed train split JSON path")
    parser.add_argument("--val_json", default=None, help="Fixed val split JSON path")
    parser.add_argument("--steps_per_batch", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=str(_repo_root() / "agent_control" / "checkpoints" / "stage2" / "offline_eval.json"))
    args = parser.parse_args()

    repo_root = _repo_root()
    load_hf_token_from_env_file(repo_root)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset_root = Path(args.dataset_root)
    feature_store = FeatureCacheStore(Path(args.cache_dir))

    if args.train_json and args.val_json:
        train_loader, val_loader, split_meta = build_stage1_dataloaders_from_split_json(
            train_json=Path(args.train_json),
            val_json=Path(args.val_json),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            mode="sequence",
        )
    elif args.split_manifest_json:
        train_loader, val_loader, split_meta = build_stage1_dataloaders_from_manifest(
            dataset_json=Path(args.dataset_json),
            split_manifest_json=Path(args.split_manifest_json),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            mode="sequence",
        )
    else:
        train_loader, val_loader, split_meta = build_stage1_dataloaders(
            dataset_json=Path(args.dataset_json),
            batch_size=int(args.batch_size),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            num_workers=int(args.num_workers),
            mode="sequence",
        )
    loader = val_loader if args.split == "val" else train_loader

    stage1_model = _load_stage1_encoder(Path(args.stage1_ckpt), device=device)
    bridge = _load_stage2_bridge(Path(args.stage2_ckpt), device=device)

    qwen = Qwen3VLWrapper(args.model_dir, torch_dtype="auto", device_map={"":str(device)}).load()
    if args.stage2_lora_dir:
        lora_dir = Path(args.stage2_lora_dir)
    else:
        lora_dir = Path(args.stage2_ckpt).resolve().parent / "lora_best"
    if lora_dir.exists():
        try:
            from peft import PeftModel
            qwen._model = PeftModel.from_pretrained(qwen.model, lora_dir)
        except Exception as e:
            print(f"Warning: failed to load LoRA from {lora_dir}: {e}")
    qwen.model.eval()

    confusion = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
    invalid_predictions = 0
    correct = 0
    total = 0
    total_ce = 0.0
    total_ce_steps = 0
    examples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.sample_limit > 0 and batch_idx >= int(args.sample_limit):
                break
            raw_batch = batch["raw_batch"]
            features, valid_steps = _prepare_features(raw_batch, feature_store, max_len=int(args.max_len), device=device)
            selected_steps = _select_steps(valid_steps, steps_per_batch=int(args.steps_per_batch))
            if not selected_steps:
                continue

            h_seq, _ = stage1_model(features)
            for i, j, item, step in selected_steps:
                pose = step.get("pose") or {}
                task = item.get("task") or "find the closest burning car"
                gt_action = str(step.get("action_name") or "").strip()
                if not gt_action:
                    continue
                gt_idx = _safe_action_index(gt_action)
                if gt_idx is None:
                    continue

                rgb_img, depth_img = _load_step_images(dataset_root, step)
                prompt = build_prompt(
                    x=float(pose.get("x", 0.0)),
                    y=float(pose.get("y", 0.0)),
                    z=float(pose.get("z", 0.0)),
                    rz=float(pose.get("rz", 0.0)),
                    task=task,
                )
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

                h_t = h_seq[i, j].unsqueeze(0)
                soft_prompt = bridge(h_t)
                raw_pred = generate_action_with_soft_prompt(
                    processor=qwen.processor,
                    model=qwen.model,
                    messages=messages,
                    images=[rgb_img, depth_img],
                    soft_prompt=soft_prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                )
                pred_action = parse_action(raw_pred)
                pred_idx = _safe_action_index(pred_action) if pred_action else None

                ce_loss = forward_action_ce_with_soft_prompt(
                    processor=qwen.processor,
                    model=qwen.model,
                    messages=messages,
                    images=[rgb_img, depth_img],
                    action_text=gt_action,
                    soft_prompt=soft_prompt,
                )
                total_ce += float(ce_loss.item())
                total_ce_steps += 1

                if pred_idx is None:
                    invalid_predictions += 1
                else:
                    confusion[gt_idx, pred_idx] += 1
                    if pred_idx == gt_idx:
                        correct += 1
                total += 1

                if len(examples) < 30:
                    examples.append(
                        {
                            "sample_id": step.get("sample_id"),
                            "task": task,
                            "gt_action": gt_action,
                            "pred_action": pred_action,
                            "raw_pred": raw_pred,
                        }
                    )

    valid_preds = int(confusion.sum())
    accuracy = (correct / valid_preds) if valid_preds > 0 else 0.0
    total_recall_denom = confusion.sum(axis=1)
    per_class = {}
    for idx, name in enumerate(ACTIONS):
        tp = int(confusion[idx, idx])
        pred_count = int(confusion[:, idx].sum())
        gt_count = int(total_recall_denom[idx])
        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_count if gt_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "tp": tp,
            "pred_count": pred_count,
            "gt_count": gt_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    result = {
        "summary": {
            "split": args.split,
            "total_steps": total,
            "valid_predictions": valid_preds,
            "invalid_predictions": int(invalid_predictions),
            "accuracy_valid_predictions": accuracy,
            "avg_action_ce": (total_ce / total_ce_steps) if total_ce_steps > 0 else 0.0,
            "batches_meta": split_meta,
        },
        "labels": list(ACTIONS),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "examples": examples,
        "config": {
            "stage1_ckpt": str(Path(args.stage1_ckpt)),
            "stage2_ckpt": str(Path(args.stage2_ckpt)),
            "model_dir": str(Path(args.model_dir)),
            "stage2_lora_dir": str(lora_dir) if "lora_dir" in locals() else None,
            "cache_dir": str(Path(args.cache_dir)),
            "dataset_json": str(Path(args.dataset_json)),
            "dataset_root": str(dataset_root),
            "batch_size": int(args.batch_size),
            "steps_per_batch": int(args.steps_per_batch),
            "max_len": int(args.max_len),
            "max_new_tokens": int(args.max_new_tokens),
            "seed": int(args.seed),
            "split_manifest_json": str(Path(args.split_manifest_json)) if args.split_manifest_json else None,
            "train_json": str(Path(args.train_json)) if args.train_json else None,
            "val_json": str(Path(args.val_json)) if args.val_json else None,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    print(f"Offline eval saved to: {output_path}")
    print(
        f"split={args.split} total_steps={total} valid_preds={valid_preds} invalid_preds={invalid_predictions} "
        f"acc={accuracy:.4f} avg_ce={result['summary']['avg_action_ce']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
