import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from action_mapping import ACTIONS, dispatch_action, parse_action
from dronesim_client import DroneSimClient
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from rgbd_utils import depth_bytes_to_pil, rgb_bytes_to_pil
from stage1_model import Stage1Config, Stage1GRUModel
from stage2_bridge import Stage2BridgeConfig, Stage2BridgeModel
from stage2_softprompt import forward_action_ce_with_soft_prompt, generate_action_with_soft_prompt
from verification_runtime import (
    build_movement_params,
    calculate_distance,
    calculate_summary_stats,
    create_anomaly_at_position,
    print_summary,
    read_jsonl,
    write_json,
)


def _repo_root():
    env_path = os.getenv("DRONESIM_ROOT")
    if env_path:
        return Path(env_path)
    return Path(r"E:\ToolApps\Steam\steamapps\common\Grand Theft Auto V")


def _tile_coords(width: int, height: int, window_size: int, stride: int):
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


def _normalize_minmax(array_2d: np.ndarray):
    min_v = float(np.min(array_2d))
    max_v = float(np.max(array_2d))
    if max_v - min_v < 1e-8:
        return np.zeros_like(array_2d, dtype=np.float32)
    return ((array_2d - min_v) / (max_v - min_v)).astype(np.float32)


class ClipHeatDepthExtractor:
    def __init__(
        self,
        model_name: str,
        heatmap_size: int,
        window_size: int,
        stride: int,
        tile_batch_size: int,
        use_null_text_baseline: bool,
        device: str,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.heatmap_size = int(heatmap_size)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.tile_batch_size = int(tile_batch_size)
        self.use_null_text_baseline = bool(use_null_text_baseline)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _compute_heatmap(self, rgb_image: Image.Image, task_text: str):
        width, height = rgb_image.size
        coords = _tile_coords(width, height, self.window_size, self.stride)
        score_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)

        text_list = [task_text, ""] if self.use_null_text_baseline else [task_text]
        text_inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text_outputs = self.model.text_model(**text_inputs)
            text_proj = self.model.text_projection(text_outputs.pooler_output)
            text_proj = torch.nn.functional.normalize(text_proj, p=2, dim=-1)
            logit_scale = self.model.logit_scale.exp()

        for start in range(0, len(coords), self.tile_batch_size):
            batch_coords = coords[start:start + self.tile_batch_size]
            tiles = [rgb_image.crop((x, y, x + self.window_size, y + self.window_size)) for x, y in batch_coords]
            inputs = self.image_processor(images=tiles, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                vision_outputs = self.model.vision_model(**inputs)
                image_proj = self.model.visual_projection(vision_outputs.pooler_output)
                image_proj = torch.nn.functional.normalize(image_proj, p=2, dim=-1)
                logits = torch.matmul(image_proj, text_proj.t()) * logit_scale
                rel = logits[:, 0] - logits[:, 1] if self.use_null_text_baseline else logits[:, 0]
                rel_np = rel.detach().cpu().numpy().astype(np.float32)

            for i, (x, y) in enumerate(batch_coords):
                score_map[y:y + self.window_size, x:x + self.window_size] += rel_np[i]
                count_map[y:y + self.window_size, x:x + self.window_size] += 1.0

        return score_map / (count_map + 1e-8)

    def extract(self, rgb_img: Image.Image, depth_img: Image.Image, task_text: str):
        heatmap = self._compute_heatmap(rgb_img, task_text)
        heatmap_resized = _resize_float_map(heatmap, self.heatmap_size)
        heatmap_norm = _normalize_minmax(heatmap_resized)

        depth_gray = np.array(depth_img.convert("L"), dtype=np.float32) / 255.0
        depth_resized = _resize_float_map(depth_gray, self.heatmap_size)
        depth_norm = np.clip(depth_resized, 0.0, 1.0).astype(np.float32)
        return np.concatenate([heatmap_norm.reshape(-1), depth_norm.reshape(-1)], axis=0).astype(np.float32)


def _load_stage1_encoder(stage1_ckpt: Path, device: torch.device):
    payload = torch.load(stage1_ckpt, map_location=device)
    cfg = payload.get("config", {}).get("model")
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid stage1 checkpoint: missing config.model")
    model = Stage1GRUModel(Stage1Config(**cfg)).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_stage2_bridge(stage2_ckpt: Path, device: torch.device):
    payload = torch.load(stage2_ckpt, map_location=device)
    config = payload.get("config", {}).get("stage2")
    stage1_cfg = payload.get("config", {}).get("stage1")
    if not isinstance(config, dict) or not isinstance(stage1_cfg, dict):
        raise RuntimeError("Invalid stage2 checkpoint config")
    bridge_cfg = Stage2BridgeConfig(
        hidden_dim=int(stage1_cfg.get("hidden_dim", 512)),
        llm_dim=int(config.get("llm_dim", 3584)),
        num_soft_tokens=int(config.get("num_soft_tokens", 16)),
    )
    bridge = Stage2BridgeModel(bridge_cfg).to(device)
    bridge.load_state_dict(payload["bridge_state_dict"], strict=True)
    bridge.eval()
    return bridge


def _rank_action_by_ce(processor, model, messages, images, soft_prompt):
    best_action = None
    best_loss = None
    for action_name in ACTIONS:
        loss = forward_action_ce_with_soft_prompt(
            processor=processor,
            model=model,
            messages=messages,
            images=images,
            action_text=action_name,
            soft_prompt=soft_prompt,
        )
        value = float(loss.item())
        if best_loss is None or value < best_loss:
            best_loss = value
            best_action = action_name
    return best_action, best_loss


def run_single_verification_stage2(
    cli,
    qwen: Qwen3VLWrapper,
    stage1_model,
    bridge,
    extractor: ClipHeatDepthExtractor,
    sample: dict,
    max_steps: int,
    movement_params: dict,
    action_select_mode: str,
    expected_feature_dim: int,
):
    scenario_id = sample["scenario_id"]
    anomaly_type = sample["anomaly_type"]
    anomaly_pos = sample["anomaly_position"]
    start_pose = sample["start_pose"]
    expected_steps = sample["expected_steps"]
    task_desc = sample["task_description"]

    print(f"\n=== Testing {scenario_id} (stage2) ===")
    anomaly_result = create_anomaly_at_position(
        cli, anomaly_type, (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"])
    )
    if anomaly_result is None:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": "Failed to create anomaly",
            "actual_steps": 0,
            "final_distance": float("inf"),
            "path_efficiency": 0.0,
            "anomaly_type": anomaly_type,
        }

    cli.set_posture(
        start_pose["x"],
        start_pose["y"],
        start_pose["z"],
        start_pose.get("rx", 0.0),
        start_pose.get("ry", 0.0),
        start_pose.get("rz", 0.0),
    )
    time.sleep(1.0)

    steps = 0
    stopped_by_model = False
    final_pose = None
    h_prev = None

    with torch.inference_mode():
        while steps < max_steps:
            pose = cli.get_pose()
            if pose is None:
                continue
            final_pose = pose

            cap = cli.capture()
            if cap is None:
                continue
            w, h, rgb_bytes, depth_bytes = cap
            rgb_pil = rgb_bytes_to_pil(w, h, rgb_bytes)
            depth_pil = depth_bytes_to_pil(w, h, depth_bytes)

            feat = extractor.extract(rgb_pil, depth_pil, task_desc)
            if int(feat.shape[0]) != int(expected_feature_dim):
                raise RuntimeError(
                    f"Feature dim mismatch in online verification: got {int(feat.shape[0])}, "
                    f"expected {int(expected_feature_dim)} from stage1 input_proj."
                )
            feat_t = torch.from_numpy(feat).to(next(stage1_model.parameters()).device).unsqueeze(0).unsqueeze(0)
            x = stage1_model.input_proj(feat_t)
            out, h_prev = stage1_model.gru(x, h_prev)
            h_t = out[:, 0, :]
            soft_prompt = bridge(h_t.to(next(bridge.parameters()).device))

            x0, y0, z0, _, _, rz0 = pose
            prompt = build_prompt(x0, y0, z0, rz0, task=task_desc)
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
            if action_select_mode == "rank_actions":
                action, best_ce = _rank_action_by_ce(
                    processor=qwen.processor,
                    model=qwen.model,
                    messages=messages,
                    images=[rgb_pil, depth_pil],
                    soft_prompt=soft_prompt,
                )
                action = action or "AUTO_FORWARD"
                print(f"  [{steps}] {action} (ce={best_ce:.4f})")
            else:
                raw = generate_action_with_soft_prompt(
                    processor=qwen.processor,
                    model=qwen.model,
                    messages=messages,
                    images=[rgb_pil, depth_pil],
                    soft_prompt=soft_prompt,
                    max_new_tokens=16,
                    do_sample=False,
                )
                action = parse_action(raw) or "AUTO_FORWARD"
                print(f"  [{steps}] {action}")
            if action == "AUTO_STOP_REACHED":
                stopped_by_model = True
                break
            dispatch_action(cli, action, **movement_params)
            steps += 1

    if final_pose is None:
        final_distance = float("inf")
    else:
        fx, fy, fz = final_pose[:3]
        final_distance = calculate_distance(
            (fx, fy, fz),
            (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"]),
        )

    success = stopped_by_model and final_distance <= 20.0
    path_efficiency = min(1.0, expected_steps / max(1, steps)) if steps > 0 else 0.0
    return {
        "scenario_id": scenario_id,
        "success": success,
        "stopped_by_model": stopped_by_model,
        "actual_steps": steps,
        "expected_steps": expected_steps,
        "final_distance": final_distance,
        "path_efficiency": path_efficiency,
        "anomaly_type": anomaly_type,
        "task_description": task_desc,
    }


def main():
    parser = argparse.ArgumentParser(description="Run stage2 verification with soft prompt bridge")
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_GTAV_20260403"))
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--stage2_lora_dir", default=None, help="Optional LoRA dir, default is sibling lora_best")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_device", default="cpu")
    parser.add_argument("--heatmap_size", type=int, default=48)
    parser.add_argument("--window_size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--tile_batch_size", type=int, default=128)
    parser.add_argument("--use_null_text_baseline", action="store_true")
    parser.add_argument(
        "--action_select_mode",
        choices=["rank_actions", "generate_parse"],
        default="rank_actions",
        help="Action selection mode in online verification",
    )
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--sleep_s", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--forward_step", type=float, default=5.0)
    parser.add_argument("--down_step", type=float, default=5.0)
    parser.add_argument("--yaw_step", type=float, default=15.0)
    parser.add_argument("--sample_limit", type=int, default=-1)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--verification_file", default=None)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    root_path = Path(args.root_path) if args.root_path else _repo_root()
    if args.verification_file is None:
        args.verification_file = str(root_path / "data" / "verification" / "samples.jsonl")
    if args.output_file is None:
        args.output_file = str(root_path / "data" / "verification" / "results_stage2.json")

    samples = read_jsonl(Path(args.verification_file))
    if not samples:
        print(f"No samples found in {args.verification_file}")
        return 1
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]

    qwen = Qwen3VLWrapper(args.model_dir).load()
    if args.stage2_lora_dir:
        lora_dir = Path(args.stage2_lora_dir)
    else:
        lora_dir = Path(args.stage2_ckpt).resolve().parent / "lora_best"
    if lora_dir.exists():
        try:
            from peft import PeftModel
            qwen._model = PeftModel.from_pretrained(qwen.model, lora_dir)
        except Exception as e:
            print(f"Warning: failed to load LoRA adapter from {lora_dir}: {e}")

    stage1_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model = _load_stage1_encoder(Path(args.stage1_ckpt), stage1_device)
    expected_feature_dim = int(stage1_model.input_proj.in_features)
    inferred_heatmap_size = int(round(math.sqrt(max(expected_feature_dim // 2, 1))))
    if inferred_heatmap_size * inferred_heatmap_size * 2 == expected_feature_dim:
        if int(args.heatmap_size) != inferred_heatmap_size:
            print(
                f"Warning: --heatmap_size={int(args.heatmap_size)} mismatches stage1 expected dim={expected_feature_dim}. "
                f"Auto-adjusting heatmap_size to {inferred_heatmap_size}."
            )
            args.heatmap_size = inferred_heatmap_size
    else:
        print(
            f"Warning: cannot infer heatmap_size from expected_feature_dim={expected_feature_dim}; "
            f"using user-provided heatmap_size={int(args.heatmap_size)}."
        )

    bridge = _load_stage2_bridge(Path(args.stage2_ckpt), stage1_device)
    extractor = ClipHeatDepthExtractor(
        model_name=args.clip_model_name,
        heatmap_size=int(args.heatmap_size),
        window_size=int(args.window_size),
        stride=int(args.stride),
        tile_batch_size=int(args.tile_batch_size),
        use_null_text_baseline=bool(args.use_null_text_baseline),
        device=args.clip_device,
    )

    cli = DroneSimClient(host=args.host, port=int(args.port))
    time.sleep(float(args.sleep_s))
    movement_params = build_movement_params(args)
    results = []
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def _save_progress(status: str, current_index: int, error_message: str = None):
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "total_samples": len(samples),
            "completed_samples": len(results),
            "current_index": int(current_index),
            "summary_statistics": calculate_summary_stats(results) if results else {},
            "results": results,
        }
        if error_message is not None:
            payload["error"] = str(error_message)
        write_json(output_file, payload)

    current_index = 0
    run_error = None
    try:
        cli.set_time(12, 0, 0)
        cli.create_camera()
        if args.fov is not None:
            cli.set_fov(float(args.fov))
        for i, sample in enumerate(samples):
            current_index = i
            print(f"\nProgress: {i+1}/{len(samples)}")
            result = run_single_verification_stage2(
                cli=cli,
                qwen=qwen,
                stage1_model=stage1_model,
                bridge=bridge,
                extractor=extractor,
                sample=sample,
                max_steps=int(args.max_steps),
                movement_params=movement_params,
                action_select_mode=str(args.action_select_mode),
                expected_feature_dim=expected_feature_dim,
            )
            results.append(result)
            # Save progress after every completed sample to avoid losing finished results.
            _save_progress(status="running", current_index=i + 1)
    except KeyboardInterrupt as e:
        run_error = e
        print("\nInterrupted by user. Saving partial results...")
        _save_progress(status="interrupted", current_index=current_index, error_message=str(e))
    except Exception as e:
        run_error = e
        print(f"\nVerification crashed: {e}")
        _save_progress(status="crashed", current_index=current_index, error_message=str(e))
    finally:
        try:
            cli.stop_camera()
            time.sleep(1.0)
            cli.restore_player()
            time.sleep(1.0)
        except Exception:
            pass

    if results:
        _save_progress(status="completed" if run_error is None else "partial", current_index=len(results))
        print(f"Results saved to: {output_file}")
        print_summary(results)
    if run_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
