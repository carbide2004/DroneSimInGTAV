import argparse
from contextlib import contextmanager
import json
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
from smt_observation import SmtObservationConfig, SmtObservationEncoder
from stage1_model import Stage1Config, Stage1GRUModel
from stage2_bridge import Stage2BridgeConfig, Stage2BridgeModel
from stage2_softprompt import forward_action_ce_with_soft_prompt, generate_action_with_soft_prompt
from train_e2e_smt_gru import E2ESmtConfig, E2ESmtGruModel
from verification_runtime import (
    build_movement_params,
    calculate_distance,
    create_anomaly_at_position,
    read_jsonl,
    write_json,
)


def _sync_cuda_for_timing(enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _timed_stage(timing: dict, stage: str, sync_cuda: bool):
    _sync_cuda_for_timing(sync_cuda)
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda_for_timing(sync_cuda)
        elapsed = time.perf_counter() - start
        timing[stage] = timing.get(stage, 0.0) + float(elapsed)


def _format_timing(timing: dict):
    total = max(float(timing.get("step_total", timing.get("sample_total", 0.0))), 1e-8)
    parts = []
    for key, value in sorted(timing.items(), key=lambda item: item[1], reverse=True):
        if key in {"index", "step_total", "sample_total"}:
            continue
        parts.append(f"{key}={value:.3f}s/{value / total * 100.0:.1f}%")
    return ", ".join(parts)


def _safe_path_name(value: str):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value)).strip("_") or "sample"


def _add_timing_totals(total: dict, timing: dict):
    for key, value in timing.items():
        if key == "index":
            continue
        total[key] = total.get(key, 0.0) + float(value)


def _aggregate_timing(results):
    aggregate = {}
    step_count = 0
    for result in results:
        timing = result.get("timing")
        if not isinstance(timing, dict):
            continue
        for key, value in timing.get("sample", {}).items():
            aggregate[f"sample.{key}"] = aggregate.get(f"sample.{key}", 0.0) + float(value)
        steps_total = timing.get("steps_total")
        if isinstance(steps_total, dict):
            step_count += int(timing.get("step_count", 0))
            for key, value in steps_total.items():
                aggregate[f"step.{key}"] = aggregate.get(f"step.{key}", 0.0) + float(value)
            continue
        for step in timing.get("steps", []):
            if not isinstance(step, dict):
                continue
            step_count += 1
            for key, value in step.items():
                if key == "index":
                    continue
                aggregate[f"step.{key}"] = aggregate.get(f"step.{key}", 0.0) + float(value)
    return aggregate, step_count


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        rgb_gray = np.array(rgb_img.convert("L"), dtype=np.float32) / 255.0
        rgb_resized = _resize_float_map(rgb_gray, self.heatmap_size)
        rgb_norm = np.clip(rgb_resized, 0.0, 1.0).astype(np.float32)
        return np.concatenate(
            [heatmap_norm.reshape(-1), depth_norm.reshape(-1), rgb_norm.reshape(-1)],
            axis=0,
        ).astype(np.float32)


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
        smt_dim=int(config.get("smt_dim", payload.get("config", {}).get("smt", {}).get("d_model", 128))),
        num_smt_soft_tokens=int(config.get("num_smt_soft_tokens", 0)),
    )
    bridge = Stage2BridgeModel(bridge_cfg).to(device)
    bridge.load_state_dict(payload["bridge_state_dict"], strict=True)
    bridge.eval()
    return bridge, payload


def _load_smt_encoder(stage2_payload: dict, feature_dim: int, device: torch.device):
    config = stage2_payload.get("config", {})
    smt_cfg_data = config.get("smt")
    stage2_cfg = config.get("stage2", {})
    if not isinstance(smt_cfg_data, dict) or int(stage2_cfg.get("num_smt_soft_tokens", 0)) <= 0:
        return None
    smt_cfg = SmtObservationConfig(**{**smt_cfg_data, "feature_dim": int(feature_dim)})
    encoder = SmtObservationEncoder(smt_cfg).to(device)
    state = stage2_payload.get("smt_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Invalid SMT stage2 checkpoint: missing smt_state_dict")
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


def _load_e2e_policy(e2e_ckpt: Path, device: torch.device):
    payload = torch.load(e2e_ckpt, map_location=device)
    config = payload.get("config", {})
    e2e_cfg_data = config.get("e2e_smt_gru")
    bridge_cfg_data = config.get("bridge")
    if not isinstance(e2e_cfg_data, dict) or not isinstance(bridge_cfg_data, dict):
        raise RuntimeError("Invalid e2e checkpoint: missing e2e_smt_gru/bridge config")

    e2e_model = E2ESmtGruModel(E2ESmtConfig(**e2e_cfg_data)).to(device)
    e2e_state = payload.get("e2e_state_dict")
    if not isinstance(e2e_state, dict):
        raise RuntimeError("Invalid e2e checkpoint: missing e2e_state_dict")
    e2e_model.load_state_dict(e2e_state, strict=True)
    e2e_model.eval()

    bridge = Stage2BridgeModel(Stage2BridgeConfig(**bridge_cfg_data)).to(device)
    bridge_state = payload.get("bridge_state_dict")
    if not isinstance(bridge_state, dict):
        raise RuntimeError("Invalid e2e checkpoint: missing bridge_state_dict")
    bridge.load_state_dict(bridge_state, strict=True)
    bridge.eval()
    return e2e_model, bridge, payload


def _pil_rgbd_tensor(rgb_img: Image.Image, depth_img: Image.Image, image_size: int, device: torch.device):
    rgb = rgb_img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    depth = depth_img.convert("L").resize((image_size, image_size), Image.BILINEAR)
    rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
    depth_np = np.asarray(depth, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)
    depth_t = torch.from_numpy(depth_np).unsqueeze(0)
    return torch.cat([rgb_t, depth_t], dim=0).to(device)


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


def _calculate_stage2_metrics(results):
    if not results:
        return {
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "sr": 0.0,
            "spl": 0.0,
            "avg_stop_distance_overall": 0.0,
            "avg_stop_distance_success": 0.0,
        }

    total = len(results)
    successful = sum(1 for r in results if bool(r.get("success")))
    sr = successful / total if total > 0 else 0.0

    spl_values = []
    for r in results:
        expected = float(r.get("expected_steps", 0))
        actual = float(r.get("actual_steps", 0))
        success = bool(r.get("success"))
        if expected <= 0:
            spl = 0.0
        else:
            spl = (expected / max(expected, actual, 1.0)) if success else 0.0
        spl_values.append(float(spl))
    spl = float(sum(spl_values) / total) if total > 0 else 0.0

    finite_distances = []
    finite_success_distances = []
    for r in results:
        d = float(r.get("final_distance", float("inf")))
        if math.isfinite(d):
            finite_distances.append(d)
            if bool(r.get("success")):
                finite_success_distances.append(d)
    avg_stop_distance_overall = (
        float(sum(finite_distances) / len(finite_distances)) if finite_distances else float("inf")
    )
    avg_stop_distance_success = (
        float(sum(finite_success_distances) / len(finite_success_distances))
        if finite_success_distances
        else float("inf")
    )

    return {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": total - successful,
        "sr": sr,
        "spl": spl,
        "avg_stop_distance_overall": avg_stop_distance_overall,
        "avg_stop_distance_success": avg_stop_distance_success,
    }


def _print_stage2_summary(results):
    metrics = _calculate_stage2_metrics(results)
    timing_aggregate, timing_steps = _aggregate_timing(results)
    print(f"\n{'=' * 50}")
    print("STAGE2 VERIFICATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successful: {metrics['successful_samples']}")
    print(f"Failed: {metrics['failed_samples']}")
    print(f"SR: {metrics['sr']:.4f}")
    print(f"SPL: {metrics['spl']:.4f}")
    if math.isfinite(metrics["avg_stop_distance_success"]):
        print(f"Avg stop distance (success): {metrics['avg_stop_distance_success']:.2f}m")
    else:
        print("Avg stop distance (success): inf")
    if math.isfinite(metrics["avg_stop_distance_overall"]):
        print(f"Avg stop distance (overall): {metrics['avg_stop_distance_overall']:.2f}m")
    else:
        print("Avg stop distance (overall): inf")
    if timing_aggregate:
        print("\nTiming totals:")
        for key, value in sorted(timing_aggregate.items(), key=lambda item: item[1], reverse=True):
            if key.endswith(".step_total") or key.endswith(".sample_total"):
                continue
            if key.startswith("step.") and timing_steps > 0:
                print(f"{key}: total={value:.3f}s, avg/step={value / timing_steps:.3f}s")
            else:
                print(f"{key}: total={value:.3f}s")
        if timing_steps > 0:
            step_total = timing_aggregate.get("step.step_total", 0.0)
            print(f"step.total: total={step_total:.3f}s, avg/step={step_total / timing_steps:.3f}s")
    print(f"{'=' * 50}")


def run_single_verification_stage2(
    cli,
    qwen: Qwen3VLWrapper,
    stage1_model,
    smt_encoder,
    e2e_model,
    bridge,
    extractor: ClipHeatDepthExtractor,
    sample: dict,
    max_steps: int,
    movement_params: dict,
    action_select_mode: str,
    expected_feature_dim: int,
    policy_mode: str,
    image_size: int,
    profile_timing: bool,
    timing_sync_cuda: bool,
    timing_step_log: bool,
    save_step_traces: bool,
    trace_root: Path,
):
    sample_start = time.perf_counter()
    sample_timing = {}
    step_timing_totals = {}
    step_count = 0
    trace_steps = []
    scenario_id = sample["scenario_id"]
    anomaly_type = sample["anomaly_type"]
    anomaly_pos = sample["anomaly_position"]
    start_pose = sample["start_pose"]
    expected_steps = sample["expected_steps"]
    task_desc = sample["task_description"]
    sample_trace_dir = None
    if save_step_traces:
        sample_trace_dir = trace_root / _safe_path_name(str(scenario_id))
        (sample_trace_dir / "RGB").mkdir(parents=True, exist_ok=True)
        (sample_trace_dir / "Depth").mkdir(parents=True, exist_ok=True)

    print(f"\n=== Testing {scenario_id} (stage2) ===")
    with _timed_stage(sample_timing, "create_anomaly", bool(profile_timing and timing_sync_cuda)):
        anomaly_result = create_anomaly_at_position(
            cli, anomaly_type, (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"])
        )
    if anomaly_result is None:
        sample_timing["sample_total"] = float(time.perf_counter() - sample_start)
        timing_payload = {"sample": sample_timing, "steps_total": step_timing_totals, "step_count": 0}
        result = {
            "scenario_id": scenario_id,
            "success": False,
            "error": "Failed to create anomaly",
            "actual_steps": 0,
            "final_distance": float("inf"),
            "path_efficiency": 0.0,
            "anomaly_type": anomaly_type,
            "timing": timing_payload,
        }
        if sample_trace_dir is not None:
            result["trace_dir"] = str(sample_trace_dir)
        return result

    with _timed_stage(sample_timing, "set_start_pose_wait", bool(profile_timing and timing_sync_cuda)):
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
    prev_action_id = int(getattr(stage1_model.config, "action_dim", len(ACTIONS))) if stage1_model is not None else len(ACTIONS)
    feature_history = []
    pose_history = []
    prev_action_history = []
    e2e_image_history = []
    e2e_pose_history = []
    e2e_action_history = []

    with torch.inference_mode():
        while steps < max_steps:
            step_start = time.perf_counter()
            step_timing = {"index": int(steps)}
            with _timed_stage(step_timing, "get_pose", bool(profile_timing and timing_sync_cuda)):
                pose = cli.get_pose()
            if pose is None:
                continue
            final_pose = pose

            with _timed_stage(step_timing, "capture", bool(profile_timing and timing_sync_cuda)):
                cap = cli.capture()
            if cap is None:
                continue
            w, h, rgb_bytes, depth_bytes = cap
            with _timed_stage(step_timing, "decode_rgbd", bool(profile_timing and timing_sync_cuda)):
                rgb_pil = rgb_bytes_to_pil(w, h, rgb_bytes)
                depth_pil = depth_bytes_to_pil(w, h, depth_bytes)
            rgb_trace_rel = None
            depth_trace_rel = None
            if sample_trace_dir is not None:
                with _timed_stage(step_timing, "save_observation", bool(profile_timing and timing_sync_cuda)):
                    rgb_trace_rel = f"RGB/step_{steps:06d}.png"
                    depth_trace_rel = f"Depth/step_{steps:06d}.png"
                    rgb_pil.save(sample_trace_dir / rgb_trace_rel)
                    depth_pil.save(sample_trace_dir / depth_trace_rel)

            soft_prompt = None
            raw = None
            best_ce = None
            if policy_mode == "stage2_softprompt":
                with _timed_stage(step_timing, "stage2_clip_extract", bool(profile_timing and timing_sync_cuda)):
                    feat = extractor.extract(rgb_pil, depth_pil, task_desc)
                with _timed_stage(step_timing, "stage2_gru_smt_bridge", bool(profile_timing and timing_sync_cuda)):
                    if int(feat.shape[0]) != int(expected_feature_dim):
                        raise RuntimeError(
                            f"Feature dim mismatch in online verification: got {int(feat.shape[0])}, "
                            f"expected {int(expected_feature_dim)} from stage1 input_proj."
                        )
                    feat_t = torch.from_numpy(feat).to(next(stage1_model.parameters()).device).unsqueeze(0).unsqueeze(0)
                    x = stage1_model.input_proj(feat_t)
                    out, h_prev = stage1_model.gru(x, h_prev)
                    h_t = out[:, 0, :]
                    smt_context = None
                    if smt_encoder is not None:
                        pose_t = torch.tensor(
                            [[pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]]],
                            dtype=torch.float32,
                            device=next(smt_encoder.parameters()).device,
                        )
                        feature_history.append(feat_t.to(next(smt_encoder.parameters()).device))
                        pose_history.append(pose_t.unsqueeze(1))
                        prev_action_history.append(
                            torch.tensor(
                                [[prev_action_id]],
                                dtype=torch.long,
                                device=next(smt_encoder.parameters()).device,
                            )
                        )
                        smt_features = torch.cat(feature_history, dim=1)
                        smt_poses = torch.cat(pose_history, dim=1)
                        smt_prev_actions = torch.cat(prev_action_history, dim=1)
                        smt_seq = smt_encoder(
                            smt_features,
                            smt_poses,
                            smt_prev_actions,
                            action_ids_are_previous=True,
                        )
                        smt_context = smt_seq[:, -1, :].to(next(bridge.parameters()).device)
                    soft_prompt = bridge(h_t.to(next(bridge.parameters()).device), smt_context=smt_context)
            elif policy_mode == "e2e_smt_gru":
                with _timed_stage(step_timing, "e2e_rgbd_preprocess", bool(profile_timing and timing_sync_cuda)):
                    e2e_device = next(e2e_model.parameters()).device
                    image_t = _pil_rgbd_tensor(rgb_pil, depth_pil, int(image_size), e2e_device)
                    pose_t = torch.tensor(
                        [[pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]]],
                        dtype=torch.float32,
                        device=e2e_device,
                    )
                with _timed_stage(step_timing, "e2e_smt_gru_bridge", bool(profile_timing and timing_sync_cuda)):
                    e2e_image_history.append(image_t.unsqueeze(0).unsqueeze(0))
                    e2e_pose_history.append(pose_t.unsqueeze(1))
                    # E2ESmtGruModel 会在内部平移 action_ids，因此当前位置可以使用占位值。
                    action_seq = list(e2e_action_history) + [0]
                    actions_t = torch.tensor([action_seq], dtype=torch.long, device=e2e_device)
                    images_t = torch.cat(e2e_image_history, dim=1)
                    poses_t = torch.cat(e2e_pose_history, dim=1)
                    smt_seq, h_seq, _ = e2e_model(images_t, poses_t, actions_t)
                    h_t = h_seq[:, -1, :]
                    smt_context = smt_seq[:, -1, :]
                    soft_prompt = bridge(
                        h_t.to(next(bridge.parameters()).device),
                        smt_context=smt_context.to(next(bridge.parameters()).device),
                    )

            with _timed_stage(step_timing, "build_prompt", bool(profile_timing and timing_sync_cuda)):
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
            if policy_mode in ("stage2_softprompt", "e2e_smt_gru") and action_select_mode == "rank_actions":
                with _timed_stage(step_timing, "vlm_rank_actions", bool(profile_timing and timing_sync_cuda)):
                    action, best_ce = _rank_action_by_ce(
                        processor=qwen.processor,
                        model=qwen.model,
                        messages=messages,
                        images=[rgb_pil, depth_pil],
                        soft_prompt=soft_prompt,
                    )
                action = action or "AUTO_FORWARD"
                print(f"  [{steps}] {action} (ce={best_ce:.4f})")
            elif policy_mode in ("stage2_softprompt", "e2e_smt_gru"):
                with _timed_stage(step_timing, "vlm_generate_parse", bool(profile_timing and timing_sync_cuda)):
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
            else:
                # 不使用 GRU 软提示，直接通过 VLA 生成动作。
                with _timed_stage(step_timing, "vla_direct_generate_parse", bool(profile_timing and timing_sync_cuda)):
                    raw = qwen.generate_action(
                        prompt_text=prompt,
                        rgb_pil=rgb_pil,
                        depth_pil=depth_pil,
                        max_new_tokens=16,
                        do_sample=False,
                    )
                action = parse_action(raw) or "AUTO_FORWARD"
                print(f"  [{steps}] {action}")
            if action == "AUTO_STOP_REACHED":
                stopped_by_model = True
                step_timing["step_total"] = float(time.perf_counter() - step_start)
                _add_timing_totals(step_timing_totals, step_timing)
                step_count += 1
                if sample_trace_dir is not None:
                    trace_steps.append({
                        "step": int(steps),
                        "pose": {
                            "x": float(pose[0]),
                            "y": float(pose[1]),
                            "z": float(pose[2]),
                            "rx": float(pose[3]),
                            "ry": float(pose[4]),
                            "rz": float(pose[5]),
                        },
                        "rgb_path": rgb_trace_rel,
                        "depth_path": depth_trace_rel,
                        "action": action,
                        "raw_output": raw,
                        "best_ce": float(best_ce) if best_ce is not None else None,
                        "timing": step_timing,
                    })
                if profile_timing and timing_step_log:
                    print(f"      timing: total={step_timing['step_total']:.3f}s, {_format_timing(step_timing)}")
                break
            with _timed_stage(step_timing, "dispatch_action", bool(profile_timing and timing_sync_cuda)):
                dispatch_action(cli, action, **movement_params)
            if policy_mode == "stage2_softprompt":
                prev_action_id = ACTIONS.index(action) if action in ACTIONS else 0
            elif policy_mode == "e2e_smt_gru":
                e2e_action_history.append(ACTIONS.index(action) if action in ACTIONS else 0)
            step_timing["step_total"] = float(time.perf_counter() - step_start)
            _add_timing_totals(step_timing_totals, step_timing)
            step_count += 1
            if sample_trace_dir is not None:
                trace_steps.append({
                    "step": int(steps),
                    "pose": {
                        "x": float(pose[0]),
                        "y": float(pose[1]),
                        "z": float(pose[2]),
                        "rx": float(pose[3]),
                        "ry": float(pose[4]),
                        "rz": float(pose[5]),
                    },
                    "rgb_path": rgb_trace_rel,
                    "depth_path": depth_trace_rel,
                    "action": action,
                    "raw_output": raw,
                    "best_ce": float(best_ce) if best_ce is not None else None,
                    "timing": step_timing,
                })
            if profile_timing and timing_step_log:
                print(f"      timing: total={step_timing['step_total']:.3f}s, {_format_timing(step_timing)}")
            steps += 1

    if final_pose is None:
        final_distance = float("inf")
        final_position = None
    else:
        fx, fy, fz = final_pose[:3]
        final_position = {"x": float(fx), "y": float(fy), "z": float(fz)}
        final_distance = calculate_distance(
            (fx, fy, fz),
            (anomaly_pos["x"], anomaly_pos["y"], anomaly_pos["z"]),
        )
    target_position = {
        "x": float(anomaly_pos["x"]),
        "y": float(anomaly_pos["y"]),
        "z": float(anomaly_pos["z"]),
    }

    success = stopped_by_model and final_distance <= 25.0
    if expected_steps > 0:
        spl = (expected_steps / max(float(expected_steps), float(max(steps, 1)))) if success else 0.0
    else:
        spl = 0.0
    if final_position is None:
        print(
            "  Final: drone=(unknown), "
            f"target=({target_position['x']:.2f}, {target_position['y']:.2f}, {target_position['z']:.2f}), "
            "distance=inf"
        )
    else:
        print(
            f"  Final: drone=({final_position['x']:.2f}, {final_position['y']:.2f}, {final_position['z']:.2f}), "
            f"target=({target_position['x']:.2f}, {target_position['y']:.2f}, {target_position['z']:.2f}), "
            f"distance={final_distance:.2f}"
        )
    sample_timing["sample_total"] = float(time.perf_counter() - sample_start)
    if profile_timing:
        print(f"  Sample timing: total={sample_timing['sample_total']:.3f}s, {_format_timing(sample_timing)}")
    timing_payload = {
        "sample": sample_timing,
        "steps_total": step_timing_totals,
        "step_count": int(step_count),
    }
    result = {
        "scenario_id": scenario_id,
        "success": success,
        "stopped_by_model": stopped_by_model,
        "actual_steps": steps,
        "expected_steps": expected_steps,
        "final_distance": final_distance,
        "final_position": final_position,
        "target_position": target_position,
        "spl": float(spl),
        "anomaly_type": anomaly_type,
        "task_description": task_desc,
        "timing": timing_payload,
    }
    if sample_trace_dir is not None:
        result["trace_dir"] = str(sample_trace_dir)
        write_json(
            sample_trace_dir / "steps.json",
            {
                "scenario_id": scenario_id,
                "anomaly_type": anomaly_type,
                "task_description": task_desc,
                "target_position": target_position,
                "start_pose": start_pose,
                "result": {
                    "success": success,
                    "stopped_by_model": stopped_by_model,
                    "actual_steps": steps,
                    "expected_steps": expected_steps,
                    "final_distance": final_distance,
                    "final_position": final_position,
                    "spl": float(spl),
                },
                "timing": timing_payload,
                "steps": trace_steps,
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run stage2 verification with soft prompt bridge")
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_GTAV_20260403"))
    parser.add_argument("--stage1_ckpt", default=None)
    parser.add_argument("--stage2_ckpt", default=None)
    parser.add_argument("--e2e_ckpt", default=None, help="Path to e2e_smt_gru best.pt/last.pt")
    parser.add_argument("--stage2_lora_dir", default=None, help="Optional LoRA dir, default is sibling lora_best")
    parser.add_argument(
        "--policy_mode",
        choices=["stage2_softprompt", "e2e_smt_gru", "vla_direct"],
        default="stage2_softprompt",
        help="Policy to run in online verification",
    )
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_device", default="cpu")
    parser.add_argument("--heatmap_size", type=int, default=48)
    parser.add_argument("--window_size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--tile_batch_size", type=int, default=128)
    parser.add_argument("--use_null_text_baseline", action="store_true")
    parser.add_argument("--image_size", type=int, default=96, help="RGBD resize size for policy_mode=e2e_smt_gru")
    parser.add_argument(
        "--action_select_mode",
        choices=["rank_actions", "generate_parse"],
        default="rank_actions",
        help="Action selection mode in online verification",
    )
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--sleep_s", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--forward_step", type=float, default=5.0)
    parser.add_argument("--down_step", type=float, default=5.0)
    parser.add_argument("--yaw_step", type=float, default=15.0)
    parser.add_argument("--sample_limit", type=int, default=-1)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--verification_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument(
        "--save_step_traces",
        action="store_true",
        help="Save per-sample steps.json plus RGBD observations for trajectory visualization",
    )
    parser.add_argument(
        "--trace_dir",
        default=None,
        help="Directory for --save_step_traces; default is <output_file_parent>/traces_<output_file_stem>",
    )
    parser.add_argument("--resume_from_output", dest="resume_from_output", action="store_true", help="Resume from existing output JSON")
    parser.add_argument("--no_resume_from_output", dest="resume_from_output", action="store_false", help="Disable resume from output JSON")
    parser.add_argument("--profile_timing", action="store_true", help="Print per-step online verification timing")
    parser.add_argument(
        "--no_timing_sync_cuda",
        dest="timing_sync_cuda",
        action="store_false",
        help="Do not synchronize CUDA before/after timed stages",
    )
    parser.add_argument(
        "--no_timing_step_log",
        dest="timing_step_log",
        action="store_false",
        help="Disable per-step timing logs while keeping timing in result JSON",
    )
    parser.set_defaults(resume_from_output=True)
    parser.set_defaults(timing_sync_cuda=True, timing_step_log=True)
    args = parser.parse_args()
    if args.policy_mode == "stage2_softprompt":
        if not args.stage1_ckpt or not args.stage2_ckpt:
            raise RuntimeError("policy_mode=stage2_softprompt requires --stage1_ckpt and --stage2_ckpt")
    if args.policy_mode == "e2e_smt_gru" and not args.e2e_ckpt:
        raise RuntimeError("policy_mode=e2e_smt_gru requires --e2e_ckpt")
    if args.policy_mode == "vla_direct" and args.action_select_mode == "rank_actions":
        print("Warning: policy_mode=vla_direct does not support rank_actions; switching to generate_parse.")
        args.action_select_mode = "generate_parse"

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
    elif args.e2e_ckpt:
        lora_dir = Path(args.e2e_ckpt).resolve().parent / "lora_best"
    elif args.stage2_ckpt:
        lora_dir = Path(args.stage2_ckpt).resolve().parent / "lora_best"
    else:
        lora_dir = None
    if lora_dir is not None and lora_dir.exists():
        try:
            from peft import PeftModel
            qwen._model = PeftModel.from_pretrained(qwen.model, lora_dir)
        except Exception as e:
            print(f"Warning: failed to load LoRA adapter from {lora_dir}: {e}")

    stage1_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model = None
    smt_encoder = None
    e2e_model = None
    bridge = None
    extractor = None
    expected_feature_dim = -1
    if args.policy_mode == "stage2_softprompt":
        stage1_model = _load_stage1_encoder(Path(args.stage1_ckpt), stage1_device)
        expected_feature_dim = int(stage1_model.input_proj.in_features)
        inferred = None
        for channels in (3, 2):
            size = int(round(math.sqrt(max(expected_feature_dim // channels, 1))))
            if size * size * channels == expected_feature_dim:
                inferred = (size, channels)
                break
        if inferred is not None:
            inferred_heatmap_size, inferred_channels = inferred
            if int(args.heatmap_size) != inferred_heatmap_size:
                print(
                    f"Warning: --heatmap_size={int(args.heatmap_size)} mismatches stage1 expected dim={expected_feature_dim} "
                    f"(channels={inferred_channels}). Auto-adjusting heatmap_size to {inferred_heatmap_size}."
                )
                args.heatmap_size = inferred_heatmap_size
        else:
            print(
                f"Warning: cannot infer heatmap_size from expected_feature_dim={expected_feature_dim}; "
                f"using user-provided heatmap_size={int(args.heatmap_size)}."
            )

        bridge, stage2_payload = _load_stage2_bridge(Path(args.stage2_ckpt), stage1_device)
        smt_encoder = _load_smt_encoder(stage2_payload, expected_feature_dim, stage1_device)
        extractor = ClipHeatDepthExtractor(
            model_name=args.clip_model_name,
            heatmap_size=int(args.heatmap_size),
            window_size=int(args.window_size),
            stride=int(args.stride),
            tile_batch_size=int(args.tile_batch_size),
            use_null_text_baseline=bool(args.use_null_text_baseline),
            device=args.clip_device,
        )
    elif args.policy_mode == "e2e_smt_gru":
        e2e_model, bridge, _ = _load_e2e_policy(Path(args.e2e_ckpt), stage1_device)

    cli = DroneSimClient(host=args.host, port=int(args.port))
    time.sleep(float(args.sleep_s))
    movement_params = build_movement_params(args)
    results = []
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    trace_root = Path(args.trace_dir) if args.trace_dir else output_file.parent / f"traces_{output_file.stem}"
    if bool(args.save_step_traces):
        trace_root.mkdir(parents=True, exist_ok=True)
    start_index = 0

    def _save_progress(status: str, current_index: int, error_message: str = None):
        timing_aggregate, timing_steps = _aggregate_timing(results)
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "total_samples": len(samples),
            "completed_samples": len(results),
            "current_index": int(current_index),
            "summary_statistics": _calculate_stage2_metrics(results),
            "timing_statistics": {
                "step_count": int(timing_steps),
                "totals_sec": timing_aggregate,
            },
            "results": results,
        }
        if error_message is not None:
            payload["error"] = str(error_message)
        write_json(output_file, payload)

    if bool(args.resume_from_output) and output_file.exists():
        try:
            previous = _read_json(output_file)
            if isinstance(previous, dict):
                previous_results = previous.get("results")
                previous_completed = int(previous.get("completed_samples", 0))
                if isinstance(previous_results, list):
                    results = previous_results
                    previous_completed = min(previous_completed, len(results))
                    start_index = min(max(previous_completed, 0), len(samples))
                else:
                    start_index = min(max(previous_completed, 0), len(samples))
                if start_index > 0:
                    print(f"Resuming from sample index {start_index}/{len(samples)} using {output_file}")
        except Exception as e:
            print(f"Warning: failed to load resume state from {output_file}: {e}")

    current_index = 0
    run_error = None
    try:
        cli.set_time(12, 0, 0)
        cli.create_camera()
        if args.fov is not None:
            cli.set_fov(float(args.fov))
        for i in range(start_index, len(samples)):
            sample = samples[i]
            current_index = i
            print(f"\nProgress: {i+1}/{len(samples)}")
            result = run_single_verification_stage2(
                cli=cli,
                qwen=qwen,
                stage1_model=stage1_model,
                smt_encoder=smt_encoder,
                e2e_model=e2e_model,
                bridge=bridge,
                extractor=extractor,
                sample=sample,
                max_steps=int(args.max_steps),
                movement_params=movement_params,
                action_select_mode=str(args.action_select_mode),
                expected_feature_dim=expected_feature_dim,
                policy_mode=str(args.policy_mode),
                image_size=int(args.image_size),
                profile_timing=bool(args.profile_timing),
                timing_sync_cuda=bool(args.timing_sync_cuda),
                timing_step_log=bool(args.timing_step_log),
                save_step_traces=bool(args.save_step_traces),
                trace_root=trace_root,
            )
            results.append(result)
            # 每完成一个样本就保存进度，避免丢失已完成结果。
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
        _print_stage2_summary(results)
    if run_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
