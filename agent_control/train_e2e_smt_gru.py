import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

from action_mapping import ACTIONS
from hf_auth import load_hf_token_from_env_file
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from smt_observation import AttentionBlock
from stage1_model import Stage1Config, Stage1GRUModel
from stage2_bridge import Stage2BridgeConfig, Stage2BridgeModel
from stage2_softprompt import forward_action_ce_with_soft_prompt


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _dist_is_active():
    return dist.is_available() and dist.is_initialized()


def _is_main_process():
    return not _dist_is_active() or dist.get_rank() == 0


def _unwrap_model(model):
    return getattr(model, "module", model)


def _setup_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA in this script.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=str(args.dist_backend), init_method="env://")
    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.device(f"cuda:{local_rank}"),
    }


def _cleanup_distributed():
    if _dist_is_active():
        dist.destroy_process_group()


def _print_rank0(message: str):
    if _is_main_process():
        print(message, flush=True)


def _parse_gpu_ids(gpu_ids_text: str):
    if gpu_ids_text is None:
        return []
    gpu_ids = []
    for part in str(gpu_ids_text).split(","):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    return gpu_ids


def _maybe_launch_with_gpu_ids(args):
    gpu_ids = _parse_gpu_ids(getattr(args, "gpu_ids", ""))
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ or not gpu_ids:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu_ids requires CUDA.")

    script_path = Path(__file__).resolve()
    cleaned = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--gpu_ids":
            skip_next = True
            continue
        if arg.startswith("--gpu_ids="):
            continue
        cleaned.append(arg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    if len(gpu_ids) == 1:
        env.setdefault("LOCAL_RANK", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
        args.device = "cuda:0"
        return False

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(gpu_ids)}",
        str(script_path),
        *cleaned,
    ]
    print(
        f"Launching DDP on physical GPUs {gpu_ids}; each process sees one local CUDA device.",
        flush=True,
    )
    raise SystemExit(subprocess.call(cmd, env=env))


def _masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_texts(texts: List[str], tokenizer, model, device: torch.device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return _masked_mean_pool(out.last_hidden_state, encoded["attention_mask"])


def _info_nce_loss(h: torch.Tensor, e: torch.Tensor, temperature: float):
    h = F.normalize(h, p=2, dim=-1)
    e = F.normalize(e, p=2, dim=-1)
    logits = torch.matmul(h, e.transpose(0, 1)) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def _safe_action_id(step: Dict):
    if "action_id" in step:
        action_id = int(step["action_id"])
        if 0 <= action_id < len(ACTIONS):
            return action_id
    action = step.get("action")
    name = ""
    if isinstance(action, dict):
        name = str(action.get("name", "")).strip()
    else:
        name = str(step.get("action_name", "")).strip()
    if name in ACTIONS:
        return ACTIONS.index(name)
    raise RuntimeError(f"Invalid action in sample_id={step.get('sample_id')}")


def _pose_dict(step: Dict):
    pose = step.get("pose")
    if not isinstance(pose, dict):
        raise RuntimeError(f"Missing pose in sample_id={step.get('sample_id')}")
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "rx": float(pose.get("rx", 0.0)),
        "ry": float(pose.get("ry", 0.0)),
        "rz": float(pose["rz"]),
    }


def _image_paths(step: Dict):
    observations = step.get("observations")
    if not isinstance(observations, dict):
        raise RuntimeError(f"Missing observations in sample_id={step.get('sample_id')}")
    rgb = observations.get("rgb")
    depth = observations.get("depth")
    if not isinstance(rgb, dict) or not isinstance(depth, dict):
        raise RuntimeError(f"Missing observations.rgb/depth in sample_id={step.get('sample_id')}")
    rgb_path = str(rgb.get("path", "")).strip()
    depth_path = str(depth.get("path", "")).strip()
    if not rgb_path or not depth_path:
        raise RuntimeError(f"Empty rgb/depth path in sample_id={step.get('sample_id')}")
    return rgb_path, depth_path


def _trajectory_id(entry: Dict, fallback_index: int):
    value = str(entry.get("trajectory_id", f"trajectory_{fallback_index:06d}")).strip()
    if not value:
        raise RuntimeError("trajectory_id must be non-empty")
    return value


def _step_index(entry: Dict):
    return int(entry.get("step_index", 0))


class TrajectoryDataset:
    def __init__(self, entries: Sequence[Dict], trajectory_ids: Sequence[str]):
        groups: Dict[str, List[Dict]] = {}
        for i, entry in enumerate(entries):
            groups.setdefault(_trajectory_id(entry, i), []).append(entry)
        for tid in groups:
            groups[tid] = sorted(groups[tid], key=_step_index)
        self.groups = groups
        self.trajectory_ids = [str(tid) for tid in trajectory_ids if str(tid) in groups]

    def __len__(self):
        return len(self.trajectory_ids)

    def __getitem__(self, index: int):
        trajectory_id = self.trajectory_ids[index]
        entries = self.groups[trajectory_id]
        steps = []
        for entry in entries:
            rgb_path, depth_path = _image_paths(entry)
            pose = _pose_dict(entry)
            action_id = _safe_action_id(entry)
            action = entry.get("action")
            action_name = ACTIONS[action_id]
            if isinstance(action, dict) and str(action.get("name", "")).strip():
                action_name = str(action["name"]).strip()
            steps.append(
                {
                    "sample_id": str(entry.get("sample_id", f"{trajectory_id}:{_step_index(entry):06d}")),
                    "step_index": _step_index(entry),
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "pose": pose,
                    "action_id": action_id,
                    "action_name": action_name,
                    "awareness": entry.get("awareness"),
                }
            )
        return {
            "trajectory_id": trajectory_id,
            "task": entries[0].get("task") if entries else None,
            "steps": steps,
        }


def _collate_trajectories(batch):
    return {"raw_batch": list(batch)}


def _split_trajectory_ids(entries: Sequence[Dict], val_ratio: float, seed: int):
    ids = sorted({_trajectory_id(entry, i) for i, entry in enumerate(entries)})
    random.Random(seed).shuffle(ids)
    if not ids:
        return [], []
    val_count = max(1, int(round(len(ids) * float(val_ratio)))) if val_ratio > 0 else 0
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    if not train_ids and val_ids:
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def _build_loaders(
    dataset_json: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    distributed: bool,
):
    data = _read_json(dataset_json)
    if not isinstance(data, list):
        raise RuntimeError("Dataset JSON must be a list.")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Entry {i} must be an object.")
        if int(entry.get("schema_version", 2)) != 2:
            raise RuntimeError(f"Entry {i} must have schema_version == 2.")
        _trajectory_id(entry, i)
        _image_paths(entry)
        _pose_dict(entry)
        _safe_action_id(entry)
    train_ids, val_ids = _split_trajectory_ids(data, val_ratio=val_ratio, seed=seed)
    train_dataset = TrajectoryDataset(data, train_ids)
    val_dataset = TrajectoryDataset(data, val_ids)
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
    except Exception as e:
        raise RuntimeError("PyTorch DataLoader is required for e2e training.") from e
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=int(seed)) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=int(seed)) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(num_workers),
        collate_fn=_collate_trajectories,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(num_workers),
        collate_fn=_collate_trajectories,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler, val_sampler, {
        "dataset_json": str(dataset_json),
        "total_trajectories": len(train_ids) + len(val_ids),
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
    }


def _load_rgbd_tensor(dataset_root: Path, rgb_rel: str, depth_rel: str, image_size: int):
    rgb_path = dataset_root / rgb_rel
    depth_path = dataset_root / depth_rel
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    with Image.open(rgb_path) as rgb_src:
        rgb = rgb_src.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    with Image.open(depth_path) as depth_src:
        depth = depth_src.convert("L").resize((image_size, image_size), Image.BILINEAR)
    rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
    depth_np = np.asarray(depth, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)
    depth_t = torch.from_numpy(depth_np).unsqueeze(0)
    return torch.cat([rgb_t, depth_t], dim=0)


def _prepare_batch(raw_batch, dataset_root: Path, image_size: int, max_len: int, device: torch.device):
    bsz = len(raw_batch)
    images = torch.zeros((bsz, max_len, 4, image_size, image_size), dtype=torch.float32, device=device)
    poses = torch.zeros((bsz, max_len, 6), dtype=torch.float32, device=device)
    action_ids = torch.full((bsz, max_len), fill_value=-1, dtype=torch.long, device=device)
    valid_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    valid_steps = []
    align_positions: List[Tuple[int, int]] = []
    align_texts: List[str] = []

    for i, item in enumerate(raw_batch):
        steps = item["steps"][:max_len]
        for j, step in enumerate(steps):
            images[i, j] = _load_rgbd_tensor(dataset_root, step["rgb_path"], step["depth_path"], image_size).to(device)
            pose = step["pose"]
            poses[i, j] = torch.tensor(
                [pose["x"], pose["y"], pose["z"], pose["rx"], pose["ry"], pose["rz"]],
                dtype=torch.float32,
                device=device,
            )
            action_ids[i, j] = int(step["action_id"])
            valid_mask[i, j] = True
            valid_steps.append((i, j, item, step))
            awareness = step.get("awareness")
            if isinstance(awareness, str) and awareness.strip():
                align_positions.append((i, j))
                align_texts.append(awareness.strip())
    return {
        "images": images,
        "poses": poses,
        "action_ids": action_ids,
        "valid_mask": valid_mask,
        "valid_steps": valid_steps,
        "align_positions": align_positions,
        "align_texts": align_texts,
    }


@dataclass
class E2ESmtConfig:
    d_model: int = 128
    visual_embed_dim: int = 96
    pose_embed_dim: int = 16
    action_embed_dim: int = 16
    num_actions: int = 6
    pose_scale: float = 50.0
    n_heads: int = 4
    d_ff: int = 256
    gru_hidden_dim: int = 512
    align_dim: int = 256
    text_dim: int = 384
    max_len: int = 100

    def to_dict(self):
        return asdict(self)


class RgbdVisualEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(64 * 2 * 2, embed_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc(h.flatten(start_dim=1))


class E2ESmtGruModel(nn.Module):
    def __init__(self, config: E2ESmtConfig):
        super().__init__()
        self.config = config
        self.visual_encoder = RgbdVisualEncoder(in_channels=4, embed_dim=config.visual_embed_dim)
        self.pose_encoder = nn.Linear(5, config.pose_embed_dim)
        self.action_encoder = nn.Embedding(config.num_actions + 1, config.action_embed_dim)
        obs_dim = config.visual_embed_dim + config.pose_embed_dim + config.action_embed_dim
        self.obs_fc = nn.Linear(obs_dim, config.d_model)
        self.pose_re_embed = nn.Linear(5, config.d_model)
        self.memory_encoder = AttentionBlock(config.d_model, config.n_heads, config.d_ff)
        self.memory_decoder = AttentionBlock(config.d_model, config.n_heads, config.d_ff)
        gru_cfg = Stage1Config(
            input_dim=config.d_model,
            hidden_dim=config.gru_hidden_dim,
            action_dim=config.num_actions,
            align_dim=config.align_dim,
            text_dim=config.text_dim,
            max_len=config.max_len,
        )
        self.gru = Stage1GRUModel(gru_cfg)

    def _current_relative_pose(self, batch_size: int, device):
        pose = torch.zeros((batch_size, 5), device=device)
        pose[:, 2] = 1.0
        pose[:, 4] = 1.0
        return pose

    def _raw_planar_pose(self, poses: torch.Tensor):
        rz_rad = poses[..., 5] * (torch.pi / 180.0)
        return torch.stack([poses[..., 0], poses[..., 1], rz_rad], dim=-1)

    def _relative_poses(self, poses: torch.Tensor, current_pose: torch.Tensor):
        x_c = current_pose[:, 0].unsqueeze(1)
        y_c = current_pose[:, 1].unsqueeze(1)
        rz_c = current_pose[:, 2].unsqueeze(1)
        x_i = poses[:, :, 0]
        y_i = poses[:, :, 1]
        rz_i = poses[:, :, 2]
        dx = x_i - x_c
        dy = y_i - y_c
        cos_c = torch.cos(rz_c)
        sin_c = torch.sin(rz_c)
        x_rel = dx * cos_c + dy * sin_c
        y_rel = -dx * sin_c + dy * cos_c
        d_rz = rz_i - rz_c
        steps = poses.shape[1]
        t_rel = torch.arange(steps - 1, -1, -1, device=poses.device, dtype=poses.dtype)
        temporal = torch.exp(-t_rel).unsqueeze(0).expand(poses.shape[0], -1)
        return torch.stack(
            [
                x_rel / float(self.config.pose_scale),
                y_rel / float(self.config.pose_scale),
                torch.cos(d_rz),
                torch.sin(d_rz),
                temporal,
            ],
            dim=-1,
        )

    def _prev_actions(self, action_ids: torch.Tensor):
        start_id = torch.full(
            (action_ids.shape[0], 1),
            fill_value=int(self.config.num_actions),
            device=action_ids.device,
            dtype=torch.long,
        )
        filled = action_ids.clamp(min=0, max=self.config.num_actions - 1)
        return torch.cat([start_id, filled[:, :-1]], dim=1)

    def forward(self, images: torch.Tensor, poses: torch.Tensor, action_ids: torch.Tensor):
        batch_size, steps = images.shape[:2]
        flat_images = images.reshape(batch_size * steps, *images.shape[2:])
        visual = self.visual_encoder(flat_images)
        rel_pose = self._current_relative_pose(batch_size * steps, images.device)
        prev_actions = self._prev_actions(action_ids).reshape(-1)
        obs = self.obs_fc(
            torch.cat(
                [
                    visual,
                    self.pose_encoder(rel_pose),
                    self.action_encoder(prev_actions),
                ],
                dim=-1,
            )
        ).view(batch_size, steps, -1)

        raw_poses = self._raw_planar_pose(poses)
        smt_contexts = []
        for t in range(steps):
            memory = obs[:, : t + 1, :]
            rel_poses = self._relative_poses(raw_poses[:, : t + 1, :], raw_poses[:, t, :])
            memory = memory + self.pose_re_embed(rel_poses)
            encoded = self.memory_encoder(memory, memory)
            decoded = self.memory_decoder(obs[:, t : t + 1, :], encoded)
            smt_contexts.append(decoded.squeeze(1))
        smt_seq = torch.stack(smt_contexts, dim=1)
        h_seq, logits = self.gru(smt_seq)
        return smt_seq, h_seq, logits

    def project_h(self, h_seq: torch.Tensor):
        return self.gru.project_h(h_seq)

    def project_e(self, text_embeddings: torch.Tensor):
        return self.gru.project_e(text_embeddings)


def _attach_lora(model, r: int, alpha: int, dropout: float, target_modules):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError("peft is required for LoRA end-to-end training.") from e
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


def _select_steps(valid_steps, steps_per_batch: int):
    if steps_per_batch <= 0 or len(valid_steps) <= steps_per_batch:
        return valid_steps
    return random.sample(valid_steps, k=int(steps_per_batch))


def _load_step_images(dataset_root: Path, step: dict):
    with Image.open(dataset_root / step["rgb_path"]) as rgb_raw:
        rgb_img = rgb_raw.convert("RGB")
    with Image.open(dataset_root / step["depth_path"]) as depth_raw:
        depth_img = depth_raw.convert("RGB")
    return rgb_img, depth_img


def _run_epoch(
    e2e_model: E2ESmtGruModel,
    bridge: Stage2BridgeModel,
    qwen: Qwen3VLWrapper,
    loader,
    dataset_root: Path,
    tokenizer,
    text_encoder,
    optimizer,
    device: torch.device,
    image_size: int,
    max_len: int,
    steps_per_batch: int,
    lambda_gru_action: float,
    lambda_awareness: float,
    lambda_vlm_action: float,
    temperature: float,
    train: bool,
    epoch: int,
    log_interval: int,
    rank: int,
):
    e2e_model.train(mode=train)
    bridge.train(mode=train)
    qwen.model.train(mode=train)
    text_encoder.eval()
    totals = {"loss": 0.0, "vlm_action_loss": 0.0, "gru_action_loss": 0.0, "awareness_loss": 0.0, "batches": 0}
    phase = "train" if train else "val"
    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        raw_batch = batch["raw_batch"]
        prepared = _prepare_batch(raw_batch, dataset_root, image_size=image_size, max_len=max_len, device=device)
        smt_seq, h_seq, gru_logits = e2e_model(prepared["images"], prepared["poses"], prepared["action_ids"])

        mask_flat = prepared["valid_mask"].reshape(-1)
        action_flat = prepared["action_ids"].reshape(-1)
        logits_flat = gru_logits.reshape(-1, gru_logits.shape[-1])
        if mask_flat.any():
            gru_action_loss = F.cross_entropy(logits_flat[mask_flat], action_flat[mask_flat])
        else:
            gru_action_loss = torch.tensor(0.0, device=device)

        align_texts = prepared["align_texts"]
        if len(align_texts) >= 2:
            text_embeddings = _encode_texts(align_texts, tokenizer, text_encoder, device=device)
            e2e_base = _unwrap_model(e2e_model)
            h_vectors = torch.stack([h_seq[i, j] for i, j in prepared["align_positions"]], dim=0)
            awareness_loss = _info_nce_loss(
                e2e_base.project_h(h_vectors),
                e2e_base.project_e(text_embeddings),
                temperature=temperature,
            )
        else:
            awareness_loss = torch.tensor(0.0, device=device)

        selected_steps = _select_steps(prepared["valid_steps"], steps_per_batch=steps_per_batch)
        vlm_losses = []
        for i, j, item, step in selected_steps:
            pose = step["pose"]
            task = item.get("task") or "find the closest burning car"
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
            rgb_img, depth_img = _load_step_images(dataset_root, step)
            soft_prompt = bridge(h_seq[i, j].unsqueeze(0), smt_context=smt_seq[i, j].unsqueeze(0))
            vlm_losses.append(
                forward_action_ce_with_soft_prompt(
                    processor=qwen.processor,
                    model=qwen.model,
                    messages=messages,
                    images=[rgb_img, depth_img],
                    action_text=step["action_name"],
                    soft_prompt=soft_prompt,
                )
            )
        vlm_action_loss = torch.stack(vlm_losses).mean() if vlm_losses else torch.tensor(0.0, device=device)
        loss = (
            float(lambda_vlm_action) * vlm_action_loss
            + float(lambda_gru_action) * gru_action_loss
            + float(lambda_awareness) * awareness_loss
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.item())
        totals["vlm_action_loss"] += float(vlm_action_loss.item())
        totals["gru_action_loss"] += float(gru_action_loss.item())
        totals["awareness_loss"] += float(awareness_loss.item())
        totals["batches"] += 1
        if rank == 0 and int(log_interval) > 0 and (batch_idx % int(log_interval) == 0):
            elapsed = max(time.time() - start_time, 1e-6)
            avg = {k: totals[k] / max(totals["batches"], 1) for k in totals if k != "batches"}
            _print_rank0(
                f"[{phase}] epoch={epoch} step={batch_idx}/{len(loader)} "
                f"loss={avg['loss']:.4f} vlm={avg['vlm_action_loss']:.4f} "
                f"gru={avg['gru_action_loss']:.4f} aw={avg['awareness_loss']:.4f} "
                f"batches_per_s={totals['batches'] / elapsed:.3f}"
            )

    if totals["batches"] == 0:
        return totals
    if _dist_is_active():
        stats = torch.tensor(
            [
                totals["loss"],
                totals["vlm_action_loss"],
                totals["gru_action_loss"],
                totals["awareness_loss"],
                float(totals["batches"]),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        totals = {
            "loss": float(stats[0].item()),
            "vlm_action_loss": float(stats[1].item()),
            "gru_action_loss": float(stats[2].item()),
            "awareness_loss": float(stats[3].item()),
            "batches": int(stats[4].item()),
        }
    batches = totals["batches"]
    return {k: (v / batches if k != "batches" else v) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="End-to-end SMT observation + GRU awareness + VLM soft-token training")
    parser.add_argument("--dataset_json", default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"))
    parser.add_argument("--dataset_root", default=str(_repo_root() / "dataset"))
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_merged"))
    parser.add_argument("--output_dir", default=str(_repo_root() / "agent_control" / "checkpoints" / "e2e_smt_gru"))
    parser.add_argument("--text_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--steps_per_batch", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu_ids",
        default="",
        help="Comma-separated physical GPU ids. If set to multiple ids, this script launches one DDP process per GPU.",
    )
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--visual_embed_dim", type=int, default=96)
    parser.add_argument("--pose_embed_dim", type=int, default=16)
    parser.add_argument("--action_embed_dim", type=int, default=16)
    parser.add_argument("--gru_hidden_dim", type=int, default=512)
    parser.add_argument("--align_dim", type=int, default=256)
    parser.add_argument("--smt_heads", type=int, default=4)
    parser.add_argument("--smt_ff_dim", type=int, default=256)
    parser.add_argument("--pose_scale", type=float, default=50.0)
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--num_smt_soft_tokens", type=int, default=4)
    parser.add_argument("--lr_e2e", type=float, default=1e-4)
    parser.add_argument("--lr_bridge", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_vlm_action", type=float, default=1.0)
    parser.add_argument("--lambda_gru_action", type=float, default=0.2)
    parser.add_argument("--lambda_awareness", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    args = parser.parse_args()

    if _maybe_launch_with_gpu_ids(args):
        return 0

    dist_state = _setup_distributed(args)
    rank = int(dist_state["rank"])
    local_rank = int(dist_state["local_rank"])
    world_size = int(dist_state["world_size"])
    device = dist_state["device"]

    repo_root = _repo_root()
    hf_token = load_hf_token_from_env_file(repo_root)
    random.seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    output_dir = Path(args.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if _dist_is_active():
        dist.barrier()
    dataset_root = Path(args.dataset_root)
    train_loader, val_loader, train_sampler, val_sampler, split_meta = _build_loaders(
        dataset_json=Path(args.dataset_json),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        distributed=bool(dist_state["distributed"]),
    )
    _print_rank0(
        f"Distributed: enabled={bool(dist_state['distributed'])} world_size={world_size} "
        f"device={device} output_dir={output_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, token=hf_token)
    text_encoder = AutoModel.from_pretrained(args.text_model_name, token=hf_token).to(device)
    text_encoder.eval()
    text_dim = int(getattr(text_encoder.config, "hidden_size", 384))

    e2e_cfg = E2ESmtConfig(
        d_model=int(args.d_model),
        visual_embed_dim=int(args.visual_embed_dim),
        pose_embed_dim=int(args.pose_embed_dim),
        action_embed_dim=int(args.action_embed_dim),
        num_actions=len(ACTIONS),
        pose_scale=float(args.pose_scale),
        n_heads=int(args.smt_heads),
        d_ff=int(args.smt_ff_dim),
        gru_hidden_dim=int(args.gru_hidden_dim),
        align_dim=int(args.align_dim),
        text_dim=text_dim,
        max_len=int(args.max_len),
    )
    e2e_model = E2ESmtGruModel(e2e_cfg).to(device)

    qwen = Qwen3VLWrapper(args.model_dir, torch_dtype="auto", device_map={"": str(device)}).load()
    lora_targets = [x.strip() for x in str(args.lora_targets).split(",") if x.strip()]
    qwen._model = _attach_lora(
        qwen.model,
        r=int(args.lora_r),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=lora_targets,
    )
    llm_dim = int(getattr(qwen.model.config, "hidden_size", 3584))
    bridge_cfg = Stage2BridgeConfig(
        hidden_dim=int(args.gru_hidden_dim),
        llm_dim=llm_dim,
        num_soft_tokens=int(args.num_soft_tokens),
        smt_dim=int(args.d_model),
        num_smt_soft_tokens=int(args.num_smt_soft_tokens),
    )
    bridge = Stage2BridgeModel(bridge_cfg).to(device)

    if bool(dist_state["distributed"]):
        e2e_model = DDP(e2e_model, device_ids=[local_rank], output_device=local_rank)
        bridge = DDP(bridge, device_ids=[local_rank], output_device=local_rank)
        qwen._model = DDP(
            qwen.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    lora_params = [p for p in qwen.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": e2e_model.parameters(), "lr": float(args.lr_e2e)},
            {"params": bridge.parameters(), "lr": float(args.lr_bridge)},
            {"params": lora_params, "lr": float(args.lr_lora)},
        ],
        weight_decay=float(args.weight_decay),
    )

    config_payload = {
        "e2e_smt_gru": e2e_cfg.to_dict(),
        "bridge": bridge_cfg.to_dict(),
        "train": {
            "dataset_json": str(Path(args.dataset_json)),
            "dataset_root": str(dataset_root),
            "model_dir": str(Path(args.model_dir)),
            "text_model_name": str(args.text_model_name),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_len": int(args.max_len),
            "image_size": int(args.image_size),
            "steps_per_batch": int(args.steps_per_batch),
            "lambda_vlm_action": float(args.lambda_vlm_action),
            "lambda_gru_action": float(args.lambda_gru_action),
            "lambda_awareness": float(args.lambda_awareness),
            "temperature": float(args.temperature),
            "lr_e2e": float(args.lr_e2e),
            "lr_bridge": float(args.lr_bridge),
            "lr_lora": float(args.lr_lora),
            "lora_targets": lora_targets,
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "seed": int(args.seed),
        },
        "split_meta": split_meta,
    }
    if _is_main_process():
        _write_json(output_dir / "config.json", config_payload)

    best_val = float("inf")
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            e2e_model=e2e_model,
            bridge=bridge,
            qwen=qwen,
            loader=train_loader,
            dataset_root=dataset_root,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            optimizer=optimizer,
            device=device,
            image_size=int(args.image_size),
            max_len=int(args.max_len),
            steps_per_batch=int(args.steps_per_batch),
            lambda_gru_action=float(args.lambda_gru_action),
            lambda_awareness=float(args.lambda_awareness),
            lambda_vlm_action=float(args.lambda_vlm_action),
            temperature=float(args.temperature),
            train=True,
            epoch=epoch,
            log_interval=int(args.log_interval),
            rank=rank,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                e2e_model=e2e_model,
                bridge=bridge,
                qwen=qwen,
                loader=val_loader,
                dataset_root=dataset_root,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                optimizer=optimizer,
                device=device,
                image_size=int(args.image_size),
                max_len=int(args.max_len),
                steps_per_batch=int(args.steps_per_batch),
                lambda_gru_action=float(args.lambda_gru_action),
                lambda_awareness=float(args.lambda_awareness),
                lambda_vlm_action=float(args.lambda_vlm_action),
                temperature=float(args.temperature),
                train=False,
                epoch=epoch,
                log_interval=0,
                rank=rank,
            )

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if _is_main_process():
            history.append(record)
            _print_rank0(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_vlm={train_metrics['vlm_action_loss']:.4f} "
                f"train_gru={train_metrics['gru_action_loss']:.4f} train_aw={train_metrics['awareness_loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_vlm={val_metrics['vlm_action_loss']:.4f} "
                f"val_gru={val_metrics['gru_action_loss']:.4f} val_aw={val_metrics['awareness_loss']:.4f}"
            )
            payload = {
                "epoch": epoch,
                "e2e_state_dict": _unwrap_model(e2e_model).state_dict(),
                "bridge_state_dict": _unwrap_model(bridge).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_payload,
                "history": history,
            }
            torch.save(payload, output_dir / "last.pt")
            _unwrap_model(qwen.model).save_pretrained(output_dir / "lora_last")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(payload, output_dir / "best.pt")
                _unwrap_model(qwen.model).save_pretrained(output_dir / "lora_best")

    if _is_main_process():
        _write_json(output_dir / "history.json", history)
        _print_rank0(f"E2E SMT+GRU training finished. best_val_loss={best_val:.4f}")
        _print_rank0(f"Output dir: {output_dir}")
    _cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
