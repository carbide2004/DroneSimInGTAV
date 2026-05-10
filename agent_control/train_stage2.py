import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from hf_auth import load_hf_token_from_env_file
from prompting import build_prompt
from qwen3vl_wrapper import Qwen3VLWrapper
from smt_observation import SmtObservationConfig, SmtObservationEncoder
from stage1_model import Stage1Config, Stage1GRUModel
from stage2_bridge import Stage2BridgeConfig, Stage2BridgeModel
from stage2_softprompt import forward_action_ce_with_soft_prompt
from train_stage1 import FeatureCacheStore
from trajectory_dataset import (
    build_stage1_dataloaders,
    build_stage1_dataloaders_from_manifest,
    build_stage1_dataloaders_from_split_json,
)


def _dist_is_active():
    return dist.is_available() and dist.is_initialized()


def _is_main_process():
    return not _dist_is_active() or dist.get_rank() == 0


def _unwrap_model(model):
    return getattr(model, "module", model)


def _print_rank0(message: str):
    if _is_main_process():
        print(message, flush=True)


def _parse_gpu_ids(gpu_ids_text: str):
    if gpu_ids_text is None:
        return []
    gpu_ids = []
    for part in str(gpu_ids_text).split(","):
        part = part.strip()
        if part:
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
    print(f"Launching DDP on physical GPUs {gpu_ids}; each process sees one local CUDA device.", flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


def _setup_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        return {"distributed": False, "rank": 0, "local_rank": 0, "world_size": 1, "device": device}
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


def _wrap_loader_for_distributed(loader, shuffle: bool, seed: int):
    if not _dist_is_active():
        return loader, None
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
    except Exception as e:
        raise RuntimeError("PyTorch DataLoader/DistributedSampler is required for DDP stage2 training.") from e
    sampler = DistributedSampler(loader.dataset, shuffle=bool(shuffle), seed=int(seed))
    wrapped = DataLoader(
        loader.dataset,
        batch_size=int(loader.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(loader.num_workers),
        collate_fn=loader.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return wrapped, sampler


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
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, stage1_cfg


def _attach_lora(model, r: int, alpha: int, dropout: float, target_modules):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError("Stage2 LoRA 需要 peft，请先安装 peft。") from e

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    return peft_model


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
    poses = torch.zeros((batch_size, max_len, 6), dtype=torch.float32, device=device)
    action_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    valid_steps = []

    for i, item in enumerate(raw_batch):
        steps = item["steps"][:max_len]
        for j, step in enumerate(steps):
            sample_id = step["sample_id"]
            feat = feature_store.get(sample_id)
            if feat is None:
                raise RuntimeError(f"Missing feature cache for sample_id={sample_id}")
            features[i, j] = torch.from_numpy(feat).to(device=device)
            pose = step.get("pose") or {}
            poses[i, j] = torch.tensor(
                [
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                    float(pose.get("rx", 0.0)),
                    float(pose.get("ry", 0.0)),
                    float(pose.get("rz", 0.0)),
                ],
                dtype=torch.float32,
                device=device,
            )
            action_ids[i, j] = int(step.get("action_id", 0))
            valid_steps.append((i, j, item, step))
    return features, poses, action_ids, valid_steps


def _select_steps(valid_steps, steps_per_batch: int):
    if steps_per_batch <= 0 or len(valid_steps) <= steps_per_batch:
        return valid_steps
    return random.sample(valid_steps, k=int(steps_per_batch))


def _run_epoch(
    stage1_model: Stage1GRUModel,
    smt_encoder: SmtObservationEncoder,
    bridge: Stage2BridgeModel,
    qwen_wrapper: Qwen3VLWrapper,
    loader,
    feature_store: FeatureCacheStore,
    dataset_root: Path,
    device: torch.device,
    max_len: int,
    steps_per_batch: int,
    optimizer=None,
):
    train = optimizer is not None
    smt_encoder.train(mode=train)
    bridge.train(mode=train)
    qwen_wrapper.model.train(mode=train)
    total_loss = 0.0
    total_steps = 0

    for batch in loader:
        raw_batch = batch["raw_batch"]
        features, poses, action_ids, valid_steps = _prepare_features(raw_batch, feature_store, max_len=max_len, device=device)
        selected_steps = _select_steps(valid_steps, steps_per_batch=steps_per_batch)
        if not selected_steps:
            continue

        with torch.no_grad():
            h_seq, _ = stage1_model(features)
        smt_seq = smt_encoder(features, poses, action_ids)

        losses = []
        for i, j, item, step in selected_steps:
            pose = step.get("pose") or {}
            task = item.get("task") or "find the closest burning car"
            action_name = str(step.get("action_name") or "").strip()
            if not action_name:
                raise RuntimeError(f"Missing action_name for sample_id={step.get('sample_id')}")

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
            smt_context = smt_seq[i, j].unsqueeze(0)
            soft_prompt = bridge(h_t, smt_context=smt_context)
            loss = forward_action_ce_with_soft_prompt(
                processor=qwen_wrapper.processor,
                model=qwen_wrapper.model,
                messages=messages,
                images=[rgb_img, depth_img],
                action_text=action_name,
                soft_prompt=soft_prompt,
            )
            losses.append(loss)

        if not losses:
            continue

        batch_loss = torch.stack(losses, dim=0).mean()
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_loss += float(batch_loss.item())
        total_steps += 1

    if _dist_is_active():
        stats = torch.tensor([total_loss, float(total_steps)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_steps = int(stats[1].item())

    if total_steps == 0:
        return {"loss": 0.0, "batches": 0}
    return {"loss": total_loss / total_steps, "batches": total_steps}


def main():
    parser = argparse.ArgumentParser(description="Stage2 training: frozen Stage1 + soft prompt bridge + Qwen LoRA")
    parser.add_argument("--dataset_json", default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"))
    parser.add_argument("--cache_dir", default=str(_repo_root() / "dataset" / "clip_cache"))
    parser.add_argument("--dataset_root", default=str(_repo_root() / "dataset"))
    parser.add_argument("--stage1_ckpt", required=True, help="Path to stage1 best.pt or last.pt")
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "models" / "qwen3_vl_sft_GTAV_20260403"))
    parser.add_argument("--output_dir", default=str(_repo_root() / "agent_control" / "checkpoints" / "stage2"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split_manifest_json", default=None, help="Fixed split manifest JSON path")
    parser.add_argument("--train_json", default=None, help="Fixed train split JSON path")
    parser.add_argument("--val_json", default=None, help="Fixed val split JSON path")
    parser.add_argument("--steps_per_batch", type=int, default=4, help="Timesteps sampled per trajectory batch")
    parser.add_argument("--num_soft_tokens", type=int, default=16)
    parser.add_argument("--num_smt_soft_tokens", type=int, default=4)
    parser.add_argument("--smt_dim", type=int, default=128)
    parser.add_argument("--smt_feature_embed_dim", type=int, default=128)
    parser.add_argument("--smt_pose_embed_dim", type=int, default=16)
    parser.add_argument("--smt_action_embed_dim", type=int, default=16)
    parser.add_argument("--smt_heads", type=int, default=4)
    parser.add_argument("--smt_ff_dim", type=int, default=256)
    parser.add_argument("--smt_pose_scale", type=float, default=50.0)
    parser.add_argument("--smt_use_factorization", action="store_true")
    parser.add_argument("--smt_num_centers", type=int, default=64)
    parser.add_argument("--lr_bridge", type=float, default=1e-4)
    parser.add_argument("--lr_smt", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument(
        "--gpu_ids",
        default="",
        help="Comma-separated physical GPU ids. If multiple ids are set, launches one DDP process per GPU.",
    )
    parser.add_argument("--dist_backend", default="nccl")
    args = parser.parse_args()

    if _maybe_launch_with_gpu_ids(args):
        return 0

    dist_state = _setup_distributed(args)
    rank = int(dist_state["rank"])
    local_rank = int(dist_state["local_rank"])
    world_size = int(dist_state["world_size"])
    device = dist_state["device"]

    repo_root = _repo_root()
    load_hf_token_from_env_file(repo_root)

    random.seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank)

    device_str = str(device)
    output_dir = Path(args.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if _dist_is_active():
        dist.barrier()

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
    train_loader, train_sampler = _wrap_loader_for_distributed(train_loader, shuffle=True, seed=int(args.seed))
    val_loader, val_sampler = _wrap_loader_for_distributed(val_loader, shuffle=False, seed=int(args.seed))
    _print_rank0(
        f"Distributed: enabled={bool(dist_state['distributed'])} world_size={world_size} "
        f"device={device} output_dir={output_dir}"
    )

    feature_store = FeatureCacheStore(Path(args.cache_dir))
    dataset_root = Path(args.dataset_root)

    stage1_model, stage1_cfg = _load_stage1_encoder(Path(args.stage1_ckpt), device=device)
    qwen = Qwen3VLWrapper(args.model_dir, torch_dtype="auto", device_map={"":device_str}).load()

    lora_targets = [x.strip() for x in str(args.lora_targets).split(",") if x.strip()]
    qwen._model = _attach_lora(
        qwen.model,
        r=int(args.lora_r),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
        target_modules=lora_targets,
    )
    qwen.model.train()

    llm_dim = int(getattr(qwen.model.config, "hidden_size", 3584))
    bridge_cfg = Stage2BridgeConfig(
        hidden_dim=int(stage1_cfg.hidden_dim),
        llm_dim=llm_dim,
        num_soft_tokens=int(args.num_soft_tokens),
        smt_dim=int(args.smt_dim),
        num_smt_soft_tokens=int(args.num_smt_soft_tokens),
    )
    bridge = Stage2BridgeModel(bridge_cfg).to(device)
    smt_cfg = SmtObservationConfig(
        feature_dim=int(feature_store.feature_dim),
        d_model=int(args.smt_dim),
        feature_embed_dim=int(args.smt_feature_embed_dim),
        pose_embed_dim=int(args.smt_pose_embed_dim),
        action_embed_dim=int(args.smt_action_embed_dim),
        num_actions=int(stage1_cfg.action_dim),
        pose_scale=float(args.smt_pose_scale),
        n_heads=int(args.smt_heads),
        d_ff=int(args.smt_ff_dim),
        num_centers=int(args.smt_num_centers),
        use_factorization=bool(args.smt_use_factorization),
    )
    smt_encoder = SmtObservationEncoder(smt_cfg).to(device)

    if bool(dist_state["distributed"]):
        bridge = DDP(bridge, device_ids=[local_rank], output_device=local_rank)
        smt_encoder = DDP(smt_encoder, device_ids=[local_rank], output_device=local_rank)
        qwen._model = DDP(
            qwen.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    lora_params = [p for p in qwen.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": bridge.parameters(), "lr": float(args.lr_bridge)},
            {"params": smt_encoder.parameters(), "lr": float(args.lr_smt)},
            {"params": lora_params, "lr": float(args.lr_lora)},
        ],
        weight_decay=float(args.weight_decay),
    )

    config_payload = {
        "stage1": {
            "ckpt": str(Path(args.stage1_ckpt)),
            "hidden_dim": int(stage1_cfg.hidden_dim),
            "frozen": True,
        },
        "stage2": {
            "num_soft_tokens": int(args.num_soft_tokens),
            "num_smt_soft_tokens": int(args.num_smt_soft_tokens),
            "smt_dim": int(args.smt_dim),
            "llm_dim": llm_dim,
            "steps_per_batch": int(args.steps_per_batch),
            "loss": "action_ce_only",
            "vla_trainable": "lora_only",
            "lora_targets": lora_targets,
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
        },
        "smt": smt_cfg.to_dict(),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_len": int(args.max_len),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "lr_bridge": float(args.lr_bridge),
            "lr_smt": float(args.lr_smt),
            "lr_lora": float(args.lr_lora),
            "weight_decay": float(args.weight_decay),
            "dataset_json": str(Path(args.dataset_json)),
            "cache_dir": str(Path(args.cache_dir)),
            "dataset_root": str(dataset_root),
            "split_manifest_json": str(Path(args.split_manifest_json)) if args.split_manifest_json else None,
            "train_json": str(Path(args.train_json)) if args.train_json else None,
            "val_json": str(Path(args.val_json)) if args.val_json else None,
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
            stage1_model=stage1_model,
            smt_encoder=smt_encoder,
            bridge=bridge,
            qwen_wrapper=qwen,
            loader=train_loader,
            feature_store=feature_store,
            dataset_root=dataset_root,
            device=device,
            max_len=int(args.max_len),
            steps_per_batch=int(args.steps_per_batch),
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                stage1_model=stage1_model,
                smt_encoder=smt_encoder,
                bridge=bridge,
                qwen_wrapper=qwen,
                loader=val_loader,
                feature_store=feature_store,
                dataset_root=dataset_root,
                device=device,
                max_len=int(args.max_len),
                steps_per_batch=int(args.steps_per_batch),
                optimizer=None,
            )

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if _is_main_process():
            history.append(record)
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_batches={train_metrics['batches']} "
                f"val_loss={val_metrics['loss']:.4f} val_batches={val_metrics['batches']}",
                flush=True,
            )

            payload = {
                "epoch": epoch,
                "bridge_state_dict": _unwrap_model(bridge).state_dict(),
                "smt_state_dict": _unwrap_model(smt_encoder).state_dict(),
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
        print(f"Stage2 training finished. best_val_loss={best_val:.4f}")
        print(f"Output dir: {output_dir}")
    _cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
