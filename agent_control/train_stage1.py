import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from hf_auth import load_hf_token_from_env_file
from stage1_model import Stage1Config, Stage1GRUModel
from trajectory_dataset import build_stage1_dataloaders


ACTION_SET = (
    "AUTO_DOWN",
    "AUTO_UP",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_texts(texts: List[str], tokenizer, model, device: torch.device):
    if not texts:
        return None
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        out = model(**encoded)
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        else:
            emb = _masked_mean_pool(out.last_hidden_state, encoded["attention_mask"])
    return emb


class DINOFeatureStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"DINO cache manifest not found: {self.manifest_path}")
        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError("Invalid cache manifest format.")
        self.manifest = manifest

    def get(self, sample_id: str):
        rel = self.manifest.get(str(sample_id))
        if not rel:
            return None
        path = self.cache_dir / rel
        if not path.exists():
            return None
        arr = np.load(path)
        if arr.ndim != 1:
            return None
        return arr.astype(np.float32)


def _prepare_batch(batch, feature_store: DINOFeatureStore, max_len: int, device: torch.device):
    raw_batch = batch["raw_batch"]
    bsz = len(raw_batch)
    features = torch.zeros((bsz, max_len, 768), dtype=torch.float32, device=device)
    action_ids = torch.full((bsz, max_len), fill_value=-1, dtype=torch.long, device=device)
    valid_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    align_positions: List[Tuple[int, int]] = []
    align_texts: List[str] = []

    for i, item in enumerate(raw_batch):
        steps = item["steps"][:max_len]
        for j, step in enumerate(steps):
            action_id = int(step.get("action_id", -1))
            action_ids[i, j] = action_id
            feat = feature_store.get(step.get("sample_id"))
            if feat is None:
                continue
            features[i, j] = torch.from_numpy(feat).to(device=device)
            if action_id >= 0:
                valid_mask[i, j] = True
            awareness = step.get("awareness")
            if valid_mask[i, j] and isinstance(awareness, str) and awareness.strip():
                align_positions.append((i, j))
                align_texts.append(awareness.strip())

    return {
        "features": features,
        "action_ids": action_ids,
        "valid_mask": valid_mask,
        "align_positions": align_positions,
        "align_texts": align_texts,
    }


def _info_nce_loss(h: torch.Tensor, e: torch.Tensor, temperature: float):
    h = F.normalize(h, p=2, dim=-1)
    e = F.normalize(e, p=2, dim=-1)
    logits = torch.matmul(h, e.transpose(0, 1)) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    loss_h2e = F.cross_entropy(logits, labels)
    loss_e2h = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_h2e + loss_e2h)


def _run_epoch(
    model: Stage1GRUModel,
    loader,
    feature_store: DINOFeatureStore,
    tokenizer,
    text_encoder,
    optimizer,
    device: torch.device,
    max_len: int,
    lambda_b1: float,
    temperature: float,
    train: bool,
):
    model.train(mode=train)
    text_encoder.eval()
    total_loss = 0.0
    total_action = 0.0
    total_b1 = 0.0
    steps = 0

    for batch in loader:
        prepared = _prepare_batch(batch, feature_store, max_len=max_len, device=device)
        features = prepared["features"]
        action_ids = prepared["action_ids"]
        valid_mask = prepared["valid_mask"]

        h_seq, logits = model(features)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        action_flat = action_ids.reshape(-1)
        mask_flat = valid_mask.reshape(-1)

        if mask_flat.any():
            action_loss = F.cross_entropy(logits_flat[mask_flat], action_flat[mask_flat])
        else:
            action_loss = torch.tensor(0.0, device=device)

        align_positions = prepared["align_positions"]
        align_texts = prepared["align_texts"]
        if len(align_texts) >= 2:
            text_embeddings = _encode_texts(align_texts, tokenizer, text_encoder, device=device)
            h_vectors = torch.stack([h_seq[i, j] for i, j in align_positions], dim=0)
            h_proj = model.project_h(h_vectors)
            e_proj = model.project_e(text_embeddings)
            b1_loss = _info_nce_loss(h_proj, e_proj, temperature=temperature)
        else:
            b1_loss = torch.tensor(0.0, device=device)

        loss = action_loss + float(lambda_b1) * b1_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_action += float(action_loss.item())
        total_b1 += float(b1_loss.item())
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "action_loss": 0.0, "b1_loss": 0.0}
    return {
        "loss": total_loss / steps,
        "action_loss": total_action / steps,
        "b1_loss": total_b1 / steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage1 training: DINOv2 CLS + GRU(512) + InfoNCE")
    parser.add_argument(
        "--dataset_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Dataset JSON path",
    )
    parser.add_argument(
        "--cache_dir",
        default=str(_repo_root() / "dataset" / "dino_cache"),
        help="DINO feature cache directory",
    )
    parser.add_argument(
        "--output_dir",
        default=str(_repo_root() / "agent_control" / "checkpoints" / "stage1"),
        help="Checkpoint output directory",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_len", type=int, default=100, help="Fixed max trajectory length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lambda_b1", type=float, default=0.1, help="B1 loss weight")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio by trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--text_model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model",
    )
    parser.add_argument("--device", default="cuda", help="Training device")
    args = parser.parse_args()

    repo_root = _repo_root()
    hf_token = load_hf_token_from_env_file(repo_root)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, split_meta = build_stage1_dataloaders(
        dataset_json=Path(args.dataset_json),
        batch_size=int(args.batch_size),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        mode="sequence",
    )

    feature_store = DINOFeatureStore(Path(args.cache_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, token=hf_token)
    text_encoder = AutoModel.from_pretrained(args.text_model_name, token=hf_token).to(device)
    text_encoder.eval()

    text_dim = int(getattr(text_encoder.config, "hidden_size", 384))
    config = Stage1Config(
        input_dim=768,
        hidden_dim=512,
        action_dim=len(ACTION_SET),
        align_dim=256,
        text_dim=text_dim,
        max_len=int(args.max_len),
    )
    model = Stage1GRUModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    history = []

    config_payload = {
        "model": config.to_dict(),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_len": int(args.max_len),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_b1": float(args.lambda_b1),
            "temperature": float(args.temperature),
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "text_model_name": str(args.text_model_name),
            "cache_dir": str(Path(args.cache_dir)),
            "dataset_json": str(Path(args.dataset_json)),
        },
        "split_meta": split_meta,
    }
    _write_json(output_dir / "config.json", config_payload)

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            feature_store=feature_store,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            optimizer=optimizer,
            device=device,
            max_len=int(args.max_len),
            lambda_b1=float(args.lambda_b1),
            temperature=float(args.temperature),
            train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            feature_store=feature_store,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            optimizer=optimizer,
            device=device,
            max_len=int(args.max_len),
            lambda_b1=float(args.lambda_b1),
            temperature=float(args.temperature),
            train=False,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_act={train_metrics['action_loss']:.4f} "
            f"train_b1={train_metrics['b1_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_act={val_metrics['action_loss']:.4f} "
            f"val_b1={val_metrics['b1_loss']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_payload,
            "history": history,
        }
        torch.save(payload, output_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(payload, output_dir / "best.pt")

    _write_json(output_dir / "history.json", history)
    print(f"Training finished. best_val_loss={best_val:.4f}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
