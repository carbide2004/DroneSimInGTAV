from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmtObservationConfig:
    feature_dim: int
    d_model: int = 128
    feature_embed_dim: int = 128
    pose_embed_dim: int = 16
    action_embed_dim: int = 16
    num_actions: int = 6
    pose_scale: float = 50.0
    n_heads: int = 4
    d_ff: int = 256
    num_centers: int = 64
    use_factorization: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query=x, key=y, value=y, need_weights=False)
        h = self.ln1(x + attn_out)
        return self.ln2(h + self.ffn(h))


class SmtObservationEmbedding(nn.Module):
    def __init__(self, config: SmtObservationConfig):
        super().__init__()
        self.config = config
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pose_encoder = nn.Linear(5, config.pose_embed_dim)
        # 额外保留一个 id 表示轨迹开始时的上一动作。
        self.action_encoder = nn.Embedding(config.num_actions + 1, config.action_embed_dim)
        total_dim = config.feature_embed_dim + config.pose_embed_dim + config.action_embed_dim
        self.out = nn.Linear(total_dim, config.d_model)

    def forward(
        self,
        features: torch.Tensor,
        relative_pose: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        e_feature = self.feature_encoder(features)
        e_pose = self.pose_encoder(relative_pose)
        e_action = self.action_encoder(prev_actions.long())
        return self.out(torch.cat([e_feature, e_pose, e_action], dim=-1))


class SmtObservationEncoder(nn.Module):
    def __init__(self, config: SmtObservationConfig):
        super().__init__()
        self.config = config
        self.obs_embed = SmtObservationEmbedding(config)
        self.pose_re_embed = nn.Linear(5, config.d_model)
        self.self_attn = AttentionBlock(config.d_model, config.n_heads, config.d_ff)
        self.compress_attn = AttentionBlock(config.d_model, config.n_heads, config.d_ff)
        self.broadcast_attn = AttentionBlock(config.d_model, config.n_heads, config.d_ff)
        self.decoder = AttentionBlock(config.d_model, config.n_heads, config.d_ff)

    def _current_relative_pose(self, batch_size: int, device) -> torch.Tensor:
        pose = torch.zeros((batch_size, 5), device=device)
        pose[:, 2] = 1.0
        pose[:, 4] = 1.0
        return pose

    def _relative_poses(self, poses: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
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

    def _centers(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.shape[1] <= int(self.config.num_centers):
            return memory
        step = max(memory.shape[1] // int(self.config.num_centers), 1)
        centers = memory[:, ::step, :]
        return centers[:, : int(self.config.num_centers), :]

    def _encode_memory(self, memory: torch.Tensor) -> torch.Tensor:
        if self.config.use_factorization and memory.shape[1] > int(self.config.num_centers):
            centers = self._centers(memory)
            compressed = self.compress_attn(centers, memory)
            return self.broadcast_attn(memory, compressed)
        return self.self_attn(memory, memory)

    def _pose_tensor(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.shape[-1] < 6:
            raise RuntimeError("SMT observation encoder expects pose tensors with at least 6 values.")
        # SMT 记忆只需要平面位置和偏航角。
        rz_rad = poses[..., 5] * (torch.pi / 180.0)
        return torch.stack([poses[..., 0], poses[..., 1], rz_rad], dim=-1)

    def _prev_actions(self, action_ids: torch.Tensor, action_ids_are_previous: bool) -> torch.Tensor:
        if action_ids_are_previous:
            return action_ids.long().clamp(0, self.config.num_actions)
        start_id = torch.full(
            (action_ids.shape[0], 1),
            fill_value=int(self.config.num_actions),
            device=action_ids.device,
            dtype=torch.long,
        )
        return torch.cat([start_id, action_ids[:, :-1].long()], dim=1).clamp(0, self.config.num_actions)

    def forward(
        self,
        features: torch.Tensor,
        poses: torch.Tensor,
        action_ids: torch.Tensor,
        action_ids_are_previous: bool = False,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise RuntimeError("features must have shape [B, T, D]")
        if poses.ndim != 3:
            raise RuntimeError("poses must have shape [B, T, P]")
        if action_ids.ndim != 2:
            raise RuntimeError("action_ids must have shape [B, T]")
        if features.shape[:2] != poses.shape[:2] or features.shape[:2] != action_ids.shape[:2]:
            raise RuntimeError("features, poses, and action_ids must share [B, T] shape")

        batch_size, steps, _ = features.shape
        current_rel_pose = self._current_relative_pose(batch_size * steps, features.device)
        prev_actions = self._prev_actions(action_ids, action_ids_are_previous).reshape(-1)
        obs_embeds = self.obs_embed(
            features.reshape(batch_size * steps, -1),
            current_rel_pose,
            prev_actions,
        ).view(batch_size, steps, -1)

        raw_poses = self._pose_tensor(poses)
        outputs = []
        for t in range(steps):
            memory = obs_embeds[:, : t + 1, :]
            memory_poses = raw_poses[:, : t + 1, :]
            rel_poses = self._relative_poses(memory_poses, raw_poses[:, t, :])
            memory = memory + self.pose_re_embed(rel_poses)
            encoded = self._encode_memory(memory)
            decoded = self.decoder(obs_embeds[:, t : t + 1, :], encoded)
            outputs.append(decoded.squeeze(1))
        return torch.stack(outputs, dim=1)
