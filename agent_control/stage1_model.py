from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class Stage1Config:
    input_dim: int = 768
    hidden_dim: int = 512
    action_dim: int = 6
    align_dim: int = 256
    text_dim: int = 384
    max_len: int = 0  # Deprecated: kept only for loading older checkpoints.

    def to_dict(self) -> Dict:
        return asdict(self)


class Stage1GRUModel(nn.Module):
    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.proj_h = nn.Linear(config.hidden_dim, config.align_dim)
        self.proj_e = nn.Linear(config.text_dim, config.align_dim)

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        h_seq, _ = self.gru(x)
        logits = self.action_head(h_seq)
        return h_seq, logits

    def project_h(self, h_seq: torch.Tensor):
        return self.proj_h(h_seq)

    def project_e(self, e: torch.Tensor):
        return self.proj_e(e)
