from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class Stage2BridgeConfig:
    hidden_dim: int = 512
    llm_dim: int = 3584
    num_soft_tokens: int = 16
    smt_dim: int = 128
    num_smt_soft_tokens: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class Stage2BridgeModel(nn.Module):
    def __init__(self, config: Stage2BridgeConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_dim, config.num_soft_tokens * config.llm_dim)
        self.smt_fc = nn.Sequential(
            nn.Linear(config.smt_dim, config.smt_dim),
            nn.GELU(),
            nn.Linear(config.smt_dim, config.num_smt_soft_tokens * config.llm_dim),
        ) if config.num_smt_soft_tokens > 0 else None

    def forward(self, h_t: torch.Tensor, smt_context: torch.Tensor = None):
        x = self.fc1(h_t)
        x = self.act(x)
        x = self.fc2(x)
        soft_prompt = x.view(h_t.shape[0], self.config.num_soft_tokens, self.config.llm_dim)
        if smt_context is None or self.smt_fc is None:
            return soft_prompt
        smt_tokens = self.smt_fc(smt_context).view(
            h_t.shape[0],
            self.config.num_smt_soft_tokens,
            self.config.llm_dim,
        )
        return torch.cat([soft_prompt, smt_tokens], dim=1)
