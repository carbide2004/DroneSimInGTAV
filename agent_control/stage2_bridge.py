from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class Stage2BridgeConfig:
    hidden_dim: int = 512
    llm_dim: int = 3584
    num_soft_tokens: int = 16

    def to_dict(self) -> Dict:
        return asdict(self)


class Stage2BridgeModel(nn.Module):
    def __init__(self, config: Stage2BridgeConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_dim, config.num_soft_tokens * config.llm_dim)

    def forward(self, h_t: torch.Tensor):
        x = self.fc1(h_t)
        x = self.act(x)
        x = self.fc2(x)
        return x.view(h_t.shape[0], self.config.num_soft_tokens, self.config.llm_dim)
