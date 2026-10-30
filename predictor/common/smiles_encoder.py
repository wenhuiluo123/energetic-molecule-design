from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
PoolingType = Literal['cls', 'mean']

@dataclass
class SmilesModelConfig:
    vocab_size: int
    pad_token_id: int = 0
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1
    pooling: PoolingType = 'mean'
    num_targets: int = 1
    reg_hidden: int | None = None
    head_type: str = 'gelu'
    lr: float = 0.0003
    weight_decay: float = 0.01
    warmup_steps: int = 0
    loss_name: str = 'mse'

class SmilesBackbone(nn.Module):
    def __init__(self, cfg: SmilesModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_encoder = PositionalEncoding1D(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_hidden,
            dropout=cfg.dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.cfg.pad_token_id).long()
        emb = self.embedding(input_ids)
        emb = emb + self.pos_encoder(emb)
        mask = self.generate_square_subsequent_mask(input_ids.size(1), input_ids.device)
        hidden = self.decoder(emb, mask=mask, src_key_padding_mask=~attention_mask.bool())
        valid = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return hidden, pooled


class ResidualPredictionHead(nn.Module):

    def __init__(self, d_model: int, num_targets: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.out = nn.Linear(d_model, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return self.out(x + h)


class PredictionHead(nn.Module):

    def __init__(self, d_model: int, num_targets: int, dropout: float, *, head_type: str='gelu', reg_hidden: int | None=None) -> None:
        super().__init__()
        head_name = str(head_type or 'gelu').lower()
        if head_name in {'mlp', 'relu'}:
            hidden_dim = reg_hidden if reg_hidden is not None else d_model
            self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_targets))
        elif head_name == 'gelu':
            hidden_dim = reg_hidden if reg_hidden is not None else d_model
            self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_targets))
        elif head_name == 'residual':
            hidden_dim = reg_hidden if reg_hidden is not None else d_model
            self.net = ResidualPredictionHead(d_model, num_targets, hidden_dim, dropout)
        else:
            raise ValueError('model.head_type must be one of: mlp, relu, gelu, residual')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
