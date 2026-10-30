from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, TransformerEncoder, TransformerEncoderLayer
import lightning.pytorch as pl
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from predictor.gpt.data import inverse_transform_tensor

@dataclass
class RLGPTRegressorConfig:
    vocab_size: int
    max_seq_len: int = 256
    pad_token_id: int = 0
    num_targets: int = 1
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    reg_hidden: int | None = None
    head_type: str = 'mlp'
    lr: float = 0.0002
    weight_decay: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    loss_name: str = 'mse'

class ResidualRegressionHead(nn.Module):

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


def build_regression_head(cfg: RLGPTRegressorConfig) -> nn.Module:
    head_type = str(cfg.head_type or 'mlp').lower()
    if head_type in {'mlp', 'relu'}:
        hidden_dim = cfg.reg_hidden if cfg.reg_hidden is not None else cfg.d_model // 2
        return nn.Sequential(nn.Linear(cfg.d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, cfg.num_targets))
    if head_type == 'gelu':
        hidden_dim = cfg.reg_hidden if cfg.reg_hidden is not None else cfg.d_model // 2
        return nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden_dim, cfg.num_targets))
    if head_type == 'residual':
        hidden_dim = cfg.reg_hidden if cfg.reg_hidden is not None else cfg.d_model
        return ResidualRegressionHead(cfg.d_model, cfg.num_targets, hidden_dim, cfg.dropout)
    raise ValueError('model.head_type must be one of: mlp, relu, gelu, residual')

class RLGPTRegressor(pl.LightningModule):

    def __init__(self, cfg: RLGPTRegressorConfig, target_columns: list[str], target_stats: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_columns = target_columns
        self.target_stats = target_stats
        self.save_hyperparameters({'cfg': asdict(cfg), 'target_columns': target_columns, 'target_stats': target_stats})
        self.vocab_size = cfg.vocab_size
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.d_model
        self.padding_idx = cfg.pad_token_id
        self.embedding = Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_encoder = PositionalEncoding1D(cfg.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, batch_first=True)
        self.decoder = TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.reg_head = build_regression_head(cfg)
        self.val_mae_metric = MeanAbsoluteError()
        self.val_rmse_metric = MeanSquaredError(squared=False, num_outputs=cfg.num_targets)
        self.val_r2_metric = R2Score(multioutput='uniform_average')
        self.test_mae_metric = MeanAbsoluteError()
        self.test_rmse_metric = MeanSquaredError(squared=False, num_outputs=cfg.num_targets)
        self.test_r2_metric = R2Score(multioutput='uniform_average')
        self.validation_metric_prefix = 'val'

    def generate_square_subsequent_mask(self, size: int, device: torch.device | None=None) -> torch.Tensor:
        if device is None:
            device = self.device
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        emb = emb + self.pos_encoder(emb)
        seq_len = input_ids.size(1)
        causal_mask = self.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        hidden = self.decoder(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return pooled

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        pooled = self.encode(input_ids=input_ids, pad_mask=pad_mask)
        return self.reg_head(pooled)

    def _loss(self, preds_norm: torch.Tensor, labels_norm: torch.Tensor) -> torch.Tensor:
        loss_name = self.cfg.loss_name.lower()
        if loss_name == 'mse':
            return F.mse_loss(preds_norm, labels_norm)
        if loss_name == 'l1':
            return F.l1_loss(preds_norm, labels_norm)
        if loss_name in {'smooth_l1', 'smoothl1', 'huber'}:
            return F.smooth_l1_loss(preds_norm, labels_norm)
        raise ValueError(f'Unsupported loss_name={self.cfg.loss_name!r}; expected mse, l1, or smooth_l1')

    def _shared_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        preds_norm = self(input_ids=batch['input_ids'], pad_mask=batch['pad_mask'])
        labels_norm = batch['labels']
        return (self._loss(preds_norm, labels_norm), preds_norm)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        (loss, _) = self._shared_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=False, on_step=True, on_epoch=False, batch_size=batch['input_ids'].size(0))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        (loss, preds_norm) = self._shared_loss(batch)
        preds_raw = inverse_transform_tensor(preds_norm.detach(), self.target_columns, self.target_stats)
        labels_raw = batch['labels_raw'].detach()
        self.val_mae_metric.update(preds_raw, labels_raw)
        self.val_rmse_metric.update(preds_raw, labels_raw)
        self.val_r2_metric.update(preds_raw, labels_raw)
        self.log('val_loss', loss, prog_bar=False, logger=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['input_ids'].size(0))
        if self.validation_metric_prefix != 'val':
            self.log(f'{self.validation_metric_prefix}_loss', loss, prog_bar=False, logger=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['input_ids'].size(0))

    def test_step(self, batch: dict, batch_idx: int) -> None:
        (loss, preds_norm) = self._shared_loss(batch)
        preds_raw = inverse_transform_tensor(preds_norm.detach(), self.target_columns, self.target_stats)
        labels_raw = batch['labels_raw'].detach()
        self.test_mae_metric.update(preds_raw, labels_raw)
        self.test_rmse_metric.update(preds_raw, labels_raw)
        self.test_r2_metric.update(preds_raw, labels_raw)
        self.log('test_loss', loss, prog_bar=False, logger=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['input_ids'].size(0))

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int=0) -> dict:
        preds_norm = self(input_ids=batch['input_ids'], pad_mask=batch['pad_mask'])
        preds_raw = inverse_transform_tensor(preds_norm.detach(), self.target_columns, self.target_stats)
        return {'smiles': batch['smiles'], 'preds': preds_raw.detach().cpu(), 'labels_raw': batch.get('labels_raw', torch.empty(0)).detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        val_mae = self.val_mae_metric.compute()
        val_rmse = self.val_rmse_metric.compute()
        val_r2 = self.val_r2_metric.compute()
        if val_mae.ndim > 0:
            val_mae = val_mae.mean()
        if val_rmse.ndim > 0:
            val_rmse = val_rmse.mean()
        self.log('val_mae', val_mae, prog_bar=True, logger=False, on_epoch=True)
        self.log('val_rmse', val_rmse, prog_bar=True, logger=False, on_epoch=True)
        self.log('val_r2', val_r2, prog_bar=True, logger=False, on_epoch=True)
        if self.validation_metric_prefix != 'val':
            self.log(f'{self.validation_metric_prefix}_mae', val_mae, prog_bar=True, logger=False, on_epoch=True)
            self.log(f'{self.validation_metric_prefix}_rmse', val_rmse, prog_bar=True, logger=False, on_epoch=True)
            self.log(f'{self.validation_metric_prefix}_r2', val_r2, prog_bar=True, logger=False, on_epoch=True)
        self.val_mae_metric.reset()
        self.val_rmse_metric.reset()
        self.val_r2_metric.reset()

    def on_test_epoch_end(self) -> None:
        test_mae = self.test_mae_metric.compute()
        test_rmse = self.test_rmse_metric.compute()
        test_r2 = self.test_r2_metric.compute()
        if test_mae.ndim > 0:
            test_mae = test_mae.mean()
        if test_rmse.ndim > 0:
            test_rmse = test_rmse.mean()
        self.log('test_mae', test_mae, prog_bar=True, logger=False, on_epoch=True)
        self.log('test_rmse', test_rmse, prog_bar=True, logger=False, on_epoch=True)
        self.log('test_r2', test_r2, prog_bar=True, logger=False, on_epoch=True)
        self.test_mae_metric.reset()
        self.test_rmse_metric.reset()
        self.test_r2_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.cfg.scheduler_factor, patience=self.cfg.scheduler_patience)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
