from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from predictor.common.smiles_encoder import PredictionHead, SmilesBackbone, SmilesModelConfig

@dataclass
class EmbeddingAlignmentConfig:
    projection_dim: int = 128
    projection_space: str = 'shared'
    projection_type: str = 'mlp'
    projection_dropout: float = 0.0
    projection_norm: bool = True
    temperature: float = 0.07
    loss_name: str = 'cosine'
    contrastive_weight: float = 1.0
    contrastive_ignore_same_molecule_negatives: bool = True
    mse_weight: float | None = None
    cosine_weight: float | None = None
    l1_weight: float = 1.0
    smooth_l1_weight: float = 1.0
    embedding_distill_weight: float = 1.0
    cosine_distill_weight: float = 1.0

def regression_loss(preds: torch.Tensor, targets: torch.Tensor, loss_name: str) -> torch.Tensor:
    name = loss_name.lower()
    if name == 'l1':
        return F.l1_loss(preds, targets)
    if name == 'smooth_l1':
        return F.smooth_l1_loss(preds, targets)
    return F.mse_loss(preds, targets)

def normalize_smiles_student_type(student_type: str | None) -> str:
    value = str(student_type or 'gpt_regression').lower()
    if value in {'smiles', 'smiles_backbone', 'default'}:
        return 'smiles'
    if value in {'gpt', 'gpt_regression', 'rl_gpt', 'rl_regression'}:
        return 'gpt_regression'
    raise ValueError('student_type must be one of: smiles, gpt_regression')

class GPTStyleSmilesBackbone(SmilesBackbone):
    pass

def build_student_backbone(student_type: str | None, cfg: SmilesModelConfig) -> nn.Module:
    normalized = normalize_smiles_student_type(student_type)
    if normalized == 'gpt_regression':
        return GPTStyleSmilesBackbone(cfg)
    return SmilesBackbone(cfg)

class ResidualAlignmentProjectionHead(nn.Module):

    def __init__(self, input_dim: int, projection_dim: int, *, dropout: float=0.0, use_norm: bool=True) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, projection_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(projection_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        delta = self.residual_proj(self.dropout(self.act(h)))
        return self.norm(h + delta)

def normalize_alignment_projection_space(projection_space: str | None) -> str:
    value = str(projection_space or 'shared').lower()
    if value in {'shared', 'common', 'joint', 'alignment'}:
        return 'shared'
    if value in {'teacher', 'teacher_space', 'teacher_embedding'}:
        return 'teacher'
    raise ValueError('alignment.projection_space must be one of: shared, teacher')

def build_alignment_projection_head(input_dim: int, align_cfg: EmbeddingAlignmentConfig, *, output_dim: int | None=None) -> nn.Module:
    projection_type = str(align_cfg.projection_type or 'mlp').lower()
    projection_dim = int(output_dim if output_dim is not None else align_cfg.projection_dim)
    if projection_type in {'mlp', 'default'}:
        return nn.Sequential(nn.Linear(input_dim, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim))
    if projection_type in {'residual', 'residual_mlp', 'res'}:
        return ResidualAlignmentProjectionHead(input_dim, projection_dim, dropout=float(align_cfg.projection_dropout), use_norm=bool(align_cfg.projection_norm))
    raise ValueError('alignment.projection_type must be one of: mlp, residual')

def symmetric_contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float, *, molecule_ids: list[str] | None=None, ignore_same_molecule_negatives: bool=True) -> torch.Tensor:
    if left.size(0) <= 1 or right.size(0) <= 1:
        return left.new_zeros(())
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.transpose(0, 1) / max(float(temperature), 1e-06)
    if ignore_same_molecule_negatives and molecule_ids is not None:
        molecule_to_index: dict[str, int] = {}
        encoded_molecule_ids = []
        for molecule_id in molecule_ids:
            if molecule_id not in molecule_to_index:
                molecule_to_index[molecule_id] = len(molecule_to_index)
            encoded_molecule_ids.append(molecule_to_index[molecule_id])
        molecule_id_tensor = torch.tensor(encoded_molecule_ids, dtype=torch.long, device=logits.device)
        same_molecule = molecule_id_tensor[:, None].eq(molecule_id_tensor[None, :])
        eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_molecule & ~eye, torch.finfo(logits.dtype).min)
    labels = torch.arange(left.size(0), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

def _mse_weight(cfg: EmbeddingAlignmentConfig) -> float:
    return cfg.embedding_distill_weight if cfg.mse_weight is None else cfg.mse_weight

def _cosine_weight(cfg: EmbeddingAlignmentConfig) -> float:
    return cfg.cosine_distill_weight if cfg.cosine_weight is None else cfg.cosine_weight

def _alignment_components(loss_name: str) -> tuple[bool, bool, bool, bool, bool]:
    name = loss_name.lower().strip().replace('-', '_')
    aliases = {'clip': 'contrastive', 'info_nce': 'infonce', 'embedding_mse': 'mse', 'embedding_cosine': 'cosine', 'mae': 'l1', 'embedding_l1': 'l1', 'huber': 'smooth_l1', 'embedding_smooth_l1': 'smooth_l1', 'contrastive_mse': 'infonce_mse', 'contrastive_cosine': 'infonce_cosine', 'contrastive_l1': 'infonce_l1'}
    name = aliases.get(name, name)
    if name == 'hybrid':
        return (True, True, True, False, False)
    elif name == 'all':
        return (True, True, True, True, True)
    single_names = {'infonce', 'contrastive', 'mse', 'cosine', 'l1', 'smooth_l1'}
    if name in single_names:
        return (name in {'infonce', 'contrastive'}, name == 'mse', name == 'cosine', name == 'l1', name == 'smooth_l1')
    parts = {part for part in name.split('_') if part}
    use_contrastive = bool(parts & {'infonce', 'contrastive', 'clip'})
    use_mse = 'mse' in parts
    use_cosine = 'cosine' in parts
    use_smooth_l1 = 'smooth' in parts and 'l1' in parts
    use_l1 = 'l1' in parts and (not use_smooth_l1)
    if not (use_contrastive or use_mse or use_cosine or use_l1 or use_smooth_l1):
        raise ValueError('alignment.loss_name must be one of: cosine, mse, l1, smooth_l1, contrastive/infonce/clip, infonce_mse, infonce_cosine, mse_cosine, infonce_l1, infonce_smooth_l1, hybrid, all')
    return (use_contrastive, use_mse, use_cosine, use_l1, use_smooth_l1)

class MulTransformerEmbeddingStudent(nn.Module):

    def __init__(self, cfg: SmilesModelConfig, target_columns: list[str], target_stats: dict, teacher_embedding_dim: int, align_cfg: EmbeddingAlignmentConfig, student_type: str='gpt_regression') -> None:
        super().__init__()
        self.cfg = cfg
        self.target_columns = target_columns
        self.target_stats = target_stats
        self.align_cfg = align_cfg
        self.student_type = normalize_smiles_student_type(student_type)
        self.backbone = build_student_backbone(self.student_type, cfg)
        self.head = PredictionHead(d_model=cfg.d_model, num_targets=cfg.num_targets, dropout=cfg.dropout)
        projection_space = normalize_alignment_projection_space(align_cfg.projection_space)
        if projection_space == 'teacher':
            self.student_align_proj = build_alignment_projection_head(cfg.d_model, align_cfg, output_dim=teacher_embedding_dim)
            self.teacher_align_proj = nn.Identity()
        else:
            self.student_align_proj = build_alignment_projection_head(cfg.d_model, align_cfg)
            self.teacher_align_proj = build_alignment_projection_head(teacher_embedding_dim, align_cfg)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        (_, pooled) = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        preds = self.head(pooled)
        return (preds, pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        (preds, _) = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return preds

    def compute_alignment_loss(self, batch: dict, teacher_embeddings: torch.Tensor, molecule_ids: list[str] | None=None, collect_metrics: bool=True) -> tuple[torch.Tensor, dict[str, float]]:
        (_, pooled) = self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        student_proj = self.student_align_proj(pooled)
        teacher_proj = self.teacher_align_proj(teacher_embeddings.detach())
        (use_contrastive, use_mse, use_cosine, use_l1, use_smooth_l1) = _alignment_components(self.align_cfg.loss_name)
        loss = student_proj.new_zeros(())
        metrics: dict[str, float] = {}
        if use_contrastive:
            contrastive = symmetric_contrastive_loss(student_proj, teacher_proj, self.align_cfg.temperature, molecule_ids=molecule_ids, ignore_same_molecule_negatives=self.align_cfg.contrastive_ignore_same_molecule_negatives)
            loss = loss + self.align_cfg.contrastive_weight * contrastive
            if collect_metrics:
                metrics['contrastive_loss'] = float(contrastive.detach().item())
        if use_mse:
            mse = F.mse_loss(student_proj, teacher_proj)
            loss = loss + _mse_weight(self.align_cfg) * mse
            if collect_metrics:
                metrics['embedding_mse_loss'] = float(mse.detach().item())
        if use_cosine:
            cosine = 1.0 - F.cosine_similarity(student_proj, teacher_proj, dim=-1).mean()
            loss = loss + _cosine_weight(self.align_cfg) * cosine
            if collect_metrics:
                metrics['embedding_cosine_loss'] = float(cosine.detach().item())
        if use_l1:
            l1 = F.l1_loss(student_proj, teacher_proj)
            loss = loss + self.align_cfg.l1_weight * l1
            if collect_metrics:
                metrics['embedding_l1_loss'] = float(l1.detach().item())
        if use_smooth_l1:
            smooth_l1 = F.smooth_l1_loss(student_proj, teacher_proj)
            loss = loss + self.align_cfg.smooth_l1_weight * smooth_l1
            if collect_metrics:
                metrics['embedding_smooth_l1_loss'] = float(smooth_l1.detach().item())
        if collect_metrics:
            metrics['alignment_loss'] = float(loss.detach().item())
        return (loss, metrics)

    def compute_regression_loss(self, batch: dict, loss_name: str) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        preds = self(batch['input_ids'], batch['attention_mask'])
        loss = regression_loss(preds, batch['labels'], loss_name)
        return (loss, {'regression_loss': float(loss.detach().item())}, preds)
