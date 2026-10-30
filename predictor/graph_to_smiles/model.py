from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, TransformerEncoder, TransformerEncoderLayer
from predictor.graph.model import GraphV2ModelConfig, LitGraphV2Regressor
from predictor.gpt.model import PositionalEncoding1D
from predictor.common.smiles_encoder import PredictionHead, SmilesBackbone, SmilesModelConfig

@dataclass
class DistillLossConfig:
    stage1_alignment: str = 'infonce'
    contrastive_weight: float = 1.0
    contrastive_ignore_same_molecule_negatives: bool = True
    supervised_weight: float = 1.0
    prediction_distill_weight: float = 1.0
    embedding_distill_weight: float = 0.5
    cosine_distill_weight: float = 0.5
    temperature: float = 0.1
    stage2_prediction_distill_weight: float | None = None
    stage2_embedding_distill_weight: float | None = None
    supervised_loss_name: str = 'mse'
    prediction_distill_loss_name: str = 'mse'
    embedding_distill_loss_name: str = 'mse'

@dataclass
class RLGPTStudentConfig:
    vocab_size: int
    pad_token_id: int = 0
    max_seq_len: int = 256
    num_targets: int = 1
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    reg_hidden: int | None = None
    head_type: str = 'gelu'
    lr: float = 0.0002
    weight_decay: float = 0.0

def regression_loss(preds: torch.Tensor, targets: torch.Tensor, loss_name: str) -> torch.Tensor:
    name = loss_name.lower()
    if name == 'mse':
        return F.mse_loss(preds, targets)
    if name == 'l1':
        return F.l1_loss(preds, targets)
    if name in {'smooth_l1', 'smoothl1', 'huber'}:
        return F.smooth_l1_loss(preds, targets)
    raise ValueError(f'Unsupported regression loss_name={loss_name!r}; expected mse, l1, or smooth_l1')

class GraphV2Teacher(nn.Module):

    def __init__(self, cfg: GraphV2ModelConfig, target_columns: list[str], target_stats: dict, checkpoint_path: str) -> None:
        super().__init__()
        self.model = LitGraphV2Regressor(cfg=cfg, target_columns=target_columns, target_stats=target_stats, pretrained_graph_path=None)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, positions: torch.Tensor, charges: torch.Tensor, atom_mask: torch.Tensor, extra_node_features: torch.Tensor | None=None, graph_features: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        graph_outputs = self.model.encoder(positions, charges, atom_mask, extra_node_features=extra_node_features)
        (embedding, preds) = self.encode_head(graph_outputs['node_embeddings'], graph_outputs['node_mask'], graph_features=graph_features)
        return {'embedding': embedding, 'preds': preds}

    def encode_head(self, node_embeddings: torch.Tensor, node_mask: torch.Tensor | None=None, graph_features: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.model.head
        h = head.node_dec(node_embeddings)
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        pooled = h.sum(dim=1)
        if head.graph_feature_dim > 0 and graph_features is not None:
            pooled = torch.cat([pooled, graph_features.to(pooled.dtype)], dim=-1)
        embedding = head.graph_dec[1](head.graph_dec[0](pooled))
        preds = head.graph_dec[2](embedding)
        return (embedding, preds)

class SmilesStudent(nn.Module):

    def __init__(self, cfg: SmilesModelConfig, teacher_dim: int, num_targets: int, loss_cfg: DistillLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = loss_cfg
        self.backbone = SmilesBackbone(cfg)
        self.proj = nn.Sequential(nn.Linear(cfg.d_model, teacher_dim), nn.GELU(), nn.LayerNorm(teacher_dim))
        self.head = PredictionHead(d_model=teacher_dim, num_targets=num_targets, dropout=cfg.dropout, head_type=cfg.head_type, reg_hidden=cfg.reg_hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        (_, pooled) = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embedding = self.proj(pooled)
        preds = self.head(embedding)
        return {'embedding': embedding, 'preds': preds}

    def contrastive_loss(self, student_embedding: torch.Tensor, teacher_embedding: torch.Tensor, molecule_ids: list[str] | None=None) -> torch.Tensor:
        student_z = F.normalize(student_embedding, dim=-1)
        teacher_z = F.normalize(teacher_embedding.detach(), dim=-1)
        logits = student_z @ teacher_z.t() / self.loss_cfg.temperature
        if self.loss_cfg.contrastive_ignore_same_molecule_negatives and molecule_ids is not None:
            same_molecule = torch.tensor([[left == right for right in molecule_ids] for left in molecule_ids], dtype=torch.bool, device=logits.device)
            eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(same_molecule & ~eye, torch.finfo(logits.dtype).min)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def embedding_mse_loss(self, student_embedding: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_embedding, teacher_embedding.detach())

    def embedding_cosine_loss(self, student_embedding: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(student_embedding, teacher_embedding.detach(), dim=-1).mean()

    def stage1_loss(self, student_out: dict[str, torch.Tensor], teacher_out: dict[str, torch.Tensor], molecule_ids: list[str] | None=None) -> tuple[torch.Tensor, dict[str, float]]:
        mode = self.loss_cfg.stage1_alignment.lower()
        use_contrastive = mode in {'infonce', 'contrastive', 'clip', 'infonce_mse', 'contrastive_mse', 'infonce_cosine', 'contrastive_cosine', 'hybrid'}
        use_mse = mode in {'mse', 'embedding_mse', 'infonce_mse', 'contrastive_mse', 'mse_cosine', 'hybrid'}
        use_cosine = mode in {'cosine', 'embedding_cosine', 'infonce_cosine', 'contrastive_cosine', 'mse_cosine', 'hybrid'}
        if not (use_contrastive or use_mse or use_cosine):
            raise ValueError('distill.stage1_alignment must be one of: infonce, mse, cosine, infonce_mse, infonce_cosine, mse_cosine, hybrid')
        embedding = student_out['embedding']
        teacher_embedding = teacher_out['embedding']
        loss = embedding.new_zeros(())
        parts: dict[str, float] = {}
        if use_contrastive:
            contrastive = self.contrastive_loss(embedding, teacher_embedding, molecule_ids=molecule_ids)
            loss = loss + self.loss_cfg.contrastive_weight * contrastive
            parts['contrastive_loss'] = float(contrastive.detach().item())
        if use_mse:
            emb_mse = self.embedding_mse_loss(embedding, teacher_embedding)
            loss = loss + self.loss_cfg.embedding_distill_weight * emb_mse
            parts['embedding_mse_loss'] = float(emb_mse.detach().item())
        if use_cosine:
            emb_cosine = self.embedding_cosine_loss(embedding, teacher_embedding)
            loss = loss + self.loss_cfg.cosine_distill_weight * emb_cosine
            parts['embedding_cosine_loss'] = float(emb_cosine.detach().item())
        return (loss, parts)

    def stage2_loss(self, student_out: dict[str, torch.Tensor], teacher_out: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        supervised = regression_loss(student_out['preds'], labels, self.loss_cfg.supervised_loss_name)
        pred_distill = regression_loss(student_out['preds'], teacher_out['preds'].detach(), self.loss_cfg.prediction_distill_loss_name)
        emb_distill = regression_loss(student_out['embedding'], teacher_out['embedding'].detach(), self.loss_cfg.embedding_distill_loss_name)
        pred_weight = self.loss_cfg.prediction_distill_weight
        if self.loss_cfg.stage2_prediction_distill_weight is not None:
            pred_weight = self.loss_cfg.stage2_prediction_distill_weight
        emb_weight = self.loss_cfg.embedding_distill_weight
        if self.loss_cfg.stage2_embedding_distill_weight is not None:
            emb_weight = self.loss_cfg.stage2_embedding_distill_weight
        loss = self.loss_cfg.supervised_weight * supervised + pred_weight * pred_distill + emb_weight * emb_distill
        return (loss, {'supervised_loss': float(supervised.detach().item()), 'prediction_distill_loss': float(pred_distill.detach().item()), 'embedding_distill_loss': float(emb_distill.detach().item()), 'prediction_distill_weight': float(pred_weight), 'embedding_distill_weight': float(emb_weight)})

class RLGPTSmilesStudent(SmilesStudent):
    """Graph-v2 distillation student using the GPT-style gpt_regression backbone."""

    def __init__(self, cfg: RLGPTStudentConfig, teacher_dim: int, num_targets: int, loss_cfg: DistillLossConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.loss_cfg = loss_cfg
        self.embedding = Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_encoder = PositionalEncoding1D(cfg.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, batch_first=True)
        self.decoder = TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.proj = nn.Sequential(nn.Linear(cfg.d_model, teacher_dim), nn.GELU(), nn.LayerNorm(teacher_dim))
        self.head = PredictionHead(d_model=teacher_dim, num_targets=num_targets, dropout=cfg.dropout, head_type=cfg.head_type, reg_hidden=cfg.reg_hidden)

    def generate_square_subsequent_mask(self, size: int, device: torch.device | None=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pad_mask = ~attention_mask.bool()
        emb = self.embedding(input_ids)
        emb = emb + self.pos_encoder(emb)
        causal_mask = self.generate_square_subsequent_mask(input_ids.size(1), device=input_ids.device)
        hidden = self.decoder(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        valid = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        embedding = self.proj(pooled)
        preds = self.head(embedding)
        return {'embedding': embedding, 'preds': preds}

def normalize_student_type(student_type: str | None) -> str:
    value = str(student_type or 'rl_gpt').lower()
    if value in {'smiles', 'smiles_backbone', 'default'}:
        return 'smiles'
    if value in {'rl_gpt', 'rl_regression', 'gpt', 'gpt_regression'}:
        return 'rl_gpt'
    raise ValueError('student_type must be one of: smiles, rl_gpt')

def _filter_dataclass_kwargs(cls, values: dict) -> dict:
    allowed = set(cls.__dataclass_fields__)
    return {key: value for (key, value) in values.items() if key in allowed}

def build_smiles_student(student_type: str | None, student_cfg_dict: dict, *, vocab_size: int, pad_token_id: int, max_seq_len: int, num_targets: int, teacher_dim: int, loss_cfg: DistillLossConfig) -> nn.Module:
    student_type = normalize_student_type(student_type)
    raw_cfg = dict(student_cfg_dict)
    if student_type == 'smiles':
        cfg = SmilesModelConfig(vocab_size=vocab_size, pad_token_id=pad_token_id, max_seq_len=max_seq_len, num_targets=num_targets, **raw_cfg)
        return SmilesStudent(cfg=cfg, teacher_dim=teacher_dim, num_targets=num_targets, loss_cfg=loss_cfg)
    if 'n_heads' in raw_cfg and 'nhead' not in raw_cfg:
        raw_cfg['nhead'] = raw_cfg.pop('n_heads')
    if 'n_layers' in raw_cfg and 'num_layers' not in raw_cfg:
        raw_cfg['num_layers'] = raw_cfg.pop('n_layers')
    if 'ffn_hidden' in raw_cfg and 'dim_feedforward' not in raw_cfg:
        raw_cfg['dim_feedforward'] = raw_cfg.pop('ffn_hidden')
    cfg = RLGPTStudentConfig(vocab_size=vocab_size, pad_token_id=pad_token_id, max_seq_len=max_seq_len, num_targets=num_targets, **_filter_dataclass_kwargs(RLGPTStudentConfig, raw_cfg))
    return RLGPTSmilesStudent(cfg=cfg, teacher_dim=teacher_dim, num_targets=num_targets, loss_cfg=loss_cfg)
