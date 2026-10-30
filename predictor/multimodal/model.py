from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from predictor.common.graph_encoder import GraphModelConfig, QModelGraphEncoder, load_qmodel_encoder_weights
from predictor.common.paths import normalize_path_string
from predictor.common.smiles_encoder import SmilesBackbone, SmilesModelConfig

@dataclass
class MultimodalTransformerConfig:
    graph_encoder_type: str = '3d'
    smiles_backbone_type: str = 'gpt_regression'
    model_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_hidden: int = 512
    reg_hidden: int | None = None
    head_type: str = 'gelu'
    pooling: str = 'cls'
    dropout: float = 0.1
    lr: float = 0.0001
    weight_decay: float = 0.0001
    loss_name: str = 'smooth_l1'
    graph_mask_prob: float = 0.15
    smiles_mask_prob: float = 0.15
    bond_mask_prob: float = 0.15

def make_graph_model_config(raw_cfg: dict) -> GraphModelConfig:
    cfg = dict(raw_cfg)
    mapped = {'hidden_dim': cfg.get('hidden_dim', 128), 'n_layers': cfg.get('n_layers', cfg.get('num_layers', 7)), 'cutoff': cfg.get('cutoff', 1.6), 'head_dropout': cfg.get('head_dropout', cfg.get('dropout', 0.1)), 'max_atom_type': cfg.get('max_atom_type', cfg.get('max_z', 100)), 'charge_power': cfg.get('charge_power', 2), 'attention': cfg.get('attention', True), 'no_edge_types': cfg.get('no_edge_types', False), 'layernorm': cfg.get('layernorm', False), 'node_feature_dim': cfg.get('node_feature_dim', 0), 'graph_feature_dim': cfg.get('graph_feature_dim', 0), 'lr': cfg.get('lr', 0.0005), 'weight_decay': cfg.get('weight_decay', 1e-06), 'warmup_steps': cfg.get('warmup_steps', 0), 'freeze_encoder': cfg.get('freeze_encoder', False)}
    return GraphModelConfig(**mapped)

def _load_smiles_backbone_weights(backbone: nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    state = torch.load(normalize_path_string(checkpoint_path), map_location='cpu', weights_only=False)
    raw_state = state.get('state_dict', state.get('model', state)) if isinstance(state, dict) else state
    model_state = backbone.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for (key, value) in raw_state.items():
        candidates = [key]
        for prefix in ('backbone.', 'smiles_backbone.', 'model.', 'student.'):
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                cleaned[candidate] = value
                break
    load_result = backbone.load_state_dict(cleaned, strict=False)
    return (list(load_result.missing_keys), list(load_result.unexpected_keys))

def _safe_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int=-100) -> torch.Tensor:
    valid_mask = targets.ne(ignore_index)
    if not torch.any(valid_mask):
        return logits.new_zeros(())
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)

def normalize_smiles_backbone_type(backbone_type: str | None) -> str:
    value = str(backbone_type or 'gpt_regression').lower()
    if value in {'smiles', 'smiles_backbone', 'default'}:
        return 'smiles'
    if value in {'gpt', 'gpt_regression', 'rl_gpt', 'rl_regression'}:
        return 'gpt_regression'
    raise ValueError('smiles_backbone_type must be one of: smiles, gpt_regression')

def normalize_multimodal_pooling(pooling: str | None) -> str:
    value = str(pooling or 'cls').lower()
    if value in {'cls', 'cls_token', 'cls_readout'}:
        return 'cls'
    if value in {'masked_mean', 'mean', 'masked_pooling', 'masked_mean_pooling'}:
        return 'masked_mean'
    raise ValueError('model.pooling must be one of: cls, masked_mean')

class GPTRegressionSmilesBackbone(SmilesBackbone):
    pass

def build_smiles_backbone(backbone_type: str | None, cfg: SmilesModelConfig) -> nn.Module:
    if normalize_smiles_backbone_type(backbone_type) == 'gpt_regression':
        return GPTRegressionSmilesBackbone(cfg)
    return SmilesBackbone(cfg)

class ResidualRegressionHead(nn.Module):

    def __init__(self, input_dim: int, num_targets: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.out = nn.Linear(input_dim, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return self.out(x + h)


def build_regression_head(input_dim: int, num_targets: int, *, head_type: str, reg_hidden: int | None, default_hidden: int, dropout: float) -> nn.Module:
    head_name = str(head_type or 'gelu').lower()
    if head_name in {'mlp', 'relu'}:
        hidden_dim = reg_hidden if reg_hidden is not None else default_hidden
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_targets))
    if head_name == 'gelu':
        hidden_dim = reg_hidden if reg_hidden is not None else default_hidden
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_targets))
    if head_name == 'residual':
        hidden_dim = reg_hidden if reg_hidden is not None else input_dim
        return ResidualRegressionHead(input_dim, num_targets, hidden_dim, dropout)
    raise ValueError('model.head_type must be one of: mlp, relu, gelu, residual')

class MultimodalTransformerBackbone(nn.Module):

    def __init__(self, graph_cfg: GraphModelConfig, smiles_cfg: SmilesModelConfig, mm_cfg: MultimodalTransformerConfig, *, pretrained_graph_path: str | None=None, pretrained_smiles_checkpoint: str | None=None) -> None:
        super().__init__()
        self.mm_cfg = mm_cfg
        self.graph_encoder_type = mm_cfg.graph_encoder_type.lower()
        self.smiles_backbone_type = normalize_smiles_backbone_type(mm_cfg.smiles_backbone_type)
        self.pooling = normalize_multimodal_pooling(mm_cfg.pooling)
        self.vocab_size = smiles_cfg.vocab_size
        self.graph_num_classes = graph_cfg.max_atom_type + 1
        if self.graph_encoder_type != '3d':
            raise ValueError('graph_encoder_type must be 3d')
        self.graph_encoder = QModelGraphEncoder(graph_cfg)
        if pretrained_graph_path:
            (self.graph_missing_keys, self.graph_unexpected_keys) = load_qmodel_encoder_weights(self.graph_encoder.encoder, pretrained_graph_path)
        else:
            (self.graph_missing_keys, self.graph_unexpected_keys) = ([], [])
        graph_hidden_dim = graph_cfg.hidden_dim
        self.smiles_backbone = build_smiles_backbone(self.smiles_backbone_type, smiles_cfg)
        if pretrained_smiles_checkpoint:
            (self.smiles_missing_keys, self.smiles_unexpected_keys) = _load_smiles_backbone_weights(self.smiles_backbone, pretrained_smiles_checkpoint)
        else:
            (self.smiles_missing_keys, self.smiles_unexpected_keys) = ([], [])
        self.graph_proj = nn.Linear(graph_hidden_dim, mm_cfg.model_dim)
        self.smiles_proj = nn.Linear(smiles_cfg.d_model, mm_cfg.model_dim)
        self.graph_type_emb = nn.Parameter(torch.randn(1, 1, mm_cfg.model_dim) * 0.02)
        self.smiles_type_emb = nn.Parameter(torch.randn(1, 1, mm_cfg.model_dim) * 0.02)
        if self.pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mm_cfg.model_dim) * 0.02)
        else:
            self.cls_token = None
        encoder_layer = nn.TransformerEncoderLayer(d_model=mm_cfg.model_dim, nhead=mm_cfg.n_heads, dim_feedforward=mm_cfg.ff_hidden, dropout=mm_cfg.dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=mm_cfg.n_layers)
        self.final_norm = nn.LayerNorm(mm_cfg.model_dim)

    def _strip_smiles_cls(self, smiles_tokens: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (smiles_tokens, attention_mask.bool())

    def encode_graph(self, positions, charges, atom_mask):
        out = self.graph_encoder(positions, charges, atom_mask)
        return (out['node_embeddings'], out['node_mask'])

    def encode_joint(self, positions, charges, atom_mask, input_ids, attention_mask, *, causal_smiles: bool=False) -> dict[str, torch.Tensor]:
        (graph_nodes, graph_mask) = self.encode_graph(positions, charges, atom_mask)
        (smiles_tokens, _) = self.smiles_backbone(input_ids=input_ids, attention_mask=attention_mask)
        (smiles_tokens, smiles_mask) = self._strip_smiles_cls(smiles_tokens, attention_mask)
        graph_tokens = self.graph_proj(graph_nodes) + self.graph_type_emb
        smiles_tokens = self.smiles_proj(smiles_tokens) + self.smiles_type_emb
        if self.pooling == 'cls':
            if self.cls_token is None:
                raise RuntimeError("cls_token is not initialized while model.pooling='cls'")
            cls = self.cls_token.expand(graph_tokens.size(0), 1, -1)
            all_tokens = torch.cat([cls, graph_tokens, smiles_tokens], dim=1)
            cls_mask = torch.ones(graph_tokens.size(0), 1, dtype=torch.bool, device=graph_tokens.device)
            joint_mask = torch.cat([cls_mask, graph_mask, smiles_mask], dim=1)
            graph_offset = 1
        else:
            all_tokens = torch.cat([graph_tokens, smiles_tokens], dim=1)
            joint_mask = torch.cat([graph_mask, smiles_mask], dim=1)
            graph_offset = 0
        attn_mask = None
        if causal_smiles:
            total_len = all_tokens.size(1)
            graph_prefix = graph_offset + graph_tokens.size(1)
            attn_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=all_tokens.device)
            smiles_len = smiles_tokens.size(1)
            for i in range(smiles_len):
                row = graph_prefix + i
                attn_mask[row, graph_prefix + i + 1:] = True
        encoded = self.transformer(all_tokens, mask=attn_mask, src_key_padding_mask=~joint_mask)
        encoded = self.final_norm(encoded)
        encoded_graph_tokens = encoded[:, graph_offset:graph_offset + graph_tokens.size(1)]
        encoded_smiles_tokens = encoded[:, graph_offset + graph_tokens.size(1):]
        pooled_tokens = torch.cat([encoded_graph_tokens, encoded_smiles_tokens], dim=1)
        pooled_mask = torch.cat([graph_mask, smiles_mask], dim=1).unsqueeze(-1).to(pooled_tokens.dtype)
        pooled = (pooled_tokens * pooled_mask).sum(dim=1) / pooled_mask.sum(dim=1).clamp_min(1.0)
        readout = encoded[:, 0] if self.pooling == 'cls' else pooled
        return {'all_tokens': encoded, 'cls': readout, 'pooled': pooled, 'readout': readout, 'graph_tokens': encoded_graph_tokens, 'smiles_tokens': encoded_smiles_tokens, 'graph_mask': graph_mask, 'smiles_mask': smiles_mask}

class MultimodalTransformerRegressor(nn.Module):

    def __init__(self, graph_cfg: GraphModelConfig, smiles_cfg: SmilesModelConfig, mm_cfg: MultimodalTransformerConfig, num_targets: int, *, pretrained_graph_path: str | None=None, pretrained_smiles_checkpoint: str | None=None) -> None:
        super().__init__()
        self.mm_cfg = mm_cfg
        self.backbone = MultimodalTransformerBackbone(graph_cfg=graph_cfg, smiles_cfg=smiles_cfg, mm_cfg=mm_cfg, pretrained_graph_path=pretrained_graph_path, pretrained_smiles_checkpoint=pretrained_smiles_checkpoint)
        self.head = build_regression_head(mm_cfg.model_dim, num_targets, head_type=mm_cfg.head_type, reg_hidden=mm_cfg.reg_hidden, default_hidden=mm_cfg.ff_hidden, dropout=mm_cfg.dropout)

    def forward(self, positions, charges, atom_mask, input_ids, attention_mask):
        encoded = self.backbone.encode_joint(positions, charges, atom_mask, input_ids, attention_mask)
        return self.head(encoded['readout'])

    def compute_loss(self, preds_norm: torch.Tensor, labels_norm: torch.Tensor) -> torch.Tensor:
        loss_name = self.mm_cfg.loss_name.lower()
        if loss_name == 'mse':
            return F.mse_loss(preds_norm, labels_norm)
        if loss_name == 'l1':
            return F.l1_loss(preds_norm, labels_norm)
        return F.smooth_l1_loss(preds_norm, labels_norm)

class MaskedMultimodalTransformerPretrainer(nn.Module):

    def __init__(self, backbone: MultimodalTransformerBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.smiles_head = nn.Linear(backbone.mm_cfg.model_dim, backbone.vocab_size)
        self.graph_node_head = nn.Linear(backbone.mm_cfg.model_dim, backbone.graph_num_classes)
        self.bond_head = nn.Sequential(nn.Linear(backbone.mm_cfg.model_dim * 2, backbone.mm_cfg.ff_hidden), nn.GELU(), nn.Linear(backbone.mm_cfg.ff_hidden, 5))
        self.distance_head = nn.Sequential(nn.Linear(backbone.mm_cfg.model_dim * 2, backbone.mm_cfg.ff_hidden), nn.GELU(), nn.Linear(backbone.mm_cfg.ff_hidden, 32))

    def forward(self, positions, charges, atom_mask, input_ids, attention_mask):
        return self.backbone.encode_joint(positions, charges, atom_mask, input_ids, attention_mask, causal_smiles=False)

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        encoded = self(batch['positions'], batch['masked_charges'], batch['atom_mask'], batch['masked_input_ids'], batch['attention_mask'])
        smiles_logits = self.smiles_head(encoded['smiles_tokens'])
        graph_node_logits = self.graph_node_head(encoded['graph_tokens'])
        smiles_loss = _safe_cross_entropy(smiles_logits.reshape(-1, smiles_logits.size(-1)), batch['smiles_mlm_labels'].reshape(-1), ignore_index=-100)
        node_loss = _safe_cross_entropy(graph_node_logits.reshape(-1, graph_node_logits.size(-1)), batch['graph_node_labels'].reshape(-1), ignore_index=-100)
        bond_loss = smiles_loss.new_zeros(())
        if 'bond_labels' in batch:
            graph_tokens = encoded['graph_tokens']
            left = graph_tokens.unsqueeze(2).expand(-1, -1, graph_tokens.size(1), -1)
            right = graph_tokens.unsqueeze(1).expand(-1, graph_tokens.size(1), -1, -1)
            bond_logits = self.bond_head(torch.cat([left, right], dim=-1))
            bond_loss = _safe_cross_entropy(bond_logits.reshape(-1, bond_logits.size(-1)), batch['bond_labels'].reshape(-1), ignore_index=-100)
        loss = smiles_loss + node_loss + bond_loss
        return (loss, {'smiles_mlm_loss': float(smiles_loss.detach().item()), 'node_mlm_loss': float(node_loss.detach().item()), 'bond_mlm_loss': float(bond_loss.detach().item())})

class GenerativeMultimodalTransformerPretrainer(nn.Module):

    def __init__(self, backbone: MultimodalTransformerBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(backbone.mm_cfg.model_dim, backbone.vocab_size)

    def forward(self, positions, charges, atom_mask, input_ids, attention_mask):
        return self.backbone.encode_joint(positions, charges, atom_mask, input_ids, attention_mask, causal_smiles=True)

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        encoded = self(batch['positions'], batch['charges'], batch['atom_mask'], batch['decoder_input_ids'], batch['decoder_attention_mask'])
        logits = self.lm_head(encoded['smiles_tokens'])
        loss = _safe_cross_entropy(logits.reshape(-1, logits.size(-1)), batch['generation_labels'].reshape(-1), ignore_index=-100)
        return (loss, {'generation_loss': float(loss.detach().item())})

def load_pretrained_multimodal_transformer_backbone(model: nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    state = torch.load(normalize_path_string(checkpoint_path), map_location='cpu', weights_only=False)
    raw_state = state.get('model', state.get('state_dict', state)) if isinstance(state, dict) else state
    model_state = model.state_dict()
    filtered = {}
    for (key, value) in raw_state.items():
        new_key = key
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
        if new_key in model_state and model_state[new_key].shape == value.shape:
            filtered[new_key] = value
    load_result = model.load_state_dict(filtered, strict=False)
    return (list(load_result.missing_keys), list(load_result.unexpected_keys))
