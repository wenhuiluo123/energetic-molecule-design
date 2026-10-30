from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from predictor.common.paths import normalize_path_string

@dataclass
class GraphModelConfig:
    max_atom_type: int = 100
    charge_power: int = 2
    hidden_dim: int = 128
    n_layers: int = 7
    attention: bool = True
    no_edge_types: bool = False
    layernorm: bool = False
    cutoff: float = 1.6
    node_feature_dim: int = 0
    graph_feature_dim: int = 0
    head_dropout: float = 0.1
    lr: float = 0.0005
    weight_decay: float = 1e-06
    warmup_steps: int = 0
    freeze_encoder: bool = False

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_GCL(nn.Module):

    def __init__(self, input_nf: int, output_nf: int, hidden_nf: int, edges_in_d: int=0, act_fn: nn.Module=nn.SiLU(), residual: bool=True, attention: bool=False, normalize: bool=False, coords_agg: str='mean', tanh: bool=False, use_layer_norm: bool=True) -> None:
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-08
        self.use_layer_norm = use_layer_norm
        edge_coords_nf = 1
        if use_layer_norm:
            self.node_ln = LayerNorm(input_nf)
            self.edge_ln = LayerNorm(hidden_nf)
        self.edge_mlp = nn.Sequential(nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf))
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_layers = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source: torch.Tensor, target: torch.Tensor, radial: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        out = torch.cat([source, target, radial], dim=1) if edge_attr is None else torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_attr: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        (row, _) = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg, node_attr], dim=1) if node_attr is not None else torch.cat([x, agg], dim=1)
        return (self.node_mlp(agg), agg)

    def coord_model(self, coord: torch.Tensor, edge_index: torch.Tensor, coord_diff: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        (row, _) = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f'Unsupported coords_agg={self.coords_agg}')
        return coord + agg

    def coord2radial(self, edge_index: torch.Tensor, coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (row, col) = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return (radial, coord_diff)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, coord: torch.Tensor, edge_attr: torch.Tensor | None=None, node_attr: torch.Tensor | None=None, edge_mask: torch.Tensor | None=None, update_coords: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        (row, _) = edge_index
        (radial, coord_diff) = self.coord2radial(edge_index, coord)
        h0 = h
        if self.use_layer_norm:
            h = self.node_ln(h)
        edge_feat = self.edge_model(h[row], h[edge_index[1]], radial, edge_attr)
        if self.use_layer_norm:
            edge_feat = self.edge_ln(edge_feat)
        if edge_mask is not None:
            edge_feat = edge_feat * edge_mask
        if update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        (h, _) = self.node_model(h, edge_index, edge_feat, node_attr)
        if self.residual:
            h = h0 + h
        return (h, coord, edge_attr)

class EGNN_last(nn.Module):

    def __init__(self, in_node_nf: int, hidden_nf: int, in_edge_nf: int=0, act_fn: nn.Module=nn.SiLU(), n_layers: int=4, residual: bool=True, attention: bool=False, normalize: bool=False, tanh: bool=False, use_layer_norm: bool=False) -> None:
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_ln = LayerNorm(self.hidden_nf)
        for i in range(n_layers):
            self.add_module(f'gcl_{i}', E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh, use_layer_norm=use_layer_norm))

    def forward(self, h: torch.Tensor, x: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor | None, edge_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding_in(h)
        for i in range(self.n_layers - 1):
            (h, x, _) = self._modules[f'gcl_{i}'](h, edges, x, edge_attr=edge_attr, edge_mask=edge_mask, update_coords=False)
        (h, x, _) = self._modules[f'gcl_{self.n_layers - 1}'](h, edges, x, edge_attr=edge_attr, edge_mask=edge_mask, update_coords=True)
        if self.use_layer_norm:
            h = self.final_ln(h)
        return (h, x)

def process_input(atom_type: torch.Tensor, max_atom_type: int, charge_power: int) -> torch.Tensor:
    one_hot = nn.functional.one_hot(atom_type, max_atom_type)
    charge_tensor = (atom_type / max_atom_type).pow(torch.arange(charge_power + 1.0, dtype=torch.float32, device=atom_type.device))
    charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
    return (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))

def binarize(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_adj_matrix(adj: torch.Tensor, order: int) -> torch.Tensor:
    adj_mats = [torch.eye(adj.size(0), device=adj.device), binarize(adj + torch.eye(adj.size(0), device=adj.device))]
    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj).float()
    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
    return order_mat.long()

def dense_to_sparse(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = dense.nonzero(as_tuple=False).t().contiguous()
    values = dense[indices[0], indices[1]]
    return (indices, values)

def gen_fully_connected(pos: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (batch, nodes) = mask.shape
    batch_mask = mask[:, :, None] * mask[:, None, :]
    batch_mask = batch_mask.bool() & ~torch.eye(nodes, device=mask.device, dtype=torch.bool)
    batch_mask = torch.block_diag(*batch_mask)
    return dense_to_sparse(batch_mask)

def gen_fully_connected_with_hop(pos: torch.Tensor, mask: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    (batch, nodes) = mask.shape
    batch_adj = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1)
    batch_mask_fc = mask[:, :, None] * mask[:, None, :]
    batch_mask = batch_mask_fc.bool() & (batch_adj <= cutoff) & ~torch.eye(nodes, device=mask.device, dtype=torch.bool)
    batch_mask = torch.block_diag(*batch_mask)
    adj_order = get_higher_order_adj_matrix(batch_mask, 3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    fc_mask = batch_mask_fc.bool() & ~torch.eye(nodes, device=mask.device, dtype=torch.bool)
    fc_mask = torch.block_diag(*fc_mask)
    type_new = batch_mask + type_highorder + fc_mask
    (edge_index, edge_type) = dense_to_sparse(type_new)
    return (edge_index, edge_type - 1)

def _strip_prefixes(state_dict: dict[str, torch.Tensor], prefixes: list[str]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for (key, value) in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                cleaned[key[len(prefix):]] = value
                break
    return cleaned

def load_qmodel_encoder_weights(model: nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    state = torch.load(normalize_path_string(checkpoint_path), map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        raw_state = state.get('model', state.get('state_dict', state))
    else:
        raw_state = state
    cleaned = _strip_prefixes(raw_state, prefixes=['module.model.', 'model.', 'encoder.encoder.', 'encoder.', 'graph_encoder.encoder.', 'graph_encoder.'])
    model_state = model.state_dict()
    filtered = {}
    for (key, value) in cleaned.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if target.shape == value.shape:
            filtered[key] = value
        elif key == 'embedding_in.weight' and target.shape[0] == value.shape[0] and (target.shape[1] >= value.shape[1]):
            merged = target.clone()
            merged[:, :value.shape[1]] = value
            filtered[key] = merged
    load_result = model.load_state_dict(filtered, strict=False)
    return (list(load_result.missing_keys), list(load_result.unexpected_keys))

class QModelGraphEncoder(nn.Module):

    def __init__(self, cfg: GraphModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = EGNN_last(in_node_nf=cfg.max_atom_type * (cfg.charge_power + 1) + cfg.node_feature_dim, hidden_nf=cfg.hidden_dim, in_edge_nf=0 if cfg.no_edge_types else 4, n_layers=cfg.n_layers, attention=cfg.attention, use_layer_norm=cfg.layernorm)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, atom_mask: torch.Tensor, extra_node_features: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        (batch_size, n_nodes, _) = positions.size()
        charges_flat = charges.view(batch_size * n_nodes, 1)
        node_features = process_input(charges_flat, self.cfg.max_atom_type, self.cfg.charge_power)
        node_features = node_features.view(batch_size * n_nodes, -1)
        if self.cfg.node_feature_dim > 0 and extra_node_features is not None:
            node_features = torch.cat([node_features, extra_node_features.view(batch_size * n_nodes, -1).to(node_features.dtype)], dim=-1)
        if self.cfg.no_edge_types:
            (edges, edge_types) = gen_fully_connected(positions, atom_mask)
            edge_attr = None
        else:
            (edges, edge_types) = gen_fully_connected_with_hop(positions, atom_mask, self.cfg.cutoff)
            edge_attr = nn.functional.one_hot(edge_types, 4).to(torch.float32)
        (node_embeddings, _) = self.encoder(h=node_features, x=positions.view(batch_size * n_nodes, -1), edges=edges, edge_attr=edge_attr, edge_mask=None)
        flat_mask = atom_mask.view(batch_size * n_nodes, 1).to(node_embeddings.dtype)
        node_embeddings = (node_embeddings * flat_mask).view(batch_size, n_nodes, -1)
        return {'node_embeddings': node_embeddings, 'graph_embedding': node_embeddings.sum(dim=1), 'node_mask': atom_mask.bool()}
