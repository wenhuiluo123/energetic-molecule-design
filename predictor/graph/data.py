from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from predictor.common.paths import normalize_path_string

@dataclass
class GraphV2DataConfig:
    metadata_path: str
    graph_cache_dir: str
    molecule_id_column: str = 'molecule_id'
    smiles_column: str = 'smiles'
    split_column: str = 'split'
    train_split: str = 'train'
    valid_split: str = 'valid'
    test_split: str = 'test'
    target_columns: list[str] | None = None
    graph_feature_columns: list[str] | None = None
    node_feature_key: str | None = None
    normalization: str = 'mean_mad'
    batch_size: int = 4
    num_workers: int = 4

def fit_target_stats(train_df: pd.DataFrame, target_columns: list[str], normalization: str) -> dict:
    stats = {'normalization': normalization, 'targets': {}}
    if normalization == 'none':
        for col in target_columns:
            stats['targets'][col] = {}
        return stats
    if normalization not in {'zscore', 'mean_mad'}:
        raise ValueError(f'Unsupported normalization={normalization}')
    for col in target_columns:
        values = train_df[col].astype(float)
        mean = float(values.mean())
        if normalization == 'zscore':
            scale = float(values.std(ddof=0))
            scale_key = 'std'
        else:
            scale = float((values - mean).abs().mean())
            scale_key = 'mad'
        if scale == 0.0:
            scale = 1.0
        stats['targets'][col] = {'mean': mean, scale_key: scale}
    return stats

def _get_scale(target_stats: dict, col: str) -> float:
    item = target_stats['targets'][col]
    if 'std' in item:
        return float(item['std'])
    if 'mad' in item:
        return float(item['mad'])
    return 1.0

def normalize_targets(target_values: list[float], target_columns: list[str], target_stats: dict) -> list[float]:
    if target_stats['normalization'] == 'none':
        return [float(value) for value in target_values]
    normalized = []
    for (value, col) in zip(target_values, target_columns):
        mean = float(target_stats['targets'][col]['mean'])
        scale = _get_scale(target_stats, col)
        normalized.append((float(value) - mean) / scale)
    return normalized

def inverse_transform_tensor(tensor: torch.Tensor, target_columns: list[str], target_stats: dict) -> torch.Tensor:
    if target_stats['normalization'] == 'none':
        return tensor
    means = [float(target_stats['targets'][col]['mean']) for col in target_columns]
    scales = [_get_scale(target_stats, col) for col in target_columns]
    means_t = torch.tensor(means, dtype=tensor.dtype, device=tensor.device)
    scales_t = torch.tensor(scales, dtype=tensor.dtype, device=tensor.device)
    return tensor * scales_t + means_t

def pad_graph_tensors(batch: list[dict]) -> dict:
    max_atoms = max((int(item['num_atoms']) for item in batch))
    batch_size = len(batch)
    positions = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    charges = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    for (idx, item) in enumerate(batch):
        num_atoms = int(item['num_atoms'])
        positions[idx, :num_atoms] = item['positions'][:num_atoms]
        charges[idx, :num_atoms] = item['charges'][:num_atoms]
        atom_mask[idx, :num_atoms] = True
    edge_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(max_atoms, dtype=torch.bool).unsqueeze(0)
    edge_mask = (edge_mask & diag_mask).view(batch_size * max_atoms * max_atoms, 1)
    node_feature_dim = 0
    for item in batch:
        extra = item.get('extra_node_features')
        if torch.is_tensor(extra):
            node_feature_dim = int(extra.size(-1))
            break
    extra_node_features = torch.zeros(batch_size, max_atoms, node_feature_dim, dtype=torch.float32)
    if node_feature_dim > 0:
        for (idx, item) in enumerate(batch):
            extra = item.get('extra_node_features')
            if torch.is_tensor(extra):
                num_atoms = int(item['num_atoms'])
                extra_node_features[idx, :num_atoms] = extra[:num_atoms]
    graph_feature_dim = max((len(item.get('graph_features', [])) for item in batch), default=0)
    graph_features = torch.zeros(batch_size, graph_feature_dim, dtype=torch.float32)
    if graph_feature_dim > 0:
        for (idx, item) in enumerate(batch):
            values = item.get('graph_features', [])
            if len(values) > 0:
                graph_features[idx, :len(values)] = torch.tensor(values, dtype=torch.float32)
    return {'positions': positions, 'charges': charges, 'atom_mask': atom_mask, 'edge_mask': edge_mask, 'extra_node_features': extra_node_features, 'graph_features': graph_features, 'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.float32), 'labels_raw': torch.tensor([item['labels_raw'] for item in batch], dtype=torch.float32), 'molecule_id': [item['molecule_id'] for item in batch], 'smiles': [item['smiles'] for item in batch], 'num_atoms': torch.tensor([item['num_atoms'] for item in batch], dtype=torch.long)}

class GraphV2MoleculeDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, cfg: GraphV2DataConfig, target_stats: dict) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.target_stats = target_stats
        self.graph_cache_dir = Path(normalize_path_string(cfg.graph_cache_dir))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        molecule_id = str(row[self.cfg.molecule_id_column])
        graph_path = self.graph_cache_dir / f'{molecule_id}.pt'
        graph_obj = torch.load(graph_path, weights_only=False)
        target_values = [float(row[col]) for col in self.cfg.target_columns or []]
        norm_targets = normalize_targets(target_values, self.cfg.target_columns or [], self.target_stats)
        charges = graph_obj.get('atomic_numbers')
        positions = graph_obj.get('positions')
        if charges is None or positions is None:
            raise KeyError(f'{graph_path} is missing atomic_numbers or positions')
        num_atoms = int(graph_obj.get('num_atoms', len(charges)))
        extra_node_features = None
        if self.cfg.node_feature_key:
            extra_node_features = graph_obj.get(self.cfg.node_feature_key)
            if extra_node_features is None:
                raise KeyError(f'{graph_path} is missing node feature key: {self.cfg.node_feature_key}')
            extra_node_features = extra_node_features[:num_atoms].to(torch.float32)
        graph_features = []
        for col in self.cfg.graph_feature_columns or []:
            graph_features.append(float(row[col]))
        return {'molecule_id': molecule_id, 'smiles': str(row[self.cfg.smiles_column]), 'positions': positions[:num_atoms].to(torch.float32), 'charges': charges[:num_atoms].to(torch.long), 'num_atoms': num_atoms, 'extra_node_features': extra_node_features, 'graph_features': graph_features, 'labels': norm_targets, 'labels_raw': target_values}

class GraphV2DataModule:

    def __init__(self, cfg: GraphV2DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_stats: Optional[dict] = None

    def setup(self, stage: Optional[str]=None) -> None:
        if not self.cfg.target_columns:
            raise ValueError('data.target_columns must not be empty')
        df = pd.read_csv(normalize_path_string(self.cfg.metadata_path))
        required_cols = {self.cfg.molecule_id_column, self.cfg.smiles_column, self.cfg.split_column, *self.cfg.target_columns, *(self.cfg.graph_feature_columns or [])}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f'metadata.csv is missing columns: {sorted(missing_cols)}')
        df = df.dropna(subset=self.cfg.target_columns).reset_index(drop=True)
        self.train_df = df[df[self.cfg.split_column] == self.cfg.train_split].copy()
        self.valid_df = df[df[self.cfg.split_column] == self.cfg.valid_split].copy()
        self.test_df = df[df[self.cfg.split_column] == self.cfg.test_split].copy()
        if len(self.train_df) == 0:
            raise ValueError('train split is empty')
        self.target_stats = fit_target_stats(self.train_df, self.cfg.target_columns, self.cfg.normalization)
        self.train_ds = GraphV2MoleculeDataset(self.train_df, self.cfg, self.target_stats)
        self.valid_ds = GraphV2MoleculeDataset(self.valid_df, self.cfg, self.target_stats)
        self.test_ds = GraphV2MoleculeDataset(self.test_df, self.cfg, self.target_stats)

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=pad_graph_tensors)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.valid_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)

    def export_artifacts(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.target_stats is not None
        with open(save_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_stats, f, ensure_ascii=False, indent=2)
        schema = {'metadata_path': self.cfg.metadata_path, 'graph_cache_dir': self.cfg.graph_cache_dir, 'molecule_id_column': self.cfg.molecule_id_column, 'smiles_column': self.cfg.smiles_column, 'split_column': self.cfg.split_column, 'train_split': self.cfg.train_split, 'valid_split': self.cfg.valid_split, 'test_split': self.cfg.test_split, 'target_columns': self.cfg.target_columns, 'graph_feature_columns': self.cfg.graph_feature_columns, 'node_feature_key': self.cfg.node_feature_key, 'normalization': self.cfg.normalization}
        with open(save_dir / 'data_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
