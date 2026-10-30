from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from predictor.graph.data import fit_target_stats, normalize_targets, pad_graph_tensors
from predictor.common.paths import normalize_path_string
from predictor.common.tokenizer import SmilesTokenizer

@dataclass
class DistillDataConfig:
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
    graph_feature_prefixes: list[str] | None = None
    node_feature_key: str | None = None
    normalization: str = 'mean_mad'
    batch_size: int = 32
    num_workers: int = 4
    max_seq_len: int = 256
    smiles_randomize_train: bool = False
    smiles_randomize_stage1: bool | None = None
    smiles_randomize_stage2: bool | None = None
    smiles_randomize_prob: float = 1.0
    smiles_randomize_multiplier: int = 1
    smiles_randomize_validate: bool = True
    smiles_randomize_fail_on_error: bool = False

def randomize_smiles(smiles: str, validate: bool=True) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
    if validate and Chem.MolFromSmiles(randomized) is None:
        raise ValueError(f'Randomized SMILES is invalid: {randomized}')
    return randomized

class DistillDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, cfg: DistillDataConfig, target_stats: dict, randomize_smiles_train: bool=False) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.target_stats = target_stats
        self.graph_cache_dir = Path(normalize_path_string(cfg.graph_cache_dir))
        self.randomize_smiles_train = randomize_smiles_train
        self.load_graph = True

    def set_graph_loading(self, enabled: bool) -> None:
        self.load_graph = bool(enabled)

    def __len__(self) -> int:
        if not self.randomize_smiles_train:
            return len(self.df)
        return len(self.df) * max(1, int(self.cfg.smiles_randomize_multiplier))

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx % len(self.df)]
        molecule_id = str(row[self.cfg.molecule_id_column])
        target_values = [float(row[col]) for col in self.cfg.target_columns or []]
        labels = normalize_targets(target_values, self.cfg.target_columns or [], self.target_stats)
        smiles = str(row[self.cfg.smiles_column])
        if self.randomize_smiles_train and torch.rand(()) < float(self.cfg.smiles_randomize_prob):
            try:
                smiles = randomize_smiles(smiles, validate=self.cfg.smiles_randomize_validate)
            except Exception:
                if self.cfg.smiles_randomize_fail_on_error:
                    raise
        sample = {'molecule_id': molecule_id, 'smiles': smiles, 'labels': labels, 'labels_raw': target_values}
        if not self.load_graph:
            return sample
        graph_obj = torch.load(self.graph_cache_dir / f'{molecule_id}.pt', weights_only=False)
        charges = graph_obj.get('atomic_numbers')
        positions = graph_obj.get('positions')
        if charges is None or positions is None:
            raise KeyError(f'graph cache for {molecule_id} missing atomic_numbers or positions')
        num_atoms = int(graph_obj.get('num_atoms', len(charges)))
        extra_node_features = None
        if self.cfg.node_feature_key:
            extra_node_features = graph_obj.get(self.cfg.node_feature_key)
            if extra_node_features is None:
                raise KeyError(f'graph cache for {molecule_id} missing node feature key: {self.cfg.node_feature_key}')
            extra_node_features = extra_node_features[:num_atoms].to(torch.float32)
        sample.update({'positions': positions[:num_atoms].to(torch.float32), 'charges': charges[:num_atoms].to(torch.long), 'num_atoms': num_atoms, 'extra_node_features': extra_node_features, 'graph_features': [float(row[col]) for col in self.cfg.graph_feature_columns or []]})
        return sample

class DistillDataModule:

    def __init__(self, cfg: DistillDataConfig, tokenizer_path: str | None=None) -> None:
        self.cfg = cfg
        self.tokenizer_path = tokenizer_path
        self.tokenizer: Optional[SmilesTokenizer] = None
        self.target_stats: Optional[dict] = None

    def setup(self) -> None:
        if not self.cfg.target_columns:
            raise ValueError('data.target_columns must not be empty')
        df = pd.read_csv(normalize_path_string(self.cfg.metadata_path))
        graph_feature_columns = list(self.cfg.graph_feature_columns or [])
        for prefix in self.cfg.graph_feature_prefixes or []:
            graph_feature_columns.extend([col for col in df.columns if str(col).startswith(prefix)])
        self.cfg.graph_feature_columns = list(dict.fromkeys(graph_feature_columns))
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
        if len(self.valid_df) == 0:
            raise ValueError('valid split is empty')
        if len(self.test_df) == 0:
            self.test_df = self.valid_df.copy()
        if self.tokenizer_path:
            self.tokenizer = SmilesTokenizer.load_json(normalize_path_string(self.tokenizer_path))
        if self.tokenizer is None:
            self.tokenizer = SmilesTokenizer.build_from_smiles(self.train_df[self.cfg.smiles_column].astype(str).tolist())
        self.target_stats = fit_target_stats(self.train_df, self.cfg.target_columns, self.cfg.normalization)
        stage1_randomize = self.cfg.smiles_randomize_train if self.cfg.smiles_randomize_stage1 is None else bool(self.cfg.smiles_randomize_stage1)
        stage2_randomize = self.cfg.smiles_randomize_train if self.cfg.smiles_randomize_stage2 is None else bool(self.cfg.smiles_randomize_stage2)
        self.stage1_train_ds = DistillDataset(self.train_df, self.cfg, self.target_stats, randomize_smiles_train=stage1_randomize)
        self.stage2_train_ds = DistillDataset(self.train_df, self.cfg, self.target_stats, randomize_smiles_train=stage2_randomize)
        self.train_ds = self.stage1_train_ds
        self.valid_ds = DistillDataset(self.valid_df, self.cfg, self.target_stats)
        self.test_ds = DistillDataset(self.test_df, self.cfg, self.target_stats)
        cache_df = pd.concat([self.train_df, self.valid_df, self.test_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=[self.cfg.molecule_id_column]).reset_index(drop=True)
        self.teacher_cache_ds = DistillDataset(cache_df, self.cfg, self.target_stats)

    def disable_graph_loading_for_cached_teacher(self) -> None:
        for dataset in (self.stage1_train_ds, self.stage2_train_ds, self.valid_ds, self.test_ds):
            dataset.set_graph_loading(False)

    def collate_fn(self, batch: list[dict]) -> dict:
        assert self.tokenizer is not None
        if 'positions' in batch[0]:
            out = pad_graph_tensors(batch)
        else:
            out = {'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.float32), 'labels_raw': torch.tensor([item['labels_raw'] for item in batch], dtype=torch.float32), 'molecule_id': [item['molecule_id'] for item in batch], 'smiles': [item['smiles'] for item in batch]}
        encoded = [self.tokenizer.encode(item['smiles'], add_cls=False)[:self.cfg.max_seq_len] for item in batch]
        max_len = max((len(ids) for ids in encoded))
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        out['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        out['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        return out

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, True)

    def stage1_train_dataloader(self) -> DataLoader:
        return self._make_loader(self.stage1_train_ds, True)

    def stage2_train_dataloader(self) -> DataLoader:
        return self._make_loader(self.stage2_train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.valid_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, False)

    def teacher_cache_dataloader(self) -> DataLoader:
        return self._make_loader(self.teacher_cache_ds, False)

    def export_artifacts(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.tokenizer is not None
        assert self.target_stats is not None
        self.tokenizer.save_json(save_dir / 'tokenizer.json')
        with open(save_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_stats, f, ensure_ascii=False, indent=2)
        schema = {'metadata_path': self.cfg.metadata_path, 'graph_cache_dir': self.cfg.graph_cache_dir, 'molecule_id_column': self.cfg.molecule_id_column, 'smiles_column': self.cfg.smiles_column, 'split_column': self.cfg.split_column, 'target_columns': self.cfg.target_columns, 'graph_feature_columns': self.cfg.graph_feature_columns or [], 'graph_feature_prefixes': self.cfg.graph_feature_prefixes or [], 'node_feature_key': self.cfg.node_feature_key, 'normalization': self.cfg.normalization, 'max_seq_len': self.cfg.max_seq_len, 'smiles_randomize_train': self.cfg.smiles_randomize_train, 'smiles_randomize_stage1': self.cfg.smiles_randomize_train if self.cfg.smiles_randomize_stage1 is None else bool(self.cfg.smiles_randomize_stage1), 'smiles_randomize_stage2': self.cfg.smiles_randomize_train if self.cfg.smiles_randomize_stage2 is None else bool(self.cfg.smiles_randomize_stage2), 'smiles_randomize_prob': self.cfg.smiles_randomize_prob, 'smiles_randomize_multiplier': self.cfg.smiles_randomize_multiplier, 'smiles_randomize_validate': self.cfg.smiles_randomize_validate, 'smiles_randomize_fail_on_error': self.cfg.smiles_randomize_fail_on_error, 'train_size': int(len(self.train_df)), 'stage1_train_dataset_size': int(len(self.stage1_train_ds)), 'stage2_train_dataset_size': int(len(self.stage2_train_ds)), 'valid_size': int(len(self.valid_df)), 'test_size': int(len(self.test_df)), 'test_uses_valid_fallback': bool(len(self.test_df) == len(self.valid_df) and self.test_df.equals(self.valid_df))}
        with open(save_dir / 'data_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
