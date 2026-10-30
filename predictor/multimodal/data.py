from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from predictor.common.paths import normalize_path_string
from predictor.common.tokenizer import SmilesTokenizer

@dataclass
class MultimodalTransformerDataConfig:
    metadata_path: str
    graph_cache_dir: str
    molecule_id_column: str = 'molecule_id'
    smiles_column: str = 'smiles'
    split_column: str = 'split'
    train_split: str = 'train'
    valid_split: str = 'valid'
    test_split: str = 'test'
    batch_size: int = 32
    num_workers: int = 4
    max_seq_len: int = 256
    graph_encoder_type: str = '3d'

class MultimodalTransformerDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, cfg: MultimodalTransformerDataConfig, target_columns: list[str]) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.target_columns = target_columns
        self.graph_cache_dir = Path(normalize_path_string(cfg.graph_cache_dir))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        molecule_id = str(row[self.cfg.molecule_id_column])
        graph_obj = torch.load(self.graph_cache_dir / f'{molecule_id}.pt', weights_only=False)
        charges = graph_obj.get('atomic_numbers')
        positions = graph_obj.get('positions')
        if charges is None or positions is None:
            raise KeyError(f'graph cache for {molecule_id} missing atomic_numbers or positions')
        num_atoms = int(graph_obj.get('num_atoms', len(charges)))
        target_values = [float(row[col]) for col in self.target_columns]
        return {'molecule_id': molecule_id, 'smiles': str(row[self.cfg.smiles_column]), 'positions': positions[:num_atoms].to(torch.float32), 'charges': charges[:num_atoms].to(torch.long), 'num_atoms': num_atoms, 'target_values': target_values}

def pad_batch(batch: list[dict], tokenizer: SmilesTokenizer, max_seq_len: int, target_columns: list[str]) -> dict:
    max_atoms = max((int(item['num_atoms']) for item in batch))
    batch_size = len(batch)
    positions = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    charges = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    smiles_list = [item['smiles'] for item in batch]
    encoded = [tokenizer.encode(smiles, add_cls=False)[:max_seq_len] for smiles in smiles_list]
    max_len = max((len(ids) for ids in encoded))
    input_ids = []
    attention_mask = []
    labels = []
    for (idx, item) in enumerate(batch):
        num_atoms = int(item['num_atoms'])
        positions[idx, :num_atoms] = item['positions']
        charges[idx, :num_atoms] = item['charges']
        atom_mask[idx, :num_atoms] = True
        ids = encoded[idx]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append([float(v) for v in item['target_values'][:len(target_columns)]])
    return {'positions': positions, 'charges': charges, 'atom_mask': atom_mask, 'input_ids': torch.tensor(input_ids, dtype=torch.long), 'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 'labels_raw': torch.tensor(labels, dtype=torch.float32), 'molecule_id': [item['molecule_id'] for item in batch], 'smiles': smiles_list}

class MultimodalTransformerDataModule:

    def __init__(self, cfg: MultimodalTransformerDataConfig, target_columns: list[str], tokenizer_path: str | None=None) -> None:
        self.cfg = cfg
        self.target_columns = target_columns
        self.tokenizer_path = tokenizer_path
        self.tokenizer: Optional[SmilesTokenizer] = None

    def setup(self) -> None:
        df = pd.read_csv(normalize_path_string(self.cfg.metadata_path))
        required_cols = {self.cfg.molecule_id_column, self.cfg.smiles_column, self.cfg.split_column, *self.target_columns}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f'metadata.csv is missing columns: {sorted(missing_cols)}')
        df = df.dropna(subset=self.target_columns).reset_index(drop=True)
        self.train_df = df[df[self.cfg.split_column] == self.cfg.train_split].copy()
        self.valid_df = df[df[self.cfg.split_column] == self.cfg.valid_split].copy()
        self.test_df = df[df[self.cfg.split_column] == self.cfg.test_split].copy()
        if len(self.train_df) == 0:
            raise ValueError('train split is empty')
        if self.tokenizer_path:
            self.tokenizer = SmilesTokenizer.load_json(normalize_path_string(self.tokenizer_path))
        if self.tokenizer is None:
            self.tokenizer = SmilesTokenizer.build_from_smiles(self.train_df[self.cfg.smiles_column].astype(str).tolist())
        self.train_ds = MultimodalTransformerDataset(self.train_df, self.cfg, self.target_columns)
        self.valid_ds = MultimodalTransformerDataset(self.valid_df, self.cfg, self.target_columns)
        self.test_ds = MultimodalTransformerDataset(self.test_df, self.cfg, self.target_columns)

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        assert self.tokenizer is not None
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=lambda batch: pad_batch(batch, self.tokenizer, self.cfg.max_seq_len, self.target_columns))

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.valid_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, False)

    def export_artifacts(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.tokenizer is not None
        self.tokenizer.save_json(save_dir / 'tokenizer.json')
        schema = {'metadata_path': self.cfg.metadata_path, 'graph_cache_dir': self.cfg.graph_cache_dir, 'target_columns': self.target_columns, 'max_seq_len': self.cfg.max_seq_len, 'graph_encoder_type': self.cfg.graph_encoder_type}
        with open(save_dir / 'data_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
