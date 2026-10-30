from __future__ import annotations
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from predictor.common.paths import normalize_path_string
from predictor.common.tokenizer import SmilesTokenizer

@dataclass
class RLGPTAugmentationConfig:
    enabled: bool = True
    num_randomizations: int = 10
    include_canonical: bool = True
    deduplicate: bool = True
    random_seed: int = 42

@dataclass
class RLGPTRegDataConfig:
    metadata_path: str
    tokenizer_path: str = ''
    smiles_column: str = 'smiles'
    split_column: str = 'split'
    train_split: str = 'train'
    valid_split: str = 'valid'
    test_split: str = 'test'
    target_columns: list[str] | None = None
    kfolds: int = 1
    fold_index: int = 1
    valid_fraction: float = 0.2
    stratify_bins: int = 10
    normalization: str = 'zscore'
    log_transform_columns: list[str] | None = None
    auto_log_h50: bool = True
    batch_size: int = 256
    num_workers: int = 4
    max_seq_len: int = 256
    augmentation_enabled: bool = True
    augmentation_num_randomizations: int = 10
    augmentation_include_canonical: bool = True
    augmentation_deduplicate: bool = True
    augmentation_random_seed: int = 42

    def to_augmentation_config(self) -> RLGPTAugmentationConfig:
        return RLGPTAugmentationConfig(enabled=self.augmentation_enabled, num_randomizations=self.augmentation_num_randomizations, include_canonical=self.augmentation_include_canonical, deduplicate=self.augmentation_deduplicate, random_seed=self.augmentation_random_seed)

def _target_transform(col: str, cfg: RLGPTRegDataConfig) -> str:
    log_cols = set(cfg.log_transform_columns or [])
    if col in log_cols:
        return 'log'
    if cfg.auto_log_h50 and col.lower() in {'h50(obs)', 'h50_obs', 'h50'}:
        return 'log'
    return 'none'

def _transformed_values(values: pd.Series, col: str, cfg: RLGPTRegDataConfig) -> pd.Series:
    transform = _target_transform(col, cfg)
    values = values.astype(float)
    if transform == 'log':
        if (values <= 0).any():
            raise ValueError(f'Column {col!r} contains non-positive values, cannot apply log transform')
        return np.log(values)
    return values

def _scale_from_stats(target_stats: dict[str, Any], col: str) -> float:
    item = target_stats['targets'][col]
    if 'std' in item:
        return float(item['std'])
    if 'mad' in item:
        return float(item['mad'])
    return 1.0

def fit_target_stats(train_df: pd.DataFrame, target_columns: list[str], cfg: RLGPTRegDataConfig) -> dict[str, Any]:
    stats: dict[str, Any] = {'normalization': cfg.normalization, 'targets': {}}
    if cfg.normalization not in {'zscore', 'mean_mad', 'none'}:
        raise ValueError(f'Unsupported normalization={cfg.normalization!r}')
    for col in target_columns:
        values = _transformed_values(train_df[col], col, cfg)
        transform = _target_transform(col, cfg)
        if cfg.normalization == 'none':
            stats['targets'][col] = {'transform': transform}
            continue
        mean = float(values.mean())
        if cfg.normalization == 'zscore':
            scale = float(values.std(ddof=0))
            scale_key = 'std'
        else:
            scale = float((values - mean).abs().mean())
            scale_key = 'mad'
        if scale == 0.0:
            scale = 1.0
        stats['targets'][col] = {'mean': mean, scale_key: scale, 'transform': transform}
    return stats

def normalize_targets(target_values: list[float], target_columns: list[str], target_stats: dict[str, Any]) -> list[float]:
    normalized = []
    for (value, col) in zip(target_values, target_columns):
        item = target_stats['targets'][col]
        transformed = float(value)
        if item.get('transform') == 'log':
            if transformed <= 0:
                raise ValueError(f'Target {col!r} contains non-positive value after split: {value}')
            transformed = float(np.log(transformed))
        if target_stats['normalization'] == 'none':
            normalized.append(transformed)
        else:
            normalized.append((transformed - float(item['mean'])) / _scale_from_stats(target_stats, col))
    return normalized

def inverse_transform_tensor(tensor: torch.Tensor, target_columns: list[str], target_stats: dict[str, Any]) -> torch.Tensor:
    columns = []
    for (idx, col) in enumerate(target_columns):
        values = tensor[:, idx]
        item = target_stats['targets'][col]
        if target_stats['normalization'] != 'none':
            values = values * _scale_from_stats(target_stats, col) + float(item['mean'])
        if item.get('transform') == 'log':
            values = torch.exp(values)
        columns.append(values)
    return torch.stack(columns, dim=1)

def randomize_smiles(smiles: str, rng: random.Random) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    try:
        atom_order = list(range(mol.GetNumAtoms()))
        rng.shuffle(atom_order)
        randomized_mol = Chem.RenumberAtoms(mol, atom_order)
        return Chem.MolToSmiles(randomized_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return str(smiles)

def build_augmented_train_df(train_df: pd.DataFrame, smiles_column: str, aug_cfg: RLGPTAugmentationConfig) -> pd.DataFrame:
    if not aug_cfg.enabled or aug_cfg.num_randomizations <= 0:
        augmented = train_df.copy()
        augmented['canonical_smiles'] = augmented[smiles_column].astype(str)
        augmented['is_augmented'] = False
        return augmented.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    rng = random.Random(aug_cfg.random_seed)
    for (_, row) in train_df.iterrows():
        row_dict = row.to_dict()
        canonical = str(row_dict[smiles_column])
        variants: list[str] = []
        if aug_cfg.include_canonical:
            variants.append(canonical)
        for _ in range(aug_cfg.num_randomizations):
            variants.append(randomize_smiles(canonical, rng))
        if aug_cfg.deduplicate:
            seen = set()
            variants = [smi for smi in variants if not (smi in seen or seen.add(smi))]
        if not variants:
            variants = [canonical]
        for smi in variants:
            sample = dict(row_dict)
            sample[smiles_column] = smi
            sample['canonical_smiles'] = canonical
            sample['is_augmented'] = smi != canonical
            rows.append(sample)
    return pd.DataFrame(rows).reset_index(drop=True)

def collate_regression_batch(batch: list[dict[str, Any]], tokenizer: SmilesTokenizer, max_seq_len: int) -> dict[str, Any]:
    pad_idx = tokenizer.pad_token_id
    input_ids = []
    for item in batch:
        ids = tokenizer.encode(str(item['smiles']), add_cls=False)[:max_seq_len]
        ids = ids + [pad_idx] * (max_seq_len - len(ids))
        input_ids.append(ids)
    x = torch.tensor(input_ids, dtype=torch.long)
    out = {'input_ids': x, 'pad_mask': x == pad_idx, 'smiles': [item['smiles'] for item in batch]}
    if 'labels' in batch[0]:
        out['labels'] = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
        out['labels_raw'] = torch.tensor([item['labels_raw'] for item in batch], dtype=torch.float32)
    return out

class RLGPTRegDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, smiles_column: str, target_columns: list[str], target_stats: dict[str, Any]) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.smiles_column = smiles_column
        self.target_columns = target_columns
        self.target_stats = target_stats

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_column])
        target_values = [float(row[col]) for col in self.target_columns]
        return {'smiles': smiles, 'labels': normalize_targets(target_values, self.target_columns, self.target_stats), 'labels_raw': target_values}

class RLGPTInferenceDataset(Dataset):

    def __init__(self, smiles_list: list[str]) -> None:
        self.smiles_list = smiles_list

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {'smiles': self.smiles_list[idx]}

class RLGPTRegDataModule(pl.LightningDataModule):

    def __init__(self, cfg: RLGPTRegDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.aug_cfg = cfg.to_augmentation_config()
        self.tokenizer: SmilesTokenizer | None = None
        if cfg.tokenizer_path:
            self.tokenizer = SmilesTokenizer.load_json(normalize_path_string(cfg.tokenizer_path))
        self.pad_idx = 0
        self.target_stats: dict[str, Any] | None = None

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError('DataModule.setup() must be called before reading vocab_size')
        return len(self.tokenizer.vocab)

    def _split_without_split_column(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.cfg.kfolds <= 1:
            (train_df, valid_df) = train_test_split(df, test_size=self.cfg.valid_fraction, random_state=42, shuffle=True)
            return (train_df.copy(), valid_df.copy(), valid_df.copy())
        first_target = self.cfg.target_columns[0]
        values = df[first_target].astype(float).to_numpy()
        bins = min(max(2, int(self.cfg.stratify_bins)), len(np.unique(values)))
        try:
            if bins > 1:
                y_bins = pd.qcut(values, q=bins, duplicates='drop', labels=False)
            else:
                y_bins = np.zeros(len(df), dtype=int)
            splitter = StratifiedKFold(n_splits=self.cfg.kfolds, shuffle=True, random_state=42)
            folds = list(splitter.split(df, y_bins))
        except ValueError:
            splitter = KFold(n_splits=self.cfg.kfolds, shuffle=True, random_state=42)
            folds = list(splitter.split(df))
        fold_idx = min(max(1, int(self.cfg.fold_index)), len(folds)) - 1
        (train_idx, valid_idx) = folds[fold_idx]
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()
        return (train_df, valid_df, valid_df.copy())

    def setup(self, stage: str | None=None) -> None:
        if not self.cfg.target_columns:
            raise ValueError('data.target_columns cannot be empty')
        df = pd.read_csv(normalize_path_string(self.cfg.metadata_path))
        required = {self.cfg.smiles_column, *self.cfg.target_columns}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'metadata CSV is missing required columns: {sorted(missing)}')
        df = df.dropna(subset=[self.cfg.smiles_column, *self.cfg.target_columns]).reset_index(drop=True)
        split_col = self.cfg.split_column
        if split_col and split_col in df.columns:
            train_df = df[df[split_col] == self.cfg.train_split].copy()
            valid_df = df[df[split_col] == self.cfg.valid_split].copy()
            test_df = df[df[split_col] == self.cfg.test_split].copy()
            if len(test_df) == 0:
                test_df = valid_df.copy()
        else:
            (train_df, valid_df, test_df) = self._split_without_split_column(df)
        if len(train_df) == 0:
            raise ValueError('train split is empty')
        if len(valid_df) == 0:
            raise ValueError('valid split is empty')
        self.train_df_raw = train_df.reset_index(drop=True)
        self.train_df = build_augmented_train_df(self.train_df_raw, self.cfg.smiles_column, self.aug_cfg)
        self.valid_df = valid_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        if self.tokenizer is None:
            self.tokenizer = SmilesTokenizer.build_from_smiles(self.train_df[self.cfg.smiles_column].astype(str).tolist())
        self.pad_idx = self.tokenizer.pad_token_id
        self.target_stats = fit_target_stats(self.train_df_raw, self.cfg.target_columns, self.cfg)
        self.train_ds = RLGPTRegDataset(self.train_df, self.cfg.smiles_column, self.cfg.target_columns, self.target_stats)
        self.valid_ds = RLGPTRegDataset(self.valid_df, self.cfg.smiles_column, self.cfg.target_columns, self.target_stats)
        self.test_ds = RLGPTRegDataset(self.test_df, self.cfg.smiles_column, self.cfg.target_columns, self.target_stats)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.tokenizer is not None
        return collate_regression_batch(batch, self.tokenizer, self.cfg.max_seq_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def export_artifacts(self, save_dir: str | Path) -> None:
        if self.target_stats is None:
            raise RuntimeError('setup() must be called before export_artifacts()')
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.tokenizer is not None
        with open(save_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_stats, f, ensure_ascii=False, indent=2)
        self.tokenizer.save_json(save_dir / 'tokenizer.json')
        schema = {'metadata_path': self.cfg.metadata_path, 'tokenizer_path': self.cfg.tokenizer_path, 'smiles_column': self.cfg.smiles_column, 'split_column': self.cfg.split_column, 'target_columns': self.cfg.target_columns, 'normalization': self.cfg.normalization, 'max_seq_len': self.cfg.max_seq_len, 'kfolds': self.cfg.kfolds, 'fold_index': self.cfg.fold_index, 'augmentation': asdict(self.aug_cfg), 'train_size_raw': int(len(self.train_df_raw)), 'train_size_augmented': int(len(self.train_df)), 'valid_size': int(len(self.valid_df)), 'test_size': int(len(self.test_df)), 'vocab_size': int(self.vocab_size), 'pad_token_id': int(self.pad_idx), 'tokenizer_format': 'smiles_tokenizer'}
        with open(save_dir / 'data_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
