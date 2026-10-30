from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from predictor.graph.data import fit_target_stats, normalize_targets, pad_graph_tensors
from predictor.common.paths import normalize_path_string
from predictor.common.tokenizer import SmilesTokenizer
from predictor.common.augmentation import SmilesAugmentationConfig, build_augmented_train_df

@dataclass
class EmbedToSmiDataConfig:
    metadata_path: str
    graph_cache_dir: str
    molecule_id_column: str = 'molecule_id'
    smiles_column: str = 'smiles'
    split_column: str = 'split'
    train_split: str = 'train'
    valid_split: str = 'valid'
    test_split: str = 'test'
    target_columns: list[str] | None = None
    normalization: str = 'zscore'
    batch_size: int = 32
    num_workers: int = 4
    max_seq_len: int = 256
    augmentation_enabled: bool = True
    augmentation_stage1: bool | None = None
    augmentation_stage2: bool | None = None
    augmentation_num_randomizations: int = 10
    augmentation_include_canonical: bool = True
    augmentation_deduplicate: bool = True
    augmentation_random_seed: int = 42

    def to_augmentation_config(self) -> SmilesAugmentationConfig:
        return SmilesAugmentationConfig(enabled=self.augmentation_enabled, num_randomizations=self.augmentation_num_randomizations, include_canonical=self.augmentation_include_canonical, deduplicate=self.augmentation_deduplicate, random_seed=self.augmentation_random_seed)

class EmbedToSmiDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, cfg: EmbedToSmiDataConfig, target_stats: dict) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.target_stats = target_stats
        self.graph_cache_dir = Path(normalize_path_string(cfg.graph_cache_dir))
        self.load_graph = True

    def set_graph_loading(self, enabled: bool) -> None:
        self.load_graph = bool(enabled)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        molecule_id = str(row[self.cfg.molecule_id_column])
        target_values = [float(row[col]) for col in self.cfg.target_columns or []]
        norm_targets = normalize_targets(target_values, self.cfg.target_columns or [], self.target_stats)
        sample = {'molecule_id': molecule_id, 'smiles': str(row[self.cfg.smiles_column]), 'labels': norm_targets, 'labels_raw': target_values}
        if not self.load_graph:
            return sample
        graph_path = self.graph_cache_dir / f'{molecule_id}.pt'
        graph_obj = torch.load(graph_path, weights_only=False)
        charges = graph_obj.get('atomic_numbers')
        positions = graph_obj.get('positions')
        if charges is None or positions is None:
            raise KeyError(f'{graph_path} is missing atomic_numbers or positions')
        num_atoms = int(graph_obj.get('num_atoms', len(charges)))
        sample.update({'positions': positions[:num_atoms].to(torch.float32), 'charges': charges[:num_atoms].to(torch.long), 'num_atoms': num_atoms, 'extra_node_features': None, 'graph_features': []})
        return sample

class EmbedToSmiDataModule:

    def __init__(self, cfg: EmbedToSmiDataConfig, *, student_tokenizer_path: str | None=None, teacher_tokenizer_path: str, teacher_graph_encoder_type: str, teacher_max_seq_len: int) -> None:
        self.cfg = cfg
        self.student_tokenizer_path = student_tokenizer_path
        self.teacher_tokenizer_path = teacher_tokenizer_path
        self.teacher_graph_encoder_type = teacher_graph_encoder_type.lower()
        self.teacher_max_seq_len = int(teacher_max_seq_len)
        self.student_tokenizer: Optional[SmilesTokenizer] = None
        self.teacher_tokenizer: Optional[SmilesTokenizer] = None
        self.target_stats: Optional[dict] = None
        self.aug_cfg = cfg.to_augmentation_config()

    def setup(self, stage: Optional[str]=None) -> None:
        if not self.cfg.target_columns:
            raise ValueError('data.target_columns must not be empty')
        df = pd.read_csv(normalize_path_string(self.cfg.metadata_path))
        required_cols = {self.cfg.molecule_id_column, self.cfg.smiles_column, self.cfg.split_column, *self.cfg.target_columns}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f'metadata.csv is missing columns: {sorted(missing_cols)}')
        df = df.dropna(subset=self.cfg.target_columns).reset_index(drop=True)
        self.train_df_raw = df[df[self.cfg.split_column] == self.cfg.train_split].copy().reset_index(drop=True)
        self.valid_df = df[df[self.cfg.split_column] == self.cfg.valid_split].copy().reset_index(drop=True)
        self.test_df = df[df[self.cfg.split_column] == self.cfg.test_split].copy().reset_index(drop=True)
        if len(self.train_df_raw) == 0:
            raise ValueError('train split is empty')
        stage1_aug_cfg = self.cfg.to_augmentation_config()
        stage2_aug_cfg = self.cfg.to_augmentation_config()
        if self.cfg.augmentation_stage1 is not None:
            stage1_aug_cfg.enabled = bool(self.cfg.augmentation_stage1)
        if self.cfg.augmentation_stage2 is not None:
            stage2_aug_cfg.enabled = bool(self.cfg.augmentation_stage2)
        self.stage1_train_df = build_augmented_train_df(train_df=self.train_df_raw, smiles_column=self.cfg.smiles_column, aug_cfg=stage1_aug_cfg)
        self.stage2_train_df = build_augmented_train_df(train_df=self.train_df_raw, smiles_column=self.cfg.smiles_column, aug_cfg=stage2_aug_cfg)
        self.train_df = self.stage1_train_df
        if self.student_tokenizer_path:
            self.student_tokenizer = SmilesTokenizer.load_json(normalize_path_string(self.student_tokenizer_path))
        if self.student_tokenizer is None:
            self.student_tokenizer = SmilesTokenizer.build_from_smiles(self.stage1_train_df[self.cfg.smiles_column].astype(str).tolist() + self.stage2_train_df[self.cfg.smiles_column].astype(str).tolist())
        self.teacher_tokenizer = SmilesTokenizer.load_json(normalize_path_string(self.teacher_tokenizer_path))
        self.target_stats = fit_target_stats(self.train_df_raw, self.cfg.target_columns, self.cfg.normalization)
        self.stage1_train_ds = EmbedToSmiDataset(self.stage1_train_df, self.cfg, self.target_stats)
        self.stage2_train_ds = EmbedToSmiDataset(self.stage2_train_df, self.cfg, self.target_stats)
        self.train_ds = self.stage1_train_ds
        self.valid_ds = EmbedToSmiDataset(self.valid_df, self.cfg, self.target_stats)
        self.test_ds = EmbedToSmiDataset(self.test_df, self.cfg, self.target_stats)
        cache_df = pd.concat([self.stage1_train_df, self.valid_df, self.test_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=[self.cfg.molecule_id_column, self.cfg.smiles_column]).reset_index(drop=True)
        self.teacher_cache_ds = EmbedToSmiDataset(cache_df, self.cfg, self.target_stats)

    def disable_graph_loading_for_cached_teacher(self) -> None:
        for dataset in (self.stage1_train_ds, self.stage2_train_ds, self.valid_ds, self.test_ds):
            dataset.set_graph_loading(False)

    def collate_fn(self, batch: list[dict]) -> dict:
        assert self.student_tokenizer is not None
        assert self.teacher_tokenizer is not None
        has_graph = 'positions' in batch[0]
        if has_graph:
            out = pad_graph_tensors(batch)
        else:
            out = {'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.float32), 'labels_raw': torch.tensor([item['labels_raw'] for item in batch], dtype=torch.float32), 'molecule_id': [item['molecule_id'] for item in batch], 'smiles': [item['smiles'] for item in batch]}
        smiles_list = [item['smiles'] for item in batch]
        student_encoded = [self.student_tokenizer.encode(smiles, add_cls=False)[:self.cfg.max_seq_len] for smiles in smiles_list]
        max_student_len = max((len(ids) for ids in student_encoded))
        student_input_ids = []
        student_attention_mask = []
        for ids in student_encoded:
            pad_len = max_student_len - len(ids)
            student_input_ids.append(ids + [self.student_tokenizer.pad_token_id] * pad_len)
            student_attention_mask.append([1] * len(ids) + [0] * pad_len)
        out['student_input_ids'] = torch.tensor(student_input_ids, dtype=torch.long)
        out['student_attention_mask'] = torch.tensor(student_attention_mask, dtype=torch.long)
        if has_graph:
            teacher_encoded = [self.teacher_tokenizer.encode(smiles, add_cls=False)[:self.teacher_max_seq_len] for smiles in smiles_list]
            max_teacher_len = max((len(ids) for ids in teacher_encoded))
            teacher_input_ids = []
            teacher_attention_mask = []
            for ids in teacher_encoded:
                pad_len = max_teacher_len - len(ids)
                teacher_input_ids.append(ids + [self.teacher_tokenizer.pad_token_id] * pad_len)
                teacher_attention_mask.append([1] * len(ids) + [0] * pad_len)
            out['teacher_input_ids'] = torch.tensor(teacher_input_ids, dtype=torch.long)
            out['teacher_attention_mask'] = torch.tensor(teacher_attention_mask, dtype=torch.long)
            out['teacher_atom_mask'] = out['atom_mask']
        return out

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def stage1_train_dataloader(self) -> DataLoader:
        return self._make_loader(self.stage1_train_ds, shuffle=True)

    def stage2_train_dataloader(self) -> DataLoader:
        return self._make_loader(self.stage2_train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.valid_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)

    def teacher_cache_dataloader(self) -> DataLoader:
        return self._make_loader(self.teacher_cache_ds, shuffle=False)

    def export_artifacts(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.student_tokenizer is not None
        assert self.target_stats is not None
        self.student_tokenizer.save_json(save_dir / 'tokenizer.json')
        with open(save_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_stats, f, ensure_ascii=False, indent=2)
        schema = {'metadata_path': self.cfg.metadata_path, 'graph_cache_dir': self.cfg.graph_cache_dir, 'molecule_id_column': self.cfg.molecule_id_column, 'smiles_column': self.cfg.smiles_column, 'split_column': self.cfg.split_column, 'train_split': self.cfg.train_split, 'valid_split': self.cfg.valid_split, 'test_split': self.cfg.test_split, 'target_columns': self.cfg.target_columns, 'normalization': self.cfg.normalization, 'student_max_seq_len': self.cfg.max_seq_len, 'teacher_max_seq_len': self.teacher_max_seq_len, 'teacher_graph_encoder_type': self.teacher_graph_encoder_type, 'augmentation': asdict(self.aug_cfg), 'augmentation_stage1': self.cfg.augmentation_enabled if self.cfg.augmentation_stage1 is None else bool(self.cfg.augmentation_stage1), 'augmentation_stage2': self.cfg.augmentation_enabled if self.cfg.augmentation_stage2 is None else bool(self.cfg.augmentation_stage2), 'train_size_raw': int(len(self.train_df_raw)), 'train_size_augmented': int(len(self.train_df)), 'stage1_train_size': int(len(self.stage1_train_df)), 'stage2_train_size': int(len(self.stage2_train_df))}
        with open(save_dir / 'data_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
