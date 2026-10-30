from __future__ import annotations
import random
from dataclasses import dataclass
import pandas as pd
from rdkit import Chem

@dataclass
class SmilesAugmentationConfig:
    enabled: bool = True
    num_randomizations: int = 10
    include_canonical: bool = True
    deduplicate: bool = True
    random_seed: int = 42

def randomize_smiles(smiles: str, rng: random.Random) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        atom_order = list(range(mol.GetNumAtoms()))
        rng.shuffle(atom_order)
        randomized_mol = Chem.RenumberAtoms(mol, atom_order)
        return Chem.MolToSmiles(randomized_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return smiles

def build_augmented_train_df(train_df: pd.DataFrame, smiles_column: str, aug_cfg: SmilesAugmentationConfig) -> pd.DataFrame:
    if not aug_cfg.enabled or aug_cfg.num_randomizations <= 0:
        augmented = train_df.copy()
        augmented['canonical_smiles'] = augmented[smiles_column].astype(str)
        augmented['is_augmented'] = False
        return augmented.reset_index(drop=True)
    rows: list[dict] = []
    rng = random.Random(aug_cfg.random_seed)
    for (_, row) in train_df.iterrows():
        row_dict = row.to_dict()
        canonical_smiles = str(row_dict[smiles_column])
        variants: list[str] = []
        if aug_cfg.include_canonical:
            variants.append(canonical_smiles)
        for _ in range(aug_cfg.num_randomizations):
            variants.append(randomize_smiles(canonical_smiles, rng))
        if aug_cfg.deduplicate:
            seen = set()
            deduped = []
            for smi in variants:
                if smi not in seen:
                    deduped.append(smi)
                    seen.add(smi)
            variants = deduped
        if not variants:
            variants = [canonical_smiles]
        for smi in variants:
            sample = dict(row_dict)
            sample[smiles_column] = smi
            sample['canonical_smiles'] = canonical_smiles
            sample['is_augmented'] = smi != canonical_smiles
            rows.append(sample)
    return pd.DataFrame(rows).reset_index(drop=True)
