from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alldata_utils import atom_features, bond_features, read_coors
from rdkit import Chem


def process_sdf_with_targets(sdf_path: str, target_values: dict[str, float]) -> dict:
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse SDF: {sdf_path}")

    mol_atom_prop = []
    mol_bond_prop = []
    charges = []

    num_atoms = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        mol_atom_prop.append(atom_features(atom))
        charges.append(atom.GetAtomicNum())
    for bond in mol.GetBonds():
        mol_bond_prop.append(bond_features(bond))

    atom_position = read_coors(sdf_path)
    molecule = {
        "mol_atom_prop": mol_atom_prop,
        "charges": charges,
        "mol_bond_prop": mol_bond_prop,
        "num_atoms": num_atoms,
        "positions": atom_position,
    }
    for key, value in target_values.items():
        molecule[key] = float(value)

    return {
        key: (val if isinstance(val, np.ndarray) else np.array(val) if not hasattr(val, "shape") else val)
        for key, val in molecule.items()
    }


def pad_stack(arrays: list) -> np.ndarray:
    if arrays[0].ndim == 0:
        return np.stack(arrays, axis=0)

    max_shape = [max(arr.shape[dim] for arr in arrays) for dim in range(arrays[0].ndim)]
    out = np.zeros((len(arrays), *max_shape), dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        slices = tuple(slice(0, s) for s in arr.shape)
        out[(i, *slices)] = arr
    return out


def convert(records: list[dict]) -> dict[str, np.ndarray]:
    keys = list(records[0].keys())
    grouped: dict[str, list] = {key: [] for key in keys}
    for record in records:
        for key in keys:
            value = record[key]
            if hasattr(value, "numpy"):
                value = value.numpy()
            grouped[key].append(np.asarray(value))
    return {key: pad_stack(vals) for key, vals in grouped.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 3D_GNN non-MDN train/valid/test npz files from metadata and source SDF paths."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/processed/qmodel_778_recovered/metadata_with_bde.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictor/3D_GNN/data/778_nonmdn_q_bde",
    )
    parser.add_argument(
        "--target_columns",
        nargs="+",
        default=["Q", "BDE"],
    )
    parser.add_argument("--split_column", type=str, default="split")
    parser.add_argument("--source_sdf_column", type=str, default="source_sdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {args.split_column, args.source_sdf_column, *args.target_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for split_name in ["train", "valid", "test"]:
        split_df = df.loc[df[args.split_column] == split_name].reset_index(drop=True)
        if len(split_df) == 0:
            raise ValueError(f"No rows found for split={split_name}")

        records = []
        for _, row in split_df.iterrows():
            target_values = {col: row[col] for col in args.target_columns}
            records.append(process_sdf_with_targets(str(row[args.source_sdf_column]), target_values))

        arrays = convert(records)
        save_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(save_path, **arrays)
        print(f"Saved {split_name} -> {save_path}")
        print({k: arrays[k].shape for k in arrays.keys()})


if __name__ == "__main__":
    main()
