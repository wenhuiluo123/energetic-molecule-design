import argparse
from pathlib import Path
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def build_graph(smiles, seed, num_confs):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        raise ValueError(f'Embedding failed: {smiles}')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        energies = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=1000)
    else:
        energies = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=1000)
    best = min(zip(conf_ids, energies), key=lambda item: item[1][1])[0]
    return {'atomic_numbers': torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]), 'positions': torch.tensor(mol.GetConformer(best).GetPositions(), dtype=torch.float32), 'num_atoms': mol.GetNumAtoms(), 'smiles': smiles}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/metadata.csv')
    parser.add_argument('--output_dir', default='data/graph_cache')
    parser.add_argument('--num_confs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metadata)
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        path = output / f'{row.molecule_id}.pt'
        if not path.exists():
            torch.save(build_graph(row.smiles, args.seed, args.num_confs), path)
if __name__ == '__main__':
    main()
