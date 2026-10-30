import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', default='data/metadata.csv')
    parser.add_argument('--smiles_column', default='smiles')
    parser.add_argument('--targets', nargs='+', default=['D', 'r0'])
    parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VALID', 'TEST'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv).rename(columns={args.smiles_column: 'smiles'})
    df = df.dropna(subset=['smiles']).reset_index(drop=True)
    canonical = df.smiles.map(lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
    if 'molecule_id' not in df:
        df.insert(0, 'molecule_id', [f'mol_{i:06d}' for i in range(len(df))])
    ratios = np.asarray(args.split_ratios, dtype=float)
    if np.any(ratios <= 0):
        parser.error('split_ratios must contain three positive values')
    ratios = ratios / ratios.sum()
    molecules = np.random.default_rng(args.seed).permutation(sorted(canonical.unique()))
    train_end = round(len(molecules) * ratios[0])
    valid_end = train_end + round(len(molecules) * ratios[1])
    train, valid, test = np.split(molecules, [train_end, valid_end])
    splits = {**dict.fromkeys(train, 'train'), **dict.fromkeys(valid, 'valid'), **dict.fromkeys(test, 'test')}
    df['split'] = canonical.map(splits)
    columns = ['molecule_id', 'smiles', *args.targets, 'split']
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, index=False)
    print(df['split'].value_counts())
if __name__ == '__main__':
    main()
