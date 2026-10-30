import argparse
import pandas as pd
import torch
from rdkit import Chem
from predictor.inference import PropertyPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--architecture', required=True, choices=['graph_distilled', 'multimodal_distilled'])
    parser.add_argument('--artifacts_dir')
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--smiles_column', default='smiles')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model = PropertyPredictor(args.checkpoint, args.architecture, args.artifacts_dir).to(args.device)
    df = pd.read_csv(args.input_csv)
    valid = [i for (i, smi) in enumerate(df[args.smiles_column]) if isinstance(smi, str) and smi.strip() and (Chem.MolFromSmiles(smi) is not None)]
    column = f'pred_{model.targets[0]}'
    df[column] = float('nan')
    for start in range(0, len(valid), args.batch_size):
        rows = valid[start:start + args.batch_size]
        df.loc[rows, column] = model(df.loc[rows, args.smiles_column].tolist()).cpu().numpy()
    df.to_csv(args.output_csv, index=False)
if __name__ == '__main__':
    main()
