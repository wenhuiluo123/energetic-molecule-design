import argparse
import json
from pathlib import Path

import pandas as pd

from generator.tokenizer import SmilesTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True)
    parser.add_argument('--smiles_column', default='smiles')
    parser.add_argument('--output', default='data/vocab.json')
    args = parser.parse_args()
    smiles = pd.read_csv(args.data_csv)[args.smiles_column].dropna().tolist()
    tokenizer = SmilesTokenizer.build(smiles)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokenizer.vocab, indent=2), encoding='utf-8')
    print(f'{len(tokenizer.vocab)} tokens -> {path}')


if __name__ == '__main__':
    main()
