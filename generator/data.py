import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_seq_len=256, randomize=True,
                 num_augmentations=1, seed=42):
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.randomize = randomize
        self.num_augmentations = num_augmentations
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        sequences = []
        for _ in range(self.num_augmentations):
            smi = smiles
            if self.randomize:
                mol = Chem.MolFromSmiles(smiles)
                order = self.rng.permutation(mol.GetNumAtoms()).tolist()
                smi = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
            sequences.append(torch.tensor(self.tokenizer.encode(smi, self.max_seq_len)))
        return sequences


def collate_batch(batch):
    sequences = [seq for views in batch for seq in views]
    src = pad_sequence(sequences, batch_first=True, padding_value=0)
    target = torch.cat([src[:, 1:], torch.zeros(src.size(0), 1, dtype=torch.long)], dim=1)
    return src, src.eq(0), target
