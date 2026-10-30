import re


SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"
)
SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<MASK>', '<GLOBAL>', '<START>', '<END>']


class SmilesTokenizer:
    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.pad_id = self.vocab['<PAD>']
        self.start_id = self.vocab['<START>']
        self.end_id = self.vocab['<END>']

    @classmethod
    def build(cls, smiles):
        tokens = dict.fromkeys(SPECIAL_TOKENS)
        for smi in smiles:
            tokens.update(dict.fromkeys(SMILES_PATTERN.findall(smi)))
        return cls({token: idx for idx, token in enumerate(tokens)})

    def encode(self, smiles, max_length):
        ids = [self.vocab.get(token, self.vocab['<UNK>'])
               for token in SMILES_PATTERN.findall(smiles)]
        return [self.start_id] + ids[:max_length - 2] + [self.end_id]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            idx = int(idx)
            if idx == self.end_id:
                break
            token = self.id_to_token[idx]
            if token not in SPECIAL_TOKENS:
                tokens.append(token)
        return ''.join(tokens)
