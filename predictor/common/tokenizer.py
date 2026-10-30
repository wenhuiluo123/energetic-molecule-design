from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
SMILES_PATTERN = re.compile('(\\[[^\\]]+\\]|%[0-9]{2}|Br|Cl|Si|Se|Na|Ca|Li|Al|Mg|Zn|Fe|Cu|Mn|As|Co|Ni|[BCNOPSFIbcnops]|[0-9]|\\(|\\)|\\.|=|#|-|\\+|\\\\|/|:|~|@|\\?|\\*|\\$)')

@dataclass
class TokenizerSpecialTokens:
    pad_token: str = '[PAD]'
    unk_token: str = '[UNK]'
    cls_token: str = '[CLS]'

class SmilesTokenizer:

    def __init__(self, vocab: Dict[str, int], specials: TokenizerSpecialTokens | None=None) -> None:
        self.specials = specials or TokenizerSpecialTokens()
        self.vocab = vocab
        self.pad_token_id = vocab[self.specials.pad_token]
        self.unk_token_id = vocab[self.specials.unk_token]
        self.cls_token_id = vocab[self.specials.cls_token]

    @classmethod
    def build_from_smiles(cls, smiles_list: List[str]) -> 'SmilesTokenizer':
        specials = TokenizerSpecialTokens()
        vocab: Dict[str, int] = {specials.pad_token: 0, specials.unk_token: 1, specials.cls_token: 2}
        token_set = set()
        for smiles in smiles_list:
            token_set.update(cls._tokenize_static(smiles))
        for tok in sorted(token_set):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab=vocab, specials=specials)

    @staticmethod
    def _tokenize_static(smiles: str) -> List[str]:
        tokens = SMILES_PATTERN.findall(smiles)
        if not tokens:
            return list(smiles)
        if ''.join(tokens) != smiles:
            return list(smiles)
        return tokens

    def tokenize(self, smiles: str) -> List[str]:
        return self._tokenize_static(smiles)

    def encode(self, smiles: str, add_cls: bool=False) -> List[int]:
        tokens = self.tokenize(smiles)
        token_ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_cls:
            token_ids = [self.cls_token_id] + token_ids
        return token_ids

    def decode(self, token_ids: List[int]) -> List[str]:
        inv_vocab = {v: k for (k, v) in self.vocab.items()}
        return [inv_vocab.get(i, self.specials.unk_token) for i in token_ids]

    def to_dict(self) -> dict:
        return {'vocab': self.vocab, 'specials': asdict(self.specials)}

    @classmethod
    def from_dict(cls, data: dict) -> 'SmilesTokenizer':
        special_fields = TokenizerSpecialTokens.__dataclass_fields__
        specials_data = {key: value for (key, value) in data['specials'].items() if key in special_fields}
        specials = TokenizerSpecialTokens(**specials_data)
        return cls(vocab=data['vocab'], specials=specials)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load_json(cls, path: str | Path) -> 'SmilesTokenizer':
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls.from_dict(data)
