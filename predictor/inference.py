import json
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pad_sequence
from predictor.common.tokenizer import SmilesTokenizer
from predictor.gpt.data import inverse_transform_tensor
from predictor.gpt.model import RLGPTRegressor, RLGPTRegressorConfig
from predictor.graph_to_smiles.model import DistillLossConfig, build_smiles_student

class PropertyPredictor(nn.Module):

    def __init__(self, checkpoint, architecture, artifacts_dir=None):
        super().__init__()
        artifacts = Path(artifacts_dir) if artifacts_dir else Path(checkpoint).parent.parent
        cfg = yaml.safe_load((artifacts / 'resolved_config.yaml').read_text(encoding='utf-8'))
        schema = json.loads((artifacts / 'data_schema.json').read_text(encoding='utf-8'))
        self.stats = json.loads((artifacts / 'stats.json').read_text(encoding='utf-8'))
        self.tokenizer = SmilesTokenizer.load_json(artifacts / 'tokenizer.json')
        self.targets = schema['target_columns']
        self.max_seq_len = int(schema['max_seq_len'])
        self.architecture = architecture
        common = dict(vocab_size=len(self.tokenizer.vocab), max_seq_len=self.max_seq_len, pad_token_id=self.tokenizer.pad_token_id, num_targets=len(self.targets))
        if architecture == 'graph_distilled':
            self.model = build_smiles_student('rl_gpt', cfg['student_model'], **common, teacher_dim=cfg['teacher_model']['hidden_dim'], loss_cfg=DistillLossConfig(**cfg['distill']))
        elif architecture == 'multimodal_distilled':
            self.model = RLGPTRegressor(RLGPTRegressorConfig(**common, **cfg['model']), self.targets, self.stats)
        else:
            raise ValueError(architecture)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state.get('state_dict', state.get('model', state)))
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, smiles):
        device = next(self.model.parameters()).device
        sequences = [torch.tensor(self.tokenizer.encode(smi, add_cls=False)[:self.max_seq_len], dtype=torch.long) for smi in smiles]
        ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(device)
        mask = ids.eq(self.tokenizer.pad_token_id)
        if self.architecture == 'graph_distilled':
            values = self.model(ids, (~mask).long())['preds']
        else:
            values = self.model(ids, mask)
        return inverse_transform_tensor(values, self.targets, self.stats).squeeze(-1)
