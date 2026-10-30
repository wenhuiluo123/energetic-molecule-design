import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D


class GPTGenerator(pl.LightningModule):
    def __init__(self, vocab, max_seq_len=256, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = dict(vocab)
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding1D(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, len(vocab), bias=False)

    def forward(self, src, src_key_padding_mask=None):
        emb = self.embedding(src)
        emb = emb + self.pos_encoder(emb)
        mask = torch.triu(torch.ones(src.size(1), src.size(1),
                                    dtype=torch.bool, device=src.device), diagonal=1)
        hidden = self.decoder(emb, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.lm_head(hidden)

    def _step(self, batch, name):
        src, pad_mask, target = batch
        logits = self(src, pad_mask)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=0)
        self.log(name, loss, prog_bar=True, batch_size=src.size(0), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['vocab'] = self.vocab

    @torch.no_grad()
    def generate(self, max_length=None, temperature=1.0, top_k=10):
        self.eval()
        generated = torch.tensor([[self.vocab['<START>']]], device=self.device)
        for _ in range((max_length or self.max_seq_len) - 1):
            logits = self(generated)[:, -1] / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                sampled = torch.multinomial(values.softmax(-1), 1)
                token = indices.gather(1, sampled)
            else:
                token = torch.multinomial(logits.softmax(-1), 1)
            generated = torch.cat([generated, token], dim=1)
            if token.item() == self.vocab['<END>']:
                break
        return generated
