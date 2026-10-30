import argparse
import json
from pathlib import Path
import shutil

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from generator.data import SmilesDataset, collate_batch
from generator.model import GPTGenerator
from generator.tokenizer import SmilesTokenizer


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    if (path / 'checkpoints').is_dir():
        path = path / 'checkpoints'
    if (path / 'best.ckpt').is_file():
        return path / 'best.ckpt'
    return min(path.glob('best-epoch=*-val_loss=*.ckpt'),
               key=lambda checkpoint: float(checkpoint.stem.split('val_loss=')[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='generator/configs/base.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    pl.seed_everything(cfg['seed'], workers=True)
    torch.set_num_threads(cfg['cpu_threads'])
    data = cfg['data']
    smiles = pd.read_csv(data['csv'])[data['smiles_column']].dropna().tolist()

    # 按分子划分，再对训练集做 SMILES 增强。
    indices = np.random.default_rng(cfg['seed']).permutation(len(smiles))
    split = int(len(smiles) * data['train_ratio'])
    train_smiles = [smiles[i] for i in indices[:split]]
    valid_smiles = [smiles[i] for i in indices[split:]]
    pretrained = cfg.get('pretrained_checkpoint', '')
    if pretrained:
        pretrained = resolve_checkpoint(pretrained)
        cfg['pretrained_checkpoint'] = str(pretrained)
        model = GPTGenerator.load_from_checkpoint(pretrained, map_location='cpu', lr=cfg['model']['lr'])
        tokenizer = SmilesTokenizer(model.vocab)
    else:
        vocab_path = data.get('vocab_path', '')
        tokenizer = (SmilesTokenizer(json.loads(Path(vocab_path).read_text(encoding='utf-8')))
                     if vocab_path else SmilesTokenizer.build(train_smiles))
        model = GPTGenerator(vocab=tokenizer.vocab, **cfg['model'])
    length = model.max_seq_len
    train_ds = SmilesDataset(train_smiles, tokenizer, length,
                             num_augmentations=data['num_augmentations'], seed=cfg['seed'])
    valid_ds = SmilesDataset(valid_smiles, tokenizer, length, randomize=False)
    loader_args = dict(batch_size=data['batch_size'], num_workers=data['num_workers'],
                       collate_fn=collate_batch)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_args)
    output = Path(cfg['trainer']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=output/'checkpoints',
                                 filename='best-{epoch:03d}-{val_loss:.4f}',
                                 monitor='val_loss', save_top_k=1, save_last=True,
                                 enable_version_counter=False)
    callbacks = [checkpoint]
    early_stopping = cfg.get('early_stopping', {})
    if early_stopping.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor=early_stopping.get('monitor', 'val_loss'),
            mode=early_stopping.get('mode', 'min'),
            patience=early_stopping.get('patience', 10),
            min_delta=early_stopping.get('min_delta', 0.001)))
    trainer_cfg = {k: v for k,v in cfg['trainer'].items() if k != 'output_dir'}
    devices = trainer_cfg.get('devices', 1)
    multiple_devices = (len(devices) > 1 if isinstance(devices, list)
                        else isinstance(devices, int) and (devices > 1 or devices == -1))
    strategy = trainer_cfg.get('strategy', 'auto')
    if strategy == 'ddp' or (strategy == 'auto' and multiple_devices):
        # 位置编码缓存随批次长度变化，不参与跨卡广播。
        trainer_cfg['strategy'] = DDPStrategy(broadcast_buffers=False)
    trainer = pl.Trainer(default_root_dir=str(output), callbacks=callbacks,
                         gradient_clip_val=1.0, **trainer_cfg)
    if trainer.is_global_zero:
        (output/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        if pretrained:
            print(f'Loaded pretrained generator: {pretrained}', flush=True)
    trainer.fit(model, train_loader, valid_loader)
    if trainer.is_global_zero and checkpoint.best_model_path:
        shutil.copy2(checkpoint.best_model_path, output/'checkpoints'/'best.ckpt')


if __name__ == '__main__':
    main()
