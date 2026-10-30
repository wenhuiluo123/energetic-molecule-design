import argparse
import json
from pathlib import Path
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from predictor.gpt.data import RLGPTRegDataConfig, RLGPTRegDataModule
from predictor.gpt.model import RLGPTRegressor, RLGPTRegressorConfig

def load_backbone(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = state.get('state_dict', state.get('model', state))
    weights = {key.removeprefix('backbone.'): value for (key, value) in weights.items()}
    backbone = {key: value for (key, value) in weights.items() if key.startswith(('embedding.', 'pos_encoder.', 'decoder.'))}
    model.load_state_dict(backbone, strict=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='predictor/gpt/configs/D_gpt.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    pl.seed_everything(cfg.get('seed', 42), workers=True)
    pretrained = cfg.get('pretrained', {})
    if pretrained.get('tokenizer_path'):
        cfg['data']['tokenizer_path'] = pretrained['tokenizer_path']
    data_cfg = RLGPTRegDataConfig(**cfg['data'])
    dm = RLGPTRegDataModule(data_cfg)
    dm.setup()
    model_cfg = RLGPTRegressorConfig(vocab_size=dm.vocab_size, max_seq_len=data_cfg.max_seq_len, pad_token_id=dm.pad_idx, num_targets=len(data_cfg.target_columns), **cfg['model'])
    model = RLGPTRegressor(model_cfg, data_cfg.target_columns, dm.target_stats)
    if pretrained.get('checkpoint'):
        load_backbone(model, pretrained['checkpoint'])
    if pretrained.get('freeze_backbone', False):
        for module in (model.embedding, model.pos_encoder, model.decoder):
            module.requires_grad_(False)
    trainer_cfg = cfg['trainer']
    log_dir = Path(trainer_cfg['default_root_dir']) / trainer_cfg['run_name']
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    dm.export_artifacts(log_dir)
    callbacks = [ModelCheckpoint(dirpath=log_dir / 'checkpoints', filename='best', monitor='val_rmse', mode='min', save_last=True, enable_version_counter=False)]
    if trainer_cfg.get('use_early_stopping', True):
        callbacks.append(EarlyStopping(monitor='val_rmse', mode='min', patience=trainer_cfg.get('early_stopping_patience', 20), min_delta=trainer_cfg.get('early_stopping_min_delta', 0.0)))
    trainer = pl.Trainer(max_epochs=trainer_cfg['max_epochs'], accelerator=trainer_cfg.get('accelerator', 'auto'), devices=trainer_cfg.get('devices', [0]), precision=trainer_cfg.get('precision', '32-true'), gradient_clip_val=trainer_cfg.get('gradient_clip_val', 1.0), deterministic=trainer_cfg.get('deterministic', True), callbacks=callbacks, logger=False, enable_progress_bar=trainer_cfg.get('enable_progress_bar', True), enable_model_summary=trainer_cfg.get('enable_model_summary', True))
    trainer.fit(model, datamodule=dm)
    outputs = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
    preds = torch.cat([item['preds'] for item in outputs]).numpy()
    labels = torch.cat([item['labels_raw'] for item in outputs]).numpy()
    summary = {'best_model_path': 'checkpoints/best.ckpt', 'test_metrics': {'mae': float(mean_absolute_error(labels, preds)), 'rmse': float(mean_squared_error(labels, preds) ** 0.5), 'r2': float(r2_score(labels, preds))}}
    (log_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
