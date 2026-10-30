from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT_DIR = Path(__file__).resolve().parents[2]
from predictor.multimodal.data import MultimodalTransformerDataConfig, MultimodalTransformerDataModule
from predictor.graph.data import fit_target_stats, inverse_transform_tensor, normalize_targets
from predictor.multimodal.model import MultimodalTransformerConfig, MultimodalTransformerRegressor, load_pretrained_multimodal_transformer_backbone, make_graph_model_config
from predictor.common.paths import to_portable_path_str
from predictor.common.smiles_encoder import SmilesModelConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='predictor/multimodal/configs/r0_multimodal.yaml')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for (k, v) in batch.items()}

def is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if mode.lower() == 'max':
        return current > best + min_delta
    return current < best - min_delta

def evaluate(model, loader, device, target_columns, target_stats):
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels_norm = torch.tensor([normalize_targets(row.tolist(), target_columns, target_stats) for row in batch['labels_raw'].cpu()], dtype=torch.float32, device=device)
            preds_norm = model(batch['positions'], batch['charges'], batch['atom_mask'], batch['input_ids'], batch['attention_mask'])
            loss = model.compute_loss(preds_norm, labels_norm)
            loss_sum += float(loss.item())
            batch_count += 1
            preds_raw = inverse_transform_tensor(preds_norm, target_columns, target_stats)
            preds.append(preds_raw.cpu().numpy())
            labels.append(batch['labels_raw'].cpu().numpy())
    gathered = [{'loss_sum': loss_sum, 'batch_count': batch_count, 'preds': np.concatenate(preds, axis=0), 'labels': np.concatenate(labels, axis=0)}]
    metrics = None
    total_loss = sum((float(item['loss_sum']) for item in gathered))
    total_batches = sum((int(item['batch_count']) for item in gathered))
    y_pred = np.concatenate([item['preds'] for item in gathered], axis=0)
    y_true = np.concatenate([item['labels'] for item in gathered], axis=0)
    metrics = {'loss': float(total_loss / max(total_batches, 1)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'mae': float(mean_absolute_error(y_true, y_pred)), 'r2': float(r2_score(y_true, y_pred))}
    return metrics

def main() -> None:
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    trainer_cfg = cfg['trainer']
    device = torch.device('cuda:' + str(trainer_cfg.get('devices', [0])[0]) if trainer_cfg.get('accelerator') != 'cpu' and torch.cuda.is_available() else 'cpu')
    target_columns = cfg['data']['target_columns']
    data_cfg_dict = {k: v for (k, v) in cfg['data'].items() if k not in ('target_columns', 'normalization')}
    data_cfg_dict['graph_encoder_type'] = cfg['model'].get('graph_encoder_type', '3d')
    data_cfg = MultimodalTransformerDataConfig(**data_cfg_dict)
    tokenizer_path = cfg.get('tokenizer_path') or cfg.get('pretrained', {}).get('tokenizer_path')
    dm = MultimodalTransformerDataModule(data_cfg, target_columns=target_columns, tokenizer_path=tokenizer_path)
    dm.setup()
    target_stats = fit_target_stats(dm.train_df, target_columns, cfg['data'].get('normalization', 'zscore'))
    smiles_cfg = SmilesModelConfig(vocab_size=len(dm.tokenizer.vocab), pad_token_id=dm.tokenizer.pad_token_id, max_seq_len=data_cfg.max_seq_len, num_targets=len(target_columns), **cfg['smiles_model'])
    model = MultimodalTransformerRegressor(graph_cfg=make_graph_model_config(cfg['graph_model']), smiles_cfg=smiles_cfg, mm_cfg=MultimodalTransformerConfig(**cfg['model']), num_targets=len(target_columns), pretrained_graph_path=cfg.get('pretrained', {}).get('graph_checkpoint'), pretrained_smiles_checkpoint=cfg.get('pretrained', {}).get('smiles_checkpoint'))
    pretrained_backbone_checkpoint = cfg.get('pretrained', {}).get('backbone_checkpoint', '')
    if pretrained_backbone_checkpoint:
        (mm_missing_keys, mm_unexpected_keys) = load_pretrained_multimodal_transformer_backbone(model.backbone, pretrained_backbone_checkpoint)
    else:
        (mm_missing_keys, mm_unexpected_keys) = ([], [])
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['model'].get('lr', 0.0001), weight_decay=cfg['model'].get('weight_decay', 0.0001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(trainer_cfg.get('scheduler_t_max', trainer_cfg.get('max_epochs', 100))))
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    log_dir = Path(trainer_cfg['default_root_dir']) / trainer_cfg['run_name']
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'resolved_config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    dm.export_artifacts(log_dir)
    with open(log_dir / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(target_stats, f, ensure_ascii=False, indent=2)
    monitor_name = str(trainer_cfg.get('monitor', 'val_rmse'))
    monitor_mode = str(trainer_cfg.get('monitor_mode', 'min'))
    monitor_min_delta = float(trainer_cfg.get('early_stopping_min_delta', 0.0))
    best_metric = -float('inf') if monitor_mode.lower() == 'max' else float('inf')
    best_epoch = -1
    best_model_path = ckpt_dir / 'best_model.pt'
    last_model_path = ckpt_dir / 'last_model.pt'
    patience = int(trainer_cfg.get('early_stopping_patience', 20))
    epochs_without_improve = 0
    history = []
    fit_start = time.time()
    for epoch in range(int(trainer_cfg.get('max_epochs', 100))):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            labels_norm = torch.tensor([normalize_targets(row.tolist(), target_columns, target_stats) for row in batch['labels_raw'].cpu()], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            preds_norm = model(batch['positions'], batch['charges'], batch['atom_mask'], batch['input_ids'], batch['attention_mask'])
            loss = model.compute_loss(preds_norm, labels_norm)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            step_count += 1
        scheduler.step()
        train_loss = epoch_loss / max(step_count, 1)
        val_metrics = evaluate(model, val_loader, device, target_columns, target_stats)
        test_metrics = evaluate(model, test_loader, device, target_columns, target_stats)
        epoch_record = {'epoch': epoch + 1, 'train_loss': train_loss, **{f'val_{k}': float(v) for (k, v) in val_metrics.items()}, **{f'test_{k}': float(v) for (k, v) in test_metrics.items()}}
        history.append(epoch_record)
        if monitor_name not in epoch_record:
            raise ValueError(f'trainer.monitor={monitor_name!r} is not available. Available keys: {sorted(epoch_record.keys())}')
        current_metric = float(epoch_record[monitor_name])
        print(f"[mul-transformer][Epoch {epoch + 1}] train_loss={train_loss:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | test_rmse={test_metrics['rmse']:.4f} | test_r2={test_metrics['r2']:.4f} | {monitor_name}={current_metric:.4f}", flush=True)
        if is_improved(current_metric, best_metric, monitor_mode, monitor_min_delta):
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improve = 0
            torch.save({'model': model.state_dict(), 'epoch': best_epoch, 'monitor': monitor_name, 'monitor_value': current_metric}, best_model_path)
        else:
            epochs_without_improve += 1
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'config': cfg, 'monitor': monitor_name, 'monitor_value': current_metric}, last_model_path)
        should_stop = bool(True and trainer_cfg.get('use_early_stopping', True) and (epochs_without_improve >= patience))
        if should_stop:
            break
    fit_walltime = time.time() - fit_start
    final_metrics = evaluate(model, test_loader, device, target_columns, target_stats)
    with open(log_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(log_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'fit_walltime_sec': fit_walltime, 'log_dir': to_portable_path_str(log_dir, base_dir=ROOT_DIR), 'best_model_path': to_portable_path_str(best_model_path, base_dir=log_dir), 'last_model_path': to_portable_path_str(last_model_path, base_dir=log_dir), 'best_epoch': best_epoch, 'best_model_score': best_metric, 'monitor': monitor_name, 'monitor_mode': monitor_mode, 'monitor_uses_test_split': monitor_name.startswith('test_'), 'test_metrics': final_metrics, 'pretrained_backbone_missing_keys': mm_missing_keys, 'pretrained_backbone_unexpected_keys': mm_unexpected_keys}, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    main()
