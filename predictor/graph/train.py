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
from predictor.graph.data import GraphV2DataConfig, GraphV2DataModule, inverse_transform_tensor
from predictor.graph.model import GraphV2ModelConfig, LitGraphV2Regressor
from predictor.common.output import infer_path_tag, target_tag
from predictor.common.paths import normalize_path_string, to_portable_path_str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='predictor/graph/configs/D_graph.yaml')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for (key, value) in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved

def is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if mode.lower() == 'max':
        return current > best + min_delta
    return current < best - min_delta

def compute_loss(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return model.compute_loss(batch)

def evaluate(model: LitGraphV2Regressor, loader, device: torch.device) -> dict:
    model.eval()
    model_ref = model
    loss_sum = 0.0
    batch_count = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            (loss, preds_norm) = compute_loss(model, batch)
            loss_sum += float(loss.item())
            batch_count += 1
            preds_raw = inverse_transform_tensor(preds_norm, model_ref.target_columns, model_ref.target_stats)
            preds.append(preds_raw.cpu().numpy())
            labels.append(batch['labels_raw'].cpu().numpy())
    gathered = [{'loss_sum': loss_sum, 'batch_count': batch_count, 'preds': np.concatenate(preds, axis=0) if preds else np.empty((0, len(model_ref.target_columns)), dtype=np.float32), 'labels': np.concatenate(labels, axis=0) if labels else np.empty((0, len(model_ref.target_columns)), dtype=np.float32)}]
    metrics = None
    total_loss = sum((float(item['loss_sum']) for item in gathered))
    total_batches = sum((int(item['batch_count']) for item in gathered))
    y_pred = np.concatenate([item['preds'] for item in gathered], axis=0)
    y_true = np.concatenate([item['labels'] for item in gathered], axis=0)
    metrics = {'loss': float(total_loss / max(total_batches, 1)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'mae': float(mean_absolute_error(y_true, y_pred)), 'r2': float(r2_score(y_true, y_pred))}
    return metrics

def main() -> None:
    args = parse_args()
    with open(normalize_path_string(args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    trainer_cfg = cfg['trainer']
    print(f'[graph_v2] config loaded: {args.config}', flush=True)
    device = torch.device('cuda:' + str(trainer_cfg.get('devices', [0])[0]) if trainer_cfg.get('accelerator') != 'cpu' and torch.cuda.is_available() else 'cpu')
    data_cfg = GraphV2DataConfig(**cfg['data'])
    print(f'[graph_v2] data setup start | metadata={data_cfg.metadata_path} | graph_cache={data_cfg.graph_cache_dir}', flush=True)
    dm = GraphV2DataModule(data_cfg)
    dm.setup()
    print(f'[graph_v2] data setup done | train={len(dm.train_ds)} | valid={len(dm.valid_ds)} | test={len(dm.test_ds)}', flush=True)
    model_cfg = GraphV2ModelConfig(**cfg['model'])
    pretrained_graph_path = cfg.get('pretrained', {}).get('graph_checkpoint')
    if pretrained_graph_path:
        pretrained_graph_path = normalize_path_string(pretrained_graph_path)
    print(f"[graph_v2] model build start | pretrained_graph_path={pretrained_graph_path or 'None'}", flush=True)
    model = LitGraphV2Regressor(model_cfg, data_cfg.target_columns, dm.target_stats, pretrained_graph_path)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(trainer_cfg.get('scheduler_t_max', trainer_cfg.get('max_epochs', 300))), eta_min=float(model_cfg.min_lr))
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    dataset_name = trainer_cfg.get('dataset_name', 'default_dataset')
    run_name = trainer_cfg.get('run_name', 'graph_v2_base')
    property_tag = target_tag(data_cfg.target_columns)
    pretrain_tag = infer_path_tag(pretrained_graph_path, fallback='scratch')
    log_dir = Path(trainer_cfg['default_root_dir']) / run_name
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'resolved_config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    dm.export_artifacts(log_dir)
    history = []
    monitor_name = str(trainer_cfg.get('monitor', 'val_rmse'))
    monitor_mode = str(trainer_cfg.get('monitor_mode', 'min'))
    monitor_min_delta = float(trainer_cfg.get('early_stopping_min_delta', 0.0))
    best_metric = -float('inf') if monitor_mode.lower() == 'max' else float('inf')
    best_epoch = -1
    best_model_path = ckpt_dir / 'best_model.pt'
    last_model_path = ckpt_dir / 'last_model.pt'
    patience = trainer_cfg.get('early_stopping_patience', 40)
    epochs_without_improve = 0
    log_every_n_steps = max(int(trainer_cfg.get('log_every_n_steps', 10)), 1)
    fit_start = time.time()
    for epoch in range(trainer_cfg.get('max_epochs', 300)):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_batch_count = 0
        total_train_steps = len(train_loader)
        print(f"[graph_v2][Epoch {epoch + 1}/{trainer_cfg.get('max_epochs', 300)}] start | lr={optimizer.param_groups[0]['lr']:.3e} | steps={total_train_steps}", flush=True)
        for (batch_idx, batch) in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            (loss, _) = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_cfg.get('gradient_clip_val', 1.0))
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batch_count += 1
            if (batch_idx + 1) % log_every_n_steps == 0 or batch_idx + 1 == total_train_steps:
                avg_loss = train_loss_sum / max(train_batch_count, 1)
                print(f"[graph_v2][Epoch {epoch + 1}/{trainer_cfg.get('max_epochs', 300)}] step {batch_idx + 1}/{total_train_steps} | loss={loss.item():.6f} | avg_loss={avg_loss:.6f}", flush=True)
        scheduler.step()
        train_loss = train_loss_sum / max(train_batch_count, 1.0)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'val_rmse': val_metrics['rmse'], 'val_mae': val_metrics['mae'], 'val_r2': val_metrics['r2'], 'test_loss': test_metrics['loss'], 'test_rmse': test_metrics['rmse'], 'test_mae': test_metrics['mae'], 'test_r2': test_metrics['r2'], 'lr': optimizer.param_groups[0]['lr']})
        epoch_record = history[-1]
        if monitor_name not in epoch_record:
            raise ValueError(f'trainer.monitor={monitor_name!r} is not available. Available keys: {sorted(epoch_record.keys())}')
        current_metric = float(epoch_record[monitor_name])
        if is_improved(current_metric, best_metric, monitor_mode, monitor_min_delta):
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'config': cfg, 'monitor': monitor_name, 'monitor_value': current_metric}, best_model_path)
        else:
            epochs_without_improve += 1
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'config': cfg, 'monitor': monitor_name, 'monitor_value': current_metric}, last_model_path)
        print(f"[graph_v2][Epoch {epoch + 1}/{trainer_cfg.get('max_epochs', 300)}] done | train_loss={train_loss:.6f} | val_rmse={val_metrics['rmse']:.6f} | val_mae={val_metrics['mae']:.6f} | val_r2={val_metrics['r2']:.6f} | test_rmse={test_metrics['rmse']:.6f} | test_mae={test_metrics['mae']:.6f} | test_r2={test_metrics['r2']:.6f} | {monitor_name}={current_metric:.6f} | best_{monitor_name}={best_metric:.6f} | epoch_time={time.time() - epoch_start:.1f}s", flush=True)
        if trainer_cfg.get('use_early_stopping', True) and epochs_without_improve >= patience:
            print(f'[graph_v2] early stopping at epoch {epoch + 1}, best_epoch={best_epoch + 1}, best_{monitor_name}={best_metric:.6f}', flush=True)
            break
    fit_walltime_sec = time.time() - fit_start
    load_best_for_final_test = bool(trainer_cfg.get('load_best_for_final_test', True))
    if load_best_for_final_test:
        best_state = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state['model'])
    final_checkpoint_role = 'best_model' if load_best_for_final_test else 'last_model'
    test_start = time.time()
    final_test_metrics = evaluate(model, test_loader, device)
    test_walltime_sec = time.time() - test_start
    print(f"[graph_v2] final evaluation | checkpoint={final_checkpoint_role} | test_rmse={final_test_metrics['rmse']:.6f} | test_mae={final_test_metrics['mae']:.6f} | test_r2={final_test_metrics['r2']:.6f}", flush=True)
    summary = {'dataset_name': dataset_name, 'run_name': run_name, 'target_columns': data_cfg.target_columns, 'target_tag': property_tag, 'normalization': data_cfg.normalization, 'fit_walltime_sec': fit_walltime_sec, 'test_walltime_sec': test_walltime_sec, 'pretrained_graph_path': pretrained_graph_path, 'pretrained_graph_tag': pretrain_tag, 'log_dir': to_portable_path_str(log_dir), 'best_model_path': to_portable_path_str(best_model_path, base_dir=log_dir), 'last_model_path': to_portable_path_str(last_model_path, base_dir=log_dir), 'best_model_score': best_metric, 'best_epoch': best_epoch, 'monitor': monitor_name, 'monitor_mode': monitor_mode, 'monitor_uses_test_split': monitor_name.startswith('test_'), 'final_eval_checkpoint': final_checkpoint_role, 'test_metrics': {'test_loss': final_test_metrics['loss'], 'test_rmse': final_test_metrics['rmse'], 'test_mae': final_test_metrics['mae'], 'test_r2': final_test_metrics['r2']}, 'encoder_missing_keys': model.missing_keys, 'encoder_unexpected_keys': model.unexpected_keys}
    with open(log_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    main()
