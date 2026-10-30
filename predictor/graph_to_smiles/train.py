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
from predictor.graph_to_smiles.data import DistillDataConfig, DistillDataModule
from predictor.graph.data import inverse_transform_tensor
from predictor.graph.model import GraphV2ModelConfig
from predictor.graph_to_smiles.model import DistillLossConfig, GraphV2Teacher, SmilesStudent, build_smiles_student, normalize_student_type
from predictor.common.paths import normalize_path_string, to_portable_path_str

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='predictor/graph_to_smiles/configs/D_graph_to_smiles.yaml')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for (k, v) in batch.items()}

def load_smiles_checkpoint(student: SmilesStudent, checkpoint_path: str) -> tuple[list[str], list[str]]:
    state = torch.load(normalize_path_string(checkpoint_path), map_location='cpu', weights_only=False)
    raw_state = state.get('model', state.get('state_dict', state)) if isinstance(state, dict) else state
    cleaned = {}
    for (key, value) in raw_state.items():
        if key.startswith('backbone.'):
            cleaned[key] = value
        elif key.startswith(('embedding.', 'pos_encoder.', 'decoder.')):
            cleaned[key] = value
        elif key.startswith('smiles_backbone.'):
            cleaned['backbone.' + key[len('smiles_backbone.'):]] = value
        elif key.startswith('student.backbone.'):
            cleaned[key[len('student.'):]] = value
        elif key.startswith('student.'):
            cleaned[key[len('student.'):]] = value
        elif key.startswith('model.'):
            cleaned[key[len('model.'):]] = value
    model_state = student.state_dict()
    filtered = {key: value for (key, value) in cleaned.items() if key in model_state and model_state[key].shape == value.shape}
    result = student.load_state_dict(filtered, strict=False)
    return (list(result.missing_keys), list(result.unexpected_keys))

def set_student_trainable(student, train_mode: str) -> list[str]:
    student_ref = student
    mode = train_mode.lower()
    if mode in {'all', 'full'}:
        for param in student_ref.parameters():
            param.requires_grad = True
    elif mode in {'encoder_proj', 'backbone_proj', 'embedding_proj'}:
        trainable_prefixes = ('backbone.', 'embedding.', 'pos_encoder.', 'decoder.', 'proj.')
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif mode in {'head', 'head_only', 'regression_head'}:
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith('head.')
    else:
        raise ValueError(f'Unsupported train_mode={train_mode!r}; expected all, encoder_proj, or head_only')
    return [name for (name, param) in student_ref.named_parameters() if param.requires_grad]

def build_optimizer_and_scheduler(student, optimizer_cfg: dict, student_cfg, stage_name: str, stage_epochs: int):
    stage_lr_key = f'{stage_name}_lr'
    stage_weight_decay_key = f'{stage_name}_weight_decay'
    lr = float(optimizer_cfg.get(stage_lr_key, optimizer_cfg.get('lr', student_cfg.lr)))
    weight_decay = float(optimizer_cfg.get(stage_weight_decay_key, optimizer_cfg.get('weight_decay', student_cfg.weight_decay)))
    trainable_params = [param for param in student.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(f'No trainable parameters for {stage_name}')
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler_t_max = int(optimizer_cfg.get(f'{stage_name}_scheduler_t_max', stage_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(scheduler_t_max, 1))
    return (optimizer, scheduler, {'lr': lr, 'weight_decay': weight_decay, 'scheduler_t_max': scheduler_t_max, 'trainable_param_tensors': len(trainable_params)})

def teacher_forward(teacher: GraphV2Teacher, batch: dict) -> dict[str, torch.Tensor]:
    return teacher(batch['positions'], batch['charges'], batch['atom_mask'], extra_node_features=batch.get('extra_node_features'), graph_features=batch.get('graph_features'))

class TeacherOutputCache:

    def __init__(self, molecule_to_index: dict[str, int], embeddings: torch.Tensor, predictions: torch.Tensor) -> None:
        self.molecule_to_index = molecule_to_index
        self.embeddings = embeddings
        self.predictions = predictions

    def __len__(self) -> int:
        return len(self.molecule_to_index)

    @property
    def memory_bytes(self) -> int:
        return self.embeddings.numel() * self.embeddings.element_size() + self.predictions.numel() * self.predictions.element_size()

    def lookup(self, molecule_ids: list[str]) -> dict[str, torch.Tensor]:
        missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in self.molecule_to_index]
        if missing:
            preview = ', '.join(missing[:5])
            raise KeyError(f'Teacher output cache is missing molecule IDs: {preview}')
        indices = torch.tensor([self.molecule_to_index[molecule_id] for molecule_id in molecule_ids], dtype=torch.long, device=self.embeddings.device)
        return {'embedding': self.embeddings.index_select(0, indices), 'preds': self.predictions.index_select(0, indices)}

def build_teacher_output_cache(teacher, loader, device: torch.device) -> TeacherOutputCache:
    teacher.eval()
    molecule_to_index: dict[str, int] = {}
    embeddings: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            teacher_out = teacher_forward(teacher, batch)
            for (row_index, molecule_id) in enumerate(batch['molecule_id']):
                if molecule_id in molecule_to_index:
                    continue
                molecule_to_index[molecule_id] = len(embeddings)
                embeddings.append(teacher_out['embedding'][row_index].detach().clone())
                predictions.append(teacher_out['preds'][row_index].detach().clone())
    if not embeddings:
        raise ValueError('Cannot build teacher output cache from an empty dataset')
    return TeacherOutputCache(molecule_to_index=molecule_to_index, embeddings=torch.stack(embeddings, dim=0), predictions=torch.stack(predictions, dim=0))

def get_teacher_outputs(teacher: GraphV2Teacher, batch: dict, teacher_cache: TeacherOutputCache | None) -> dict[str, torch.Tensor]:
    if teacher_cache is not None:
        return teacher_cache.lookup(batch['molecule_id'])
    return teacher_forward(teacher, batch)

def is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if mode.lower() == 'max':
        return current > best + min_delta
    return current < best - min_delta

def evaluate(student, loader, teacher, device, target_columns, target_stats, teacher_cache=None) -> dict:
    student.eval()
    student_ref = student
    loss_sum = 0.0
    batch_count = 0
    extra_loss_sums: dict[str, float] = {}
    preds = []
    labels = []
    teacher_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            student_out = student_ref(batch['input_ids'], batch['attention_mask'])
            teacher_out = get_teacher_outputs(teacher, batch, teacher_cache)
            loss = torch.nn.functional.mse_loss(student_out['preds'], batch['labels'])
            (stage1_loss, stage1_parts) = student_ref.stage1_loss(student_out, teacher_out, molecule_ids=batch.get('molecule_id'))
            (stage2_loss, stage2_parts) = student_ref.stage2_loss(student_out, teacher_out, batch['labels'])
            loss_sum += float(loss.item())
            batch_count += 1
            extra_loss_sums['stage1_loss'] = extra_loss_sums.get('stage1_loss', 0.0) + float(stage1_loss.item())
            extra_loss_sums['stage2_loss'] = extra_loss_sums.get('stage2_loss', 0.0) + float(stage2_loss.item())
            for (key, value) in stage1_parts.items():
                extra_loss_sums[f'stage1_{key}'] = extra_loss_sums.get(f'stage1_{key}', 0.0) + float(value)
            for (key, value) in stage2_parts.items():
                extra_loss_sums[f'stage2_{key}'] = extra_loss_sums.get(f'stage2_{key}', 0.0) + float(value)
            preds.append(inverse_transform_tensor(student_out['preds'], target_columns, target_stats).cpu().numpy())
            teacher_preds.append(inverse_transform_tensor(teacher_out['preds'], target_columns, target_stats).cpu().numpy())
            labels.append(batch['labels_raw'].cpu().numpy())
    gathered = [{'loss_sum': loss_sum, 'batch_count': batch_count, 'extra_loss_sums': extra_loss_sums, 'preds': np.concatenate(preds, axis=0) if preds else np.empty((0, len(target_columns)), dtype=np.float32), 'teacher_preds': np.concatenate(teacher_preds, axis=0) if teacher_preds else np.empty((0, len(target_columns)), dtype=np.float32), 'labels': np.concatenate(labels, axis=0) if labels else np.empty((0, len(target_columns)), dtype=np.float32)}]
    metrics = None
    total_loss = sum((float(item['loss_sum']) for item in gathered))
    total_batches = sum((int(item['batch_count']) for item in gathered))
    y_pred = np.concatenate([item['preds'] for item in gathered], axis=0)
    y_teacher = np.concatenate([item['teacher_preds'] for item in gathered], axis=0)
    y_true = np.concatenate([item['labels'] for item in gathered], axis=0)
    extra_totals: dict[str, float] = {}
    for item in gathered:
        for (key, value) in item['extra_loss_sums'].items():
            extra_totals[key] = extra_totals.get(key, 0.0) + float(value)
    metrics = {'loss': float(total_loss / max(total_batches, 1)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'mae': float(mean_absolute_error(y_true, y_pred)), 'r2': float(r2_score(y_true, y_pred)), 'teacher_rmse': float(np.sqrt(mean_squared_error(y_true, y_teacher))), 'teacher_r2': float(r2_score(y_true, y_teacher))}
    for (key, value) in extra_totals.items():
        metrics[key] = float(value / max(total_batches, 1))
    return metrics

def train_one_stage(*, stage_name: str, stage_epochs: int, student, teacher, teacher_cache, train_loader, val_loader, test_loader, optimizer, scheduler, device, target_columns, target_stats, history: list[dict], best: dict, ckpt_dir: Path, early_stopping_patience: int=0, monitor: str='val_loss', monitor_mode: str='min', min_delta: float=0.0) -> None:
    student_ref = student
    stage_best = float('inf') if monitor_mode.lower() != 'max' else -float('inf')
    epochs_without_improve = 0
    for epoch in range(stage_epochs):
        student.train()
        loss_sum = 0.0
        step_count = 0
        loss_parts_sum: dict[str, float] = {}
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                teacher_out = get_teacher_outputs(teacher, batch, teacher_cache)
            student_out = student(batch['input_ids'], batch['attention_mask'])
            if stage_name == 'contrastive':
                (loss, loss_parts) = student_ref.stage1_loss(student_out, teacher_out, molecule_ids=batch.get('molecule_id'))
            else:
                (loss, loss_parts) = student_ref.stage2_loss(student_out, teacher_out, batch['labels'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            step_count += 1
            for (key, value) in loss_parts.items():
                loss_parts_sum[key] = loss_parts_sum.get(key, 0.0) + float(value)
        scheduler.step()
        train_loss = loss_sum / max(step_count, 1)
        val_metrics = evaluate(student, val_loader, teacher, device, target_columns, target_stats, teacher_cache)
        test_metrics = evaluate(student, test_loader, teacher, device, target_columns, target_stats, teacher_cache)
        record = {'stage': stage_name, 'epoch': epoch + 1, 'train_loss': train_loss, **{f'train_{key}': value / max(step_count, 1) for (key, value) in loss_parts_sum.items()}, **{f'val_{key}': float(value) for (key, value) in val_metrics.items()}, **{f'test_{key}': float(value) for (key, value) in test_metrics.items()}}
        if monitor not in record:
            raise ValueError(f"Early stopping monitor '{monitor}' is not available. Available keys: {sorted(record.keys())}")
        current_monitor = float(record.get(monitor, float('nan')))
        history.append(record)
        print(f"[graph-v2->smiles:{stage_name}][Epoch {epoch + 1}/{stage_epochs}] train_loss={train_loss:.4f} | val_rmse={val_metrics['rmse']:.4f} | val_r2={val_metrics['r2']:.4f} | test_rmse={test_metrics['rmse']:.4f} | test_r2={test_metrics['r2']:.4f} | val_teacher_r2={val_metrics['teacher_r2']:.4f} | test_teacher_r2={test_metrics['teacher_r2']:.4f} | {monitor}={current_monitor:.4f}", flush=True)
        improved = is_improved(current_monitor, stage_best, monitor_mode, min_delta)
        if improved:
            stage_best = current_monitor
            epochs_without_improve = 0
            stage_payload = {'model': student_ref.state_dict(), 'stage': stage_name, 'epoch': epoch + 1, 'monitor': monitor, 'monitor_value': current_monitor}
            torch.save(stage_payload, ckpt_dir / f'{stage_name}_best.pt')
            if stage_name == 'contrastive':
                torch.save(stage_payload, ckpt_dir / 'stage1_best.pt')
            elif stage_name == 'regression':
                torch.save(stage_payload, ckpt_dir / 'stage2_best.pt')
        else:
            epochs_without_improve += 1
        if stage_name == 'contrastive':
            torch.save({'model': student_ref.state_dict(), 'stage': stage_name, 'epoch': epoch + 1}, ckpt_dir / 'stage1_last.pt')
        elif stage_name == 'regression':
            torch.save({'model': student_ref.state_dict(), 'stage': stage_name, 'epoch': epoch + 1}, ckpt_dir / 'stage2_last.pt')
            if improved:
                best['score'] = current_monitor
                best['stage'] = stage_name
                best['epoch'] = epoch + 1
                best['monitor'] = monitor
                torch.save({'model': student_ref.state_dict(), 'stage': stage_name, 'epoch': epoch + 1}, ckpt_dir / 'best_model.pt')
        should_stop = bool(True and early_stopping_patience > 0 and (epochs_without_improve >= early_stopping_patience))
        if should_stop:
            print(f'[graph-v2->smiles:{stage_name}] early stopping at epoch {epoch + 1} | best_{monitor}={stage_best:.4f}', flush=True)
            break

def main() -> None:
    args = parse_args()
    with open(normalize_path_string(args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda:' + str(cfg['trainer'].get('devices', [0])[0]) if cfg['trainer'].get('accelerator') != 'cpu' and torch.cuda.is_available() else 'cpu')
    data_cfg = DistillDataConfig(**cfg['data'])
    dm = DistillDataModule(data_cfg, tokenizer_path=cfg.get('student_pretrained', {}).get('tokenizer_path') or None)
    dm.setup()
    target_columns = data_cfg.target_columns or []
    graph_feature_dim = len(dm.cfg.graph_feature_columns or [])
    cfg['data']['graph_feature_columns'] = dm.cfg.graph_feature_columns or []
    cfg['teacher_model']['graph_feature_dim'] = graph_feature_dim
    teacher_checkpoint = cfg['teacher'].get('checkpoint', '')
    if not teacher_checkpoint:
        raise ValueError('teacher.checkpoint must point to a trained graph_v2 checkpoint')
    teacher_cfg = GraphV2ModelConfig(**cfg['teacher_model'])
    teacher = GraphV2Teacher(cfg=teacher_cfg, target_columns=target_columns, target_stats=dm.target_stats, checkpoint_path=normalize_path_string(teacher_checkpoint))
    loss_cfg = DistillLossConfig(**cfg['distill'])
    student_type = normalize_student_type(cfg.get('student_type', cfg.get('student', {}).get('type', 'rl_gpt')))
    cfg['student_type'] = student_type
    student = build_smiles_student(student_type, cfg['student_model'], vocab_size=len(dm.tokenizer.vocab), pad_token_id=dm.tokenizer.pad_token_id, max_seq_len=data_cfg.max_seq_len, num_targets=len(target_columns), teacher_dim=teacher_cfg.hidden_dim, loss_cfg=loss_cfg)
    student_cfg = student.cfg
    (smiles_missing_keys, smiles_unexpected_keys) = ([], [])
    smiles_checkpoint = cfg.get('student_pretrained', {}).get('smiles_checkpoint', '')
    if smiles_checkpoint:
        (smiles_missing_keys, smiles_unexpected_keys) = load_smiles_checkpoint(student, smiles_checkpoint)
    teacher.to(device)
    student.to(device)
    teacher_cache = None
    cache_teacher_outputs = bool(cfg['trainer'].get('cache_teacher_outputs', False))
    if cache_teacher_outputs:
        cache_start = time.time()
        teacher_cache = build_teacher_output_cache(teacher, dm.teacher_cache_dataloader(), device)
        dm.disable_graph_loading_for_cached_teacher()
        print(f'[graph-v2->smiles] teacher cache | molecules={len(teacher_cache)} | device_memory={teacher_cache.memory_bytes / 1024 ** 2:.2f} MiB | build_time={time.time() - cache_start:.2f}s', flush=True)
    stage1_train_loader = dm.stage1_train_dataloader()
    stage2_train_loader = dm.stage2_train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    log_dir = Path(cfg['trainer']['default_root_dir']) / cfg['trainer']['run_name']
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'resolved_config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    dm.export_artifacts(log_dir)
    with open(log_dir / 'teacher_info.json', 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': teacher_checkpoint}, f, ensure_ascii=False, indent=2)
    history: list[dict] = []
    best = {'score': float('inf'), 'stage': '', 'epoch': -1}
    optimizer_cfg = cfg.get('optimizer', {})
    stage1_epochs = int(cfg['trainer'].get('stage1_epochs', 50))
    stage2_epochs = int(cfg['trainer'].get('stage2_epochs', 150))
    stage1_trainable_names = set_student_trainable(student, str(optimizer_cfg.get('stage1_train_mode', 'all')))
    (stage1_optimizer, stage1_scheduler, stage1_optimizer_info) = build_optimizer_and_scheduler(student, optimizer_cfg, student_cfg, stage_name='stage1', stage_epochs=stage1_epochs)
    print(f"[graph-v2->smiles] stage1 optimizer | lr={stage1_optimizer_info['lr']:.3g} | weight_decay={stage1_optimizer_info['weight_decay']:.3g} | trainable_tensors={stage1_optimizer_info['trainable_param_tensors']}", flush=True)
    start = time.time()
    # 先对齐表征，再训练回归输出。
    train_one_stage(stage_name='contrastive', stage_epochs=stage1_epochs, student=student, teacher=teacher, teacher_cache=teacher_cache, train_loader=stage1_train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=stage1_optimizer, scheduler=stage1_scheduler, device=device, target_columns=target_columns, target_stats=dm.target_stats, history=history, best=best, ckpt_dir=ckpt_dir, early_stopping_patience=int(cfg['trainer'].get('stage1_early_stopping_patience', 10)), monitor=str(cfg['trainer'].get('stage1_monitor', 'val_stage1_loss')), monitor_mode=str(cfg['trainer'].get('stage1_monitor_mode', 'min')), min_delta=float(cfg['trainer'].get('stage1_min_delta', cfg['trainer'].get('early_stopping_min_delta', 0.0))))
    if bool(cfg['trainer'].get('load_stage1_best_for_stage2', True)):
        stage1_best_path = ckpt_dir / 'stage1_best.pt'
        if stage1_best_path.exists():
            state = torch.load(stage1_best_path, map_location=device, weights_only=False)
            student.load_state_dict(state['model'], strict=False)
            print(f'[graph-v2->smiles] loaded stage1 best checkpoint for stage2: {stage1_best_path}', flush=True)
        else:
            print(f'[graph-v2->smiles] stage1 best checkpoint not found, using current student weights: {stage1_best_path}', flush=True)
    stage2_enabled = bool(cfg['trainer'].get('stage2_enabled', cfg.get('stage2', {}).get('enabled', True))) and stage2_epochs > 0
    stage2_train_mode = 'disabled'
    stage2_trainable_names: list[str] = []
    stage2_optimizer_info = {}
    best_model_path = ckpt_dir / 'best_model.pt'
    test_metrics_source = 'not_evaluated'
    test_metrics = {}
    if stage2_enabled:
        stage2_train_mode = 'head_only' if bool(optimizer_cfg.get('stage2_train_head_only', False)) else str(optimizer_cfg.get('stage2_train_mode', 'all'))
        stage2_trainable_names = set_student_trainable(student, stage2_train_mode)
        (stage2_optimizer, stage2_scheduler, stage2_optimizer_info) = build_optimizer_and_scheduler(student, optimizer_cfg, student_cfg, stage_name='stage2', stage_epochs=stage2_epochs)
        preview = ', '.join(stage2_trainable_names[:8])
        if len(stage2_trainable_names) > 8:
            preview = f'{preview}, ...'
        print(f"[graph-v2->smiles] stage2 optimizer | mode={stage2_train_mode} | lr={stage2_optimizer_info['lr']:.3g} | weight_decay={stage2_optimizer_info['weight_decay']:.3g} | trainable_tensors={stage2_optimizer_info['trainable_param_tensors']} | trainable={preview}", flush=True)
        train_one_stage(stage_name='regression', stage_epochs=stage2_epochs, student=student, teacher=teacher, teacher_cache=teacher_cache, train_loader=stage2_train_loader, val_loader=val_loader, test_loader=test_loader, optimizer=stage2_optimizer, scheduler=stage2_scheduler, device=device, target_columns=target_columns, target_stats=dm.target_stats, history=history, best=best, ckpt_dir=ckpt_dir, early_stopping_patience=int(cfg['trainer'].get('stage2_early_stopping_patience', 20)), monitor=str(cfg['trainer'].get('stage2_monitor', 'val_rmse')), monitor_mode=str(cfg['trainer'].get('stage2_monitor_mode', 'min')), min_delta=float(cfg['trainer'].get('stage2_min_delta', cfg['trainer'].get('early_stopping_min_delta', 0.0))))
        test_metrics_source = 'last_epoch'
    else:
        best_model_path = ckpt_dir / 'stage1_best.pt'
        print('[graph-v2->smiles] stage2 disabled; keeping stage1 best checkpoint as final init output.', flush=True)
    if stage2_enabled:
        load_best_for_final_test = bool(cfg['trainer'].get('load_best_for_final_test', True))
        if load_best_for_final_test and best_model_path.exists():
            best_state = torch.load(best_model_path, map_location=device, weights_only=False)
            student.load_state_dict(best_state['model'], strict=False)
            test_metrics_source = 'best_model'
            print(f'[graph-v2->smiles] loaded best regression checkpoint for final test: {best_model_path}', flush=True)
        else:
            test_metrics_source = 'last_epoch'
        test_metrics = evaluate(student, test_loader, teacher, device, target_columns, dm.target_stats, teacher_cache)
    with open(log_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(log_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'fit_walltime_sec': time.time() - start, 'log_dir': to_portable_path_str(log_dir, base_dir=ROOT_DIR), 'best_model_path': to_portable_path_str(best_model_path, base_dir=log_dir), 'stage1_last_model_path': to_portable_path_str(ckpt_dir / 'stage1_last.pt', base_dir=log_dir), 'stage2_last_model_path': to_portable_path_str(ckpt_dir / 'stage2_last.pt', base_dir=log_dir) if stage2_enabled else None, 'best': best, 'stage1_monitor': str(cfg['trainer'].get('stage1_monitor', 'val_stage1_loss')), 'stage2_monitor': str(cfg['trainer'].get('stage2_monitor', 'val_rmse')), 'stage2_enabled': stage2_enabled, 'teacher_output_cache_enabled': teacher_cache is not None, 'teacher_output_cache_size': len(teacher_cache) if teacher_cache is not None else 0, 'stage1_optimizer': stage1_optimizer_info, 'stage2_optimizer': stage2_optimizer_info, 'stage1_trainable_param_names': stage1_trainable_names, 'stage2_train_mode': stage2_train_mode, 'stage2_trainable_param_names': stage2_trainable_names, 'monitor_uses_test_split': str(cfg['trainer'].get('stage2_monitor', 'val_rmse')).startswith('test_'), 'test_metrics': test_metrics, 'test_metrics_source': test_metrics_source, 'smiles_pretrained_missing_keys': smiles_missing_keys, 'smiles_pretrained_unexpected_keys': smiles_unexpected_keys}, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    main()
