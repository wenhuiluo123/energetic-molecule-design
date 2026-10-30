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
from predictor.graph.data import inverse_transform_tensor
from predictor.multimodal_to_smiles.data import EmbedToSmiDataConfig, EmbedToSmiDataModule
from predictor.multimodal_to_smiles.model import EmbeddingAlignmentConfig, MulTransformerEmbeddingStudent
from predictor.common.output import infer_path_tag, target_tag
from predictor.common.paths import normalize_path_string, resolve_portable_path, to_portable_path_str
from predictor.common.smiles_encoder import SmilesModelConfig
from predictor.common.tokenizer import SmilesTokenizer
from predictor.multimodal.model import MultimodalTransformerConfig, MultimodalTransformerRegressor, make_graph_model_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='predictor/multimodal_to_smiles/configs/r0_alignment.yaml')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_load_full(path: str | Path, map_location=None):
    try:
        return torch.load(normalize_path_string(str(path)), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(normalize_path_string(str(path)), map_location=map_location, weights_only=False)

def get_rng_state() -> dict:
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for (key, value) in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

def resolve_resume_checkpoint_path(raw_path: str | None, log_dir: Path, ckpt_dir: Path, *, auto_resume: bool) -> Path | None:
    if raw_path:
        resolved = resolve_portable_path(raw_path, base_dir=log_dir)
        candidate = resolved if resolved is not None else Path(normalize_path_string(raw_path))
        if not candidate.exists():
            raise FileNotFoundError(f'resume checkpoint not found: {candidate}')
        return candidate
    if auto_resume:
        candidate = ckpt_dir / 'last_checkpoint.pt'
        if candidate.exists():
            return candidate
    return None

def save_last_checkpoint(path: Path, *, student, optimizer: torch.optim.Optimizer, scheduler, cfg: dict, history: list[dict], stage: str, next_epoch: int, stage_done: bool, training_state: dict) -> None:
    torch.save({'model': student.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'config': cfg, 'history': history, 'stage': stage, 'next_epoch': next_epoch, 'stage_done': stage_done, 'training_state': training_state, 'rng_state': get_rng_state()}, path)

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for (k, v) in batch.items()}

def is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if mode.lower() == 'max':
        return current > best + min_delta
    return current < best - min_delta

def resolve_run_paths(run_dir: str | None, checkpoint: str | None, artifacts_dir: str | None) -> tuple[str | None, str | None]:
    ckpt = checkpoint or None
    artifacts = artifacts_dir or None
    run = run_dir or None
    if run:
        run_path = Path(normalize_path_string(run))
        summary_path = run_path / 'run_summary.json'
        common_candidates = [run_path / 'best_model.pt', run_path / 'best.ckpt', run_path / 'checkpoints' / 'best_model.pt', run_path / 'checkpoints' / 'best.ckpt']
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            if not ckpt:
                resolved = resolve_portable_path(summary.get('best_model_path'), base_dir=run_path, fallback_candidates=common_candidates)
                ckpt = str(resolved) if resolved is not None else None
        else:
            for candidate in common_candidates:
                if candidate.exists():
                    ckpt = ckpt or str(candidate)
                    break
        artifacts = artifacts or str(run_path)
    return (ckpt, artifacts)

def resolve_student_init(student_cfg: dict) -> tuple[str | None, str | None]:
    return resolve_run_paths(student_cfg.get('smiles_run_dir'), student_cfg.get('smiles_checkpoint'), student_cfg.get('smiles_artifacts_dir'))

def load_student_weights(student: MulTransformerEmbeddingStudent, checkpoint_path: str | None) -> tuple[list[str], list[str]]:
    if not checkpoint_path:
        return ([], [])
    state = torch.load(normalize_path_string(checkpoint_path), map_location='cpu', weights_only=False)
    raw_state = state.get('state_dict', state.get('model', state))
    student_state = student.state_dict()
    filtered = {}
    full_prefixes = ('backbone.', 'head.', 'student_align_proj.', 'teacher_align_proj.')
    for (key, value) in raw_state.items():
        stripped = key
        for prefix in ('model.', 'student.'):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        candidates = [stripped]
        if not stripped.startswith(full_prefixes):
            candidates.append(f'backbone.{stripped}')
        for candidate in candidates:
            if candidate in student_state and student_state[candidate].shape == value.shape:
                filtered[candidate] = value
                break
    load_result = student.load_state_dict(filtered, strict=False)
    return (list(load_result.missing_keys), list(load_result.unexpected_keys))

def normalize_student_model_cfg(raw_cfg: dict) -> dict:
    cfg = dict(raw_cfg)
    if 'nhead' in cfg and 'n_heads' not in cfg:
        cfg['n_heads'] = cfg.pop('nhead')
    if 'num_layers' in cfg and 'n_layers' not in cfg:
        cfg['n_layers'] = cfg.pop('num_layers')
    if 'dim_feedforward' in cfg and 'ffn_hidden' not in cfg:
        cfg['ffn_hidden'] = cfg.pop('dim_feedforward')
    allowed = set(SmilesModelConfig.__dataclass_fields__)
    return {key: value for (key, value) in cfg.items() if key in allowed}

def set_student_trainable(student, train_mode: str) -> list[str]:
    student_ref = student
    mode = str(train_mode or 'all').lower()
    if mode in {'all', 'full'}:
        for param in student_ref.parameters():
            param.requires_grad = True
    elif mode in {'encoder_proj', 'backbone_proj', 'embedding_proj'}:
        prefixes = ('backbone.', 'student_align_proj.', 'teacher_align_proj.')
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith(prefixes)
    elif mode in {'alignment_proj', 'proj', 'projection'}:
        prefixes = ('student_align_proj.', 'teacher_align_proj.')
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith(prefixes)
    elif mode in {'head', 'head_only', 'regression_head'}:
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith('head.')
    elif mode in {'backbone_head', 'encoder_head'}:
        prefixes = ('backbone.', 'head.')
        for (name, param) in student_ref.named_parameters():
            param.requires_grad = name.startswith(prefixes)
    else:
        raise ValueError(f'Unsupported train_mode={train_mode!r}; expected all, encoder_proj, alignment_proj, head_only, or backbone_head')
    return [name for (name, param) in student_ref.named_parameters() if param.requires_grad]

def build_teacher(cfg: dict):
    teacher_cfg = cfg['teacher']
    (teacher_checkpoint, teacher_artifacts_dir) = resolve_run_paths(teacher_cfg.get('multimodal_run_dir'), teacher_cfg.get('multimodal_checkpoint'), teacher_cfg.get('multimodal_artifacts_dir'))
    if not teacher_checkpoint or not teacher_artifacts_dir:
        raise ValueError('mul-transformer teacher checkpoint/artifacts are required')
    artifacts_path = Path(normalize_path_string(teacher_artifacts_dir))
    with open(artifacts_path / 'resolved_config.yaml', 'r', encoding='utf-8') as f:
        resolved_cfg = yaml.safe_load(f)
    with open(artifacts_path / 'stats.json', 'r', encoding='utf-8') as f:
        teacher_target_stats = json.load(f)
    with open(artifacts_path / 'data_schema.json', 'r', encoding='utf-8') as f:
        teacher_schema = json.load(f)
    teacher_tokenizer_path = artifacts_path / 'tokenizer.json'
    if not teacher_tokenizer_path.exists():
        raise ValueError(f'Teacher tokenizer not found: {teacher_tokenizer_path}')
    teacher_tokenizer = SmilesTokenizer.load_json(teacher_tokenizer_path)
    target_columns = teacher_schema['target_columns']
    teacher_max_seq_len = int(resolved_cfg['data'].get('max_seq_len', teacher_schema.get('max_seq_len', 256)))
    mm_cfg = MultimodalTransformerConfig(**resolved_cfg['model'])
    teacher = MultimodalTransformerRegressor(graph_cfg=make_graph_model_config(resolved_cfg['graph_model']), smiles_cfg=SmilesModelConfig(vocab_size=len(teacher_tokenizer.vocab), pad_token_id=teacher_tokenizer.pad_token_id, max_seq_len=teacher_max_seq_len, num_targets=len(target_columns), **resolved_cfg['smiles_model']), mm_cfg=mm_cfg, num_targets=len(target_columns), pretrained_graph_path=None, pretrained_smiles_checkpoint=None)
    state = torch.load(normalize_path_string(teacher_checkpoint), map_location='cpu', weights_only=False)
    teacher.load_state_dict(state['model'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return {'model': teacher, 'target_stats': teacher_target_stats, 'checkpoint': teacher_checkpoint, 'artifacts_dir': teacher_artifacts_dir, 'tokenizer_path': str(teacher_tokenizer_path), 'graph_encoder_type': mm_cfg.graph_encoder_type.lower(), 'max_seq_len': teacher_max_seq_len, 'target_columns': target_columns, 'embedding_dim': int(mm_cfg.model_dim)}

def teacher_embeddings_from_batch(teacher, batch: dict) -> torch.Tensor:
    encoded = teacher.backbone.encode_joint(batch['positions'], batch['charges'], batch['teacher_atom_mask'], batch['teacher_input_ids'], batch['teacher_attention_mask'])
    return encoded['cls']

class TeacherEmbeddingCache:

    def __init__(self, sample_to_index: dict[tuple[str, str], int], embeddings: torch.Tensor) -> None:
        self.sample_to_index = sample_to_index
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.sample_to_index)

    @property
    def memory_bytes(self) -> int:
        return self.embeddings.numel() * self.embeddings.element_size()

    def lookup(self, molecule_ids: list[str], smiles_list: list[str]) -> torch.Tensor:
        keys = [(str(molecule_id), str(smiles)) for (molecule_id, smiles) in zip(molecule_ids, smiles_list)]
        missing = [key for key in keys if key not in self.sample_to_index]
        if missing:
            preview = ', '.join((f'{molecule_id}:{smiles}' for (molecule_id, smiles) in missing[:3]))
            raise KeyError(f'Teacher embedding cache is missing samples: {preview}')
        indices = torch.tensor([self.sample_to_index[key] for key in keys], dtype=torch.long, device=self.embeddings.device)
        return self.embeddings.index_select(0, indices)

def build_teacher_embedding_cache(teacher, loader, device: torch.device) -> TeacherEmbeddingCache:
    teacher.eval()
    sample_to_index: dict[tuple[str, str], int] = {}
    embeddings: list[torch.Tensor] = []
    total_steps = len(loader)
    progress_every = max(total_steps // 20, 1)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            teacher_embeddings = teacher_embeddings_from_batch(teacher, batch)
            for (row_index, (molecule_id, smiles)) in enumerate(zip(batch['molecule_id'], batch['smiles'])):
                key = (str(molecule_id), str(smiles))
                if key in sample_to_index:
                    continue
                sample_to_index[key] = len(embeddings)
                embeddings.append(teacher_embeddings[row_index].detach().clone())
            if (batch_idx + 1) % progress_every == 0 or batch_idx + 1 == total_steps:
                print(f'[mul_transformer_embed_to_smiles] building teacher cache {batch_idx + 1}/{total_steps}', flush=True)
    if not embeddings:
        raise ValueError('Cannot build teacher embedding cache from an empty dataset')
    return TeacherEmbeddingCache(sample_to_index, torch.stack(embeddings, dim=0))

def get_teacher_embeddings(teacher, batch: dict, teacher_cache: TeacherEmbeddingCache | None) -> torch.Tensor:
    if teacher_cache is not None:
        return teacher_cache.lookup(batch['molecule_id'], batch['smiles'])
    return teacher_embeddings_from_batch(teacher, batch)

def evaluate_alignment(student, teacher, loader, device: torch.device, teacher_cache: TeacherEmbeddingCache | None=None) -> dict:
    student.eval()
    student_ref = student
    loss_sum = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            teacher_embeddings = get_teacher_embeddings(teacher, batch, teacher_cache)
            (loss, _) = student_ref.compute_alignment_loss({'input_ids': batch['student_input_ids'], 'attention_mask': batch['student_attention_mask']}, teacher_embeddings=teacher_embeddings, molecule_ids=batch.get('molecule_id'), collect_metrics=False)
            loss_sum += float(loss.item())
            batch_count += 1
    gathered = [{'loss_sum': loss_sum, 'batch_count': batch_count}]
    metrics = None
    total_loss = sum((float(item['loss_sum']) for item in gathered))
    total_batches = sum((int(item['batch_count']) for item in gathered))
    metrics = {'alignment_loss': float(total_loss / max(total_batches, 1))}
    return metrics

def evaluate_regression(student, loader, device: torch.device, loss_name: str) -> dict:
    student.eval()
    student_ref = student
    loss_sum = 0.0
    batch_count = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            (loss, _, preds_norm) = student_ref.compute_regression_loss({'input_ids': batch['student_input_ids'], 'attention_mask': batch['student_attention_mask'], 'labels': batch['labels']}, loss_name)
            loss_sum += float(loss.item())
            batch_count += 1
            preds_raw = inverse_transform_tensor(preds_norm, student_ref.target_columns, student_ref.target_stats)
            preds.append(preds_raw.cpu().numpy())
            labels.append(batch['labels_raw'].cpu().numpy())
    gathered = [{'loss_sum': loss_sum, 'batch_count': batch_count, 'preds': np.concatenate(preds, axis=0) if preds else np.empty((0, len(student_ref.target_columns)), dtype=np.float32), 'labels': np.concatenate(labels, axis=0) if labels else np.empty((0, len(student_ref.target_columns)), dtype=np.float32)}]
    metrics = None
    total_loss = sum((float(item['loss_sum']) for item in gathered))
    total_batches = sum((int(item['batch_count']) for item in gathered))
    y_pred = np.concatenate([item['preds'] for item in gathered], axis=0)
    y_true = np.concatenate([item['labels'] for item in gathered], axis=0)
    metrics = {'loss': float(total_loss / max(total_batches, 1)), 'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'mae': float(mean_absolute_error(y_true, y_pred)), 'r2': float(r2_score(y_true, y_pred))}
    return metrics

def set_backbone_freeze(student: MulTransformerEmbeddingStudent, freeze_backbone: bool) -> None:
    for param in student.backbone.parameters():
        param.requires_grad = not freeze_backbone
    for param in student.head.parameters():
        param.requires_grad = True
    for param in student.student_align_proj.parameters():
        param.requires_grad = True
    for param in student.teacher_align_proj.parameters():
        param.requires_grad = True

def main() -> None:
    args = parse_args()
    with open(normalize_path_string(args.config), 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    trainer_cfg = cfg['trainer']
    stage1_cfg = cfg['stage1']
    stage2_cfg = cfg['stage2']
    device = torch.device('cuda:' + str(trainer_cfg.get('devices', [0])[0]) if trainer_cfg.get('accelerator') != 'cpu' and torch.cuda.is_available() else 'cpu')
    print(f'[mul_transformer_embed_to_smiles] config loaded: {args.config}', flush=True)
    teacher_info = build_teacher(cfg)
    teacher = teacher_info['model']
    (student_checkpoint, student_artifacts_dir) = resolve_student_init(cfg.get('student_init', {}))
    student_tokenizer_path = str(Path(normalize_path_string(student_artifacts_dir)) / 'tokenizer.json') if student_artifacts_dir else None
    data_cfg = EmbedToSmiDataConfig(**cfg['data'])
    if data_cfg.target_columns != teacher_info['target_columns']:
        raise ValueError(f"Teacher target_columns={teacher_info['target_columns']} do not match data target_columns={data_cfg.target_columns}")
    dm = EmbedToSmiDataModule(data_cfg, student_tokenizer_path=student_tokenizer_path, teacher_tokenizer_path=teacher_info['tokenizer_path'], teacher_graph_encoder_type=teacher_info['graph_encoder_type'], teacher_max_seq_len=teacher_info['max_seq_len'])
    dm.setup()
    if student_artifacts_dir:
        with open(Path(normalize_path_string(student_artifacts_dir)) / 'resolved_config.yaml', 'r', encoding='utf-8') as f:
            student_resolved_cfg = yaml.safe_load(f)
        resolved_student_model_cfg = student_resolved_cfg.get('model', student_resolved_cfg.get('student_model', cfg['student_model']))
    else:
        resolved_student_model_cfg = cfg['student_model']
    resolved_student_model_cfg = normalize_student_model_cfg(resolved_student_model_cfg)
    resolved_student_model_cfg['pooling'] = 'cls'
    student_model_cfg = SmilesModelConfig(vocab_size=len(dm.student_tokenizer.vocab), pad_token_id=dm.student_tokenizer.pad_token_id, max_seq_len=data_cfg.max_seq_len, num_targets=len(data_cfg.target_columns), **resolved_student_model_cfg)
    student = MulTransformerEmbeddingStudent(cfg=student_model_cfg, target_columns=data_cfg.target_columns, target_stats=dm.target_stats, teacher_embedding_dim=teacher_info['embedding_dim'], align_cfg=EmbeddingAlignmentConfig(**cfg['alignment']), student_type=cfg.get('student_type', cfg.get('student', {}).get('type', 'gpt_regression')))
    (student_missing_keys, student_unexpected_keys) = load_student_weights(student, student_checkpoint)
    student.to(device)
    teacher.to(device)
    teacher.eval()
    teacher_cache = None
    cache_teacher_outputs = bool(trainer_cfg.get('cache_teacher_outputs', False))
    if cache_teacher_outputs:
        cache_start = time.time()
        print('[mul_transformer_embed_to_smiles] building frozen teacher embedding cache', flush=True)
        teacher_cache = build_teacher_embedding_cache(teacher, dm.teacher_cache_dataloader(), device)
        dm.disable_graph_loading_for_cached_teacher()
        teacher.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'[mul_transformer_embed_to_smiles] teacher cache ready | samples={len(teacher_cache)} | device_memory={teacher_cache.memory_bytes / 1024 ** 2:.2f} MiB | build_time={time.time() - cache_start:.2f}s | teacher_released_from_gpu=true', flush=True)
    stage1_train_loader = dm.stage1_train_dataloader()
    stage2_train_loader = dm.stage2_train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    dataset_name = trainer_cfg.get('dataset_name', 'default_dataset')
    run_name = trainer_cfg.get('run_name', 'mul_transformer_embed_to_smiles')
    property_tag = target_tag(data_cfg.target_columns)
    teacher_tag = infer_path_tag(teacher_info['artifacts_dir'] or teacher_info['checkpoint'], fallback='mul-transformer-teacher')
    student_tag = infer_path_tag(student_artifacts_dir or student_checkpoint, fallback='scratch')
    aug_tag = f'aug-{data_cfg.augmentation_num_randomizations}' if data_cfg.augmentation_enabled else 'aug-disabled'
    align_tag = f"align-{cfg['alignment'].get('loss_name', 'cosine')}"
    log_dir = Path(trainer_cfg['default_root_dir']) / run_name
    ckpt_dir = log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = ckpt_dir / 'last_checkpoint.pt'
    resolved_cfg_to_save = dict(cfg)
    resolved_cfg_to_save['student_model_resolved'] = resolved_student_model_cfg
    resolved_cfg_to_save['alignment'] = {**cfg['alignment'], 'teacher_embedding_dim': teacher_info['embedding_dim']}
    with open(log_dir / 'resolved_config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(resolved_cfg_to_save, f, sort_keys=False, allow_unicode=True)
    dm.export_artifacts(log_dir)
    history: list[dict] = []
    resume_checkpoint_path = resolve_resume_checkpoint_path(args.resume_checkpoint or trainer_cfg.get('resume_from_checkpoint', ''), log_dir, ckpt_dir, auto_resume=bool(trainer_cfg.get('auto_resume', False)))
    resume_state = None
    resume_stage = None
    resume_next_epoch = 0
    resume_stage_done = False
    resume_training_state: dict = {}
    if resume_checkpoint_path is not None:
        resume_state = torch_load_full(resume_checkpoint_path, map_location=device)
        student.load_state_dict(resume_state['model'])
        restore_rng_state(resume_state.get('rng_state'))
        history = list(resume_state.get('history', []))
        resume_stage = str(resume_state.get('stage', ''))
        resume_next_epoch = int(resume_state.get('next_epoch', 0))
        resume_stage_done = bool(resume_state.get('stage_done', False))
        resume_training_state = dict(resume_state.get('training_state', {}))
        print(f'[mul_transformer_embed_to_smiles] resumed from {resume_checkpoint_path} | stage={resume_stage} | next_epoch={resume_next_epoch} | stage_done={resume_stage_done}', flush=True)
    fit_start = time.time()
    log_every = max(int(trainer_cfg.get('log_every_n_steps', 10)), 1)
    stage1_best_path = None
    stage1_best_metric = None
    stage1_best_epoch = -1
    stage1_stopped_early = False
    stage1_trainable_names: list[str] = []
    stage1_monitor = str(stage1_cfg.get('monitor', 'val_alignment_loss'))
    stage1_monitor_mode = str(stage1_cfg.get('monitor_mode', 'min'))
    # 对齐多模态教师与 SMILES 学生的表征。
    if stage1_cfg.get('enabled', True) and int(stage1_cfg.get('max_epochs', 0)) > 0:
        stage1_train_mode = str(stage1_cfg.get('train_mode', 'alignment_proj' if bool(stage1_cfg.get('freeze_backbone', False)) else 'encoder_proj'))
        stage1_trainable_names = set_student_trainable(student, stage1_train_mode)
        stage1_params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(stage1_params, lr=float(stage1_cfg['lr']), weight_decay=float(stage1_cfg['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(stage1_cfg.get('scheduler_t_max', stage1_cfg['max_epochs'])))
        best_stage1 = -float('inf') if stage1_monitor_mode.lower() == 'max' else float('inf')
        stage1_best_path = ckpt_dir / 'stage1_best_model.pt'
        stage1_use_early_stopping = bool(stage1_cfg.get('use_early_stopping', trainer_cfg.get('use_early_stopping', True)))
        stage1_patience = int(stage1_cfg.get('early_stopping_patience', trainer_cfg.get('early_stopping_patience', 20)))
        stage1_min_delta = float(stage1_cfg.get('early_stopping_min_delta', trainer_cfg.get('early_stopping_min_delta', 0.0)))
        stage1_epochs_without_improve = 0
        stage1_state = dict(resume_training_state.get('stage1', {}))
        if stage1_state:
            best_stage1 = float(stage1_state.get('best_stage1', best_stage1))
            stage1_best_metric = stage1_state.get('stage1_best_metric', stage1_best_metric)
            stage1_best_epoch = int(stage1_state.get('stage1_best_epoch', stage1_best_epoch))
            stage1_stopped_early = bool(stage1_state.get('stage1_stopped_early', stage1_stopped_early))
            stage1_epochs_without_improve = int(stage1_state.get('stage1_epochs_without_improve', stage1_epochs_without_improve))
        stage1_start_epoch = 0
        if resume_stage == 'stage1':
            if resume_state.get('optimizer') is not None:
                optimizer.load_state_dict(resume_state['optimizer'])
                move_optimizer_state_to_device(optimizer, device)
            if resume_state.get('scheduler') is not None:
                scheduler.load_state_dict(resume_state['scheduler'])
            stage1_start_epoch = min(resume_next_epoch, int(stage1_cfg['max_epochs']))
            if resume_stage_done:
                stage1_start_epoch = int(stage1_cfg['max_epochs'])
        elif resume_stage == 'stage2':
            stage1_start_epoch = int(stage1_cfg['max_epochs'])
        for epoch in range(stage1_start_epoch, int(stage1_cfg['max_epochs'])):
            student.train()
            loss_sum = torch.zeros((), device=device)
            batch_count = 0
            total_train_steps = len(stage1_train_loader)
            print(f"[mul_transformer_embed_to_smiles][stage1][Epoch {epoch + 1}/{stage1_cfg['max_epochs']}] start | lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
            for (batch_idx, batch) in enumerate(stage1_train_loader):
                batch = move_batch_to_device(batch, device)
                with torch.no_grad():
                    teacher_embeddings = get_teacher_embeddings(teacher, batch, teacher_cache)
                optimizer.zero_grad(set_to_none=True)
                (loss, _) = student.compute_alignment_loss({'input_ids': batch['student_input_ids'], 'attention_mask': batch['student_attention_mask']}, teacher_embeddings=teacher_embeddings, molecule_ids=batch.get('molecule_id'), collect_metrics=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage1_params, trainer_cfg.get('gradient_clip_val', 1.0))
                optimizer.step()
                loss_sum += loss.detach()
                batch_count += 1
                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == total_train_steps:
                    current_loss = float(loss.detach().item())
                    avg_loss = float(loss_sum.item()) / max(batch_count, 1)
                    print(f"[mul_transformer_embed_to_smiles][stage1][Epoch {epoch + 1}/{stage1_cfg['max_epochs']}] step {batch_idx + 1}/{total_train_steps} | loss={current_loss:.6f} | avg_align={avg_loss:.6f}", flush=True)
            scheduler.step()
            train_align = float(loss_sum.item()) / max(batch_count, 1.0)
            val_align = evaluate_alignment(student, teacher, val_loader, device, teacher_cache)
            test_align = evaluate_alignment(student, teacher, test_loader, device, teacher_cache)
            stage1_record = {'stage': 'align', 'epoch': epoch, 'train_alignment_loss': train_align, 'val_alignment_loss': val_align['alignment_loss'], 'test_alignment_loss': test_align['alignment_loss'], 'lr': optimizer.param_groups[0]['lr']}
            history.append(stage1_record)
            if stage1_monitor not in stage1_record:
                raise ValueError(f'stage1.monitor={stage1_monitor!r} is not available. Available keys: {sorted(stage1_record.keys())}')
            stage1_current_metric = float(stage1_record[stage1_monitor])
            if is_improved(stage1_current_metric, best_stage1, stage1_monitor_mode, stage1_min_delta):
                best_stage1 = stage1_current_metric
                stage1_best_metric = best_stage1
                stage1_best_epoch = epoch
                stage1_epochs_without_improve = 0
                torch.save({'model': student.state_dict(), 'epoch': epoch, 'config': cfg, 'monitor': stage1_monitor, 'monitor_value': stage1_current_metric}, stage1_best_path)
            else:
                stage1_epochs_without_improve += 1
            stage1_will_stop = stage1_use_early_stopping and stage1_epochs_without_improve >= stage1_patience
            if stage1_will_stop:
                stage1_stopped_early = True
            save_last_checkpoint(last_checkpoint_path, student=student, optimizer=optimizer, scheduler=scheduler, cfg=cfg, history=history, stage='stage1', next_epoch=epoch + 1, stage_done=stage1_will_stop or epoch + 1 >= int(stage1_cfg['max_epochs']), training_state={'stage1': {'best_stage1': best_stage1, 'stage1_best_metric': stage1_best_metric, 'stage1_best_epoch': stage1_best_epoch, 'stage1_stopped_early': stage1_stopped_early, 'stage1_monitor': stage1_monitor, 'stage1_monitor_mode': stage1_monitor_mode, 'stage1_epochs_without_improve': stage1_epochs_without_improve}, 'stage2': dict(resume_training_state.get('stage2', {}))})
            print(f"[mul_transformer_embed_to_smiles][stage1][Epoch {epoch + 1}/{stage1_cfg['max_epochs']}] done | train_align={train_align:.6f} | val_align={val_align['alignment_loss']:.6f} | test_align={test_align['alignment_loss']:.6f} | {stage1_monitor}={stage1_current_metric:.6f} | best_{stage1_monitor}={best_stage1:.6f}", flush=True)
            if stage1_will_stop:
                print(f'[mul_transformer_embed_to_smiles][stage1] early stopping at epoch {epoch + 1} | best_epoch={stage1_best_epoch + 1} | best_{stage1_monitor}={best_stage1:.6f}', flush=True)
                break
        if resume_stage != 'stage2' and stage1_best_path.exists():
            state = torch.load(stage1_best_path, map_location=device, weights_only=False)
            student.load_state_dict(state['model'])
    stage2_enabled = bool(stage2_cfg.get('enabled', True)) and int(stage2_cfg.get('max_epochs', 0)) > 0
    stage2_train_mode = 'disabled'
    stage2_trainable_names: list[str] = []
    stage2_monitor = str(stage2_cfg.get('monitor', trainer_cfg.get('monitor', 'val_rmse')))
    stage2_monitor_mode = str(stage2_cfg.get('monitor_mode', trainer_cfg.get('monitor_mode', 'min')))
    best_metric = None
    best_epoch = -1
    best_model_path = ckpt_dir / 'best_model.pt'
    best_test_metrics = {}
    test_walltime_sec = 0.0
    stage2_epochs_without_improve = 0
    if stage2_enabled:
        stage2_train_mode = str(stage2_cfg.get('train_mode', 'head_only' if bool(stage2_cfg.get('freeze_backbone', False)) else 'all'))
        stage2_trainable_names = set_student_trainable(student, stage2_train_mode)
        student.to(device)
        stage2_params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(stage2_params, lr=float(stage2_cfg['lr']), weight_decay=float(stage2_cfg['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(stage2_cfg.get('scheduler_t_max', stage2_cfg['max_epochs'])))
        stage2_min_delta = float(stage2_cfg.get('early_stopping_min_delta', trainer_cfg.get('early_stopping_min_delta', 0.0)))
        best_metric = -float('inf') if stage2_monitor_mode.lower() == 'max' else float('inf')
        patience = trainer_cfg.get('early_stopping_patience', 20)
        epochs_without_improve = 0
        stage2_state = dict(resume_training_state.get('stage2', {}))
        if stage2_state:
            best_metric = float(stage2_state.get('best_metric', best_metric))
            best_epoch = int(stage2_state.get('best_epoch', best_epoch))
            epochs_without_improve = int(stage2_state.get('stage2_epochs_without_improve', epochs_without_improve))
            stage2_epochs_without_improve = epochs_without_improve
            best_test_metrics = dict(stage2_state.get('best_test_metrics', best_test_metrics))
        stage2_start_epoch = 0
        if resume_stage == 'stage2':
            if resume_state.get('optimizer') is not None:
                optimizer.load_state_dict(resume_state['optimizer'])
                move_optimizer_state_to_device(optimizer, device)
            if resume_state.get('scheduler') is not None:
                scheduler.load_state_dict(resume_state['scheduler'])
            stage2_start_epoch = min(resume_next_epoch, int(stage2_cfg['max_epochs']))
            if resume_stage_done:
                stage2_start_epoch = int(stage2_cfg['max_epochs'])
        for epoch in range(stage2_start_epoch, int(stage2_cfg['max_epochs'])):
            student.train()
            epoch_start = time.time()
            train_loss_sum = 0.0
            train_batch_count = 0
            total_train_steps = len(stage2_train_loader)
            print(f"[mul_transformer_embed_to_smiles][stage2][Epoch {epoch + 1}/{stage2_cfg['max_epochs']}] start | lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
            for (batch_idx, batch) in enumerate(stage2_train_loader):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                (loss, _, _) = student.compute_regression_loss({'input_ids': batch['student_input_ids'], 'attention_mask': batch['student_attention_mask'], 'labels': batch['labels']}, stage2_cfg['loss_name'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stage2_params, trainer_cfg.get('gradient_clip_val', 1.0))
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batch_count += 1
                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == total_train_steps:
                    avg_loss = train_loss_sum / max(train_batch_count, 1)
                    print(f"[mul_transformer_embed_to_smiles][stage2][Epoch {epoch + 1}/{stage2_cfg['max_epochs']}] step {batch_idx + 1}/{total_train_steps} | loss={loss.item():.6f} | avg_reg={avg_loss:.6f}", flush=True)
            scheduler.step()
            train_loss = train_loss_sum / max(train_batch_count, 1.0)
            val_metrics = evaluate_regression(student, val_loader, device, stage2_cfg['loss_name'])
            test_metrics = evaluate_regression(student, test_loader, device, stage2_cfg['loss_name'])
            stage2_record = {'stage': 'regress', 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'val_rmse': val_metrics['rmse'], 'val_mae': val_metrics['mae'], 'val_r2': val_metrics['r2'], 'test_loss': test_metrics['loss'], 'test_rmse': test_metrics['rmse'], 'test_mae': test_metrics['mae'], 'test_r2': test_metrics['r2'], 'lr': optimizer.param_groups[0]['lr']}
            history.append(stage2_record)
            if stage2_monitor not in stage2_record:
                raise ValueError(f'stage2.monitor/trainer.monitor={stage2_monitor!r} is not available. Available keys: {sorted(stage2_record.keys())}')
            stage2_current_metric = float(stage2_record[stage2_monitor])
            if is_improved(stage2_current_metric, best_metric, stage2_monitor_mode, stage2_min_delta):
                best_metric = stage2_current_metric
                best_epoch = epoch
                epochs_without_improve = 0
                best_test_metrics = test_metrics
                torch.save({'model': student.state_dict(), 'epoch': epoch, 'config': cfg, 'monitor': stage2_monitor, 'monitor_value': stage2_current_metric}, best_model_path)
            else:
                epochs_without_improve += 1
            stage2_epochs_without_improve = epochs_without_improve
            stage2_will_stop = trainer_cfg.get('use_early_stopping', True) and epochs_without_improve >= patience
            save_last_checkpoint(last_checkpoint_path, student=student, optimizer=optimizer, scheduler=scheduler, cfg=cfg, history=history, stage='stage2', next_epoch=epoch + 1, stage_done=stage2_will_stop or epoch + 1 >= int(stage2_cfg['max_epochs']), training_state={'stage1': {'best_stage1': stage1_best_metric if stage1_best_metric is not None else None, 'stage1_best_metric': stage1_best_metric, 'stage1_best_epoch': stage1_best_epoch, 'stage1_stopped_early': stage1_stopped_early, 'stage1_monitor': stage1_monitor, 'stage1_monitor_mode': stage1_monitor_mode}, 'stage2': {'best_metric': best_metric, 'best_epoch': best_epoch, 'stage2_epochs_without_improve': stage2_epochs_without_improve, 'stage2_monitor': stage2_monitor, 'stage2_monitor_mode': stage2_monitor_mode, 'best_test_metrics': best_test_metrics}})
            print(f"[mul_transformer_embed_to_smiles][stage2][Epoch {epoch + 1}/{stage2_cfg['max_epochs']}] done | train_loss={train_loss:.6f} | val_rmse={val_metrics['rmse']:.6f} | val_mae={val_metrics['mae']:.6f} | val_r2={val_metrics['r2']:.6f} | test_rmse={test_metrics['rmse']:.6f} | test_mae={test_metrics['mae']:.6f} | test_r2={test_metrics['r2']:.6f} | {stage2_monitor}={stage2_current_metric:.6f} | best_{stage2_monitor}={best_metric:.6f} | epoch_time={time.time() - epoch_start:.1f}s", flush=True)
            if stage2_will_stop:
                break
        if best_model_path.exists():
            best_state = torch.load(best_model_path, map_location=device, weights_only=False)
            student.load_state_dict(best_state['model'])
        else:
            print(f'[mul_transformer_embed_to_smiles] best model not found at {best_model_path}; keeping current checkpoint state.', flush=True)
        test_start = time.time()
        best_test_metrics = evaluate_regression(student, test_loader, device, stage2_cfg['loss_name'])
        test_walltime_sec = time.time() - test_start
    else:
        print('[mul_transformer_embed_to_smiles] stage2 disabled; keeping stage1 best checkpoint as final init output.', flush=True)
    fit_walltime_sec = time.time() - fit_start
    summary = {'dataset_name': dataset_name, 'run_name': run_name, 'target_columns': data_cfg.target_columns, 'target_tag': property_tag, 'normalization': data_cfg.normalization, 'fit_walltime_sec': fit_walltime_sec, 'test_walltime_sec': test_walltime_sec, 'teacher_checkpoint': teacher_info['checkpoint'], 'teacher_artifacts_dir': teacher_info['artifacts_dir'], 'teacher_graph_encoder_type': teacher_info['graph_encoder_type'], 'teacher_tag': teacher_tag, 'student_init_checkpoint': student_checkpoint, 'student_init_artifacts_dir': student_artifacts_dir, 'student_init_tag': student_tag, 'augmentation_enabled': data_cfg.augmentation_enabled, 'augmentation_num_randomizations': data_cfg.augmentation_num_randomizations, 'augmentation_stage1': data_cfg.augmentation_enabled if data_cfg.augmentation_stage1 is None else bool(data_cfg.augmentation_stage1), 'augmentation_stage2': data_cfg.augmentation_enabled if data_cfg.augmentation_stage2 is None else bool(data_cfg.augmentation_stage2), 'cache_teacher_outputs': cache_teacher_outputs, 'teacher_cache_samples': len(teacher_cache) if teacher_cache is not None else 0, 'teacher_cache_memory_bytes': teacher_cache.memory_bytes if teacher_cache is not None else 0, 'student_type': cfg.get('student_type', cfg.get('student', {}).get('type', 'gpt_regression')), 'student_pooling': 'mean', 'log_dir': to_portable_path_str(log_dir), 'best_model_path': to_portable_path_str(best_model_path, base_dir=log_dir), 'last_checkpoint_path': to_portable_path_str(last_checkpoint_path, base_dir=log_dir), 'resumed_from_checkpoint': to_portable_path_str(resume_checkpoint_path, base_dir=log_dir) if resume_checkpoint_path is not None else '', 'best_model_score': best_metric, 'best_epoch': best_epoch, 'stage2_enabled': stage2_enabled, 'stage2_monitor': stage2_monitor, 'stage2_monitor_mode': stage2_monitor_mode, 'monitor_uses_test_split': stage2_monitor.startswith('test_'), 'student_missing_keys': student_missing_keys, 'student_unexpected_keys': student_unexpected_keys, 'alignment': {**cfg['alignment'], 'teacher_embedding_dim': teacher_info['embedding_dim']}, 'stage1': cfg['stage1'], 'stage1_trainable_param_names': stage1_trainable_names, 'stage2': cfg['stage2'], 'stage2_trainable_param_names': stage2_trainable_names, 'test_metrics': best_test_metrics}
    if stage1_best_path is not None:
        summary['stage1_best_model_path'] = to_portable_path_str(stage1_best_path, base_dir=log_dir)
        summary['stage1_best_alignment_loss'] = stage1_best_metric
        summary['stage1_best_epoch'] = stage1_best_epoch
        summary['stage1_stopped_early'] = stage1_stopped_early
        summary['stage1_monitor'] = stage1_monitor
        summary['stage1_monitor_mode'] = stage1_monitor_mode
    with open(log_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    main()
