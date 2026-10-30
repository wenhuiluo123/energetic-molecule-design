from pathlib import Path

def target_tag(columns, fallback='target'):
    return '__'.join(columns) if columns else fallback

def infer_path_tag(value, fallback='scratch'):
    return Path(value).stem if value else fallback

