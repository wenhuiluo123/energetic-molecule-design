from __future__ import annotations
import re
from pathlib import Path
_WINDOWS_DRIVE_RE = re.compile('^[A-Za-z]:[/\\\\]')

def to_portable_path_str(path: str | Path, base_dir: str | Path | None=None) -> str:
    path_obj = Path(path)
    if base_dir is not None:
        base_path = Path(base_dir)
        try:
            return path_obj.resolve().relative_to(base_path.resolve()).as_posix()
        except Exception:
            pass
    return path_obj.as_posix()

def normalize_path_string(path_str: str) -> str:
    return str(path_str).replace('\\', '/')

def is_windows_absolute_path(path_str: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(normalize_path_string(path_str)))

def resolve_portable_path(path_str: str | None, base_dir: str | Path | None=None, fallback_candidates: list[str | Path] | None=None) -> Path | None:
    if path_str:
        normalized = normalize_path_string(path_str)
        candidate = Path(normalized)
        candidates: list[Path] = []
        if not is_windows_absolute_path(normalized):
            candidates.append(candidate)
        if base_dir is not None:
            base_path = Path(base_dir)
            candidates.append(base_path / candidate)
        for resolved in candidates:
            if resolved.exists():
                return resolved
        if candidates:
            return candidates[-1]
    for item in fallback_candidates or []:
        candidate = Path(item)
        if candidate.exists():
            return candidate
    if fallback_candidates:
        return Path(fallback_candidates[0])
    return None
