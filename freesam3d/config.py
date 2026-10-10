"""YAML config loading with ``a.b.c=value`` command-line overrides."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, _, raw = item.partition('=')
        if not _:
            raise ValueError(f'override must be key=value, got {item!r}')
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_path(path):
    """Relative paths in configs are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
