import os
from typing import Any

def _load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if ":" in line and not line.startswith("-"):
                    k, _, v = line.partition(":")
                    v = v.strip()
                    if v: cfg[k.strip()] = v
        return cfg

def load_config(path="configs/default.yaml"):
    raw = _load_yaml(path)
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict): _flatten(v, key)
            else: flat[key] = v
    _flatten(raw)
    return flat

def get(cfg, key, default=None):
    return cfg.get(key, default)
