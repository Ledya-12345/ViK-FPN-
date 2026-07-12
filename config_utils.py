

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_config_to_args(args, cfg: dict):
   
    for key, value in cfg.items():
        setattr(args, key, value)
    return args
