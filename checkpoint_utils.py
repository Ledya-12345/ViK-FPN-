

import glob
import os
import re
from typing import Optional

import torch


def save_checkpoint(model: torch.nn.Module, path: str, extra: Optional[dict] = None) -> None:
   
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: str = "cpu",
                     strict: bool = False) -> torch.nn.Module:
    
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # If loading a Lightning checkpoint directly into a bare ViK_CD (i.e. the
    # keys are prefixed with "net."), strip the prefix.
    model_keys = set(model.state_dict().keys())
    if not any(k in model_keys for k in state_dict.keys()):
        stripped = {re.sub(r"^net\.", "", k): v for k, v in state_dict.items()}
        if any(k in model_keys for k in stripped.keys()):
            state_dict = stripped

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[load_checkpoint] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def find_best_checkpoint(checkpoint_dir: str, metric_name: str = "val_mIoU",
                          mode: str = "max") -> Optional[str]:
   

    pattern = os.path.join(checkpoint_dir, f"*{metric_name}=*.ckpt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _extract(path):
        m = re.search(rf"{metric_name}=([0-9.]+)", os.path.basename(path))
        return float(m.group(1).rstrip(".")) if m else float("-inf")

    key = _extract
    best = max(candidates, key=key) if mode == "max" else min(candidates, key=key)
    return best


def list_checkpoints(checkpoint_dir: str) -> list:
    
    files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    def _epoch(path):
        m = re.search(r"epoch=(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(files, key=_epoch)
