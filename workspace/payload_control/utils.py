import torch
import glob
import re
import os

from dataclasses import asdict, is_dataclass
from config_env import SCRIPT_DIR

# Helper converting dataclass to python dict - Used for converting TrainingConfig to dict as expected by RSL_RL
def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)

# Random bounded sampling
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# Resolving a model (.pt file)
def resolve_ckpt(run_name, ckpt):
    # Explicit path wins; otherwise pick model_<ckpt>.pt (or the latest one) from logs/<run_name>
    if ckpt is not None and os.path.isfile(ckpt):
        return ckpt
    log_dir = os.path.join(SCRIPT_DIR, "logs", run_name)
    if ckpt is not None:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {log_dir}")
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))

