import genesis as gs
import torch
from dataclasses import asdict, is_dataclass

# Helper converting dataclass to python dict - Used for converting TrainingConfig to dict as expected by RSL_RL
def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)

# Random bounded sampling
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


