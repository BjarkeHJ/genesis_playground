import os
import types
import yaml
from typing import get_type_hints, get_origin, get_args, Union
from dataclasses import asdict, is_dataclass, fields

def dataclass_to_dict(cls):
    assert is_dataclass(cls) and not isinstance(cls, type), f"{type(cls).__name__} is not a dataclass instance"
    return asdict(cls)

def dict_to_dataclass(cls, data: dict):
    assert is_dataclass(cls) and isinstance(cls, type), f"{cls} is not a dataclass type"
    unknown = set(data) - {f.name for f in fields(cls)}
    assert not unknown, f"{cls.__name__}: unknown keys {unknown}"

    hints = get_type_hints(cls)
    kwargs = {
        f.name: _convert(hints[f.name], data[f.name])
        for f in fields(cls) if f.init and f.name in data # init=False are the ones derived from set params
    }
    return cls(**kwargs)

def _convert(tp, val):
    if val is None:
        return None
    if is_dataclass(tp):
        return dict_to_dataclass(tp, val)
    
    origin = get_origin(tp) # origin is a type hint 
    args = get_args(tp) # returns args eg if tp is a list[int] this return (int,), if not parameterized (eg. int) return ()

    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return _convert(non_none[0], val) if len(non_none) == 1 else val
    if origin is list and args:
        return [_convert(args[0], v) for v in val]
    if origin is tuple:
        return tuple(val)
    if origin is dict and args:
        return {k: _convert(args[1], v) for k, v in val.items()}

    return val

def save_yaml(data: dict, path, name):
    os.makedirs(path, exist_ok=True)
    f_path = os.path.join(path, name)
    with open(f_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_yaml(path) -> dict:
    assert os.path.isfile(path), f"{path} is not a file"
    with open(path) as f:
        return yaml.safe_load(f)

def resolve_ckpt(dir, run_name, ckpt: int=None):
    assert os.path.isdir(dir), f"{dir} is not a directory"
    
    run_dir = os.path.join(dir, run_name)
    assert os.path.isdir(run_dir), f"run name '{run_name}' is not a trained model name"

    if ckpt is not None:
        assert isinstance(ckpt, int), f"ckpt must be int type"
        model_ckpt = f"model_{ckpt}.pt"
        assert os.path.isfile(os.path.join(run_dir, model_ckpt)), f"ckpt '{ckpt}' is not a valid ckpt"
        return os.path.join(run_dir, model_ckpt)
    else:
        # fetch latest ckpt
        model_ckpts = []
        for f in os.listdir(run_dir):
            fs = str(f)
            if fs is not None and fs.startswith("model_"):
                model_ckpts.append(fs)
        if len(model_ckpts) == 0:
            return None
        latest_ckpt = max(model_ckpts, key = lambda s: int(s.split('_')[1].split('.')[0]))
        return os.path.join(dir, run_name, latest_ckpt)
    
# TEST (Completed)
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from config import EnvConfig
#     from config.paths import MODELS_DIR

#     p = resolve_ckpt(MODELS_DIR, "run_w_noise", 0)
#     print("max ckpt", p)


    # e = EnvConfig(num_actions=6)
    # e_dict = dataclass_to_dict(e)
    # save_yaml(e_dict, ".", "test.yaml")

    # e_dict_load = load_yaml("./test.yaml")
    # e_load = dict_to_dataclass(EnvConfig, e_dict_load)
    # print(e_load)