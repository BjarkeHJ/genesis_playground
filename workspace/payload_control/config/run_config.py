"""Per-run files under logs/<run>: config snapshots, checkpoints and plot folders.

Config snapshots: train.py copies config/config_env.py / config/config_training.py into logs/<run>/config, and eval
scripts load those copies so a run is always evaluated with the configs it was trained with.
use_run_configs() must be called BEFORE anything imports config.config_env / config.config_training (env, plot_utils, ...):
it registers the snapshots under those module names, so every later `from config.config_env import ...` gets the run's copy."""
import argparse
import filecmp
import glob
import os
import re
import shutil
import sys
import time
import types
import importlib
from dataclasses import asdict, is_dataclass

from config.paths import PC_DIR, LOG_DIR

CONFIG_PKG = "config" # current configs live in payload_control/config/
CONFIG_SRC_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILES = ("config_env.py", "config_training.py")

def config_dir(run):
    return os.path.join(LOG_DIR, run, "config")

def plots_dir(log_dir, *sub):
    """<log_dir>/plots[/sub...], created if missing. Training-log plots go in plots/, evals of a checkpoint in plots/model_<iter>/."""
    d = os.path.join(log_dir, "plots", *sub)
    os.makedirs(d, exist_ok=True)
    return d

def snapshot_configs(log_dir):
    """Copy the current config files into <log_dir>/config. A differing older snapshot (resumed run) is kept as config_<time>."""
    dst = os.path.join(log_dir, "config")
    if os.path.isdir(dst):
        same = all(os.path.isfile(os.path.join(dst, f)) and filecmp.cmp(os.path.join(CONFIG_SRC_DIR, f), os.path.join(dst, f), shallow=False)
                   for f in CONFIG_FILES)
        if same:
            return dst
        old = f"{dst}_{time.strftime('%Y%m%d-%H%M%S')}"
        os.rename(dst, old)
        print(f"Configs changed since the last snapshot -- previous one moved to {old}")
    os.makedirs(dst)
    for f in CONFIG_FILES:
        shutil.copy2(os.path.join(CONFIG_SRC_DIR, f), dst)
    print(f"Saved configs to {dst}")
    return dst

def _load_as(name, path):
    # Register the snapshot as config.<name>. __file__ is payload_control/<name>.py: snapshots from before paths.py
    # derive SCRIPT_DIR / REPO_ROOT from it (now unused, see config/paths.py)
    mod = types.ModuleType(f"{CONFIG_PKG}.{name}")
    mod.__file__ = os.path.join(PC_DIR, f"{name}.py")
    with open(path) as fh:
        code = compile(fh.read(), path, "exec") # tracebacks point at the snapshot
    sys.modules[mod.__name__] = mod
    setattr(importlib.import_module(CONFIG_PKG), name, mod) # config.<name> attribute access, as for a regular import
    exec(code, mod.__dict__)
    return mod

def use_run_configs(run=None):
    """Load logs/<run>/config/*.py in place of the current configs. run=None takes --run from the command line.
    Falls back to the current configs (with a warning) for runs trained before snapshots existed."""
    if run is None:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--run", type=str, default=None)
        run = p.parse_known_args()[0].run
    if run is None:
        return False
    for f in CONFIG_FILES:
        name = f"{CONFIG_PKG}.{f[:-3]}"
        if name in sys.modules:
            raise RuntimeError(f"{name} was imported before use_run_configs() -- call it before importing env/utils")
    d = config_dir(run)

    if not all(os.path.isfile(os.path.join(d, f)) for f in CONFIG_FILES):
        print(f"WARNING: no config snapshot in {d} -- using the current config files")
        return False
    
    sys.path.insert(0, PC_DIR) # config_env imports system_model.* the same way as from payload_control/
    for f in CONFIG_FILES:
        _load_as(f[:-3], os.path.join(d, f))
    print(f"Using configs from {d}")
    return True

def resolve_ckpt(run_name, ckpt):
    """Checkpoint path: an explicit path wins, else model_<ckpt>.pt (or the latest one) from logs/<run_name>."""
    if ckpt is not None and os.path.isfile(ckpt):
        return ckpt
    log_dir = os.path.join(LOG_DIR, run_name)
    if ckpt is not None:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {log_dir}")
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))

def dataclass_to_dict(obj):
    """Config dataclass instance -> nested dict, as rsl_rl expects for the training config."""
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)
