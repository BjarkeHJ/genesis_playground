"""Per-run config snapshots: train.py copies config_env.py / config_training.py into logs/<run>/config, and eval
scripts load those copies so a run is always evaluated with the configs it was trained with.

use_run_configs() must be called BEFORE anything imports config_env / config_training (env, utils, plot_utils, ...):
it registers the snapshots under those module names, so every later `from config_env import ...` gets the run's copy."""
import argparse
import filecmp
import os
import shutil
import sys
import time
import types

PC_DIR = os.path.dirname(os.path.realpath(__file__)) # payload_control/
CONFIG_FILES = ("config_env.py", "config_training.py")

def config_dir(run):
    return os.path.join(PC_DIR, "logs", run, "config")

def plots_dir(log_dir, *sub):
    """<log_dir>/plots[/sub...], created if missing. Training-log plots go in plots/, evals of a checkpoint in plots/model_<iter>/."""
    d = os.path.join(log_dir, "plots", *sub)
    os.makedirs(d, exist_ok=True)
    return d

def snapshot_configs(log_dir):
    """Copy the current config files into <log_dir>/config. A differing older snapshot (resumed run) is kept as config_<time>."""
    dst = os.path.join(log_dir, "config")
    if os.path.isdir(dst):
        same = all(os.path.isfile(os.path.join(dst, f)) and filecmp.cmp(os.path.join(PC_DIR, f), os.path.join(dst, f), shallow=False)
                   for f in CONFIG_FILES)
        if same:
            return dst
        old = f"{dst}_{time.strftime('%Y%m%d-%H%M%S')}"
        os.rename(dst, old)
        print(f"Configs changed since the last snapshot -- previous one moved to {old}")
    os.makedirs(dst)
    for f in CONFIG_FILES:
        shutil.copy2(os.path.join(PC_DIR, f), dst)
    print(f"Saved configs to {dst}")
    return dst

def _load_as(name, path):
    # __file__ stays at payload_control/ so SCRIPT_DIR / REPO_ROOT in config_env resolve as in the original
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(PC_DIR, f"{name}.py")
    with open(path) as fh:
        code = compile(fh.read(), path, "exec") # tracebacks point at the snapshot
    sys.modules[name] = mod
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
    for name in ("config_env", "config_training"):
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
