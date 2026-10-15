"""Shared tools for the training-log figures in eval/: config access, TensorBoard loading, smoothing, house style,
CLI parsing and saving. Configs are read from config_env so every figure stays in sync with env.py."""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (re-exported: scripts take plt from here so the Agg backend is set first)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

PC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # payload_control/
LOG_DIR = os.path.join(PC_DIR, "logs")
sys.path.insert(0, PC_DIR)
from run_config import use_run_configs, plots_dir  # noqa: E402
use_run_configs()  # logs/<--run>/config when given, else the current configs
from config_env import EnvConfig  # noqa: E402

cfg = EnvConfig()
rw = cfg.rew_cfg
dr = cfg.dr_cfg
STEPS_PER_S = 1.0 / cfg.dt

# ====== STYLE ======
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
LW = 2.0
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold", "axes.titlesize": 14,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.fontsize": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
}

def apply_style(**overrides):
    plt.rcParams.update({**STYLE, **overrides})

def legend_outside(ax, fontsize=None, ncol=1, anchor=(1.02, 1.0)):
    # Right of the axes: training curves fill the whole plot area at some point
    ax.legend(loc="upper left", bbox_to_anchor=anchor, ncol=ncol, fontsize=fontsize, borderaxespad=0.0)

# ====== LOADING ======
def run_dir(run):
    return os.path.join(LOG_DIR, run)

def load_scalars(run):
    """{tag: (steps, values)} for every scalar in the run's TensorBoard event files."""
    files = sorted(glob.glob(os.path.join(run_dir(run), "events.*")))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files in {run_dir(run)}")
    data, first = {}, None
    for f in files:  # later files win on overlapping steps (resumed runs)
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        start = min((ea.Scalars(t)[0].step for t in tags), default=None)
        if start is None:
            continue
        if first is not None and start <= first: # restarted from scratch in the same run dir, not a resume: drop the older run
            data = {}
        first = start if first is None or start <= first else first
        for tag in tags:
            d = data.setdefault(tag, {})
            for s in ea.Scalars(tag):
                d[s.step] = s.value
    return {t: (np.array(sorted(d)), np.array([d[k] for k in sorted(d)])) for t, d in data.items()}

def require(S, tag):
    if tag not in S:
        prefix = tag.split("/", 1)[0] + "/"
        have = sorted(t.split("/", 1)[1] for t in S if t.startswith(prefix))
        raise KeyError(f"'{tag}' not logged in this run (has under '{prefix}': {have}) -- was it trained with an older env.py?")
    return S[tag]

# ====== DATA ======
def ema(y, alpha=0.1):
    out, acc = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out

def plot_raw_ema(ax, x, y, color, label, ls="-", alpha=0.1, dots=False):
    """Raw values faint (thin line, or dots for sparse series) under a thick EMA. label is formatted with the final EMA."""
    if dots:
        ax.plot(x, y, color=color, lw=0, marker="o", ms=3.5, alpha=0.4)
    else:
        ax.plot(x, y, color=color, lw=1, alpha=0.25)
    ys = ema(y, alpha)
    ax.plot(x, ys, color=color, lw=LW, ls=ls, label=label.format(final=ys[-1]))
    return ys

# ====== CLI / OUTPUT ======
def parse_run_args(description=None, run_required=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run", type=str, required=run_required, help="Run name under payload_control/logs")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: logs/<run>/plots, else logs/plots)")
    return parser.parse_args()

def save_figure(fig, args, name, exts=("svg", "png"), dpi=200):
    out = args.out or plots_dir(run_dir(args.run) if args.run else LOG_DIR)
    for ext in exts:
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi)
        print("saved", path)
