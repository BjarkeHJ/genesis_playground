"""Reward terms during training: task terms (left) and penalty terms (right), from a run's TensorBoard logs.
Reward weights are read from config_env so the normalisation stays in sync with env.py."""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # payload_control/
sys.path.insert(0, PC_DIR)
from config_env import EnvConfig  # noqa: E402

cfg = EnvConfig()
rw = cfg.rew_cfg
STEPS_PER_S = 1.0 / cfg.dt

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# Fixed colour per reward term (categorical slots, validated light-mode), shared by every figure
C_TRACK, C_CRASH = "#2a78d6", "#e34948"
PENALTIES = [ # (tag, label, colour, dash) -- dashes double-encode identity, three hues sit below 3:1 on the surface
    ("damping", "damping near target", "#eb6834", "-"),
    ("swing_energy", "swing energy", "#1baf7a", (0, (6, 3))),
    ("yaw_error", "payload yaw err", "#eda100", (0, (2, 2))),
    ("yaw_damping", "yaw damping near target", "#8a6d00", (0, (8, 2, 1, 2))),
    ("tilt", "payload tilt", "#e87ba4", (0, (6, 2, 2, 2))),
    ("vmax", "overspeed", "#008300", (0, (1, 3))),
    ("smooth_actions", "action smoothness", "#4a3aa7", (0, (10, 3))),
    ("action_bound", "action bound", "#52514e", (0, (3, 1, 1, 1))),
    ("yaw", "payload yaw (older runs)", "#eda100", (0, (2, 2))), # tag before the yaw_error rename; absent in new runs
]
LW = 2.0

STYLE = {
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold", "axes.titlesize": 14,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.fontsize": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
}

def load_scalars(run):
    files = sorted(glob.glob(os.path.join(PC_DIR, "logs", run, "events.*")))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files in {os.path.join(PC_DIR, 'logs', run)}")
    data = {}
    for f in files:  # later files win on overlapping steps (resumed runs)
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            d = data.setdefault(tag, {})
            for s in ea.Scalars(tag):
                d[s.step] = s.value
    return {t: (np.array(sorted(d)), np.array([d[k] for k in sorted(d)])) for t, d in data.items()}

def ema(y, alpha=0.1):
    out, acc = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out

def require(S, name):
    tag = f"Episode_Reward/{name}"
    if tag not in S:
        have = sorted(t.split("/", 1)[1] for t in S if t.startswith("Episode_Reward/"))
        raise KeyError(f"'{tag}' not logged in this run (has: {have}) -- was it trained with an older env.py?")
    return S[tag]

def plot_task_terms(ax, S, title="Task terms during training"):
    # Logged value = episode sum / episode_length_s, so the per-second maximum is (max reward per step) * steps per second
    it, track = require(S, "track")
    _, crash = require(S, "crash")
    track_pct = 100 * track / ((rw.w_track_rough + rw.w_track_fine) * STEPS_PER_S)
    crash_pct = 100 * crash * cfg.episode_length_s / rw.w_crash  # one w_crash per crashed episode
    for y, col, ls, name in [(track_pct, C_TRACK, "-", "position tracking (% of max)"),
                             (crash_pct, C_CRASH, (0, (2, 2)), "episodes ending in crash")]:
        ax.plot(it, y, color=col, lw=1, alpha=0.25)
        ys = ema(y)
        ax.plot(it, ys, color=col, lw=LW, ls=ls, label=f"{name}  (final {ys[-1]:.0f}%)")
    ax.set_xlim(it[0], it[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("% of maximum  /  % of episodes")
    ax.set_title(title)
    ax.legend(loc="center right")

def plot_penalty_terms(ax, S, title="Penalty terms", legend_fs=None, legend_ncol=1, legend_anchor=(1.02, 1.0)):
    lo = 0.0
    for tag, name, col, ls in PENALTIES:
        if f"Episode_Reward/{tag}" not in S:
            continue
        it, y = S[f"Episode_Reward/{tag}"]
        ax.plot(it, y, color=col, lw=1, alpha=0.25)
        ys = ema(y)
        ax.plot(it, ys, color=col, lw=LW, ls=ls, label=f"{name}  ({ys[-1]:.2f})")
        lo = min(lo, np.percentile(y, 2)) # ignore the first-iteration spikes when scaling
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(it[0], it[-1])
    ax.set_ylim(lo * 1.1, -0.05 * lo)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("penalty per second of episode")
    ax.set_title(title)
    # Outside the axes: the curves fill the whole plot area at some point in training
    if legend_anchor is not None: # None: caller places the legend (e.g. fig.legend across the figure)
        ax.legend(loc="upper left", bbox_to_anchor=legend_anchor, ncol=legend_ncol, fontsize=legend_fs, borderaxespad=0.0)

def footnote(fig, x, y, fs):
    fig.text(x, y, "Thin lines: raw per-iteration values · thick: EMA (α = 0.1). "
             f"Logged values are episode sums divided by the {cfg.episode_length_s:g} s episode length; "
             "legend values are the final EMA.", fontsize=fs, color=MUTED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Run name under payload_control/logs")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: the run's log dir)")
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    S = load_scalars(args.run)

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(13.33, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.22))
    fig.subplots_adjust(left=0.06, right=0.8, top=0.9, bottom=0.17)
    plot_task_terms(ax, S, title=f"Task terms during training  —  run '{args.run}'")
    plot_penalty_terms(axd, S, legend_fs=10)
    footnote(fig, 0.06, 0.02, 9.5)

    out = args.out or os.path.join(PC_DIR, "logs", args.run)
    for ext in ("svg", "png"):
        path = os.path.join(out, f"training_terms.{ext}")
        fig.savefig(path, dpi=200)
        print("saved", path)

if __name__ == "__main__":
    main()
