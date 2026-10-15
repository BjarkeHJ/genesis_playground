"""Training dashboard from a run's TensorBoard logs: task outcome, success rate, penalties and return (top row), and
optimisation health -- action std, adaptive learning rate, value loss, throughput (bottom row). Standalone: depends only on plot_utils."""
import glob
import os
import re

import numpy as np
from matplotlib.transforms import blended_transform_factory

from plot_utils import (plt, cfg, rw, STEPS_PER_S, INK, INK2, MUTED, LW, apply_style, load_scalars, require,
                        plot_raw_ema, parse_run_args, run_dir, save_figure)
from config_training import PPOConfig

ppo = PPOConfig()
LR_MIN, LR_MAX = 1e-5, 1e-2 # rsl_rl's adaptive-schedule clamps (algorithms/ppo.py)

# Categorical slots from the shared validated set; a colour means one entity across the whole dashboard
C_TRACK, C_CRASH, C_SUCCESS = "#2a78d6", "#e34948", "#4a3aa7"
C_SINGLE = INK2 # single-series panels: the title names the series, no hue needed
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

# ====== SHARED MARKERS ======
def checkpoints(run):
    pts = [int(m.group(1)) for f in glob.glob(os.path.join(run_dir(run), "model_*.pt"))
           if (m := re.search(r"model_(\d+)\.pt$", f))]
    return np.array(sorted(pts))

def finish(ax, it, title, ylabel, xlabel=False):
    ax.set_xlim(it[0], it[-1])
    ax.set_title(title, fontsize=12.5)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("PPO iteration")

# ====== TOP ROW: WHAT IS IT LEARNING ======
def plot_outcome(ax, S, ckpts):
    # Logged reward terms are episode sums / episode_length_s, so the per-second maximum is (max reward per step) * steps per second
    it, track = require(S, "Episode_Reward/track")
    _, crash = require(S, "Episode_Reward/crash")
    _, eplen = require(S, "Train/mean_episode_length")
    series = [(100 * track / ((rw.w_track_rough + rw.w_track_fine) * STEPS_PER_S), C_TRACK, "-", "tracking (% of max)"),
              (100 * crash * cfg.episode_length_s / rw.w_crash, C_CRASH, (0, (2, 2)), "episodes ending in crash"),
              (100 * eplen / (cfg.episode_length_s * STEPS_PER_S), C_SINGLE, (0, (1, 2)), "episode length (% of max)")]
    for y, col, ls, name in series:
        plot_raw_ema(ax, it, y, col, name + "  ({final:.0f}%)", ls=ls)
    if len(ckpts):
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot(ckpts, np.zeros_like(ckpts), transform=tr, lw=0, marker="^", ms=8, color=INK, clip_on=False,
                label=f"checkpoint saved ({len(ckpts)})")
    ax.set_ylim(0, 103)
    finish(ax, it, "Task outcome", "%")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=9)

def plot_success(ax, S):
    if "Task/success_rate" not in S:
        ax.text(0.5, 0.5, "no success rate logged", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title("Success rate", fontsize=12.5)
        return
    it, sr = S["Task/success_rate"] # sparse (one point per window): dots and a faster EMA
    plot_raw_ema(ax, it, 100 * sr, C_SUCCESS, "success rate  ({final:.0f}%)", alpha=0.3, dots=True)
    ax.set_ylim(0, 103)
    finish(ax, it, "Success rate", "% of episodes per window")
    ax.legend(loc="lower right", fontsize=9)

def plot_penalties(ax, S):
    lo, it = 0.0, None
    for tag, name, col, ls in PENALTIES:
        if f"Episode_Reward/{tag}" not in S:
            continue
        it, y = S[f"Episode_Reward/{tag}"]
        plot_raw_ema(ax, it, y, col, name + "  ({final:.2f})", ls=ls)
        lo = min(lo, np.percentile(y, 2)) # ignore the first-iteration spikes when scaling
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(lo * 1.1, -0.05 * lo)
    finish(ax, it, "Penalty terms", "penalty per second of episode")
    ax.legend(loc="lower left", fontsize=8, ncol=2, columnspacing=1.0, handlelength=2.2)

def plot_return(ax, S):
    it, y = require(S, "Train/mean_reward")
    plot_raw_ema(ax, it, y, C_SINGLE, "final {final:.0f}")
    ax.axhline(0, color=MUTED, lw=1)
    finish(ax, it, "Mean episode return", "undiscounted return")
    ax.legend(loc="lower right", fontsize=9)

# ====== BOTTOM ROW: IS THE OPTIMISATION HEALTHY ======
def plot_std(ax, S):
    it, y = require(S, "Policy/mean_std")
    plot_raw_ema(ax, it, y, C_SINGLE, "final {final:.3f}")
    finish(ax, it, "Action std (exploration)", "mean policy std (action units)", xlabel=True)
    ax.legend(loc="upper right", fontsize=9)

def plot_lr(ax, S):
    it, y = require(S, "Loss/learning_rate")
    ax.plot(it, y, color=C_SINGLE, lw=LW, drawstyle="steps-post")
    for b in (LR_MIN, LR_MAX):
        ax.axhline(b, color=MUTED, lw=1, ls=(0, (4, 2)))
    tail = y[-max(1, len(y) // 5):] # last 20% of the run
    at_max, at_min = np.mean(tail >= LR_MAX * 0.999), np.mean(tail <= LR_MIN * 1.001)
    note = (f"last 20%: {at_max:.0%} at upper clamp → updates smaller than KL target" if at_max > 0.5 else
            f"last 20%: {at_min:.0%} at lower clamp → updates overshoot KL target" if at_min > 0.5 else
            "tracking the KL target")
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(LR_MIN / 2, LR_MAX * 2)
    finish(ax, it, f"Learning rate  (adaptive, KL target {ppo.desired_kl:g})", "learning rate (log)", xlabel=True)

def plot_value_loss(ax, S):
    it, y = require(S, "Loss/value")
    plot_raw_ema(ax, it, y, C_SINGLE, "final {final:.3g}")
    ax.set_yscale("log")
    finish(ax, it, "Value loss (critic fit)", "MSE (log)", xlabel=True)
    ax.legend(loc="upper left", fontsize=9)

def plot_throughput(ax, S):
    it, fps = require(S, "Perf/total_fps")
    hours = (require(S, "Perf/collection_time")[1] + require(S, "Perf/learning_time")[1]).sum() / 3600
    plot_raw_ema(ax, it, fps / 1e3, C_SINGLE, "final {final:.0f}k steps/s")
    ax.set_ylim(0, None)
    finish(ax, it, f"Throughput  ({hours:.1f} h wall-clock)", "env steps / s (×1000)", xlabel=True)
    ax.legend(loc="lower right", fontsize=9)

def footnote(fig, x, y, fs):
    fig.text(x, y, "Thin lines / dots: raw per-iteration values · thick: EMA (α = 0.1; success rate α = 0.3). "
             f"Reward terms are episode sums divided by the {cfg.episode_length_s:g} s episode length; legend values are the final EMA. "
             "▲ checkpoints.",
             fontsize=fs, color=MUTED)

def main():
    args = parse_run_args(__doc__)
    apply_style()
    S = load_scalars(args.run)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5), sharex=True)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.87, bottom=0.1, wspace=0.28, hspace=0.38)
    (a_out, a_succ, a_pen, a_ret), (a_std, a_lr, a_val, a_fps) = axes
    plot_outcome(a_out, S, checkpoints(args.run))
    plot_success(a_succ, S)
    plot_penalties(a_pen, S)
    plot_return(a_ret, S)
    plot_std(a_std, S)
    plot_lr(a_lr, S)
    plot_value_loss(a_val, S)
    plot_throughput(a_fps, S)
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)

    it = require(S, "Train/mean_reward")[0]
    fig.suptitle(f"Training dashboard  —  run '{args.run}'  ·  {it[-1] + 1} iterations", x=0.045, ha="left",
                 y=0.97, fontsize=16, fontweight="bold", color=INK)
    for row, label in zip(axes, ["What is it learning?", "Is the optimisation healthy?"]):
        fig.text(0.045, row[0].get_position().y1 + 0.045, label, fontsize=11, color=MUTED, style="italic")
    footnote(fig, 0.045, 0.02, 9.5)
    save_figure(fig, args, "training_dashboard")

if __name__ == "__main__":
    main()
