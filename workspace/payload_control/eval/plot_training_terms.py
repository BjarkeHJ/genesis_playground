"""Reward terms during training: task terms (left) and penalty terms (right), from a run's TensorBoard logs.
Reward weights are read from config_env so the normalisation stays in sync with env.py."""
import numpy as np

from plot_utils import (plt, cfg, rw, STEPS_PER_S, MUTED, apply_style, legend_outside, load_scalars, require,
                        plot_raw_ema, parse_run_args, save_figure)

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

def plot_task_terms(ax, S, title="Task terms during training"):
    # Logged value = episode sum / episode_length_s, so the per-second maximum is (max reward per step) * steps per second
    it, track = require(S, "Episode_Reward/track")
    _, crash = require(S, "Episode_Reward/crash")
    track_pct = 100 * track / ((rw.w_track_rough + rw.w_track_fine) * STEPS_PER_S)
    crash_pct = 100 * crash * cfg.episode_length_s / rw.w_crash  # one w_crash per crashed episode
    for y, col, ls, name in [(track_pct, C_TRACK, "-", "position tracking (% of max)"),
                             (crash_pct, C_CRASH, (0, (2, 2)), "episodes ending in crash")]:
        plot_raw_ema(ax, it, y, col, name + "  (final {final:.0f}%)", ls=ls)
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
        plot_raw_ema(ax, it, y, col, name + "  ({final:.2f})", ls=ls)
        lo = min(lo, np.percentile(y, 2)) # ignore the first-iteration spikes when scaling
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(it[0], it[-1])
    ax.set_ylim(lo * 1.1, -0.05 * lo)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("penalty per second of episode")
    ax.set_title(title)
    if legend_anchor is not None: # None: caller places the legend (e.g. fig.legend across the figure)
        legend_outside(ax, fontsize=legend_fs, ncol=legend_ncol, anchor=legend_anchor)

def footnote(fig, x, y, fs):
    fig.text(x, y, "Thin lines: raw per-iteration values · thick: EMA (α = 0.1). "
             f"Logged values are episode sums divided by the {cfg.episode_length_s:g} s episode length; "
             "legend values are the final EMA.", fontsize=fs, color=MUTED)

def main():
    args = parse_run_args(__doc__)
    apply_style()
    S = load_scalars(args.run)

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(13.33, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.22))
    fig.subplots_adjust(left=0.06, right=0.8, top=0.9, bottom=0.17)
    plot_task_terms(ax, S, title=f"Task terms during training  —  run '{args.run}'")
    plot_penalty_terms(axd, S, legend_fs=10)
    footnote(fig, 0.06, 0.02, 9.5)
    save_figure(fig, args, "training_terms")

if __name__ == "__main__":
    main()
