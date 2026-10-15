"""Curriculum progress during training: the domain-randomization level (top) and the success rate driving it (bottom),
from a run's TensorBoard logs. Thresholds and enabled groups are read from config_env so they stay in sync with env.py."""
import numpy as np

from plot_utils import (plt, dr, INK, MUTED, GRID, LW, apply_style, legend_outside, load_scalars, plot_raw_ema,
                        parse_run_args, require, save_figure)

C_LEVEL = "#1baf7a"
C_BACKOFF = "#e34948"
C_SUCCESS = "#2a78d6"


def plot_level(ax, S, title="Curriculum level"):
    # Logged level is the mean over an iteration's episode resets, so each step shows up as a one-iteration ramp
    it, y = require(S, "Curriculum/level")
    full = np.flatnonzero(y >= 1.0 - 1e-6)
    when = f"full @ {it[full[0]]}" if len(full) else f"final {y[-1]:.1f}"
    ax.fill_between(it, 0, y, color=C_LEVEL, alpha=0.12, lw=0)
    ax.plot(it, y, color=C_LEVEL, lw=LW, label=f"level  ({when})")
    if len(full):
        ax.axvline(it[full[0]], color=C_LEVEL, lw=0.8, ls=":", alpha=0.8)
    down = np.flatnonzero(np.diff(y) < -1e-6) + 1 # back-offs (both iterations of a split step land here, so mark the first)
    down = down[np.r_[True, np.diff(down) > 1]] if len(down) else down
    if len(down):
        ax.plot(it[down], y[down], lw=0, marker="v", ms=7, color=C_BACKOFF, label=f"back-off  ({len(down)}×)")
    ax.set_xlim(it[0], it[-1])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("DR level (0 nominal → 1 full)")
    ax.set_title(title)
    ax.text(1.02, 0.0, "scales: " + ", ".join(dr.groups), transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")
    legend_outside(ax, fontsize=10)

def plot_success(ax, S, title="Success rate per window"):
    it, y = require(S, "Curriculum/success_rate")
    y = 100 * y
    ax.axhspan(100 * dr.curriculum_down, 100 * dr.curriculum_up, color=GRID, alpha=0.6, lw=0)
    ax.axhline(100 * dr.curriculum_up, color=INK, lw=1, ls=(0, (4, 2)), label=f"advance  (> {dr.curriculum_up:.0%})")
    ax.axhline(100 * dr.curriculum_down, color=MUTED, lw=1, ls=(0, (1, 2)), label=f"back off  (< {dr.curriculum_down:.0%})")
    # Sparse (one point per window), so dots and a faster EMA than the per-iteration curves
    plot_raw_ema(ax, it, y, C_SUCCESS, "success rate  (final {final:.0f}%)", alpha=0.3, dots=True)
    ax.set_ylim(0, 102)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("% of episodes")
    ax.set_title(title)
    legend_outside(ax, fontsize=10)

def footnote(fig, x, y, fs):
    rule = (f"Every {dr.curriculum_interval} policy steps the level steps +{dr.curriculum_step:g} if success > {dr.curriculum_up:.0%} "
            f"and −{dr.curriculum_step_down:g} if success < {dr.curriculum_down:.0%} (shaded: hold); after a change one episode passes "
            "before success is measured again. " if dr.curriculum else "Curriculum off: full ranges from the start, success logged for comparison. ")
    fig.text(x, y, rule + f"Success = timeout without crash, mean payload error < {dr.curriculum_pos_err:g} m over the last "
             f"{dr.curriculum_window_s:g} s. Dots: raw · line: EMA (α = 0.3).", fontsize=fs, color=MUTED, wrap=True)

def main():
    args = parse_run_args(__doc__)
    apply_style()
    S = load_scalars(args.run)

    fig, (ax, axs) = plt.subplots(2, 1, figsize=(13.33, 8.0), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.18))
    fig.subplots_adjust(left=0.06, right=0.76, top=0.94, bottom=0.12)
    plot_level(ax, S, title=f"Curriculum level during training  —  run '{args.run}'")
    plot_success(axs, S)
    footnote(fig, 0.06, 0.02, 9.5)
    save_figure(fig, args, "training_curriculum")

if __name__ == "__main__":
    main()
