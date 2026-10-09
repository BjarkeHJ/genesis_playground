"""Reward-shaping slide figure: (a) tracking kernels, (b) speed shaping near the target, (c) attitude/swing/yaw penalties,
(d) task terms during training, (e) penalty terms during training. Reward parameters are read from config_env so the
figure stays in sync with env.py."""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_training_terms import (PC_DIR, cfg, rw, STEPS_PER_S, STYLE, INK, INK2, MUTED, SURF, PENALTIES, LW,
                                 load_scalars, plot_task_terms, plot_penalty_terms, footnote)

C = {tag: (col, ls) for tag, _, col, ls in PENALTIES} # same colour per term as the training panels
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]   # panel a (ordinal)
ORANGE_RAMP = ["#f0a07c", "#eb6834", "#a8401a"] # panel b (ordinal: payload speed)

def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center", ha="left", fontsize=10, color=INK)

def panel_tracking(ax):
    d = np.linspace(0, 5, 400)
    coarse = rw.w_track_rough * np.exp(-d / rw.sigma_track_rough)
    fine = rw.w_track_fine * np.exp(-d / rw.sigma_track_fine)
    ax.plot(d, coarse, color=BLUE_RAMP[0], lw=LW, label=f"coarse  σ = {rw.sigma_track_rough:g} m  (gradient far away)")
    ax.plot(d, fine, color=BLUE_RAMP[1], lw=LW, label=f"fine  σ = {rw.sigma_track_fine:g} m  (precision near target)")
    ax.plot(d, coarse + fine, color=BLUE_RAMP[2], lw=3.0, label="sum")
    peak = rw.w_track_rough + rw.w_track_fine
    at1 = rw.w_track_rough * math.exp(-1 / rw.sigma_track_rough) + rw.w_track_fine * math.exp(-1 / rw.sigma_track_fine)
    ax.plot([0, 1], [peak, at1], "o", ms=7, color=BLUE_RAMP[2], mec=SURF, mew=2, zorder=5)
    ax.annotate(f"{peak:.1f} at target", (0, peak), xytext=(10, 2), textcoords="offset points", fontsize=10.5)
    ax.annotate(f"{at1:.2f} at 1 m", (1, at1), xytext=(8, 6), textcoords="offset points", fontsize=10.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, peak * 1.25)
    ax.set_xlabel("payload distance to target [m]")
    ax.set_ylabel("reward per step")
    ax.set_title("(a) Position tracking: two-scale kernel")
    ax.legend(loc="center right")

def panel_speed(ax):
    # Damping: speed penalty gated to the target region, so braking is enforced without slowing the approach
    d = np.linspace(0, 3, 300)
    for v, col in zip((0.5, 1.0, 2.0), ORANGE_RAMP):
        pen = rw.w_damping * v * np.exp(-d / rw.sigma_damping)
        ax.plot(d, pen, color=col, lw=LW, label=f"|v| = {v:g} m/s")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 3)
    ax.set_xlabel("payload distance to target [m]")
    ax.set_ylabel("penalty per step")
    ax.set_title(f"(b) Damping near target  σ = {rw.sigma_damping:g} m", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, borderaxespad=0.2)

    # Inset: global overspeed penalty
    ins = ax.inset_axes([0.45, 0.42, 0.5, 0.22])
    v = np.linspace(0, rw.v_max + 2, 200)
    col, ls = C["vmax"]
    ins.plot(v, rw.w_vmax * np.clip(v - rw.v_max, 0, None) ** 2, color=col, lw=LW, ls=ls)
    ins.axvline(rw.v_max, color=MUTED, lw=1, ls=":")
    ins.set_title("overspeed vs |v| [m/s]", fontsize=9, fontweight="normal", loc="left", color=INK2)
    ins.tick_params(labelsize=8)
    ins.set_facecolor("#ffffff")

def panel_attitude(ax):
    deg = np.linspace(0, 90, 300)
    th = np.radians(deg)
    curves = [
        ("swing_energy", rw.w_swing_energy * th, "swing amplitude (linear)"),
        ("tilt", rw.w_tilt * (1 - np.cos(th)), "payload tilt  1 − cos"),
        ("yaw_error", rw.w_yaw * (np.sqrt(th**2 + rw.delta_yaw**2) - rw.delta_yaw), "payload yaw err  pseudo-Huber"),
    ]
    # Yaw damping vs yaw error at the max yaw rate: gated to the yaw target like the position damping
    w_max = cfg.sys_cfg.max_yaw_rate
    yaw_rate_pen = np.sqrt(w_max**2 + rw.delta_yaw_damping**2) - rw.delta_yaw_damping
    curves.append(("yaw_damping", rw.w_yaw_damping * yaw_rate_pen * np.exp(-th / rw.sigma_yaw_damping),
                   f"yaw damping at {math.degrees(w_max):.0f}°/s  (vs yaw err)"))
    for tag, y, name in curves:
        col, ls = C[tag]
        ax.plot(deg, y, color=col, lw=LW, ls=ls, label=name)
        end_label(ax, deg, y, f"{y[-1]:.2f}")
    tilt_t, swing_t = math.degrees(cfg.terminate_if_payload_tilt_greater_than), math.degrees(cfg.terminate_if_swingangle_greater_than)
    for val in (tilt_t, swing_t):
        ax.axvline(val, color=MUTED, lw=1, ls=":", label=f"terminate: tilt {tilt_t:.0f}°, swing {swing_t:.0f}°" if val == tilt_t else None)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("angle [deg]")
    ax.set_ylabel("penalty per step")
    ax.set_title("(c) Swing & attitude penalties", fontsize=12)
    ax.legend(loc="lower left", fontsize=9.5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Run name under payload_control/logs")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: the run's log dir)")
    args = parser.parse_args()

    plt.rcParams.update({**STYLE, "font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10})
    S = load_scalars(args.run)

    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.75, 0.75], hspace=0.42, wspace=0.3,
                          left=0.06, right=0.96, top=0.86, bottom=0.2)

    fig.text(0.06, 0.945, "Reward shaping", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.905,
             r"$r = r_\mathrm{track} - r_\mathrm{damping} - r_\mathrm{vmax} - r_\mathrm{swing} - r_\mathrm{tilt} - r_\mathrm{yaw}"
             r" - r_\mathrm{yaw\,damping} - r_\mathrm{smooth} - r_\mathrm{bound} - r_\mathrm{crash}$"
             f"      (per policy step, {STEPS_PER_S:.0f} Hz)", fontsize=13, color=INK2)

    panel_tracking(fig.add_subplot(gs[0, 0]))
    panel_speed(fig.add_subplot(gs[0, 1]))
    panel_attitude(fig.add_subplot(gs[0, 2]))
    plot_task_terms(fig.add_subplot(gs[1, 0:2]), S, title=f"(d) Task terms during training  —  run '{args.run}'")
    ax_pen = fig.add_subplot(gs[1, 2])
    plot_penalty_terms(ax_pen, S, title="(e) Penalty terms", legend_anchor=None)
    # Panel (e) is too narrow for its own legend: one row across the bottom of the figure
    fig.legend(*ax_pen.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.51, 0.025), ncol=4, fontsize=9,
               title="(e) penalty terms, final EMA:", title_fontsize=9, alignment="left")
    footnote(fig, 0.06, 0.012, 9)

    out = args.out or os.path.join(PC_DIR, "logs", args.run)
    for ext in ("svg", "png"):
        path = os.path.join(out, f"reward_shaping.{ext}")
        fig.savefig(path, dpi=200)
        print("saved", path)

if __name__ == "__main__":
    main()
