"""Reward-shaping figure: every reward / penalty term in env.py as a function of the quantity it acts on.
(a) position tracking, (b) speed near the target + overspeed, (c) swing & payload tilt, (d) payload yaw error,
(e) payload yaw damping, (f) action terms + crash. Parameters are read from config_env so the figure stays in sync."""
import math

import numpy as np

from plot_utils import (plt, cfg, rw, STEPS_PER_S, INK, INK2, MUTED, SURF, LW, apply_style, parse_run_args, save_figure)

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

C_CRASH = "#2a78d6"
C = {tag: (col, ls) for tag, _, col, ls in PENALTIES} # same colour per term as the training figures
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]   # panel a (ordinal)
ORANGE_RAMP = ["#f0a07c", "#eb6834", "#a8401a"] # panel b (ordinal: payload speed)
OCHRE_RAMP = ["#c9a400", "#8a6d00", "#4f3e00"]  # panel e (ordinal: yaw error gate)
TRACK_PEAK = rw.w_track_rough + rw.w_track_fine

def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center", ha="left", fontsize=9.5, color=INK)

def vline_label(ax, x, label):
    # Dotted vertical marker, label running up from the bottom of the axes
    ax.axvline(x, color=MUTED, lw=1, ls=":")
    ax.text(x, 0.03, label, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="bottom", fontsize=8.5, color=MUTED)

def panel_tracking(ax):
    d = np.linspace(0, 5, 400)
    coarse = rw.w_track_rough * np.exp(-d / rw.sigma_track_rough)
    fine = rw.w_track_fine * np.exp(-d / rw.sigma_track_fine)
    ax.plot(d, coarse, color=BLUE_RAMP[0], lw=LW, label=f"coarse  σ = {rw.sigma_track_rough:g} m  (gradient far away)")
    ax.plot(d, fine, color=BLUE_RAMP[1], lw=LW, label=f"fine  σ = {rw.sigma_track_fine:g} m  (precision near target)")
    ax.plot(d, coarse + fine, color=BLUE_RAMP[2], lw=3.0, label="track  (sum)")
    at1 = rw.w_track_rough * math.exp(-1 / rw.sigma_track_rough) + rw.w_track_fine * math.exp(-1 / rw.sigma_track_fine)
    ax.plot([0, 1], [TRACK_PEAK, at1], "o", ms=7, color=BLUE_RAMP[2], mec=SURF, mew=2, zorder=5)
    ax.annotate(f"{TRACK_PEAK:.1f} at target", (0, TRACK_PEAK), xytext=(10, 2), textcoords="offset points", fontsize=10)
    ax.annotate(f"{at1:.2f} at 1 m", (1, at1), xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, TRACK_PEAK * 1.25)
    ax.set_xlabel("payload distance to target [m]")
    ax.set_ylabel("reward per step")
    ax.set_title("(a) Position tracking: two-scale kernel")
    ax.legend(loc="center right", fontsize=9.5)

def panel_speed(ax):
    # Damping: speed penalty gated to the target region, so braking is enforced without slowing the approach
    d = np.linspace(0, 3, 300)
    for v, col in zip((0.5, 1.0, 2.0), ORANGE_RAMP):
        ax.plot(d, rw.w_damping * v * np.exp(-d / rw.sigma_damping), color=col, lw=LW, label=f"|v| = {v:g} m/s")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 3)
    ax.set_xlabel("payload distance to target [m]")
    ax.set_ylabel("penalty per step")
    ax.set_title(f"(b) Damping near target  (σ = {rw.sigma_damping:g} m)")
    ax.legend(loc="lower right", fontsize=9.5, title="payload speed", title_fontsize=9.5)

    # Inset: global overspeed penalty
    ins = ax.inset_axes([0.42, 0.42, 0.52, 0.22])
    v = np.linspace(0, rw.v_max + 2, 200)
    col, ls = C["vmax"]
    ins.plot(v, rw.w_vmax * np.clip(v - rw.v_max, 0, None) ** 2, color=col, lw=LW, ls=ls)
    ins.axvline(rw.v_max, color=MUTED, lw=1, ls=":")
    ins.set_title("overspeed vs |v| [m/s]", fontsize=8.5, fontweight="normal", loc="left", color=INK2)
    ins.tick_params(labelsize=8)
    ins.set_facecolor("#ffffff")

def panel_swing_tilt(ax):
    deg = np.linspace(0, 90, 300)
    th = np.radians(deg)
    for tag, y, name in [
        ("swing_energy", rw.w_swing_energy * th, "swing amplitude θ  (linear)"),
        ("tilt", rw.w_tilt * (1 - np.cos(th)), "payload tilt  (1 − cos)"),
    ]:
        col, ls = C[tag]
        ax.plot(deg, y, color=col, lw=LW, ls=ls, label=name)
        end_label(ax, deg, y, f"{y[-1]:.2f}")
    ax.axhline(0, color=MUTED, lw=1)
    tilt_t, swing_t = math.degrees(cfg.terminate_if_payload_tilt_greater_than), math.degrees(cfg.terminate_if_swingangle_greater_than)
    for val in (tilt_t, swing_t):
        ax.axvline(val, color=MUTED, lw=1, ls=":", label=f"terminate: tilt {tilt_t:.0f}°, swing {swing_t:.0f}°" if val == tilt_t else None)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("angle [deg]")
    ax.set_ylabel("penalty per step")
    ax.set_title("(c) Swing & payload tilt")
    ax.legend(loc="lower left", fontsize=9.5)

def panel_yaw_error(ax):
    deg = np.linspace(0, 180, 400)
    e = np.radians(deg)
    d = rw.delta_yaw
    col, ls = C["yaw_error"]
    y = rw.w_yaw * (np.sqrt(e**2 + d**2) - d)
    ax.plot(deg, y, color=col, lw=LW, ls=ls, label=f"pseudo-Huber  (knee δ = {math.degrees(d):.0f}°)")
    end_label(ax, deg, y, f"{y[-1]:.2f}")
    # Inset: the knee, where the pull switches from linear to quadratic and fades instead of driving bang-bang yaw
    ins = ax.inset_axes([0.5, 0.52, 0.45, 0.4])
    zd = np.linspace(0, 4 * math.degrees(d), 200)
    ze = np.radians(zd)
    ins.plot(zd, rw.w_yaw * (np.sqrt(ze**2 + d**2) - d), color=col, lw=LW, ls=ls)
    ins.plot(zd, rw.w_yaw * ze, color=MUTED, lw=1, ls=(0, (1, 2)))
    ins.axvline(math.degrees(d), color=MUTED, lw=1, ls=":")
    ins.set_title("near zero  (dotted: pure linear)", fontsize=8.5, fontweight="normal", loc="left", color=INK2)
    ins.tick_params(labelsize=8)
    ins.set_facecolor("#ffffff")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(y[-1] * 1.1, -0.05 * y[-1])
    ax.set_xlabel("payload yaw error [deg]")
    ax.set_ylabel("penalty per step")
    ax.set_title("(d) Payload yaw error")
    ax.legend(loc="lower left", fontsize=9.5)

def panel_yaw_damping(ax):
    # Payload yaw rate penalty, gated by the yaw error: braking into the yaw target, free turning far from it
    rate_deg = np.linspace(0, 90, 300)
    rate = np.radians(rate_deg)
    d, s = rw.delta_yaw_damping, rw.sigma_yaw_damping
    for err, col in zip((0.0, s, 2 * s), OCHRE_RAMP):
        y = rw.w_yaw_damping * (np.sqrt(rate**2 + d**2) - d) * math.exp(-err / s)
        ax.plot(rate_deg, y, color=col, lw=LW, label=f"{math.degrees(err):.0f}°")
    vline_label(ax, math.degrees(cfg.sys_cfg.max_yaw_rate), "max yaw-rate setpoint")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 90)
    ax.set_xlabel("payload yaw rate [deg/s]")
    ax.set_ylabel("penalty per step")
    ax.set_title(f"(e) Yaw damping near yaw target  (σ = {math.degrees(s):.0f}°)")
    ax.legend(loc="lower left", fontsize=9.5, title="yaw error", title_fontsize=9.5)

def panel_actions(ax):
    a = np.linspace(0, 2, 300)
    col, ls = C["smooth_actions"]
    w = rw.w_smooth_actions
    if len(set(w)) == 1:
        ax.plot(a, w[0] * a**2, color=col, lw=LW, ls=ls, label=f"smoothness vs |Δa|  ({w[0]:g} · Δa², each axis)")
    else: # per-axis weights differ: one curve each, ordered thrust / roll / pitch / yaw rate
        for wi, name, alpha in zip(w, ("thrust", "roll", "pitch", "yaw rate"), (1.0, 0.8, 0.6, 0.4)):
            ax.plot(a, wi * a**2, color=col, lw=LW, ls=ls, alpha=alpha, label=f"smoothness {name}  ({wi:g} · Δa²)")
    col, ls = C["action_bound"]
    clip = cfg.clip_actions
    ax.plot(a, rw.w_action_bound * np.clip(a - clip, 0, None) ** 2, color=col, lw=LW, ls=ls,
            label=f"bound vs pre-clip |a|  ({rw.w_action_bound:g} · excess², each axis)")
    ax.axvline(clip, color=MUTED, lw=1, ls=":")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(0, 2)
    ax.set_xlabel("action units [-]  (actions clipped to ±1)")
    ax.set_ylabel("penalty per step")
    ax.set_title("(f) Action terms & crash")
    ax.legend(loc="lower left", fontsize=9.5)
    n = abs(rw.w_crash) / TRACK_PEAK
    ax.text(0.03, 0.55, f"crash: {rw.w_crash:g} once, at termination\n= {n:.0f} steps of max tracking reward,\n"
            "plus all reward the episode would still earn", transform=ax.transAxes, fontsize=9, color=C_CRASH, va="top")

def main():
    args = parse_run_args(__doc__, run_required=False)
    apply_style(**{"font.size": 11, "axes.titlesize": 12.5, "legend.fontsize": 10})

    fig = plt.figure(figsize=(13.33, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32, left=0.06, right=0.965, top=0.84, bottom=0.115)

    fig.text(0.06, 0.94, "Reward shaping", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.895,
             r"$r = r_\mathrm{track} + r_\mathrm{damping} + r_\mathrm{vmax} + r_\mathrm{swing} + r_\mathrm{tilt} + r_\mathrm{yaw}"
             r" + r_\mathrm{yaw\,damping} + r_\mathrm{smooth} + r_\mathrm{bound} + r_\mathrm{crash}$"
             f"      (per policy step, {STEPS_PER_S:.0f} Hz; penalty weights are negative)", fontsize=12.5, color=INK2)

    panel_tracking(fig.add_subplot(gs[0, 0]))
    panel_speed(fig.add_subplot(gs[0, 1]))
    panel_swing_tilt(fig.add_subplot(gs[0, 2]))
    panel_yaw_error(fig.add_subplot(gs[1, 0]))
    panel_yaw_damping(fig.add_subplot(gs[1, 1]))
    panel_actions(fig.add_subplot(gs[1, 2]))
    fig.text(0.06, 0.008, "Swing amplitude θ: energy-equivalent angle of the swing mode, 1 − cos θ = E_kin + E_pot, measured about "
             "the drone thrust axis.\nAll weights and shape parameters read from config_env.RewardConfig; "
             "colours match the training figures.", fontsize=9, color=MUTED, va="bottom")
    save_figure(fig, args, "reward_shaping")

if __name__ == "__main__":
    main()
