"""Routes drawn by RouteSampler, one row per route keyword, solved with MinSnapSolver. No simulator.

    route_samples.png       top-down (E-N) view: raw waypoints and the solved min-snap path colored by speed / v_des
    trajectory_samples.png  speed, |a| and |j| against time, each over its limit (v_des, a_max, j_max), fillet arcs shaded,
                            followed by --hold seconds of the held end state (the stop and stabilization phase)

    python utils/plot_route.py [--n 4] [--seed 0] [--keywords random survey ...] [--hold 5] [--out DIR] [--show]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

PC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # payload_control/
sys.path.insert(0, PC_DIR)
from components.route import Route, RouteSampler  # noqa: E402
from components.trajectory import MinSnapSolver  # noqa: E402
from config.config_env import RouteConfig, RouteSamplerConfig  # noqa: E402
from config.run_config import plots_dir  # noqa: E402

# ====== STYLE (house style of eval/plot_utils.py) ======
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
C_WAYPOINT, C_START, C_END = "#8a8984", "#2d6a3e", "#b03a2e"
C_SPEED, C_ACC, C_JERK, C_BAND, C_HOLD = "#1f5c86", "#d9822b", "#7b5ea7", "#eef1f0", "#e3edf4"
CMAP_SPEED = LinearSegmentedColormap.from_list("speed", ["#1f5fbf", "#d62728"]) # slow (blue) -> fast (red)
NORM_SPEED = Normalize(0.0, 1.1) # speed / v_des
TRACE_DT = 0.02 # sampling step of the solved trajectory [s]
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.fontsize": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
}

# ====== SAMPLING ======
def sample_routes(sampler, solver, cfg, keywords, n, seed):
    # {keyword: [(seed, route, traj, solver node positions)]}, seeds seed + 1000 * row + col
    out = {}
    for row, kw in enumerate(keywords):
        out[kw] = []
        for col in range(n):
            s = seed + 1000 * row + col
            points, v_des = sampler.sample(s, kw)
            route = Route(points, v_des, cfg)
            pos, vel, fix, T = solver.plan(route)
            out[kw].append((s, route, solver.time_scale(solver.solve_fixed_times(pos, vel, fix, T)), pos))
    return out

def fillet_intervals(route, traj, solver_pos):
    # Time interval of each filleted corner (entry -> exit node) on the solved trajectory. Route nodes are a subset of
    # the solver nodes, found by position
    out = []
    for a, b in route.corner_idx:
        if a == b:
            continue
        ia, ib = (int(np.argmin(np.linalg.norm(solver_pos - route.node_pos[k], axis=-1))) for k in (a, b))
        out.append((traj.t_nodes[ia], traj.t_nodes[ib]))
    return out

# ====== PLOT ======
def plot_route(ax, kw, seed, route, traj):
    W = route.waypoints
    _, d = traj.sample(TRACE_DT, max_order=1)
    v_ref = max(route.v_des, 1e-9)
    speed = np.linalg.norm(d[1], axis=-1) / v_ref

    ax.plot(W[:, 0], W[:, 1], color=C_WAYPOINT, lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(W[:, 0], W[:, 1], marker="x", s=18, color=C_WAYPOINT, lw=0.9, zorder=2, label="waypoints")
    xy = d[0][:, :2]
    lc = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1), cmap=CMAP_SPEED, norm=NORM_SPEED, lw=1.8, zorder=3)
    lc.set_array(0.5 * (speed[:-1] + speed[1:]))
    ax.add_collection(lc)
    ax.scatter(*W[0, :2], marker="o", s=40, color=C_START, zorder=4, label="start")
    ax.scatter(*W[-1, :2], marker="s", s=40, color=C_END, zorder=4, label="end")

    z = route.points[:, 2]
    ax.set_title(f"{kw} · seed {seed} · " + ("hover" if route.is_hover else f"v_des {route.v_des:.1f} m/s"))
    info = (f"hold at z {z[0]:.1f} m" if route.is_hover else
            f"{len(W)} pts · {route.length:.0f} m · z {z.min():.1f}–{z.max():.1f} m\n"
            f"{traj.num_segments} seg · {traj.duration:.0f} s · k {traj.k_scale:.2f} · "
            f"mean v/v_des {route.length / max(traj.duration, 1e-9) / v_ref:.2f}")
    ax.text(0.0, -0.2, info, transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")
    if route.is_hover: # a single point: fixed window instead of autoscaling to nothing
        ax.set_xlim(W[0, 0] - 2.0, W[0, 0] + 2.0)
        ax.set_ylim(W[0, 1] - 2.0, W[0, 1] + 2.0)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
    return lc

def plot_traces(ax, kw, seed, route, traj, solver_pos, cfg, hold):
    # Speed, |a| and |j| over their limits: 1 is the limit (v_des for speed), shaded bands are the fillet arcs, then
    # `hold` seconds of the held end state. A hover route has no motion: all of it is hold
    t_end = 0.0 if route.is_hover else traj.duration
    t = np.arange(0.0, t_end + hold + 0.5 * TRACE_DT, TRACE_DT)
    d = traj.eval(np.where(t <= t_end, t, traj.duration + 1.0) if route.is_hover else t, max_order=3)
    speed, acc, jerk = (np.linalg.norm(d[k], axis=-1) for k in (1, 2, 3))
    for k, (t0, t1) in enumerate(fillet_intervals(route, traj, solver_pos)):
        ax.axvspan(t0, t1, color=C_BAND, lw=0, zorder=0, label="fillet arc" if k == 0 else None)
    ax.axvspan(t_end, t_end + hold, color=C_HOLD, lw=0, zorder=0, label="hold (end state)")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.plot(t, speed / max(route.v_des, 1e-9), color=C_SPEED, lw=1.6, label="speed / v_des")
    ax.plot(t, acc / cfg.a_max, color=C_ACC, lw=1.2, label="|a| / a_max")
    ax.plot(t, jerk / cfg.j_max, color=C_JERK, lw=1.0, alpha=0.8, label="|j| / j_max")

    ax.set_title(f"{kw} · seed {seed} · " + ("hover" if route.is_hover else f"v_des {route.v_des:.1f} m/s"))
    ax.set_xlabel("t [s]")
    ax.set_ylim(0.0, 1.25)
    ax.set_xlim(0.0, t_end + hold)

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=4, help="Routes per keyword (columns)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; route seed = seed + 1000 * row + col")
    parser.add_argument("--keywords", nargs="+", default=None, help="Route keywords (rows), default all of RouteSamplerConfig")
    parser.add_argument("--hold", type=float, default=5.0, help="Seconds of held end state shown after each route [s]")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: utils/plots)")
    parser.add_argument("--show", action="store_true", help="Open an interactive window instead of only saving")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)

    cfg, sampler_cfg = RouteConfig(), RouteSamplerConfig()
    keywords = args.keywords or list(sampler_cfg.route_keywords)
    routes = sample_routes(RouteSampler(sampler_cfg), MinSnapSolver(cfg), cfg, keywords, args.n, args.seed)
    rows, cols = len(keywords), args.n

    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), constrained_layout=True, squeeze=False)
    for r, kw in enumerate(keywords):
        for c, (seed, route, traj, _) in enumerate(routes[kw]):
            lc = plot_route(axes[r, c], kw, seed, route, traj)
    fig.colorbar(lc, ax=axes, shrink=0.6, pad=0.01, label="speed / v_des")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(f"RouteSampler routes, min-snap solved  (corridor {cfg.corridor:g} m, "
                 f"a_max {cfg.a_max:g} m/s², j_max {cfg.j_max:g} m/s³)", x=0.01, ha="left", fontweight="bold")

    fig_t, axes_t = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), constrained_layout=True, squeeze=False)
    for r, kw in enumerate(keywords):
        for c, (seed, route, traj, pos) in enumerate(routes[kw]):
            plot_traces(axes_t[r, c], kw, seed, route, traj, pos, cfg, args.hold)
    legend = {}
    for ax in axes_t.flat: # union over panels: a panel without fillets has no fillet entry
        legend.update({l: h for h, l in zip(*ax.get_legend_handles_labels())})
    fig_t.legend(legend.values(), legend.keys(), loc="outside lower center", ncol=5)
    fig_t.suptitle("Min-snap trajectory over its limits (time-scaled by k)", x=0.01, ha="left", fontweight="bold")

    out = args.out or plots_dir(os.path.join(PC_DIR, "utils"))
    os.makedirs(out, exist_ok=True)
    for f, name in ((fig, "route_samples.png"), (fig_t, "trajectory_samples.png")):
        path = os.path.join(out, name)
        f.savefig(path, dpi=130)
        print("saved", path)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
