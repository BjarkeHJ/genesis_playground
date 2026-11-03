"""Single-route evaluation of a trained policy, in detail. One route (a sampler route type with --seed, or figure8) is
flown at each of --speeds, all speeds in parallel and overlaid in one figure: nominal system, starting at rest on the
reference, deterministic policy, recorded until --settle seconds after the route end (the final stop).

Panels against progress along the reference [m] (aligns the same corner across speeds), the final stop against time
since the route end: path (top-down), reference and payload speed, tracking error E/N/U, payload roll/pitch, swing
angles x/y (heading frame) with the cone angle (cable from vertical), actions. Color = speed, line style = component. Per-speed metrics are printed as a
table (eval_route_batch.py for many routes and checkpoint comparisons).

    python eval/eval_route.py <route> --run <run> [--ckpt N] [--seed 0] [--speeds 1 1.5 2] [--settle 5] [--viewer] [--record] [--follow]
    route: random | survey | loop | zigzag | stop (sampled with --seed) | figure8
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # payload_control/

from config.run_config import use_run_configs, plots_dir, resolve_ckpt, dataclass_to_dict
use_run_configs() # logs/<--run>/config in place of the current configs -- must precede env/config imports

import dataclasses # noqa: E402
from env import PayloadControlEnv # noqa: E402
from components.reference import TrajectoryBank # noqa: E402
from components.route import Route, RouteSampler # noqa: E402
from config.config_env import EnvConfig # noqa: E402
from config.config_training import TrainConfig # noqa: E402
from eval.routes import TEST_ROUTES # noqa: E402
from eval.route_metrics import rollout, case_metrics, print_table, recorded_steps # noqa: E402

FIXED_ROUTES = ("figure8", ) # from eval/routes.py; every other route type is sampled

def route_waypoints(name, seed, sampler_cfg):
    if name in FIXED_ROUTES:
        return TEST_ROUTES[name]
    return RouteSampler(sampler_cfg).sample(seed, name)[0] # the sampler's v_des is ignored: --speeds instead

# ====== PLOT ======
def plot(name, seed, speeds, routes, results, h, crash, durations, settle_s, dt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Color = speed, line style = component within a panel (black legend per panel)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(speeds)))
    styles = ("-", "--", ":", "-.")
    T = h["payload"].shape[1]
    t = (np.arange(T) + 1) * dt
    hover = durations.max() == 0.0
    xlabel = "time [s]" if hover else "progress along reference [m]"

    fig = plt.figure(figsize=(16, 23), constrained_layout=True)
    grid = fig.add_gridspec(6, 2, height_ratios=(1.3, 1, 1, 1, 1, 0.9))
    ax_path, ax_speed = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax = {k: fig.add_subplot(grid[r, :]) for k, r in (("err", 1), ("rp", 2), ("swing", 3), ("act", 4))}
    ax["end_err"], ax["end_cone"] = fig.add_subplot(grid[5, 0]), fig.add_subplot(grid[5, 1])

    # Panel -> components (label, history key or function of the speed index)
    panels = {
        "err": [("E", lambda j: h["ref"][j][:, 0] - h["payload"][j][:, 0]), ("N", lambda j: h["ref"][j][:, 1] - h["payload"][j][:, 1]),
                ("U", lambda j: h["ref"][j][:, 2] - h["payload"][j][:, 2])],
        "rp": [("roll", lambda j: np.degrees(h["pl_rp"][j][:, 0])), ("pitch", lambda j: np.degrees(h["pl_rp"][j][:, 1]))],
        "swing": [("swing x", lambda j: np.degrees(h["swing_xy"][j][:, 0])), ("swing y", lambda j: np.degrees(h["swing_xy"][j][:, 1])),
                  ("cone (cable from vertical)", lambda j: np.degrees(h["cone"][j]))],
        "act": [(a, lambda j, k=k: h["actions"][j][:, k]) for k, a in enumerate(("thrust", "roll sp", "pitch sp", "yaw-rate sp"))],
    }

    W = routes[0].waypoints
    ax_path.plot(W[:, 0], W[:, 1], "x-", color="#b9b8b2", lw=1, ms=5, label="waypoints", zorder=1)
    for j, v in enumerate(speeds):
        end = recorded_steps(crash[j], durations[j], settle_s, dt)
        on = (np.arange(T) < end) if hover else (t <= durations[j]) & (np.arange(T) < end)
        after = (t > durations[j]) & (np.arange(T) < end)
        col = colors[j]
        ref, pl, rv = h["ref"][j], h["payload"][j], h["ref_vel"][j]
        x = t if hover else np.cumsum(np.linalg.norm(rv, axis=-1) * dt) # progress: arc length travelled by the reference

        ax_path.plot(ref[:end, 0], ref[:end, 1], "--", color=col, lw=1, zorder=2)
        ax_path.plot(pl[:end, 0], pl[:end, 1], color=col, lw=1.8, label=f"{v:g} m/s", zorder=3)
        if crash[j] >= 0 and end > 0:
            ax_path.plot(*pl[end - 1, :2], "x", color="red", ms=12, mew=2.5, zorder=4)
        ax_speed.plot(x[on], np.linalg.norm(rv, axis=-1)[on], "--", color=col, lw=1)
        ax_speed.plot(x[on], h["payload_speed"][j][on], color=col, lw=1.3)

        for k, comps in panels.items():
            for (label, f), ls in zip(comps, styles):
                ax[k].plot(x[on], f(j)[on], ls, color=col, lw=1.2)
        if after.any():
            ts = t[after] - durations[j]
            ax["end_err"].plot(ts, np.linalg.norm(ref - pl, axis=-1)[after], color=col, lw=1.3)
            ax["end_cone"].plot(ts, np.degrees(h["cone"][j][after]), color=col, lw=1.3)

    titles = {"err": ("Tracking error, reference - payload (world)", "[m]"), "rp": ("Payload roll / pitch", "[deg]"),
              "swing": ("Swing angles (heading frame) and cone angle", "[deg]"), "act": ("Actions (clipped)", "[-]"),
              "end_err": ("Final stop: |tracking error|", "[m]"), "end_cone": ("Final stop: cone angle", "[deg]")}
    for k, (title, unit) in titles.items():
        ax[k].set_title(title, loc="left", fontsize=10)
        ax[k].set_ylabel(unit)
        ax[k].set_xlabel("time since route end [s]" if k.startswith("end_") else xlabel)
        ax[k].grid(alpha=0.3)
        if k in panels:
            ax[k].legend(handles=[Line2D([], [], color="k", ls=ls, lw=1.2, label=label) for (label, _), ls in zip(panels[k], styles)],
                         loc="upper right", fontsize=8, ncol=len(panels[k]))
    ax_speed.set_title("Speed (dashed: reference)", loc="left", fontsize=10)
    ax_speed.set_xlabel(xlabel)
    ax_speed.set_ylabel("[m/s]")
    ax_speed.grid(alpha=0.3)
    ax_path.set_title("Path, top-down (dashed: reference)", loc="left", fontsize=10)
    ax_path.set_xlabel("E [m]")
    ax_path.set_ylabel("N [m]")
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.grid(alpha=0.3)
    ax_path.legend(loc="best", fontsize=9)

    summary = "   ".join(f"{r['speed']:g} m/s: err {r['err_mean']:.3f} m, dev max {r['route_dev_max']:.2f} m, "
                         f"swing max {r['swing_max_deg']:.1f}°{', CRASH' if r['crashed'] else ''}" for r in results)
    seed_txt = "" if name in FIXED_ROUTES else f", seed {seed}"
    fig.suptitle(f"{name}{seed_txt}: {len(W)} waypoints, {routes[0].length:.0f} m   (color = speed)\n{summary}",
                 x=0.01, ha="left", fontweight="bold")
    fig.savefig(path, dpi=100)
    plt.close(fig)

# ====== MAIN ======
def main():
    cfg0 = EnvConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("route", choices=(*cfg0.sampler_cfg.route_keywords, *FIXED_ROUTES), help="Route type")
    parser.add_argument("--run", type=str, required=True, help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--seed", type=int, default=0, help="Sampler seed of the route (sampled route types)")
    parser.add_argument("--speeds", type=float, nargs="+", default=(1.0, ), help="v_des the route is flown at, one env each [m/s]")
    parser.add_argument("--settle", type=float, default=5.0, help="Time recorded after the route end [s]")
    parser.add_argument("--settle_tol", type=float, default=0.1, help="Position error that counts as settled after the stop [m]")
    parser.add_argument("--viewer", action="store_true", help="Show the first speed")
    parser.add_argument("--record", action="store_true", help="Save a video of the first speed next to the plot (works without --viewer)")
    parser.add_argument("--follow", action="store_true", help="Viewer/video camera chases the drone from above and behind")
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile (faster startup, slower stepping)")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True, seed=args.seed)

    points = route_waypoints(args.route, args.seed, cfg0.sampler_cfg)
    routes = [Route(points, v, cfg0.route_cfg) for v in args.speeds]
    bank = TrajectoryBank.from_routes(routes, (args.route, ), keyword_ids=[0] * len(routes), route_cfg=cfg0.route_cfg, device="cuda")
    durations = bank.duration.cpu().numpy()
    print(f"{args.route}: {len(points)} waypoints, {routes[0].length:.1f} m, route durations " +
          ", ".join(f"{d:.1f} s @{v:g}" for d, v in zip(durations, args.speeds)))

    # One env per speed: nominal system, at rest on the reference start, no timeout before the longest run ends
    env_cfg = EnvConfig(num_envs=len(routes), episode_length_s=float(durations.max()) + args.settle + 2.0,
                        torch_compile=not args.no_compile, ref_cfg=dataclasses.replace(cfg0.ref_cfg, p_mid_route_start=0.0))
    e = PayloadControlEnv(env_cfg, show_viewer=args.viewer, device="cuda", bank=bank, record=args.record, follow=args.follow)
    e.dr.enabled = False
    e.ref.fix_routes(torch.arange(len(routes)))

    runner = OnPolicyRunner(e, dataclass_to_dict(TrainConfig(run_name=args.run)), log_dir=None, device="cuda")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device) # deterministic: actor mean

    out_dir = plots_dir(os.path.dirname(ckpt_path), os.path.splitext(os.path.basename(ckpt_path))[0])
    tag = args.route if args.route in FIXED_ROUTES else f"{args.route}_s{args.seed}"
    if args.record:
        e.start_recording(os.path.join(out_dir, f"eval_route_{tag}_{args.speeds[0]:g}mps.mp4"))
    h, crash = rollout(e, policy, bank.duration, args.settle, e.dt)
    e.stop_recording()
    cases = [{"name": f"{args.route} @{v:g}", "group": args.route, "speed": v, "route": r} for v, r in zip(args.speeds, routes)]
    results = [case_metrics(c, h, j, crash[j], float(durations[j]), args.settle, args.settle_tol, e.dt) for j, c in enumerate(cases)]
    print(f"\ncheckpoint {ckpt_path} | err/cross/dev/refdev/final [m], lag/settle [s], swing/yaw/tilt [deg]")
    print_table([{**r, "crash_rate": float(r["crashed"])} for r in results], "Per speed")

    path = os.path.join(out_dir, f"eval_route_{tag}.png")
    plot(args.route, args.seed, args.speeds, routes, results, h, crash, durations, args.settle, e.dt, path)
    print(f"\nsaved {path}")

if __name__ == "__main__":
    main()
