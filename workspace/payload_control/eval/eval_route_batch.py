"""Batch route-tracking evaluation of a trained policy, for comparing checkpoints and runs (single route in detail:
eval_route.py). Every env flies one test case from the start of its route until --settle
seconds after its end (the final stop), all cases in parallel, deterministic policy. No random routes, rotations or
mid-route starts; nominal system unless --dr.

Suites:
    test     fixed routes of eval/routes.py x --speeds (default)
    heldout  routes freshly sampled with the run's RouteSamplerConfig from seeds outside the bank: generalization

Per case: crash, position error to the reference (mean / RMS / max), its cross-track part and the along-track lag,
deviation of the payload from the waypoint polyline (the mission metric) next to the reference's own deviation from it
(what the generator contributes, not the policy), swing, payload yaw error, drone tilt, and the final stop (error after
the route end and the time to settle within --settle_tol). Writes a table to stdout, a JSON and
two figures (paths, time traces) to logs/<run>/plots/model_<iter>/.

    python eval/eval_route_batch.py --run <run> [--ckpt N] [--suite test|heldout] [--speeds 1 1.5 2] [--dr --repeats 5] [--viewer]
"""
import argparse
import json
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
from eval.route_metrics import MOVING_SPEED, rollout, case_metrics, print_table, recorded_steps # noqa: E402

# ====== CASES ======
def build_cases(args, env_cfg):
    # -> list of {name, group, route, rep}: group is the plot row (route name or sampler keyword)
    cases = []
    if args.suite == "test":
        names = args.routes or list(TEST_ROUTES)
        for name in names:
            for v in (args.speeds if len(TEST_ROUTES[name]) > 1 else args.speeds[:1]): # hover: speed doesn't apply
                for rep in range(args.repeats):
                    cases.append({"name": f"{name} @{v:g}", "group": name, "speed": v, "rep": rep,
                                  "route": Route(TEST_ROUTES[name], v, env_cfg.route_cfg)})
    else:
        sampler = RouteSampler(env_cfg.sampler_cfg)
        rng = np.random.default_rng(10**6 + args.seed) # bank seeds come from the bank's own rng: collisions are negligible
        for kw in env_cfg.sampler_cfg.route_keywords:
            for i in range(args.heldout_n):
                seed = int(rng.integers(0, 2**31 - 1))
                route = Route(*sampler.sample(seed, kw), env_cfg.route_cfg)
                cases.append({"name": f"{kw} #{i}", "group": kw, "speed": route.v_des, "rep": 0, "route": route, "seed": seed})
    return cases

# ====== METRICS ======
def aggregate(results):
    # Mean over repeats per case name (crash rate as a fraction)
    out = {}
    for r in results:
        out.setdefault(r["name"], []).append(r)
    agg = []
    for name, rs in out.items():
        a = {"name": name, "group": rs[0]["group"], "speed": rs[0]["speed"], "n": len(rs),
             "crash_rate": float(np.mean([r["crashed"] for r in rs]))}
        for k in rs[0]:
            if isinstance(rs[0][k], float) and k not in ("speed", "crash_t"):
                a[k] = float(np.nanmean([r[k] for r in rs])) if not all(np.isnan(r[k]) for r in rs) else float("nan")
        agg.append(a)
    return agg

# ====== PLOTS ======
def plot(cases, results, h, crash, durations, settle_s, dt, out_dir, suite, max_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    # One plotted case per (group, column): repeat 0, up to max_cols per group
    groups = list(dict.fromkeys(c["group"] for c in cases))
    picks = {g: [i for i, c in enumerate(cases) if c["group"] == g and c["rep"] == 0][:max_cols] for g in groups}
    cols = max(len(p) for p in picks.values())
    norm = Normalize(0.0, 0.5)
    t_all = (np.arange(h["payload"].shape[1]) + 1) * dt

    # Paths: waypoint polyline, reference, payload colored by position error
    fig, axes = plt.subplots(len(groups), cols, figsize=(3.6 * cols, 3.4 * len(groups)), squeeze=False, constrained_layout=True)
    for r, g in enumerate(groups):
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(picks[g]):
                ax.axis("off")
                continue
            i = picks[g][c]
            end = recorded_steps(crash[i], durations[i], settle_s, dt)
            W, ref, pl = cases[i]["route"].waypoints, h["ref"][i][:end], h["payload"][i][:end]
            err = np.linalg.norm(ref - pl, axis=-1)
            ax.plot(W[:, 0], W[:, 1], "x-", color="#b9b8b2", lw=1, ms=4, zorder=1)
            ax.plot(ref[:, 0], ref[:, 1], "--", color="#52514e", lw=1, zorder=2)
            if len(pl) > 1:
                lc = LineCollection(np.stack([pl[:-1, :2], pl[1:, :2]], axis=1), cmap="viridis", norm=norm, lw=2, zorder=3)
                lc.set_array(0.5 * (err[:-1] + err[1:]))
                ax.add_collection(lc)
            if crash[i] >= 0 and len(pl):
                ax.plot(*pl[-1, :2], "rx", ms=10, mew=2, zorder=4)
            res = results[i]
            ax.set_title(f"{cases[i]['name']}{'  CRASH' if res['crashed'] else ''}\nerr {res['err_mean']:.2f} m, dev max {res['route_dev_max']:.2f} m (ref {res['ref_dev_max']:.2f})",
                         fontsize=8, loc="left")
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=7)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, shrink=0.5, label="position error [m]")
    fig.suptitle(f"Route tracking ({suite}): waypoints (grey), reference (dashed), payload (colored by error)", x=0.01, ha="left", fontweight="bold")
    path_png = os.path.join(out_dir, f"eval_route_{suite}_paths.png")
    fig.savefig(path_png, dpi=110)
    plt.close(fig)

    # Time traces: position error with its cross-track part, swing amplitude; route end marked
    fig, axes = plt.subplots(len(groups), cols, figsize=(3.6 * cols, 2.4 * len(groups)), squeeze=False, constrained_layout=True)
    for r, g in enumerate(groups):
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(picks[g]):
                ax.axis("off")
                continue
            i = picks[g][c]
            end = recorded_steps(crash[i], durations[i], settle_s, dt)
            t, ev = t_all[:end], h["ref"][i][:end] - h["payload"][i][:end]
            v = h["ref_vel"][i][:end]
            sp = np.linalg.norm(v, axis=-1)
            tan = v / np.maximum(sp, 1e-9)[:, None]
            cross = np.where(sp > MOVING_SPEED, np.linalg.norm(ev - np.sum(ev * tan, axis=-1)[:, None] * tan, axis=-1), np.nan)
            ax.plot(t, np.linalg.norm(ev, axis=-1), color="#1f5c86", lw=1.2, label="|pos err|")
            ax.plot(t, cross, color="#d9822b", lw=0.9, label="cross-track")
            ax.axvline(durations[i], color="#8a8984", ls=":", lw=1)
            ax2 = ax.twinx()
            ax2.plot(t, np.degrees(h["swing"][i][:end]), color="#7b5ea7", lw=0.8, alpha=0.7)
            ax2.tick_params(labelsize=7, colors="#7b5ea7")
            ax.set_title(cases[i]["name"], fontsize=8, loc="left")
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Position error [m] (blue), cross-track [m] (orange), swing [deg] (purple, right axis); dotted: route end",
                 x=0.01, ha="left", fontweight="bold")
    trace_png = os.path.join(out_dir, f"eval_route_{suite}_traces.png")
    fig.savefig(trace_png, dpi=110)
    plt.close(fig)
    return path_png, trace_png

# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=str, required=True, help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--suite", choices=("test", "heldout"), default="test")
    parser.add_argument("--routes", nargs="+", default=None, choices=list(TEST_ROUTES), help="test suite: subset of routes")
    parser.add_argument("--speeds", type=float, nargs="+", default=(1.0, 1.5, 2.0), help="test suite: v_des per route [m/s]")
    parser.add_argument("--heldout_n", type=int, default=20, help="heldout suite: routes per sampler keyword")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per case (differ only with --dr)")
    parser.add_argument("--dr", action="store_true", help="Domain-randomized system and initial state (default: nominal)")
    parser.add_argument("--settle", type=float, default=5.0, help="Time recorded after each route's end [s]")
    parser.add_argument("--settle_tol", type=float, default=0.1, help="Position error that counts as settled after the stop [m]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true", help="Show env 0 (the first case)")
    parser.add_argument("--record", action="store_true", help="Save a video of env 0 (the first case) next to the plots (works without --viewer)")
    parser.add_argument("--follow", action="store_true", help="Viewer/video camera chases env 0's drone from above and behind")
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile (faster startup, slower stepping)")
    parser.add_argument("--plot_cols", type=int, default=4, help="Cases per row in the figures")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True, seed=args.seed)

    base_cfg = EnvConfig()
    cases = build_cases(args, base_cfg)
    bank = TrajectoryBank.from_routes([c["route"] for c in cases], tuple(dict.fromkeys(c["group"] for c in cases)),
                                      route_cfg=base_cfg.route_cfg, device="cuda")
    durations = bank.duration.cpu().numpy()
    print(f"{len(cases)} cases, route durations {durations.min():.1f}-{durations.max():.1f} s, + {args.settle:g} s settle")

    # One env per case, no timeout before the longest case ends, routes from their start
    env_cfg = EnvConfig(num_envs=len(cases), episode_length_s=float(durations.max()) + args.settle + 2.0,
                        torch_compile=not args.no_compile, ref_cfg=dataclasses.replace(base_cfg.ref_cfg, p_mid_route_start=0.0))
    e = PayloadControlEnv(env_cfg, show_viewer=args.viewer, device="cuda", bank=bank, record=args.record, follow=args.follow)
    e.dr.enabled = args.dr
    e.ref.fix_routes(torch.arange(len(cases)))

    runner = OnPolicyRunner(e, dataclass_to_dict(TrainConfig(run_name=args.run)), log_dir=None, device="cuda")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device) # deterministic: actor mean

    out_dir = plots_dir(os.path.dirname(ckpt_path), os.path.splitext(os.path.basename(ckpt_path))[0])
    tag = f"{args.suite}{'_dr' if args.dr else ''}"
    if args.record:
        e.start_recording(os.path.join(out_dir, f"eval_route_{tag}_{cases[0]['name'].replace(' ', '').replace('@', '_')}.mp4"))
    h, crash = rollout(e, policy, bank.duration, args.settle, e.dt)
    e.stop_recording()
    results = [case_metrics(c, h, i, crash[i], float(durations[i]), args.settle, args.settle_tol, e.dt) for i, c in enumerate(cases)]
    agg = aggregate(results)

    print(f"\ncheckpoint {ckpt_path} | suite {args.suite} | {'domain-randomized' if args.dr else 'nominal'} system | "
          f"err/cross/dev/refdev/final [m], lag/settle [s], swing/yaw/tilt [deg]. dev: payload to the waypoint polyline, "
          f"refdev: the reference to it (generator)")
    print_table(agg, "Per case" + (f" (mean of {args.repeats} repeats)" if args.repeats > 1 else ""))
    by_group = []
    for g in dict.fromkeys(a["group"] for a in agg):
        rows = [a for a in agg if a["group"] == g]
        by_group.append({"name": g, **{k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float) and k != "speed"}})
    print_table(by_group, "Per route" if args.suite == "test" else "Per keyword")
    overall = {k: float(np.nanmean([r[k] for r in agg])) for k in agg[0] if isinstance(agg[0][k], float) and k != "speed"}
    print_table([{"name": "ALL", **overall}], "Overall")

    with open(os.path.join(out_dir, f"eval_route_{tag}.json"), "w") as fh:
        json.dump({"checkpoint": ckpt_path, "args": vars(args), "overall": overall, "per_group": by_group, "per_case": agg,
                   "runs": results}, fh, indent=1)
    paths = plot(cases, results, h, crash, durations, args.settle, e.dt, out_dir, tag, args.plot_cols)
    print(f"\nsaved {os.path.join(out_dir, f'eval_route_{tag}.json')}\nsaved {paths[0]}\nsaved {paths[1]}")

if __name__ == "__main__":
    main()
