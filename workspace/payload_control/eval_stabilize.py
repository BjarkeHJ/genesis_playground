import argparse
import os

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from env import *
from utils import *
from config_training import TrainConfig
from config_env import EnvConfig

# Disturbance-rejection eval: the target is the nominal hover point (payload hanging straight below the drone at spawn),
# and every episode starts with the system knocked out of equilibrium:
#   - payload swung out by a random angle/azimuth on a sphere of tether length around the drone
#   - random lateral payload velocity kick
#   - random drone body-rate kick
# The policy only has to bring the payload back to rest at the target.

def sample_disturbance(e, idx, args):
    k, dev = len(idx), e.device
    L = e.sys_cfg.rest_length

    # Swing: angle in [0.5, 1] * max so every episode is a real disturbance, uniform azimuth
    theta = math.radians(args.swing_deg) * gs_rand_float(0.5, 1.0, (k, ), dev)
    phi = gs_rand_float(-math.pi, math.pi, (k, ), dev)
    drone_pos = e.drone.get_pos(idx)
    payload_pos = drone_pos + L * torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), -torch.cos(theta)], dim=-1)
    e.payload.set_pos(payload_pos, zero_velocity=True, envs_idx=idx)

    # Payload lateral velocity kick
    psi = gs_rand_float(-math.pi, math.pi, (k, ), dev)
    v_mag = args.payload_vel * gs_rand_float(0.5, 1.0, (k, ), dev)
    payload_vel = torch.stack([v_mag * torch.cos(psi), v_mag * torch.sin(psi), torch.zeros_like(v_mag)], dim=-1)
    e.payload.set_dofs_velocity(payload_vel, dofs_idx_local=[0, 1, 2], envs_idx=idx)

    # Drone angular velocity kick (random direction)
    w_dir = torch.randn((k, 3), device=dev)
    w_dir = w_dir / torch.clamp(torch.norm(w_dir, dim=-1, keepdim=True), min=EPS)
    drone_ang = w_dir * args.drone_rate * gs_rand_float(0.5, 1.0, (k, 1), dev)
    e.drone.set_dofs_velocity(drone_ang, dofs_idx_local=[3, 4, 5], envs_idx=idx)

    # Target = undisturbed hover point
    e.commands[idx] = e.payload_init_pos
    e.target.set_pos(e.commands[idx], zero_velocity=True, envs_idx=idx)

def refresh_obs(e):
    # Recompute the state-derived buffers from the (disturbed) sim state without counting a step
    e._update_buffers()
    e.episode_length_buf -= 1
    e._update_observations()
    return e.get_observations()

def swing_cone(e):
    # Total swing angle from vertical of the drone->payload vector
    r = e.payload.get_pos() - e.drone.get_pos()
    return torch.atan2(torch.norm(r[:, :2], dim=-1), -r[:, 2])

def pm_lines(ax, val, color, label):
    # Dotted +-val threshold pair with a single legend entry
    for sgn in (1, -1):
        ax.axhline(sgn * val, color=color, ls=":", label=label if sgn > 0 else None)

def save_plot(log, path, args, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axs[0, 0]
    for i, name in enumerate("xyz"):
        ax.plot(t, [v[i] for v in log["err_vec"]], lw=1, label=f"e_{name}")
    ax.plot(t, log["err"], "k", label="|e|")
    pm_lines(ax, args.settle_pos, "g", f"settle ±{args.settle_pos} m")
    ax.set_ylabel("pos error [m]"); ax.legend(ncol=2); ax.grid(alpha=0.3); ax.set_title("Payload position error (target - payload, world)")

    ax = axs[0, 1]
    ax.plot(t, [math.degrees(v) for v in log["cone"]], "k", label="cone")
    ax.plot(t, [math.degrees(s[0]) for s in log["swing"]], lw=1, label="swing x (heading)")
    ax.plot(t, [math.degrees(s[1]) for s in log["swing"]], lw=1, label="swing y (heading)")
    pm_lines(ax, args.settle_swing_deg, "g", f"settle ±{args.settle_swing_deg:g} deg")
    pm_lines(ax, math.degrees(cfg.terminate_if_swingangle_greater_than), "r", "terminate")
    ax.set_ylabel("angle [deg]"); ax.legend(ncol=2); ax.grid(alpha=0.3); ax.set_title("Swing")

    ax = axs[1, 0]
    for i, name in enumerate("xyz"):
        ax.plot(t, [v[i] for v in log["vel"]], lw=1, label=f"v_{name}")
    ax.plot(t, log["speed"], "k", label="|v|")
    pm_lines(ax, args.settle_vel, "g", f"settle ±{args.settle_vel} m/s")
    ax.set_ylabel("velocity [m/s]"); ax.legend(ncol=2); ax.grid(alpha=0.3); ax.set_title("Payload velocity (world)")

    ax = axs[1, 1]
    for i, name in enumerate(("thrust", "roll_sp", "pitch_sp", "yawrate_sp")):
        ax.plot(t, [a[i] for a in log["act"]], lw=1, label=name)
    ax.set_ylabel("action [-]"); ax.legend(); ax.grid(alpha=0.3); ax.set_title("Actions")

    ax = axs[2, 0]
    ax.plot(t, [math.degrees(v[0]) for v in log["rp"]], label="roll")
    ax.plot(t, [math.degrees(v[1]) for v in log["rp"]], label="pitch")
    pm_lines(ax, math.degrees(cfg.terminate_if_rollpitch_greater_than), "r", "terminate")
    ax.set_xlabel("t [s]"); ax.set_ylabel("angle [deg]"); ax.legend(); ax.grid(alpha=0.3); ax.set_title("Payload roll/pitch")

    ax = axs[2, 1]
    ax.plot(t, [math.degrees(v[0]) for v in log["rp_rate"]], label="roll rate")
    ax.plot(t, [math.degrees(v[1]) for v in log["rp_rate"]], label="pitch rate")
    ax.set_xlabel("t [s]"); ax.set_ylabel("rate [deg/s]"); ax.legend(); ax.grid(alpha=0.3); ax.set_title("Payload roll/pitch rates (body)")

    if log["settle_t"] is not None and not math.isnan(log["settle_t"]):
        for ax in axs.flat:
            ax.axvline(log["settle_t"], color="g", ls="--", lw=1)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved stabilization plot: {path}")

def save_motion_plot(log, path, snapshot_dt=0.2):
    # Drone vs payload motion, projected onto the initial swing plane so the drone's counter-movement is visible
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(log["t"])
    drone = np.asarray(log["drone"]) - np.asarray(log["target"])   # positions relative to the target
    payload = np.asarray(log["payload"]) - np.asarray(log["target"])

    # Swing-plane axes: u along the initial horizontal payload offset from the drone, v perpendicular (horizontal)
    d0 = payload[0, :2] - drone[0, :2]
    u = d0 / max(np.linalg.norm(d0), 1e-9)
    v = np.array([-u[1], u[0]])
    drone_u, drone_v = drone[:, :2] @ u, drone[:, :2] @ v
    pl_u, pl_v = payload[:, :2] @ u, payload[:, :2] @ v

    step = max(1, int(round(snapshot_dt / (t[1] - t[0])))) if len(t) > 1 else 1
    snaps = range(0, len(t), step)
    cmap = plt.get_cmap("viridis")
    col = lambda k: cmap(t[k] / t[-1])

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")

    ax = axs[0, 0]
    ax.plot(t, drone_u, "C3", label="drone")
    ax.plot(t, pl_u, "C0", label="payload")
    ax.fill_between(t, drone_u, pl_u, color="0.5", alpha=0.15, label="drone - payload offset")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pos [m]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Along initial swing direction (rel. target)")

    ax = axs[0, 1]
    ax.plot(t, drone_v, "C3", label="drone")
    ax.plot(t, pl_v, "C0", label="payload")
    ax.fill_between(t, drone_v, pl_v, color="0.5", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pos [m]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Across initial swing direction (rel. target)")

    # Side view in the swing plane: pendulum snapshots (tether lines) coloured by time
    ax = axs[1, 0]
    for k in snaps:
        ax.plot([drone_u[k], pl_u[k]], [drone[k, 2], payload[k, 2]], color=col(k), lw=0.8, alpha=0.7)
    ax.plot(drone_u, drone[:, 2], "C3", lw=1.5, label="drone")
    ax.plot(pl_u, payload[:, 2], "C0", lw=1.5, label="payload")
    ax.plot(0.0, 0.0, "kx", ms=10, mew=2, label="target")
    ax.set_xlabel("along swing [m]"); ax.set_ylabel("z rel. target [m]"); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title(f"Side view, swing plane (tether every {snapshot_dt:.1f} s)")

    # Top-down view
    ax = axs[1, 1]
    for k in snaps:
        ax.plot([drone[k, 0], payload[k, 0]], [drone[k, 1], payload[k, 1]], color=col(k), lw=0.8, alpha=0.7)
    ax.plot(drone[:, 0], drone[:, 1], "C3", lw=1.5, label="drone")
    ax.plot(payload[:, 0], payload[:, 1], "C0", lw=1.5, label="payload")
    ax.plot(0.0, 0.0, "kx", ms=10, mew=2, label="target")
    ax.set_xlabel("x rel. target [m]"); ax.set_ylabel("y rel. target [m]"); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Top-down view")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, t[-1]))
    fig.colorbar(sm, ax=axs[1, :].tolist(), label="t [s]", shrink=0.8)

    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved motion plot: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="dev_test", help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=5, help="Stop after this many finished episodes")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer (use with many envs for stats)")
    parser.add_argument("--duration", type=float, default=8.0, help="Episode length [s]")
    # Disturbance magnitudes (each sampled in [0.5, 1] x value)
    parser.add_argument("--swing_deg", type=float, default=45.0, help="Initial payload swing angle [deg]")
    parser.add_argument("--payload_vel", type=float, default=1.0, help="Initial lateral payload velocity [m/s]")
    parser.add_argument("--drone_rate", type=float, default=2.0, help="Initial drone angular velocity [rad/s]")
    # Settled = all of these hold from some time until the end of the episode
    parser.add_argument("--settle_pos", type=float, default=0.25, help="Settled position error [m]")
    parser.add_argument("--settle_swing_deg", type=float, default=5.0, help="Settled swing cone angle [deg]")
    parser.add_argument("--settle_vel", type=float, default=0.2, help="Settled payload speed [m/s]")
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the env-0 plot")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    gs.init(backend=gs.cpu, logging_level="warning")

    # Same TrainConfig as train.py -- actor/critic architecture must match the checkpoint's saved weights
    train_cfg_dict = dataclass_to_dict(TrainConfig(run_name=args.run))
    env_cfg = EnvConfig(num_envs=args.num_envs, episode_length_s=args.duration)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=not args.headless, device="cpu")

    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=None, device="cpu")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device)

    n, dev = e.num_envs, e.device
    settle_swing = math.radians(args.settle_swing_deg)

    # Per-env running accumulators for the episode in progress
    ep_return = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    init_err = torch.zeros(n, device=dev)
    init_cone = torch.zeros(n, device=dev)
    max_err = torch.zeros(n, device=dev)
    max_cone = torch.zeros(n, device=dev)
    last_err = torch.zeros(n, device=dev)
    last_cone = torch.zeros(n, device=dev)
    last_unsettled = torch.zeros(n, device=dev) # last step index where the settle criteria were violated

    results = {k: [] for k in ("return", "length_s", "crashed", "settled", "settle_time_s", "init_err", "init_swing_deg",
                               "max_err", "max_swing_deg", "final_err", "final_swing_deg")}
    plot_log = {"t": [], "err": [], "err_vec": [], "vel": [], "cone": [], "swing": [], "speed": [], "act": [], "rp": [], "rp_rate": [],
                "drone": [], "payload": [], "target": [], "settle_t": None}
    plot_done = args.no_plot

    def start_episodes(idx):
        sample_disturbance(e, idx, args)
        obs = refresh_obs(e)
        init_err[idx] = torch.norm(e.payload_pos_err[idx], dim=1)
        init_cone[idx] = swing_cone(e)[idx]
        return obs

    e.reset()
    obs = start_episodes(torch.arange(n, device=dev))
    with torch.inference_mode():
        while len(results["return"]) < args.num_episodes:
            obs, rew, dones, extras = e.step(policy(obs))
            dones = dones.bool()

            # Env auto-resets done envs inside step(), so their buffers already hold the new episode -- only accumulate alive envs
            alive = ~dones
            err = torch.norm(e.payload_pos_err, dim=1)
            cone = swing_cone(e)
            speed = torch.norm(e.payload.get_vel(), dim=1)
            unsettled = (err > args.settle_pos) | (cone > settle_swing) | (speed > args.settle_vel)

            ep_return += rew
            ep_len += 1
            max_err = torch.where(alive, torch.maximum(max_err, err), max_err)
            max_cone = torch.where(alive, torch.maximum(max_cone, cone), max_cone)
            last_err = torch.where(alive, err, last_err)
            last_cone = torch.where(alive, cone, last_cone)
            last_unsettled = torch.where(alive & unsettled, ep_len, last_unsettled)

            if not plot_done:
                if alive[0]:
                    plot_log["t"].append(ep_len[0].item() * e.dt)
                    plot_log["err"].append(err[0].item())
                    plot_log["err_vec"].append(e.payload_pos_err[0].tolist())
                    plot_log["vel"].append(e.payload.get_vel()[0].tolist())
                    plot_log["cone"].append(cone[0].item())
                    plot_log["swing"].append(e.payload_swing_angles[0].tolist())
                    plot_log["speed"].append(speed[0].item())
                    plot_log["act"].append(e.actions[0].tolist())
                    plot_log["rp"].append(e.payload_roll_pitch[0].tolist())
                    plot_log["rp_rate"].append(e.payload_roll_pitch_rates[0].tolist())
                    plot_log["drone"].append(e.drone.get_pos()[0].tolist())
                    plot_log["payload"].append(e.payload.get_pos()[0].tolist())
                    plot_log["target"].append(e.commands[0].tolist())

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) == 0:
                continue

            crashed = dones & ~extras["time_outs"].bool()
            live_steps = ep_len - 1 # the step that triggered the done is already the reset state
            for i in done_idx.tolist():
                settled = (not bool(crashed[i])) and last_unsettled[i].item() < live_steps[i].item()
                ep = {
                    "return": ep_return[i].item(),
                    "length_s": ep_len[i].item() * e.dt,
                    "crashed": bool(crashed[i]),
                    "settled": settled,
                    "settle_time_s": last_unsettled[i].item() * e.dt if settled else float("nan"),
                    "init_err": init_err[i].item(),
                    "init_swing_deg": math.degrees(init_cone[i].item()),
                    "max_err": max_err[i].item(),
                    "max_swing_deg": math.degrees(max_cone[i].item()),
                    "final_err": last_err[i].item(),
                    "final_swing_deg": math.degrees(last_cone[i].item()),
                }
                for k, v in ep.items():
                    results[k].append(v)
                print(f"[ep {len(results['return']):4d}] {'CRASH  ' if ep['crashed'] else 'timeout'}  len={ep['length_s']:5.2f}s  "
                      f"init: err={ep['init_err']:.2f}m swing={ep['init_swing_deg']:4.1f}deg  |  "
                      f"max: err={ep['max_err']:.2f}m swing={ep['max_swing_deg']:4.1f}deg  |  "
                      f"final: err={ep['final_err']:.3f}m swing={ep['final_swing_deg']:4.1f}deg  |  "
                      f"settle={ep['settle_time_s']:.2f}s")

                if i == 0 and not plot_done:
                    plot_log["settle_t"] = ep["settle_time_s"]
                    save_plot(plot_log, os.path.join(os.path.dirname(ckpt_path), "eval_stabilize.png"), args, e.cfg)
                    save_motion_plot(plot_log, os.path.join(os.path.dirname(ckpt_path), "eval_stabilize_motion.png"))
                    plot_done = True

            for buf in (ep_return, ep_len, max_err, max_cone, last_err, last_cone, last_unsettled):
                buf[done_idx] = 0.0
            obs = start_episodes(done_idx)

    # Summary
    res = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    settled = res["settled"].bool()
    print("\n========== STABILIZATION EVAL SUMMARY ==========")
    print(f"checkpoint        : {ckpt_path}")
    print(f"disturbance       : swing <= {args.swing_deg:.0f}deg, payload vel <= {args.payload_vel:.1f} m/s, drone rate <= {args.drone_rate:.1f} rad/s")
    print(f"settle criteria   : err < {args.settle_pos} m, swing < {args.settle_swing_deg} deg, speed < {args.settle_vel} m/s (held to end)")
    print(f"episodes          : {len(results['return'])}")
    print(f"crash rate        : {res['crashed'].mean().item() * 100:.1f}%")
    print(f"settle rate       : {settled.float().mean().item() * 100:.1f}%")
    if settled.any():
        st = res["settle_time_s"][settled]
        print(f"settle time       : {st.mean().item():.2f} s mean, {st.max().item():.2f} s worst")
    print(f"initial err/swing : {res['init_err'].mean().item():.2f} m / {res['init_swing_deg'].mean().item():.1f} deg")
    print(f"max err/swing     : {res['max_err'].mean().item():.2f} m / {res['max_swing_deg'].mean().item():.1f} deg")
    print(f"final err/swing   : {res['final_err'].mean().item():.3f} m / {res['final_swing_deg'].mean().item():.1f} deg")
    print(f"return            : {res['return'].mean().item():.2f} +- {res['return'].std().nan_to_num().item():.2f}")


if __name__ == "__main__":
    main()
