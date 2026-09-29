import argparse
import os

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from env import *
from utils import *
from config_training import TrainConfig
from config_env import EnvConfig

# Moving-waypoint tracking eval: the payload target follows a tilted 3D figure-8 around the payload spawn point.
#   x = A sin(wt), y = B sin(2wt), z = C sin(wt/2)   (starts at the payload, peak speed ~2 m/s, peak accel ~1.7 m/s² at defaults)
# The policy was trained on static targets, so this tests how well the approach law generalizes to a moving goal.

def smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

class Figure8:
    def __init__(self, center, period, amp_x, amp_y, amp_z, ramp_s, device):
        self.center = center
        self.w = 2.0 * math.pi / period
        self.amp = torch.tensor([amp_x, amp_y, amp_z], device=device, dtype=gs.tc_float)
        self.ramp_s = ramp_s

    def __call__(self, t, heading):
        # t: (n,) episode time [s], heading: (n,) yaw of the pattern [rad] -> (n, 3) world target
        wt = self.w * t
        shape = torch.stack([torch.sin(wt), torch.sin(2.0 * wt), torch.sin(0.5 * wt)], dim=-1)
        ramp = smoothstep(t / self.ramp_s).unsqueeze(-1) if self.ramp_s > 0 else 1.0 # ease in so the target doesn't start at full speed
        p = shape * self.amp * ramp
        c, s = torch.cos(heading), torch.sin(heading)
        p = torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], dim=-1)
        return self.center + p

def set_targets(e, pos):
    # Move the target for all envs and refresh the observation the policy is about to act on
    e.commands[:] = pos
    e.target.set_pos(pos, zero_velocity=True)
    e.payload_pos_err[:] = e.commands - e.payload.get_pos()
    e.payload_vel_err[:] = e._desired_vel(e.payload_pos_err) - e.payload.get_vel()
    e._update_observations()
    return e.get_observations()

def draw_reference(e, traj, duration):
    t = torch.arange(0.0, duration, 0.1, device=e.device)
    pts = traj(t, torch.zeros_like(t)).cpu().numpy()
    try:
        e.scene.draw_debug_spheres(pts, radius=0.02, color=(0.1, 0.1, 0.1, 0.6))
    except Exception as ex:
        print(f"(could not draw reference path: {ex})")

def save_plot(log, path, at_target_th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    ref, pl = log["ref"], log["payload"]
    err = [((r[0] - p[0])**2 + (r[1] - p[1])**2 + (r[2] - p[2])**2) ** 0.5 for r, p in zip(ref, pl)]

    fig = plt.figure(figsize=(14, 8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(*zip(*ref), "g--", label="target")
    ax3d.plot(*zip(*pl), "b", label="payload")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.legend()
    ax3d.set_title("Path")

    ax = fig.add_subplot(2, 2, 2)
    for i, (name, col) in enumerate(zip("xyz", "rgb")):
        ax.plot(t, [r[i] for r in ref], col + "--", lw=1)
        ax.plot(t, [p[i] for p in pl], col, lw=1.5, label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pos [m]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Position (dashed = target)")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, err, "k")
    ax.axhline(at_target_th, color="r", ls=":", label=f"{at_target_th} m")
    ax.set_xlabel("t [s]"); ax.set_ylabel("error [m]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Tracking error")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, [math.degrees(s[0]) for s in log["swing"]], label="swing x (heading)")
    ax.plot(t, [math.degrees(s[1]) for s in log["swing"]], label="swing y (heading)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("angle [deg]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Swing angles")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved tracking plot: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="dev_test", help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3, help="Stop after this many finished episodes")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer (use with many envs for stats)")
    parser.add_argument("--duration", type=float, default=24.0, help="Episode length [s] (default: two laps)")
    parser.add_argument("--period", type=float, default=12.0, help="Figure-8 lap time [s] (lower = harder)")
    parser.add_argument("--amp", type=float, nargs=3, default=(2.5, 1.5, 1.0), metavar=("AX", "AY", "AZ"), help="Figure-8 amplitudes [m]")
    parser.add_argument("--ramp", type=float, default=2.0, help="Ease-in time for the pattern amplitude [s]")
    parser.add_argument("--rand_heading", action="store_true", help="Randomly rotate the pattern about z each episode")
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the env-0 tracking plot")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    # gs.init(backend=gs.cuda, logging_level="warning")
    gs.init(backend=gs.cpu)

    # Same TrainConfig as train.py -- actor/critic architecture must match the checkpoint's saved weights
    train_cfg_dict = dataclass_to_dict(TrainConfig(run_name=args.run))
    env_cfg = EnvConfig(num_envs=args.num_envs, episode_length_s=args.duration)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=not args.headless, device="cpu")

    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=None, device="cpu")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device)

    n, dev = e.num_envs, e.device
    traj = Figure8(e.payload_init_pos, args.period, *args.amp, ramp_s=args.ramp, device=dev)
    if not args.headless:
        draw_reference(e, traj, args.duration)

    def sample_heading(k):
        return gs_rand_float(-math.pi, math.pi, (k, ), dev) if args.rand_heading else torch.zeros(k, device=dev)
    heading = sample_heading(n)

    # Per-env running accumulators for the episode in progress
    ep_return = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    err_sum = torch.zeros(n, device=dev)
    err_sq_sum = torch.zeros(n, device=dev)
    err_max = torch.zeros(n, device=dev)
    in_tol = torch.zeros(n, device=dev)
    max_swing = torch.zeros(n, device=dev)

    results = {k: [] for k in ("return", "length_s", "crashed", "mean_err", "rms_err", "max_err", "in_tol_frac", "max_swing_deg")}
    plot_log = {"t": [], "ref": [], "payload": [], "swing": []}
    plot_done = args.no_plot

    obs = e.reset()
    with torch.inference_mode():
        while len(results["return"]) < args.num_episodes:
            # Target at the time this policy step ends, so the post-step error compares like with like
            t_next = (ep_len + 1) * e.dt
            ref = traj(t_next, heading)
            obs = set_targets(e, ref)

            obs, rew, dones, extras = e.step(policy(obs))
            dones = dones.bool()

            # Env auto-resets done envs inside step(), so their buffers already hold the new episode -- only accumulate alive envs
            alive = ~dones
            err = torch.norm(e.payload_pos_err, dim=1)
            swing = torch.max(torch.abs(e.payload_swing_angles), dim=1).values

            ep_return += rew
            ep_len += 1
            err_sum = torch.where(alive, err_sum + err, err_sum)
            err_sq_sum = torch.where(alive, err_sq_sum + err**2, err_sq_sum)
            err_max = torch.where(alive, torch.maximum(err_max, err), err_max)
            in_tol = torch.where(alive & (err < e.cfg.at_target_th), in_tol + 1, in_tol)
            max_swing = torch.where(alive, torch.maximum(max_swing, swing), max_swing)

            if not plot_done:
                if alive[0]:
                    plot_log["t"].append(t_next[0].item())
                    plot_log["ref"].append(ref[0].tolist())
                    plot_log["payload"].append(e.payload.get_pos()[0].tolist())
                    plot_log["swing"].append(e.payload_swing_angles[0].tolist())
                else:
                    save_plot(plot_log, os.path.join(os.path.dirname(ckpt_path), "eval_tracking.png"), e.cfg.at_target_th)
                    plot_done = True

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) == 0:
                continue

            crashed = dones & ~extras["time_outs"].bool()
            live_steps = torch.clamp(ep_len - 1, min=1)
            for i in done_idx.tolist():
                ep = {
                    "return": ep_return[i].item(),
                    "length_s": ep_len[i].item() * e.dt,
                    "crashed": bool(crashed[i]),
                    "mean_err": (err_sum[i] / live_steps[i]).item(),
                    "rms_err": torch.sqrt(err_sq_sum[i] / live_steps[i]).item(),
                    "max_err": err_max[i].item(),
                    "in_tol_frac": (in_tol[i] / live_steps[i]).item(),
                    "max_swing_deg": math.degrees(max_swing[i].item()),
                }
                for k, v in ep.items():
                    results[k].append(v)
                print(f"[ep {len(results['return']):4d}] return={ep['return']:8.2f}  len={ep['length_s']:5.2f}s  "
                      f"{'CRASH' if ep['crashed'] else 'timeout'}  mean_err={ep['mean_err']:.3f}m  rms_err={ep['rms_err']:.3f}m  "
                      f"max_err={ep['max_err']:.3f}m  in_tol={ep['in_tol_frac'] * 100:5.1f}%  max_swing={ep['max_swing_deg']:5.1f}deg")

            for buf in (ep_return, ep_len, err_sum, err_sq_sum, err_max, in_tol, max_swing):
                buf[done_idx] = 0.0
            heading[done_idx] = sample_heading(len(done_idx))

    # Summary
    res = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    w = traj.w
    v_peak = max(args.amp[0] * w, 2 * args.amp[1] * w, 0.5 * args.amp[2] * w)
    print("\n========== TRACKING EVAL SUMMARY ==========")
    print(f"checkpoint        : {ckpt_path}")
    print(f"trajectory        : figure-8, period={args.period:.1f}s, amp={tuple(args.amp)} m, peak axis speed ~{v_peak:.2f} m/s")
    print(f"episodes          : {len(results['return'])}")
    print(f"crash rate        : {res['crashed'].mean().item() * 100:.1f}%")
    print(f"return            : {res['return'].mean().item():.2f} +- {res['return'].std().nan_to_num().item():.2f}")
    print(f"episode length    : {res['length_s'].mean().item():.2f} s")
    print(f"mean track err    : {res['mean_err'].mean().item():.3f} m")
    print(f"rms track err     : {res['rms_err'].mean().item():.3f} m")
    print(f"max track err     : {res['max_err'].mean().item():.3f} m")
    print(f"time within {e.cfg.at_target_th} m : {res['in_tol_frac'].mean().item() * 100:.1f}%")
    print(f"max swing         : {res['max_swing_deg'].mean().item():.1f} deg")


if __name__ == "__main__":
    main()
