import argparse
import os
import sys

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # payload_control/ (env, utils, configs)
from env import *
from utils import *
from config_training import TrainConfig
from config_env import EnvConfig

# Moving-waypoint tracking eval: the payload target follows a tilted 3D figure-8 around the payload spawn point.
#   x = A sin(wt), y = B sin(2wt), z = C sin(wt/2)   (starts at the payload, peak speed ~2 m/s, peak accel ~1.7 m/s² at defaults)
# Yaw reference: held at the reset payload heading ("fixed") or aligned with the path direction ("tangent").
# The policy was trained on static targets, so this tests how well its learned approach generalizes to a moving goal.

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

    def tangent_yaw(self, t, heading, h=0.05):
        # Horizontal direction of travel (central difference); falls back to the pattern heading at t=0 where the path is at rest
        v = self(t + h, heading) - self(torch.clamp(t - h, min=0.0), heading)
        moving = torch.norm(v[:, :2], dim=1) > 1e-6
        return torch.where(moving, torch.atan2(v[:, 1], v[:, 0]), heading)

def set_targets(e, pos, yaw):
    # Move the command [x, y, z, yaw_ref] for all envs and refresh the observation the policy is about to act on
    e.commands[:, :3] = pos
    e.commands[:, 3] = wrap_angle(yaw)
    e.payload_pos_err[:] = rotate_to_heading(e.commands[:, :3] - e.payload.get_pos(), e.drone_yaw)
    e.payload_yaw_err[:] = wrap_angle(e.commands[:, 3] - quat_yaw(e.payload.get_quat()))
    e._draw_target(torch.zeros(1, device=e.device, dtype=torch.long))
    e._update_observations()
    return e.get_observations()

def payload_roll_pitch(e):
    # ZYX roll/pitch from gravity in payload body axes g_b = [sin(p), -sin(r)cos(p), -cos(r)cos(p)] (yaw-independent)
    g = e.payload_proj_g
    return torch.stack([torch.atan2(-g[:, 1], -g[:, 2]), torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))], dim=-1)

def draw_reference(e, traj, duration):
    t = torch.arange(0.0, duration, 0.1, device=e.device)
    pts = traj(t, torch.zeros_like(t)).cpu().numpy()
    try:
        e.scene.draw_debug_spheres(pts, radius=0.02, color=(0.1, 0.1, 0.1, 0.6))
    except Exception as ex:
        print(f"(could not draw reference path: {ex})")

def build_paths(traj, t_path, heading):
    # Densely sampled reference path per env: (k, M, 3), used for the timing-free contour error
    k, m = len(heading), len(t_path)
    return traj(t_path.repeat(k), heading.repeat_interleave(m)).view(k, m, 3)

def estimate_lag(pl, ref, dt, max_lag_s):
    # Time shift tau minimizing mean |payload(t) - ref(t - tau)|  ->  (tau [s], mean error at that shift [m])
    # pl, ref: (L, 3) histories sampled every dt
    L = pl.shape[0]
    best_k, best_err = 0, torch.norm(pl - ref, dim=1).mean().item()
    for k in range(1, min(int(round(max_lag_s / dt)), L // 2) + 1):
        err = torch.norm(pl[k:] - ref[:L - k], dim=1).mean().item()
        if err < best_err:
            best_k, best_err = k, err
    return best_k * dt, best_err

def save_plot(log, path, tilt_limit):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(log["t"])
    ref, pl = np.asarray(log["ref"]), np.asarray(log["payload"])

    fig = plt.figure(figsize=(15, 13), layout="constrained")
    ax3d = fig.add_subplot(3, 2, 1, projection="3d")
    ax3d.plot(*ref.T, "g--", label="target")
    ax3d.plot(*pl.T, "b", label="payload")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.legend()
    ax3d.set_title("Path")

    ax = fig.add_subplot(3, 2, 2)
    for i, (name, col) in enumerate(zip("xyz", "rgb")):
        ax.plot(t, ref[:, i], col + "--", lw=1)
        ax.plot(t, pl[:, i], col, lw=1.5, label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pos [m]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Position (dashed = target)")

    ax = fig.add_subplot(3, 2, 3)
    ax.plot(t, np.degrees(log["yaw_ref"]), "k--", lw=1, label="yaw ref")
    ax.plot(t, np.degrees(log["payload_yaw"]), "C0", label="payload yaw")
    ax.plot(t, np.degrees(log["drone_yaw"]), "C3", lw=1, label="drone yaw")
    ax.set_xlabel("t [s]"); ax.set_ylabel("yaw [deg]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Heading")

    ax = fig.add_subplot(3, 2, 4)
    swing = np.degrees(np.asarray(log["swing"]))
    ax.plot(t, swing[:, 0], label="swing x (heading)")
    ax.plot(t, swing[:, 1], label="swing y (heading)")
    ax.plot(t, np.degrees(log["swing_amp"]), "k", lw=1, label="swing amplitude")
    ax.set_xlabel("t [s]"); ax.set_ylabel("angle [deg]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Swing")

    ax = fig.add_subplot(3, 2, 5)
    rp = np.degrees(np.asarray(log["rp"]))
    ax.plot(t, rp[:, 0], label="roll")
    ax.plot(t, rp[:, 1], label="pitch")
    for sgn in (1, -1):
        ax.axhline(sgn * math.degrees(tilt_limit), color="r", ls=":", label="terminate (tilt)" if sgn > 0 else None)
    ax.set_xlabel("t [s]"); ax.set_ylabel("angle [deg]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Payload roll/pitch")

    ax = fig.add_subplot(3, 2, 6)
    act, act_raw = np.asarray(log["act"]), np.asarray(log["act_raw"])
    for i, name in enumerate(("thrust", "roll_sp", "pitch_sp", "yawrate_sp")):
        ax.plot(t, act[:, i], color=f"C{i}", lw=1, label=name)
        ax.plot(t, act_raw[:, i], color=f"C{i}", lw=0.8, ls=":", alpha=0.7) # pre-clip: shows saturation past +-1
    for sgn in (1, -1):
        ax.axhline(sgn, color="k", lw=0.8, ls="--", label="clip (dotted: pre-clip)" if sgn > 0 else None)
    ax.set_xlabel("t [s]"); ax.set_ylabel("action [-]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Actions")

    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved tracking plot: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1, help="Stop after this many finished episodes")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer (use with many envs for stats)")
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile of the per-sim-step code (faster startup, ~3x slower stepping)")
    parser.add_argument("--duration", type=float, default=24.0, help="Episode length [s] (default: two laps)")
    parser.add_argument("--period", type=float, default=12.0, help="Figure-8 lap time [s] (lower = harder)")
    parser.add_argument("--amp", type=float, nargs=3, default=(2.5, 1.5, 1.0), metavar=("AX", "AY", "AZ"), help="Figure-8 amplitudes [m]")
    parser.add_argument("--ramp", type=float, default=2.0, help="Ease-in time for the pattern amplitude [s]")
    parser.add_argument("--rand_heading", action="store_true", help="Randomly rotate the pattern about z each episode")
    parser.add_argument("--yaw_mode", choices=("fixed", "tangent"), default="fixed", help="Payload yaw reference: reset heading or path direction")
    parser.add_argument("--tol", type=float, default=0.3, help="Position tolerance for the time-within-tolerance metric [m]")
    parser.add_argument("--max_lag", type=float, default=3.0, help="Largest time lag searched when estimating the tracking delay [s]")
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the env-0 tracking plot")
    args = parser.parse_args()

    if args.run == None:
        raise ValueError("Run argument is required - Add with --run <name>")


    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)

    # Same TrainConfig as train.py -- actor/critic architecture must match the checkpoint's saved weights
    train_cfg_dict = dataclass_to_dict(TrainConfig(run_name=args.run))
    env_cfg = EnvConfig(num_envs=args.num_envs, episode_length_s=args.duration, torch_compile=not args.no_compile)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=not args.headless, device="cuda")

    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=None, device="cuda")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device)

    n, dev = e.num_envs, e.device
    traj = Figure8(e.payload_init_pos, args.period, *args.amp, ramp_s=args.ramp, device=dev)
    if not args.headless:
        draw_reference(e, traj, args.duration)

    def sample_heading(k):
        return gs_rand_float(-math.pi, math.pi, (k, ), dev) if args.rand_heading else torch.zeros(k, device=dev)
    heading = sample_heading(n)
    yaw0 = quat_yaw(e.payload_init_quat.expand(n, -1)).clone() # reset payload heading, used by the "fixed" yaw mode
    t_path = torch.arange(0.0, args.duration + e.dt, e.dt, device=dev)
    paths = build_paths(traj, t_path, heading)

    def yaw_ref(t):
        return traj.tangent_yaw(t, heading) if args.yaw_mode == "tangent" else yaw0

    # Per-env running accumulators for the episode in progress
    ep_return = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    err_sum = torch.zeros(n, device=dev)
    err_sq_sum = torch.zeros(n, device=dev)
    err_max = torch.zeros(n, device=dev)
    in_tol = torch.zeros(n, device=dev)
    max_swing = torch.zeros(n, device=dev)
    swing_amp_sum = torch.zeros(n, device=dev)
    yaw_err_sum = torch.zeros(n, device=dev)
    contour_sum = torch.zeros(n, device=dev)
    # Per-env histories for the lag estimate
    hist_len = e.max_episode_length + 2
    hist_pl = torch.zeros((n, hist_len, 3), device=dev)
    hist_ref = torch.zeros((n, hist_len, 3), device=dev)

    results = {k: [] for k in ("return", "length_s", "crashed", "mean_err", "rms_err", "max_err", "in_tol_frac", "max_swing_deg",
                               "mean_swing_amp_deg", "mean_yaw_err_deg", "lag_s", "shifted_err", "contour_err")}
    plot_log = {"t": [], "ref": [], "payload": [], "yaw_ref": [], "payload_yaw": [], "drone_yaw": [],
                "swing": [], "swing_amp": [], "rp": [], "act": [], "act_raw": []}
    plot_done = args.no_plot

    obs = e.reset()
    with torch.inference_mode():
        while len(results["return"]) < args.num_episodes:
            # Target at the time this policy step ends, so the post-step error compares like with like
            t_next = (ep_len + 1) * e.dt
            ref = traj(t_next, heading)
            ref_yaw = yaw_ref(t_next)
            obs = set_targets(e, ref, ref_yaw)

            obs, rew, dones, extras = e.step(policy(obs))
            dones = dones.bool()

            # Env auto-resets done envs inside step(), so their buffers already hold the new episode -- only accumulate alive envs
            alive = ~dones
            err = torch.norm(e.payload_pos_err, dim=1)
            yaw_err = torch.abs(e.payload_yaw_err)
            swing = torch.max(torch.abs(e.payload_swing_angles), dim=1).values
            payload_pos = e.payload.get_pos()
            contour = torch.cdist(payload_pos.unsqueeze(1), paths).squeeze(1).min(dim=1).values

            alive_idx = alive.nonzero(as_tuple=False).flatten()
            step_idx = ep_len[alive_idx].long()
            hist_pl[alive_idx, step_idx] = payload_pos[alive_idx]
            hist_ref[alive_idx, step_idx] = ref[alive_idx]

            ep_return += rew
            ep_len += 1
            err_sum = torch.where(alive, err_sum + err, err_sum)
            err_sq_sum = torch.where(alive, err_sq_sum + err**2, err_sq_sum)
            err_max = torch.where(alive, torch.maximum(err_max, err), err_max)
            in_tol = torch.where(alive & (err < args.tol), in_tol + 1, in_tol)
            max_swing = torch.where(alive, torch.maximum(max_swing, swing), max_swing)
            swing_amp_sum = torch.where(alive, swing_amp_sum + e.swing_amp, swing_amp_sum)
            yaw_err_sum = torch.where(alive, yaw_err_sum + yaw_err, yaw_err_sum)
            contour_sum = torch.where(alive, contour_sum + contour, contour_sum)

            if not plot_done and alive[0]:
                plot_log["t"].append(t_next[0].item())
                plot_log["ref"].append(ref[0].tolist())
                plot_log["payload"].append(payload_pos[0].tolist())
                plot_log["yaw_ref"].append(e.commands[0, 3].item())
                plot_log["payload_yaw"].append(quat_yaw(e.payload.get_quat())[0].item())
                plot_log["drone_yaw"].append(e.drone_yaw[0].item())
                plot_log["swing"].append(e.payload_swing_angles[0].tolist())
                plot_log["swing_amp"].append(e.swing_amp[0].item())
                plot_log["rp"].append(payload_roll_pitch(e)[0].tolist())
                plot_log["act"].append(e.actions[0].tolist())
                plot_log["act_raw"].append(e.actions_raw[0].tolist())

            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) == 0:
                continue

            crashed = dones & ~extras["time_outs"].bool()
            live_steps = torch.clamp(ep_len - 1, min=1)
            for i in done_idx.tolist():
                L = int(live_steps[i].item())
                lag_s, shifted_err = estimate_lag(hist_pl[i, :L], hist_ref[i, :L], e.dt, args.max_lag)
                if i == 0 and not plot_done:
                    save_plot(plot_log, os.path.join(os.path.dirname(ckpt_path), "eval_tracking.png"), e.cfg.terminate_if_payload_tilt_greater_than)
                    plot_done = True
                ep = {
                    "return": ep_return[i].item(),
                    "length_s": ep_len[i].item() * e.dt,
                    "crashed": bool(crashed[i]),
                    "mean_err": (err_sum[i] / live_steps[i]).item(),
                    "rms_err": torch.sqrt(err_sq_sum[i] / live_steps[i]).item(),
                    "max_err": err_max[i].item(),
                    "in_tol_frac": (in_tol[i] / live_steps[i]).item(),
                    "max_swing_deg": math.degrees(max_swing[i].item()),
                    "mean_swing_amp_deg": math.degrees((swing_amp_sum[i] / live_steps[i]).item()),
                    "mean_yaw_err_deg": math.degrees((yaw_err_sum[i] / live_steps[i]).item()),
                    "lag_s": lag_s,
                    "shifted_err": shifted_err,
                    "contour_err": (contour_sum[i] / live_steps[i]).item(),
                }
                for k, v in ep.items():
                    results[k].append(v)
                print(f"[ep {len(results['return']):4d}] {'CRASH  ' if ep['crashed'] else 'timeout'}  len={ep['length_s']:5.2f}s  "
                      f"mean_err={ep['mean_err']:.3f}m  lag={ep['lag_s']:.2f}s  shifted_err={ep['shifted_err']:.3f}m  "
                      f"contour_err={ep['contour_err']:.3f}m  max_err={ep['max_err']:.3f}m  yaw_err={ep['mean_yaw_err_deg']:5.1f}deg  "
                      f"max_swing={ep['max_swing_deg']:5.1f}deg")

            for buf in (ep_return, ep_len, err_sum, err_sq_sum, err_max, in_tol, max_swing, swing_amp_sum, yaw_err_sum, contour_sum):
                buf[done_idx] = 0.0
            heading[done_idx] = sample_heading(len(done_idx))
            paths[done_idx] = build_paths(traj, t_path, heading[done_idx])

    # Summary
    res = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    w = traj.w
    v_peak = max(args.amp[0] * w, 2 * args.amp[1] * w, 0.5 * args.amp[2] * w)
    print("\n========== TRACKING EVAL SUMMARY ==========")
    print(f"checkpoint        : {ckpt_path}")
    print(f"trajectory        : figure-8, period={args.period:.1f}s, amp={tuple(args.amp)} m, peak axis speed ~{v_peak:.2f} m/s, yaw={args.yaw_mode}")
    print(f"episodes          : {len(results['return'])}")
    print(f"crash rate        : {res['crashed'].mean().item() * 100:.1f}%")
    print(f"return            : {res['return'].mean().item():.2f} +- {res['return'].std().nan_to_num().item():.2f}")
    print(f"episode length    : {res['length_s'].mean().item():.2f} s")
    print(f"mean track err    : {res['mean_err'].mean().item():.3f} m")
    print(f"rms track err     : {res['rms_err'].mean().item():.3f} m")
    print(f"max track err     : {res['max_err'].mean().item():.3f} m")
    print(f"time within {args.tol} m : {res['in_tol_frac'].mean().item() * 100:.1f}%")
    print(f"tracking lag      : {res['lag_s'].mean().item():.2f} s  (time shift that best aligns payload with target)")
    print(f"lag-removed err   : {res['shifted_err'].mean().item():.3f} m  (shape error once the delay is removed)")
    print(f"contour err       : {res['contour_err'].mean().item():.3f} m  (distance to the reference path, timing ignored)")
    print(f"mean yaw err      : {res['mean_yaw_err_deg'].mean().item():.1f} deg")
    print(f"max swing         : {res['max_swing_deg'].mean().item():.1f} deg")
    print(f"mean swing amp    : {res['mean_swing_amp_deg'].mean().item():.1f} deg  (pendulum energy-equivalent amplitude)")


if __name__ == "__main__":
    main()
