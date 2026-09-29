import argparse

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from env import *
from utils import *
from config_training import TrainConfig
from config_env import EnvConfig

def set_targets(e, idx, hard=True):
    if hard:
        n = len(idx)
        e.commands[idx, 0] = gs_rand_float(-3.0, 3.0, (n,), e.device)
        e.commands[idx, 1] = gs_rand_float(-3.0, 3.0, (n,), e.device)
        e.commands[idx, 2] = gs_rand_float(2.0, 7.0, (n,), e.device)
        e.target.set_pos(e.commands[idx], zero_velocity=True, envs_idx=idx)
    else:
        e._resample_commands(idx) # training distribution
    # Env already computed errors/obs for its own target on reset -- recompute them for the new one
    e.payload_pos_err[idx] = e.commands[idx] - e.payload.get_pos(idx)
    e.payload_vel_err[idx] = e._desired_vel(e.payload_pos_err[idx]) - e.payload.get_vel(idx)
    e._update_observations()
    return e.get_observations()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="test", help="Run name under payload_control/logs")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=10, help="Stop after this many finished episodes")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer (use with many envs for stats)")
    parser.add_argument("--hard_targets", action="store_true", help="Override env targets with the wider set_targets() distribution")
    parser.add_argument("--switch_every", type=float, default=None, help="Resample the target every N seconds mid-episode (moving waypoints)")
    args = parser.parse_args()

    ckpt_path = resolve_ckpt(args.run, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    gs.init(backend=gs.cuda, logging_level="warning")

    # Same TrainConfig as train.py -- actor/critic architecture must match the checkpoint's saved weights
    train_cfg_dict = dataclass_to_dict(TrainConfig(run_name=args.run))
    env_cfg = EnvConfig(num_envs=args.num_envs)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=not args.headless, device="cuda")

    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=None, device="cuda")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=e.device)

    # Per-env running accumulators for the episode in progress
    n, dev = e.num_envs, e.device
    ep_return = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    dist_sum = torch.zeros(n, device=dev)
    last_dist = torch.zeros(n, device=dev)
    max_swing = torch.zeros(n, device=dev)
    time_to_target = torch.full((n, ), float("nan"), device=dev)

    results = {k: [] for k in ("return", "length_s", "crashed", "final_dist", "mean_dist", "max_swing_deg", "time_to_target_s")}

    switch_steps = max(1, round(args.switch_every / e.dt)) if args.switch_every else None

    obs = e.reset()
    if args.hard_targets:
        obs = set_targets(e, torch.arange(n, device=dev))
    with torch.inference_mode():
        while len(results["return"]) < args.num_episodes:
            obs, rew, dones, extras = e.step(policy(obs))
            dones = dones.bool()

            # Env auto-resets done envs inside step(), so their state buffers already hold the new episode.
            # Only accumulate state metrics for envs that are still running (last_dist is then from the step before termination).
            alive = ~dones
            dist = torch.norm(e.payload_pos_err, dim=1)
            swing = torch.max(torch.abs(e.payload_swing_angles), dim=1).values

            ep_return += rew
            ep_len += 1
            dist_sum = torch.where(alive, dist_sum + dist, dist_sum)
            last_dist = torch.where(alive, dist, last_dist)
            max_swing = torch.where(alive, torch.maximum(max_swing, swing), max_swing)
            reached = alive & torch.isnan(time_to_target) & (dist < e.cfg.at_target_th)
            time_to_target = torch.where(reached, ep_len * e.dt, time_to_target)

            if switch_steps is not None:
                switch_idx = (alive & (ep_len % switch_steps == 0)).nonzero(as_tuple=False).flatten()
                if len(switch_idx) > 0:
                    obs = set_targets(e, switch_idx, hard=args.hard_targets)

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
                    "final_dist": last_dist[i].item(),
                    "mean_dist": (dist_sum[i] / live_steps[i]).item(),
                    "max_swing_deg": math.degrees(max_swing[i].item()),
                    "time_to_target_s": time_to_target[i].item(),
                }
                for k, v in ep.items():
                    results[k].append(v)
                print(f"[ep {len(results['return']):4d}] return={ep['return']:8.2f}  len={ep['length_s']:5.2f}s  "
                      f"{'CRASH' if ep['crashed'] else 'timeout'}  final_dist={ep['final_dist']:.3f}m  "
                      f"mean_dist={ep['mean_dist']:.3f}m  max_swing={ep['max_swing_deg']:5.1f}deg  "
                      f"t_target={ep['time_to_target_s']:.2f}s")

            for buf in (ep_return, ep_len, dist_sum, last_dist, max_swing):
                buf[done_idx] = 0.0
            time_to_target[done_idx] = float("nan")

            if args.hard_targets:
                obs = set_targets(e, done_idx)

    # Summary
    res = {k: torch.tensor(v, dtype=torch.float32) for k, v in results.items()}
    success = (~res["crashed"].bool()) & (res["final_dist"] < e.cfg.at_target_th)
    reached = ~torch.isnan(res["time_to_target_s"])
    print("\n========== EVAL SUMMARY ==========")
    print(f"checkpoint        : {ckpt_path}")
    print(f"episodes          : {len(results['return'])}")
    print(f"crash rate        : {res['crashed'].mean().item() * 100:.1f}%")
    print(f"success rate      : {success.float().mean().item() * 100:.1f}%  (no crash, final dist < {e.cfg.at_target_th} m)")
    print(f"return            : {res['return'].mean().item():.2f} +- {res['return'].std().nan_to_num().item():.2f}")
    print(f"episode length    : {res['length_s'].mean().item():.2f} s")
    print(f"final dist        : {res['final_dist'].mean().item():.3f} m")
    print(f"mean dist         : {res['mean_dist'].mean().item():.3f} m")
    print(f"max swing         : {res['max_swing_deg'].mean().item():.1f} deg")
    if reached.any():
        print(f"time to target    : {res['time_to_target_s'][reached].mean().item():.2f} s  ({reached.float().mean().item() * 100:.1f}% reached)")
    else:
        print("time to target    : never reached")


if __name__ == "__main__":
    main()
