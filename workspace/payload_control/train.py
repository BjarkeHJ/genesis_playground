import os
import argparse
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from env import *
from config.config_training import TrainConfig
from config.config_env import EnvConfig
from config.paths import LOG_DIR
from config.run_config import snapshot_configs, resolve_ckpt, dataclass_to_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run name under payload_control/logs")
    parser.add_argument("--resume", action="store_true", help="Continue training from a checkpoint")
    parser.add_argument("--load_run", type=str, default=None, help="Run to load from (default: --run)")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.run == None:
        raise ValueError("Run argument is required - Add with --run <name>")

    # gs.init(backend=gs.cuda, performance_mode=True)
    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)

    train_cfg = TrainConfig(run_name=args.run)
    train_cfg_dict = dataclass_to_dict(train_cfg)

    env_cfg = EnvConfig(num_envs=args.num_envs)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=not args.headless, device="cuda")

    log_dir = os.path.join(LOG_DIR, train_cfg.run_name or "model")
    snapshot_configs(log_dir) # eval scripts load these copies (run_config.use_run_configs)
    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=log_dir, device="cuda")

    if args.resume:
        ckpt_path = resolve_ckpt(args.load_run or args.run, args.ckpt)
        print(f"Resuming from {ckpt_path}")
        runner.load(ckpt_path) # restores policy, optimizer and iteration counter
        runner.current_learning_iteration += 1 # step one forward instead of redoing the last iter of the previous

    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
