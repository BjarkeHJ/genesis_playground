import os
import argparse
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from spc_env import SlungPayloadControlEnv
from config.paths import MODELS_DIR
from config import EnvConfig, TrainConfig 

from utils.dataclass_helper import dataclass_to_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Training run name")
    parser.add_argument("--iters", type=int, default=None, help="Number of training iterations")
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--resume", action="store_true", help="Continue from 'run' checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Iteration number or path to model_*.pt (default: latest)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.run == None or args.run == "":
        raise ValueError("Run name argument is required; add with --run <name>")
    if args.iters == None:
        raise ValueError("Number of training iterations is required; add with --iters <n>")

    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)

    tcfg = TrainConfig(run_nam=args.run)
    tcfg = dataclass_to_dict(tcfg)

    ecfg = EnvConfig()
    e = SlungPayloadControlEnv(num_envs=args.num_envs, env_cfg=ecfg, show_viever=not args.headless, device="cuda")

    log_dir = os.path.join(MODELS_DIR, args.run)

    # Dump config and store alongside trained model
    
    runner = OnPolicyRunner(e, tcfg, log_dir=log_dir, device="cuda")

    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()