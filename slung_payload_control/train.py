import os
import argparse
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from spc_env import SlungPayloadControlEnv
from config.paths import MODELS_DIR
from config import EnvConfig, TrainConfig 

from slung_payload_control.utils.config_helpers import dataclass_to_dict, dict_to_dataclass, save_yaml, load_yaml, resolve_ckpt

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
    
    log_dir = os.path.join(MODELS_DIR, args.run)
    cfg_path = os.path.join(log_dir, "config")
    ckpt_path = None
    if args.resume:
        # Resume from existing model (optional: at a ckpt, default: latest)
        ckpt_path = resolve_ckpt(MODELS_DIR, args.run, args.ckpt)
        tcfg_path = os.path.join(cfg_path, "tcfg.yaml")
        ecfg_path = os.path.join(cfg_path, "ecfg.yaml")
        tcfg = load_yaml(tcfg_path)
        ecfg = dict_to_dataclass(EnvConfig, load_yaml(ecfg_path)) # construct the EnvConfig dataclass from .yaml file
    else:
        tcfg = TrainConfig(run_nam=args.run)
        tcfg = dataclass_to_dict(tcfg)
        ecfg = EnvConfig() # Use live config values
        # Dump configs and store alongside trained model (for eval / resumed training)
        save_yaml(dataclass_to_dict(ecfg), cfg_path, "ecfg.yaml")
        save_yaml(tcfg, cfg_path, "tcfg.yaml")

    env = SlungPayloadControlEnv(num_envs=args.num_envs, env_cfg=ecfg, show_viever=not args.headless, device="cuda")

    runner = OnPolicyRunner(env, tcfg, log_dir=log_dir, device="cuda")
    if ckpt_path is not None:
        print(f"Resumed Training: loaded checkpoint {ckpt_path}. Continues training with stored configuration")
        runner.load(ckpt_path)
        runner.learn(num_learning_iterations=args.iters+1, init_at_random_ep_len=True) # runner starts from last iter of prev run, so end +1 later (keeps ckpts clean)
    else:
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
        
if __name__ == "__main__":
    main()