import sys
import os
import argparse
from dataclasses import replace
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import MODELS_DIR
from config.config_env import EnvConfig
from utils.config_helpers import load_yaml, dict_to_dataclass, resolve_ckpt
from spc_env import SlungPayloadControlEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Which training 'run' to export")
    parser.add_argument("--ckpt", type=int, default=None, help="Chose ckpt for model - defaults to latest")
    args = parser.parse_args()

    if args.run is None:
        raise ValueError("Run name argument is required; add with --run <name>")

    run_dir = os.path.join(MODELS_DIR, args.run)
    model_path = resolve_ckpt(MODELS_DIR, args.run, args.ckpt)
    
    cfg_path = os.path.join(run_dir, "config")
    tcfg = load_yaml(os.path.join(cfg_path, "tcfg.yaml"))
    ecfg = dict_to_dataclass(EnvConfig, load_yaml(os.path.join(cfg_path, "ecfg.yaml")))
    ecfg = replace(ecfg, torch_compile=False)
    
    # Star environment
    gs.init(backend=gs.cpu, logging_level="info")
    env = SlungPayloadControlEnv(num_envs=1, env_cfg=ecfg, show_viever=False, device="cpu")

    # Load the trained model
    runner = OnPolicyRunner(env, tcfg, log_dir=None, device="cpu")
    runner.load(model_path, map_location="cpu")

    out_dir = os.path.join(run_dir, "exported")
    runner.export_policy_to_onnx(out_dir, filename="policy.onnx")
    print(f"Exported -> {out_dir}/policy.onnx")

if __name__ == "__main__":
    main()