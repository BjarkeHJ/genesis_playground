import argparse

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from slung_payload_env import *
from config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a saved model_*.pt checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2000)
    args = parser.parse_args()

    gs.init(backend=gs.cuda)

    # Same TrainConfig as train.py -- actor/critic architecture must match the checkpoint's saved weights.
    train_config = TrainConfig()
    train_config_dict = dataclass_to_dict(train_config)

    env_config = EnvConfig(num_envs=args.num_envs)
    obs_config = ObservationConfig()
    command_cfg = CommandConfig()
    reward_cfg = RewardConfig()
    tether_config = TetherConfig()

    env = SlungPayloadEnv(
        env_cfg=env_config,
        obs_cfg=obs_config,
        command_cfg=command_cfg,
        reward_cfg=reward_cfg,
        tether_cfg=tether_config,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_config_dict, log_dir=None, device="cuda")
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args.num_steps):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)


if __name__ == "__main__":
    main()
