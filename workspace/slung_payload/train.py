import os

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from slung_payload_env import *
from config import *


def main():
    gs.init(backend=gs.cuda, logging_level="warning")

    train_config = TrainConfig(run_name="test")
    train_config_dict = dataclass_to_dict(train_config)

    env_config = EnvConfig(num_envs=4096*2)
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

    log_dir = os.path.join("logs", train_config.run_name or "slung_payload")
    runner = OnPolicyRunner(env, train_config_dict, log_dir=log_dir, device="cuda")
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
