import os
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from env import *
from utils import *
from config_training import TrainConfig
from config_env import EnvConfig

def main():
    gs.init(backend=gs.cuda, logging_level="warning")
    # gs.init(backend=gs.cuda)

    train_cfg = TrainConfig(run_name="test")
    train_cfg_dict = dataclass_to_dict(train_cfg)

    env_cfg = EnvConfig(num_envs=8192)
    e = PayloadControlEnv(env_cfg=env_cfg, show_viewer=True, device="cuda")

    log_dir = os.path.join(SCRIPT_DIR, "logs", train_cfg.run_name or "model")
    runner = OnPolicyRunner(e, train_cfg_dict, log_dir=log_dir, device="cuda")
    runner.learn(num_learning_iterations=256, init_at_random_ep_len=True)
    
if __name__ == "__main__":
    main()