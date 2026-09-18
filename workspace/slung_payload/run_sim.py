import genesis as gs
import torch

import slung_payload_env as spe
from config import *

def run():
    gs.init(backend=gs.cuda)

    # Config
    env_cfg = EnvConfig()
    obs_cfg = ObservationConfig()
    command_cfg = CommandConfig()
    reward_cfg = RewardConfig()
    tether_cfg = TetherConfig()

    # Environment
    env = spe.SlungPayloadEnv(env_cfg=env_cfg, obs_cfg=obs_cfg, command_cfg=command_cfg, reward_cfg=reward_cfg, tether_cfg=tether_cfg, show_viewer=True)

    # Constant-thrust, zero-rate flight test: sanity-checks the mixer/rate-PID against
    # the sim physics without any policy in the loop. Thrust is set to hover the full
    # drone+payload weight; watch for level, non-diverging flight in the viewer.
    hover_thrust = env.total_mass * 9.81
    thrust_action = 2.0 * (hover_thrust / env.thrust_max) - 1.0  # inverse of the [-1,1] -> [0, thrust_max] map in step()
    actions = torch.zeros((env.num_envs, env_cfg.num_actions), device=env.device)
    actions[:, 0] = thrust_action

    # Step sim
    for i in range(env_cfg.max_sim_step_n):
        env.step(actions)

if __name__ == "__main__":
    run()
