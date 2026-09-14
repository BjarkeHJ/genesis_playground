import genesis as gs

import slung_payload_env as spe
import config as cfg

gs.init(backend=gs.cuda)

env_cfg = cfg.EnvConfig()
cable_cfg = cfg.CableConfig()

env = spe.SlungPayloadEnv(num_envs=1, env_cfg=env_cfg, cable_cfg=cable_cfg, show_viewer=True)

actions = []

for i in range(env_cfg.max_sim_step_n):
    env.step_sim(actions)