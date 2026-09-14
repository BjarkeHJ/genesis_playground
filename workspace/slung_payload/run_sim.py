import genesis as gs

import slung_payload_env as spe
import config as cfg

def run():
    gs.init(backend=gs.cuda)

    # Config
    num_envs = 1
    env_cfg = cfg.EnvConfig()
    cable_cfg = cfg.CableConfig()

    # Environment
    env = spe.SlungPayloadEnv(num_envs=num_envs, env_cfg=env_cfg, cable_cfg=cable_cfg, show_viewer=True)

    # Step sim
    actions = []
    for i in range(env_cfg.max_sim_step_n):
        env.step(actions)

if __name__ == "__main__": 
    run()