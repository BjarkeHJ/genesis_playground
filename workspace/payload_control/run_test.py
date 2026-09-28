import genesis as gs
import torch
import env
from config_env import *

def run():
    gs.init(backend=gs.cuda)

    env_cfg = EnvConfig()
    e = env.PayloadControlEnv(env_cfg=env_cfg, show_viewer=True, device="cuda")

    actions = torch.zeros((e.num_envs, 4), device=e.device)
    actions[:,0] = 0.0
    actions[:,3] = 1

    for i in range(5000):
        e.step(actions)


if __name__ == "__main__":
    run()