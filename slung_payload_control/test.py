import genesis as gs
import torch
from config.config_env import EnvConfig
from spc_env import SlungPayloadControlEnv

def main():

    gs.init(backend=gs.cuda, logging_level="info", performance_mode=True)

    env_cfg = EnvConfig()
    e = SlungPayloadControlEnv(1, env_cfg, show_viever=True)
    e.toggle_cam_follow()

    actions = torch.tensor([[0.1, 0.0, 0.0, 0.0]], device=e.device, dtype=gs.tc_float)
    for i in range(1000):
        e.step(actions)    
    
if __name__ == "__main__":
    main()
    print("ok")