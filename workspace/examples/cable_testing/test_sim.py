import genesis as gs
import drone_payload_env as dpe

gs.init(backend=gs.cuda)

env = dpe.DronePayloadEnv(num_envs=1, env_cfg=None, obs_cfg=None, reward_cfg=None, target_cfg=None, show_viewer=True)


