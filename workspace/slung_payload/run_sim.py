import genesis as gs
import slung_payload_env as spe

gs.init(backend=gs.cuda)

env = spe.SlungPayloadEnv(num_envs=1, show_viewer=True)

for i in range(5000):
    env.step_sim()