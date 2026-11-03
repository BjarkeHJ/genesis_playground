import genesis as gs

import math
import torch

from config.config_env import EPS
from .math_utils import yaw_to_quat, gs_rand_float, rand_uniform, NOMINAL_SCALE

class DomainRandomizer:
    # Per-episode randomization of the system and the initial state, on while enabled (else nominal). Owns the body masses;
    # tether and actuation parameters live in the models that use them and are resampled through their randomize().
    # Writes nothing to the simulator: the env applies the masses
    def __init__(self, env_cfg, tether, ctrl, drone_mass0, payload_mass0, gravity, device):
        # drone_mass0, payload_mass0 (nominal entity masses) and gravity: from the built scene
        self.cfg = env_cfg.dr_cfg
        self.tether = tether
        self.ctrl = ctrl
        self.gravity = gravity
        self.device = device
        self.enabled = self.cfg.enabled # runtime copy of the config switch, evals turn it off for nominal params and initial state

        self.drone_mass0, self.payload_mass0 = drone_mass0, payload_mass0
        self.payload_init_pos = torch.tensor(env_cfg.sys_cfg.payload_reset_pos, device=device, dtype=gs.tc_float)
        full = lambda v: torch.full((env_cfg.num_envs, ), v, device=device, dtype=gs.tc_float)
        self.drone_mass = full(drone_mass0)
        self.payload_mass = full(payload_mass0)
        self.tether.mass_eff[:] = 1.0 / (1.0 / drone_mass0 + 1.0 / payload_mass0)
        self.g_eff = (1.0 + self.payload_mass / self.drone_mass) * gravity # swing-mode gravity, see PayloadControlEnv._compute_buffers

    def randomize(self, envs_idx):
        # Resample all system parameters of envs_idx
        self.tether.randomize(envs_idx, self.enabled)

        n = len(envs_idx)
        pm = self.payload_mass0 * rand_uniform(self.cfg.payload_mass_scale if self.enabled else NOMINAL_SCALE, n, self.device)
        dm = self.drone_mass0 * rand_uniform(self.cfg.drone_mass_scale if self.enabled else NOMINAL_SCALE, n, self.device)
        self.payload_mass[envs_idx] = pm
        self.drone_mass[envs_idx] = dm
        self.tether.mass_eff[envs_idx] = 1.0 / (1.0 / dm + 1.0 / pm)
        self.g_eff[envs_idx] = (1.0 + pm / dm) * self.gravity

        self.ctrl.randomize(envs_idx, self.enabled)

    def sample_initial_state(self, envs_idx, payload_yaw=None):
        # -> drone (pos, quat, lin vel), payload (pos, quat, lin vel) of envs_idx, with tethers slightly taut. Payload at
        # payload_init_pos (the env moves the state to its start point). payload_yaw (n, ): payload heading, the drone's
        # within init_rel_yaw of it; None: uniform heading
        dr, n = self.cfg, len(envs_idx)
        lvl = 1.0 if self.enabled else 0.0 # scales every initial perturbation (0: nominal hover at rest)
        L = self.tether.rest_length[envs_idx]

        # Swing: r = drone->payload, built so atan2(r_x, -r_z) = theta_x
        th = gs_rand_float(-dr.init_swing_angle * lvl, dr.init_swing_angle * lvl, (n, 2), self.device)
        r_hat = torch.stack([torch.tan(th[:, 0]), torch.tan(th[:, 1]), -torch.ones_like(L)], dim=-1)
        r_hat = r_hat / torch.norm(r_hat, dim=-1, keepdim=True)
        payload_pos = self.payload_init_pos.expand(n, -1)
        attach_dz = 0.045
        drone_pos = payload_pos - r_hat * (L + attach_dz + 0.01).unsqueeze(-1) # slightly taut

        # Heading
        rel_yaw = gs_rand_float(-dr.init_rel_yaw * lvl, dr.init_rel_yaw * lvl, (n, ), self.device)
        if payload_yaw is None:
            yaw_d = gs_rand_float(-math.pi, math.pi, (n, ), self.device)
            yaw_p = yaw_d + rel_yaw
        else:
            yaw_p = payload_yaw
            yaw_d = yaw_p - rel_yaw

        # Velocities
        v_d = dr.init_drone_vel * lvl * torch.randn((n, 3), device=self.device)
        u = torch.randn((n, 3), device=self.device)
        u = u - (u * r_hat).sum(-1, keepdim=True) * r_hat
        v_rel = u / torch.clamp(torch.norm(u, dim=-1, keepdim=True), min=EPS) * (dr.init_swing_rate * lvl * L).unsqueeze(-1) * torch.rand((n, 1), device=self.device)
        v_p = v_d + v_rel

        return drone_pos, yaw_to_quat(yaw_d), v_d, payload_pos, yaw_to_quat(yaw_p), v_p

    def privileged_obs(self):
        # (N, 8) randomized parameters relative to nominal, for the critic: [tether (4), payload mass, drone mass, actuation (2)]
        masses = torch.stack([self.payload_mass / self.payload_mass0 - 1.0, self.drone_mass / self.drone_mass0 - 1.0], dim=-1)
        return torch.cat([self.tether.privileged_obs(), masses, self.ctrl.privileged_obs()], dim=-1)
