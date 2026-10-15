import genesis as gs

import torch

from config.config_env import EPS
from .math_utils import rand_uniform, rand_log_uniform, NOMINAL_SCALE

class TetherModel:
    # Drone-payload suspension: n elastic tethers between matching attach points, applied as external wrenches on the
    # attach links. Pure torch: the env reads the attach link states and applies the wrench, see PayloadControlEnv._tethers_step.
    # Per-env (N, ) parameters are nominal at construction and resampled per episode by randomize(). They are always
    # written in place (self.x[envs_idx] = ...): the compiled wrench holds on to these tensors
    def __init__(self, env_cfg, device):
        self.cfg = env_cfg.sys_cfg
        self.dr_cfg = env_cfg.dr_cfg
        self.num_tethers = self.cfg.num_tethers
        self.dt = env_cfg.sim_dt
        self.device = device

        full = lambda v: torch.full((env_cfg.num_envs, ), v, device=device, dtype=gs.tc_float)
        self.rest_length = full(self.cfg.rest_length)
        self.stiffness = full(self.cfg.stiffness)
        self.damping = full(self.cfg.damping)
        self.torsional_damping = full(self.cfg.torsional_damping)
        self.mass_eff = full(float("nan")) # drone-payload reduced mass for the implicit damping term, owned by DomainRandomizer

    def randomize(self, envs_idx, active=True):
        # Resample the tether parameters of envs_idx from the dr_cfg ranges (active=False: nominal)
        n, L0, dr_cfg = len(envs_idx), self.cfg.rest_length, self.dr_cfg
        scale = lambda rng: rng if active else NOMINAL_SCALE
        L = L0 * rand_uniform(scale(dr_cfg.rest_length_scale), n, self.device)
        self.rest_length[envs_idx] = L
        # Rope axial stiffness is EA/L: a longer rope is softer
        self.stiffness[envs_idx] = self.cfg.stiffness * (L0 / L) * rand_log_uniform(scale(dr_cfg.stiffness_scale), n, self.device)
        self.damping[envs_idx] = self.cfg.damping * rand_log_uniform(scale(dr_cfg.damping_scale), n, self.device)
        self.torsional_damping[envs_idx] = self.cfg.torsional_damping * rand_log_uniform(scale(dr_cfg.torsional_damping_scale), n, self.device)

    def privileged_obs(self):
        # (N, 4) parameters relative to nominal, for the critic
        return torch.stack([
            self.rest_length / self.cfg.rest_length - 1.0,
            torch.log(self.stiffness / self.cfg.stiffness),
            torch.log(self.damping / self.cfg.damping),
            torch.log(self.torsional_damping / self.cfg.torsional_damping),
        ], dim=-1)

    def wrench(self, pos, vel):
        # Attach link states (N, 2n, 3) ordered [drone..., payload...] -> (force, torque) on each of those links, world frame
        n = self.num_tethers
        drone_pos, drone_vel, payload_pos, payload_vel = pos[:, :n], vel[:, :n], pos[:, n:], vel[:, n:]
        force_on_drone = self._forces(drone_pos, drone_vel, payload_pos, payload_vel)
        torque = self._torques(drone_pos, drone_vel, payload_pos, payload_vel)
        return torch.cat([force_on_drone, -force_on_drone], dim=1), torque

    def _forces(self, drone_pos, drone_vel, payload_pos, payload_vel):
        # Tension-force model per tether -> force on the drone attach points (payload gets the opposite)
        dt = self.dt
        rest_length, stiffness, damping, mass_eff = self.rest_length[:, None], self.stiffness[:, None], self.damping[:, None], self.mass_eff[:, None]

        vec = payload_pos - drone_pos
        vec_len = torch.linalg.norm(vec, dim=-1)
        uhat = vec / torch.clamp(vec_len, min=EPS).unsqueeze(-1)

        extension = vec_len - rest_length
        activation = torch.tanh(extension / self.cfg.activation_delta)
        extension_rate = ((payload_vel - drone_vel) * uhat).sum(dim=-1)

        damping_impl = damping + dt * stiffness
        extension_rate_impl = (mass_eff * extension_rate - dt * stiffness * extension) / (mass_eff + dt * damping_impl)
        tension = activation * (stiffness * extension + damping_impl * extension_rate_impl)
        tension = torch.where(vec_len > rest_length, tension, torch.zeros_like(tension))
        tension = torch.clamp(tension, min=0.0)

        return tension.unsqueeze(-1) * uhat

    def _torques(self, drone_pos, drone_vel, payload_pos, payload_vel):
        # Suspension yaw friction (rope internal friction, attachments): the tension model only damps along each line, so the
        # twist mode is otherwise undamped. Opposes the payload-drone relative yaw rate, zero when both turn together.
        # Torque per attach link [drone..., payload...], split evenly over the n links of each body
        n = self.num_tethers
        rel_yaw_rate = self._points_yaw_rate(payload_pos, payload_vel) - self._points_yaw_rate(drone_pos, drone_vel)
        tau = -self.torsional_damping * rel_yaw_rate / n # on payload, about world z
        zeros = torch.zeros_like(tau)
        tau_payload = torch.stack([zeros, zeros, tau], dim=-1).unsqueeze(1).expand(-1, n, -1)
        return torch.cat([-tau_payload, tau_payload], dim=1)

    @staticmethod
    def _points_yaw_rate(pos, vel):
        # World-z rotation rate of a rigid body from its attach points (N, n, 3): least-squares fit of v - v_mean = w_z x r in xy
        r = pos - pos.mean(dim=1, keepdim=True)
        dv = vel - vel.mean(dim=1, keepdim=True)
        num = (r[..., 0] * dv[..., 1] - r[..., 1] * dv[..., 0]).sum(dim=1)
        den = torch.clamp((r[..., 0]**2 + r[..., 1]**2).sum(dim=1), min=EPS)
        return num / den
