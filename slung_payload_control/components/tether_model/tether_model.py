import torch
from config.config_components import TetherModelConfig

class TetherModel:
    def __init__(self, cfg: TetherModelConfig, num_envs: int, sim_dt: float, device, dtype):
        self.cfg = cfg
        self.num_envs = num_envs
        self.sim_dt = sim_dt
        self.device = device
        self.dtype = dtype

        # Constructed with nominal values - Use set_parameters to do per-env randomization
        full = lambda v: torch.full((self.num_envs, ), v, device=self.device, dtype=self.dtype)
        self.rest_length = full(self.cfg.rest_length)
        self.stiffness = full(self.cfg.stiffness)
        self.damping = full(self.cfg.damping)
        self.torsional_damping = full(self.cfg.torsional_damping)
        self.mass_eff = full(float("nan")) # drone-payload reduced mass for implicit dampening term (set by domain rand)

    def compile(self):
        self.step = torch.compile(self.step, dynamic=False)

        # Run one step after to trigger compile (run with dummy tensors)
        n = self.cfg.num_tethers
        dummy_pos = torch.zeros((self.num_envs, 2*n, 3), device=self.device, dtype=self.dtype)
        dummy_vel = torch.zeros((self.num_envs, 2*n, 3), device=self.device, dtype=self.dtype)
        self.step(dummy_pos, dummy_vel)

    def set_parameters(self, envs_idx, rest_length, stiffness, damping, torsional_damping, mass_eff):
        self.rest_length[envs_idx] = rest_length
        self.stiffness[envs_idx] = stiffness
        self.damping[envs_idx] = damping
        self.torsional_damping[envs_idx] = torsional_damping
        self.mass_eff[envs_idx] = mass_eff

    def step(self, pos, vel):
        n = self.cfg.num_tethers
        drone_pos = pos[:, :n]
        drone_vel = vel[:, :n]
        payload_pos = pos[:, n:]
        payload_vel = vel[:, n:]
        force_on_drone = self._forces(drone_pos, drone_vel, payload_pos, payload_vel)
        torque = self._torques(drone_pos, drone_vel, payload_pos, payload_vel)
        force = torch.cat([force_on_drone, -force_on_drone], dim=1)
        return force, torque

    def _forces(self, drone_pos, drone_vel, payload_pos, payload_vel, EPS=1e-9):
        # Tether tension model per tether -> force on drone attach points (payload gets equal but opposite force)
        dt = self.sim_dt
        l = self.rest_length[:, None]
        s = self.stiffness[:, None]
        d = self.damping[:, None]
        me = self.mass_eff[:, None] / self.cfg.num_tethers

        vec = payload_pos - drone_pos
        vec_l = torch.linalg.norm(vec, dim=-1)
        uhat = vec / torch.clamp(vec_l, min=EPS).unsqueeze(-1)

        extension = vec_l - l
        extension_rate = ((payload_vel - drone_vel) * uhat).sum(dim=-1)

        d_eff = d * me / (me + dt * d)
        elastic = s * extension
        gate = torch.clamp(elastic / self.cfg.gate_tension, 0.0, 1.0)
        tension = elastic + gate * d_eff * extension_rate
        tension = torch.where(vec_l > l, tension, torch.zeros_like(tension)).clamp(min=0.0)

        return tension.unsqueeze(-1) * uhat

    def _torques(self, drone_pos, drone_vel, payload_pos, payload_vel):
        # Suspension yaw friction (rope internal frcition, attachments)
        n = self.cfg.num_tethers
        rel_yaw_rate = self._points_yaw_rate(payload_pos, payload_vel) - self._points_yaw_rate(drone_pos, drone_vel)
        tau = -self.torsional_damping * rel_yaw_rate / n
        zeros = torch.zeros_like(tau)
        tau_payload = torch.stack([zeros, zeros, tau], dim=-1).unsqueeze(1).expand(-1, n, -1)
        return torch.cat([-tau_payload, tau_payload], dim=1)

    @staticmethod
    def _points_yaw_rate(pos, vel, EPS=1e-9):
        # World-z rotation rate of a rigid body from its attach points (N, n, 3)
        # least-squares fit of v - v_mean = w_z x r in xy
        # will be zero for only one cable, which makes sense for the torque calculation as well 
        r = pos - pos.mean(dim=1, keepdim=True)
        dv = vel - vel.mean(dim=1, keepdim=True)
        num = (r[..., 0] * dv[..., 1] - r[..., 1] * dv[..., 0]).sum(dim=1)
        den = torch.clamp((r[..., 0] * r[..., 0] + r[..., 1] * r[..., 1]).sum(dim=1), min=EPS)
        return num / den