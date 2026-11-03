import math
import torch

from config.config_components import FlightControllerConfig
from utils.torch_math_helpers import quat_rotate_inverse
from .airframe import Airframe
from .px4_vectorized import QCAttitudeController, QCRateController

class FlightController:
    # Policy action -> motor RPM: action scaling, setpoint latency, attitude + rate control and mixing @ sim rate.
    # Pure torch. Lifecycle: construct (config only) -> compile() (optional, before the scene exists) -> bind(airframe)
    # -> reset / policy_step / sim_step. Per-env actuation parameters come in through set_actuation (domain randomizer)
    def __init__(self, cfg: FlightControllerConfig, num_envs: int, sim_dt: float, decimation: int, device, dtype):
        self.cfg = cfg
        self.num_envs = num_envs
        self.decimation = decimation
        self.device = device
        self.dtype = dtype

        self.att_ctrl = QCAttitudeController(num_envs, sim_dt, device, dtype)
        self.rate_ctrl = QCRateController(num_envs, cfg.n_rotors, sim_dt, device, dtype)

        # Hover collective is airframe independent (see MotorOutput.setpoint_for_thrust_fraction)
        self.hover_thrust = self.rate_ctrl.thrust_setpoint_for_thrust_fraction(1.0 / cfg.twr_max)
        self.min_cos_tilt = math.cos(cfg.tilt_comp_max)
        self.reset_sp = torch.tensor([self.hover_thrust, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.prev_sp = self.reset_sp.repeat(num_envs, 1) # [thrust, roll, pitch, yaw rate] of the last policy step

        # Actuation parameters (nominal until set_actuation): thrust scale applied as rpm * sqrt(scale), and setpoint
        # latency in sim steps (substep i of a policy step holds the previous setpoint while i < latency_steps)
        self.thrust_scale = torch.ones(num_envs, device=device, dtype=dtype)
        self.latency_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.substeps_idx = torch.arange(decimation, device=device)

    # ======================== LIFECYCLE ========================

    def compile(self):
        # torch.compile both per-step stages and trace them now on dummy inputs shaped like the sim's, so the first real
        # steps are fast. Needs no airframe: the airframe tensors are only read at runtime. Leaves every env reset
        self.policy_step = torch.compile(self.policy_step, dynamic=False)
        self.sim_step = torch.compile(self.sim_step, dynamic=False)

        quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=self.dtype)
        quat[:, 0] = 1.0
        ang = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        _, substep_sps = self.policy_step(torch.zeros((self.num_envs, 4), device=self.device, dtype=self.dtype))
        self.sim_step(quat, ang, substep_sps[0])
        self.reset(torch.arange(self.num_envs, device=self.device), quat)

    def bind(self, airframe: Airframe):
        self.rate_ctrl.bind(airframe.rotors, airframe.rpm_max)

    def reset(self, envs_idx, quat):
        # Hover setpoint and spinning motors, yaw setpoint at the current heading (quat: (len(envs_idx), 4) wxyz)
        self.rate_ctrl.reset(envs_idx, thrust_sp=self.hover_thrust)
        self.att_ctrl.reset(envs_idx, quat)
        self.prev_sp[envs_idx] = self.reset_sp

    def set_actuation(self, envs_idx, thrust_scale, latency_steps):
        # thrust_scale: motor thrust relative to nominal (KF / battery), latency_steps: setpoint delay in sim steps
        self.thrust_scale[envs_idx] = thrust_scale
        self.latency_steps[envs_idx] = latency_steps

    # ======================== STEP ========================

    def policy_step(self, actions):
        # Raw policy actions (N, 4) -> clipped actions and the per-substep setpoints (decimation, N, 4) with latency.
        # Latches the new setpoint as prev_sp for the next policy step
        actions = torch.clip(actions, -self.cfg.clip_values, self.cfg.clip_values)
        sp = self._scale_actions(actions)
        late = self.substeps_idx[:, None, None] < self.latency_steps[None, :, None]
        substep_sps = torch.where(late, self.prev_sp, sp)
        self.prev_sp = sp
        return actions, substep_sps

    def sim_step(self, quat, ang, sp):
        # One sim-rate control step: attitude (N, 4) wxyz body->world, world angular velocity (N, 3), setpoint (N, 4)
        # -> motor RPM (N, n_rotors)
        thrust_sp, roll_sp, pitch_sp, yawrate_sp = sp.unbind(-1)
        rate_sp = self.att_ctrl.update(quat, roll_sp, pitch_sp, yawrate_sp)
        rpms = self.rate_ctrl.update(thrust_sp, rate_sp, quat_rotate_inverse(quat, ang))
        return rpms * torch.sqrt(self.thrust_scale)[:, None] # thrust ~ rpm²

    def privileged_obs(self):
        # (N, 2) actuation parameters relative to nominal, for the critic. Latency normalized by one policy step
        return torch.stack([self.thrust_scale - 1.0, self.latency_steps.to(self.dtype) / self.decimation], dim=-1)

    def _scale_actions(self, actions):
        # actions in [-1, 1] -> [thrust, roll, pitch, yaw rate] setpoints
        # Thrust: piecewise linear so a=0 is hover, a=-1 is zero thrust, a=+1 is max_throttle
        a_thrust = actions[:, 0]
        thrust_sp = torch.where(
            a_thrust >= 0.0,
            self.hover_thrust + a_thrust * (self.cfg.max_throttle - self.hover_thrust),
            self.hover_thrust * (1.0 + a_thrust),
        )
        roll_sp = actions[:, 1] * self.cfg.max_tilt
        pitch_sp = actions[:, 2] * self.cfg.max_tilt
        yawrate_sp = actions[:, 3] * self.cfg.max_yaw_rate

        if self.cfg.thrust_tilt_comp:
            cos_tilt = torch.clamp(torch.cos(roll_sp) * torch.cos(pitch_sp), min=self.min_cos_tilt)
            thrust_sp = torch.clamp(thrust_sp / cos_tilt, max=self.cfg.max_throttle)

        return torch.stack([thrust_sp, roll_sp, pitch_sp, yawrate_sp], dim=-1)
