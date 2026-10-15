import genesis as gs

import math
import torch

from system_model.utils.qc_rate_control import QCRateController, rotors_from_genesis
from system_model.utils.qc_attitude_control import QCAttitudeController
from .math_utils import rotate_by_quat, rand_uniform, NOMINAL_SCALE

class FlightController:
    # Policy action -> motor RPM: action scaling, setpoint latency, attitude + rate control and mixing (sim rate).
    # Owns the actuation parameters randomized per episode (thrust scale, latency), written in place like TetherModel's
    def __init__(self, env_cfg, drone, total_mass, gravity, device):
        # drone: the built Genesis drone entity (rotor layout, KF). total_mass, gravity: from the built scene
        self.cfg = sys_cfg = env_cfg.sys_cfg
        self.dr_cfg = dr_cfg = env_cfg.dr_cfg
        self.clip_actions = env_cfg.clip_actions
        self.device = device
        num_envs, sim_dt, dtype = env_cfg.num_envs, env_cfg.sim_dt, gs.tc_float

        max_motor_thrust = sys_cfg.twr_max * total_mass * gravity / drone.n_propellers
        rotors = rotors_from_genesis(drone, sys_cfg.propellers_link_name, sys_cfg.propellers_spin, ct=sys_cfg.rotor_ct)
        self.rate_ctrl = QCRateController(
            num_envs=num_envs,
            dt=sim_dt,
            rotors=rotors,
            rpm_max=math.sqrt(max_motor_thrust / drone.KF),
            device=device,
            params=sys_cfg.rate_control_params,
            plant=sys_cfg.motor_plant_params,
            dtype=dtype,
        )
        self.att_ctrl = QCAttitudeController(
            num_envs=num_envs,
            dt=sim_dt,
            device=device,
            params=sys_cfg.attitude_control_params,
            dtype=dtype,
        )
        self.hover_thrust = self.rate_ctrl.thrust_setpoint_for_thrust_fraction(1.0 / sys_cfg.twr_max)
        self.min_cos_tilt = math.cos(sys_cfg.tilt_comp_max)

        # Setpoint latency in sim steps: substep i of a policy step holds the previous setpoint while i < latency_steps.
        # Config snapshots from before latency_s give the range in sim steps directly
        if hasattr(dr_cfg, "latency_s"):
            self.latency_range_steps = tuple(round(s / sim_dt) for s in dr_cfg.latency_s)
        else:
            self.latency_range_steps = tuple(dr_cfg.latency_sim_steps)
        self.latency_max_steps = max(self.latency_range_steps[1], 1) # privileged obs normalization: max latency -> 1
        self.substeps_idx = torch.arange(env_cfg.decimation, device=device)

        self.rpm_scale = torch.ones((num_envs, ), device=device, dtype=dtype)
        self.latency_steps = torch.zeros((num_envs, ), device=device, dtype=gs.tc_int)
        self.reset_sp = torch.tensor([self.hover_thrust, 0.0, 0.0, 0.0], device=device, dtype=dtype) # hover
        self.prev_sp = self.reset_sp.repeat(num_envs, 1) # [thrust, roll, pitch, yaw rate] of the last policy step

    def setpoints(self, actions):
        # Raw policy actions -> clipped actions and the per-substep setpoints (decimation, N, 4) with latency applied.
        # Latches the new setpoint as prev_sp for the next policy step
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions) # [-1; 1]
        sp = self._scale_actions(actions)
        late = self.substeps_idx[:, None, None] < self.latency_steps[None, :, None]
        substep_sps = torch.where(late, self.prev_sp, sp)
        self.prev_sp = sp
        return actions, substep_sps

    def rpms(self, drone_quat, drone_ang, sp):
        # One sim-rate control step: drone attitude (wxyz) and world angular velocity, setpoint (N, 4) -> motor RPM (N, n_rotors)
        thrust_sp, roll_sp, pitch_sp, yawrate_sp = sp.unbind(-1)
        drone_body_rates = rotate_by_quat(drone_ang, drone_quat, inverse=True)
        rate_sp = self.att_ctrl.update(drone_quat, roll_sp, pitch_sp, yawrate_sp)
        rpms = self.rate_ctrl.update(thrust_sp, rate_sp, drone_body_rates)
        return rpms * self.rpm_scale[:, None]

    def reset(self, envs_idx, drone_quat):
        self.rate_ctrl.reset(envs_idx, thrust_sp=self.hover_thrust) # hover after reset
        self.att_ctrl.reset(envs_idx, quat=drone_quat) # yaw setpoint = current heading
        self.prev_sp[envs_idx] = self.reset_sp

    def randomize(self, envs_idx, active=True):
        # Resample the actuation parameters of envs_idx from the dr_cfg ranges (active=False: nominal thrust, min latency)
        n = len(envs_idx)
        thrust_scale = self.dr_cfg.thrust_scale if active else NOMINAL_SCALE
        self.rpm_scale[envs_idx] = torch.sqrt(rand_uniform(thrust_scale, n, self.device)) # thrust ~ rpm²
        lo, hi = self.latency_range_steps
        self.latency_steps[envs_idx] = torch.randint(lo, (hi if active else lo) + 1, (n, ), device=self.device, dtype=gs.tc_int)

    def privileged_obs(self):
        # (N, 2) actuation parameters relative to nominal, for the critic
        return torch.stack([self.rpm_scale**2 - 1.0, self.latency_steps.float() / self.latency_max_steps], dim=-1)

    def _scale_actions(self, actions):
        # Thrust: piecewise linear so a=0 is hover, a=-1 is zero thrust, a=+1 is max thrust
        a_thrust = actions[:, 0]
        thrust_sp = torch.where(
            a_thrust >= 0.0,
            self.hover_thrust + a_thrust * (self.cfg.max_throttle - self.hover_thrust),
            self.hover_thrust * (1.0 + a_thrust),
        )

        # a[1:3] roll/pitch setpoints, a[3] yaw-rate setpoint
        roll_sp = actions[:, 1] * self.cfg.max_tilt
        pitch_sp = actions[:, 2] * self.cfg.max_tilt
        yawrate_sp = actions[:, 3] * self.cfg.max_yaw_rate

        if self.cfg.thrust_tilt_comp:
            cos_tilt = torch.clamp(torch.cos(roll_sp) * torch.cos(pitch_sp), min=self.min_cos_tilt)
            thrust_sp = torch.clamp(thrust_sp / cos_tilt, max=self.cfg.max_throttle)

        return torch.stack([thrust_sp, roll_sp, pitch_sp, yawrate_sp], dim=-1)
