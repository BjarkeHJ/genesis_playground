import math
from typing import Optional, Sequence
import torch

from utils.torch_math_helpers import quat_mul, quat_conj, quat_canonical, quat_from_euler, quat_yaw, dcm_z, quat_between, wrap_pi
from .airframe import Rotor
from .filters import LowPassFilter2, AlphaFilter
from .params import AttitudeControlParams, RateControlParams, MotorPlantParams, ROLL, PITCH, YAW, THRUST_Z, FLT_EPS

# Vectorized PX4 multicopter attitude and rate control, (N, ...) batched over envs. Pure torch: the airframe-dependent
# tensors (mixer, rpm range) are allocated at construction and written in place by QCRateController.bind, so the
# per-step functions can be compiled before the airframe is known

class QCAttitudeController:
    def __init__(self, num_envs, dt, device, dtype, params: AttitudeControlParams = AttitudeControlParams()):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.params = params

        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        self.gain_p = t([params.MC_ROLL_P, params.MC_PITCH_P, params.MC_YAW_P])
        self.rate_max = t([math.radians(params.MC_ROLLRATE_MAX), math.radians(params.MC_PITCHRATE_MAX), math.radians(params.MC_YAWRATE_MAX)])
        self.yaw_w = min(max(params.MC_YAW_WEIGHT, 0.0), 1.0)

        self.yaw_sp = torch.zeros(num_envs, device=device, dtype=dtype)
        self.q_sp = torch.zeros((num_envs, 4), device=device, dtype=dtype) # logging
        self.q_sp[:, 0] = 1.0

    def update(self, quat, roll_sp, pitch_sp, yawrate_sp):
        # quat: (N,4) wxyz body->world, roll_sp/pitch_sp [rad], yawrate_sp [rad/s] (world z) -> (N,3) body rates FLU
        q = quat / quat.norm(dim=-1, keepdim=True)
        yaw = quat_yaw(q)

        if self.params.yaw_hold:
            yaw_err = torch.clamp(wrap_pi(self.yaw_sp + yawrate_sp * self.dt - yaw), -self.params.yaw_err_max, self.params.yaw_err_max)
            self.yaw_sp = wrap_pi(yaw + yaw_err)
        else:
            self.yaw_sp = yaw # yaw error ~0, heading driven by feed-forward only

        qd = quat_from_euler(roll_sp, pitch_sp, self.yaw_sp)
        self.q_sp = qd
        return self._attitude_to_rate(q, qd, yawrate_sp)

    def _attitude_to_rate(self, q, qd, yawspeed_sp):
        # reduced attitude: tilt (thrust vector) first, then yaw mixed in with weight yaw_w
        qd_red = quat_between(dcm_z(q), dcm_z(qd))
        degenerate = (qd_red[:, 1].abs() > 1.0 - 1e-5) | (qd_red[:, 2].abs() > 1.0 - 1e-5)
        qd_red = torch.where(degenerate.unsqueeze(-1), qd, quat_mul(qd_red, q))

        q_mix = quat_canonical(quat_mul(quat_conj(qd_red), qd))
        mw = torch.clamp(q_mix[:, 0], -1.0, 1.0)
        mz = torch.clamp(q_mix[:, 3], -1.0, 1.0)
        zeros = torch.zeros_like(mw)
        q_yaw = torch.stack([torch.cos(self.yaw_w * torch.acos(mw)), zeros, zeros, torch.sin(self.yaw_w * torch.asin(mz))], dim=-1)
        qd = quat_mul(qd_red, q_yaw)

        # attitude error -> rate setpoint
        qe = quat_canonical(quat_mul(quat_conj(q), qd))
        rate_sp = 2.0 * qe[:, 1:] * self.gain_p

        # yaw rate feed-forward: world z expressed in body (= R^T e_z = dcm_z(q^-1))
        rate_sp = rate_sp + dcm_z(quat_conj(q)) * yawspeed_sp.unsqueeze(-1)

        rate_sp = torch.nan_to_num(rate_sp, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.maximum(torch.minimum(rate_sp, self.rate_max), -self.rate_max)

    def reset(self, envs_idx, quat: Optional[torch.Tensor] = None):
        # align yaw setpoint with the current heading (quat: (len(envs_idx),4) wxyz), else 0
        self.yaw_sp[envs_idx] = 0.0 if quat is None else quat_yaw(quat).to(self.dtype)
        self.q_sp[envs_idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)

class RateControl:
    # torque = P*e + I - D*angular_accel + FF*rate_sp (normalized torque, FRD)
    def __init__(self, num_envs, dt, device, dtype, params: RateControlParams):
        self.dt = dt
        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        k = t([params.MC_ROLLRATE_K, params.MC_PITCHRATE_K, params.MC_YAWRATE_K])

        self.gain_p = k * t([params.MC_ROLLRATE_P, params.MC_PITCHRATE_P, params.MC_YAWRATE_P])
        self.gain_i = k * t([params.MC_ROLLRATE_I, params.MC_PITCHRATE_I, params.MC_YAWRATE_I])
        self.gain_d = k * t([params.MC_ROLLRATE_D, params.MC_PITCHRATE_D, params.MC_YAWRATE_D])
        self.gain_ff = t([params.MC_ROLLRATE_FF, params.MC_PITCHRATE_FF, params.MC_YAWRATE_FF])
        self.lim_int = t([params.MC_RR_INT_LIM, params.MC_PR_INT_LIM, params.MC_YR_INT_LIM])

        self.rate_int = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.sat_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.bool) # allocator saturation, from the previous cycle
        self.sat_neg = torch.zeros((num_envs, 3), device=device, dtype=torch.bool)
        self._i_ref = math.radians(400.0)

    def update(self, rate, rate_sp, angular_accel):
        rate_error = rate_sp - rate
        torque = self.gain_p * rate_error + self.rate_int - self.gain_d * angular_accel + self.gain_ff * rate_sp
        self._update_integral(rate_error)
        return torque

    def _update_integral(self, rate_error):
        # prevent further saturation in the direction the allocator could not deliver
        e = torch.where(self.sat_pos, torch.clamp(rate_error, max=0.0), rate_error)
        e = torch.where(self.sat_neg, torch.clamp(e, min=0.0), e)
        # I-term attenuation for large errors (bounce back)
        i_factor = e / self._i_ref
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)
        rate_i = self.rate_int + i_factor * self.gain_i * e * self.dt
        rate_i = torch.maximum(torch.minimum(rate_i, self.lim_int), -self.lim_int)
        self.rate_int = torch.where(torch.isfinite(rate_i), rate_i, self.rate_int)

    def reset(self, envs_idx):
        self.rate_int[envs_idx] = 0.0
        self.sat_pos[envs_idx] = False
        self.sat_neg[envs_idx] = False

def mixer(rotors: Sequence[Rotor]):
    # PX4 ActuatorEffectivenessRotors + normalization: rotor layout -> effectiveness E (4, n), normalized mix (n, 4), scale (4, )
    E = torch.tensor([[-r.ct * r.py, r.ct * r.px, r.ct * r.km, -r.ct] for r in rotors], dtype=torch.float64).T # rows ROLL, PITCH, YAW, THRUST_Z
    E[(E.abs() <= 0.05).all(dim=1)] = 0.0 # weak axes

    mix = torch.linalg.pinv(E)
    scale = torch.ones(4, dtype=torch.float64)
    nz_r = int((mix[:, ROLL].abs() > 1e-3).sum())
    nz_p = int((mix[:, PITCH].abs() > 1e-3).sum())
    roll_s = math.sqrt(float(mix[:, ROLL].pow(2).sum()) / (nz_r / 2.0)) if nz_r > 0 else 1.0
    pitch_s = math.sqrt(float(mix[:, PITCH].pow(2).sum()) / (nz_p / 2.0)) if nz_p > 0 else 1.0

    scale[ROLL] = scale[PITCH] = max(roll_s, pitch_s)
    scale[YAW] = float(mix[:, YAW].max())
    t_abs = mix[:, THRUST_Z].abs()
    nz_t = int((t_abs > FLT_EPS).sum())
    scale[THRUST_Z] = float(t_abs.sum()) / nz_t if nz_t > 0 else 1.0

    if scale[ROLL] > FLT_EPS:
        mix[:, ROLL] /= scale[ROLL]
        mix[:, PITCH] /= scale[PITCH]
    if scale[YAW] > FLT_EPS:
        mix[:, YAW] /= scale[YAW]
    if scale[THRUST_Z] > FLT_EPS:
        mix[:, THRUST_Z] /= scale[THRUST_Z]
    mix[mix.abs() < 1e-3] = 0.0
    return E, mix, scale

class ControlAllocator:
    # control vector c = [roll, pitch, yaw, thrust_z] (normalized, FRD, thrust_z=-collective thrust) -> actuator setpoints [0, 1]
    def __init__(self, num_envs, n_rotors, dt, device, dtype, params: RateControlParams):
        self.n = n_rotors
        self.dt = dt
        self.airmode = params.MC_AIRMODE
        self.slew_s = params.CA_R_SLEW

        # Filled by set_rotors
        self.E = torch.zeros((4, n_rotors), device=device, dtype=dtype)
        self.mix = torch.zeros((n_rotors, 4), device=device, dtype=dtype)
        self.scale = torch.ones(4, device=device, dtype=dtype)

        self.act_min = torch.zeros(n_rotors, device=device, dtype=dtype)
        self.act_max = torch.ones(n_rotors, device=device, dtype=dtype)
        self.actuator_sp = torch.zeros((num_envs, n_rotors), device=device, dtype=dtype)

    def set_rotors(self, rotors: Sequence[Rotor]):
        assert len(rotors) == self.n, f"allocator built for {self.n} rotors, got {len(rotors)}"
        for buf, value in zip((self.E, self.mix, self.scale), mixer(rotors)):
            buf.copy_(value)

    def _desaturation_gain(self, vec, sp, act_max):
        usable = vec.abs() >= 0.2 # do not desaturate with weakly effective actuators
        denom = torch.where(usable, vec,  torch.ones_like(vec))
        k = torch.zeros_like(sp)
        k = torch.where(sp < self.act_min, (self.act_min - sp) / denom, k)
        k = torch.where(sp > act_max, (act_max - sp) / denom, k)
        k = torch.where(usable, k, torch.zeros_like(k))
        k_min = torch.clamp(k.min(dim=-1).values, max=0.0)
        k_max = torch.clamp(k.max(dim=-1).values, min=0.0)
        return k_min + k_max

    def _desaturate(self, sp, vec, increase_only=False, act_max=None):
        act_max = self.act_max if act_max is None else act_max
        gain = self._desaturation_gain(vec, sp, act_max)
        skip = (gain < 0.0) if increase_only else torch.zeros_like(gain, dtype=torch.bool)
        gain = torch.where(skip, torch.zeros_like(gain), gain)
        sp = sp + gain.unsqueeze(-1) * vec
        gain = 0.5 * self._desaturation_gain(vec, sp, act_max)
        gain = torch.where(skip, torch.zeros_like(gain), gain)
        return sp + gain.unsqueeze(-1) * vec

    def _mix_yaw(self, sp, c):
        yaw, thrust = self.mix[:, YAW], self.mix[:, THRUST_Z]
        sp = sp + c[:, YAW:YAW + 1] * yaw
        # allow some yaw response at maximum thrust
        sp = self._desaturate(sp, yaw, act_max=self.act_max + 0.15 * (self.act_max - self.act_min))
        return self._desaturate(sp, thrust, increase_only=True) # reduce thrust only (obs negative sign conv)

    def allocate(self, c):
        M = self.mix
        rpt = c[:, [ROLL, PITCH, THRUST_Z]] @ M[:, [ROLL, PITCH, THRUST_Z]].T
        if self.airmode == 1:
            sp = self._desaturate(rpt, M[:, THRUST_Z])
            sp = self._mix_yaw(sp, c)
        elif self.airmode == 2:
            sp = self._desaturate(c @ M.T, M[:, THRUST_Z])
            sp = self._desaturate(sp, M[:, YAW]) # prioritize roll/pitch over yaw
        else:
            sp = self._desaturate(rpt, M[:, THRUST_Z], increase_only=True) # never increase thrust (obs negative sign conv)
            sp = self._desaturate(sp, M[:, ROLL])
            sp = self._desaturate(sp, M[:, PITCH])
            sp = self._mix_yaw(sp, c)

        if self.slew_s > FLT_EPS:
            d_max = self.dt * (self.act_max - self.act_min) / self.slew_s
            sp = self.actuator_sp + torch.maximum(torch.minimum(sp - self.actuator_sp, d_max), -d_max)

        sp = torch.maximum(torch.minimum(sp, self.act_max), self.act_min)
        self.actuator_sp = sp
        return sp

    def saturation_status(self, c):
        allocated = (self.actuator_sp @ self.E.T) * self.scale
        unallocated = c[:, :3] - allocated[:, :3]
        achieved = unallocated.pow(2).sum(dim=-1, keepdim=True) < 1e-6
        sat_pos = (~achieved) & (unallocated > FLT_EPS)
        sat_neg = (~achieved) & (unallocated < -FLT_EPS)
        return sat_pos, sat_neg, unallocated

    def reset(self, envs_idx, actuator_sp=None):
        self.actuator_sp[envs_idx] = 0.0 if actuator_sp is None else actuator_sp

class MotorOutput:
    # Output stage: actuator setpoint [0, 1] -> thrust model inversion -> rpm command -> first-order ESC/motor response
    def __init__(self, num_envs, n_rotors, dt, device, dtype, thr_mdl_fac, plant: MotorPlantParams):
        self.fac = float(thr_mdl_fac)
        self.rpm_min_frac = plant.rpm_min_frac
        a = lambda tau: 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau) # ZOH discretization
        self.alpha_up = a(plant.tau_up)
        self.alpha_down = a(plant.tau_down)

        # Filled by set_rpm_max
        self.rpm_max = torch.zeros((), device=device, dtype=dtype)
        self.rpm_min = torch.zeros((), device=device, dtype=dtype)

        self.rpm = torch.zeros((num_envs, n_rotors), device=device, dtype=dtype)

    def set_rpm_max(self, rpm_max):
        self.rpm_max.fill_(rpm_max)
        self.rpm_min.fill_(self.rpm_min_frac * rpm_max)

    def thrust_model(self, u):
        # invert rel_thrust = f*x² + (1-f)*x
        f = self.fac
        if not ( 0.0 < f <= 1.0):
            return u
        b = 1.0 - f
        tmp1 = b / (2.0 * f)
        tmp2 = b * b / (4.0 * f * f)
        return torch.where(u > 0.0, -tmp1 + torch.sqrt(torch.clamp(tmp2 + u / f, min=0.0)), torch.zeros_like(u))

    def cmd_to_rpm(self, cmd):
        return self.rpm_min + cmd * (self.rpm_max - self.rpm_min)

    def step(self, u):
        rpm_cmd = self.cmd_to_rpm(self.thrust_model(u))
        alpha = torch.where(rpm_cmd > self.rpm, self.alpha_up, self.alpha_down)
        self.rpm = self.rpm + alpha * (rpm_cmd - self.rpm)
        return self.rpm

    def setpoint_for_thrust_fraction(self, f):
        # normalized actuator setpoint that makes one motor produce the fraction f of its max thrust (thrust ~ rpm²).
        x = min(max((math.sqrt(max(f, 0.0)) - self.rpm_min_frac) / (1.0 - self.rpm_min_frac), 0.0), 1.0)
        fac = self.fac if 0.0 < self.fac <= 1.0 else 0.0
        return fac * x * x + (1.0 - fac) * x

    def reset(self, envs_idx, rpm):
        self.rpm[envs_idx] = rpm

class QCRateController:
    # Top level: body rates -> filtered rate PID -> allocation -> motor rpm. Runs once per sim step (dt).
    # Rotor layout and rpm range come in through bind (all tensors allocated here)
    def __init__(self, num_envs, n_rotors, dt, device, dtype, params: RateControlParams = RateControlParams(), plant: MotorPlantParams = MotorPlantParams()):
        self.dt = dt
        self.device = device
        self.dtype = dtype

        fs = 1.0 / dt
        self._flu_to_frd = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)

        self.gyro_lpf = LowPassFilter2((num_envs, 3), fs, params.IMU_GYRO_CUTOFF, device, dtype)
        self.dgyro_lpf = AlphaFilter((num_envs, 3), fs, params.IMU_DGYRO_CUTOFF, device, dtype)
        self.rate_prev = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.rate_control = RateControl(num_envs, dt, device, dtype, params)
        self.allocator = ControlAllocator(num_envs, n_rotors, dt, device, dtype, params)
        self.motors = MotorOutput(num_envs, n_rotors, dt, device, dtype, params.THR_MDL_FAC, plant)
        self.unallocated_torque = torch.zeros((num_envs, 3), device=device, dtype=dtype) # logging

    def bind(self, rotors: Sequence[Rotor], rpm_max: float):
        # Airframe-dependent parameters, written in place (compiled update() keeps its graph)
        self.allocator.set_rotors(rotors)
        self.motors.set_rpm_max(rpm_max)

    def update(self, thrust_sp, rates_sp, rates):
        # thrust_sp (N, ) normalized collective, rates_sp / rates (N,3) body FLU [rad/s] -> (N, n_rotors) rpm
        frd = self._flu_to_frd

        rate = self.gyro_lpf.apply(rates * frd)
        accel = self.dgyro_lpf.apply((rate - self.rate_prev) / self.dt)
        self.rate_prev = rate

        torque = self.rate_control.update(rate, rates_sp * frd, accel)
        torque = torch.nan_to_num(torque, nan=0.0, posinf=0.0, neginf=0.0)

        c = torch.cat([torque, -thrust_sp.unsqueeze(-1)], dim=-1)
        u = self.allocator.allocate(c)
        sat_pos, sat_neg, self.unallocated_torque = self.allocator.saturation_status(c)
        self.rate_control.sat_pos, self.rate_control.sat_neg = sat_pos, sat_neg # used from next cycle

        return self.motors.step(u)

    def thrust_setpoint_for_thrust_fraction(self, f):
        # eg hover: f = weight / (n_rotors * max_motor_thrust) = 1 / twr_max
        return self.motors.setpoint_for_thrust_fraction(f)

    def reset(self, envs_idx, thrust_sp: float = 0.0, rates=None):
        # reset integrator, filters and allocator; spin motors at the rpm matching thrust_sp
        rate0 = 0.0 if rates is None else rates * self._flu_to_frd
        self.gyro_lpf.reset(envs_idx, rate0)
        self.dgyro_lpf.reset(envs_idx, 0.0)
        self.rate_prev[envs_idx] = rate0
        self.rate_control.reset(envs_idx)
        u0 = torch.full((self.allocator.n, ), float(thrust_sp), device=self.device, dtype=self.dtype)
        self.allocator.reset(envs_idx, u0)
        self.motors.reset(envs_idx, self.motors.cmd_to_rpm(self.motors.thrust_model(u0)))
        self.unallocated_torque[envs_idx] = 0.0
