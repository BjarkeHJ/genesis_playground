import math
from dataclasses import dataclass
from typing import Optional, Sequence
import torch

FLT_EPS = 1.1920929e-07

# Axis indices of the control vector
ROLL, PITCH, YAW, THRUST_Z = 0, 1, 2, 3

@dataclass(frozen=True)
class Rotor:
    # One rotor position relative to CoM
    px: float 
    py: float
    km: float # moment ratio, > 0 for CCW rotor
    ct: float = 6.5

@dataclass(frozen=True)
class RateControlParams:
    # PX4 Parameters (defaults)
    MC_ROLLRATE_P: float = 0.15
    MC_ROLLRATE_I: float = 0.2
    MC_ROLLRATE_D: float = 0.003
    MC_ROLLRATE_FF: float = 0.0
    MC_ROLLRATE_K: float = 1.0
    MC_RR_INT_LIM: float = 0.30
    MC_PITCHRATE_P: float = 0.15
    MC_PITCHRATE_I: float = 0.2
    MC_PITCHRATE_D: float = 0.003
    MC_PITCHRATE_FF: float = 0.0
    MC_PITCHRATE_K: float = 1.0
    MC_PR_INT_LIM: float = 0.30
    MC_YAWRATE_P: float = 0.2
    MC_YAWRATE_I: float = 0.1
    MC_YAWRATE_D: float = 0.0
    MC_YAWRATE_FF: float = 0.0
    MC_YAWRATE_K: float = 1.0
    MC_YR_INT_LIM: float = 0.30

    IMU_GYRO_CUTOFF: float = 40.0 # [Hz] 2nd-order Butterworth on the rate, 0 disables
    IMU_DGYRO_CUTOFF: float = 30.0 # [HZ] 1st-order lowpass on D-term input, 0 disables

    MC_AIRMODE: int = 0 # 0 diabled, 1 roll/pitch, 2 roll/pitch/yaw
    CA_R_SLEW: float = 0.0 

    THR_MDL_FAC: float = 1.0 # orig 0.0

    status_interval_s: float = 0.005

@dataclass(frozen=True)
class MotorPlantParams:
    # ESC + motor model (From hardware)
    rpm_min_frac: float = 0.0 # rpm at cmd = 0.0
    tau_up: float = 0.0125 # [s] spin-up time constant
    tau_down: float = 0.025 # [s] spin-down time constant

class LowPassFilter2:
    # 2nd order Butterworth
    def __init__(self, shape, sample_freq, cutoff_freq, device, dtype):
        self.d1 = torch.zeros(shape, device=device, dtype=dtype)
        self.d2 = torch.zeros(shape, device=device, dtype=dtype)
        self.enabled = sample_freq > 0.0 and 0.0 < cutoff_freq < sample_freq / 2.0
        if not self.enabled:
            self.bo, self.b1, self.b2, self.a1, self.a2 = 1.0, 0.0, 0.0, 0.0, 0.0
            return

        cutoff = max(cutoff_freq, sample_freq * 0.001)
        ohm = math.tan(math.pi / (sample_freq / cutoff))
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm

        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c

    def apply(self, x):
        if not self.enabled:
            return x
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2 = self.d1
        self.d1 = d0
        return y

    def reset(self, envs_idx, value):
        # steady state of the delay line for a constant input
        dval = value / (1.0 + self.a1 + self.a2)
        self.d1[envs_idx] = dval
        self.d2[envs_idx] = dval

class AlphaFilter:
    # 1st order low-pass y += alpha * (x - y), alpha = dt / (dt + 1/(2pi*fc))
    def __init__(self, shape, sample_freq, cutoff_freq, device, dtype):
        self.state = torch.zeros(shape, device=device, dtype=dtype)
        if sample_freq > 0.0 and 0.0 < cutoff_freq < sample_freq / 2.0:
            dt = 1.0 / sample_freq
            tau = 1.0 / (2.0 * math.pi * cutoff_freq)
            self.alpha = dt / (dt + tau)
        else:
            self.alpha = 1.0 

    def apply(self, x):
        self.state = self.state + self.alpha * (x - self.state)
        return self.state

    def reset(self, envs_idx, value):
        self.state[envs_idx] = value

class RateControl:
    # torque = P*e + I - D*angular_accel + FF*rate_sp (normalized torque, FRD)
    def __init__(self, num_envs, params: RateControlParams, device, dtype):
        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        k = t([params.MC_ROLLRATE_K, params.MC_PITCHRATE_K, params.MC_YAWRATE_K])
        
        self.gain_p = k * t([params.MC_ROLLRATE_P, params.MC_PITCHRATE_P, params.MC_YAWRATE_P])
        self.gain_i = k * t([params.MC_ROLLRATE_I, params.MC_PITCHRATE_I, params.MC_YAWRATE_I])
        self.gain_d = k * t([params.MC_ROLLRATE_D, params.MC_PITCHRATE_D, params.MC_YAWRATE_D])
        self.gain_ff = t([params.MC_ROLLRATE_FF, params.MC_PITCHRATE_FF, params.MC_YAWRATE_FF])
        self.lim_int = t([params.MC_RR_INT_LIM, params.MC_PR_INT_LIM, params.MC_YR_INT_LIM])

        self.rate_int = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.sat_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.bool)
        self.sat_neg = torch.zeros((num_envs, 3), device=device, dtype=torch.bool)
        self._i_ref = math.radians(400.0)

    def set_saturation_status(self, sat_pos, sat_neg):
        self.sat_pos.copy_(sat_pos)
        self.sat_neg.copy_(sat_neg)

    def update(self, rate, rate_sp, angular_accel, dt, landed: Optional[torch.Tensor] = None):
        rate_error = rate_sp - rate
        torque = self.gain_p * rate_error + self.rate_int - self.gain_d * angular_accel + self.gain_ff * rate_sp
        self._update_integral(rate_error, dt, landed)
        return torque

    def _update_integral(self, rate_error, dt, landed):
        # prevent further saturation in the direction the allocator could not deliver
        e = torch.where(self.sat_pos, torch.clamp(rate_error, max=0.0), rate_error)
        e = torch.where(self.sat_neg, torch.clamp(e, min=0.0), e)
        # I-term attenuation for large errors (bounce back)
        i_factor = e / self._i_ref
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)
        rate_i = self.rate_int + i_factor * self.gain_i * e * dt
        rate_i = torch.maximum(torch.minimum(rate_i, self.lim_int), -self.lim_int)
        keep = ~torch.isfinite(rate_i)
        if landed is not None:
            keep = keep | landed.unsqueeze(-1)
        self.rate_int = torch.where(keep, self.rate_int, rate_i)

    def reset(self, envs_idx):
        self.rate_int[envs_idx] = 0.0
        self.sat_pos[envs_idx] = False
        self.sat_neg[envs_idx] = False

class ControlAllocator:
    # control vector c = [roll, pitch, yaw, thrust_z] (normalized, FRD, thrust_z=-collective thrust)
    def __init__(self, rotors: Sequence[Rotor], num_envs, airmode, slew_s, device, dtype):
        self.n = len(rotors)
        self.airmode = airmode
        self.device = device
        self.dtype = dtype

        # ActuatorEffectiveness
        E = torch.zeros((4, self.n), dtype=torch.float64)
        for i, r in enumerate(rotors):
            E[ROLL, i] = -r.ct * r.py
            E[PITCH, i] = r.ct * r.px
            E[YAW, i] = r.ct * r.km
            E[THRUST_Z, i] = -r.ct

        weak = (E.abs() <= 0.05).all(dim=1)
        E[weak] = 0.0

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

        self.E = E.to(device=device, dtype=dtype)
        self.mix = mix.to(device=device, dtype=dtype)
        self.scale = scale.to(device=device, dtype=dtype)
        self.act_min = torch.zeros(self.n, device=device, dtype=dtype)
        self.act_max = torch.ones(self.n, device=device, dtype=dtype)
        self.slew_s = slew_s
        self.actuator_sp = torch.zeros((num_envs, self.n), device=device, dtype=dtype)

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

    def allocate(self, c, dt):
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
            d_max = dt * (self.act_max - self.act_min) / self.slew_s
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
    # output stage
    def __init__(self, num_envs, n, rpm_max, thr_mdl_fac, plant: MotorPlantParams, dt, device, dtype):
        self.rpm_max = float(rpm_max)
        self.rpm_min = plant.rpm_min_frac * self.rpm_max
        self.fac = float(thr_mdl_fac)
        a = lambda tau: 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau) # ZOH discretization
        self.alpha_up = a(plant.tau_up)
        self.alpha_down = a(plant.tau_down)
        self.rpm = torch.zeros((num_envs, n), device=device, dtype=dtype)

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
        # normalized actuator setpoint that makes one motor produce the fraction of its max thrust
        rpm = math.sqrt(max(f, 0.0)) * self.rpm_max
        x = min(max((rpm - self.rpm_min) / (self.rpm_max - self.rpm_min), 0.0), 1.0)
        fac = self.fac if 0.0 < self.fac <= 1.0 else 0.0
        return fac * x * x + (1.0 - fac) * x

    def reset(self, envs_idx, rpm):
        self.rpm[envs_idx] = rpm

class QCRateController:
    # Top level
    # Note: match IMU_GYRO_RATEMAX (default 400Hz) to genesis sim_dt=0.0025

    def __init__(self, num_envs, dt, rotors: Sequence[Rotor], rpm_max, device, params: RateControlParams, plant: MotorPlantParams = MotorPlantParams(), dtype=torch.float32):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.params = params

        fs = 1.0 / dt
        self._flu_to_frd = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)

        self.gyro_lpf = LowPassFilter2((num_envs, 3), fs, params.IMU_GYRO_CUTOFF, device, dtype)
        self.dgyro_lpf = AlphaFilter((num_envs, 3), fs, params.IMU_DGYRO_CUTOFF, device, dtype)
        self.rate_prev = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.rate_control = RateControl(num_envs, params, device, dtype)
        self.allocator = ControlAllocator(rotors, num_envs, params.MC_AIRMODE, params.CA_R_SLEW, device, dtype)
        self.motors = MotorOutput(num_envs, len(rotors), rpm_max, params.THR_MDL_FAC, plant, dt, device, dtype)

        self._status_every = max(1, math.ceil(params.status_interval_s / dt - 1e-9))
        self._step = 0
        self.unallocated_torque = torch.zeros((num_envs, 3), device=device, dtype=dtype) # logging

    def update(self, thrust_sp, rates_sp, rates, landed: Optional[torch.Tensor]=None):
        frd = self._flu_to_frd

        rate = self.gyro_lpf.apply(rates * frd)
        accel = self.dgyro_lpf.apply((rate - self.rate_prev) / self.dt)
        self.rate_prev = rate

        torque = self.rate_control.update(rate, rates_sp * frd, accel, self.dt, landed)
        torque = torch.nan_to_num(torque, nan=0.0, posinf=0.0, neginf=0.0)

        c = torch.cat([torque, -thrust_sp.unsqueeze(-1)], dim=-1)
        u = self.allocator.allocate(c, self.dt)
        self._step += 1
        if self._step % self._status_every == 0:
            sat_pos, sat_neg, self.unallocated_torque = self.allocator.saturation_status(c)
            self.rate_control.set_saturation_status(sat_pos, sat_neg) # used from next cycle

        return self.motors.step(u)

    def thrust_setpoint_for_thrust_fraction(self, f):
        # eg hover: f = weight / (n_rotors * max_motor_thrust)
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

# Genesis helper 
def rotors_from_genesis(drone, prop_link_names, spin, ct=6.5, com_offset_flu=(0.0, 0.0, 0.0)):
    from genesis.utils.geom import inv_quat, transform_by_quat
    km = drone.KM / drone.KF
    idx = [drone.get_link(n).idx_local for n in prop_link_names]
    p_links = drone.get_links_pos(idx)[0]
    p_base = drone.get_pos()[0]
    q_inv = inv_quat(drone.get_quat()[0])
    com = torch.as_tensor(com_offset_flu, device=p_links.device, dtype=p_links.dtype)
    rotors = []
    for i in range(len(idx)):
        r = transform_by_quat(p_links[i] - p_base, q_inv) - com #body flu
        rotors.append(Rotor(px=float(r[0]), py=float(-r[1]), km=float(-spin[i]*km), ct=ct)) # allocator expects FRD
    return rotors