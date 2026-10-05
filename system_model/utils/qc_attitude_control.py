import math
from dataclasses import dataclass
from typing import Optional
import torch

# Attitude stage in front of QCRateController (PX4 AttitudeControl port).
# Conventions: quaternions wxyz (genesis), world ENU / body FLU, body rates out in FLU
# (QCRateController.update converts to FRD itself).
# Euler setpoints are ZYX in FLU: roll about +x (right side down), pitch about +y (nose down), yaw about +z (CCW).

@dataclass(frozen=True)
class AttitudeControlParams:
    # PX4 Parameters (defaults)
    MC_ROLL_P: float = 6.5
    MC_PITCH_P: float = 6.5
    MC_YAW_P: float = 2.8
    MC_YAW_WEIGHT: float = 0.4 # yaw priority relative to roll/pitch in the reduced attitude mix

    MC_ROLLRATE_MAX: float = 220.0 # [deg/s]
    MC_PITCHRATE_MAX: float = 220.0 # [deg/s]
    MC_YAWRATE_MAX: float = 200.0 # [deg/s]

    yaw_hold: bool = True # True: integrate yaw setpoint from yaw rate (heading hold), False: pure yaw-rate
    yaw_err_max: float = math.radians(45.0) # [rad] cap on |yaw_sp - yaw| so the integrated setpoint cannot run away

# Batched quaternion helpers (wxyz)
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)

def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def quat_canonical(q):
    # w >= 0
    return torch.where(q[..., :1] < 0.0, -q, q)

def quat_from_euler(roll, pitch, yaw):
    # ZYX: q = q_z(yaw) * q_y(pitch) * q_x(roll)
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)

def quat_yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def dcm_z(q):
    # body z-axis expressed in world (third column of R)
    w, x, y, z = q.unbind(-1)
    return torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)

def quat_between(src, dst):
    # shortest rotation taking unit vector src to dst
    cr = torch.linalg.cross(src, dst)
    dt = (src * dst).sum(-1, keepdim=True)
    q = torch.cat([dt + torch.sqrt((src * src).sum(-1, keepdim=True) * (dst * dst).sum(-1, keepdim=True)), cr], dim=-1)
    # antiparallel: 180 deg about an axis orthogonal to src (the least aligned basis axis)
    # Branchless (no flip.any()): avoids a GPU->CPU sync per call and keeps torch.compile in one graph
    flip = (cr.norm(dim=-1) < 1e-6) & (dt.squeeze(-1) < 0.0)
    eye = torch.eye(3, device=src.device, dtype=src.dtype)
    axis = torch.linalg.cross(src, eye[src.abs().argmin(dim=-1)])
    q_flip = torch.cat([torch.zeros_like(dt), axis], dim=-1)
    q = torch.where(flip.unsqueeze(-1), q_flip, q)
    return q / q.norm(dim=-1, keepdim=True)

def wrap_pi(a):
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi

class QCAttitudeController:
    # Top level: (roll_sp, pitch_sp, yawrate_sp) + attitude -> body rate setpoint
    # Run at sim rate (every inner step) with the current attitude; the setpoint may be held from the policy.

    def __init__(self, num_envs, dt, device, params: AttitudeControlParams = AttitudeControlParams(), dtype=torch.float32):
        self.num_envs = num_envs
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
