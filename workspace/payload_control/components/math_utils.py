import math
import torch

# Batched torch helpers: frames/rotations (pure torch, so torch.compile fuses them into the calling stage) and random sampling

# ====== FRAMES / ROTATIONS ======

def rotate_to_heading(v, yaw):
    # World ENU vector -> heading frame: rotate by -yaw about world z. Gravity stays on z, no tilt involved
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=-1)

def wrap_angle(a):
    # Wrap to (-pi, pi]
    return torch.atan2(torch.sin(a), torch.cos(a))

def yaw_to_quat(yaw):
    # Pure-yaw rotation about world z, wxyz
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)

def rotate_by_quat(v, quat, inverse=False):
    # Rotate v by quat (wxyz), or by its inverse: same math as gu.transform_by_quat(v, [inv_quat(quat)]), written inline so
    # torch.compile fuses it into the caller (the Genesis helper is a separately compiled call with its own dispatch cost)
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    if inverse:
        q_x, q_y, q_z = -q_x, -q_y, -q_z
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2
    v_x, v_y, v_z = (v / (q_ww + q_xx + q_yy + q_zz)[..., None]).unbind(-1)
    return torch.stack([
        v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy),
        v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx),
        v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz),
    ], dim=-1)

# ====== RANDOM SAMPLING ======
def gs_rand_float(lower, upper, shape, device):
    # Uniform in [lower, upper)
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def rand_uniform(rng, n, device):
    return gs_rand_float(rng[0], rng[1], (n, ), device)

def rand_log_uniform(rng, n, device):
    return torch.exp(gs_rand_float(math.log(rng[0]), math.log(rng[1]), (n, ), device))

NOMINAL_SCALE = (1.0, 1.0) # multiplicative scale range of a parameter left at nominal (still drawn, so the RNG stream is unchanged)
