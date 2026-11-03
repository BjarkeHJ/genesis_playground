import torch
import math

# most formulas found at: 
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles


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
    # q = w + xi + yj + zk
    # conj(q) = w - xi - yj - zk 
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def quat_canonical(q):
    # w >= 0
    return torch.where(q[..., :1] < 0.0, -q, q)

def quat_from_euler(roll, pitch, yaw):
    # ZYX: q = q_z(yaw) * q_y(pitch) * q_x(roll)
    cr = torch.cos(0.5 * roll)
    sr = torch.sin(0.5 * roll)
    cp = torch.cos(0.5 * pitch)
    sp = torch.sin(0.5 * pitch)
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ], dim=-1)

def quat_rotate(q, v):
    # rotate v by q: v' = v + w*t + q_vec x t, t = 2 * q_vec x v
    q_vec = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_vec, v)
    return v + q[..., :1] * t + torch.linalg.cross(q_vec, t)

def quat_rotate_inverse(q, v):
    # rotate v by q^-1 (eg world -> body for q body->world)
    return quat_rotate(quat_conj(q), v)

def quat_yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def dcm_z(q):
    # direct-cosine-matrix but only z 
    # body z-axis expressed in world (third colum of R)
    # assuming world up is [0, 0, 1]
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        2.0 * (x * z + w * y), 
        2.0 * (y * z - w * x), 
        1.0 - 2.0 * (x * x + y * y)
    ], dim=-1)

def quat_between(src, dst):
    # shortes rotation taking unit vector src to dst
    cr = torch.linalg.cross(src, dst)
    dt = (src * dst).sum(-1, keepdim=True)
    q = torch.cat([dt + torch.sqrt((src * src).sum(-1, keepdim=True) * (dst * dst).sum(-1, keepdim=True)), cr], dim=-1)
    
    # antiparallel: 180 deg about an exis orthogonal to src (the least aligned basis axis)
    flip = (cr.norm(dim=-1) < 1e-6) & (dt.squeeze(-1) < 0.0)
    eye = torch.eye(3, device=src.device, dtype=src.dtype)
    axis = torch.linalg.cross(src, eye[src.abs().argmin(dim=-1)])
    q_flip = torch.cat([torch.zeros_like(dt), axis], dim=-1)
    q = torch.where(flip.unsqueeze(-1), q_flip, q)
    return q / q.norm(dim=-1, keepdim=True)

def wrap_pi(a):
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi