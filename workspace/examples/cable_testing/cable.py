from typing import NamedTuple

import torch

_EPS = 1e-9


def _rotate_vector_by_quat(v: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate `v` by the unit quaternion `quat` (w, x, y, z), broadcasting over leading dims."""
    q_w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def attach_point_kinematics(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    link_vel: torch.Tensor,
    link_ang: torch.Tensor,
    local_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r = _rotate_vector_by_quat(local_offset, link_quat.unsqueeze(1))
    pos = link_pos.unsqueeze(1) + r
    vel = link_vel.unsqueeze(1) + torch.cross(link_ang.unsqueeze(1), r, dim=-1)
    return pos, vel


class CableWrench(NamedTuple):
    drone_force: torch.Tensor  # [n_envs, num_cables, 3], world frame
    payload_force: torch.Tensor  # [n_envs, num_cables, 3], world frame
    tension: torch.Tensor  # [n_envs, num_cables]


# Model the tether forces between drone and payload.
# Modelled as independent unilateral spring-damper systems
class TetherModel:
    """
    Unilateral spring-damper force law for `num_cables` independent tether cables, each connecting
    a world-frame attach point on the drone to one on the payload.
    """

    def __init__(self, cable_cfg: dict, num_envs: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.num_envs = num_envs
        self.device = device
        self.num_cables = len(cable_cfg["cables"])

        def stacked_scalar(key: str) -> torch.Tensor:
            return torch.tensor([c[key] for c in cable_cfg["cables"]], dtype=dtype, device=device).reshape(
                1, self.num_cables
            )

        self.stiffness = stacked_scalar("stiffness")
        self.damping = stacked_scalar("damping")
        self.rest_length = stacked_scalar("rest_length")
        self.cable_diameter = stacked_scalar("cable_diameter")

        slack_transition_width = stacked_scalar("slack_transition_width")
        self.activation_delta = torch.clamp(slack_transition_width * self.rest_length, min=_EPS)

        self.air_density = 1.225  # [kg/m^3], sea level
        self.cylinder_drag_coeff = 1.2  # cylinder in subcritical crossflow

    def compute(
        self,
        drone_pos: torch.Tensor,
        drone_vel: torch.Tensor,
        payload_pos: torch.Tensor,
        payload_vel: torch.Tensor,
    ) -> CableWrench:
        vec = payload_pos - drone_pos
        length = torch.linalg.norm(vec, dim=-1)
        uhat = vec / torch.clamp(length, min=_EPS).unsqueeze(-1)

        extension = length - self.rest_length
        activation = 0.5 * (1.0 + torch.tanh(extension / self.activation_delta))
        extension_rate = ((payload_vel - drone_vel) * uhat).sum(dim=-1)
        tension = activation * (self.stiffness * extension + self.damping * extension_rate)
        tension = torch.clamp(tension, min=0.0)

        force_on_payload = -tension.unsqueeze(-1) * uhat
        force_on_drone = -force_on_payload
        
        half_drag = self._crossflow_drag(drone_vel, payload_vel, uhat, length)
        force_on_drone = force_on_drone + half_drag
        force_on_payload = force_on_payload + half_drag

        return CableWrench(
            drone_force=force_on_drone,
            payload_force=force_on_payload,
            tension=tension,
        )

    def _crossflow_drag(
        self, v_drone: torch.Tensor, v_payload: torch.Tensor, uhat: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        """Half of the cylinder-crossflow drag on each cable, applied at both endpoints."""
        v_cable = 0.5 * (v_drone + v_payload)
        v_perp = v_cable - (v_cable * uhat).sum(dim=-1, keepdim=True) * uhat
        v_perp_mag = torch.linalg.norm(v_perp, dim=-1, keepdim=True)

        f_drag = (
            0.5
            * self.air_density
            * self.cable_diameter.unsqueeze(-1)
            * self.cylinder_drag_coeff
            * v_perp_mag**2
            * length.unsqueeze(-1)
        )
        v_perp_dir = v_perp / torch.clamp(v_perp_mag, min=_EPS)
        return -0.5 * f_drag * v_perp_dir
