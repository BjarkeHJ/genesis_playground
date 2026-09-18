import torch


class QuadcopterMixer:
    """Maps [thrust, tau_x, tau_y, tau_z] to per-motor RPM via the fixed rigid-body allocation matrix.

    Matches Genesis's own per-propeller physics exactly (accessor.py kernel_set_drone_rpm):
        F_i   = kf * rpm_i^2                  (thrust, along body +z, applied at the propeller's own link)
        tau_i = km * rpm_i^2 * spin_i          (reaction torque about body z)
    Because F_i acts at the propeller's own offset (x_i, y_i) from the COM, it also
    contributes roll/pitch moment: tau_x = sum(y_i * F_i), tau_y = -sum(x_i * F_i).
    """

    def __init__(self, kf: float, km: float, prop_offsets: torch.Tensor, spin: torch.Tensor, device):
        # prop_offsets: [n_motors, 3] body-frame (x, y, z) of each propeller relative to the COM
        # spin: [n_motors] +1/-1 per motor, matches propellers_spin
        x, y = prop_offsets[:, 0], prop_offsets[:, 1]
        n_motors = prop_offsets.shape[0]

        A = torch.zeros((4, n_motors), device=device, dtype=torch.float32)
        A[0, :] = kf
        A[1, :] = kf * y
        A[2, :] = -kf * x
        A[3, :] = km * spin

        self.A_pinv = torch.linalg.pinv(A)  # [n_motors, 4], pinv so layouts other than 4 motors still work

    def wrench_to_rpm(self, thrust: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        # thrust: [n_envs], torque: [n_envs, 3] -> rpm: [n_envs, n_motors]
        wrench = torch.cat([thrust.unsqueeze(-1), torque], dim=-1)
        rpm_sq = torch.clamp(wrench @ self.A_pinv.T, min=0.0)
        return torch.sqrt(rpm_sq)


class RateController:
    """PID on body-rate error -> body torque command.

    Uses derivative-on-measurement (d(rate_meas)/dt) rather than derivative-on-error,
    since the setpoint here is the RL policy's raw action and can jump step to step --
    differentiating it directly would spike the torque command every time it does.
    """

    def __init__(self, kp, ki, kd, num_envs: int, device, integral_limit: float = None):
        self.kp = torch.as_tensor(kp, device=device, dtype=torch.float32)
        self.ki = torch.as_tensor(ki, device=device, dtype=torch.float32)
        self.kd = torch.as_tensor(kd, device=device, dtype=torch.float32)
        self.integral_limit = integral_limit
        self.integral = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self.prev_meas = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)

    def reset(self, envs_idx):
        self.integral[envs_idx] = 0.0
        self.prev_meas[envs_idx] = 0.0

    def compute(self, rate_cmd: torch.Tensor, rate_meas: torch.Tensor, dt: float) -> torch.Tensor:
        error = rate_cmd - rate_meas
        self.integral = self.integral + error * dt
        if self.integral_limit is not None:
            self.integral = torch.clamp(self.integral, -self.integral_limit, self.integral_limit)
        d_meas = (rate_meas - self.prev_meas) / dt
        self.prev_meas = rate_meas.clone()
        return self.kp * error + self.ki * self.integral - self.kd * d_meas


class QuadcopterController:
    def __init__(self, mixer: QuadcopterMixer, rate_pid: RateController):
        self.mixer = mixer
        self.rate_pid = rate_pid

    def reset(self, envs_idx):
        self.rate_pid.reset(envs_idx)

    def compute_rpms(self, thrust_cmd: torch.Tensor, rate_cmd: torch.Tensor, rate_meas: torch.Tensor, dt: float) -> torch.Tensor:
        torque_cmd = self.rate_pid.compute(rate_cmd, rate_meas, dt)
        return self.mixer.wrench_to_rpm(thrust_cmd, torque_cmd)
