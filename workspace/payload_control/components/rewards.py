import genesis as gs
import torch

class RewardsMixin:
    # Reward terms of PayloadControlEnv, computed from its state buffers. Each name in REWARD_TERMS maps to a method
    # _reward_<name> returning its weighted per-env reward. The names are the logged Episode_Reward/<name> tags
    REWARD_TERMS = ("track", "vmax", "damping", "swing_energy", "tilt", "yaw_error", "yaw_damping", "smooth_actions", "action_bound", "crash")

    def _init_rewards(self):
        self.reward_fns = {name: getattr(self, f"_reward_{name}") for name in self.REWARD_TERMS}
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.episode_sums = torch.zeros((len(self.REWARD_TERMS), self.num_envs), device=self.device, dtype=gs.tc_float) # rows ordered as REWARD_TERMS
        self.w_smooth_actions = torch.tensor(self.rew_cfg.w_smooth_actions, device=self.device, dtype=gs.tc_float)

    def _compute_reward(self):
        rews = torch.stack([fn() for fn in self.reward_fns.values()]) # (num_terms, N)
        self.rew_buf[:] = rews.sum(dim=0)
        self.episode_sums += rews

    def _log_episode_rewards(self, envs_idx):
        # Per-second mean of each term over the finished episodes of envs_idx, then restart their sums
        ep_means = torch.mean(self.episode_sums[:, envs_idx], dim=1) / self.cfg.episode_length_s
        for i, name in enumerate(self.REWARD_TERMS):
            self.extras["episode"][f"Episode_Reward/{name}"] = ep_means[i]
        self.episode_sums[:, envs_idx] = 0.0

    def _reward_track(self):
        # Sharper gradient closer to the target but avoiding completely flat gradient further away
        dist = torch.norm(self.payload_pos_err, dim=1)
        track_rew = torch.exp(-dist / self.rew_cfg.sigma_track_rough) * self.rew_cfg.w_track_rough
        track_rew += torch.exp(-dist / self.rew_cfg.sigma_track_fine) * self.rew_cfg.w_track_fine
        return track_rew

    def _reward_vmax(self):
        overspeed = torch.clamp(torch.norm(self.payload_vel, dim=1) - self.rew_cfg.v_max, min=0.0)
        vmax_rew = torch.square(overspeed) * self.rew_cfg.w_vmax
        return vmax_rew

    def _reward_damping(self):
        # Penalise payload speed only near the target: forces braking without slowing the approach
        # NOTE: assumes a static setpoint. for trajectory tracking use (payload_vel - ref_vel) instead
        dist = torch.norm(self.payload_pos_err, dim=1)
        speed = torch.norm(self.payload_vel, dim=1)
        damping_rew = speed * torch.exp(-dist / self.rew_cfg.sigma_damping) * self.rew_cfg.w_damping
        return damping_rew

    def _reward_swing_energy(self):
        # Linear in amplitude so small residual swing near the target is still penalised
        swing_energy_rew = self.swing_amp * self.rew_cfg.w_swing_energy
        return swing_energy_rew

    def _reward_yaw_error(self):
        # Payload heading tracking, pseudo-Huber: linear outside delta so small errors are still worth correcting,
        # quadratic inside so the pull fades near zero instead of driving bang-bang yaw commands
        d = self.rew_cfg.delta_yaw
        yaw_rew = (torch.sqrt(torch.square(self.payload_yaw_err) + d**2) - d) * self.rew_cfg.w_yaw
        return yaw_rew

    def _reward_yaw_damping(self):
        # Yaw analogue of _reward_damping: penalise payload yaw rate only near the yaw target, so turning past it is
        # worse than arriving slowly (braking) and residual twist swing at the target costs. Drone yaw is left free
        d = self.rew_cfg.delta_yaw_damping
        yaw_rate = self.payload_body_rates[:, 2] # payload tilt stays small, so body z ~ world z
        gate = torch.exp(-torch.abs(self.payload_yaw_err) / self.rew_cfg.sigma_yaw_damping)
        yaw_damping_rew = (torch.sqrt(torch.square(yaw_rate) + d**2) - d) * gate * self.rew_cfg.w_yaw_damping
        return yaw_damping_rew

    def _reward_tilt(self):
        # Payload attitude away from level: 1 - cos(tilt) ~ tilt²/2, so small tilts during manoeuvres are nearly free
        # (unlike the linear swing term) while large tilts are increasingly penalised
        tilt_rew = (1.0 + self.payload_proj_g[:, 2]) * self.rew_cfg.w_tilt
        return tilt_rew

    def _reward_action_bound(self):
        # Pre-clip actions beyond the clip bound: once the policy mean sits past the bound every sample clips to the same
        # value, so PPO gets no gradient to bring it back (saturation trap). Zero inside the bound
        excess = torch.clamp(torch.abs(self.actions_raw) - self.cfg.clip_actions, min=0.0)
        action_bound_rew = torch.sum(torch.square(excess), dim=1) * self.rew_cfg.w_action_bound
        return action_bound_rew

    def _reward_smooth_actions(self):
        # Per-axis weights (w_smooth_actions: [thrust, roll, pitch, yawrate])
        action_rew = torch.sum(torch.square(self.actions - self.prev_actions) * self.w_smooth_actions, dim=1)
        return action_rew

    def _reward_on_trajectory(self):
        traj_rew = 0.0
        # Some error between line-segment between trajectory setpoints.
        # Distance from payload projected onto line (3D)
        # Will it be any different than simply following the line from payload pos to target pos - i dont think so??

        return traj_rew

    def _reward_crash(self):
        crash_rew = torch.where(self.crash_condition, self.rew_cfg.w_crash, 0.0)
        return crash_rew
