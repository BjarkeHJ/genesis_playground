import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat, transform_by_quat, inv_quat, transform_quat_by_quat
import torch
from tensordict import TensorDict
import math
from config import *
from quadcopter_control import QuadcopterMixer, RateController, QuadcopterController

class SlungPayloadEnv:
    def __init__(self, env_cfg: EnvConfig, obs_cfg: ObservationConfig, command_cfg: CommandConfig, reward_cfg: RewardConfig, tether_cfg: TetherConfig, show_viewer: bool=False, device: str="cuda"):
        self.show_viewer = show_viewer
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.command_cfg = command_cfg
        self.reward_cfg = reward_cfg
        self.tether_cfg = tether_cfg
        self.dt = env_cfg.dt

        # What RSL_RL VecEnv/OnPolicyRunner looks for:
        self.device = torch.device(device)
        self.cfg = dataclass_to_dict(self.env_cfg)
        self.num_envs = env_cfg.num_envs
        self.num_actions = env_cfg.num_actions
        self.max_episode_length = math.ceil(env_cfg.episode_length_s / self.dt)

        # Genesis-World Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt), # physics rate

            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=60,
            ), # viewer (default 60 fps update rate)
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1)),
                                                background_color=(0.2, 0.2, 0.2)), # scene visual settings
            rigid_options=gs.options.RigidOptions(
                dt=self.dt, # double time rigid solver
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,
            ), # rigid solver settings
            show_viewer=show_viewer, # visualiser on/off
        )

        # Ground Plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        # Command target 
        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.07,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.3, 0.3, 0.9),
                    ),
                ),
            )

        # Drone
        drone_attach_links = [f"attach{x}_link" for x in range(self.tether_cfg.num_tethers)]
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=DRONE_PATH,
                pos=self.env_cfg.drone_reset_pos,
                quat=self.env_cfg.drone_reset_quat,
                propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                propellers_spin=(-1, 1, -1, 1), # per prop -1=CW, +1=CCW
                prioritize_urdf_material=True,
                links_to_keep=drone_attach_links,
            ),
        )
        self.drone_attach_links = [self.drone.get_link(x) for x in drone_attach_links]
        self.drone_attach_links_idx = [link.idx_local for link in self.drone_attach_links]

        # Payload
        payload_attach_links = [f"attach_{x}" for x in range(self.tether_cfg.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=self.env_cfg.payload_reset_pos,
                quat=self.env_cfg.payload_reset_quat,
                scale=(1.0, 1.0, 1.0),
                links_to_keep=payload_attach_links,
            ),
        )
        self.payload_attach_links = [self.payload.get_link(x) for x in payload_attach_links]
        self.payload_attach_links_idx = [link.idx_local for link in self.payload_attach_links]
        
        # Build scene
        self.scene.build(n_envs=self.num_envs)

        # Effective mass of two-body system (good approx?)
        drone_mass = self.drone.get_mass()
        payload_mass = self.payload.get_mass()
        self.mass_eff = 1.0 / (1.0 / drone_mass + 1.0 / payload_mass)
        self.total_mass = drone_mass + payload_mass
        self.thrust_max = self.env_cfg.thrust_to_weight * self.total_mass * 9.81

        # Low-level controller
        prop_link_idx = [self.drone.get_link(n).idx_local for n in ("prop0_link", "prop1_link", "prop2_link", "prop3_link")]
        prop_offsets = (self.drone.get_links_pos(prop_link_idx) - self.drone.get_pos().unsqueeze(1))[0]
        mixer = QuadcopterMixer(self.drone.KF, self.drone.KM, prop_offsets, self.drone.propellers_spin, self.device)
        rate_pid = RateController(
            kp=self.env_cfg.rate_kp, ki=self.env_cfg.rate_ki, kd=self.env_cfg.rate_kd,
            num_envs=self.num_envs, device=self.device, integral_limit=self.env_cfg.rate_integral_limit,
        )
        self.controller = QuadcopterController(mixer, rate_pid)

        # Initialized state (reset pos/quat)
        self.drone_base_link_init_pos = torch.tensor(self.env_cfg.drone_reset_pos, device=self.device, dtype=gs.tc_float)
        self.drone_base_link_init_quat = torch.tensor(self.env_cfg.drone_reset_quat, device=self.device, dtype=gs.tc_float)
        self.payload_init_pos = torch.tensor(self.env_cfg.payload_reset_pos, device=self.device, dtype=gs.tc_float)
        self.payload_init_quat = torch.tensor(self.env_cfg.payload_reset_quat, device=self.device, dtype=gs.tc_float)
        self.world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=gs.tc_float)

        # Initialize system state buffers
        self.obs_buf = torch.zeros((self.num_envs, self.obs_cfg.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.at_target_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        self.commands = torch.zeros((self.num_envs, self.command_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.next_commands = torch.zeros((self.num_envs, self.command_cfg.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)

        self.drone_base_link_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_attach_pos = torch.zeros((self.num_envs, self.tether_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)
        self.drone_attach_vel = torch.zeros((self.num_envs, self.tether_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)

        self.payload_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_last_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.payload_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_last_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_pos = torch.zeros((self.num_envs, self.tether_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_vel = torch.zeros((self.num_envs, self.tether_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)

        self.payload_pos_err = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_last_pos_err = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_payload_relative_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_payload_relative_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_payload_relative_yaw = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)

        self.extras = dict() # Extra information for logging

        self.reset()

    def get_observations(self):
            return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.drone_base_link_pos[envs_idx] = self.drone_base_link_init_pos
        self.drone_base_link_quat[envs_idx] = self.drone_base_link_init_quat
        self.drone.set_pos(self.drone_base_link_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.drone_base_link_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone_base_link_lin_vel[envs_idx] = 0.0
        self.drone_base_link_ang_vel[envs_idx] = 0.0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.payload_pos[envs_idx] = self.payload_init_pos
        self.payload_quat[envs_idx] = self.payload_init_quat
        self.payload.set_pos(self.payload_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.payload.set_quat(self.payload_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.payload_lin_vel[envs_idx] = 0.0
        self.payload_last_lin_vel[envs_idx] = 0.0
        self.payload_ang_vel[envs_idx] = 0.0
        self.payload.zero_all_dofs_velocity(envs_idx)

        self.payload_pos_err[envs_idx] = self.commands[envs_idx] - self.payload_pos[envs_idx]
        self.payload_last_pos_err[envs_idx] = self.payload_pos_err[envs_idx]
        self.drone_payload_relative_pos[envs_idx] = self.drone_base_link_pos[envs_idx] - self.payload_pos[envs_idx]
        self.drone_payload_relative_vel[envs_idx] = 0.0
        self.drone_payload_relative_yaw[envs_idx] = 0.0

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.at_target_buf[envs_idx] = 0
        self.controller.reset(envs_idx)
        self.reset_buf[envs_idx] = True # redundant True-set

        self.extras["episode"] = {}

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_observation()
        return self.get_observations()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg.clip_actions, self.env_cfg.clip_actions)
        exec_actions = self.last_actions if self.env_cfg.simulate_action_latency else self.actions

        thrust_cmd = (exec_actions[:, 0] * 0.5 + 0.5) * self.thrust_max  # [-1,1] -> [0, thrust_max]
        rate_cmd = exec_actions[:, 1:4] * self.env_cfg.max_rate

        # Update tether dynamics
        self._step_tethers()

        # Fly drone: rate PID (on last measured body rate) + mixer -> per-motor RPM
        rpms = self.controller.compute_rpms(thrust_cmd, rate_cmd, self.drone_base_link_ang_vel, self.dt)
        self.drone.set_propellers_rpm(rpms)

        # Step simulation scene
        self.scene.step()

        # Update buffers
        self.episode_length_buf += 1

        self.drone_base_link_pos[:] = self.drone.get_pos()
        self.drone_base_link_quat[:] = self.drone.get_quat()

        self.payload_last_pos[:] = self.payload_pos[:]
        self.payload_pos[:] = self.payload.get_pos()
        self.payload_quat[:] = self.payload.get_quat()

        self.payload_last_lin_vel[:] = self.payload_lin_vel[:] # save last payload velocity
        self.drone_base_link_lin_vel[:] = self.drone.get_vel() # World frame
        self.payload_lin_vel[:] = self.payload.get_vel() # World frame

        inv_base_link_quat = inv_quat(self.drone_base_link_quat)
        self.drone_base_link_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_link_quat) # Body frmae

        inv_payload_quat = inv_quat(self.payload_quat)
        self.payload_ang_vel[:] = transform_by_quat(self.payload.get_ang(), inv_payload_quat) # Body frame 

        self.payload_pos_err[:] = self.commands - self.payload_pos # Error between command and current position
        self.payload_last_pos_err[:] = self.commands - self.payload_last_pos # Error between command and previous position
        self.drone_payload_relative_pos[:] = self.drone_base_link_pos - self.payload_pos
        self.drone_payload_relative_vel[:] = self.drone_base_link_lin_vel - self.payload_lin_vel

        # Resample commands for envs with reached command setpoint
        # envs_idx = self._at_target()
        resample_idxs = self._get_resample_idxs()
        self.at_target_buf[resample_idxs] = 0 # reset dwell counter
        self._resample_commands(resample_idxs)

        # Check termination conditions
        payload_tilt = self._tilt_from_vertical(self.payload_quat)
        drone_tilt = self._tilt_from_vertical(self.drone_base_link_quat)
        self.crash_condition = (
            (torch.abs(payload_tilt[:, 0]) > self.env_cfg.terminate_if_roll_greater_than)
            | (torch.abs(payload_tilt[:, 1]) > self.env_cfg.terminate_if_pitch_greater_than)
            | (torch.abs(drone_tilt[:, 0]) >= 90.0)
            | (torch.abs(drone_tilt[:, 1]) >= 90.0)
            | (torch.abs(self.payload_pos_err[:, 0]) > self.env_cfg.terminate_if_x_greater_than)
            | (torch.abs(self.payload_pos_err[:, 1]) > self.env_cfg.terminate_if_y_greater_than)
            | (self.payload_pos_err[:, 2] > self.env_cfg.terminate_if_z_greater_than)
            | (self.drone_base_link_pos[:, 2] <= 0.5)
        )

        # Reset: Set true/false per envs
        timeout_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout_buf | self.crash_condition
        self.extras["time_outs"] = timeout_buf # rsl_rl should not treat timeout as terminal
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # Compute reward
        self._compute_reward()

        # Compute observations
        self._update_observation()
        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _step_tethers(self):
        # Drone attachment point pos/vel
        self.drone_attach_pos = self.drone.get_links_pos(self.drone_attach_links_idx)
        self.drone_attach_vel = self.drone.get_links_vel(self.drone_attach_links_idx)
        
        # Payload attachment points pos/vel
        self.payload_attach_pos = self.payload.get_links_pos(self.payload_attach_links_idx)
        self.payload_attach_vel = self.payload.get_links_vel(self.payload_attach_links_idx)

        # Calculate the implicit euler solution to unilateral spring-damper system
        vec = self.payload_attach_pos - self.drone_attach_pos # [num_envs, num_tethers, 3]
        length = torch.linalg.norm(vec, dim=-1) # norm over xyz [num_envs, num_tethers]
        uhat = vec / torch.clamp(length, min=EPS).unsqueeze(-1) # [num_envs, num_tethers, 3]
        
        extension = length - self.tether_cfg.rest_length # [num_envs, num_tethers]
        activation = torch.tanh(extension / self.tether_cfg.activation_delta) # [num_envs, num_tethers], 0 at extension=0, continuous with the hard cutoff below
        extension_rate = ((self.payload_attach_vel - self.drone_attach_vel) * uhat).sum(dim=-1) # [num_envs, num_tethers]

        # Backward euler
        damping_implicit = self.tether_cfg.damping + self.dt * self.tether_cfg.stiffness
        extension_rate_implicit = (self.mass_eff * extension_rate - self.dt * self.tether_cfg.stiffness * extension) / (self.mass_eff + self.dt * damping_implicit)
        tension = activation * (self.tether_cfg.stiffness * extension + damping_implicit * extension_rate_implicit)
        tension = torch.where(length > self.tether_cfg.rest_length, tension, torch.zeros_like(tension))
        tension = torch.clamp(tension, min=0.0)

        force_on_payload = -tension.unsqueeze(-1) * uhat
        force_on_drone = -force_on_payload

        # Apply forces on drone/payload - Automatically computes torques based on link offsets
        self.drone.apply_links_external_wrench(force_on_drone, links_idx_local=self.drone_attach_links_idx)
        self.payload.apply_links_external_wrench(force_on_payload, links_idx_local=self.payload_attach_links_idx)

        self._step_tether_twist()

    def _step_tether_twist(self):
        relative_yaw_rate = self.payload.get_ang()[: ,2] - self.drone.get_ang()[:, 2]
        self.drone_payload_relative_yaw += relative_yaw_rate * self.dt # unwrapped -> keeps accum

        twist_torque = -(
            self.tether_cfg.twist_stiffness * self.drone_payload_relative_yaw
            + self.tether_cfg.twist_damping * relative_yaw_rate
        )
        torque_vec = twist_torque.unsqueeze(-1) * self.world_up

        self.payload.apply_links_external_wrench(torque=torque_vec, links_idx_local=self.payload.base_link_idx)
        self.drone.apply_links_external_wrench(torque=-torque_vec, links_idx_local=self.drone.base_link_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = self._gs_rand_float(*self.command_cfg.pos_x_range, (len(envs_idx),))
        self.commands[envs_idx, 1] = self._gs_rand_float(*self.command_cfg.pos_y_range, (len(envs_idx),))
        self.commands[envs_idx, 2] = self._gs_rand_float(*self.command_cfg.pos_z_range, (len(envs_idx),))
        self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _tilt_from_vertical(self, quat: torch.Tensor) -> torch.Tensor:
        up = transform_by_quat(self.world_up, quat)
        roll = torch.rad2deg(torch.atan2(up[:, 1], up[:, 2]))
        pitch = torch.rad2deg(torch.atan2(up[:, 0], up[:, 2]))
        return torch.stack([roll, pitch], dim=1)

    def _at_target(self):
        return (torch.norm(self.payload_pos_err, dim=1) < self.env_cfg.at_target_th).nonzero(as_tuple=False).reshape((-1,))
    
    def _get_resample_idxs(self):
        self.at_target_buf = torch.where(
            torch.norm(self.payload_pos_err, dim=1) < self.env_cfg.at_target_th,
            self.at_target_buf + 1, 
            torch.zeros_like(self.at_target_buf)
        )

        hold_steps = int(self.env_cfg.resampling_time_s / self.dt)
        return (self.at_target_buf >= hold_steps).nonzero(as_tuple=False).reshape(-1)
    
    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                torch.clip(self.payload_pos_err * self.obs_cfg.scale_rel_pos, -1, 1),
                torch.clip(self.payload_lin_vel * self.obs_cfg.scale_lin_vel, -1, 1),
                torch.clip(self.payload_ang_vel * self.obs_cfg.scale_ang_vel, -1, 1),
                self.payload_quat,
                torch.clip(self.drone_base_link_lin_vel * self.obs_cfg.scale_lin_vel, -1, 1),
                torch.clip(self.drone_base_link_ang_vel * self.obs_cfg.scale_ang_vel, -1, 1),
                self.drone_base_link_quat,
                torch.clip(self.drone_payload_relative_pos * self.obs_cfg.scale_rel_swing, -1, 1),
                torch.clip(self.drone_payload_relative_vel * self.obs_cfg.scale_rel_swing, -1, 1),
                self.last_actions,
            ],
            dim=1
        )

    # ==== REWARDS ====
    def _reward_target(self):
        err = self.payload_pos_err
        dist_z  = torch.abs(err[:, 2])
        dist_xy = torch.norm(err[:, :2], dim=1)

        r_z  = 2.0 * torch.exp(-dist_z  / self.reward_cfg.sigma_target) - 1.0
        r_xy = 2.0 * torch.exp(-dist_xy / self.reward_cfg.sigma_target) - 1.0
        target_rew = 2.0 * r_z + r_xy

        dist_z_last  = torch.abs(self.payload_last_pos_err[:, 2])
        dist_xy_last = torch.norm(self.payload_last_pos_err[:, :2], dim=1)
        target_rew += 2.0 * (dist_z_last - dist_z) + (dist_xy_last - dist_xy)
        return target_rew

    def _reward_motion(self):
        motion_rew = 0.0

        payload_acc = (self.payload_lin_vel - self.payload_last_lin_vel) / self.dt
        rel_pos_xy = self.drone_payload_relative_pos[:, :2]
        rel_vel_xy = self.drone_payload_relative_vel[:, :2]

        close_gate = torch.exp(-torch.norm(self.payload_pos_err, dim=1) / self.reward_cfg.sigma_target)
        vel_err = self.payload_lin_vel
        
        motion_rew -= torch.sum(payload_acc * payload_acc, dim=1)
        motion_rew -= torch.sum(rel_pos_xy * rel_pos_xy, dim=1)
        motion_rew -= torch.sum(rel_vel_xy * rel_vel_xy, dim=1)
        motion_rew -= close_gate * torch.sum(vel_err * vel_err, dim=1) 
        return motion_rew

    def _reward_tension(self):
        dist_drone_pl = torch.norm(self.drone_payload_relative_pos, dim=1)
        extension_frac = (dist_drone_pl - self.tether_cfg.rest_length) / self.tether_cfg.rest_length
        tension_rew = 1.0 - torch.clip(torch.abs(extension_frac), min=0.0, max=1.0)
        return tension_rew

    def _reward_attitude(self):
        up_payload = transform_by_quat(self.world_up, self.payload_quat)
        attitude_rew = up_payload[:, 2] - 1.0 # 0 when level, negative as tilt increase
        return attitude_rew

    def _reward_action_smooth(self):
        action_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return action_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _compute_reward(self):
        self.rew_buf[:] = (
              self.reward_cfg.scale_target * self._reward_target()
            + self.reward_cfg.scale_motion * self._reward_motion()
            + self.reward_cfg.scale_attitude * self._reward_attitude()
            + self.reward_cfg.scale_action * self._reward_action_smooth()
            + self.reward_cfg.scale_tension * self._reward_tension()
            + self.reward_cfg.scale_crash * self._reward_crash()
        )

    def _gs_rand_float(self, lower, upper, shape):
        return (upper - lower) * torch.rand(size=shape, device=self.device) + lower

    
