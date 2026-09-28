import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

import torch
import math
from tensordict import TensorDict

from system_model.utils.qc_rate_control import QCRateController, rotors_from_genesis
from config_env import *
from utils import *

class PayloadControlEnv:
    def __init__(self, env_cfg: EnvConfig, show_viewer: bool=False, device: str="cuda"):
        self.show_viewer = show_viewer
        self.device = torch.device(device)

        self.cfg = env_cfg
        self.dt = self.cfg.dt # policy step
        self.sim_dt = self.cfg.sim_dt # physics / rate controller step
        self.decimation = self.cfg.decimation
        self.num_envs = self.cfg.num_envs
        self.num_actions = self.cfg.num_actions
        self.max_episode_length = math.ceil(self.cfg.episode_length_s / self.dt)

        self.sys_cfg = self.cfg.sys_cfg
        self.cmd_cfg = self.cfg.cmd_cfg
        self.rew_cfg = self.cfg.rew_cfg

        # ======================== SCENE SETUP ========================
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.cfg.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=60,
                refresh_rate=30,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(1)), 
                background_color=(0.2, 0.2, 0.2),
                shadow=False,
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=self.show_viewer,
        )

        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=True,
                collision=False
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.8, 0.3, 0.2),
                ),
            ),
        )

        drone_attach_links = [f"attach{x}_link" for x in range(self.sys_cfg.num_tethers)]
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=DRONE_PATH,
                pos=self.sys_cfg.drone_reset_pos,
                quat=self.sys_cfg.drone_reset_quat,
                propellers_link_name=self.sys_cfg.propellers_link_name,
                propellers_spin=self.sys_cfg.propellers_spin,
                prioritize_urdf_material=True,
                links_to_keep=drone_attach_links,
            ),
        )
        self.drone_attach_links = [self.drone.get_link(x) for x in drone_attach_links]
        self.drone_attach_links_idx = [link.idx_local for link in self.drone_attach_links]

        payload_attach_links = [f"attach_{x}" for x in range(self.sys_cfg.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=self.sys_cfg.payload_reset_pos,
                quat=self.sys_cfg.payload_reset_quat,
                scale=(1.0,1.0,1.0),
                links_to_keep=payload_attach_links,
            ),
        )
        self.payload_attach_links = [self.payload.get_link(x) for x in payload_attach_links]
        self.payload_attach_links_idx = [link.idx_local for link in self.payload_attach_links]

        self.scene.build(n_envs=self.num_envs)
        # ======================== END OF SCENE SETUP ========================

        # Rate Controller
        gravity = abs(self.scene.sim.options.gravity[2])
        max_motor_thrust = self.sys_cfg.twr_max * (self.drone.get_mass() + self.payload.get_mass()) * gravity / self.drone.n_propellers
        rotors = rotors_from_genesis(
            self.drone,
            self.sys_cfg.propellers_link_name,
            self.sys_cfg.propellers_spin,
            ct=self.sys_cfg.rotor_ct,
        )
        self.controller = QCRateController(
            num_envs=self.num_envs,
            dt=self.sim_dt,
            rotors=rotors,
            rpm_max=math.sqrt(max_motor_thrust / self.drone.KF),
            device=self.device,
            params=self.sys_cfg.rate_control_params,
            plant=self.sys_cfg.motor_plant_params,
            dtype=gs.tc_float
        )
        self.hover_thrust = self.controller.thrust_setpoint_for_thrust_fraction(1.0 / self.sys_cfg.twr_max)
        self.max_rates = torch.tensor([self.sys_cfg.max_roll_rate, self.sys_cfg.max_pitch_rate, self.sys_cfg.max_yaw_rate], device=self.device, dtype=gs.tc_float)

        # Initialized state
        self.drone_init_pos = torch.tensor(self.sys_cfg.drone_reset_pos, device=self.device, dtype=gs.tc_float)
        self.drone_init_quat = torch.tensor(self.sys_cfg.drone_reset_quat, device=self.device, dtype=gs.tc_float)
        self.payload_init_pos = torch.tensor(self.sys_cfg.payload_reset_pos, device=self.device, dtype=gs.tc_float)
        self.payload_init_quat = torch.tensor(self.sys_cfg.payload_reset_quat, device=self.device, dtype=gs.tc_float)
        self.mass_eff = 1.0 / (1.0 / self.drone.get_mass() + 1.0 / self.payload.get_mass())
        self.world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1) # body z-axis for tilt check

        # Environment state buffers
        self.obs_buf = torch.zeros((self.num_envs, self.cfg.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.cmd_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device, dtype=gs.tc_float)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device, dtype=gs.tc_float)
        self.crash_condition = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_bool)

        # Observation state buffers
        self.payload_pos_err = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_vel_err = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_roll_pitch = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float) 
        self.payload_roll_pitch_rates = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float) 
        self.payload_swing_angles = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.payload_swing_angles_rates = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.drone_payload_rel_yaw = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)

        self.extras= dict() # logging information

        # Reward terms (each returns its weighted per-env reward) and per-episode sums for logging
        self.reward_fns = {
            "track": self._reward_track,
            "vel_track": self._reward_vel_track,
            "swing": self._reward_min_swing,
            "smooth_actions": self._reward_smooth_actions,
            "thrust_effort": self._reward_low_thrust_effort,
            "crash": self._reward_crash,
        }
        self.episode_sums = {name: torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) for name in self.reward_fns}

        self.reset()

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        self.extras["episode"] = {} # cleared every step so rsl_rl never re-logs stale values
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)

        for name, ep_sum in self.episode_sums.items():
            self.extras["episode"][f"Episode_Reward/{name}"] = torch.mean(ep_sum[envs_idx]) / self.cfg.episode_length_s
            ep_sum[envs_idx] = 0.0
        
        self._resample_commands(envs_idx)

        # Reset drone
        self.drone.set_pos(self.drone_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.drone_init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)

        # Reset payload
        self.payload.set_pos(self.payload_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.payload.set_quat(self.payload_init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)

        # Reset internal states
        self.drone_payload_rel_yaw[envs_idx] = 0.0
        self.payload_pos_err[envs_idx] = self.commands[envs_idx] - self.payload.get_pos(envs_idx)
        self.payload_vel_err[envs_idx] = self._desired_vel(self.payload_pos_err[envs_idx]) # spawned at rest
        self.payload_roll_pitch[envs_idx] = 0.0
        self.payload_roll_pitch_rates[envs_idx] = 0.0
        self.payload_swing_angles[envs_idx] = 0.0
        self.payload_swing_angles_rates[envs_idx] = 0.0

        self.actions[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0
        self.crash_condition[envs_idx] = 0
        self.controller.reset(envs_idx, thrust_sp=self.hover_thrust) # hover after reset

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_observations()
        return self.get_observations()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions) # [-1; 1]
        thrust_des, rate_des = self._scale_actions(self.actions) # [[-1;1], [-max_rate;max_rate]]

        # Setpoints held constant (zero-order hold) while the inner loop runs at sim rate
        for _ in range(self.decimation):
            self._sim_step(thrust_des, rate_des)

        self._update_buffers()
        self._update_crash_conditions()
        self._compute_reward()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1, )))
        
        self._update_observations()

        self.prev_actions = self.actions

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _scale_actions(self, actions):
            # Thrust: piecewise linear so a=0 is hover, a=-1 is zero thrust, a=+1 is max thrust
            a_thrust = actions[:, 0]
            thrust_des = torch.where(
                a_thrust >= 0.0,
                self.hover_thrust + a_thrust * (self.sys_cfg.max_throttle - self.hover_thrust),
                self.hover_thrust * (1.0 + a_thrust),
            )
            # Rates: [-1, 1] -> [-max_rate, max_rate] rad/s
            rate_des = actions[:, 1:] * self.max_rates
            return thrust_des, rate_des
    
    def _sim_step(self, thrust_des, rate_des):
        # Read drone body rate state
        drone_body_rates = transform_by_quat(self.drone.get_ang(), inv_quat(self.drone.get_quat()))

        # Get motor RPM from ratecontroller/mixer
        rpms = self.controller.update(thrust_des, rate_des, drone_body_rates)
        self.drone.set_propellers_rpm(rpms)

        # Step tether model
        self._tethers_step()

        # Step sim scene
        self.scene.step()

    def _tethers_step(self):
        # Step tether tension-force model
        drone_attach_pos = self.drone.get_links_pos(self.drone_attach_links_idx)
        drone_attach_vel = self.drone.get_links_vel(self.drone_attach_links_idx)
        payload_attach_pos = self.payload.get_links_pos(self.payload_attach_links_idx)
        payload_attach_vel = self.payload.get_links_vel(self.payload_attach_links_idx)

        vec = payload_attach_pos - drone_attach_pos
        vec_len = torch.linalg.norm(vec, dim=-1)
        uhat = vec / torch.clamp(vec_len, min=EPS).unsqueeze(-1)

        extension = vec_len - self.sys_cfg.rest_length
        activation = torch.tanh(extension / self.sys_cfg.activation_delta)
        extension_rate = ((payload_attach_vel - drone_attach_vel) * uhat).sum(dim=-1)

        damping_impl = self.sys_cfg.damping + self.sim_dt * self.sys_cfg.stiffness
        extension_rate_impl = (self.mass_eff * extension_rate - self.sim_dt * self.sys_cfg.stiffness * extension) / (self.mass_eff + self.sim_dt * damping_impl)
        tension = activation * (self.sys_cfg.stiffness * extension + damping_impl * extension_rate_impl)
        tension = torch.where(vec_len > self.sys_cfg.rest_length, tension, torch.zeros_like(tension))
        tension = torch.clamp(tension, min=0.0)

        force_on_payload = -tension.unsqueeze(-1) * uhat
        force_on_drone = -force_on_payload

        self.drone.apply_links_external_wrench(force=force_on_drone, links_idx_local=self.drone_attach_links_idx)
        self.payload.apply_links_external_wrench(force=force_on_payload, links_idx_local=self.payload_attach_links_idx)

        relative_yaw_rate = self.payload.get_ang()[:, 2] - self.drone.get_ang()[:, 2]
        self.drone_payload_rel_yaw += relative_yaw_rate * self.sim_dt

    def _update_buffers(self):
        self.episode_length_buf += 1

        self.payload_pos_err[:] = self.commands - self.payload.get_pos()
        self.payload_vel_err[:] = self._desired_vel(self.payload_pos_err) - self.payload.get_vel()

        self.payload_roll_pitch[:] = quat_to_xyz(self.payload.get_quat(), rpy=True, degrees=False)[:, :2]
        self.payload_roll_pitch_rates[:] = transform_by_quat(self.payload.get_ang(), inv_quat(self.payload.get_quat()))[:, :2]

        r = self.payload.get_pos() - self.drone.get_pos()
        r_dot = self.payload.get_vel() - self.drone.get_vel()
        self.payload_swing_angles[:, 0] = torch.atan2(r[:, 0], -r[:, 2])
        self.payload_swing_angles[:, 1] = torch.atan2(r[:, 1], -r[:, 2])
        # analytical derivative of atan2(a, b): (b*a_dot - a*b_dot) / (a² + b²), with b=-r_z
        den_x = torch.clamp(r[:, 0]**2 + r[:, 2]**2, min=EPS)
        den_y = torch.clamp(r[:, 1]**2 + r[:, 2]**2, min=EPS)
        self.payload_swing_angles_rates[:, 0] = (r[:, 0] * r_dot[:, 2] - r[:, 2] * r_dot[:, 0]) / den_x
        self.payload_swing_angles_rates[:, 1] = (r[:, 1] * r_dot[:, 2] - r[:, 2] * r_dot[:, 1]) / den_y

    def _update_crash_conditions(self):
        drone_up_z = transform_by_quat(self.world_up, self.drone.get_quat())[:, 2]  # z-component of body z-axis
        self.crash_condition = (
              (torch.abs(self.payload_roll_pitch[:, 0]) > self.cfg.terminate_if_rollpitch_greater_than)
            | (torch.abs(self.payload_roll_pitch[:, 1]) > self.cfg.terminate_if_rollpitch_greater_than)
            | (torch.abs(self.payload_swing_angles[:, 0]) > self.cfg.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_swing_angles[:, 1]) > self.cfg.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_pos_err[:, 0]) > self.cfg.terminate_if_x_greater_than)
            | (torch.abs(self.payload_pos_err[:, 1]) > self.cfg.terminate_if_y_greater_than)
            | (torch.abs(self.payload_pos_err[:, 2]) > self.cfg.terminate_if_z_greater_than)
            | (torch.abs(self.drone_payload_rel_yaw[:]) > self.cfg.terminate_if_relyaw_greater_than)
            | (self.payload.get_pos()[:, 2] <= self.cfg.terminate_if_payload_below_z)
            | (drone_up_z < math.cos(self.cfg.terminate_if_drone_tilt_greater_than))
        )
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout | self.crash_condition
        self.extras["time_outs"] = timeout

    def _update_observations(self):
        self.obs_buf = torch.cat(
            [
                torch.clamp(self.payload_pos_err[:, 0:1] * self.cfg.obs_scales.px, -1.0, 1.0), # rel_pos_x, dim=1
                torch.clamp(self.payload_pos_err[:, 1:2] * self.cfg.obs_scales.py, -1.0, 1.0), # rel_pos_y, dim=1
                torch.clamp(self.payload_pos_err[:, 2:3] * self.cfg.obs_scales.pz, -1.0, 1.0), # rel_pos_z, dim=1
                torch.clamp(self.payload_vel_err[:, 0:1] * self.cfg.obs_scales.vx, -1.0, 1.0), # vel_err_x, dim=1
                torch.clamp(self.payload_vel_err[:, 1:2] * self.cfg.obs_scales.vy, -1.0, 1.0), # vel_err_y, dim=1
                torch.clamp(self.payload_vel_err[:, 2:3] * self.cfg.obs_scales.vz, -1.0, 1.0), # vel_err_z, dim=1
                torch.clamp(self.payload_roll_pitch * self.cfg.obs_scales.rp, -1.0, 1.0), # pitch/roll, dim=2
                torch.clamp(self.payload_roll_pitch_rates * self.cfg.obs_scales.rpr, -1.0, 1.0), # pitch/roll rates, dim=2
                torch.clamp(self.payload_swing_angles * self.cfg.obs_scales.sa, -1.0, 1.0), # swing angles (alpha and beta), dim=2
                torch.clamp(self.payload_swing_angles_rates * self.cfg.obs_scales.sar, -1.0, 1.0), # swing angles rates, dim=2
                torch.clamp(self.drone_payload_rel_yaw.unsqueeze(-1) * self.cfg.obs_scales.ryaw, -1.0, 1.0), # drone-payload relative yaw, dim=1
            ],
            axis=1
        )
    
    def _resample_commands(self, envs_idx, n=1):
        if len(envs_idx) == 0:
            return
        # TODO: Make n the number of close waypoint in n-horizon trajectory
        pl_init = self.payload_init_pos
        self.commands[envs_idx, 0] = gs_rand_float(pl_init[0] - 0.5, pl_init[0] + 0.5, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 1] = gs_rand_float(pl_init[1] - 0.5, pl_init[1] + 0.5, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 2] = gs_rand_float(pl_init[2] - 2.0, pl_init[2] + 2.0, shape=(len(envs_idx), ), device=self.device)
            
        self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _desired_vel(self, pos_err):
        # Saturated P-law: pos_err * gain, norm clipped to v_max (smooth at the target, no division by zero)
        v_des = pos_err * self.cmd_cfg.approach_gain
        speed = torch.norm(v_des, dim=-1, keepdim=True)
        return v_des * torch.clamp(self.cmd_cfg.approach_v_max / torch.clamp(speed, min=EPS), max=1.0)

    def _reward_track(self):
        # Sharper gradient closer to the target but avoiding completely flat gradient further away
        dist = torch.norm(self.payload_pos_err, dim=1)
        track_rew = torch.exp(-dist / self.rew_cfg.sigma_track_rough) * self.rew_cfg.w_track_rough # rough  may not be needed if vel_track works
        track_rew += torch.exp(-dist / self.rew_cfg.sigma_track_fine) * self.rew_cfg.w_track_fine 
        return track_rew

    def _reward_vel_track(self):
        vel_err_sq = torch.sum(torch.square(self.payload_vel_err), dim=1)
        return torch.exp(-vel_err_sq / self.rew_cfg.sigma_vel_track**2) * self.rew_cfg.w_vel_track

    def _reward_min_swing(self):
        angles_rew = torch.sum(torch.square(self.payload_swing_angles), dim=1) * self.rew_cfg.w_swing_angles
        rates_rew = torch.sum(torch.square(self.payload_swing_angles_rates), dim=1) * self.rew_cfg.w_swing_rate
        return angles_rew + rates_rew

    def _reward_smooth_actions(self):
        action_rew = torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * self.rew_cfg.w_smooth_actions
        return action_rew

    def _reward_low_thrust_effort(self):
        thrust_rew = torch.square(self.actions[:, 0]) * self.rew_cfg.w_thrust_effort
        return thrust_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = self.rew_cfg.w_crash
        return crash_rew

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_fns.items():
            rew = fn()
            self.rew_buf += rew
            self.episode_sums[name] += rew