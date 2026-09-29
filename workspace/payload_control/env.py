import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

import torch
import math
from tensordict import TensorDict

from system_model.utils.qc_rate_control import QCRateController, rotors_from_genesis
from system_model.utils.qc_attitude_control import QCAttitudeController, quat_yaw
from config_env import *
from utils import *

def rotate_to_heading(v, yaw):
    # World ENU vector -> heading frame: rotate by -yaw about world z. Gravity stays on z, no tilt involved
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=-1)

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
        self.gravity = abs(self.scene.sim.options.gravity[2])
        max_motor_thrust = self.sys_cfg.twr_max * (self.drone.get_mass() + self.payload.get_mass()) * self.gravity / self.drone.n_propellers
        rotors = rotors_from_genesis(
            self.drone,
            self.sys_cfg.propellers_link_name,
            self.sys_cfg.propellers_spin,
            ct=self.sys_cfg.rotor_ct,
        )
        self.rate_ctrl = QCRateController(
            num_envs=self.num_envs,
            dt=self.sim_dt,
            rotors=rotors,
            rpm_max=math.sqrt(max_motor_thrust / self.drone.KF),
            device=self.device,
            params=self.sys_cfg.rate_control_params,
            plant=self.sys_cfg.motor_plant_params,
            dtype=gs.tc_float
        )
        self.att_ctrl = QCAttitudeController(
            num_envs=self.num_envs,
            dt=self.sim_dt,
            device=self.device,
            params=self.sys_cfg.attitude_control_params,
            dtype=gs.tc_float,
        )
        self.hover_thrust = self.rate_ctrl.thrust_setpoint_for_thrust_fraction(1.0 / self.sys_cfg.twr_max)
        self.min_cos_tilt = math.cos(self.sys_cfg.tilt_comp_max)

        # Initialized state
        self.mass_eff = 1.0 / (1.0 / self.drone.get_mass() + 1.0 / self.payload.get_mass())
        self.drone_init_pos = torch.tensor(self.sys_cfg.drone_reset_pos, device=self.device, dtype=gs.tc_float)
        self.drone_init_quat = torch.tensor(self.sys_cfg.drone_reset_quat, device=self.device, dtype=gs.tc_float)
        self.payload_init_pos = torch.tensor(self.sys_cfg.payload_reset_pos, device=self.device, dtype=gs.tc_float)
        self.payload_init_quat = torch.tensor(self.sys_cfg.payload_reset_quat, device=self.device, dtype=gs.tc_float)
        self.world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1) # body z-axis for tilt check

        s = self.cfg.obs_scales
        self.pos_obs_scale = torch.tensor([s.px, s.py, s.pz], device=self.device, dtype=gs.tc_float)
        self.vel_obs_scale = torch.tensor([s.vx, s.vy, s.vz], device=self.device, dtype=gs.tc_float)

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

        self.payload_ref_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) 
        self.payload_swing_ref = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float) # heading frame
        self.payload_swing_cone = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) # total swing angle

        self.drone_roll_pitch = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.drone_yaw = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)

        self.extras= dict() # logging information

        # Reward terms (each returns its weighted per-env reward) and per-episode sums for logging
        self.reward_fns = {
            "track": self._reward_track,
            "vel_track": self._reward_vel_track,
            "swing": self._reward_excess_swing,
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
        drone_quat0 = self.drone_init_quat.expand(n, -1)
        self.drone.set_pos(self.drone_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(drone_quat0, zero_velocity=True, envs_idx=envs_idx)

        # Reset payload
        self.payload.set_pos(self.payload_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.payload.set_quat(self.payload_init_quat.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)

        # Reset internal states (at rest)
        self.drone_payload_rel_yaw[envs_idx] = 0.0
        self.drone_roll_pitch[envs_idx] = 0.0
        self.drone_yaw[envs_idx] = quat_yaw(drone_quat0)
        self.payload_pos_err[envs_idx] = self.commands[envs_idx] - self.payload.get_pos(envs_idx)

        v_des0 = self._desired_vel(self.payload_pos_err[envs_idx])
        self.payload_vel_err[envs_idx] = v_des0
        self.payload_ref_acc[envs_idx] = self._reference_accel(self.payload_pos_err[envs_idx], v_des0, torch.zeros_like(v_des0))
        self.payload_roll_pitch[envs_idx] = 0.0
        self.payload_roll_pitch_rates[envs_idx] = 0.0
        self.payload_swing_angles[envs_idx] = 0.0
        self.payload_swing_angles_rates[envs_idx] = 0.0

        self.payload_swing_ref[envs_idx] = 0.0
        self.payload_swing_cone[envs_idx] = 0.0

        self.actions[envs_idx] = 0
        self.prev_actions[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0
        self.crash_condition[envs_idx] = 0
        self.rate_ctrl.reset(envs_idx, thrust_sp=self.hover_thrust) # hover after reset
        self.att_ctrl.reset(envs_idx, quat=drone_quat0) # yaw setpoint = current heading

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_observations()
        return self.get_observations()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions) # [-1; 1]
        thrust_sp, roll_sp, pitch_sp, yawrate_sp = self._scale_actions(self.actions)

        # Setpoints held constant (zero-order hold) while the inner loop runs at sim rate
        for _ in range(self.decimation):
            self._sim_step(thrust_sp, roll_sp, pitch_sp, yawrate_sp)

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
            thrust_sp = torch.where(
                a_thrust >= 0.0,
                self.hover_thrust + a_thrust * (self.sys_cfg.max_throttle - self.hover_thrust),
                self.hover_thrust * (1.0 + a_thrust),
            )

            # a[1:3] roll/pitch setpoints, a[3] yaw-rate setpoint
            roll_sp = actions[:,1] * self.sys_cfg.max_tilt
            pitch_sp = actions[:,2] * self.sys_cfg.max_tilt
            yawrate_sp = actions[:,3] * self.sys_cfg.max_yaw_rate

            if self.sys_cfg.thrust_tilt_comp:
                cos_tilt = torch.clamp(torch.cos(roll_sp) * torch.cos(pitch_sp), min=self.min_cos_tilt)
                thrust_sp = torch.clamp(thrust_sp / cos_tilt, max=self.sys_cfg.max_throttle)
            
            return thrust_sp, roll_sp, pitch_sp, yawrate_sp
    
    def _sim_step(self, thrust_sp, roll_sp, pitch_sp, yawrate_sp):
        # Read drone body rate state
        drone_quat = self.drone.get_quat()
        drone_body_rates = transform_by_quat(self.drone.get_ang(), inv_quat(drone_quat))

        # Get motor RPM from ratecontroller/mixer
        rate_sp = self.att_ctrl.update(drone_quat, roll_sp, pitch_sp, yawrate_sp)
        rpms = self.rate_ctrl.update(thrust_sp, rate_sp, drone_body_rates)
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

        # Drone tilt
        drone_quat = self.drone.get_quat()
        self.drone_roll_pitch[:] = quat_to_xyz(drone_quat, rpy=True, degrees=False)[:, :2]
        self.drone_yaw[:] = quat_yaw(drone_quat)
        yaw = self.drone_yaw

        # Payload tracking errors + reference acceleration
        payload_pos = self.payload.get_pos()
        payload_vel = self.payload.get_vel()
        self.payload_pos_err[:] = self.commands - payload_pos
        v_des = self._desired_vel(self.payload_pos_err)
        self.payload_vel_err[:] = v_des - payload_vel
        self.payload_ref_acc[:] = self._reference_accel(self.payload_pos_err, v_des, payload_vel)

        self.payload_roll_pitch[:] = quat_to_xyz(self.payload.get_quat(), rpy=True, degrees=False)[:, :2]
        self.payload_roll_pitch_rates[:] = transform_by_quat(self.payload.get_ang(), inv_quat(self.payload.get_quat()))[:, :2]

        # Swing in the heading frame. r_dot is the inertial relative velocity expresed in headed axes
        r_w = payload_pos - self.drone.get_pos()
        r = rotate_to_heading(r_w, yaw)
        r_dot = rotate_to_heading(payload_vel - self.drone.get_vel(), yaw)
        self.payload_swing_angles[:, 0] = torch.atan2(r[:, 0], -r[:, 2])
        self.payload_swing_angles[:, 1] = torch.atan2(r[:, 1], -r[:, 2])
        # analytical derivative of atan2(a, b): (b*a_dot - a*b_dot) / (a² + b²), with b=-r_z
        den_x = torch.clamp(r[:, 0]**2 + r[:, 2]**2, min=EPS)
        den_y = torch.clamp(r[:, 1]**2 + r[:, 2]**2, min=EPS)
        self.payload_swing_angles_rates[:, 0] = (r[:, 0] * r_dot[:, 2] - r[:, 2] * r_dot[:, 0]) / den_x
        self.payload_swing_angles_rates[:, 1] = (r[:, 1] * r_dot[:, 2] - r[:, 2] * r_dot[:, 1]) / den_y

        # Quasi-static swing the reference acceleration requires
        # T*q = m*(a + g*e_Z), q = unit(drone - payload) = -r/|r| -> swing_x = atan2(-a_x, g + a_z)
        a_ref = rotate_to_heading(self.payload_ref_acc, yaw)
        g_plus_az = torch.clamp(self.gravity + a_ref[:, 2], min=EPS)
        self.payload_swing_ref[:, 0] = torch.atan2(-a_ref[:, 0], g_plus_az)
        self.payload_swing_ref[:, 1] = torch.atan2(-a_ref[:, 1], g_plus_az)

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
        s = self.cfg.obs_scales
        pos_err_h = rotate_to_heading(self.payload_pos_err, self.drone_yaw)
        vel_err_h = rotate_to_heading(self.payload_vel_err, self.drone_yaw)

        self.obs_buf = torch.cat(
            [
                # -- Task (heading frame)
                torch.clamp(pos_err_h * self.pos_obs_scale, -1.0, 1.0), # payload pos err, dim=3
                torch.clamp(vel_err_h * self.vel_obs_scale, -1.0, 1.0), # payload vel err vs approach law, dim=3
                # -- Payload
                torch.clamp(self.payload_roll_pitch * s.rp, -1.0, 1.0), # payload roll/pitch, dim=2
                torch.clamp(self.payload_roll_pitch_rates * s.rpr, -1.0, 1.0), # payload roll/pitch rates, dim=2
                torch.clamp(self.payload_swing_angles * s.sa, -1.0, 1.0), # swing angles (heading frame), dim=2
                torch.clamp(self.payload_swing_angles_rates * s.sar, -1.0, 1.0), # swing angles rates (heading frame), dim=2
                torch.clamp(self.drone_payload_rel_yaw.unsqueeze(-1) * s.ryaw, -1.0, 1.0), # drone-payload relative yaw, dim=1
                # -- Drone
                torch.clamp(self.drone_roll_pitch * s.drp, -1.0, 1.0), # drone tilt, dim=2
                self.actions, # last setpoint [thrust, roll_sp, pitch_sp, yawrate_sp], dim=4
            ],
            axis=1
        )
    
    def _resample_commands(self, envs_idx, n=1):
        if len(envs_idx) == 0:
            return
        # TODO: Make n the number of close waypoint in n-horizon trajectory
        pl_init = self.payload_init_pos
        self.commands[envs_idx, 0] = gs_rand_float(pl_init[0] - 3.0, pl_init[0] + 3.0, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 1] = gs_rand_float(pl_init[1] - 3.0, pl_init[1] + 3.0, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 2] = gs_rand_float(pl_init[2] - 2.0, pl_init[2] + 2.0, shape=(len(envs_idx), ), device=self.device)
            
        self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _desired_vel(self, pos_err):
        # Constant-decel (sqrt) profile far out, linear near the target, capped at cruise speed
        dist = torch.norm(pos_err, dim=-1, keepdim=True)
        speed = torch.minimum(torch.sqrt(2 * self.cmd_cfg.a_brake * dist), self.cmd_cfg.approach_gain * dist)
        speed = torch.clamp(speed, max=self.cmd_cfg.approach_v_max)
        return pos_err / torch.clamp(dist, min=EPS) * speed

    def _reference_accel(self, pos_err, v_des, vel):
        c = self.cmd_cfg
        dist = torch.norm(pos_err, dim=-1, keepdim=True)
        u = pos_err / torch.clamp(dist, min=EPS)
        s_brake = torch.sqrt(2 * c.a_brake * dist)
        s_lin = c.approach_gain * dist
        decel = torch.where(s_lin <= s_brake, c.approach_gain**2 * dist, torch.full_like(dist, c.a_brake))
        decel = torch.where(torch.minimum(s_brake, s_lin) >= c.approach_v_max, torch.zeros_like(dist), decel)
        a = -u * decel + c.vel_fb_gain * (v_des - vel)
        a_norm = torch.norm(a, dim=-1, keepdim=True)
        return a * torch.clamp(c.a_ref_max / torch.clamp(a_norm, min=EPS), max=1.0)

    def _reward_track(self):
        # Sharper gradient closer to the target but avoiding completely flat gradient further away
        dist = torch.norm(self.payload_pos_err, dim=1)
        track_rew = torch.exp(-dist / self.rew_cfg.sigma_track_rough) * self.rew_cfg.w_track_rough # rough  may not be needed if vel_track works
        track_rew += torch.exp(-dist / self.rew_cfg.sigma_track_fine) * self.rew_cfg.w_track_fine 
        return track_rew

    def _reward_vel_track(self):
        vel_err_sq = torch.sum(torch.square(self.payload_vel_err), dim=1)
        return torch.exp(-vel_err_sq / self.rew_cfg.sigma_vel_track**2) * self.rew_cfg.w_vel_track

    def _reward_excess_swing(self):
        excess = self.payload_swing_angles - self.payload_swing_ref
        angles_rew = torch.sum(torch.square(excess), dim=1) * self.rew_cfg.w_swing_angles
        rates_rew = torch.sum(torch.square(self.payload_swing_angles_rates), dim=1) * self.rew_cfg.w_swing_rate
        return angles_rew + rates_rew

    def _reward_smooth_actions(self):
        action_rew = torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * self.rew_cfg.w_smooth_actions
        return action_rew

    def _reward_low_thrust_effort(self):
        thrust_rew = torch.square(self.actions[:, 0]) * self.rew_cfg.w_thrust_effort
        return thrust_rew

    def _reward_on_trajectory(self):
        traj_rew = 0.0
        # Some error between line-segment between trajectory setpoints. 
        # Distance from payload projected onto line (3D)
        # Will it be any different than simply following the line from payload pos to target pos - i dont think so??
        
        return traj_rew

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