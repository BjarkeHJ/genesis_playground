import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat

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

def wrap_angle(a):
    # Wrap to (-pi, pi]
    return torch.atan2(torch.sin(a), torch.cos(a))

def tether_forces(drone_attach_pos, drone_attach_vel, payload_attach_pos, payload_attach_vel, rest_length, activation_delta, stiffness, damping, mass_eff, dt):
    # Tension-force model per tether -> force on the drone attach points (payload gets the opposite)
    vec = payload_attach_pos - drone_attach_pos
    vec_len = torch.linalg.norm(vec, dim=-1)
    uhat = vec / torch.clamp(vec_len, min=EPS).unsqueeze(-1)

    extension = vec_len - rest_length
    activation = torch.tanh(extension / activation_delta)
    extension_rate = ((payload_attach_vel - drone_attach_vel) * uhat).sum(dim=-1)

    damping_impl = damping + dt * stiffness
    extension_rate_impl = (mass_eff * extension_rate - dt * stiffness * extension) / (mass_eff + dt * damping_impl)
    tension = activation * (stiffness * extension + damping_impl * extension_rate_impl)
    tension = torch.where(vec_len > rest_length, tension, torch.zeros_like(tension))
    tension = torch.clamp(tension, min=0.0)

    return tension.unsqueeze(-1) * uhat

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

        self.target_debug_objs = [] # viewer-only target (sphere + yaw_ref arrow) for env 0, see _draw_target

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
        if self.show_viewer:
            self._attach_viewer_logs()
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

        # Per-sim-step torch code is hundreds of tiny ops, bound by Python dispatch and kernel launches regardless of num_envs.
        # torch.compile fuses them into a few kernels (first steps are slow while it compiles)
        maybe_compile = torch.compile if self.cfg.torch_compile else (lambda fn: fn)
        self._att_ctrl_update = maybe_compile(self.att_ctrl.update)
        self._rate_ctrl_update = maybe_compile(self.rate_ctrl.update)
        self._tether_forces = maybe_compile(tether_forces)

        # Tether links of both entities in one solver call (each Genesis call carries a fixed Python validation cost)
        self.rigid_solver = self.scene.sim.rigid_solver
        self.tether_links_idx = torch.tensor([link.idx for link in self.drone_attach_links + self.payload_attach_links], device=self.device, dtype=gs.tc_int)
        self.num_tethers = len(self.drone_attach_links)
        self.min_cos_tilt = math.cos(self.sys_cfg.tilt_comp_max)

        # Initialized state
        self.mass_eff = 1.0 / (1.0 / self.drone.get_mass() + 1.0 / self.payload.get_mass())
        self.drone_init_pos = torch.tensor(self.sys_cfg.drone_reset_pos, device=self.device, dtype=gs.tc_float)
        self.drone_init_quat = torch.tensor(self.sys_cfg.drone_reset_quat, device=self.device, dtype=gs.tc_float)
        self.payload_init_pos = torch.tensor(self.sys_cfg.payload_reset_pos, device=self.device, dtype=gs.tc_float)
        self.payload_init_quat = torch.tensor(self.sys_cfg.payload_reset_quat, device=self.device, dtype=gs.tc_float)
        self.world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1) # body z-axis for thrust axis
        self.world_down = -self.world_up # gravity direction for projected gravity
        self.g_eff = (1.0 + self.payload.get_mass() / self.drone.get_mass()) * self.gravity # Swing-mode effective gravity |F|/M_drone, hover approximation F = (M + m) g

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
        self.payload_pos_err = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Drone heading frame (RTK)
        self.payload_yaw_err = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) # wrap(yaw_ref - payload yaw)
        self.payload_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Drone heading frame (RTK)
        self.payload_proj_g = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Gravity in payload body axes
        self.payload_body_rates = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Body frame (IMU gyro)
        self.payload_swing_angles = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float) # Drone-Payload relative position vector
        self.payload_swing_angles_rates = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float) # --
        self.drone_proj_g = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Gravity in drone body axes
        self.drone_body_rates = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # Drone body frame (IMU)
        self.drone_payload_rel_yaw = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) # For tether tangling termination (PL RKT+MAG)

        # Non-observations
        self.drone_yaw = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) # abs drone yaw
        self.swing_amp = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float) # equivalent pendulum amplitude [rad]

        self.extras= dict() # logging information

        # Reward terms (each returns its weighted per-env reward) and per-episode sums for logging
        self.reward_fns = {
            "track": self._reward_track,
            "vmax": self._reward_vmax,
            "swing_energy": self._reward_swing_energy,
            "yaw": self._reward_yaw,
            "tilt": self._reward_tilt,
            "smooth_actions": self._reward_smooth_actions,
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

        # Reset drone
        drone_quat0 = self.drone_init_quat.expand(n, -1)
        self.drone.set_pos(self.drone_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(drone_quat0, zero_velocity=True, envs_idx=envs_idx)
        
        # Reset payload
        payload_quat0 = self.payload_init_quat.expand(n, -1)
        self.payload.set_pos(self.payload_init_pos.expand(n, -1), zero_velocity=True, envs_idx=envs_idx)
        self.payload.set_quat(payload_quat0, zero_velocity=True, envs_idx=envs_idx)

        # Commands (yaw_ref sampled relative to the reset payload heading)
        payload_yaw0 = quat_yaw(payload_quat0)
        self._resample_commands(envs_idx, payload_yaw0)

        # Reset internal state (at rest)
        self.drone_yaw[envs_idx] = quat_yaw(drone_quat0)
        self.payload_pos_err[envs_idx] = rotate_to_heading(self.commands[envs_idx, :3] - self.payload.get_pos(envs_idx), self.drone_yaw[envs_idx])
        self.payload_vel[envs_idx] = 0.0
        self.payload_proj_g[envs_idx] = transform_by_quat(self.world_down[envs_idx], inv_quat(payload_quat0))
        self.payload_body_rates[envs_idx] = 0.0
        self.payload_swing_angles[envs_idx] = 0.0
        self.payload_swing_angles_rates[envs_idx] = 0.0
        self.drone_proj_g[envs_idx] = transform_by_quat(self.world_down[envs_idx], inv_quat(drone_quat0))
        self.drone_body_rates[envs_idx] = 0.0
        self.drone_payload_rel_yaw[envs_idx] = wrap_angle(payload_yaw0 - self.drone_yaw[envs_idx])
        self.payload_yaw_err[envs_idx] = wrap_angle(self.commands[envs_idx, 3] - payload_yaw0)

        self.swing_amp[envs_idx] = 0.0

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
        # Viewer refreshed on the last substep only (policy rate), see _attach_viewer_logs for the matching pacing
        for i in range(self.decimation):
            self._sim_step(thrust_sp, roll_sp, pitch_sp, yawrate_sp, update_visualizer=(i == self.decimation - 1))

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
    
    def _sim_step(self, thrust_sp, roll_sp, pitch_sp, yawrate_sp, update_visualizer=True):
        # Read drone body rate state
        drone_quat = self.drone.get_quat()
        drone_body_rates = transform_by_quat(self.drone.get_ang(), inv_quat(drone_quat))

        # Get motor RPM from ratecontroller/mixer
        rate_sp = self._att_ctrl_update(drone_quat, roll_sp, pitch_sp, yawrate_sp)
        rpms = self._rate_ctrl_update(thrust_sp, rate_sp, drone_body_rates)
        self.drone.set_propellers_rpm(rpms)

        # Step tether model
        self._tethers_step()

        # Step sim scene
        self.scene.step(update_visualizer=update_visualizer)

    def _tethers_step(self):
        # Step tether tension-force model. Links ordered [drone attach..., payload attach...], see tether_links_idx
        n = self.num_tethers
        pos = self.rigid_solver.get_links_pos(self.tether_links_idx, relative=True)
        vel = self.rigid_solver.get_links_vel(self.tether_links_idx, relative=True)
        force_on_drone = self._tether_forces(
            pos[:, :n], vel[:, :n], pos[:, n:], vel[:, n:],
            self.sys_cfg.rest_length, self.sys_cfg.activation_delta, self.sys_cfg.stiffness, self.sys_cfg.damping, self.mass_eff, self.sim_dt,
        )
        self.rigid_solver.apply_links_external_wrench(force=torch.cat([force_on_drone, -force_on_drone], dim=1), links_idx=self.tether_links_idx)

    def _update_buffers(self):
        self.episode_length_buf += 1

        # Drone attitude (heading frame defined by drone yaw)
        drone_quat = self.drone.get_quat()
        drone_inv_quat = inv_quat(drone_quat)
        self.drone_yaw[:] = quat_yaw(drone_quat)
        yaw = self.drone_yaw
        self.drone_proj_g[:] = transform_by_quat(self.world_down, drone_inv_quat)
        self.drone_body_rates[:] = transform_by_quat(self.drone.get_ang(), drone_inv_quat)

        # Payload tracking and attitude
        payload_pos = self.payload.get_pos()
        payload_vel = self.payload.get_vel()
        payload_quat = self.payload.get_quat()
        payload_inv_quat = inv_quat(payload_quat)
        payload_yaw = quat_yaw(payload_quat)
        self.payload_pos_err[:] = rotate_to_heading(self.commands[:, :3] - payload_pos, yaw)
        self.payload_yaw_err[:] = wrap_angle(self.commands[:, 3] - payload_yaw) # can wrap: observed as sin/cos
        self.payload_vel[:] = rotate_to_heading(payload_vel, yaw)
        self.payload_proj_g[:] = transform_by_quat(self.world_down, payload_inv_quat)
        self.payload_body_rates[:] = transform_by_quat(self.payload.get_ang(), payload_inv_quat)
        self.drone_payload_rel_yaw[:] = wrap_angle(payload_yaw - yaw) # will terminate before wrap-issue

        r_w = payload_pos - self.drone.get_pos() # drone->payload vector
        r_dot_w = payload_vel - self.drone.get_vel() # drone->payload vector rate
        r = rotate_to_heading(r_w, yaw) # drone->payload vector in yaw-frame (abs yaw from drone FC)
        r_dot = rotate_to_heading(r_dot_w, yaw) # -- rates

        # Swing angles
        self.payload_swing_angles[:, 0] = torch.atan2(r[:, 0], -r[:, 2])
        self.payload_swing_angles[:, 1] = torch.atan2(r[:, 1], -r[:, 2])
        # analytical derivative of atan2(a, b): (b*a_dot - a*b_dot) / (a² + b²), with b=-r_z
        den_x = torch.clamp(r[:, 0]**2 + r[:, 2]**2, min=EPS)
        den_y = torch.clamp(r[:, 1]**2 + r[:, 2]**2, min=EPS)
        self.payload_swing_angles_rates[:, 0] = (r[:, 0] * r_dot[:, 2] - r[:, 2] * r_dot[:, 0]) / den_x
        self.payload_swing_angles_rates[:, 1] = (r[:, 1] * r_dot[:, 2] - r[:, 2] * r_dot[:, 1]) / den_y

        # Swing-mode energy. Relative motion r_ddot = T*q/mu - F/M is a pendulum in "gravity" F/M,
        # so its equilibrium is the cable along the thrust axis (body z). Normalised by L*g_eff:
        # E = 1/2 |r_dot_perp|² / (L g_eff) + (1 - q·z_body) = 1 - cos(swing_amp)
        L = torch.clamp(torch.norm(r_w, dim=1, keepdim=True), min=EPS)
        q = -r_w / L # payload -> drone
        r_dot_perp = r_dot_w - torch.sum(r_dot_w * q, dim=1, keepdim=True) * q # drop cable stretch
        thrust_axis = transform_by_quat(self.world_up, drone_quat)
        e_kin = 0.5 * torch.sum(torch.square(r_dot_perp), dim=1) / (L.squeeze(1) * self.g_eff)
        e_pot = 1.0 - torch.sum(q * thrust_axis, dim=1)
        self.swing_amp[:] = torch.acos(torch.clamp(1.0 - (e_kin + e_pot), min=-1.0, max=1.0))

    def _update_crash_conditions(self):
        payload_up_z = -self.payload_proj_g[:, 2] # cos(payload tilt)
        drone_up_z = -self.drone_proj_g[:, 2] # cos(drone tilt)

        self.crash_condition = (
              (torch.abs(self.payload_swing_angles[:, 0]) > self.cfg.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_swing_angles[:, 1]) > self.cfg.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_pos_err[:, 0]) > self.cfg.terminate_if_x_greater_than)
            | (torch.abs(self.payload_pos_err[:, 1]) > self.cfg.terminate_if_y_greater_than)
            | (torch.abs(self.payload_pos_err[:, 2]) > self.cfg.terminate_if_z_greater_than)
            | (torch.abs(self.drone_payload_rel_yaw[:]) > self.cfg.terminate_if_relyaw_greater_than)
            | (payload_up_z < math.cos(self.cfg.terminate_if_payload_tilt_greater_than))
            | (drone_up_z < math.cos(self.cfg.terminate_if_drone_tilt_greater_than))
            | (self.payload.get_pos()[:, 2] <= self.cfg.terminate_if_payload_below_z)
        )
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout | self.crash_condition
        self.extras["time_outs"] = timeout

    def _update_observations(self):
        # Yaw-agnostic: heading-frame, body-frame or relative quantities only
        s = self.cfg.obs_scales
        self.obs_buf = torch.cat(
            [
                # Task (heading frame)
                torch.clamp(self.payload_pos_err * self.pos_obs_scale, -1.0, 1.0), # payload pos err, dim=3
                torch.clamp(self.payload_vel * self.vel_obs_scale, -1.0, 1.0), # payload vel, dim=3
                # Payload
                self.payload_proj_g, # gravity in payload body axes, dim=3
                torch.clamp(self.payload_body_rates * s.pl_body_rates, -1.0, 1.0), # payload body rates, dim=3
                torch.clamp(self.payload_swing_angles * s.pl_swing_angles, -1.0, 1.0), # swing angles (heading frame), dim=2
                torch.clamp(self.payload_swing_angles_rates * s.pl_swing_angles_rates, -1.0, 1.0), # swing angles rates (heading frame), dim=2
                torch.clamp(self.drone_payload_rel_yaw.unsqueeze(-1) * s.rel_yaw, -1.0, 1.0), # drone-payload relative yaw, dim=1
                torch.sin(self.payload_yaw_err).unsqueeze(-1), # payload yaw err, dim=1
                torch.cos(self.payload_yaw_err).unsqueeze(-1), # -- dim=1
                # Drone
                self.drone_proj_g, # gravity in drone body axes, dim=3
                torch.clamp(self.drone_body_rates * s.drone_body_rates, -1.0, 1.0), # drone body rates, dim=3
                self.actions, # last setpoint [thrust, roll_sp, pitch_sp, yawrate_sp], dim=4
            ],
            axis=1
        )
    
    def _resample_commands(self, envs_idx, payload_yaw0, n=1):
        if len(envs_idx) == 0:
            return
        # TODO: Make n the number of close waypoint in n-horizon trajectory
        pl_init = self.payload_init_pos
        self.commands[envs_idx, 0] = gs_rand_float(pl_init[0] - self.cmd_cfg.pos_x_range, pl_init[0] + self.cmd_cfg.pos_x_range, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 1] = gs_rand_float(pl_init[1] - self.cmd_cfg.pos_y_range, pl_init[1] + self.cmd_cfg.pos_y_range, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 2] = gs_rand_float(pl_init[2] - self.cmd_cfg.pos_z_range, pl_init[2] + self.cmd_cfg.pos_z_range, shape=(len(envs_idx), ), device=self.device)
        self.commands[envs_idx, 3] = wrap_angle(payload_yaw0 + gs_rand_float(-self.cmd_cfg.yaw_range, self.cmd_cfg.yaw_range, shape=(len(envs_idx), ), device=self.device))

        self._draw_target(envs_idx)

    def _draw_target(self, envs_idx):
        # Viewer-only target from env 0's command: sphere at [x, y, z], arrow along yaw_ref pointing outward from the sphere
        # (debug objects only exist in the viewer, which renders env 0)
        if not self.show_viewer or not (envs_idx == 0).any():
            return
        for obj in self.target_debug_objs:
            self.scene.clear_debug_object(obj)

        radius = 0.05
        pos = self.commands[0, :3]
        yaw_ref = self.commands[0, 3].item()
        heading = torch.tensor([math.cos(yaw_ref), math.sin(yaw_ref), 0.0], device=self.device)
        color_s = (1.0, 0.0, 0.0, 1.0)
        color_a = (0.0, 0.0, 0.0, 1.0)
        self.target_debug_objs = [
            self.scene.draw_debug_sphere(pos=pos.tolist(), radius=radius, color=color_s),
            self.scene.draw_debug_arrow(pos=(pos + radius * heading).tolist(), vec=(0.3 * heading).tolist(), radius=0.008, color=color_a),
        ]

    def _attach_viewer_logs(self):
        # The viewer paces one sim_dt per update, but step() updates it once per decimation substeps: scale to stay real time
        self.scene.viewer.realtime_factor = self.decimation
        # Viewer-only text overlay of env 0 state (see utils.attach_log). Buffers are heading frame
        attach_sim_rate(self.scene)
        attach_log(self.scene, "|pos err| [m]", lambda: torch.norm(self.payload_pos_err, dim=1))
        attach_log(self.scene, "payload |v| [m/s]", lambda: torch.norm(self.payload_vel, dim=1))
        attach_log(self.scene, "reward", lambda: self.rew_buf, fmt="{:+.3f}")

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

    def _reward_swing_energy(self):
        # Linear in amplitude so small residual swing near the target is still penalised
        swing_energy_rew = self.swing_amp * self.rew_cfg.w_swing_energy
        return swing_energy_rew

    def _reward_yaw(self):
        # Payload heading tracking; smooth and periodic so the wrap at +-pi has no jump
        yaw_rew = (1.0 - torch.cos(self.payload_yaw_err)) * self.rew_cfg.w_yaw
        return yaw_rew

    def _reward_tilt(self):
        # Payload attitude away from level: 1 - cos(tilt) ~ tilt²/2, so small tilts during manoeuvres are nearly free
        # (unlike the linear swing term) while large tilts are increasingly penalised
        tilt_rew = (1.0 + self.payload_proj_g[:, 2]) * self.rew_cfg.w_tilt
        return tilt_rew

    def _reward_smooth_actions(self):
        action_rew = torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * self.rew_cfg.w_smooth_actions
        return action_rew

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