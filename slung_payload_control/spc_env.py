import genesis as gs
import os
import torch
import math
from tensordict import TensorDict

from config.config_env import EnvConfig
from config.paths import DRONE_PATH, PAYLOAD_PATH, TRAJ_BANK_DIR

from utils.genesis_viewer import GenesisFollower, GenesisRecorder, GenesisViewer
from utils.genesis_helpers import silence_asset_warnings, airframe_from_genesis
from utils.torch_math_helpers import *

from components.flight_control import FlightController
from components.tether_model import TetherModel
from components.domain_randomization import DomainRandomizer, SensorModel
from components.trajectory import TrajectoryBank

class SlungPayloadControlEnv:
    def __init__(self, num_envs: int, env_cfg: EnvConfig, device: str="cuda", show_viever: bool=False):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.cfg = env_cfg

        # Timing
        self.dt = self.cfg.policy_dt # policy step
        self.sim_dt = self.cfg.sim_dt # physics step (and low-level control)
        self.decimation = self.cfg.decimation

        # System
        self.num_actions = self.cfg.num_actions
        self.max_episode_length = math.ceil(self.cfg.episode_termination.episode_length_s / self.dt)

        # Visualization / Camera / Recording
        self.show_viewer = show_viever

        # Initialize buffers 
        self._init_buffers()
        # Initialize components
        self._init_components()
        # Build scene
        self._build_scene()
        # Post init/build binds
        self._post_init()

        self.reset()

    def _build_scene(self):
        silence_asset_warnings()
        cfg = self.cfg
        # Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, gravity=(0.0, 0.0, self.gravity)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(1)),
                background_color=(0.3, 0.3, 0.3),
                shadow=False,
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,
                batch_links_info=True, # to allow domain randomization in drone/payload mass
            ),
            show_viewer=self.show_viewer,
        )

        if self.show_viewer:
            self.gs_viewer = GenesisViewer(self.scene, self.device, view_idx=0)
            self.gs_recorder = GenesisRecorder(self.scene, round(1/self.dt))
            self.gs_follower = GenesisFollower(self.scene)

        # Ground Plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        # Drone
        drone_attach_links = [f"attach{x}_link" for x in range(cfg.tether_cfg.num_tethers)]
        prop_links = [f"prop{x}_link" for x in range(4)]
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=DRONE_PATH,
                pos=(0.0, 0.0, cfg.tether_cfg.rest_length + cfg.payload_init_altitude),
                quat=(1.0, 0.0, 0.0, 0.0),
                propellers_link_name=prop_links,
                propellers_spin=(-1, 1, -1, 1),
                prioritize_urdf_material=True,
                links_to_keep=drone_attach_links,
                # default_base_ang_damping_scale=0.0,
            )
        )

        # Payload
        payload_attach_links = [f"attach{x}_link" for x in range(cfg.tether_cfg.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=(0.0, 0.0, cfg.payload_init_altitude),
                quat=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                links_to_keep=payload_attach_links
            )
        )

        # Genesis Rigid solver
        self.rigid_solver = self.scene.sim.rigid_solver
        # Link idx for tether attachments [drone..., payload...]
        self.tether_links_idx = torch.tensor([link.idx for link in [self.drone.get_link(dl) for dl in drone_attach_links] + [self.payload.get_link(pl) for pl in payload_attach_links]], device=self.device, dtype=gs.tc_int)
        # Link idx for bodies base_links
        self.base_links_idx = torch.tensor([self.drone.base_link_idx, self.payload.base_link_idx], device=self.device, dtype=gs.tc_int)
        # Mass links (for DR)
        mass_links = [[link.idx for link in e.links if not link.is_fixed] for e in (self.drone, self.payload)]
        self.mass_links_idx = sum(mass_links, [])
        self.mass_links_entity = torch.tensor([i for i, links in enumerate(mass_links) for _ in links], device=self.device)
        # Free joint indices
        self.base_qs_idx = list(range(self.drone.q_start, self.drone.q_end)) + list(range(self.payload.q_start, self.payload.q_end))
        self.base_dofs_idx = list(range(self.drone.dof_start, self.drone.dof_end)) + list(range(self.payload.dof_start, self.payload.dof_end))

        # Build scene
        self.scene.build(n_envs = self.num_envs)

    def _init_components(self):
        self.controller = FlightController(
            self.cfg.fc_cfg, 
            self.num_envs, 
            self.sim_dt, 
            self.decimation, 
            self.device, 
            gs.tc_float
        )
        self.tether_model = TetherModel(
            self.cfg.tether_cfg,
            self.num_envs,
            self.sim_dt,
            self.device,
            gs.tc_float
        )
        self.domain_rand = DomainRandomizer(
            self.cfg.dr_cfg,
            self.num_envs,
            self.device,
            gs.tc_float
        )
        self.sensors = SensorModel(
            self.cfg.sensor_cfg,
            self.num_envs,
            self.dt,
            self.device,
            gs.tc_float
        )
        self.traj_bank = TrajectoryBank.load_or_build(
            self.cfg.traj_cfg,
            os.path.join(TRAJ_BANK_DIR, "bank.npz"),
            self.device
        )

        # Compile per sim-step functions to improve runtime performance
        if self.cfg.torch_compile:
            print("[FlightController] Compiling torch kernels...")
            self.controller.compile()
            print("[TetherModel] Compiling torch kernels...")
            self.tether_model.compile()
            print("[SensorModel] Compiling torch kernels...")
            self.sensors.compile()
            print("[SlungPayloadControlEnv] Compiling torch kernels...")
            self._compute_buffers = torch.compile(self._compute_buffers, dynamic=False)
            self._update_observations = torch.compile(self._update_observations, dynamic=False)
            self._post_step = torch.compile(self._post_step, dynamic=False)

            # Step through compiled env functions to trigger compile pre-build
            dummy_3vec = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=gs.tc_float)
            dummy_4vec = torch.zeros((self.num_envs, 2, 4), device=self.device, dtype=gs.tc_float)
            dummy_4vec[..., 0] = 1.0 # identity
            self._compute_buffers(dummy_3vec, dummy_4vec, dummy_3vec.clone(), dummy_3vec.clone()) # (pos, quat, vel, ang)
            self.raw_states = (dummy_3vec, dummy_4vec, dummy_3vec.clone(), dummy_3vec.clone()) # (pos, quat, vel, ang)
            self._update_observations()
            self.raw_states = None
            self._post_step()

            self.success_count.zero_()
            self.episode_count.zero_()

    def _post_init(self):
        self.mass_links0 = self.rigid_solver.get_links_mass(self.mass_links_idx, [0])[0] # nominal mass values
        self.mass0 = torch.zeros(2, device=self.device, dtype=gs.tc_float).index_add_(0, self.mass_links_entity, self.mass_links0)

        # Bind DroneEntity to controller
        self.controller.bind(airframe_from_genesis(self.drone, self.cfg.fc_cfg.twr_max, payload_mass=self.mass0[1].item()))
        # Bind domain randomization
        self.domain_rand.bind(self._apply_domain_rand)

        n = self.cfg.tether_cfg.num_tethers
        p = self.rigid_solver.get_links_pos(self.tether_links_idx, [0], relative=True)[0]
        b = self.rigid_solver.get_links_pos(self.base_links_idx, [0], relative=True)[0]
        self.attach_off = p[:n] - b[0], p[n:] - b[1] # drone, payload (n, 3)

        # Viewer logging on-screen
        if self.show_viewer:
            self.gs_viewer.attach_sim_rate()
            self.gs_viewer.attach_log("|PL pos err| [m]", lambda: torch.norm(self.payload_pos_err[:,0], dim=1))
            self.gs_viewer.attach_log("reward", lambda: self.reward_buf)


    def _init_buffers(self):
        zeros = lambda *shape, dtype=gs.tc_float: torch.zeros((self.num_envs, *shape), device=self.device, dtype=dtype)

        # World (gravity defined here and passed to SimOptions, so compiled kernels can warm up before scene build)
        self.world_down = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=gs.tc_float)
        self.gravity = -9.81 # -z
        self.gravity_vec = torch.tensor([0.0, 0.0, self.gravity], device=self.device, dtype=gs.tc_float)

        # Environment state buffers
        self.obs_buf = zeros(self.cfg.num_obs)
        self.critic_obs_buf = zeros(self.cfg.num_obs + self.cfg.num_critic_obs)
        self.reward_buf = zeros()
        self.reset_buf = zeros(dtype=gs.tc_bool)
        self.episode_length_buf = zeros(dtype=gs.tc_int)
        self.crash_condition_buf = zeros(dtype=gs.tc_bool)

        # Trajectory
        self.traj_idx = zeros(dtype=gs.tc_int) # (E, ) which bank entry each env follows
        self.traj_time = zeros() # (E, ) seconds since traj start
        self.preview_t = torch.tensor(self.cfg.preview_points_s, device=self.device, dtype=gs.tc_float) # (K, ), the preview times

        # Observations
        self.payload_pos_err = zeros(len(self.preview_t), 3) # position errors to trajectory positions (N, K, 3)
        self.payload_vel_err = zeros(len(self.preview_t), 3) # velocity errors to trajectory velocities (N, K, 3)
        self.payload_proj_g = zeros(3) # tilt: gravity vector in payload body frame
        self.payload_body_rates = zeros(3) # angular velocity
        self.payload_swing_angles = zeros(2) # 2D projected angles alpha/beta based on relative position
        self.payload_swing_rates = zeros(2) # --- rates
        self.drone_proj_g = zeros(3) # tilt: gravity vector in drone body frame
        self.drone_body_rates = zeros(3) # angular velocity: in drone body frame
        self.actions = zeros(self.num_actions)

        # Non-observation buffers
        self.prev_actions = zeros(self.num_actions)
        self.actions_raw = zeros(self.num_actions)
        self.payload_world_pos = zeros(3) # crash-conditions
        self.swing_energy = zeros()
        self.raw_states = None # (pos, quat, vel, ang) of [drone, payload] from last _update_buffers
    
        # Rewards (OBS: on naming convention)
        self.reward_fns = {name: getattr(self, f"_reward_{name}") for name in self.cfg.rewards.reward_terms}
        self.ep_rew_sum = zeros(len(self.cfg.rewards.reward_terms))

        # Per-episode tracking accumulators
        self.ep_err_sum = zeros()
        self.ep_err_sq_sum = zeros()
        self.ep_err_max = zeros()
        self.ep_swing_sum = zeros()
        self.ep_steps = zeros()
        self.success_count = torch.zeros((), device=self.device)
        self.episode_count = torch.zeros((), device=self.device)
        self.success_steps = 0 # policy steps into the current success window

        # Visualization
        self.traj_preview_pos = zeros(len(self.preview_t), 3) # for visualizing the preview trajectory

        # Logging info (shown by genesis)
        self.extras = dict()

    def get_observations(self):
        return TensorDict({
            "actor": self.obs_buf,
            "critic": self.critic_obs_buf,
        }, batch_size=[self.num_envs])

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_buffers()
        self._update_observations()

        if self.show_viewer:
            self._update_viewer()
            self._update_camera()

        return self.get_observations()

    def reset_idx(self, envs_idx):
        self.extras["episode"] = {} # clear everytime so rsl_rl never relogs stale vals

        if len(envs_idx) == 0:
            return
        
        self._log_reset(envs_idx)

        # New parameters
        self.domain_rand.randomize(envs_idx)
        self.sensors.reset(envs_idx)

        # New trajectories
        self.traj_idx[envs_idx] = self.traj_bank.sample_ids(len(envs_idx))
        self.traj_time[envs_idx] = 0.0
        if self.show_viewer:
            self.gs_viewer.draw_route(self.traj_bank, self.traj_idx, envs_idx)

        t0 = torch.zeros(len(envs_idx), 1, device=self.device)
        p0, v0 = (x[:, 0] for x in self.traj_bank.eval(self.traj_idx[envs_idx], t0, orders=(0,1))) 

        # Init state (rought est.)
        quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float).expand(len(envs_idx), 4)
        payload_pos = p0
        # drone_pos = p0 + self.tether_model.rest_length[envs_idx, None] * -self.world_down
        
        d_off = self.attach_off[0] - self.attach_off[1]
        h = self.tether_model.rest_length[envs_idx, None] - d_off[:, 2]
        drone_pos = p0 + h.mean(dim=1, keepdim=True) * -self.world_down

        zeros = torch.zeros_like(v0)
        self.rigid_solver.set_qpos(torch.cat([drone_pos, quat0, payload_pos, quat0], dim=1), self.base_qs_idx, envs_idx)

        # Shift drone up by measured tether slack (attach-to-attach lenght vs rest_length)
        # This avoids initial drop of payload causing tension peaks of several N
        n = self.cfg.tether_cfg.num_tethers
        p = self.rigid_solver.get_links_pos(self.tether_links_idx, envs_idx, relative=True)
        slack = self.tether_model.rest_length[envs_idx] - (p[:, :n] - p[:, n:]).norm(dim=-1).mean(dim=1)
        # TODO: With randomized initial swing: self.world_down -> swing_direction
        drone_pos = drone_pos + slack[:, None] * (-self.world_down)
        self.rigid_solver.set_qpos(torch.cat([drone_pos, quat0, payload_pos, quat0], dim=1), self.base_qs_idx, envs_idx)
        self.rigid_solver.set_dofs_velocity(torch.cat([v0, zeros, v0, zeros], dim=1), self.base_dofs_idx, envs_idx) # [lin, ang] per entity

        # Reset controller with initial drone quaternion
        self.controller.reset(envs_idx, quat0)

        # Zero buffers
        self.actions[envs_idx] = 0
        self.prev_actions[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0
        self.crash_condition_buf[envs_idx] = 0
        self.ep_rew_sum[envs_idx] = 0.0

        self.ep_err_sum[envs_idx] = 0.0
        self.ep_err_sq_sum[envs_idx] = 0.0
        self.ep_err_max[envs_idx] = 0.0
        self.ep_swing_sum[envs_idx] = 0.0
        self.ep_steps[envs_idx] = 0.0

    def step(self, actions):
        self.actions_raw = actions
        self.actions, substep_setpoints = self.controller.policy_step(actions)

        # Dual-rate: decimation number of physics/control steps between policy steps
        for i in range(self.decimation):            
            self._sim_step(substep_setpoints[i])

        self.episode_length_buf += 1
        self.traj_time[:] += self.dt # update trajectory times by dt
        
        self._update_buffers()
        self.extras["time_outs"] = self._post_step()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape(-1, ))
        self._update_buffers(compute=False) # does not compute all buffers - only necessary for observations
        self._update_observations()
        self.prev_actions[:] = self.actions

        # CAMERA and VISUALIZATION
        if self.show_viewer:
            self._update_viewer()
            self._update_camera()

        # LOGGING
        self._log_step()

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras

    def _sim_step(self, ctrl_sp, update_viz=True):
        # Motor RPM from controller component
        drone_quat = self.drone.get_quat()
        drone_angv = self.drone.get_ang()
        rpm = self.controller.sim_step(drone_quat, drone_angv, ctrl_sp)
        self.drone.set_propellers_rpm(rpm)

        # TetherModel update
        t_pos = self.rigid_solver.get_links_pos(self.tether_links_idx, relative=True)
        t_vel = self.rigid_solver.get_links_vel(self.tether_links_idx, relative=True)
        force, torque = self.tether_model.step(t_pos, t_vel)
        self.rigid_solver.apply_links_external_wrench(force=force, torque=torque, links_idx=self.tether_links_idx)

        # Step the simulator
        self.scene.step()

    def _post_step(self):
        timeout = self._update_crash_conditions()
        # self.extras["time_outs"] = timeout

        self._compute_rewards()

        # Episode accumulators
        err = torch.norm(self.payload_pos_err[:,0], dim=1)
        self.ep_err_sum += err
        self.ep_err_sq_sum += torch.square(err)
        self.ep_err_max[:] = torch.maximum(self.ep_err_max, err)
        self.ep_swing_sum += self.swing_energy
        self.ep_steps += 1.0
        
        mean_err = self.ep_err_sum / torch.clamp(self.ep_steps, min=1.0)
        succes = timeout & (mean_err < self.cfg.succes_mean_track_err)
        self.success_count += succes.sum()
        self.episode_count += self.reset_buf.sum()

        return timeout

    def _update_buffers(self, compute=True):
        # Read genesis states from solver
        # [drone,..., payload, ...]
        pos = self.rigid_solver.get_links_pos(self.base_links_idx, relative=True)
        quat = self.rigid_solver.get_links_quat(self.base_links_idx, relative=True)
        vel = self.rigid_solver.get_links_vel(self.base_links_idx, relative=True)
        ang = self.rigid_solver.get_links_ang(self.base_links_idx)
        self.raw_states = (pos, quat, vel, ang)
        if compute:
            self._compute_buffers(pos, quat, vel, ang)

    def _compute_buffers(self, pos, quat, vel, ang):
        drone_p, payload_p = pos.unbind(1)
        drone_q, payload_q = quat.unbind(1)
        drone_vel, payload_vel = vel.unbind(1)
        drone_ang, payload_ang = ang.unbind(1)

        # Query TrajectoryBanks
        t_query = self.traj_time[:, None] + self.preview_t[None, :]
        traj_pos, traj_vel, traj_acc = self.traj_bank.eval(self.traj_idx, t_query, orders=(0,1,2))        
        self.traj_preview_pos[:] = traj_pos # stored for visualization
        
        # Function to convert to drone heading frame (Generally yaw invariant observations)
        yaw = quat_yaw(drone_q)
        zero = torch.zeros_like(yaw)
        q_yaw = torch.stack([torch.cos(yaw / 2.0), zero, zero, torch.sin(yaw / 2.0)], dim=-1)
        def to_heading(v):
            # broadcast q_yaw (N, 4) over any middle dims of v: (N, 3), (N, K, 3), ...
            q = q_yaw.view(q_yaw.shape[0], *([1] * (v.dim() - 2)), 4)
            return quat_rotate_inverse(q, v)

        # Tracking errors
        self.payload_pos_err[:] = to_heading(traj_pos - payload_p[:, None, :]) # (N, K, 3)
        self.payload_vel_err[:] = to_heading(traj_vel - payload_vel[:, None, :]) # (N, K, 3)

        # Orientations and rates
        self.payload_proj_g[:] = quat_rotate_inverse(payload_q, self.world_down) # down_body = drone_q^-1 * down_world
        self.drone_proj_g[:] = quat_rotate_inverse(drone_q, self.world_down) 
        self.drone_body_rates[:] = quat_rotate_inverse(drone_q, drone_ang)
        self.payload_body_rates[:] = quat_rotate_inverse(payload_q, payload_ang)

        vdp = to_heading(payload_p - drone_p) # vector from drone->payload
        vdp_dot = to_heading(payload_vel - drone_vel) # how the drone->payload vector changes
        self.payload_swing_angles[:, 0] = torch.atan2(vdp[:, 0], -vdp[:, 2]) # negative-z since vdp points from drone->payload
        self.payload_swing_angles[:, 1] = torch.atan2(vdp[:, 1], -vdp[:, 2])
        # (d/da * d/db) atan2(a,b) = (a*b_dot - b*a_dot) / (a^2 + b^2)
        den0 = torch.clamp(vdp[:,0]*vdp[:,0] + vdp[:,2]*vdp[:,2], min=1e-9)
        den1 = torch.clamp(vdp[:,1]*vdp[:,1] + vdp[:,2]*vdp[:,2], min=1e-9)
        self.payload_swing_rates[:, 0] = (vdp[:, 0] * vdp_dot[:, 2] - vdp[:, 2] * vdp_dot[:, 0]) / den0
        self.payload_swing_rates[:, 1] = (vdp[:, 1] * vdp_dot[:, 2] - vdp[:, 2] * vdp_dot[: ,1]) / den1

        # swing mode energy
        vdp_norm = torch.clamp(torch.norm(vdp, dim=1), min=1e-9) # (N, )
        vdp_hat = vdp / vdp_norm[:, None] # (N, 3)
        v_radial = (vdp_dot * vdp_hat).sum(-1)[:, None] * vdp_hat # axial: stretch/bounce mode 0 if vdp_dot is orthogonal to vpd_hat
        v_tangent = vdp_dot - v_radial # tangential: swing mode

        traj_acc = to_heading(traj_acc) # (N, K, 3)
        g_eff = self.gravity_vec - traj_acc[:, 0, :] # (N, 3) only for first reference acc
        g_mag = torch.clamp(torch.norm(g_eff, dim=1), min=1e-9) # (N, )
        g_eff_hat = g_eff / g_mag[:, None] # (N, 3)

        ke = 0.5 * (v_tangent * v_tangent).sum(-1) # (N, )
        pe = g_mag * vdp_norm * (1.0 - (vdp_hat * g_eff_hat).sum(-1))
        self.swing_energy[:] = (ke + pe) / (abs(self.gravity) * vdp_norm)

        # Misc
        self.payload_world_pos[:] = payload_p

    def _update_observations(self):
        pos, quat, vel, ang = self.raw_states
        p, q, v, rates = self.sensors.measure(pos, quat, vel, quat_rotate_inverse(quat, ang))
        self._compute_buffers(p, q, v, quat_rotate(q, rates))
        self.obs_buf = self._observations()

        # clean
        self._compute_buffers(pos, quat, vel, ang)
        self.critic_obs_buf = torch.cat([self._observations(), self.domain_rand.priviledged_obs()], dim=-1)

    def _observations(self):
        s = self.cfg.obs

        # tau-horizon scaling to not saturate the clamp
        tau = self.preview_t[:, None]
        vmax, amax = self.cfg.traj_cfg.route_vel_des_max, self.cfg.traj_cfg.a_max
        pos_scale = 1.0 / (1.0 / s.scale_pl_p + vmax * tau + 0.5 * amax * tau * tau)
        vel_scale = 1.0 / (1.0 / s.scale_pl_v + amax * tau)

        obs_buf = torch.cat(
            [
                # Task (heading frame): dim=30 for K=5
                torch.clamp(self.payload_pos_err * pos_scale, -1.0, 1.0).flatten(1), # (N, 3K), [p0x, p0y, p0z, ..., ]
                torch.clamp(self.payload_vel_err * vel_scale, -1.0, 1.0).flatten(1), # (N, 3K), [v0x, v0y, v0z, ..., ]

                # Payload: dim=10
                self.payload_proj_g,
                torch.clamp(self.payload_body_rates * s.scale_pl_br, -1.0, 1.0),
                torch.clamp(self.payload_swing_angles * s.scale_pl_sa, -1.0, 1.0),
                torch.clamp(self.payload_swing_rates * s.scale_pl_sr, -1.0, 1.0),

                # Drone: dim=6
                self.drone_proj_g,
                torch.clamp(self.drone_body_rates * s.scale_d_br, -1.0, 1.0),

                # Actions: dim=4
                self.actions,
            ],
            axis=1
        )

        return obs_buf

    def _update_crash_conditions(self):
        payload_up_z = -self.payload_proj_g[:, 2]
        term = self.cfg.episode_termination

        self.crash_condition_buf = (
              (torch.abs(self.payload_swing_angles[:, 0]) > term.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_swing_angles[:, 1]) > term.terminate_if_swingangle_greater_than)
            | (torch.abs(self.payload_pos_err[:, 0, 0]) > term.terminate_if_x_greater_than)
            | (torch.abs(self.payload_pos_err[:, 0, 1]) > term.terminate_if_y_greater_than)
            | (torch.abs(self.payload_pos_err[:, 0, 2]) > term.terminate_if_z_greater_than)
            | (payload_up_z < math.cos(term.terminate_if_payload_tilt_greater_than))
            | (self.payload_world_pos[:, 2] <= term.terminate_if_payload_below_z)
        )

        timeout = (self.episode_length_buf >= self.max_episode_length) & ~self.crash_condition_buf
        self.reset_buf = timeout | self.crash_condition_buf
        return timeout

    def _apply_domain_rand(self, envs_idx, p):
        # Actuation
        self.controller.set_actuation(
            envs_idx, 
            p["thrust_scale"], 
            (p["latency_s"] / self.sim_dt).round().long()
        )
        
        # Body masses
        mass_scales = torch.stack([p["drone_mass_scale"], p["payload_mass_scale"]], dim=-1)
        links_mass = self.mass_links0 * mass_scales[:, self.mass_links_entity]
        self.rigid_solver.set_links_mass(links_mass, self.mass_links_idx, envs_idx, scale_inertia=True)
        dm, pm = (self.mass0 * mass_scales).unbind(-1)

        # Tether
        t = self.cfg.tether_cfg
        self.tether_model.set_parameters(
            envs_idx,
            rest_length=t.rest_length * p["rest_length_scale"],
            stiffness=t.stiffness * p["stiffness_scale"],
            damping=t.damping * p["damping_scale"],
            torsional_damping=t.torsional_damping * p["torsional_damping_scale"],
            mass_eff=dm * pm / (dm + pm), # reduced mass: uses the already randomized mass
        )

    def _log_reset(self, envs_idx):
        steps = torch.clamp(self.ep_steps[envs_idx], min=1.0)
        duration = self.traj_bank.t_end[self.traj_idx[envs_idx]]
        completion = torch.clamp(self.traj_time[envs_idx] / duration, max=1.0)

        ep = self.extras.setdefault("episode", {})
        ep["Task/mean_pos_err"] = torch.mean(self.ep_err_sum[envs_idx] / steps)
        ep["Task/rms_pos_err"] = torch.mean(torch.sqrt(self.ep_err_sq_sum[envs_idx] / steps))
        ep["Task/max_pos_err"] = torch.mean(self.ep_err_max[envs_idx])
        ep["Task/mean_swing_energy"] = torch.mean(self.ep_swing_sum[envs_idx] / steps)
        ep["Task/crash_rate"] = torch.mean(self.crash_condition_buf[envs_idx].float())
        ep["Task/route_completion"] = torch.mean(completion)

        # Log rewards
        for i, name in enumerate(self.reward_fns):
            ep[f"Reward/{name}"] = torch.mean(self.ep_rew_sum[envs_idx, i] / steps)

    def _log_step(self):
        self.success_steps += 1
        if self.success_steps < self.cfg.succes_interval or self.episode_count.item() == 0:
            return
        self.extras["episode"]["Task/success_rate"] = (self.success_count / self.episode_count).item()
        self.success_count.zero_()
        self.episode_count.zero_()
        self.success_steps = 0

    def _update_viewer(self):
        self.gs_viewer.draw_preview(self.traj_preview_pos[self.gs_viewer.view_idx])
        tether_pos = self.rigid_solver.get_links_pos(self.tether_links_idx, envs_idx=[0])[0]
        self.gs_viewer.draw_tether_lines(
            tether_pos[:self.cfg.tether_cfg.num_tethers], 
            tether_pos[self.cfg.tether_cfg.num_tethers:]
        )

    def _update_camera(self):
        if not self.show_viewer:
            return 
        cam_pose = self.gs_follower.update(
            self.drone.get_pos()[0],
            self.payload.get_pos()[0],
            quat_yaw(self.drone.get_quat()[0]),
            self.dt
        )
        self.gs_recorder.capture(cam_pose) # no-op unsless recording

    # Rewards
    def _compute_rewards(self):
        rews = torch.stack([fn() for fn in self.reward_fns.values()], dim=-1) # (N, num_reward_terms)
        self.reward_buf[:] = rews.sum(dim=-1)
        self.ep_rew_sum += rews

    def _reward_pos_track(self):
        err = torch.norm(self.payload_pos_err[:, 0, :], dim=1)
        rew = torch.exp(-err / self.cfg.rewards.sigma_pos_track_far)
        rew += torch.exp(-err / self.cfg.rewards.sigma_pos_track_close)
        rew *= 0.5 # avg of the two contri.
        rew *= self.cfg.rewards.w_pos_track
        return rew

    def _reward_vel_track(self):
        err = torch.norm(self.payload_vel_err[:, 0, :], dim=1)
        rew = torch.exp(-err / self.cfg.rewards.sigma_vel_track)
        rew *= self.cfg.rewards.w_vel_track
        return rew
    
    def _reward_swing_energy(self):
        rew = self.swing_energy * self.cfg.rewards.w_swing_energy
        return rew
    
    def _reward_actions(self):
        # Smooth and pre-clip beyond clip bound
        rew = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        excess = torch.clamp(torch.abs(self.actions_raw) - self.cfg.clip_actions, min=0.0)
        rew += torch.sum(torch.square(excess), dim=1)
        rew *= self.cfg.rewards.w_action
        return rew
    
    def _reward_crash(self):
        rew = torch.where(self.crash_condition_buf, 1.0, 0.0)
        rew *= self.cfg.rewards.w_crash
        return rew

    # MISC toggles (exposed publics)
    def toggle_record(self, out_path=""):
        if not self.show_viewer:
            print("Cannot toggle record when viewer is disabled")
            return
        if self.gs_recorder.encoder is None:
            self.gs_recorder.start_record(out_path)
        else:
            self.gs_recorder.stop_record()

    def toggle_cam_follow(self):
        if not self.show_viewer:
            print("Cannot toggle camera follow when viewer is disabled")
            return
        self.gs_follower.toggle()            