import genesis as gs

import torch
import math
from tensordict import TensorDict

from system_model.utils.qc_attitude_control import quat_yaw
from config.config_env import *
from config.paths import DRONE_PATH, PAYLOAD_PATH
from components.math_utils import rotate_to_heading, wrap_angle, rotate_by_quat, gs_rand_float
from components.tether import TetherModel
from components.flight_control import FlightController
from components.domain_rand import DomainRandomizer
from components.reference import TrajectoryBank, ReferenceTrajectory
from components.rewards import RewardsMixin
from components.viewer import TargetMarker, RoutePath, PreviewMarkers, attach_viewer_logs

class PayloadControlEnv(RewardsMixin):
    # Drone carrying a tethered payload; the policy commands [thrust, roll, pitch, yaw rate] so the payload tracks a
    # reference trajectory. Components: FlightController (action -> RPM), TetherModel (suspension wrench),
    # DomainRandomizer (per-episode params and initial state), ReferenceTrajectory (per-env route from the trajectory
    # bank, its clock and yaw reference), RewardsMixin (reward terms). Only this class talks to the Genesis scene/solver
    def __init__(self, env_cfg: EnvConfig, show_viewer: bool=False, device: str="cuda", bank: TrajectoryBank=None):
        # bank: routes to draw from; None loads the trajectory bank of the configs (evaluations pass their own)
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
        self.ref_cfg = self.cfg.ref_cfg
        self.rew_cfg = self.cfg.rew_cfg

        self._build_scene()
        self._init_solver_indices()

        # Components. Nominal masses are identical across envs at build time; get_mass() returns a tensor (scalar or per-env)
        self.gravity = abs(self.scene.sim.options.gravity[2])
        drone_mass0 = self.drone.get_mass().flatten()[0].item()
        payload_mass0 = self.payload.get_mass().flatten()[0].item()
        self.drone_init_pos = torch.tensor(self.sys_cfg.drone_reset_pos, device=self.device, dtype=gs.tc_float)
        self.drone_init_quat = torch.tensor(self.sys_cfg.drone_reset_quat, device=self.device, dtype=gs.tc_float)
        self.payload_init_pos = torch.tensor(self.sys_cfg.payload_reset_pos, device=self.device, dtype=gs.tc_float)
        self.payload_init_quat = torch.tensor(self.sys_cfg.payload_reset_quat, device=self.device, dtype=gs.tc_float)

        self.ctrl = FlightController(
            self.cfg, 
            self.drone, 
            total_mass=drone_mass0 + payload_mass0, 
            gravity=self.gravity, 
            device=self.device
        )
        self.tether = TetherModel(
            self.cfg, 
            device=self.device
        )
        self.dr = DomainRandomizer(
            self.cfg, 
            self.tether, 
            self.ctrl, 
            drone_mass0, 
            payload_mass0, 
            gravity=self.gravity, 
            device=self.device
        )

        # Payload reference: routes solved offline (utils/build_bank.py builds the bank for the current configs), placed
        # per env at reset; ref_offsets are the current point and the preview times
        self.bank = bank if bank is not None else TrajectoryBank.load(self.cfg.bank_cfg, self.cfg.route_cfg, self.cfg.sampler_cfg, device=self.device)
        self.ref = ReferenceTrajectory(self.bank, self.num_envs, self.device, self.ref_cfg, compile=self.cfg.torch_compile)
        self.ref_offsets = torch.tensor((0.0, *self.ref_cfg.preview_s), device=self.device, dtype=gs.tc_float)

        # The env's torch code is hundreds of tiny ops, bound by Python dispatch and kernel launches regardless of num_envs.
        # torch.compile fuses each stage into a few kernels (first steps are slow while it compiles). Wrap stages here only:
        # helpers they call (components, math_utils) are inlined into the stage, so a separately compiled helper only adds dispatch
        maybe_compile = torch.compile if self.cfg.torch_compile else (lambda fn: fn)
        self._setpoints_fn = maybe_compile(self.ctrl.setpoints)
        self._rpms_fn = maybe_compile(self.ctrl.rpms)
        self._tether_wrench_fn = maybe_compile(self.tether.wrench)
        self._buffers_fn = maybe_compile(self._compute_buffers)
        self._post_step_fn = maybe_compile(self._post_step)
        self._observations_fn = maybe_compile(self._compute_observations)

        self._init_buffers()
        self._init_rewards()

        # Viewer-only drawing of env 0's reference: current point and yaw reference, preview points, full route
        self.target_marker = TargetMarker(self.scene) if self.show_viewer else None
        self.preview_markers = PreviewMarkers(self.scene, len(self.ref_cfg.preview_s)) if self.show_viewer else None
        self.route_path = RoutePath(self.scene) if self.show_viewer else None
        self.ref_points0 = None # env 0's reference now and at the preview times (1+K, 3), kept for the viewer
        if self.show_viewer:
            attach_viewer_logs(self)

        if self.cfg.torch_compile:
            self._warm_up_compile()
        self.reset()

    # ======================== SETUP ========================

    def _warm_up_compile(self):
        # Compile every stage now rather than inside the first training iteration: the first call of each compiled stage
        # traces and compiles it (seconds), so one throwaway hover step pays for all of them. The reset after it (in
        # __init__) restarts every episode; the success counters are cleared here since reset() leaves them alone
        self.reset()
        self.step(torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device) # compile + first launches done before construction returns
        self.success_count.zero_()
        self.episode_count.zero_()
        self.success_steps = 0

    def _build_scene(self):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.cfg.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=60,
                # refresh_rate=30,
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
                batch_links_info=True,
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

        self.scene.build(n_envs=self.num_envs)

    def _init_solver_indices(self):
        # Links/dofs of both entities in one solver call each (every Genesis call carries a fixed Python validation cost)
        self.rigid_solver = self.scene.sim.rigid_solver
        self.tether_links_idx = torch.tensor([link.idx for link in self.drone_attach_links + self.payload_attach_links], device=self.device, dtype=gs.tc_int) # [drone..., payload...]
        self.base_links_idx = torch.tensor([self.drone.base_link_idx, self.payload.base_link_idx], device=self.device, dtype=gs.tc_int) # [drone, payload]
        
        # Free joints only, see reset_idx: qpos [pos, quat] equals the base link pose (base links have no inertial offset)
        assert self.drone.n_qs == 7 and self.payload.n_qs == 7 and self.drone.n_dofs == 6 and self.payload.n_dofs == 6
        self.base_qs_idx = list(range(self.drone.q_start, self.drone.q_end)) + list(range(self.payload.q_start, self.payload.q_end))
        self.base_dofs_idx = list(range(self.drone.dof_start, self.drone.dof_end)) + list(range(self.payload.dof_start, self.payload.dof_end))
        
        # Mass-carrying links of [drone..., payload...], see _apply_masses (same selection as entity.set_mass)
        drone_mass_links = [link.idx for link in self.drone.links if not link.is_fixed]
        self.mass_links_idx = drone_mass_links + [link.idx for link in self.payload.links if not link.is_fixed]
        self.num_drone_mass_links = len(drone_mass_links)

    def _init_buffers(self):
        zeros = lambda *shape, dtype=gs.tc_float: torch.zeros((self.num_envs, *shape), device=self.device, dtype=dtype)

        self.world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=gs.tc_float).expand(self.num_envs, -1) # body z-axis for thrust axis
        self.world_down = -self.world_up # gravity direction for projected gravity
        s = self.cfg.obs_scales
        self.pos_obs_scale = torch.tensor([s.px, s.py, s.pz], device=self.device, dtype=gs.tc_float)
        self.vel_obs_scale = torch.tensor([s.vx, s.vy, s.vz], device=self.device, dtype=gs.tc_float)
        # Preview point k ~ v * τ_k + error away: scaled by the distance it can plausibly reach, so far points don't saturate
        self.preview_obs_scale = 1.0 / (self.cfg.sampler_cfg.v_des_max * self.ref_offsets[1:] + s.preview_margin)

        # Environment state buffers
        self.obs_buf = zeros(self.cfg.num_obs)
        self.critic_obs_buf = zeros(self.cfg.num_obs + self.cfg.num_privileged_obs) # obs + DomainRandomizer.privileged_obs
        self.reset_buf = zeros(dtype=gs.tc_int)
        self.episode_length_buf = zeros(dtype=gs.tc_int)
        self.actions = zeros(self.num_actions)
        self.prev_actions = zeros(self.num_actions)
        self.actions_raw = zeros(self.num_actions)
        self.crash_condition = zeros(dtype=gs.tc_bool)

        # Observation state buffers
        self.payload_pos_err = zeros(3) # reference - payload, drone heading frame (RTK)
        self.payload_vel_err = zeros(3) # reference velocity - payload velocity, drone heading frame
        self.payload_preview = zeros(len(self.ref_cfg.preview_s), 3) # reference at τ + τ_k - payload, drone heading frame
        self.payload_yaw_err = zeros() # wrap(psi_ref - payload yaw)
        self.yaw_rate_ref = zeros() # yaw-rate reference psi_dot_ref
        self.payload_vel = zeros(3) # Drone heading frame (RTK)
        self.payload_proj_g = zeros(3) # Gravity in payload body axes
        self.payload_body_rates = zeros(3) # Body frame (IMU gyro)
        self.payload_swing_angles = zeros(2) # Drone-Payload relative position vector
        self.payload_swing_angles_rates = zeros(2) # --
        self.drone_proj_g = zeros(3) # Gravity in drone body axes
        self.drone_body_rates = zeros(3) # Drone body frame (IMU)
        self.drone_payload_rel_yaw = zeros() # For tether tangling termination (PL RKT+MAG)

        # Non-observations
        self.payload_pos = zeros(3) # world, for ground crash
        self.ref_pos = zeros(3) # world, current reference point (viewer)
        self.drone_yaw = zeros() # abs drone yaw
        self.swing_amp = zeros() # equivalent pendulum amplitude [rad]

        # Per-episode tracking accumulators (success rate and Task/ logs), see _post_step and _log_episode_tracking
        self.ep_err_sum = zeros()
        self.ep_err_sq_sum = zeros()
        self.ep_err_max = zeros()
        self.ep_swing_sum = zeros()
        self.ep_steps = zeros()
        self.success_count = torch.zeros((), device=self.device)
        self.episode_count = torch.zeros((), device=self.device)
        self.success_steps = 0 # policy steps into the current success window

        self.extras = dict() # logging information

    # ======================== RL INTERFACE ========================

    def get_observations(self):
        return TensorDict({
            "policy": self.obs_buf, # TODO: noisy
            "critic": self.critic_obs_buf, # clean + priv obs
        }, batch_size=[self.num_envs])

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_buffers()
        self._update_observations()
        self._draw_reference()
        return self.get_observations()

    def step(self, actions):
        self.actions_raw = actions # pre-clip policy sample, for the action bound penalty
        self.actions, substep_sps = self._setpoints_fn(actions) # clipped actions

        # Setpoints held constant (zero-order hold) while the inner loop runs at sim rate
        # Viewer refreshed on the last substep only (policy rate), see viewer.attach_viewer_logs for the matching pacing
        for i in range(self.decimation):
            self._sim_step(substep_sps[i], update_visualizer=(i == self.decimation - 1))

        self.episode_length_buf += 1
        self.ref.step(self.dt) # reference clock and yaw follower, so the errors below compare the same instant

        self._update_buffers()
        self.extras["time_outs"] = self._post_step_fn()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1, )))
        self._update_buffers() # update again to catch initial states (cheap but could be guarded)
        self._update_observations()
        self.prev_actions = self.actions
        self._draw_reference()

        self._log_success_rate()

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        self.extras["episode"] = {} # cleared every step so rsl_rl never re-logs stale values
        if len(envs_idx) == 0:
            return

        self._log_episode_rewards(envs_idx)
        self._log_episode_tracking(envs_idx)

        # New system parameters
        self.dr.randomize(envs_idx)
        self._apply_masses(envs_idx)

        # New route (each env is its own world, so every route starts at the origin), then the initial state moved onto
        # its start point and moving with it (mid-route starts). Payload heading within init_yaw_err of the yaw reference
        n = len(envs_idx)
        ref_pos0, ref_vel0 = self.ref.assign_random(envs_idx, torch.zeros((n, 2), device=self.device), p_mid_route=self.ref_cfg.p_mid_route_start)
        self._draw_route(envs_idx)
        yaw_err = self.ref_cfg.init_yaw_err * (1.0 if self.dr.enabled else 0.0)
        payload_yaw = wrap_angle(self.ref.psi_ref[envs_idx] + gs_rand_float(-yaw_err, yaw_err, (n, ), self.device))
        drone_pos, drone_quat, drone_vel, payload_pos, payload_quat, payload_vel = self.dr.sample_initial_state(envs_idx, payload_yaw)
        shift = ref_pos0 - self.payload_init_pos
        drone_pos, payload_pos = drone_pos + shift, payload_pos + shift
        drone_vel, payload_vel = drone_vel + ref_vel0, payload_vel + ref_vel0

        # Reset drone and payload together through their free joints. Forward kinematics runs once, in the qpos write
        zeros = torch.zeros_like(drone_vel)
        self.rigid_solver.set_dofs_velocity(torch.cat([drone_vel, zeros, payload_vel, zeros], dim=1), self.base_dofs_idx, envs_idx, skip_forward=True) # [lin, ang] per entity
        qpos = torch.cat([drone_pos, drone_quat, payload_pos, payload_quat], dim=1)
        self.rigid_solver.set_qpos(qpos, self.base_qs_idx, envs_idx)

        self.ctrl.reset(envs_idx, drone_quat)

        for buf in (self.ep_err_sum, self.ep_err_sq_sum, self.ep_err_max, self.ep_swing_sum, self.ep_steps):
            buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0
        self.prev_actions[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0
        self.crash_condition[envs_idx] = 0

    # ======================== SIMULATION ========================

    def _sim_step(self, sp, update_visualizer=True):
        # Motor RPM from attitude/rate controller and mixer
        self.drone.set_propellers_rpm(self._rpms_fn(self.drone.get_quat(), self.drone.get_ang(), sp))

        # Tether wrench on the attach links [drone..., payload...], see tether_links_idx
        pos = self.rigid_solver.get_links_pos(self.tether_links_idx, relative=True)
        vel = self.rigid_solver.get_links_vel(self.tether_links_idx, relative=True)
        force, torque = self._tether_wrench_fn(pos, vel)
        self.rigid_solver.apply_links_external_wrench(force=force, torque=torque, links_idx=self.tether_links_idx)

        self.scene.step(update_visualizer=update_visualizer)

    def _apply_masses(self, envs_idx):
        # entity.set_mass for both entities in one solver write: scale each entity's link masses (and inertia) to its total
        links_mass = self.rigid_solver.get_links_mass(self.mass_links_idx, envs_idx)
        k = self.num_drone_mass_links
        drone_links_mass, payload_links_mass = links_mass[:, :k], links_mass[:, k:]
        links_mass = torch.cat([
            drone_links_mass * self.dr.drone_mass[envs_idx, None] / drone_links_mass.sum(dim=-1, keepdim=True),
            payload_links_mass * self.dr.payload_mass[envs_idx, None] / payload_links_mass.sum(dim=-1, keepdim=True),
        ], dim=1)
        self.rigid_solver.set_links_mass(links_mass, self.mass_links_idx, envs_idx, scale_inertia=True)

    # ======================== STATE ========================

    def _update_buffers(self):
        # Base link states of [drone, payload], see base_links_idx
        pos = self.rigid_solver.get_links_pos(self.base_links_idx, relative=True)
        quat = self.rigid_solver.get_links_quat(self.base_links_idx, relative=True)
        vel = self.rigid_solver.get_links_vel(self.base_links_idx, relative=True)
        ang = self.rigid_solver.get_links_ang(self.base_links_idx)
        # Reference at τ + ref_offsets (world): current point and preview, outside the compiled stage (searchsorted/gather)
        ref_pos, ref_vel = self.ref.eval(self.ref_offsets, max_order=1)
        if self.show_viewer:
            self.ref_points0 = ref_pos[0]
        self._buffers_fn(pos, quat, vel, ang, ref_pos, ref_vel[:, 0], self.ref.psi_ref, self.ref.psi_dot_ref)

    def _compute_buffers(self, pos, quat, vel, ang, ref_pos, ref_vel, psi_ref, psi_dot_ref):
        # ref_pos (N, 1+K, 3): reference now and at the preview times; ref_vel (N, 3): reference velocity now
        drone_pos, payload_pos = pos.unbind(1)
        drone_quat, payload_quat = quat.unbind(1)
        drone_vel, payload_vel = vel.unbind(1)
        drone_ang, payload_ang = ang.unbind(1)

        # Drone attitude (heading frame defined by drone yaw)
        self.drone_yaw[:] = quat_yaw(drone_quat)
        yaw = self.drone_yaw
        self.drone_proj_g[:] = rotate_by_quat(self.world_down, drone_quat, inverse=True)
        self.drone_body_rates[:] = rotate_by_quat(drone_ang, drone_quat, inverse=True)

        # Payload tracking and attitude
        payload_yaw = quat_yaw(payload_quat)
        self.payload_pos[:] = payload_pos
        self.ref_pos[:] = ref_pos[:, 0]
        self.payload_pos_err[:] = rotate_to_heading(ref_pos[:, 0] - payload_pos, yaw)
        self.payload_vel_err[:] = rotate_to_heading(ref_vel - payload_vel, yaw)
        self.payload_preview[:] = rotate_to_heading(ref_pos[:, 1:] - payload_pos[:, None], yaw[:, None])
        self.payload_yaw_err[:] = wrap_angle(psi_ref - payload_yaw) # can wrap: observed as sin/cos
        self.yaw_rate_ref[:] = psi_dot_ref
        self.payload_vel[:] = rotate_to_heading(payload_vel, yaw)
        self.payload_proj_g[:] = rotate_by_quat(self.world_down, payload_quat, inverse=True)
        self.payload_body_rates[:] = rotate_by_quat(payload_ang, payload_quat, inverse=True)

        self.drone_payload_rel_yaw[:] = wrap_angle(payload_yaw - yaw) # will terminate before wrap-issue

        r_w = payload_pos - drone_pos # drone->payload vector
        r_dot_w = payload_vel - drone_vel # drone->payload vector rate
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
        thrust_axis = rotate_by_quat(self.world_up, drone_quat)
        e_kin = 0.5 * torch.sum(torch.square(r_dot_perp), dim=1) / (L.squeeze(1) * self.dr.g_eff)
        e_pot = 1.0 - torch.sum(q * thrust_axis, dim=1)
        self.swing_amp[:] = torch.acos(torch.clamp(1.0 - (e_kin + e_pot), min=-1.0, max=1.0))

    # ======================== TERMINATION / SUCCESS ========================

    def _post_step(self):
        # Terminations, rewards and success accounting on the terminal state (before reset). Returns the timeouts
        timeout = self._update_crash_conditions()
        self._compute_reward()

        # Episode accumulators. Success: survived to timeout with a small mean error to the reference
        err = torch.norm(self.payload_pos_err, dim=1)
        self.ep_err_sum += err
        self.ep_err_sq_sum += torch.square(err)
        self.ep_err_max[:] = torch.maximum(self.ep_err_max, err)
        self.ep_swing_sum += self.swing_amp
        self.ep_steps += 1.0
        mean_err = self.ep_err_sum / torch.clamp(self.ep_steps, min=1.0)
        success = self.reset_buf & ~self.crash_condition & (mean_err < self.cfg.dr_cfg.success_pos_err)
        self.success_count += success.sum()
        self.episode_count += self.reset_buf.sum()
        return timeout

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
            | (self.payload_pos[:, 2] <= self.cfg.terminate_if_payload_below_z)
        )
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout | self.crash_condition
        return timeout

    def _log_episode_tracking(self, envs_idx):
        # Tracking quality of the finished episodes of envs_idx (means over envs), before their accumulators restart
        steps = torch.clamp(self.ep_steps[envs_idx], min=1.0)
        duration = self.ref.duration[envs_idx]
        completion = torch.where(duration > 0.0, torch.clamp(self.ref.tau[envs_idx] / torch.clamp(duration, min=EPS), max=1.0), 1.0)
        ep = self.extras["episode"]
        ep["Task/mean_pos_err"] = torch.mean(self.ep_err_sum[envs_idx] / steps)
        ep["Task/rms_pos_err"] = torch.mean(torch.sqrt(self.ep_err_sq_sum[envs_idx] / steps))
        ep["Task/max_pos_err"] = torch.mean(self.ep_err_max[envs_idx])
        ep["Task/mean_swing_deg"] = torch.rad2deg(torch.mean(self.ep_swing_sum[envs_idx] / steps))
        ep["Task/crash_rate"] = torch.mean(self.crash_condition[envs_idx].float())
        ep["Task/route_completion"] = torch.mean(completion)

    def _log_success_rate(self):
        # Fraction of episodes ending in success over each window of success_interval policy steps
        self.success_steps += 1
        if self.success_steps < self.cfg.dr_cfg.success_interval or self.episode_count.item() == 0:
            return
        self.extras["episode"]["Task/success_rate"] = (self.success_count / self.episode_count).item()
        self.success_count.zero_()
        self.episode_count.zero_()
        self.success_steps = 0

    # ======================== OBSERVATIONS ========================

    def _update_observations(self):
        self.obs_buf, self.critic_obs_buf = self._observations_fn()

    def _compute_observations(self):
        # Yaw-agnostic: heading-frame, body-frame or relative quantities only
        s = self.cfg.obs_scales
        obs = torch.cat(
            [
                # Task (heading frame)
                torch.clamp(self.payload_pos_err * self.pos_obs_scale, -1.0, 1.0), # payload pos err to the reference, dim=3
                torch.clamp(self.payload_vel * self.vel_obs_scale, -1.0, 1.0), # payload vel, dim=3
                torch.clamp(self.payload_vel_err * s.v_err, -1.0, 1.0), # reference vel - payload vel, dim=3
                torch.clamp(self.payload_preview * self.preview_obs_scale[None, :, None], -1.0, 1.0).flatten(1), # preview, dim=3K

                # Payload dim=13
                self.payload_proj_g, # gravity in payload body axes, dim=3
                torch.clamp(self.payload_body_rates * s.pl_body_rates, -1.0, 1.0), # payload body rates, dim=3
                torch.clamp(self.payload_swing_angles * s.pl_swing_angles, -1.0, 1.0), # swing angles (heading frame), dim=2
                torch.clamp(self.payload_swing_angles_rates * s.pl_swing_angles_rates, -1.0, 1.0), # swing angles rates (heading frame), dim=2
                torch.clamp(self.drone_payload_rel_yaw.unsqueeze(-1) * s.rel_yaw, -1.0, 1.0), # drone-payload relative yaw, dim=1
                
                torch.sin(self.payload_yaw_err).unsqueeze(-1), # payload yaw err, dim=1
                torch.cos(self.payload_yaw_err).unsqueeze(-1), # -- dim=1
                torch.clamp(self.yaw_rate_ref.unsqueeze(-1) / self.sys_cfg.max_yaw_rate, -1.0, 1.0), # yaw-rate reference, dim=1
                
                # Drone
                self.drone_proj_g, # gravity in drone body axes, dim=3
                torch.clamp(self.drone_body_rates * s.drone_body_rates, -1.0, 1.0), # drone body rates, dim=3
                self.actions, # last setpoint [thrust, roll_sp, pitch_sp, yawrate_sp], dim=4
            ],
            axis=1
        )
        return obs, torch.cat([obs, self.dr.privileged_obs()], dim=-1)

    def _draw_reference(self):
        # Viewer-only, every step: env 0's current reference point and yaw reference, and the preview points it observes
        if self.target_marker is not None:
            self.target_marker.update(torch.cat([self.ref_pos[0], self.ref.psi_ref[:1]]))
            self.preview_markers.update(self.ref_points0[1:])

    def _draw_route(self, envs_idx, spacing_s=0.1, max_points=1000):
        # Viewer-only, when env 0 gets a new route: its full reference path, sampled every spacing_s
        if self.route_path is None or not (envs_idx == 0).any():
            return
        duration = self.ref.duration[0].item()
        n = int(min(max(duration / spacing_s, 2), max_points)) if duration > 0.0 else 0
        t = torch.linspace(0.0, duration, max(n, 1), device=self.device) - self.ref.tau[0]
        self.route_path.set(self.ref.eval(t, max_order=0, env_ids=torch.zeros(1, device=self.device, dtype=torch.long))[0, 0][:n])
