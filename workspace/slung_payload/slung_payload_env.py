import genesis as gs
import torch
import os

import config as cfg

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

class SlungPayloadEnv:
    def __init__(self, num_envs: int, env_cfg: cfg.EnvConfig, tether_cfg: cfg.TetherConfig, show_viewer: bool=False, device: str="cuda"):
        self.device = torch.device("cuda")
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        self.env_cfg = env_cfg
        self.tether_cfg = tether_cfg
        self.dt = env_cfg.dt
        
        # Genesis-World Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt), # physics rate

            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ), # viewer (default 60 fps update rate)
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1)),
                                                background_color=(0.2, 0.2, 0.2)), # scene visual settings
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
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

        # Drone
        drone_attach_links = [f"attach{x}_link" for x in range(self.tether_cfg.num_tethers)]
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=DRONE_PATH,
                pos=self.env_cfg.reset_pos,
                euler=(0.0, 0.0, 0.0),
                propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                propellers_spin=(-1, 1, -1, 1), # per prop -1=CW, +1=CCW
                prioritize_urdf_material=True,
                links_to_keep=drone_attach_links,
            ),
        )
        self.drone_attach_links = [self.drone.get_link(x) for x in drone_attach_links]
        self.drone_attach_links_idx = [link.idx_local for link in self.drone_attach_links]

        # Payload
        payload_attach_links = [f"attach_{x}" for x in range(self.env_cfg.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=(1.0, 0.0, 0.025),
                euler=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                links_to_keep=payload_attach_links,
            ),
        )
        self.payload_attach_links = [self.payload.get_link(x) for x in payload_attach_links]
        self.payload_attach_links_idx = [link.idx_local for link in self.payload_attach_links]
        
        # Build scene
        self.scene.build(n_envs=num_envs)

        # Effective mass of two-body system (good approx?)
        drone_mass = self.drone.get_mass()
        payload_mass = self.payload.get_mass()
        self.mass_eff = 1.0 / (1.0 / drone_mass + 1.0 / payload_mass)

        # Initialized state (reset pos/quat)
        self.drone_base_link_init_pos = torch.tensor(self.env_cfg.reset_pos, device=self.device)
        self.drone_base_link_init_quat = torch.tensor(self.env_cfg.reset_quat, device=self.device)

        # Initialize system state buffers
        self.obs_buf = torch.zeros((self.num_envs, self.env_cfg.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.env_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.next_commands = torch.zeros((self.num_envs, self.env_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)

        self.drone_base_link_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_attach_pos = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)
        self.drone_attach_vel = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)

        self.payload_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.payload_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_pos = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_vel = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)

        self.relative_projected_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # relative position in 2D plane

        self.extras = dict() # Extra information for logging

        self.reset()

    def get_observations(self):
            return self.obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.drone_base_link_pos[envs_idx] = self.drone_base_link_init_pos
        self.drone_base_link_quat[envs_idx] = self.drone_base_link_init_quat

        self.drone.set_pos(self.drone_base_link_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_base_link_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone_base_link_lin_vel[envs_idx] = 0.0
        self.drone_base_link_ang_vel[envs_idx] = 0.0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs), device=self.device)
        self._u
        return self.get_observations()

    def step(self, actions):
        # Drone base_link pos/vel
        self.drone_base_link_pos = self.drone.get_pos(relative=False)
        self.drone_base_link_quat = self.drone.get_quat(relative=False)
        self.drone_base_link_lin_vel = self.drone.get_vel(relative=False)
        self.drone_base_link_ang_vel = self.drone.get_ang()
        # Drone attachment point pos/vel
        self.drone_attach_pos = self.drone.get_links_pos(self.drone_attach_links_idx, relative=False)
        self.drone_attach_vel = self.drone.get_links_vel(self.drone_attach_links_idx, relative=False)

        # Payload center pos/vel
        self.payload_pos = self.payload.get_pos(relative=False)
        self.payload_quat = self.payload.get_quat(relative=False)
        self.payload_lin_vel = self.payload.get_vel(relative=False)
        # Payload attachment points pos/vel
        self.payload_attach_pos = self.payload.get_links_pos(self.payload_attach_links_idx, relative=False)
        self.payload_attach_vel = self.payload.get_links_vel(self.payload_attach_links_idx, relative=False)

        # Update tether dynamics
        self._step_tethers()

        # Fly drone (TEMP)
        self.drone.set_propellers_rpm(16343)
        
        self.scene.step()

    def _step_tethers(self):
        # Calculate the implicit euler solution to unilateral spring-damper system

        vec = self.payload_attach_pos - self.drone_attach_pos # [num_envs, num_tethers, 3]
        length = torch.linalg.norm(vec, dim=-1) # norm over xyz [num_envs, num_tethers]
        uhat = vec / torch.clamp(length, min=cfg._EPS).unsqueeze(-1) # [num_envs, num_tethers, 3]
        
        extension = length - self.tether_cfg.rest_length # [num_envs, num_tethers]
        activation = 0.5 * (1.0 + torch.tanh(extension / self.tether_cfg.activation_delta)) # [num_envs, num_tethers]
        extension_rate = ((self.payload_attach_vel - self.drone_attach_vel) * uhat).sum(dim=-1) # [num_envs, num_tethers]

        # Backward euler
        damping_implicit = self.tether_cfg.damping + self.dt * self.tether_cfg.stiffness
        extension_rate_implicit = (self.mass_eff * extension_rate - self.dt * self.tether_cfg.stiffness * extension) / (self.mass_eff + self.dt * damping_implicit)
        tension = activation * (self.tether_cfg.stiffness * extension + damping_implicit * extension_rate_implicit)
        tension = torch.clamp(tension, min=0.0)

        force_on_payload = -tension.unsqueeze(-1) * uhat
        force_on_drone = -force_on_payload

        # Apply forces on drone/payload - Automatically computes torques based on link offsets
        self.drone.apply_links_external_wrench(force_on_drone, links_idx_local=self.drone_attach_links_idx)
        self.payload.apply_links_external_wrench(force_on_payload, links_idx_local=self.payload_attach_links_idx)

    def _at_target(self):
        pass

    def _resample_commands(self, envs_idx):
        pass

    def _update_observation(self):
        pass

    
