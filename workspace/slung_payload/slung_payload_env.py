import genesis as gs
import torch
import math
import os

import config as cfg

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

class SlungPayloadEnv:
    def __init__(self, num_envs: int, env_cfg: cfg.EnvConfig, cable_cfg: cfg.CableConfig, show_viewer: bool=False, device: str="cuda"):
        self.device = torch.device("cuda")
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        self.env_cfg = env_cfg
        self.cable_cfg = cable_cfg
        self.dt = env_cfg.dt
        
        # Drone-Payload system parameters
        self.base_link_above_gnd = 0.068

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
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=DRONE_PATH,
                pos=(0.0, 0.0, self.base_link_above_gnd),
                euler=(0.0, 0.0, 0.0),
                propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                propellers_spin=(-1, 1, -1, 1), # per prop -1=CW, +1=CCW
                prioritize_urdf_material=True,
                links_to_keep=["base_link"],
            ),
        )

        # Payload
        payload_link_names = [f"attach_{x}" for x in range(self.env_cfg.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=(1.0, 0.0, 0.025),
                euler=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                links_to_keep=payload_link_names,
            ),
        )
        self.payload_attach_links = [self.payload.get_link(x) for x in payload_link_names]
        self.payload_attach_links_idx = [link.idx_local for link in self.payload_attach_links]

        
        
        # Build scene
        self.scene.build(n_envs=num_envs)

        # Effective mass of two-body system (good approx?)
        drone_mass = self.drone.get_mass()
        payload_mass = self.payload.get_mass()
        self.mass_eff = 1.0 / (1.0 / drone_mass + 1.0 / payload_mass)

        # Initialize system state buffers
        self.obs_buf = torch.zeros((self.num_envs, self.env_cfg.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.env_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.next_commands = torch.zeros((self.num_envs, self.env_cfg.num_commands), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)
        self.next_actions = torch.zeros((self.num_envs, self.env_cfg.num_actions), device=self.device, dtype=gs.tc_float)

        self.drone_base_link_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone_base_link_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.payload_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_pos = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)
        self.payload_attach_vel = torch.zeros((self.num_envs, self.env_cfg.num_tethers, 3), device=self.device, dtype=gs.tc_float)

        self.relative_projected_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float) # relative position in 2D plane

        self.extras = dict()

    def step_sim(self, actions):

        # Get quadrotor states
        self.drone_base_link_pos = self.drone.get_pos(relative=False)
        self.drone_base_link_quat = self.drone.get_quat(relative=False)
        self.drone_base_link_lin_vel = self.drone.get_vel()
        self.drone_base_link_ang_vel = self.drone.get_ang()

        # Payload center pos/vel
        self.payload_pos = self.payload.get_pos(relative=False)
        self.payload_lin_vel = self.payload.get_vel()

        # Payload attachment points pos/vel
        self.payload_attach_pos = self.payload.get_links_pos(self.payload_attach_links_idx, relative=False)
        self.payload_attach_vel = self.payload.get_links_vel(self.payload_attach_links_idx)

        # Update cable dynamics
        self._step_cables()


        self.scene.step()
        pass 

    def _step_cables(self):
        # Run the math from previous cable.py 
        # Calculate the implicit euler solution to unilateral spring-damper system
         
        pass

    def _calculate_cable_force(self):

        pass
