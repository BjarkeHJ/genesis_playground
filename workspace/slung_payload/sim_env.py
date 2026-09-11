import genesis as gs
import torch
import math
import os

REPO_ROOT = ""
PAYLOAD_PATH = ""
DRONE_PATH = ""

class SlungPayloadEnv:
    def __init__(self, num_envs: int, show_viewer: bool=False, device: str="cuda"):
        self.device = torch.device("cuda")
        self.num_envs = num_envs
        self.show_viewer = show_viewer

        self.dt = 0.01

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

        # Payload

        # Build scene
        self.scene.build(n_envs=num_envs)

    def step_sim(self):

        self.step_cables()
        self.scene.step()
        pass 

    def step_cables(self):
        # Run the math from previous cable.py 
        # Calculate the implicit euler solution to unilateral spring-damper system
        pass
