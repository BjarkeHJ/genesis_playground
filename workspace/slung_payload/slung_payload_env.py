import genesis as gs
import torch
import math
import os
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

@dataclass
class CableConfig:
    stiffness: float
    damping: float
    rest_length: float
    diameter: float
    slack_transition_width: float

class SlungPayloadEnv:
    def __init__(self, num_envs: int, show_viewer: bool=False, device: str="cuda"):
        self.device = torch.device("cuda")
        self.num_envs = num_envs
        self.show_viewer = show_viewer

        self.dt = 0.01

        # Drone-Payload system parameters
        self.base_link_above_gnd = 0.068
        self.num_tethers = 3

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
                propellers_link_names=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                propellers_spin=(-1, 1, -1, 1), # per prop -1=CW, +1=CCW
                prioritize_urdf_material=True,
            ),
        )

        # Payload
        payload_link_names = [f"attach_{x}" for x in range(self.num_tethers)]
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=PAYLOAD_PATH,
                pos=(1.0, 0.0, 0.025),
                euler=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                links_to_keep=payload_link_names,
            ),
        )

        # Build scene
        self.scene.build(n_envs=num_envs)

        self.drone_base_link = self.drone.get_link("base_link")

    def step_sim(self):

        self.step_cables()
        self.scene.step()
        pass 

    def step_cables(self):
        # Run the math from previous cable.py 
        # Calculate the implicit euler solution to unilateral spring-damper system
        pass
