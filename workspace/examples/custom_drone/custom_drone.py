import os

import genesis as gs
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "external", "models", "Starling2MaxURDF", "model", "starling2max.urdf")

# Init scene - MUST be the first thing done
gs.init(backend=gs.cpu)

# Setup a gs scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=True,
)

# Add a ground-plane to the scene
scene.add_entity(gs.morphs.Plane())

# Spawn in a drone entity (crazy flie)
drone = scene.add_entity(
    gs.morphs.Drone(
        file=os.path.join(MODEL_PATH),
        pos=(0.0, 0.0, 2.0),  # meters, Z-up
        euler=(0.0, 0.0, 0.0),  # scipy extrinsic x-y-z, degrees
        propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
        propellers_spin=(-1, 1, -1, 1),  # per propeller: -1 = CW, +1 = CCW
    ),
)

drone_link = drone.get_link("base_link")

# Build the scene with all entities (compiles kernels, visualizers etc.)
scene.build(n_envs=1)

hover_rpm = 50000

for i in range(10000):
    drone.set_propellers_rpm([hover_rpm, hover_rpm, hover_rpm, hover_rpm])

    # Apply external force of 1N on drone in +z direction
    if (i == 250):
        drone_link.apply_external_wrench(force=(0,0,1))

    scene.step()