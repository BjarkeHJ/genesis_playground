"""
Calibrate the hover RPM of a drone URDF in Genesis.

Genesis models each propeller as F = kf * rpm^2 (along the prop link z-axis),
so for n props the hover RPM is:   rpm = sqrt(m * g / (n * kf))

1. Analytic estimate from the built entity's mass, kf and the scene gravity.
2. Empirical refinement: hold the RPM in mid-air, measure vertical acceleration a
   and correct with rpm *= sqrt(g / (g + a)) until |a| < tol. This also captures
   anything the formula misses (e.g. extra attached bodies).
"""

import argparse
import math
import os

import genesis as gs

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=DRONE_PATH)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--window", type=float, default=5.0, help="seconds of flight per measurement")
parser.add_argument("--tol", type=float, default=1e-4, help="accel tolerance [m/s^2]")
parser.add_argument("--max-iters", type=int, default=10)
parser.add_argument("--vis", action="store_true")
args = parser.parse_args()

SPAWN_POS = (0.0, 0.0, 2.0)  # high enough to never touch the ground during a window

# Init scene - MUST be the first thing done
gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=args.dt),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0.0, 2.5),
        camera_lookat=SPAWN_POS,
        camera_fov=40,
    ),
    show_viewer=args.vis,
)

drone = scene.add_entity(
    gs.morphs.Drone(
        file=args.urdf,
        pos=SPAWN_POS,
        euler=(0.0, 0.0, 0.0),
        propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
        propellers_spin=(-1, 1, -1, 1),
        prioritize_urdf_material=True,
    ),
)

scene.build()

# --- Analytic estimate ---
mass = float(drone.get_mass())
g = -float(scene.sim.options.gravity[2])
n_props = drone.n_propellers
kf = drone.KF
rpm = math.sqrt(mass * g / (n_props * kf))

print(f"mass={mass:.4f} kg  g={g:.3f} m/s^2  n_props={n_props}  kf={kf:.4e}  km={drone.KM:.4e}")
print(f"analytic hover rpm: {rpm:.8f}")


def measure_accel(rpm):
    """Reset drone at rest in mid-air, hold rpm for the window, return mean vertical accel."""
    drone.set_pos(SPAWN_POS, zero_velocity=True)
    drone.set_quat((1.0, 0.0, 0.0, 0.0), zero_velocity=True)
    n_steps = max(1, round(args.window / args.dt))
    for _ in range(n_steps):
        drone.set_propellers_rpm([rpm] * n_props)
        scene.step()
    return float(drone.get_vel()[2]) / (n_steps * args.dt)


# --- Empirical refinement ---
for it in range(args.max_iters):
    a_z = measure_accel(rpm)
    print(f"iter {it}: rpm={rpm:.3f}  a_z={a_z:+.3e} m/s^2")
    if abs(a_z) < args.tol:
        break
    rpm *= math.sqrt(g / (g + a_z))
else:
    print("warning: did not converge within max iterations")

print(f"\nhover rpm: {rpm:.8f}")
