"""Render quadcopter.urdf.j2 into a plain URDF that gs.morphs.Drone can load.

body.obj and prop.obj are both authored in millimeters: prop.obj's 46.44mm X-extent matches 2 * a
propeller radius around 2.3cm, and body.obj's own measured arm radius matches the drone's originally
authored arm length almost exactly under the same mm assumption. So every physical dimension - arm
length, motor mount height, propeller radius, collision box - is derived directly from the raw mesh
coordinates rather than fit to a chosen target size.
"""

import argparse
import os

import jinja2
import numpy as np
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

MM_TO_M = 0.001


def box_inertia(mass: float, size_x: float, size_y: float, size_z: float) -> tuple[float, float, float]:
    ixx = mass / 12.0 * (size_y**2 + size_z**2)
    iyy = mass / 12.0 * (size_x**2 + size_z**2)
    izz = mass / 12.0 * (size_x**2 + size_y**2)
    return ixx, iyy, izz


def measure_prop(mesh_path: str) -> tuple[float, float]:
    """Measure the propeller's own radius and how far its mesh extends below its local origin, in meters."""
    mesh = trimesh.load(mesh_path, force="mesh")
    radius = mesh.bounding_box.extents[0] / 2.0 * MM_TO_M
    seat_depth = -mesh.bounds[0, 2] * MM_TO_M
    return radius, seat_depth


def measure_body(mesh_path: str) -> tuple[float, float, float, float]:
    """Measure the body's own motor-mount radius, motor-mount height, overall height, and vertical center.

    The body is a cross-shaped frame (4 arms along the +-X/+-Y axes). Each motor mount is modeled as a
    disconnected stack - main arm, then a mounting-plate disc, then a thin shaft tip poking up through
    where the propeller hub goes - so its highest point is the shaft tip, not the actual mounting surface.
    The real mount surface is found by taking the arm-tip vertices (large radius from the link origin),
    then walking down from the top past the largest vertical gap: that gap is the shaft floating clear of
    the mounting plate, so the solid cluster just below it is the true mount height.

    Its bounding box is not centered on the link origin (which sits close to the mount plane), so a
    collision box needs its own origin at the box's true vertical center to avoid leaving the mesh's lower
    half uncovered.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = mesh.vertices
    radius = np.linalg.norm(verts[:, :2], axis=1)
    tip_verts = verts[radius > 0.5 * radius.max()]
    tip_z_sorted = np.sort(tip_verts[:, 2])
    gap_idx = np.argmax(np.diff(tip_z_sorted))
    mount_z_raw = tip_z_sorted[gap_idx]

    mount_band = tip_verts[(tip_verts[:, 2] > mount_z_raw - 2.0) & (tip_verts[:, 2] <= mount_z_raw)]
    arm = np.linalg.norm(mount_band[:, :2], axis=1).mean() * MM_TO_M
    mount_z = mount_z_raw * MM_TO_M

    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    height = (zmax - zmin) * MM_TO_M
    center_z = (zmax + zmin) / 2.0 * MM_TO_M
    return arm, mount_z, height, center_z


def render(mass: float) -> str:
    prop_radius, prop_seat_depth = measure_prop(os.path.join(SCRIPT_DIR, "prop.obj"))
    arm, mount_z, size_z, box_origin_z = measure_body(os.path.join(SCRIPT_DIR, "body.obj"))

    # Collision box footprint covers the body's arm span plus the propellers' spinning disks.
    size_x = size_y = 2.0 * (arm + prop_radius)
    # Seat the propeller mesh's own bottom - not its local origin - flush with the mount surface.
    prop_z = mount_z + prop_seat_depth

    ixx, iyy, izz = box_inertia(mass, size_x, size_y, size_z)

    with open(os.path.join(SCRIPT_DIR, "quadcopter.urdf.j2")) as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    return template.render(
        mass=mass,
        ixx=ixx,
        iyy=iyy,
        izz=izz,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        box_origin_z=box_origin_z,
        scale_x=MM_TO_M,
        scale_y=MM_TO_M,
        scale_z=MM_TO_M,
        prop_radius=prop_radius,
        prop_scale=MM_TO_M,
        arm=arm,
        prop_z=prop_z,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.5, help="Total drone mass in kg")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "quadcopter.urdf"), help="Output URDF path")
    args = parser.parse_args()

    urdf = render(args.mass)
    with open(args.out, "w") as f:
        f.write(urdf)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
