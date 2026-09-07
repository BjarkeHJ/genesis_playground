
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import math

TEMPLATE_DIR = Path(__file__).resolve().parent

def generate(mass=1.0, size=(0.3, 0.3, 0.1), output="payload.urdf",
             body_color=(0.0, 0.0, 0.8, 1.0),
             panel_color=(0.9, 0.0, 0.0, 1.0),
             arrow_color=(0.05, 0.85, 0.05, 1.0)):
    size_x, size_y, size_z = size
    ixx = mass / 12.0 * (size_y**2 + size_z**2)
    iyy = mass / 12.0 * (size_x**2 + size_z**2)
    izz = mass / 12.0 * (size_x**2 + size_y**2)

    # Top face arrow (x-dir)
    top_panel_thickness = 0.004
    arrow_length = 0.35 * size_x
    arrow_head_size = 0.30 * arrow_length
    arrow_shaft_len = arrow_length - arrow_head_size
    arrow_shaft_width = 0.12 * arrow_length
    arrow_height = 0.01

    # Aero parameters
    air_density = 1.225        # [kg/m^3], sea level
    drag_coeff = 1.05          # bluff cube face
    drag_coeff_x = -0.5 * air_density * drag_coeff * (size_y * size_z)
    drag_coeff_y = -0.5 * air_density * drag_coeff * (size_x * size_z)
    drag_coeff_z = -0.5 * air_density * drag_coeff * (size_x * size_y)

    # Compute attachment point geometry
    aps = compute_attachment_points(size, delta0=0)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      trim_blocks=True, lstrip_blocks=True)

    urdf = env.get_template("payload.urdf.j2").render(
        mass=mass, size_x=size_x, size_y=size_y, size_z=size_z,
        ixx=ixx, iyy=iyy, izz=izz,
        top_panel_thickness=top_panel_thickness,
        arrow_length=arrow_length,
        arrow_shaft_len=arrow_shaft_len,
        arrow_shaft_width=arrow_shaft_width,
        arrow_head_size=arrow_head_size,
        arrow_height=arrow_height,
        drag_coeff_x=drag_coeff_x,
        drag_coeff_y=drag_coeff_y,
        drag_coeff_z=drag_coeff_z,
        body_color=" ".join(str(c) for c in body_color),
        panel_color=" ".join(str(c) for c in panel_color),
        arrow_color=" ".join(str(c) for c in arrow_color),
        attachment_points=aps
    )

    out_path = TEMPLATE_DIR / output
    out_path.write_text(urdf)

    return out_path

def compute_attachment_points(size, delta0 = 0):
    x, y, _ = size;
    c = [x/2, y/2]
    r = 0.9 * min(x/2,y/2) # center to vertex distance (assuming rectangle: 90% of distance from center to closest side))
    a = math.sqrt(3) * r # side length
    attach_ps = []
    for i in range(3):
        # apx = c[0] + r * math.cos(math.radians(120) * i + delta0)
        apx = r * math.cos(math.radians(120) * i + delta0)
        # apy = c[1] + r * math.sin(math.radians(120) * i + delta0)
        apy = r * math.sin(math.radians(120) * i + delta0)
        attach_ps.append([apx, apy])

    return attach_ps


def main():
    p = argparse.ArgumentParser(
        description="Generate Payload URDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--mass", type=float, default=1.0, help="Payload mass [kg]")
    p.add_argument("--size", type=float, nargs=3, default=(0.3,0.3,0.1), metavar=("X", "Y", "Z"), help="Box size ([m],[m],[m])")
    p.add_argument("--body-color", type=float, nargs=4, default=(0.0, 0.0, 0.8, 1.0), metavar=("R", "G", "B", "A"), help="Box body RGBA color")
    p.add_argument("--panel-color", type=float, nargs=4, default=(0.9, 0.0, 0.0, 1.0), metavar=("R", "G", "B", "A"), help="Top panel RGBA color")
    p.add_argument("--arrow-color", type=float, nargs=4, default=(0.05, 0.85, 0.05, 1.0), metavar=("R", "G", "B", "A"), help="Direction arrow RGBA color")
    p.add_argument("-o", "--output", type=str, default="payload.urdf")

    args = p.parse_args()

    generate(mass=args.mass, size=tuple(args.size), output=args.output,
              body_color=tuple(args.body_color),
              panel_color=tuple(args.panel_color),
              arrow_color=tuple(args.arrow_color))

if __name__ == "__main__":
    main()