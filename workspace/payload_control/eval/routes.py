"""Fixed test routes for eval_route.py: ENU waypoints (N, 3), cage scale (legs of a few metres, 2 m payload altitude) like
the training distribution, one shape per behaviour. Each is solved per test speed with the run's RouteConfig."""
import numpy as np

Z = 2.0 # payload altitude [m]

def _figure8(a=4.0, n=16):
    # Lemniscate of Gerono (x = a sin t, y = a/2 sin 2t), sampled as a closed polyline starting and ending at the centre
    t = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return np.stack([a * np.sin(t), 0.5 * a * np.sin(2.0 * t), np.full_like(t, Z)], axis=-1)

def _survey(leg=8.0, spacing=2.0, lines=3):
    pts = []
    for i in range(lines):
        xs = (0.0, leg) if i % 2 == 0 else (leg, 0.0)
        pts += [[xs[0], i * spacing, Z], [xs[1], i * spacing, Z]]
    return np.array(pts)

TEST_ROUTES = {
    "hover": np.array([[0.0, 0.0, Z]]), # stabilization only
    "line_stop": np.array([[0.0, 0.0, Z], [8.0, 0.0, Z]]), # accelerate, cruise, final stop
    "square": np.array([[0, 0, Z], [6, 0, Z], [6, 6, Z], [0, 6, Z], [0, 0, Z]], dtype=float), # 90° corners
    "zigzag": np.array([[0, 0, Z], [2.83, 2.83, Z], [5.66, 0, Z], [8.49, 2.83, Z], [11.31, 0, Z]]), # 90° alternating turns
    "survey": _survey(), # U-turns on a short spacing
    "figure8": _figure8(), # continuous curvature change, left and right turns
    "climb": np.array([[0, 0, Z], [4, 0, Z], [8, 0, Z + 1.5], [12, 0, Z + 1.5]], dtype=float), # vertical corners
}
