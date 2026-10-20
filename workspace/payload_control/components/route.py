import math
import numpy as np

from config.config_env import EPS, RouteConfig, RouteSamplerConfig

# Route geometry: ENU waypoints + v_des -> cleaned polyline with filleted corners, as nodes with fixed node velocities.
# Pure NumPy, float64, one route at a time. The min-snap solve on top of the nodes lives in its own module

# Generator internals: geometric rules and tolerances, independent of the drone-payload system. Not meant to be tuned;
# the mission spec and system envelope are in RouteConfig
A_LAT_FRAC = 0.75 # lateral acceleration on a fillet arc, fraction of a_max: sets the corner speed v²/R
MIN_RADIUS = 1.0 # fillet radius floor [m]: at low speed v²/a_lat would round corners to near-sharp turns
FILLET_FIT = 0.5 # fillet tangent distance <= this fraction of the shorter adjacent segment (two fillets share a segment)
THETA_STOP = math.radians(170.0) # sharper turns stop at the corner: the fillet would shrink to nothing
MIN_POINT_DIST = 0.1 # waypoints closer than this to the previous one are duplicates [m]
MERGE_DIST = 0.5 # adjacent nodes closer than this merge (fillets meeting on a short segment) [m]
MIN_NODE_TIME = 0.5 # fillets shorter than this at v_corner become one pass-through node, and no segment is flown
                    # faster than this: a very short segment next to long ones distorts min-snap [s]

WGS84_A = 6378137.0 # semi-major axis [m]
WGS84_E2 = 6.69437999014e-3 # first eccentricity squared

def gps_to_enu(lat, lon, alt, origin=None):
    # Flat-earth ENU around origin (lat0, lon0, alt0) [deg, deg, m], default the first point. Good to ~cm over a few km
    lat, lon, alt = (np.asarray(x, dtype=np.float64) for x in (lat, lon, alt))
    lat0, lon0, alt0 = origin if origin is not None else (lat[0], lon[0], alt[0])
    phi0 = math.radians(lat0)
    w = 1.0 - WGS84_E2 * math.sin(phi0) ** 2
    r_n = WGS84_A / math.sqrt(w) # prime vertical radius
    r_m = WGS84_A * (1.0 - WGS84_E2) / w**1.5 # meridian radius
    east = np.radians(lon - lon0) * (r_n + alt0) * math.cos(phi0)
    north = np.radians(lat - lat0) * (r_m + alt0)
    return np.stack([east, north, alt - alt0], axis=-1)

class Route:
    # Built once from ENU waypoints, read-only after. Each interior corner becomes either two nodes at the tangent points
    # of a circular arc (velocity v_corner along the adjacent segments) or, for near-reversals, one stop node (v = 0).
    # Route start and end are rest nodes. corners / corner_* keep the per-corner data for the deviation checks
    def __init__(self, waypoints, v_des, cfg: RouteConfig = RouteConfig()):
        self.cfg = cfg
        self.v_des = float(v_des)
        self.waypoints = np.atleast_2d(np.asarray(waypoints, dtype=np.float64)) # (N, 3) raw input, kept for plotting
        if self.waypoints.shape[-1] != 3 or len(self.waypoints) == 0:
            raise ValueError(f"waypoints must be (N, 3), got {self.waypoints.shape}")
        self._build()

    # ====== CONSTRUCTORS ======
    @classmethod
    def from_gps(cls, lat, lon, alt, v_des, cfg: RouteConfig = RouteConfig(), origin=None):
        return cls(gps_to_enu(lat, lon, alt, origin), v_des, cfg)

    @classmethod
    def hover(cls, point, cfg: RouteConfig = RouteConfig()):
        return cls(np.asarray(point, dtype=np.float64)[None], 0.0, cfg)

    # ====== PIPELINE ======
    def _build(self):
        self.points = self._clean(self.waypoints) # (M, 3)
        if len(self.points) > 1 and self.v_des <= 0.0:
            raise ValueError("v_des must be > 0 for a route with more than one point")
        L, u, theta = self.segment_geometry(self.points)
        self.corners = self.points[1:-1] # (C, 3)
        self.corner_theta = theta # (C, )
        self.corner_radius, self.corner_speed, self.corner_is_stop, d = self._corner_fillet(theta, L[:-1], L[1:])
        pos, vel, corner_idx = self._place_nodes(u, d)
        self.node_pos, self.node_vel, self.corner_idx = self._merge_close(pos, vel, corner_idx) # (K, 3), (K, 3), (C, 2)

    def _clean(self, P):
        # Clamp altitude and drop near-duplicates. Small turns are kept: they become pass-through nodes (_place_nodes)
        P = P.copy()
        P[:, 2] = np.maximum(P[:, 2], self.cfg.min_altitude)
        keep = [P[0]]
        for p in P[1:]:
            if np.linalg.norm(p - keep[-1]) >= MIN_POINT_DIST:
                keep.append(p)
        return np.array(keep)

    def _corner_fillet(self, theta, L_in, L_out):
        # Arc of radius R in the plane of the two segments: tangent distance d = R tan(θ/2), deviation from the corner
        # e = R (sec(θ/2) - 1), lateral acceleration v²/R. R_max from fit and corridor, v_corner from a_lat at R_max,
        # then the smallest R that speed allows (least rounding), floored at MIN_RADIUS
        a_lat = A_LAT_FRAC * self.cfg.a_max
        tan_h = np.tan(theta / 2.0)
        sec_m1 = 1.0 / np.cos(theta / 2.0) - 1.0
        is_stop = theta > THETA_STOP

        r_max = np.minimum(FILLET_FIT * np.minimum(L_in, L_out) / np.maximum(tan_h, EPS), self.cfg.corridor / np.maximum(sec_m1, EPS))
        v_corner = np.minimum(self.v_des, np.sqrt(a_lat * r_max))
        radius = np.minimum(np.maximum(v_corner**2 / a_lat, MIN_RADIUS), r_max)
        d = radius * tan_h

        radius, v_corner, d = (np.where(is_stop, 0.0, x) for x in (radius, v_corner, d))
        return radius, v_corner, is_stop, d

    def _place_nodes(self, u, d):
        # Start node, per corner entry/exit (or one stop / pass-through node), end node. corner_idx holds each corner's
        # node indices. A corner whose fillet is shorter than MERGE_DIST, or than MIN_NODE_TIME at v_corner (small turns,
        # fast shallow corners), keeps its own point as one node, passed at v_corner along the bisector of the two segments
        P = self.points
        pos, vel, corner_idx = [P[0]], [np.zeros(3)], []
        for c in range(len(self.corners)):
            p, k = P[c + 1], len(pos)
            if self.corner_is_stop[c]:
                pos.append(p)
                vel.append(np.zeros(3))
                corner_idx.append((k, k))
            elif 2.0 * d[c] < max(MERGE_DIST, self.corner_speed[c] * MIN_NODE_TIME):
                bisector = u[c] + u[c + 1] # nonzero: θ <= THETA_STOP < π here
                pos.append(p)
                vel.append(self.corner_speed[c] * bisector / np.linalg.norm(bisector))
                corner_idx.append((k, k))
            else:
                v = self.corner_speed[c]
                pos += [p - d[c] * u[c], p + d[c] * u[c + 1]]
                vel += [v * u[c], v * u[c + 1]]
                corner_idx.append((k, k + 1))
        if len(P) > 1:
            pos.append(P[-1])
            vel.append(np.zeros(3))
        return np.array(pos), np.array(vel), np.array(corner_idx, dtype=np.int64).reshape(-1, 2)

    def _merge_close(self, pos, vel, corner_idx):
        # Adjacent nodes on the same segment closer than MERGE_DIST become one node (fillets of two corners meeting on a
        # short U-turn leg; a single corner's short fillet is already one node). Both velocities point along that
        # segment, so keep the slower one. A merge with a route endpoint keeps the endpoint's position and rest state,
        # otherwise the midpoint
        last = len(pos) - 1
        new_pos, new_vel, is_end, remap = [], [], [], np.empty(len(pos), dtype=np.int64)
        for k in range(len(pos)):
            end_k = k == 0 or k == last
            if new_pos and np.linalg.norm(pos[k] - new_pos[-1]) < MERGE_DIST:
                if end_k or is_end[-1]:
                    new_pos[-1] = pos[k] if end_k else new_pos[-1]
                    new_vel[-1] = np.zeros(3)
                else:
                    new_pos[-1] = 0.5 * (new_pos[-1] + pos[k])
                    new_vel[-1] = min(new_vel[-1], vel[k], key=np.linalg.norm)
                is_end[-1] = is_end[-1] or end_k
            else:
                new_pos.append(pos[k])
                new_vel.append(vel[k])
                is_end.append(end_k)
            remap[k] = len(new_pos) - 1
        return np.array(new_pos), np.array(new_vel), remap[corner_idx]

    # ====== GEOMETRY ======
    @staticmethod
    def segment_geometry(P):
        # Segment lengths (M-1, ), unit directions (M-1, 3), turn angles at the interior points (M-2, )
        diff = np.diff(P, axis=0)
        L = np.linalg.norm(diff, axis=-1)
        u = diff / np.maximum(L, EPS)[:, None]
        theta = np.arccos(np.clip(np.sum(u[:-1] * u[1:], axis=-1), -1.0, 1.0))
        return L, u, theta

    # ====== QUERIES ======
    @property
    def num_nodes(self):
        return len(self.node_pos)

    @property
    def num_segments(self):
        return self.num_nodes - 1

    @property
    def node_speed(self):
        return np.linalg.norm(self.node_vel, axis=-1)

    @property
    def is_hover(self):
        return self.num_nodes == 1

    @property
    def length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=-1)))

    def corner_deviation_bound(self):
        # Deviation of the ideal arc from each corner (0 for stop nodes). The solved trajectory is checked against this
        return self.corner_radius * (1.0 / np.cos(self.corner_theta / 2.0) - 1.0)

    def deviation(self, p):
        # Distance of points (T, 3) to the cleaned polyline (T, )
        p = np.atleast_2d(p)
        if len(self.points) == 1:
            return np.linalg.norm(p - self.points[0], axis=-1)
        a, ab = self.points[:-1], np.diff(self.points, axis=0) # (S, 3)
        t = np.clip(np.einsum("tsi,si->ts", p[:, None] - a, ab) / np.maximum(np.sum(ab * ab, axis=-1), EPS), 0.0, 1.0)
        return np.min(np.linalg.norm(p[:, None] - (a + t[..., None] * ab), axis=-1), axis=-1)

    def __repr__(self):
        return f"Route(points={len(self.points)}, nodes={self.num_nodes}, corners={len(self.corners)}, length={self.length:.1f} m, v_des={self.v_des})"


class RouteSampler:
    def __init__(self, cfg: RouteSamplerConfig):
        self.cfg = cfg
        self.keywords = cfg.route_keywords

    def sample(self, seed, keyword):
        rng = np.random.default_rng(seed)

        n_points = rng.integers(self.cfg.num_points_min, self.cfg.num_points_max + 1)
        points = None
        match keyword:
            case "random":
               points = self._random_polyline(rng, n_points)
            case "survey":
                points = self._survey_lines(rng, n_points)
            case "loop":
                points = self._loop(rng, n_points)
            case "zigzag":
                points = self._zigzag(rng, n_points)
            case "stop":
                points = self._stop(rng)
            case _:
                raise ValueError(f"unknown route keyword '{keyword}', expected one of {self.keywords}")

        v_des = rng.uniform(self.cfg.v_des_min, self.cfg.v_des_max)
        return points, v_des

    def _random_polyline(self, rng, n_points):
        z = rng.uniform(self.cfg.alt_min, self.cfg.alt_max)
        pts = [np.array([0.0, 0.0, z])]
        yaw = rng.uniform(-np.pi, np.pi)

        for _ in range(n_points - 1):
            if rng.random() > self.cfg.p_straight:
                yaw += rng.uniform(-self.cfg.turn_max, self.cfg.turn_max)
            seg_l = rng.uniform(self.cfg.seg_length_min, self.cfg.seg_length_max)
            
            if rng.random() < self.cfg.p_climb:
                up_bound = self.cfg.alt_max - z
                down_bound = self.cfg.alt_min - z
                z += rng.uniform(down_bound, up_bound)

            xy = pts[-1][:2] + seg_l * np.array([np.cos(yaw), np.sin(yaw)])
            pts.append(np.array([xy[0], xy[1], z]))
        return np.array(pts)

    def _survey_lines(self, rng, n_points):
        cfg = self.cfg
        n_lines = max(1, int(n_points) // 2)
        z = rng.uniform(cfg.alt_min, cfg.alt_max)
        yaw = rng.uniform(-np.pi, np.pi)
        leg = rng.uniform(cfg.seg_length_min, cfg.seg_length_max)
        spacing = rng.uniform(cfg.seg_length_min, leg)

        along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        across = rng.choice((-1.0, 1.0)) * left 

        pts, offset = [], 0.0
        for i in range(n_lines):
            start = rng.uniform(-0.15, 0.15) * leg
            length = leg * rng.uniform(0.8, 1.2)
            dpsi = np.radians(rng.uniform(-5.0, 5.0))
            direction = np.cos(dpsi) * along + np.sin(dpsi) * left

            a = offset * across + start * along
            b = a + length * direction
            pts += [a, b] if i % 2 == 0 else [b, a] # back and forth
            offset += spacing * rng.uniform(0.9, 1.1)

        pts = np.array(pts) - pts[0] # start at the origin, like the other shapes
        pts[:, 2] = z
        return pts

    def _loop(self, rng, n_points):
        # One closed lap around an ellipse, ending where it started. Sized so the mean chord is a segment length.
        # Per route: aspect, orientation, direction (CW / CCW); per point: angle and radius jitter, so the lap is a
        # lumpy polygon with uneven turns rather than a perfect circle. Constant altitude
        n = max(3, int(n_points) - 1) # vertices; the last point closes the lap
        z = rng.uniform(self.cfg.alt_min, self.cfg.alt_max)
        seg = rng.uniform(self.cfg.seg_length_min, self.cfg.seg_length_max)
        r_a = seg / (2.0 * np.sin(np.pi / n)) # circle whose n-gon has side seg
        r_b = r_a * rng.uniform(0.5, 1.0)
        yaw = rng.uniform(-np.pi, np.pi)
        direction = rng.choice((-1.0, 1.0))

        phi = direction * (np.arange(n) + rng.uniform(-0.25, 0.25, n)) * 2.0 * np.pi / n
        r = rng.uniform(0.85, 1.15, n)
        x, y = r * r_a * np.cos(phi), r * r_b * np.sin(phi)
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.stack([c * x - s * y, s * x + c * y, np.full(n, z)], axis=-1)
        pts = np.concatenate([pts, pts[:1]]) # close the lap
        pts[:, :2] -= pts[0, :2] # start at the origin, like the other shapes
        return pts

    def _zigzag(self, rng, n_points):
        # Legs alternating left and right of a base heading by a zig angle (turns of 2x that angle). Per route: base
        # heading, leg length, zig angle; per leg: length, angle and a slow drift of the base heading, so the zigzag
        # can bend. Constant altitude
        cfg = self.cfg
        z = rng.uniform(cfg.alt_min, cfg.alt_max)
        base = rng.uniform(-np.pi, np.pi)
        leg = rng.uniform(cfg.seg_length_min, cfg.seg_length_max)
        zig = np.radians(rng.uniform(20.0, 70.0)) # turns of 40-140°
        side = rng.choice((-1.0, 1.0))

        pts = [np.array([0.0, 0.0, z])]
        for i in range(int(n_points) - 1):
            base += np.radians(rng.uniform(-5.0, 5.0))
            yaw = base + side * (-1.0) ** i * zig * rng.uniform(0.85, 1.15)
            xy = pts[-1][:2] + leg * rng.uniform(0.8, 1.2) * np.array([np.cos(yaw), np.sin(yaw)])
            pts.append(np.array([xy[0], xy[1], z]))
        return np.array(pts)

    def _stop(self, rng):
        # Short move that ends at rest, so most of the episode is the final stop and holding the end point (the
        # trajectory holds its end state after the route). Longer routes rarely reach their end within an episode.
        # One of: hover in place, a straight dash, or a dog-leg with one turn; each leg may step in altitude
        cfg = self.cfg
        z = rng.uniform(cfg.alt_min, cfg.alt_max)
        yaw = rng.uniform(-np.pi, np.pi)
        pts = [np.array([0.0, 0.0, z])]
        for i in range(rng.choice((0, 1, 2), p=(0.2, 0.5, 0.3))): # hover / dash / dog-leg
            if i > 0:
                yaw += rng.choice((-1.0, 1.0)) * rng.uniform(np.radians(30.0), cfg.turn_max)
            if rng.random() < cfg.p_climb:
                z = rng.uniform(cfg.alt_min, cfg.alt_max)
            xy = pts[-1][:2] + rng.uniform(cfg.seg_length_min, cfg.seg_length_max) * np.array([np.cos(yaw), np.sin(yaw)])
            pts.append(np.array([xy[0], xy[1], z]))
        return np.array(pts)