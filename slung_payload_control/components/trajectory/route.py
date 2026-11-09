import math
import numpy as np

# Design constants
EPS = 1e-9
MIN_SEG_LENGTH = 0.5
THETA_STRAIGHT = math.radians(2.0) # below: pass straight through the waypoint
THETA_STOP = math.radians(170.0) # above: full stop at the waypoint

class Route:
    def __init__(self, waypoints, v_des, corridor, a_max, max_segments=None, meta=None):
        self.v_des = v_des
        self.a_max = a_max
        self.corridor = corridor
        self.max_segments = max_segments
        self.meta = meta
        self.wpts = np.atleast_2d(np.asarray(waypoints, dtype=np.float64))

        if self.wpts.shape[-1] != 3 or len(self.wpts) == 0:
            raise ValueError(f"waypoints must be (N, 3), got {self.wpts.shape}")

        self._build() # build self.node_pos, self.node_vel, self.node_rest, self.seg_T

    def _build(self):
        self.points = self._clean(self.wpts)
        if len(self.points) < 2: raise ValueError(f"Number of points is too small - is {len(self.points)}")
        
        L, u, theta = self.segment_geometry(self.points)
        self.corners = self.points[1:-1] # none-leaf nodes
        self.corner_theta = theta
        self.corner_radius, self.corner_speed, self.corner_is_stop, d = self._corner_smooth(theta, L[:-1], L[1:])
        self.node_pos, dirs, vcap, self.corner_idx = self._place_nodes(u, d)
        self.node_rest = vcap == 0.0 # if node vcap == 0, node_rest=true

        seg_len, seg_arc = self._segments(self.node_pos)
        v = self._speed_profile(vcap, seg_len, seg_arc)
        self.node_vel = v[:, None] * dirs
        self.seg_T = self._durations(v, seg_len, seg_arc)

        if self.max_segments and len(self.seg_T) > self.max_segments:
            raise ValueError(f"Route has {len(self.seg_T)} segments, max is {self.max_segments}")
        
        if not (np.isfinite(self.seg_T).all() and (self.seg_T > 0).all()):
            raise ValueError(f"Invalid segment durations: {self.seg_T}")

    def _clean(self, P):
        P = P.copy()
        keep = [P[0]]

        # TODO: Better handling than just discarding - May remove wanted movement
        for p in P[1:]:
            if np.linalg.norm(p - keep[-1]) >= MIN_SEG_LENGTH:
                keep.append(p)
        return np.array(keep)

    def _corner_smooth(self, theta, Lin, Lout):
        amax = self.a_max
        tan_h = np.tan(theta / 2.0) # tan(theta / 2.0)
        sec_m1 = 1.0 / np.cos(theta / 2.0) - 1.0 # sec(theta / 2.0) - 1.0 = 1.0/cos(theta / 2.0) - 1.0


        # Constraint the maximum allowable turning radius
        # 1) The distance (d) to start turning before the corner should not be further away than
        #    0.45 times the smallest segment length among seg_in and seg_out.
        #    From: d = r * tan(theta / 2) -> r = d / tan(theta / 2)
        # 2) The arc should not be further away from the corner-waypoint than "corridor"
        #    From: sec(zero) = 1 and rising with angle. Therefore the minus 1.0 to get actual deviation from arc
        r_max = np.minimum(
            0.45 * np.minimum(Lin, Lout) / np.maximum(tan_h, EPS),
            self.corridor / np.maximum(sec_m1, EPS)
        )

        # Centripetal acceleration = v² / r -> v = sqrt(a * r). Up to v_des only
        # Fastest speed the biggest allowed arc can do is v_corner.
        v_corner = np.minimum(self.v_des, np.sqrt(amax * r_max))

        # If v_des bounds the speed, the r_max is not needed
        radius = v_corner**2 / amax

        # Back to d = r*tan(theta/2)
        d = radius * tan_h

        # If the turning angle is too great, the corner becomes a full-stop
        is_stop = theta > THETA_STOP
        radius, v_corner, d = (np.where(is_stop, 0.0, x) for x in (radius, v_corner, d))

        return radius, v_corner, is_stop, d

    def _place_nodes(self, u, d):
        # Returns node positions, unit velocity directions (zero at rest nodes) and speed caps.
        # corner_idx holds (entry, exit) node indices per corner; entry == exit means no arc.
        P = self.points
        pos = [P[0]]
        dirs = [np.zeros(3)]
        vcap = [0.0]
        corner_idx = []
        for c in range(len(self.corners)):
            p = P[c + 1]
            k = len(pos)
            if self.corner_is_stop[c]: # Full-stop
                pos.append(p)
                dirs.append(np.zeros(3))
                vcap.append(0.0)
                corner_idx.append((k, k))
            elif self.corner_theta[c] < THETA_STRAIGHT: # Near-straight -> pass through along the bisector
                bisector = u[c] + u[c+1]
                pos.append(p)
                dirs.append(bisector / np.linalg.norm(bisector))
                vcap.append(self.corner_speed[c])
                corner_idx.append((k, k))
            else: # Default: add two points d away from corner point along segment
                pos.append(p - d[c] * u[c])
                pos.append(p + d[c] * u[c + 1])
                dirs.append(u[c])
                dirs.append(u[c + 1])
                vcap.append(self.corner_speed[c])
                vcap.append(self.corner_speed[c])
                corner_idx.append((k, k+1))

        pos.append(P[-1])
        dirs.append(np.zeros(3))
        vcap.append(0.0)
        return np.array(pos), np.array(dirs), np.array(vcap), np.array(corner_idx, dtype=np.int64).reshape(-1, 2)

    def _segments(self, pos):
        # Path length per node-to-node segment; segments inside a rounded corner use the arc length r*theta
        seg_len = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        seg_arc = np.zeros(len(seg_len), dtype=bool)
        for c, (k0, k1) in enumerate(self.corner_idx):
            if k1 > k0:
                seg_arc[k0] = True
                seg_len[k0] = self.corner_radius[c] * self.corner_theta[c]
        return seg_len, seg_arc

    def _speed_profile(self, vcap, seg_len, seg_arc):
        a = self.a_max
        v = vcap.copy()
        for i in range(len(seg_len)): # forward: limit how fast speed can rise
            if seg_arc[i]:
                v[i + 1] = min(v[i], v[i + 1]) # constant speed on the arc
            else:
                v[i + 1] = min(v[i + 1], math.sqrt(v[i]**2 + 2.0 * a * seg_len[i]))

        for i in reversed(range(len(seg_len))): # backward: limit how fast speed can fall
            if seg_arc[i]:
                v[i] = min(v[i], v[i + 1])
            else:
                v[i] = min(v[i], math.sqrt(v[i + 1]**2 + 2.0 * a * seg_len[i]))

        return v

    def _durations(self, v, seg_len, seg_arc):
        a = self.a_max
        vd = self.v_des
        T = np.empty(len(seg_len))
        for i, s in enumerate(seg_len):
            v0 = v[i]
            v1 = v[i + 1]
            if seg_arc[i]:
                T[i] = s / max(v0, EPS)
                continue
            vp = min(vd, math.sqrt(a * s + 0.5 * (v0**2 + v1**2))) # peak speed reachable
            s_cruise = s - (2 * vp**2 - v0**2 - v1**2) / (2 * a)
            T[i] = (2 * vp - v0 - v1) / a + max(s_cruise, 0.0) / vp
        return T

    @staticmethod
    def segment_geometry(P):
        diff = np.diff(P, axis=0)
        L = np.linalg.norm(diff, axis=-1)
        u = diff / np.maximum(L, EPS)[:, None]
        theta = np.arccos(np.clip(np.sum(u[:-1] * u[1:], axis=-1), -1.0, 1.0))
        return L, u, theta # segment lengths, segment unit vectors, angles between segments