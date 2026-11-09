import numpy as np
from config.config_components import TrajectoryConfig
from components.trajectory.route import Route

class RouteSampler:
    def __init__(self, cfg: TrajectoryConfig, max_tries=20):
        self.cfg = cfg
        self.max_tries = max_tries
        self._gen = {}
        for kw in cfg.keywords:
            fn = getattr(self, f"_gen_{kw}", None)
            if not callable(fn):
                raise NotImplementedError(f"RouteSampler has no generator '_gen_{kw}' for keyword '{kw}'")
            self._gen[kw] = fn

        if len(cfg.keyword_weights) != len(cfg.keywords):
            raise ValueError("keyword_weights must match keywords in size")

    def sample(self, seed, keyword=None):
        rng = np.random.default_rng(seed)

        keyword = keyword or rng.choice(self.cfg.keywords, p=self.cfg.keyword_weights)

        for _ in range(self.max_tries):
            n_pts = rng.integers(self.cfg.route_points_min, self.cfg.route_points_max, endpoint=True)
            wpts = self._gen[keyword](rng, n_pts)
            wpts = self._place(wpts, rng)    
            v_des = rng.uniform(self.cfg.route_vel_des_min, self.cfg.route_vel_des_max)
            try:
                meta = {"keyword": keyword}
                return Route(wpts, v_des, self.cfg.a_max, self.cfg.corridor, self.cfg.traj_max_segments, meta)
            except ValueError:
                continue

        raise RuntimeError(f"no valid '{keyword}' route after {self.max_tries} tries (seed={seed})")

    def _place(self, wpts, rng):
        cfg = self.cfg
        yaw = rng.uniform(-np.pi, np.pi)
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P = wpts @ R.T
        P[:, :2] -= P[0, :2] # center at first

        band = cfg.route_alt_max - cfg.route_alt_min
        z = P[:, 2] - P[:, 2].min()
        if z.max() > band:
            z *= band / z.max() # compress instead of clipping to avoid flat kinks
        P[:, 2] = cfg.route_alt_min + z + rng.uniform(0.0, band - z.max())
        return P

    def _gen_random(self, rng, n_pts):
        cfg = self.cfg
        n_seg = n_pts - 1
        step = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max, size=n_seg)
        dpsi = rng.uniform(-np.radians(150.0), np.radians(150.0), size=n_seg)
        dpsi[0] = 0.0
        psi = np.cumsum(dpsi)
        gamma = rng.uniform(-cfg.route_max_climb_ang, cfg.route_max_climb_ang, size=n_seg)

        d = np.stack([
            np.cos(psi) * np.cos(gamma),
            np.sin(psi) * np.cos(gamma),
            np.sin(gamma)], axis=-1) * step[:, None]

        return np.vstack([np.zeros(3), np.cumsum(d, axis=0)])

    def _gen_survey(self, rng, n_pts):
        cfg = self.cfg
        n_lanes = max(2, int(n_pts) // 2)
        leg = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        spacing = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        side = rng.choice([-1.0, 1.0])

        pts = []
        offset = 0.0
        for i in range(n_lanes):
            start = rng.uniform(-0.15, 0.15) * leg
            length = leg * rng.uniform(0.8, 1.2)
            dpsi = np.radians(rng.uniform(-5.0, 5.0))

            a = np.array([start, side * offset, 0.0])
            b = a + length * np.array([np.cos(dpsi), np.sin(dpsi), 0.0])
            pts += [a, b] if i % 2 == 0 else [b, a] # back and forth
            offset += spacing * rng.uniform(0.9, 1.1)

        return np.array(pts)

    def _gen_loop(self, rng, n_pts):
        cfg = self.cfg
        n = max(3, int(n_pts) - 1)
        seg = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        r_a = seg / (2.0 * np.sin(np.pi / n))
        r_b = r_a * rng.uniform(0.5, 1.0)
        direction = rng.choice([-1.0, 1.0])

        psi = direction * (np.arange(n) + rng.uniform(-0.25, 0.25, n)) * 2.0 * np.pi / n
        r = rng.uniform(0.85, 1.15, size=n)
        x = r * r_a * np.cos(psi)
        y = r * r_b * np.sin(psi)
        pts = np.stack([x, y, np.zeros_like(x)], axis=-1)
        pts = np.vstack([pts, pts[:1]]) # start = end
        return pts
        
    def _gen_zigzag(self, rng, n_pts):
        cfg = self.cfg
        n_seg = n_pts - 1
        leg = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        zig = np.radians(rng.uniform(20.0, 70.0))
        side = rng.choice([-1.0, 1.0])

        dir_drift = 0.0
        pts = [np.array([0.0, 0.0, 0.0])]
        for i in range(n_seg):
            dir_drift += np.radians(rng.uniform(-5.0, 5.0))
            psi = dir_drift + side * (-1.0)**i * zig * rng.uniform(0.85, 1.15)
            xy = pts[-1][:2] + leg * rng.uniform(0.8, 1.2) * np.array([np.cos(psi), np.sin(psi)])
            pts.append(np.array([xy[0], xy[1], 0.0]))
        return np.array(pts)

    def _gen_stop(self, rng, n_pts):
        cfg = self.cfg
        type_n = rng.choice((0, 1, 2), p=(0.25, 0.5, 0.25)) # hover / dash / dog-leg

        if type_n == 0: # nudge: small reposition then hold
            return np.array([
                [0.0, 0.0, 0.0],
                [rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_min * 1.5), 0.0, 0.0]
            ])

        dash = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        if type_n == 1: # dash: accelerate and brake to rest
            return np.array([
                [0.0, 0.0, 0.0],
                [dash, 0.0, 0.0]
            ])


        turn = rng.choice([-1.0, 1.0]) * np.radians(rng.uniform(30.0, 120.0))
        leg = rng.uniform(cfg.route_seg_l_min, cfg.route_seg_l_max)
        p1 = np.array([dash, 0.0, 0.0])
        p2 = p1 + leg * np.array([np.cos(turn), np.sin(turn), 0.0])
        return np.array([
            [0.0, 0.0, 0.0],
            p1,
            p2
        ])