import dataclasses
import hashlib
import inspect
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from config.config_env import ReferenceConfig, RouteConfig, RouteSamplerConfig, TrajectoryBankConfig
from config.paths import LOG_DIR
from . import route as route_module, trajectory as trajectory_module
from .math_utils import wrap_angle
from .route import Route, RouteSampler
from .trajectory import FACT, N_COEF, MinSnapSolver

# Trajectory bank and the per-env payload reference built on it. The bank holds many solved routes as padded torch
# tensors (built once, cached to disk); ReferenceTrajectory places one bank entry per env in the world (yaw rotation,
# optional mirror, translation) and evaluates it at the env's reference clock τ plus time offsets (current point and
# preview), batched over all envs with the same normalized-time Horner scheme as trajectory.Trajectory

BANK_DIR = os.path.join(LOG_DIR, "route_bank")
BANK_FORMAT = 1 # bump when the stored fields change: part of the cache key
HEADING_MIN_SPEED = 0.05 # heading0 is the reference velocity direction once its horizontal speed exceeds this [m/s]
YAW_HYSTERESIS = 0.7 # the yaw follower engages above yaw_min_speed and releases below this fraction of it (no flicker)

# ====== BUILD (worker side) ======
def _solve_entry(job):
    # (seed, keyword) -> one solved route as plain arrays. Runs in a worker process
    seed, keyword, route_cfg, sampler_cfg = job
    points, v_des = RouteSampler(sampler_cfg).sample(seed, keyword)
    route = Route(points, v_des, route_cfg)
    traj = MinSnapSolver(route_cfg).solve(route)
    # Initial heading: direction the reference starts moving in (on a short first leg it already curves into the next
    # turn, so the first leg's direction can be far off). Hover and purely vertical routes: 0
    heading0 = 0.0
    if not route.is_hover:
        v_h = traj.sample(0.01, max_order=1)[1][1][:, :2]
        moving = np.linalg.norm(v_h, axis=-1) > HEADING_MIN_SPEED
        if moving.any():
            heading0 = float(np.arctan2(*v_h[np.argmax(moving)][::-1]))
    return {"coeffs": traj.coeffs, "durations": traj.durations, "duration": 0.0 if route.is_hover else traj.duration,
            "end": route.points[-1], "heading0": heading0, "v_des": v_des, "seed": seed}

class TrajectoryBank:
    # Solved routes as padded tensors: coeffs (B, S, 3, 8) in normalized segment time, durations (B, S), t_nodes
    # (B, S+1), duration (B, ). Padding segments past a route's end are constant at its end point (duration 1 s), so
    # evaluating at t >= duration lands on the held end state. Routes start at the origin (x = y = 0, absolute z).
    # seed and keyword_id reproduce any entry: Route(*RouteSampler(cfg).sample(seed, keywords[keyword_id]), route_cfg)
    TENSORS = ("coeffs", "durations", "t_nodes", "duration", "heading0", "v_des", "keyword_id", "seed", "num_segments")

    def __init__(self, data, keywords, device):
        self.keywords = tuple(keywords)
        for name in self.TENSORS:
            setattr(self, name, data[name].to(device))

    @property
    def size(self):
        return len(self.duration)

    @property
    def max_segments(self):
        return self.durations.shape[1]

    # ====== CACHE ======
    @staticmethod
    def cache_key(bank_cfg, route_cfg, sampler_cfg):
        # Configs plus the generator source (its shaping constants live in code, a change there must rebuild the bank)
        # and the bank's own format
        cfgs = (dataclasses.replace(bank_cfg, num_workers=0), route_cfg, sampler_cfg, BANK_FORMAT, HEADING_MIN_SPEED)
        src = "".join(inspect.getsource(m) for m in (route_module, trajectory_module))
        return hashlib.sha1((repr(cfgs) + src).encode()).hexdigest()[:12]

    @classmethod
    def cache_path(cls, bank_cfg=TrajectoryBankConfig(), route_cfg=RouteConfig(), sampler_cfg=RouteSamplerConfig()):
        return os.path.join(BANK_DIR, f"bank_{cls.cache_key(bank_cfg, route_cfg, sampler_cfg)}.pt")

    @classmethod
    def load(cls, bank_cfg=TrajectoryBankConfig(), route_cfg=RouteConfig(), sampler_cfg=RouteSamplerConfig(),
             device="cpu", verbose=True):
        # The bank for these configs, from the cache. Built with utils/build_bank.py, never implicitly
        path = cls.cache_path(bank_cfg, route_cfg, sampler_cfg)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No trajectory bank for the current configs ({path}). Build it with: "
                                    f"python utils/build_bank.py")
        data = torch.load(path, map_location="cpu")
        if verbose:
            print(f"TrajectoryBank: loaded {path}")
        return cls(data, data["keywords"], device)

    @classmethod
    def build_and_save(cls, bank_cfg=TrajectoryBankConfig(), route_cfg=RouteConfig(), sampler_cfg=RouteSamplerConfig(),
                       verbose=True):
        # Build the bank for these configs and write it to its cache path (overwriting). Returns the path
        data = cls.build(bank_cfg, route_cfg, sampler_cfg, verbose)
        data["configs"] = {"bank": repr(bank_cfg), "route": repr(route_cfg), "sampler": repr(sampler_cfg)}
        path = cls.cache_path(bank_cfg, route_cfg, sampler_cfg)
        os.makedirs(BANK_DIR, exist_ok=True)
        torch.save(data, path)
        if verbose:
            print(f"TrajectoryBank: saved {path}")
        return path

    # ====== BUILD ======
    @staticmethod
    def build(bank_cfg, route_cfg, sampler_cfg, verbose=True):
        keywords = sampler_cfg.route_keywords
        if len(bank_cfg.keyword_weights) != len(keywords):
            raise ValueError(f"keyword_weights {bank_cfg.keyword_weights} must match route_keywords {keywords}")
        weights = np.asarray(bank_cfg.keyword_weights, dtype=np.float64)
        rng = np.random.default_rng(bank_cfg.seed)
        B, S = bank_cfg.size, bank_cfg.max_segments
        t_start = time.perf_counter()

        # Solve in rounds: routes longer than S segments are redrawn (new seed, same keyword) until B fit. Jobs are
        # accepted in submission order, so the bank depends only on the configs and the seed
        entries, kw_ids, n_redrawn = [], [], 0
        pending = list(rng.choice(len(keywords), size=B, p=weights / weights.sum()))
        # spawn: no fork after CUDA init in the parent. Needs the caller to run from a file (workers re-import __main__);
        # the executor raises BrokenProcessPool if workers die instead of respawning them forever like Pool does
        with ProcessPoolExecutor(bank_cfg.num_workers, mp_context=mp.get_context("spawn")) as pool:
            while pending:
                seeds = rng.integers(0, 2**31 - 1, size=len(pending))
                jobs = [(int(s), keywords[k], route_cfg, sampler_cfg) for s, k in zip(seeds, pending)]
                retry = []
                for k, entry in zip(pending, pool.map(_solve_entry, jobs, chunksize=16)):
                    if len(entry["durations"]) <= S:
                        entries.append(entry)
                        kw_ids.append(k)
                    else:
                        retry.append(k)
                n_redrawn += len(retry)
                pending = retry

        coeffs = np.zeros((B, S, 3, N_COEF))
        durations = np.ones((B, S))
        for b, e in enumerate(entries):
            n = len(e["durations"])
            coeffs[b, :n], durations[b, :n] = e["coeffs"], e["durations"]
            coeffs[b, n:, :, 0] = e["end"] # hold segments: constant at the end point
        t_nodes = np.concatenate([np.zeros((B, 1)), np.cumsum(durations, axis=1)], axis=1) # float64 sum, cast after

        f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        data = {
            "coeffs": f32(coeffs), "durations": f32(durations), "t_nodes": f32(t_nodes),
            "duration": f32([e["duration"] for e in entries]), "heading0": f32([e["heading0"] for e in entries]),
            "v_des": f32([e["v_des"] for e in entries]), "keyword_id": torch.as_tensor(kw_ids, dtype=torch.long),
            "seed": torch.as_tensor([e["seed"] for e in entries], dtype=torch.long),
            "num_segments": torch.as_tensor([len(e["durations"]) for e in entries], dtype=torch.long),
            "keywords": tuple(keywords),
        }
        if verbose:
            n_seg = data["num_segments"].float()
            print(f"TrajectoryBank: built {B} routes in {time.perf_counter() - t_start:.0f} s ({n_redrawn} redrawn for > {S} "
                  f"segments), segments median {n_seg.median():.0f} max {n_seg.max():.0f}, duration median "
                  f"{data['duration'].median():.1f} s")
        return data

class ReferenceTrajectory:
    # Per-env payload reference: a bank entry placed in the world by a yaw rotation, an optional mirror (route y -> -y)
    # and a horizontal translation, and a reference clock τ advanced by α·dt per step (α ≡ 1 in phase 1, the hook for
    # time-scaling in phase 2). Past a route's end the reference holds its end point with zero derivatives.
    # Yaw reference psi_ref: not part of the position solve (yaw doesn't affect position feasibility). A first-order
    # follower turns it towards the horizontal reference velocity direction, rate-limited to yaw_rate_max, and holds it
    # below yaw_min_speed (stops, hover, route end). psi_ref is continuous; psi_dot_ref may step, which a yaw-rate
    # command can follow
    def __init__(self, bank: TrajectoryBank, num_envs, device, cfg: ReferenceConfig = ReferenceConfig()):
        if cfg.yaw_mode not in ("velocity", "fixed"):
            raise ValueError(f"yaw_mode must be 'velocity' or 'fixed', got '{cfg.yaw_mode}'")
        self.bank, self.num_envs, self.device, self.cfg = bank, num_envs, torch.device(device), cfg
        f32 = dict(device=self.device, dtype=torch.float32)
        self.idx = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.origin = torch.zeros((num_envs, 2), **f32) # world xy of the route start
        self.cos_yaw, self.sin_yaw = torch.ones(num_envs, **f32), torch.zeros(num_envs, **f32)
        self.mirror = torch.ones(num_envs, **f32) # +1 or -1 on route y
        self.tau = torch.zeros(num_envs, **f32)
        self.alpha = torch.ones(num_envs, **f32)
        # Per-env copies of the small per-segment tables, refreshed at assign: eval then gathers only coefficients
        self.t_nodes = torch.zeros((num_envs, bank.max_segments + 1), **f32)
        self.durations = torch.ones((num_envs, bank.max_segments), **f32)
        self.duration = torch.zeros(num_envs, **f32)
        self.fact = torch.as_tensor(FACT, **f32) # (8, 8) derivative factors n! / (n-k)!
        self.psi_ref = torch.zeros(num_envs, **f32) # yaw reference [rad], wrapped
        self.psi_dot_ref = torch.zeros(num_envs, **f32) # yaw-rate reference [rad/s]
        self.yaw_moving = torch.zeros(num_envs, device=self.device, dtype=torch.bool) # follower engaged (hysteresis state)

    # ====== ASSIGNMENT ======
    def assign(self, env_ids, bank_idx, origin_xy, yaw, mirror=None, tau0=None, psi0=None):
        # Place bank entries bank_idx (M, ) for envs env_ids (M, ): route start at origin_xy (M, 2), rotated by yaw (M, ).
        # Yaw reference starts at psi0 if given, else at the reference velocity direction at tau0 if already moving
        # (mid-route start), else at the direction the route starts moving in
        self.idx[env_ids] = bank_idx
        self.origin[env_ids] = origin_xy
        self.cos_yaw[env_ids], self.sin_yaw[env_ids] = torch.cos(yaw), torch.sin(yaw)
        self.mirror[env_ids] = 1.0 if mirror is None else mirror
        self.tau[env_ids] = 0.0 if tau0 is None else tau0
        self.alpha[env_ids] = 1.0
        self.t_nodes[env_ids] = self.bank.t_nodes[bank_idx]
        self.durations[env_ids] = self.bank.durations[bank_idx]
        self.duration[env_ids] = self.bank.duration[bank_idx]
        self._reset_yaw(env_ids, psi0)

    def assign_random(self, env_ids, origin_xy, mirror=True, p_mid_route=0.0):
        # Uniform bank entry, uniform yaw, random mirror; with probability p_mid_route a uniform start time on the route
        M = len(env_ids)
        f32 = dict(device=self.device, dtype=torch.float32)
        bank_idx = torch.randint(0, self.bank.size, (M,), device=self.device)
        yaw = (torch.rand(M, **f32) * 2.0 - 1.0) * torch.pi
        flip = torch.where(torch.rand(M, **f32) < 0.5, -1.0, 1.0) if mirror else None
        tau0 = torch.where(torch.rand(M, **f32) < p_mid_route, torch.rand(M, **f32) * self.bank.duration[bank_idx], 0.0)
        self.assign(env_ids, bank_idx, origin_xy, yaw, flip, tau0)

    # ====== CLOCK ======
    def step(self, dt):
        # Advance the reference clock, then the yaw follower at the new time
        self.tau += self.alpha * dt
        self._update_yaw(dt)

    @property
    def done(self):
        # Reference has reached its end point (holding from here on)
        return self.tau >= self.duration

    def heading0(self, env_ids=None):
        # World direction each env's route starts moving in; mirroring flips the route-frame angle
        ids = slice(None) if env_ids is None else env_ids
        h = self.mirror[ids] * self.bank.heading0[self.idx[ids]]
        return wrap_angle(torch.atan2(self.sin_yaw[ids], self.cos_yaw[ids]) + h)

    # ====== YAW REFERENCE ======
    def _reset_yaw(self, env_ids, psi0=None):
        cfg = self.cfg
        v = self.eval([0.0], max_order=1, env_ids=env_ids)[1, :, 0, :2]
        speed = v.norm(dim=-1)
        moving = speed > cfg.yaw_min_speed
        if psi0 is None:
            psi0 = torch.where(moving & (cfg.yaw_mode == "velocity"), torch.atan2(v[:, 1], v[:, 0]), self.heading0(env_ids))
        self.psi_ref[env_ids] = wrap_angle(torch.as_tensor(psi0, device=self.device, dtype=torch.float32))
        self.psi_dot_ref[env_ids] = 0.0
        self.yaw_moving[env_ids] = moving

    def _update_yaw(self, dt):
        cfg = self.cfg
        if cfg.yaw_mode == "fixed":
            return
        v = self.eval([0.0], max_order=1)[1, :, 0, :2]
        speed = v.norm(dim=-1)
        self.yaw_moving = torch.where(self.yaw_moving, speed > YAW_HYSTERESIS * cfg.yaw_min_speed, speed > cfg.yaw_min_speed)
        target = torch.where(self.yaw_moving, torch.atan2(v[:, 1], v[:, 0]), self.psi_ref)
        self.psi_dot_ref = (wrap_angle(target - self.psi_ref) / cfg.yaw_tau).clamp(-cfg.yaw_rate_max, cfg.yaw_rate_max)
        self.psi_ref = wrap_angle(self.psi_ref + self.psi_dot_ref * dt)

    # ====== EVALUATION ======
    def eval(self, offsets, max_order=2, env_ids=None):
        # Derivatives 0..max_order at τ + offsets (K, ) for every env (or env_ids), in the world frame
        # -> (max_order+1, N, K, 3). Order 0 is position, 1 velocity, 2 acceleration
        ids = slice(None) if env_ids is None else env_ids
        offsets = torch.as_tensor(offsets, device=self.device, dtype=torch.float32)
        t_nodes = self.t_nodes[ids]
        t = self.tau[ids][:, None] + offsets[None, :] # (N, K)
        dur = self.duration[ids][:, None]
        inside = ((t >= 0.0) & (t <= dur))[..., None]
        t = torch.minimum(t.clamp_min(0.0), dur).contiguous()
        seg = (torch.searchsorted(t_nodes, t, right=True) - 1).clamp(0, self.bank.max_segments - 1)
        T = self.durations[ids].gather(1, seg)
        tau_n = ((t - t_nodes.gather(1, seg)) / T)[..., None] # (N, K, 1)
        c = self.bank.coeffs[self.idx[ids][:, None], seg] # (N, K, 3, 8)

        out = []
        for k in range(max_order + 1):
            dc = c[..., k:] * self.fact[k, k:] # coefficients of the k-th τ-derivative, ascending
            acc = dc[..., -1]
            for n in range(dc.shape[-1] - 2, -1, -1): # Horner
                acc = acc * tau_n + dc[..., n]
            acc = acc / T[..., None] ** k
            out.append(acc if k == 0 else acc * inside)
        return self._to_world(torch.stack(out), ids)

    def _to_world(self, d, ids):
        # Route frame -> world: mirror y, rotate xy by yaw, translate positions (order 0) by the origin
        c, s = self.cos_yaw[ids][None, :, None], self.sin_yaw[ids][None, :, None]
        x, y, z = d[..., 0], self.mirror[ids][None, :, None] * d[..., 1], d[..., 2]
        xy = torch.stack([c * x - s * y, s * x + c * y], dim=-1)
        xy[0] = xy[0] + self.origin[ids][:, None, :]
        return torch.cat([xy, z[..., None]], dim=-1)
