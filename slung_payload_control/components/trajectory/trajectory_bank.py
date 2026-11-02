import os
import time
import datetime
import json
import torch
import numpy as np
from dataclasses import asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from .trajectory import DERIV, N_COEFF
from config.config_components import TrajectoryConfig

METRIC_KEYS = ("v_peak", "a_peak", "j_peak", "t_end", "z_min")
MAX_REDRAWS = 10 # post-solve rejections per bank entry before giving up
Z_MARGIN = 0.2 # polynomial may dip this far below route_alt_min between nodes

class TrajectoryBank:
    _worker = {} # per-process sampler/solver, set by _init_worker in each build worker

    def __init__(self, arrays, device):
        # Holds the loaded bank as tensors on 'device'
        # Tensors are only built via load()
        # Bank itself is built via build()
        self.device = device
        as_t = lambda a, dtype=torch.float32: torch.as_tensor(a, device=self.device, dtype=dtype)

        self.coeffs = as_t(arrays["coeffs"]) # (N, S, 8, 3) in normalized segment time 
        self.durations = as_t(arrays["durations"]) # (N, S) padding segments have T = 1
        self.n_seg = as_t(arrays["n_seg"], torch.int64) # (N, )
        self.t_end = as_t(arrays["t_end"]) # (N, )
        self.keyword_id = as_t(arrays["keyword_id"], torch.int64) # (N, )
        self.metrics = as_t(arrays["metrics"]) # (N, len(metric_keys))
        self.metric_keys = tuple(arrays["metric_keys"].tolist())

        zeros = torch.zeros_like(self.durations[:, :1])
        self.t_nodes = torch.cat([zeros, torch.cumsum(self.durations, dim=1)], dim=1) # (N, S+1)
        self.deriv = as_t(DERIV) # (8, 8)
        self.powers = torch.arange(N_COEFF, device=self.device)
        self.size = len(self.t_end) # bank size (N, )

    # ========================== OFFLINE BUILD ==========================
    @staticmethod
    def _init_worker(cfg: TrajectoryConfig):
        from .route import RouteSampler # imported only when building
        from .min_snap_solver import MinSnapSolver
        TrajectoryBank._worker.update(cfg=cfg, sampler=RouteSampler(cfg), solver=MinSnapSolver())
    
    @staticmethod
    def _build_one(job):
        # One bank entry: sample -> solve -> limit -> validate. Redraw from child seeds on reject
        i, seq, max_duration = job
        w = TrajectoryBank._worker
        cfg = w["cfg"]
        sampler = w["sampler"]
        solver = w["solver"]

        for attempt, child in enumerate(seq.spawn(MAX_REDRAWS)):
            try:
                route = sampler.sample(child)
            except RuntimeError: # sampler exausted its own retries
                continue

            traj = solver.solve(route).enforce_limits(cfg.a_max, cfg.j_max)
            route, traj = TrajectoryBank._fit_duration(route, traj, cfg, solver)

            m = traj.metrics()
            if m["t_end"] > max_duration or m["z_min"] < cfg.route_alt_min - Z_MARGIN:
                continue
            if traj.n_segments > cfg.traj_max_segments:
                continue

            coeffs, durations, n_seg = traj.padded(cfg.traj_max_segments)
            return dict(
                coeffs=coeffs,
                durations=durations,
                n_seg=n_seg,
                t_end=m["t_end"],
                keyword_id=cfg.keywords.index(route.meta["keyword"]),
                v_des=route.v_des,
                attempt=attempt,
                metrics=[m[k] for k in METRIC_KEYS],
            )
        
        raise RuntimeError(f"bank entry {i}: no valid trajectory after {MAX_REDRAWS} redraws")

    @staticmethod
    def _fit_duration(route, traj, cfg, solver):
        # Removes route waypoints and resolves traj if input traj exceeds traj_max_duration
        from .route import Route
        wpts = route.wpts
        max_duration = cfg.traj_max_duration
        while traj.t_end > max_duration and len(wpts) > 2:
            keep = max(2, min(len(wpts) - 1, int(len(wpts) * max_duration / traj.t_end)))
            wpts = wpts[:keep]
            route = Route(wpts, route.v_des, cfg.corridor, cfg.a_max, cfg.traj_max_segments, route.meta)
            traj = solver.solve(route).enforce_limits(cfg.a_max, cfg.j_max)
        return route, traj

    @staticmethod
    def _stats(arrays, cfg, build_s):
        # summary of built bank: sizes, redraws, and metric distr overall and per keyword
        def describe(x):
            x = np.asarray(x, dtype=np.float64)
            q = np.percentile(x, [0, 10, 50, 90, 100]).round(3).tolist()
            return dict(zip(("min", "p10", "median", "p90", "max"), q), mean=round(float(x.mean()), 3))

        metrics = arrays["metrics"]
        kw = arrays["keyword_id"]
        t_end = metrics[:, METRIC_KEYS.index("t_end")]
        per_kw = {}
        for k, name in enumerate(cfg.keywords):
            m = kw == k
            per_kw[name] = {"count": int(m.sum())}
            if m.any():
                per_kw[name] |= {key: describe(metrics[m, j]) for j, key in enumerate(METRIC_KEYS)}

        return {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "build_seconds": round(build_s, 1),
            "size": int(len(kw)),
            "budget_s": cfg.traj_max_duration,
            "frac_near_budget": round(float((t_end > 0.9*cfg.traj_max_duration).mean()), 3), #pile-up chec
            "redraw_hist": np.bincount(arrays["attempt"]).tolist(), #index = redraws needed
            "n_segments": describe(arrays["n_seg"]),
            "v_des": describe(arrays["v_des"]),
            "metrics": {key: describe(metrics[:, j]) for j, key in enumerate(METRIC_KEYS)},
            "keywords": per_kw,
            "config": json.loads(str(arrays["config"])),
        }

    @classmethod
    def build(cls, cfg: TrajectoryConfig, path, num_workers=None):
        t0 = time.perf_counter()
        print(f"[TrajectoryBank] Building bank of {cfg.bank_size} trajectories...")

        max_duration = cfg.traj_max_duration
        num_workers = num_workers or cfg.num_workers
        seqs = np.random.SeedSequence(cfg.seed).spawn(cfg.bank_size) # one independent stram per entry
        jobs = [(i, seq, max_duration) for i, seq in enumerate(seqs)]

        if num_workers <= 1: # in-process: readable tracevacks for debugging
            cls._init_worker(cfg)
            results = [cls._build_one(job) for job in jobs]
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(num_workers, mp_context=ctx, initializer=cls._init_worker, initargs=(cfg,)) as pool:
                results = list(pool.map(cls._build_one, jobs, chunksize=64)) # map keeps entry order
        arrays = dict(
            coeffs = np.stack([r["coeffs"] for r in results]).astype(np.float32), # (N, S, 8, 3)
            durations = np.stack([r["durations"] for r in results]).astype(np.float32), # (N, S)
            n_seg = np.array([r["n_seg"] for r in results], dtype=np.int16),
            t_end = np.array([r["t_end"] for r in results], dtype=np.float32),
            keyword_id = np.array([r["keyword_id"] for r in results], dtype=np.int8),
            v_des = np.array([r["v_des"] for r in results], dtype=np.float32),
            attempt = np.array([r["attempt"] for r in results], dtype=np.int8),
            metrics = np.array([r["metrics"] for r in results], dtype=np.float32),
            metric_keys = np.array(METRIC_KEYS),
            config = np.array(json.dumps(asdict(cfg), sort_keys=True)),
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)

        # Save metafile with bank info
        meta_path = os.path.splitext(path)[0] + "_meta.json" # bank.npz -> bank_meta.json
        with open(meta_path, "w") as f:
            json.dump(cls._stats(arrays, cfg, time.perf_counter() - t0), f, indent=2)
        
        print(f"[TrajectoryBank] Saved {path} and {meta_path}")
        return path

    # ========================== ONLINE LOAD/USE ==========================
    @staticmethod
    def _config_matches(arrays, cfg):
        stored = json.loads(str(arrays["config"]))
        current = json.loads(json.dumps(asdict(cfg), sort_keys=True))
        return stored == current

    @classmethod
    def load(cls, path, device, cfg=None):
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        if cfg is not None and not cls._config_matches(arrays, cfg):
            raise ValueError(f"{path} was built with a different TrajectoryConfig - Rebuild instead")
        return cls(arrays, device) # Re-inits the class with the loaded bank arrays

    @classmethod
    def load_or_build(cls, cfg, path, device):
        try:
            loaded = cls.load(path, device, cfg)
            print(f"[TrajectoryBank] Bank loaded from {path}...")
            return loaded
        except (FileNotFoundError, ValueError):
            print(f"[TrajectoryBank] Bank with matching config not found at {path} - rebuilding...")
            cls.build(cfg, path)
            return cls.load(path, device, cfg)

    def sample_ids(self, n, weights=None, generator=None, dtype=torch.int32):
        # draw n random traj idxs independently and uniformly from the whole bank
        if weights is None:
            return torch.randint(high=self.size, size=(n, ), device=self.device, generator=generator, dtype=dtype)

        # weights: optional (N,) per-trajectory weights; entry i is drawn with probability weights[i] / weights.sum().
        # Without weights every entry is equally likely, so the mix follows the bank's build-time composition.
        return torch.multinomial(weights=weights, num_samples=n, replacement=True, generator=generator, dtype=dtype)

    def eval(self, idx, t, orders=(0,1)):
        # idx (E, )
        # t (E, K) seconds since traj start -> (E, K, 3) position or derivative
        t = torch.minimum(t.clamp(min=0.0), self.t_end[idx, None]) # hold the final point after t_end
        nodes = self.t_nodes[idx] # (E, S+1)
        seg = torch.searchsorted(nodes, t.contiguous(), right=True) - 1
        seg = torch.minimum(seg.clamp(min=0), self.n_seg[idx, None] - 1) # never enter padding
        T = self.durations[idx[:, None], seg] # (E, K)
        tau = (t - nodes.gather(1, seg)) / T
        c = self.coeffs[idx[:, None], seg] # (E, K, 8, 3)

        out = []
        for r in orders:
            n = self.powers[r:]
            basis = self.deriv[r, r:] * tau[..., None] ** (n - r) # (E, K, 8-r)
            out.append(torch.einsum("ekn, eknd->ekd", basis, c[:, :, r:]) / T[..., None]**r)
        return tuple(out)