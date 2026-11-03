import math
import numpy as np
from dataclasses import dataclass, field

N_COEFF = 8 # 7th order polynomial per segment and axis

# DERIV[r, n] = n! / (n-r)! for n >= r
# coefficient of tau^(n-r) in the r-th tau-derivative of tau^n
DERIV = np.array(
    [[math.perm(n, r) for n in range(N_COEFF)] for r in range(N_COEFF)], 
    dtype=np.float64)

@dataclass
class PolyTrajectory:
    coeffs: np.ndarray # (S, N_COEFF, 3), coeffs[s, n]
    durations: np.ndarray # (S, )
    meta: dict = field(default_factory=dict) # keyword, seed, v_des ...

    def __post_init__(self):
        self.coeffs = np.asarray(self.coeffs, dtype=np.float64)
        self.durations = np.asarray(self.durations, dtype=np.float64)
        if self.coeffs.ndim != 3 or self.coeffs.shape[1:] != (N_COEFF, 3):
            raise ValueError(f"coeffs must be (S, {N_COEFF}, 3), got {self.coeffs.shape}")
        if len(self.durations) != len(self.coeffs) or not (self.durations > 0).all():
            raise ValueError("duratiosn must be positive and match the number of segments")
        self.t_nodes = np.concatenate([[0.0], np.cumsum(self.durations)])

    @property
    def n_segments(self):
        return len(self.durations)

    @property
    def t_end(self):
        return self.t_nodes[-1]

    def eval(self, t, order=0):
        # Position (order=0) or the order-th time derivative at times t (t clamped 0;t_end)
        
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=np.float64)), 0.0, self.t_end)
        seg = np.searchsorted(self.t_nodes, t, side="right") # segment where t fits in wrt t_nodes
        seg = np.clip(seg-1, 0, self.n_segments-1)
        T = self.durations[seg]
        tau = (t - self.t_nodes[seg]) / T # normalized time tau in [0,1]

        n = np.arange(order, N_COEFF)
        basis = DERIV[order, n] * tau[:, None] ** (n - order) # (K, N_COEFF - order)
        out = np.einsum("kn, knd->kd", basis, self.coeffs[seg][:, order:])

        return out / T[:, None] ** order # d/dt = (1/T) d/dtau

    def sample(self, dt, order=0):
        t = np.arange(0.0, self.t_end+0.5*dt, dt)
        return t, self.eval(t, order)

    def time_scaled(self, k):
        # stretch time by k
        return PolyTrajectory(self.coeffs.copy(), self.durations * k, dict(self.meta))

    def metrics(self, dt=0.01):
        t = np.arange(0.0, self.t_end+0.5 * dt, dt)
        peak = {name: np.linalg.norm(self.eval(t, r), axis=-1).max() for r, name in ((1, "v_peak"), (2, "a_peak"), (3, "j_peak"))}
        return {**peak, "t_end": self.t_end, "n_segments": self.n_segments, "z_min": self.eval(t,0)[:,2].min()}

    def enforce_limits(self, a_max, j_max, dt=0.01):
        m = self.metrics(dt)
        k = max(1.0, math.sqrt(m["a_peak"] / a_max), (m["j_peak"] / j_max) ** (1.0 / 3.0))
        return self if k == 1.0 else self.time_scaled(k)

    def padded(self, s_max):
        if self.n_segments > s_max:
            raise ValueError(f"{self.n_segments} segments exceed s_max={s_max}")
        coeffs = np.zeros((s_max, N_COEFF, 3))
        coeffs[:self.n_segments] = self.coeffs
        coeffs[self.n_segments:, 0] = self.eval(self.t_end)[0]
        durations = np.ones(s_max)
        durations[:self.n_segments] = self.durations
        return coeffs, durations, self.n_segments