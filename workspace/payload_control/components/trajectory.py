import math
import numpy as np

from config.config_env import EPS, RouteConfig
from .route import Route, MIN_NODE_TIME

# Route nodes -> timed minimum-snap trajectory of the payload position. One 7th-order polynomial per segment and axis.
# Position and velocity are fixed at every node, acceleration and jerk are optimized at interior nodes, and the route
# ends are at rest (v = a = j = 0). Free a and j keep snap continuous at every node.
#
# Snap cost scales as T⁻⁷ per segment, so a long segment with free ends absorbs any shaping its short neighbours want
# at almost no cost (large excursions, speed overshoot). Every straight therefore gets nodes along its planned speed
# profile (trapezoid, or triangle if it can't reach v_des) at the phase changes and at most every MAX_STRAIGHT_SEG_TIME,
# with the profile's speed pinned. No segment is long, and on a cruise the line at v_des is the zero-cost optimum.
#
# Coefficients are stored in normalized segment time τ = (t - t_i) / T_i ∈ [0, 1], which keeps the solve well
# conditioned for durations from 0.2 s to minutes, and makes uniform time-scaling a change of T only

ORDER = 7
N_COEF = ORDER + 1
SNAP = 4

# Generator internals: time allocation and min-snap shaping, independent of the drone-payload system. Not meant to be
# tuned; the mission spec and system envelope are in RouteConfig
A_ALLOC_FRAC = 0.5 # planned acceleration on straights, fraction of a_max: min-snap peaks above the trapezoid it follows
A_REST_FRAC = 0.25 # ...for ramps from and to rest (route ends, stop nodes). Snap is free there and jumps from 0 at hover,
                   # a step in drone tilt via the cable (≈ L/g · snap); snap scales ~ a³
MIN_SEG_TIME = 0.2 # floor on any segment duration [s]
MIN_STRAIGHT_SEG_TIME = 0.5 # straight nodes at least this far apart in time and from the straight's ends (room to blend out of a fillet) [s]
MAX_STRAIGHT_SEG_TIME = 1.0 # longer straight pieces are split: long segments let min-snap bend them at no cost [s]
MIN_REST_SEG_TIME = 2.0 # ...and pieces into or out of rest at least this long: a short piece out of v = a = j = 0 needs a large snap [s]
SCALE_CHECK_DT = 0.01 # sampling step of the peak search for time-scaling [s]

def _derivative_factors(max_order=ORDER):
    # F[k, n] = n! / (n-k)! for n >= k else 0: coefficient of τ^(n-k) in the k-th derivative of τ^n
    F = np.zeros((max_order + 1, N_COEF))
    for k in range(max_order + 1):
        for n in range(k, N_COEF):
            F[k, n] = math.factorial(n) / math.factorial(n - k)
    return F

FACT = _derivative_factors()

def _unit_matrices():
    # On τ ∈ [0, 1]: A maps coefficients to boundary derivatives [p, v, a, j](0) + [p, v, a, j](1), Q is the snap cost
    # ∫ (d⁴p/dτ⁴)² dτ as a quadratic form in the coefficients. H = A⁻ᵀ Q A⁻¹ is the cost in boundary derivatives
    A = np.zeros((N_COEF, N_COEF))
    for k in range(4):
        A[k, k] = FACT[k, k] # at τ = 0 only the τ^k term survives
        A[4 + k] = FACT[k] # at τ = 1 every term with n >= k
    Q = np.zeros((N_COEF, N_COEF))
    for m in range(SNAP, N_COEF):
        for n in range(SNAP, N_COEF):
            Q[m, n] = FACT[SNAP, m] * FACT[SNAP, n] / (m + n - 2 * SNAP + 1)
    A_inv = np.linalg.inv(A)
    return A_inv, A_inv.T @ Q @ A_inv

A_INV, H_UNIT = _unit_matrices()
BOUNDARY_POW = np.array([0, 1, 2, 3, 0, 1, 2, 3]) # derivative order of each boundary entry

class Trajectory:
    # Piecewise polynomial in normalized segment time. coeffs (S, 3, 8) ascending powers of τ, durations (S, ).
    # Outside [0, duration] the trajectory holds its end state: position clamped, all derivatives zero
    def __init__(self, coeffs, durations, k_scale=1.0):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.t_nodes = np.concatenate([[0.0], np.cumsum(self.durations)]) # (S+1, )
        self.k_scale = k_scale # time-scaling factor applied by the solver (1 = no stretch)

    @property
    def num_segments(self):
        return len(self.durations)

    @property
    def duration(self):
        return float(self.t_nodes[-1])

    def scaled(self, k):
        # Uniform time stretch: same τ-polynomials, durations x k. Derivative of order n scales by k^-n
        return Trajectory(self.coeffs, self.durations * k, self.k_scale * k)

    def eval(self, t, max_order=0):
        # Derivatives 0..max_order at times t (N, ) -> (max_order+1, N, 3)
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        inside = (t >= 0.0) & (t <= self.duration)
        tc = np.clip(t, 0.0, self.duration)
        seg = np.clip(np.searchsorted(self.t_nodes, tc, side="right") - 1, 0, self.num_segments - 1)
        T = self.durations[seg]
        tau = (tc - self.t_nodes[seg]) / T
        c = self.coeffs[seg] # (N, 3, 8)

        out = np.empty((max_order + 1, len(t), 3))
        for k in range(max_order + 1):
            dc = c[..., k:] * FACT[k, k:] # coefficients of the k-th τ-derivative, ascending
            acc = dc[..., -1]
            for n in range(dc.shape[-1] - 2, -1, -1): # Horner
                acc = acc * tau[:, None] + dc[..., n]
            out[k] = acc / T[:, None] ** k
            if k > 0:
                out[k][~inside] = 0.0
        return out

    def sample(self, dt, max_order=0):
        t = np.arange(0.0, self.duration + 0.5 * dt, dt)
        return t, self.eval(t, max_order)

    def peaks(self, dt):
        # Max |v|, |a|, |j| over the trajectory (norm over axes), from dense sampling
        _, d = self.sample(dt, max_order=3)
        n = np.linalg.norm(d, axis=-1).max(axis=-1)
        return {"v": n[1], "a": n[2], "j": n[3]}

    def __repr__(self):
        return f"Trajectory(segments={self.num_segments}, duration={self.duration:.1f} s, k={self.k_scale:.3f})"

class MinSnapSolver:
    # Route -> Trajectory: plan solver nodes and segment times, solve the min-snap problem for fixed times, then stretch
    # time uniformly until a_max and j_max hold. Uniform stretching keeps the solution min-snap optimal for the new times
    def __init__(self, cfg: RouteConfig = RouteConfig()):
        self.cfg = cfg

    def solve(self, route: Route) -> Trajectory:
        return self.time_scale(self.solve_fixed_times(*self.plan(route)))

    # ====== NODES AND TIME ALLOCATION ======
    def plan(self, route: Route):
        # Route nodes plus straight nodes -> pos (K, 3), vel (K, 3), fix_aj (K, ), durations (K-1, ). Fillet segments
        # (entry -> exit of one corner) run at constant speed over the arc length R θ. Straights follow their speed
        # profile, cut at the phase changes and evenly within each phase so no piece exceeds MAX_STRAIGHT_SEG_TIME;
        # cuts closer than MIN_STRAIGHT_SEG_TIME to the previous cut or the straight's end (MIN_REST_SEG_TIME to a
        # rest end) are dropped
        if route.is_hover:
            return route.node_pos, route.node_vel, np.ones(1, dtype=bool), np.ones(0)
        cfg = self.cfg
        P = route.node_pos
        arc_len = {a: route.corner_radius[c] * route.corner_theta[c] for c, (a, b) in enumerate(route.corner_idx) if b == a + 1}
        L_seg = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        for a, L_arc in arc_len.items():
            L_seg[a] = max(L_seg[a], L_arc)
        rest = route.node_speed < EPS # route ends and stop nodes: ramps to and from them use A_REST_FRAC · a_max
        acc = np.where(rest, A_REST_FRAC, A_ALLOC_FRAC) * cfg.a_max
        speed = self._reachable_speeds(route.node_speed, L_seg, arc_len, rest)
        V = route.node_vel * (speed / np.maximum(route.node_speed, EPS))[:, None]

        pos, vel, fix, T = [P[0]], [V[0]], [True], [] # fix: a = j = 0 fixed (route ends only)
        for i in range(route.num_segments):
            v_a, v_b = speed[i], speed[i + 1]
            chord = P[i + 1] - P[i]
            L = np.linalg.norm(chord)
            n_first = len(T) # index of this route segment's first piece
            if i in arc_len:
                T.append(2.0 * L_seg[i] / max(v_a + v_b, EPS))
            else:
                u = chord / max(L, EPS)
                phases = self._speed_profile(L, v_a, v_b, route.v_des, acc[i], acc[i + 1])
                t_end = sum(ph[0] for ph in phases)
                t_last = 0.0
                gap_in, gap_out = (MIN_REST_SEG_TIME if r else MIN_STRAIGHT_SEG_TIME for r in (rest[i], rest[i + 1]))
                for t in self._profile_cuts(phases)[:-1]:
                    gap = gap_in if t_last == 0.0 else MIN_STRAIGHT_SEG_TIME
                    if t - t_last >= gap and t_end - t >= gap_out:
                        s, v = self._profile_state(phases, t)
                        pos.append(P[i] + s * u); vel.append(v * u); fix.append(False)
                        T.append(t - t_last)
                        t_last = t
                T.append(t_end - t_last)
            # A piece out of or into rest is never shorter than MIN_REST_SEG_TIME, whatever the profile planned (a
            # short ramp overshooting its end speed would squeeze it). The speed cap in _reachable_speeds keeps the
            # pinned end speed consistent with that duration
            if rest[i]:
                T[n_first] = max(T[n_first], MIN_REST_SEG_TIME)
            if rest[i + 1]:
                T[-1] = max(T[-1], MIN_REST_SEG_TIME)
            pos.append(P[i + 1]); vel.append(V[i + 1]); fix.append(i + 1 == route.num_segments)
        return np.array(pos), np.array(vel), np.array(fix), np.maximum(np.array(T), MIN_SEG_TIME)

    def _reachable_speeds(self, speed, L_seg, arc_len, rest):
        # Corner speeds are set per corner; a slow corner shortly before a fast one asks for more than A_ALLOC_FRAC · a_max between
        # them, and min-snap then bends the path to make up the time. Forward-backward pass: cap each node speed at
        # what its neighbours can reach at A_ALLOC_FRAC · a_max (A_REST_FRAC · a_max from or to a rest node) over the segment between them,
        # and keep each fillet at one speed. Short segments also cap the speed, so no segment takes less than
        # MIN_NODE_TIME, or MIN_REST_SEG_TIME next to a rest node (mean speed v / 2 there)
        cfg = self.cfg
        v = speed.copy()
        at_rest = rest[:-1] | rest[1:] # per segment
        a2 = 2.0 * np.where(at_rest, A_REST_FRAC, A_ALLOC_FRAC) * self.cfg.a_max
        v_short = np.where(at_rest, 2.0 * L_seg / MIN_REST_SEG_TIME, L_seg / MIN_NODE_TIME)
        v[:-1], v[1:] = np.minimum(v[:-1], v_short), np.minimum(v[1:], v_short)
        for _ in range(len(v)): # converges in a couple of passes; the bound is only a safeguard
            v_prev = v.copy()
            for i in range(len(L_seg)):
                v[i + 1] = min(v[i + 1], math.sqrt(v[i]**2 + a2[i] * L_seg[i]))
            for i in reversed(range(len(L_seg))):
                v[i] = min(v[i], math.sqrt(v[i + 1]**2 + a2[i] * L_seg[i]))
            for a in arc_len:
                if 0 < a and a + 1 < len(v) - 1: # a fillet merged into a route end keeps its rest end
                    v[a] = v[a + 1] = min(v[a], v[a + 1])
            if np.allclose(v, v_prev):
                break
        return v

    @staticmethod
    def _speed_profile(L, v_a, v_b, v_des, a_in, a_out):
        # Constant-acceleration phases [(duration, v_start, v_end)] covering L: accelerate at a_in to v_des (or the
        # triangle peak), cruise, decelerate at a_out. If L is too short to even change between v_a and v_b: one phase
        d_acc, d_dec = (v_des**2 - v_a**2) / (2.0 * a_in), (v_des**2 - v_b**2) / (2.0 * a_out)
        if d_acc + d_dec <= L:
            v_peak = v_des
        else: # (v² - v_a²) / 2a_in + (v² - v_b²) / 2a_out = L
            v_peak = math.sqrt((2.0 * L + v_a**2 / a_in + v_b**2 / a_out) / (1.0 / a_in + 1.0 / a_out))
        if v_peak < max(v_a, v_b):
            return [(2.0 * L / max(v_a + v_b, EPS), v_a, v_b)]
        d_acc, d_dec = (v_peak**2 - v_a**2) / (2.0 * a_in), (v_peak**2 - v_b**2) / (2.0 * a_out)
        phases = [((v_peak - v_a) / a_in, v_a, v_peak), ((L - d_acc - d_dec) / v_peak, v_peak, v_peak), ((v_peak - v_b) / a_out, v_peak, v_b)]
        return [ph for ph in phases if ph[0] > EPS]

    def _profile_cuts(self, phases):
        # Candidate cut times: every phase split evenly into pieces of at most MAX_STRAIGHT_SEG_TIME (ends included)
        cuts, t0 = [], 0.0
        for T_ph, _, _ in phases:
            n = max(1, math.ceil(T_ph / MAX_STRAIGHT_SEG_TIME))
            cuts += [t0 + T_ph * m / n for m in range(1, n + 1)]
            t0 += T_ph
        return cuts

    @staticmethod
    def _profile_state(phases, t):
        # Distance and speed along the profile at time t
        s = 0.0
        for T_ph, v0, v1 in phases:
            if t <= T_ph:
                return s + v0 * t + 0.5 * (v1 - v0) / T_ph * t**2, v0 + (v1 - v0) * t / T_ph
            s += 0.5 * (v0 + v1) * T_ph
            t -= T_ph
        return s, phases[-1][2]

    # ====== MIN-SNAP ======
    def solve_fixed_times(self, pos, vel, fix_aj, durations) -> Trajectory:
        # pos, vel (K, 3) fixed at every node; a, j fixed to 0 where fix_aj, optimized elsewhere. durations (K-1, )
        K, S = len(pos), len(durations)
        if K == 1: # hover: constant polynomial, held forever
            coeffs = np.zeros((1, 3, N_COEF))
            coeffs[0, :, 0] = pos[0]
            return Trajectory(coeffs, np.ones(1))
        T = np.asarray(durations, dtype=np.float64)

        # Unknowns: [p, v, a, j] per node, 4K per axis. Segment cost x_segᵀ H_seg x_seg with H_seg = T⁻⁷ D H_UNIT D,
        # D = diag(T^order) converting physical boundary derivatives to τ-derivatives
        H = np.zeros((4 * K, 4 * K))
        for i in range(S):
            D = T[i] ** BOUNDARY_POW
            H[4 * i:4 * i + 8, 4 * i:4 * i + 8] += (D[:, None] * H_UNIT * D[None, :]) / T[i] ** 7

        x = np.zeros((4 * K, 3)) # boundary derivatives, one column per axis
        x[0::4], x[1::4] = pos, vel # fixed a = j = 0 stay from zeros
        free = np.zeros((K, 4), dtype=bool)
        free[~np.asarray(fix_aj), 2:] = True
        free = free.reshape(-1)

        if free.any():
            # min over x_F of the quadratic: H_FF x_F = -H_FP x_P, same matrix for the three axes
            x[free] = np.linalg.solve(H[np.ix_(free, free)], -H[np.ix_(free, ~free)] @ x[~free])

        coeffs = np.empty((S, 3, N_COEF))
        for i in range(S):
            b = x[4 * i:4 * i + 8] * (T[i] ** BOUNDARY_POW)[:, None] # τ-derivatives (8, 3)
            coeffs[i] = (A_INV @ b).T
        return Trajectory(coeffs, T)

    # ====== TIME-SCALING ======
    def time_scale(self, traj: Trajectory) -> Trajectory:
        # a scales as k⁻², j as k⁻³ under a uniform stretch by k
        pk = traj.peaks(SCALE_CHECK_DT)
        k = max(1.0, math.sqrt(pk["a"] / self.cfg.a_max), (pk["j"] / self.cfg.j_max) ** (1.0 / 3.0))
        return traj.scaled(k) if k > 1.0 else traj
