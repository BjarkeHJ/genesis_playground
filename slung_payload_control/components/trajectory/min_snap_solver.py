import numpy as np
from .trajectory import PolyTrajectory, DERIV, N_COEFF

# derivatives shared at each node: pos, vel, acc, jerk
N_DER = N_COEFF // 2 
SNAP = 4

class MinSnapSolver:
    def __init__(self, pin_velocity=False):
        self.pin_velocity = pin_velocity

        # A unit maps tau coeffs to tau-derivs at tau=0 (rows 0..3) and tua=1 (rows 4...7)
        A = np.zeros((N_COEFF, N_COEFF))
        for r in range(N_DER):
            A[r, r] = DERIV[r, r]
            A[N_DER + r, r:] = DERIV[r, r:]
        self.A_inv = np.linalg.inv(A)

        # Q[i,j] = integral over [0,1] of the 4th tau-derivatives of tau^i and tau^j
        Q = np.zeros((N_COEFF, N_COEFF))
        for i in range(SNAP, N_COEFF):
            for j in range(SNAP, N_COEFF):
                Q[i, j] = DERIV[SNAP, i] * DERIV[SNAP, j] / (i + j - 2 * SNAP + 1)
        self.Q = Q

    def solve(self, route):
        return self.solve_nodes(route.node_pos, route.node_vel, route.node_rest, route.seg_T, route.meta)

    def solve_nodes(self, pos, vel, rest, T, meta=None):
        M = len(pos) # Number of nodes
        S = len(T) # Number of segments (with stamps)

        # Node derivative table x[node, order, axis]; fixed[node, order] marks known entries
        x = np.zeros((M, N_DER, 3))
        x[:, 0] = pos
        x[:, 1] = vel
        fixed = np.zeros((M, N_DER), dtype=bool)
        fixed[:, 0] = True
        fixed[:, 1] = self.pin_velocity | rest
        fixed[rest, 2:] = True # rest nodes: vel = acc = jerk = 0

        # Global cost x^T H x, assembled from per-segment blocks over [x_s, x_(s+1)]
        H = np.zeros((M * N_DER, M * N_DER))
        for s in range(S):
            idx = slice(s * N_DER, (s+2) * N_DER)
            H[idx, idx] += self._segment_cost(T[s])

        f = fixed.ravel() # flatten array
        xf = x.reshape(-1, 3)
        if (~f).any(): # if any False in fixed -> any not-fixed
            # Minimize over free entries: H_pp x_p = -H_pf x_f, one RHS column per axis
            xf[~f] = np.linalg.solve(H[np.ix_(~f, ~f)], -H[np.ix_(~f, f)] @ xf[f])

        coeffs = np.stack([self._segment_coeffs(xf[s * N_DER:(s + 2) * N_DER], T[s]) for s in range(S)])
        return PolyTrajectory(coeffs, T, dict(meta or {}))

    def _segment_coeffs(self, e, T):
        # Time derivatives at both ends -> tau-coeffs: tau-derivs of roder r=T^r * time-deriv
        scale = np.tile(T ** np.arange(N_DER), 2)
        return self.A_inv @ (scale[:, None] * e)

    def _segment_cost(self, T):
        # Snap cost in terms of end derivatives: integral of |d4p/dt4|^2 dt = T^-7 * c^T Q c
        B = self.A_inv * np.tile(T ** np.arange(N_DER), 2)[None, :]
        return B.T @ self.Q @ B / T**7