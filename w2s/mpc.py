"""Linear model predictive car following (2026-10-28, Section 5.2): the classical MPC the CAV can run with a preview.

State of the CAV (position, speed, acceleration), input the jerk, held over prediction steps of TM s (exact triple
integrator). Over a horizon of N steps the predecessor follows its forecast (exact over the preview, held after it).
Cost, per step of TM:
    q_g delta_i^2 + q_v (v_p,i - v_i)^2 + r_a a_i^2 + r_j j_i^2,     delta = g - S_r(v)  (r-Safety spacing deviation),
with S_r linearised about the predecessor's forecast speed. Constraints: A_MIN <= a <= A_MAX, |j| <= J_MAX, v >= 0 and
the band of Eq. (band), -R_b(v) <= delta <= R_b(v_bar), softened by slacks of weight W_SLACK.
Sparse form (states and inputs as variables, the dynamics as equality constraints), which keeps the problem well
conditioned over long horizons; solved by OSQP, one solver per horizon length, updated in place.
"""
import numpy as np
import osqp
from scipy import sparse

from lib import A_MAX, A_MIN, DT, J_MAX, L_EFF

TM = 0.5                 # prediction step (s)
W_SLACK = 1e3
EPS = 1e-5


class MPC:
    def __init__(self, rs, qg=1.0, qv=1.0, ra=1.0, rj=1.0):
        self.rs, self.w = rs, dict(qg=float(qg), qv=float(qv), ra=float(ra), rj=float(rj))
        self._cache = {}

    # variables z = [x_1..x_N, v_1..v_N, a_1..a_N, j_0..j_(N-1), s_1..s_N]
    def _mats(self, N, d1):
        T, w, I = TM, self.w, sparse.eye(N, format="csc")
        Dd = sparse.diags(d1)
        Z = sparse.csc_matrix((N, N))
        # cost: qg (x + d1 v)^2 + qv v^2 + ra a^2 + rj j^2 + W s^2 (times 2 TM, OSQP's 1/2 z'Pz)
        Pxx, Pxv, Pvv = w["qg"] * I, w["qg"] * Dd, w["qg"] * Dd @ Dd + w["qv"] * I
        P = 2 * T * sparse.bmat([[Pxx, Pxv, None, None, None], [None, Pvv, None, None, None],
                                 [None, None, w["ra"] * I, None, None], [None, None, None, w["rj"] * I, None],
                                 [None, None, None, None, W_SLACK * I]], format="csc")
        P = sparse.triu(P, format="csc")
        S = sparse.diags([np.ones(N - 1)], [-1], shape=(N, N))            # shift: row i picks state i-1
        # dynamics rows: state_i - A state_(i-1) - B j_(i-1) = (A s0 + B j0 terms for i = 1)
        ex = [I - S, -T * S, -T * T / 2 * S, -T ** 3 / 6 * I, None]
        ev = [None, I - S, -T * S, -T * T / 2 * I, None]
        ea = [None, None, I - S, -T * I, None]
        band = [-I, -Dd, None, None, I]                                  # -x - d1 v + s >= lower
        band2 = [-I, -Dd, None, None, -I]                                # -x - d1 v - s <= upper
        rows = [ex, ev, ea, band, band2, [None, None, I, None, None], [None, None, None, I, None],
                [None, I, None, None, None], [None, None, None, None, I]]
        rows = [[Z if b is None else b for b in r] for r in rows]
        A = sparse.bmat(rows, format="csc")
        return P, A

    def _solver(self, N, P, A, q, lo, hi):
        if N not in self._cache:
            s = osqp.OSQP()
            s.setup(P, q, A, lo, hi, verbose=False, warm_starting=True, polishing=False, eps_abs=EPS, eps_rel=EPS,
                    max_iter=20000)
            self._cache[N] = s
        else:
            s = self._cache[N]
            s.update(Px=P.data, Ax=A.data, q=q, l=lo, u=hi)
        return s

    def solve(self, x0, v0, a0, xpf, vpf, v_bar):
        """Jerk sequence over N = len(xpf) steps; xpf, vpf: the predecessor's forecast at steps 1..N. Returns
        (j, v, a) at steps 0..N-1 (inputs) and 1..N (states)."""
        N, T = len(xpf), TM
        vl = np.maximum(vpf, 0.0)
        d1 = self.rs.tau_h + self.rs.r * vl / self.rs.a_ref
        c = xpf - L_EFF - self.rs.sr(vl) + d1 * vl                      # delta = c - x - d1 v
        P, A = self._mats(N, d1)
        w = self.w
        q = 2 * T * np.r_[-w["qg"] * c, -w["qg"] * d1 * c - w["qv"] * vpf, np.zeros(3 * N)]
        e0 = np.zeros((3, N))
        e0[:, 0] = [x0 + T * v0 + T * T / 2 * a0, v0 + T * a0, a0]
        eq = np.r_[e0[0], e0[1], e0[2]]
        rb_lo, rb_up = self.rs.rb(vl), self.rs.rb(v_bar)
        inf = np.full(N, np.inf)
        lo = np.r_[eq, -rb_lo - c, -inf, np.full(N, A_MIN), np.full(N, -J_MAX), np.zeros(N), np.zeros(N)]
        hi = np.r_[eq, inf, rb_up - c, np.full(N, A_MAX), np.full(N, J_MAX), inf, inf]
        r = self._solver(N, P, A, q, lo, hi).solve()
        if r.x is None or not np.all(np.isfinite(r.x)):
            return np.zeros(N), np.full(N, v0), np.full(N, a0)
        z = r.x
        return z[3 * N:4 * N], z[N:2 * N], z[2 * N:3 * N]


def sample(arr, k, N, K):
    """arr at steps k + i TM / DT, i = 1..N (held after its end)."""
    m = int(round(TM / DT))
    return arr[np.minimum(k + m * np.arange(1, N + 1), K - 1)]
