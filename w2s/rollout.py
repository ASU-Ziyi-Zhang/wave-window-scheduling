"""Platoon simulator and the three ways of choosing CAV transitions (exploratory round 1).

Every vehicle depends only on its predecessor, so a platoon is simulated rank by rank over the whole horizon, and
branches (alternative plans) are simulated side by side as rows of arrays.

Scenario: recorded leader (rank 0); followers start in equilibrium at the leader's initial speed; HDVs follow their
calibrated IDM; CAVs run C0 (r-Safety feedback) until they plan at t_plan, then track their committed transition
(Eq. trk) and return to C0 after it. All CAVs receive the leader's transition at t_plan (broadcast).

Modes: 'hdv' (CAV ranks driven by their human IDM), 'feedback', 'independent', 'sequential', 'joint'.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from lib import (A_MAX, A_MIN, DT, G_MIN, J_MAX, L_EFF, SHAPE_INFO, RSafety, cost, idm_acc)


@dataclass
class Setup:
    t: np.ndarray                 # time grid (K,)
    xL: np.ndarray                # leader position (K,)
    vL: np.ndarray                # leader speed (K,)
    idm_true: list                # per rank 1..N: dict
    idm_fore: list                # forecast models per rank 1..N
    cav: tuple                    # CAV ranks
    rs: RSafety
    v2: float
    k_plan: int
    kg: float = 0.2
    kd: float = 0.6
    kv: float = 0.8
    dG: float | None = None       # constant lag limit (round 1 only); None = symmetric band -R_b(v) <= delta <= R_b(v)
    cj: float = 0.1
    s_step: float = 0.5           # start grid of the candidates (s)
    n_keep: int = 9               # candidates kept per CAV and branch in the joint tree (plus feedback)
    beam: int = 0                 # 0 = exact tree; otherwise keep this many partial branches per stage
    N: int = field(init=False)

    def __post_init__(self):
        self.N = len(self.idm_true)


# --------------------------------------------------------------------------------------------------- single vehicles
def _filter(u, g, v, vp, b_lead=3.5, b_own=3.0):
    """Vectorised safety filter: largest u' <= u keeping the one-step-ahead gap above G_MIN plus the braking
    difference (predecessor at b_lead, own at b_own)."""
    def ok(uu):
        v1 = np.maximum(v + uu * DT, 0)
        g1 = g + (vp - 0.5 * (v + v1)) * DT
        return g1 >= G_MIN + v1 ** 2 / (2 * b_own) - vp ** 2 / (2 * b_lead)
    bad = ~ok(u)
    if not bad.any():
        return u, bad
    lo = np.full_like(u, A_MIN - 3.0); hi = u.copy()
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        good = ok(mid)
        lo = np.where(good, mid, lo); hi = np.where(good, hi, mid)
    return np.where(bad, lo, u), bad


def roll_idm(th, xp, vp, x0, v0):
    B, K = xp.shape
    x = np.empty((B, K)); v = np.empty((B, K)); a = np.zeros((B, K))
    x[:, 0], v[:, 0] = x0, v0
    for k in range(K - 1):
        a[:, k] = idm_acc(th, xp[:, k] - x[:, k] - L_EFF, v[:, k], vp[:, k])
        v[:, k + 1] = np.maximum(0.0, v[:, k] + a[:, k] * DT)
        x[:, k + 1] = x[:, k] + 0.5 * (v[:, k] + v[:, k + 1]) * DT
    a[:, -1] = a[:, -2]
    return x, v, a


def roll_cav(S: Setup, xp, vp, x0, v0, u0, ref=None, ideal=False):
    """CAV behind the rows of xp. ref = dict(x, v, a, gref, e) with arrays (B, K) and end index e (B,), or None for
    C0. Before e the CAV tracks the reference (Eq. trk); after e, or without ref, it runs C0. ideal=True follows the
    reference exactly before e (planning under ideal tracking)."""
    B, K = xp.shape
    x = np.empty((B, K)); v = np.empty((B, K)); a = np.zeros((B, K))
    x[:, 0], v[:, 0] = x0, v0
    u_prev = np.broadcast_to(np.asarray(u0, float), (B,)).copy()
    n_filter = np.zeros(B, int)
    e = ref["e"] if ref is not None else np.full(B, -1)
    for k in range(K - 1):
        g = xp[:, k] - x[:, k] - L_EFF
        u = np.clip(S.kg * (g - S.rs.sr(v[:, k])) + S.kd * (vp[:, k] - v[:, k]), A_MIN, A_MAX)
        on = k < e
        if ref is not None:
            ut = ref["a"][:, k] + S.kg * (g - ref["gref"][:, k]) + S.kv * (ref["v"][:, k] - v[:, k])
            u = np.where(on, ut, u)
        u = np.clip(u, u_prev - J_MAX * DT, u_prev + J_MAX * DT)     # actuator jerk limit
        u, bad = _filter(u, g, v[:, k], vp[:, k])
        n_filter += bad
        a[:, k] = u
        v[:, k + 1] = np.maximum(0.0, v[:, k] + u * DT)
        x[:, k + 1] = x[:, k] + 0.5 * (v[:, k] + v[:, k + 1]) * DT
        if ideal and ref is not None and on.any():
            a[on, k] = ref["a"][on, k]; v[on, k + 1] = ref["v"][on, k + 1]; x[on, k + 1] = ref["x"][on, k + 1]
        u_prev = a[:, k].copy()
    a[:, -1] = a[:, -2]
    return x, v, a, n_filter


def jerk(a):
    j = np.zeros_like(a)
    j[..., 1:] = np.diff(a, axis=-1) / DT
    return j


# ------------------------------------------------------------------------------------------------------- candidates
def candidates(S: Setup, xp, vp, x0, v0):
    """Admissible AKS transitions of one CAV from state (x0, v0) at the planning instant (column 0), behind the
    forecast predecessor (xp, vp) of shape (K,). Returns dict of arrays over candidates (C, K) and descriptors."""
    K = len(xp)
    rs, v2, dG = S.rs, S.v2, S.dG
    tt = np.arange(K) * DT
    Tg = np.arange(1.0, 45.0, 0.1)
    s_idx = np.arange(0, K, int(round(S.s_step / DT)))
    rows = []
    for name, sh in SHAPE_INFO.items():
        rho = sh["rho"]
        EF = s_idx[:, None] + Tg[None, :] / DT
        valid = EF < K - 1
        xpe = np.interp(np.where(valid, EF, K - 1).ravel(), np.arange(K), xp).reshape(EF.shape)
        f = x0 + v0 * s_idx[:, None] * DT + Tg[None, :] * (v0 + (v2 - v0) * rho) - (xpe - L_EFF - rs.sr(v2))
        f = np.where(valid, f, np.nan)
        cond = (f[:, :-1] < 0) & (f[:, 1:] >= 0)
        has = cond.any(1)
        idx = np.argmax(cond, 1)
        for r in np.where(has)[0]:
            i = idx[r]
            T = Tg[i] + (Tg[i + 1] - Tg[i]) * (-f[r, i]) / (f[r, i + 1] - f[r, i])
            rows.append((name, int(s_idx[r]), float(T)))
    if not rows:
        return None
    C = len(rows)
    X = np.empty((C, K)); V = np.empty((C, K)); A = np.zeros((C, K)); E = np.empty(C, int)
    for c, (name, si, T) in enumerate(rows):
        sh = SHAPE_INFO[name]
        s = si * DT
        u = np.clip((tt - s) / T, 0, 1)
        xs = x0 + v0 * s
        X[c] = np.where(tt < s, x0 + v0 * tt, xs + T * (v0 * u + (v2 - v0) * sh["H"](u)))
        after = tt > s + T
        X[c, after] = xs + T * (v0 + (v2 - v0) * sh["rho"]) + v2 * (tt[after] - s - T)
        V[c] = v0 + (v2 - v0) * sh["h"](u)
        inside = (tt >= s) & (tt <= s + T)
        A[c, inside] = (v2 - v0) * sh["dh"](u[inside]) / T
        E[c] = min(int(np.ceil((s + T) / DT)), K - 1)
    Jk = jerk(A)
    G = xp[None, :] - X - L_EFF
    D = G - rs.sr(V)
    mask = np.arange(K)[None, :] <= E[:, None]
    up = rs.rb(V) if dG is None else np.full_like(V, dG)
    up = np.maximum(up, float(D[0, 0]))                  # a lag already present at the release is not a violation
    viol_low = np.where(mask, D + rs.rb(V), np.inf).min(1)
    viol_up = np.where(mask, up - D, np.inf).min(1)
    viol_gap = np.where(mask, G - G_MIN, np.inf).min(1)
    amax = np.where(mask, A, 0)
    jm = np.where(mask, np.abs(Jk), 0)
    jm[:, 0] = 0                                       # the step from feedback is limited by the actuator
    feas = (viol_low >= -1e-6) & (viol_up >= -1e-6) & (viol_gap >= 0) & (amax.min(1) >= A_MIN) & \
           (amax.max(1) <= A_MAX) & (jm.max(1) <= J_MAX + 1e-6)
    keep = np.where(feas)[0]
    if len(keep) == 0:
        return None
    return dict(x=X[keep], v=V[keep], a=A[keep], e=E[keep], gref=G[keep],
                shape=[rows[i][0] for i in keep], s=np.array([rows[i][1] * DT for i in keep]),
                T=np.array([rows[i][2] for i in keep]))


def take(c, idx):
    out = {k: (val[idx] if isinstance(val, np.ndarray) else [val[i] for i in idx]) for k, val in c.items()}
    return out


# ---------------------------------------------------------------------------------------------- platoon evaluation
def default_platoon(S: Setup, kind="feedback"):
    """Whole horizon with no plans: CAVs on C0 ('feedback') or replaced by their human IDM ('hdv')."""
    K = len(S.t)
    X = np.empty((S.N + 1, K)); Vv = np.empty_like(X); A = np.zeros_like(X)
    X[0], Vv[0] = S.xL, S.vL
    A[0, 1:] = np.diff(S.vL) / DT
    x0, v0 = S.init
    for m in range(1, S.N + 1):
        xp, vp = X[m - 1][None], Vv[m - 1][None]
        if m in S.cav and kind == "feedback":
            x, v, a, _ = roll_cav(S, xp, vp, x0[m], v0[m], 0.0)
        else:
            x, v, a = roll_idm(S.idm_true[m - 1], xp, vp, x0[m], v0[m])
        X[m], Vv[m], A[m] = x[0], v[0], a[0]
    return X, Vv, A


def forecast_until(S: Setup, X, Vv, A, m_from, m_to, xp, vp, plans, models):
    """Forecast ranks m_from..m_to from the planning instant behind predecessor rows (xp, vp) (B, K'), with CAV
    plans {rank: ref or None}. Returns the trajectory rows of m_to and the summed cost of m_from..m_to."""
    k0 = S.k_plan
    B = xp.shape[0]
    Jsum = np.zeros(B)
    for m in range(m_from, m_to + 1):
        x0, v0, u0 = X[m, k0], Vv[m, k0], A[m, k0 - 1]
        if m in S.cav:
            ref = plans.get(m)
            x, v, a, _ = roll_cav(S, xp, vp, x0, v0, u0, ref=ref, ideal=True)
        else:
            x, v, a = roll_idm(models[m - 1], xp, vp, np.full(B, x0), np.full(B, v0))
        Jsum += cost(a, jerk(a), S.cj)
        xp, vp = x, v
    return xp, vp, a, Jsum


def own_plan(S: Setup, X, Vv, A, m, xp, vp, models, force=False):
    """The CAV's least-effort admissible transition behind the forecast predecessor (single row), compared with
    feedback over the same horizon; returns (ref or None, info)."""
    k0 = S.k_plan
    x0, v0, u0 = X[m, k0], Vv[m, k0], A[m, k0 - 1]
    xf, vf, af, _ = roll_cav(S, xp[None], vp[None], x0, v0, u0)
    J_fb = float(cost(af, jerk(af), S.cj)[0])
    c = candidates(S, xp, vp, x0, v0)
    if c is None:
        return None, dict(n_adm=0, J_fb=J_fb)
    ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
    xx, vv, aa, _ = roll_cav(S, np.repeat(xp[None], len(c["e"]), 0), np.repeat(vp[None], len(c["e"]), 0),
                             x0, v0, u0, ref=ref, ideal=True)
    J = cost(aa, jerk(aa), S.cj)
    i = int(np.argmin(J))
    info = dict(n_adm=len(J), J_fb=J_fb, J_best=float(J[i]), shape=c["shape"][i], s=float(c["s"][i]),
                T=float(c["T"][i]), T_min=float(c["T"].min()))
    if J[i] >= J_fb and not force:
        return None, info
    return {k: v[i:i + 1] for k, v in ref.items()}, info


def schedule(S: Setup, mode, models=None):
    """Return plans {rank: ref (1 row, from k_plan)} for 'independent', 'sequential' or 'joint'."""
    models = models or S.idm_fore
    X, Vv, A = S.base                                  # default platoon (CAVs on C0) gives the state at k_plan
    k0 = S.k_plan
    xL, vL = S.xL[None, k0:], S.vL[None, k0:]
    cavs = sorted(S.cav)
    plans, infos = {}, {}
    if mode in ("independent", "sequential", "sequential_commit"):
        for m in cavs:
            known = plans if mode != "independent" else {}
            xp, vp, _, _ = forecast_until(S, X, Vv, A, 1, m - 1, xL, vL, known, models)
            ref, info = own_plan(S, X, Vv, A, m, xp[0], vp[0], models, force=(mode == "sequential_commit"))
            infos[m] = info
            if ref is not None:
                plans[m] = ref
        return plans, infos
    # joint: tree over the CAVs; each branch keeps the predecessor rows and the accumulated cost
    xp, vp, _, J0 = forecast_until(S, X, Vv, A, 1, cavs[0] - 1, xL, vL, {}, models)
    branches = [dict(xp=xp[0], vp=vp[0], J=float(J0[0]), choice={})]
    bounds = cavs[1:] + [S.N + 1]
    for j, m in enumerate(cavs):
        nxt = []
        for br in branches:
            x0, v0, u0 = X[m, k0], Vv[m, k0], A[m, k0 - 1]
            c = candidates(S, br["xp"], br["vp"], x0, v0)
            opts = [None]
            if c is not None:
                ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
                nC = len(c["e"])
                xx, vv, aa, _ = roll_cav(S, np.repeat(br["xp"][None], nC, 0), np.repeat(br["vp"][None], nC, 0),
                                         x0, v0, u0, ref=ref, ideal=True)
                J = cost(aa, jerk(aa), S.cj)
                pick = set(np.argsort(J)[:3].tolist())
                for sh in set(c["shape"]):
                    ii = [i for i in range(nC) if c["shape"][i] == sh]
                    pick |= {min(ii, key=lambda i: c["s"][i]), max(ii, key=lambda i: c["s"][i])}
                pick = sorted(pick, key=lambda i: J[i])[:S.n_keep]
                opts += [{k: v[i:i + 1] for k, v in ref.items()} for i in pick]
            # evaluate every option of this CAV and the HDVs behind it up to the next CAV, side by side
            nO = len(opts)
            K1 = len(br["xp"])
            comb = dict(x=np.zeros((nO, K1)), v=np.zeros((nO, K1)), a=np.zeros((nO, K1)),
                        gref=np.zeros((nO, K1)), e=np.full(nO, -1))
            for o, ref in enumerate(opts):
                if ref is not None:
                    for k in ("x", "v", "a", "gref"):
                        comb[k][o] = ref[k][0]
                    comb["e"][o] = ref["e"][0]
            xq, vq, _, Js = forecast_until(S, X, Vv, A, m, bounds[j] - 1, np.repeat(br["xp"][None], nO, 0),
                                           np.repeat(br["vp"][None], nO, 0), {m: comb}, models)
            for o, ref in enumerate(opts):
                plans_b = dict(br["choice"]); plans_b[m] = ref
                nxt.append(dict(xp=xq[o], vp=vq[o], J=br["J"] + float(Js[o]), choice=plans_b))
        branches = nxt
        if S.beam and len(branches) > S.beam:              # beyond a few CAVs: keep the cheapest partial branches
            branches = sorted(branches, key=lambda b: b["J"])[:S.beam]
    best = min(branches, key=lambda b: b["J"])
    plans = {m: r for m, r in best["choice"].items() if r is not None}
    return plans, dict(n_leaves=len(branches), J_pred=best["J"])


def execute(S: Setup, plans):
    """True platoon from k_plan with the committed plans; history before k_plan from the default run."""
    X, Vv, A = (a.copy() for a in S.base)
    k0 = S.k_plan
    nfil = {}
    for m in range(1, S.N + 1):
        xp, vp = X[m - 1][None, k0:], Vv[m - 1][None, k0:]
        x0, v0 = X[m, k0], Vv[m, k0]
        if m in S.cav:
            x, v, a, nf = roll_cav(S, xp, vp, x0, v0, A[m, k0 - 1], ref=plans.get(m))
            nfil[m] = int(nf[0])
        else:
            x, v, a = roll_idm(S.idm_true[m - 1], xp, vp, np.array([x0]), np.array([v0]))
        X[m, k0:], Vv[m, k0:], A[m, k0:] = x[0], v[0], a[0]
    return X, Vv, A, nfil
