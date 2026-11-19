"""Joint scheduling by the recursion of Eq. (dp), Section 3.3, and a rollout with a plan-chain base policy: two
alternatives to WaveSched.rollout_cost, which predicts the vehicles behind the deciding CAV with the CAVs on feedback.

tree_cost(ws, ex, ev, ea, k, k1) returns, per option of the deciding CAV (rows of ex/ev/ea, full-length trajectories),
its own cost plus the cost of every vehicle behind it over [k, k1), where each CAV behind
  mode 'd' (exact recursion): chooses optimally among feedback and its WS_KD best candidates, so that the value of an
           option is the least cost over all combinations of the choices behind it (Eq. dp, enumerated as a tree);
  mode 'c' (plan-chain base policy): plans on its predecessor's predicted trajectory and commits its own best candidate
           if that is cheaper than feedback for itself, as each CAV does in the plan chain.
CAVs behind plan at the decision instant k from their predicted state, under ideal tracking (Eq. platoon); before k they
are on feedback. HDVs follow the scheduler's car-following model (ws.down). A CAV behind a CAV that chose a plan targets
the end speed of that plan; behind an HDV or a CAV on feedback it takes the first plateau of its predicted predecessor.
Used through the scenario suffixes "+WSd" and "+WSc" of platoon.run_platoon.
"""
import os

import numpy as np

import single_cav as pt
from lib import A_MAX, A_MIN, DT, J_MAX, L_EFF, cost, idm_acc
from rollout import jerk, roll_cav
from transition import candidates

CFG, RS = pt.CFG, pt.RS
KD = int(os.environ.get("WS_KD", "4"))                   # candidates kept per CAV behind in the exact recursion
OPEN = int(os.environ.get("WS_OPEN", "4"))              # CAVs scheduled together (the open set of Eq. width); CAVs
                                                         # further back are predicted on feedback
ETA_DEP = 1.05


def _seg(a, k0, k1):
    s = a[:, k0:k1]
    return np.sum(s ** 2, 1) * DT + pt.CJ * np.sum((np.diff(s, axis=1) / DT) ** 2, 1) * DT


def _follow(xp, vp, gap0, v0, k1, th=None):
    """Vehicles behind the rows of (xp, vp) from the start: IDM if th is given, else feedback with the jerk limit."""
    n = xp.shape[0]
    x = np.zeros((n, k1)); v = np.zeros((n, k1)); a = np.zeros((n, k1))
    x[:, 0], v[:, 0] = xp[:, 0] - L_EFF - gap0, v0
    up = np.zeros(n)
    for k in range(k1 - 1):
        g = xp[:, k] - x[:, k] - L_EFF
        if th is None:
            u = np.clip(CFG.kg * (g - RS.sr(v[:, k])) + CFG.kd * (vp[:, k] - v[:, k]), A_MIN, A_MAX)
            u = np.clip(u, up - J_MAX * DT, up + J_MAX * DT)
        else:
            u = idm_acc(th, g, v[:, k], vp[:, k])
        a[:, k] = u
        up = u
        v[:, k + 1] = np.maximum(v[:, k] + u * DT, 0.0)
        x[:, k + 1] = x[:, k] + 0.5 * (v[:, k] + v[:, k + 1]) * DT
    return x, v, a


def _tail(vo, k, H, K, tail, rank):
    """Speed after the planning horizon: hold the last speed, or relax to the Newell copy of the leader (as simulate_cav)."""
    pad = K - k - H
    if pad <= 0:
        return np.zeros(0)
    tt = np.arange(1, pad + 1) * DT
    if tail is None:
        return np.full(pad, vo[-1])
    vlk = tail[0]
    sh = int(round((rank - 1) * ETA_DEP / DT))
    idx = np.clip(k + H + np.arange(1, pad + 1) - sh, 0, len(vlk) - 1)
    cp = vlk[idx]
    return np.maximum(cp + (vo[-1] - cp[0]) * np.exp(-tt / 3.0), 0.0)


def _cav_options(xpk, vpk, xfb, vfb, afb, tgt_pred, k, K, k1, mode, tail, rank):
    """Options of one CAV behind the predecessor trajectory (xpk, vpk) (full length): feedback, and plans from its
    feedback state at k. Returns a list of (x, v, a, end_speed_or_None)."""
    opts = [(xfb, vfb, afb, None)]
    H = min(int(pt.H_S / DT), K - k)
    if H <= 60:
        return opts
    xf, vf = xpk[k:k + H], vpk[k:k + H]
    x0, v0, u0 = float(xfb[k]), float(vfb[k]), float(afb[k - 1]) if k > 0 else 0.0
    tgt = tgt_pred if tgt_pred is not None else pt.first_plateau(vf, H - 1, v0)
    if tgt is None or abs(tgt - v0) < CFG.gate:
        return opts
    c = candidates(CFG, tgt, xf, vf, x0, v0, a0=u0, j_rel=pt.J_REL)
    if c is None:
        return opts
    n = len(c["e"])
    ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
    px, pv, aa, _ = roll_cav(CFG, np.repeat(xf[None], n, 0), np.repeat(vf[None], n, 0), x0, v0, u0, ref=ref, ideal=True)
    J = cost(aa, jerk(aa), pt.CJ)
    _, _, a_fb, _ = roll_cav(CFG, xf[None], vf[None], x0, v0, u0)
    J_fb = float(cost(a_fb, jerk(a_fb), pt.CJ)[0])
    order = np.argsort(J)
    if mode == "c":
        order = order[:1] if J[order[0]] < J_fb else []
        if len(order):
            opts = []                                        # the plan chain commits its plan instead of feedback
    else:
        order = order[:KD]
    for b in order:
        vt = _tail(pv[b], k, H, K, tail, rank)
        xo, vo, ao = px[b], pv[b], aa[b]
        if len(vt):
            xo = np.r_[xo, xo[-1] + np.cumsum(0.5 * (np.r_[vo[-1], vt[:-1]] + vt) * DT)]
            ao = np.r_[ao, np.diff(np.r_[vo[-1], vt]) / DT]
            vo = np.r_[vo, vt]
        opts.append((np.r_[xfb[:k], xo][:K], np.r_[vfb[:k], vo][:K], np.r_[afb[:k], ao][:K], float(c["v"][b][c["e"][b] - 1])))
    return opts


def tree_cost(ws, ex, ev, ea, k, k1):
    mode = ws.mode
    K = ex.shape[1]
    n0 = ex.shape[0]
    root = np.arange(n0)                                     # root option of every branch
    path = _seg(ea, k, k1)                                   # own cost of the deciding CAV
    xp, vp = ex[:, :k1], ev[:, :k1]
    tgt = [None] * n0                                        # end speed of the predecessor's plan, if it chose one
    n_open = 1
    for j, item in enumerate(ws.down):
        is_cav, th, gap0 = item[:3]
        v0r = item[3] if len(item) > 3 else ws.v0                # 5.3: recorded initial speed of that vehicle
        rank = ws.rank + 1 + j
        if not is_cav:
            x, v, a = _follow(xp, vp, gap0, v0r, k1, th)
            path = path + _seg(a, k, k1 - 1)
            xp, vp, tgt = x, v, [None] * len(root)
            continue
        xfb, vfb, afb = _follow(xp, vp, gap0, v0r, k1)     # on feedback before k (and as its feedback option)
        n_open += 1
        if n_open > OPEN:                                    # outside the open set: feedback
            xp, vp = xfb, vfb
            path, tgt = path + _seg(afb, k, k1 - 1), [None] * len(root)
            continue
        nx, nv, na, nroot, npath, ntgt = [], [], [], [], [], []
        for i in range(len(root)):
            opts = _cav_options(xp[i], vp[i], xfb[i], vfb[i], afb[i], tgt[i], k, k1, k1, mode,
                                ws.tail if ws.tail is not None else None, rank)
            for xo, vo, ao, end in opts:
                nx.append(xo[:k1]); nv.append(vo[:k1]); na.append(ao[:k1])
                nroot.append(root[i]); npath.append(path[i]); ntgt.append(end)
        xp, vp, a = np.array(nx), np.array(nv), np.array(na)
        root, path, tgt = np.array(nroot), np.array(npath) + _seg(a, k, k1 - 1), ntgt
    out = np.full(n0, np.inf)
    np.minimum.at(out, root, path)                           # value of each root option: least cost over its subtree
    return out
