"""Tests of the preview-threshold proposition (decisions D33-D36), single CAV behind a recorded human predecessor.

E1  inputs per pair: the predecessor's main transition (plateau to plateau: 1-s smoothed |a| <= 0.3 m/s^2 held 1 s,
    new plateau >= 2 m/s from the old one), its duration T_p and speed change dv, the buffers R_b(v_new), R_b(v_old),
    the feedback cost J_fb (whole window and the main window [departure - 2 s, arrival + 5 s]),
    kappa = J_fb(main) / J_KS1(T_p) and the predictions:
        tau_b = R_b(v_new)/|dv|, tau_l = R_b(v_old)/|dv|        (both directions: lag uses the new speed's buffer)
        P*    = T_p/kappa - tau_b                               (length threshold, exact information)
        sig*  = kappa R_b(v_new)/T_p                            (accuracy threshold)
        floor = T_p / (kappa (T_p + tau_l + tau_b))             (best ratio on the main window)
E2  exact preview truncated at P s (then held), P in P_GRID, and the whole remaining future ("inf").
E3  the whole future with a speed bias b (+-), ramped in over the first 2 s: v_hat = v + b min(tau/2, 1).
The planner is the one of layers_single.py (D32) except the target: the first plateau of the forecast inside the true
preview (1-s smoothed |a| <= 0.3 m/s^2 held 2 s, >= 2 m/s from the CAV's speed; D28), AKS candidates in band D24, commit only if predicted cheaper than predicted feedback, re-plan if the target moves > 2 m/s.
Run: python single_cav.py e1|sweep [n_workers]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from eval_identify import leader_transitions
from lib import A_MAX, A_MIN, DT, J_MAX, L_EFF, RES, cost, load_run
from rollout import _filter, jerk, roll_cav
from transition import POST, PRE, Cfg, candidates

H_S, TGT_WIN, RETGT = 60.0, 20.0, 2.0
CFG = Cfg(s_step=1.0)
HAND_S, HAND_J = 2.0, 1.0       # plan -> feedback handover: blend over 2 s with |jerk| <= 1 m/s^3 (D42)
if os.environ.get("PROP_NO_HANDOVER"):
    HAND_S = 0.0
J_REL = None if os.environ.get("PROP_OLD_RELEASE") else HAND_J      # D43: plan entry released at <= 1 m/s^3
RETRY_S = float(os.environ.get("PROP_RETRY", "2"))   # an uncommitted CAV re-decides every RETRY_S s (0: never)
RETRY_DEP = float(os.environ.get("PROP_RETRY_DEP", "0.5"))   # ... while within RETRY_DEP m/s of its initial speed
FC_CHECK = float(os.environ.get("FC_CHECK", "2"))   # commit only if the forecast made FC_CHECK s ago matched the
FC_TOL = float(os.environ.get("FC_TOL", "0.5"))     # predecessor's observed speed within FC_TOL m/s (0: no check)
BAND_GUARD = os.environ.get("BAND_GUARD", "1") == "1"   # D71: a plan stops pulling the CAV out of the band
REANCHOR = os.environ.get("REANCHOR", "0") == "1"       # D74 (superseded by D77): re-anchoring on the deviation D
REPLAN = os.environ.get("REPLAN", "1") == "1"           # D77: replan an AKS when execution enters the margin
RETIME = 1.0                                             # D74: end time of a re-anchored arc within this of the plan (s)
TGT_HOLD_CHECK = not os.environ.get("PROP_OLD_TARGET")              # D43: skip over-/undershoot plateaus
S_GRID = (1, 2, 3, 4, 6, 8, 10)                                      # D43: short-preview smoothed feedback
if os.environ.get("PROP_LOW_OFF"):                                     # diagnostic: no lower spacing band
    CFG.low_off = True
RS = CFG.rs
CJ = CFG.cj
P_GRID = (2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40)
B_GRID = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6)
K_ON = int(PRE / DT)


def j_ks1(T, dv):
    return dv ** 2 * (6 / (5 * T) + CJ * 12 / T ** 3)


# ------------------------------------------------------------------------------------------------------------- E1
def main_transition(vp, lead_dv):
    """The predecessor's main transition between two plateaus (stretches with 1-s smoothed |a| <= 0.3 m/s^2 held
    at least 1 s): the first pair of plateaus, the old one ending after the leader's onset - 5 s, whose levels differ
    in the leader's direction by at least max(2 m/s, half the leader's change). Returns window-relative departure
    (end of the old plateau) and arrival (start of the new one), the two levels, the plateau lengths and whether the
    speed stays between the two levels (within 1 m/s) in between; None if there is no such pair."""
    sgn, need = np.sign(lead_dv), max(2.0, 0.5 * abs(lead_dv))
    vs = np.convolve(vp, np.ones(10) / 10, mode="same")
    a = np.gradient(vs) / DT
    n = len(vs)
    steady = np.array([np.all(np.abs(a[k:k + 10]) <= 0.3) if k + 10 <= n else False for k in range(n)])
    st = []                                                          # (first index, end index exclusive, level)
    k = 0
    while k < n:
        if steady[k]:
            k1 = k
            while k1 + 1 < n and steady[k1 + 1]:
                k1 += 1
            st.append((k, k1 + 10, float(vs[k:k1 + 10].mean())))
            k = k1 + 1
        else:
            k += 1
    for i, (s0, e0, L0) in enumerate(st):
        if e0 < K_ON - 50:
            continue
        for s1, e1, L1 in st[i + 1:]:
            if sgn * (L1 - L0) <= -1.0:                              # a plateau back the other way first: no pair
                break
            if sgn * (L1 - L0) >= need:
                # departure: last time within 0.5 m/s of the old level before the new plateau; arrival: first time
                # within 0.5 m/s of the new level after it (a creeping vehicle can leave the plateau test early)
                near0 = np.where(sgn * (vs[:s1] - L0) <= 0.5)[0]
                e0 = max(e0, int(near0[-1])) if len(near0) else e0
                near1 = np.where(sgn * (L1 - vs[e0:s1 + 1]) <= 0.5)[0]
                s1 = e0 + int(near1[0]) if len(near1) else s1
                seg = vs[e0:s1]
                mono = bool(len(seg) == 0 or (seg.min() >= min(L0, L1) - 1.0 and seg.max() <= max(L0, L1) + 1.0))
                return e0, s1, L0, L1, (e0 - s0) * DT, (e1 - s1) * DT, mono
    return None


# ------------------------------------------------------------------------------------------------------- simulation
def forecast(layer, k, vp, ap, K, H):
    tau = np.arange(H) * DT
    idx = np.minimum(k + np.arange(H), K - 1)
    if layer == "inf":
        return vp[idx], H * DT
    kind, val = layer.split("=")
    val = float(val)
    if kind == "P":                                                  # exact for P s, then held
        m = int(round(val / DT))
        return vp[np.minimum(k + np.minimum(np.arange(H), m), K - 1)], val
    if kind == "b":                                                  # whole future with a ramped-in speed bias
        return np.maximum(vp[idx] + val * np.minimum(tau / 2.0, 1.0), 0.0), H * DT
    raise ValueError(layer)



def first_plateau(vf, ie, v_now, hold=20):
    """Level of the first plateau of the forecast inside the true preview [0, ie]: 1-s smoothed |a| <= 0.3 m/s^2 held
    2 s, at least the planning gate away from the CAV's own speed (D28: targets are plateaus actually reached)."""
    if ie < hold + 10:
        return None
    vs = np.convolve(vf[:ie + 1], np.ones(10) / 10, mode="valid")
    a = np.abs(np.diff(vs)) / DT
    if len(a) < hold:
        return None
    w = np.lib.stride_tricks.sliding_window_view(a, hold).max(1)
    lev = np.lib.stride_tricks.sliding_window_view(vs[:-1], hold).mean(1)
    ok = np.where((w <= 0.3) & (np.abs(lev - v_now) >= CFG.gate))[0]
    if not TGT_HOLD_CHECK:
        return float(lev[ok[0]]) if len(ok) else None
    for i in ok:                                       # the level must hold: within 0.5 m/s for 8 s after the plateau
        seg = vs[i:min(i + hold + 80, len(vs))]       # (as far as the preview shows), so a dip that recovers is skipped
        if np.abs(seg - lev[i]).max() <= 0.5:
            return float(lev[i])
    return None


def simulate(ev, layer):
    xp, vp, K = ev["xp"], ev["vp"], len(ev["xp"])
    ap = np.concatenate([[0], np.diff(np.convolve(vp, np.ones(10) / 10, mode="full")[:K]) / DT])
    x = np.zeros(K); v = np.zeros(K); a = np.zeros(K)
    x[0], v[0] = ev["x0"], ev["v0"]
    plan, basis, last_plan, plans = None, None, -10 ** 6, []
    last_anchor, force_due, last_rp = -10 ** 6, False, -10 ** 6
    hand, n_h = None, int(round(HAND_S / DT))             # handover: (start step, reference speed at the plan's end)
    hist, n_chk = {}, int(round(FC_CHECK / DT))           # forecasts kept for the pre-commit check
    for k in range(K - 1):
        g = xp[k] - x[k] - L_EFF
        if layer.startswith("FC="):                      # F capped (D53): the anticipation fades out as the spacing
            n = int(round(float(layer[3:]) / DT))       # deviation approaches the upper band R_b(max speed so far)
            off = min(0.0, float(np.mean(vp[k:min(K, k + n + 1)])) - float(vp[k]))
            dev = g - RS.sr(v[k])
            upper = RS.rb(float(v[:k + 1].max()))
            off *= float(np.clip((upper - dev) / 1.0, 0.0, 1.0))
            u = float(np.clip(CFG.kg * dev + CFG.kd * (float(vp[k]) + off - v[k]), A_MIN, A_MAX))
            u = float(np.clip(u, a[k - 1] - J_MAX * DT, a[k - 1] + J_MAX * DT)) if k > 0 else u
            u = float(_filter(np.array([u]), np.array([g]), np.array([v[k]]), np.array([vp[k]]))[0][0])
            a[k] = u
            v[k + 1] = max(v[k] + u * DT, 0.0)
            x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
            continue
        if layer.startswith(("S=", "M=", "F=")):        # smoothed feedback: S = predecessor speed averaged over [t-P, t+P];
            n = int(round(float(layer[2:]) / DT))       # M = the smaller of its averages over [t, t+P] and [t-P, t] (never borrows);
            if layer[0] == "S":                         # F = the smaller of its current speed and its average over [t, t+P]
                vr = float(np.mean(vp[max(0, k - n):min(K, k + n + 1)]))
            elif layer[0] == "F":
                vr = min(float(vp[k]), float(np.mean(vp[k:min(K, k + n + 1)])))
            else:
                vr = min(float(np.mean(vp[k:min(K, k + n + 1)])), float(np.mean(vp[max(0, k - n):k + 1])))
            u = float(np.clip(CFG.kg * (g - RS.sr(v[k])) + CFG.kd * (vr - v[k]), A_MIN, A_MAX))
            u = float(np.clip(u, a[k - 1] - J_MAX * DT, a[k - 1] + J_MAX * DT)) if k > 0 else u
            u = float(_filter(np.array([u]), np.array([g]), np.array([v[k]]), np.array([vp[k]]))[0][0])
            a[k] = u
            v[k + 1] = max(v[k] + u * DT, 0.0)
            x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
            continue
        if layer != "feedback" and k >= 10:
            H = min(int(H_S / DT), K - k)
            if H > 60:
                vf, prev = forecast(layer, k, vp, ap, K, H)
                ie = min(int(prev / DT), H - 1)
                if FC_CHECK > 0:
                    hist[k] = vf[:n_chk + 1].copy()
                    old = hist.get(k - n_chk)
                    valid = old is not None and len(old) == n_chk + 1 and \
                        float(np.abs(old - vp[k - n_chk:k + 1]).max()) <= FC_TOL
                else:
                    valid = True
                tgt = first_plateau(vf, ie, v[k])
                if tgt is None:
                    tgt = basis
                if tgt is None:
                    pass
                else:
                  due = (plan is None and (basis is None or abs(tgt - basis) > RETGT)) or \
                        (plan is not None and abs(tgt - basis) > RETGT) or \
                        (plan is None and RETRY_S > 0 and k - last_plan >= int(round(RETRY_S / DT))
                         and abs(v[k] - v[0]) < RETRY_DEP) or force_due     # only before the CAV has departed
                  if plan is not None:
                      i = k - plan["k0"]
                      if i < plan["e"]:
                          D = np.hypot((g - plan["gref"][i]) / CFG.sig_g, (v[k] - plan["v"][i]) / CFG.sig_v)
                          due = due or D >= CFG.D2
                  if due and valid and k - last_plan >= 20 and abs(tgt - v[k]) >= CFG.gate:
                      last_plan, basis, force_due = k, tgt, False
                      xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
                      u0 = a[k - 1]
                      c = candidates(CFG, tgt, xf, vf, x[k], v[k], a0=u0, j_rel=J_REL)
                      _, _, af_fb, _ = roll_cav(CFG, xf[None], vf[None], x[k], v[k], u0)
                      J_fb = float(cost(af_fb, jerk(af_fb), CJ)[0])
                      new = None
                      if c is not None:
                          n = len(c["e"])
                          ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
                          _, _, aa, _ = roll_cav(CFG, np.repeat(xf[None], n, 0), np.repeat(vf[None], n, 0), x[k], v[k],
                                                 u0, ref=ref, ideal=True)
                          J = cost(aa, jerk(aa), CJ)
                          b = int(np.argmin(J))
                          if J[b] < J_fb:
                              new = dict(k0=k, x=c["x"][b], v=c["v"][b], a=c["a"][b], gref=c["gref"][b], e=int(c["e"][b]))
                              plans.append((k, float(c["s"][b]), float(c["T"][b]), c["shape"][b], tgt))
                      if plan is not None and new is None and n_h:
                          hand = (k, float(plan["v"][min(k - plan["k0"], plan["e"] - 1)]))
                      plan = new
                      if plan is not None:
                          hand = None
                if REANCHOR and plan is not None and k - plan["k0"] < plan["e"] and k - last_anchor >= 10:
                    i = k - plan["k0"]
                    D = np.hypot((g - plan["gref"][i]) / CFG.sig_g, (v[k] - plan["v"][i]) / CFG.sig_v)
                    if CFG.D1 <= D < CFG.D2:                  # D74: moderate deviation, re-anchor
                        last_anchor = k
                        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
                        new = reanchor(plan, k, xf, vf, x[k], v[k], a[k - 1])
                        if new is not None:
                            plan = new
                        else:
                            force_due = True
                if False and REPLAN and plan is not None and k - plan["k0"] < plan["e"] and k - last_rp >= 10 and \
                        leaves_margin(g, v[k], max(float(v[:k + 1].max()), float(np.max(plan["v"]))), 0.0):
                    last_rp = k                               # D77: no margin for an exact forecast, so no replanning
                    xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
                    new = replan(plan, k, xf, vf, x[k], v[k], a[k - 1])
                    if new is None and n_h:
                        hand = (k, float(plan["v"][min(k - plan["k0"], plan["e"] - 1)]))
                    plan = new
        u = float(np.clip(CFG.kg * (g - RS.sr(v[k])) + CFG.kd * (vp[k] - v[k]), A_MIN, A_MAX))
        jcap = J_MAX
        if plan is not None:
            i = k - plan["k0"]
            if i < plan["e"]:
                u_fb = u
                u = plan["a"][i] + CFG.kg * (g - plan["gref"][i]) + CFG.kv * (plan["v"][i] - v[k])
                if BAND_GUARD:
                    u = band_guard(u, u_fb, g, v[k], max(float(v[:k + 1].max()), float(np.max(plan["v"]))))
            else:
                if n_h:
                    hand = (k, float(plan["v"][plan["e"] - 1]))
                plan = None
        if plan is None and hand is not None:
            w = 1.0 - (k - hand[0]) / n_h
            if w <= 0:
                hand = None
            else:                                          # blend from tracking the plan's end speed to feedback
                u_tr = CFG.kg * (g - RS.sr(v[k])) + CFG.kv * (hand[1] - v[k])
                u = float(np.clip(w * u_tr + (1 - w) * u, A_MIN, A_MAX))
                jcap = HAND_J
        u = float(np.clip(u, a[k - 1] - jcap * DT, a[k - 1] + jcap * DT)) if k > 0 else u
        u = float(_filter(np.array([u]), np.array([g]), np.array([v[k]]), np.array([vp[k]]))[0][0])
        a[k] = u
        v[k + 1] = max(v[k] + u * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    a[-1] = a[-2]
    gap = xp - x - L_EFF
    return a, plans, float(gap[K_ON:].min()), float((gap - RS.sr(v))[K_ON:].min())


def band_guard(u_plan, u_fb, g, v_now, v_max, extra=0.0):
    """D71: the same band rule as the band-limited CACC. Within the last metre before the lag limit R_b(v_max) (v_max:
    the faster end of the planned transition, as in the band of Eq. (band)), and
    within the last half of the borrowing buffer before -R_b(v) (at most 1 m), the command fades from tracking the plan
    to plain feedback, which pulls the spacing back towards the r-Safety value."""
    w = band_weight(g, v_now, v_max, extra)
    return w * u_plan + (1.0 - w) * u_fb


def band_weight(g, v_now, v_max, extra=0.0):
    """Weight of the plan in band_guard (1: plain tracking). extra (2026-11-09): the spacing that a buffer time adds when
    the predecessor comes on time, kept rather than closed, so the lag limit is R_b(v_max) + extra."""
    dev = g - RS.sr(v_now)
    w = float(np.clip((RS.rb(v_max) + extra - dev) / 1.0, 0.0, 1.0))
    rb = RS.rb(v_now)
    return min(w, float(np.clip((dev + rb) / max(0.05, min(1.0, 0.5 * rb)), 0.0, 1.0)))


def leaves_margin(g, v_now, v_max, margin):
    """D77: True when the realised spacing deviation has entered the margin kept by the plan (or left the band when the
    margin is zero): the band minus the margin is what the plan was allowed to use."""
    dev = g - RS.sr(v_now)
    return dev > RS.rb(v_max) - margin + 1e-6 or dev < -RS.rb(v_now) + margin - 1e-6


def replan(plan, k, xf, vf, x_k, v_k, a_prev, up_extra=0.0):
    """D77: a new AKS from the realised state to the same end speed, over the whole band and with a free end time, the
    one of least effort. None if no admissible arc exists (the CAV then returns to feedback)."""
    v_end = float(plan["v"][plan["e"] - 1])
    if abs(v_end - v_k) < 0.05:
        return None
    c = candidates(CFG, v_end, xf, vf, x_k, v_k, a0=a_prev, j_rel=J_REL, up_extra=up_extra)
    if c is None:
        return None
    b = int(np.argmin(cost(c["a"], jerk(c["a"]), CJ)))
    return dict(k0=k, x=c["x"][b], v=c["v"][b], a=c["a"][b], gref=c["gref"][b], e=int(c["e"][b]))


def reanchor(plan, k, xf, vf, x_k, v_k, a_prev, margin=0.0):
    """D74 (Section 3.4, moderate deviation D1 <= D < D2): a new arc of the candidate family from the realised state to
    the same end state, its end within RETIME of the planned end. None if no admissible arc exists, in which case the
    CAV is sent back to the schedule (it decides again)."""
    v_end = float(plan["v"][plan["e"] - 1])
    c = candidates(CFG, v_end, xf, vf, x_k, v_k, a0=a_prev, j_rel=J_REL, margin=margin)
    if c is None:
        return None
    ok = np.where(np.abs((k + c["e"]) - (plan["k0"] + plan["e"])) * DT <= RETIME)[0]
    if not len(ok):
        return None
    J = cost(c["a"][ok], jerk(c["a"][ok]), CJ)
    b = int(ok[np.argmin(J)])
    return dict(k0=k, x=c["x"][b], v=c["v"][b], a=c["a"][b], gref=c["gref"][b], e=int(c["e"][b]))


def seg_cost(a, k0, k1):
    s = a[max(k0, 0):k1]
    return float(cost(s[None], jerk(s[None]), CJ)[0])


# ------------------------------------------------------------------------------------------------------------ pairs
def pairs():
    jobs = []
    for run in range(1, 6):
        t, ids, x, v = load_run(run)
        for i in range(len(leader_transitions(v[1]))):
            jobs += [(run, i, vid) for vid in ids[1:] if vid - 1 in ids]
    return jobs


def event(run, i, vid):
    t, ids, x, v = load_run(run)
    trs = leader_transitions(v[1])
    k_end = trs[i][0]
    k_next = trs[i + 1][0] if i + 1 < len(trs) else len(t)
    ks, ke = k_end - K_ON, min(k_end + int(POST / DT), k_next - 50, len(t) - 1)
    p = vid - 1
    return dict(xp=x[p][ks:ke] - x[1][ks], vp=v[p][ks:ke], x0=x[vid][ks] - x[1][ks], v0=v[vid][ks],
                lead_dv=trs[i][2] - trs[i][1])


def inputs(job):
    run, i, vid = job
    ev = event(run, i, vid)
    out = dict(run=run, ev=i, follower=vid, lead_dv=ev["lead_dv"])
    mt = main_transition(ev["vp"], ev["lead_dv"])
    a_fb, _, gmin, dmin = simulate(ev, "feedback")
    K = len(ev["vp"])
    out.update(J_fb=seg_cost(a_fb, K_ON, K), gap_fb=gmin, dev_fb=dmin)
    if mt is None:
        return out
    k_dep, k_arr, v_old, v_new, len_old, len_new, mono = mt
    dv = v_new - v_old
    Tp = (k_arr - k_dep) * DT
    rb_new, rb_old = RS.rb(v_new), RS.rb(v_old)
    m0, m1 = max(k_dep - 20, K_ON), min(k_arr + 50, K)
    J_main = seg_cost(a_fb, m0, m1)
    kappa = J_main / j_ks1(Tp, abs(dv))
    tb, tl = rb_new / abs(dv), rb_old / abs(dv)
    out.update(k_dep=k_dep, k_arr=k_arr, v_old=v_old, v_new=v_new, len_old=len_old, len_new=len_new, mono=mono, dv=dv, Tp=Tp,
               kind="decel" if dv < 0 else "accel", rb_new=rb_new, rb_old=rb_old, tau_b=tb, tau_l=tl,
               J_fb_main=J_main, kappa=kappa, P_star=Tp / kappa - tb, sig_star=kappa * rb_new / Tp,
               floor_main=Tp / (kappa * (Tp + tl + tb)),
               floor_full=(j_ks1(Tp + tl + tb, abs(dv)) + out["J_fb"] - J_main) / out["J_fb"])
    return out


def sweep(job):
    run, i, vid = job
    ev = event(run, i, vid)
    K = len(ev["vp"])
    rows = []
    layers = ["inf"] + [f"P={p}" for p in P_GRID] + [f"b={s * b}" for b in B_GRID for s in (1, -1)] + \
        [f"S={p}" for p in S_GRID]
    if os.environ.get("PROP_LAYERS"):
        layers = os.environ["PROP_LAYERS"].split(",")
    for layer in layers:
        a, plans, gmin, dmin = simulate(ev, layer)
        rows.append(dict(run=run, ev=i, follower=vid, layer=layer, J=seg_cost(a, K_ON, K), n_commit=len(plans),
                         gap_min=gmin, dev_min=dmin,
                         T_first=plans[0][2] if plans else np.nan, s_first=plans[0][1] if plans else np.nan,
                         k_first=plans[0][0] if plans else np.nan, tgt_first=plans[0][4] if plans else np.nan,
                         T_max=max(p[2] for p in plans) if plans else np.nan, a=a.astype(np.float32)))
    return rows


if __name__ == "__main__":
    mode = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = pairs()
    print("pairs", len(jobs))
    with ProcessPoolExecutor(nw) as ex:
        if mode == "e1":
            d = pd.DataFrame(list(ex.map(inputs, jobs, chunksize=4)))
            d.to_csv(RES / "prop_e1.csv", index=False)
        else:
            out = [r for rows in ex.map(sweep, jobs, chunksize=1) for r in rows]
            acc = {(r["run"], r["ev"], r["follower"], r["layer"]): r.pop("a") for r in out}
            pd.DataFrame(out).to_csv(RES / "prop_sweep.csv", index=False)
            np.save(RES / "prop_sweep_acc.npy", acc, allow_pickle=True)
