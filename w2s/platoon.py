"""Platoon experiment v2 (agreed 2026-10-18): recorded leader transition, 24 followers, HDVs on their calibrated IDM,
all starting in equilibrium at the leader's initial speed. Followers only react to their predecessors, so the platoon is
simulated vehicle by vehicle, exactly. CAVs use the single-vehicle controller of single_cav.py (D42-D43: smooth
plan-to-feedback handover, plan entry released at <= 1 m/s^3, target = first held plateau of the forecast).

Scenarios
  A   all HDV.
  B   CAVs run feedback only.
  C1  a CAV behind an HDV observes it (onboard sensing only): once the HDV has held a new level for 2 s
      (1-s smoothed |a| <= 0.3 m/s^2) it plans an AKS transition from its own state to that level and announces it;
      a CAV behind a CAV reads the announced plan (then "hold the end speed") as its forecast. Every CAV commits only if
      the plan is predicted cheaper than feedback for itself.
  C2  as C1, but a CAV behind an HDV that has a CAV right behind it commits whenever a plan exists (it plans for its
      follower, the platoon view of wave scheduling).
  C3  end speed relayed by the CAVs (first seen when a CAV's HDV predecessor holds a new level), with the observed
      departure step and rank; a CAV forecasts its predecessor as a KS1 to that speed starting on the departure front
      (ETA_DEP s per vehicle) and plans AKS on it (or on the announced plan of a CAV predecessor).
  C4  as C3 with the end speed and onset broadcast by the leader 0.3 s after its onset.
  C5  C4's information used to anticipate slowdowns within feedback (no plan): predecessor speed in the feedback law
      = current speed + min(0, forecast mean over the next 4 s - current speed), smoothed by two 1-s low-pass filters (D47).
  C6  as C5 with C3's relayed information.
  Level 0 (informed head; layouts with a CAV at rank 2):
  HF  the CAV at rank 2 knows the leader's whole future and anticipates slowdowns within feedback (F, 4 s); the other
      CAVs have no announced plan to read and run feedback.
  HP  plan chain: the CAV at rank 2 plans AKS on the leader's whole future and announces it; every CAV behind a CAV
      plans on its predecessor's announced plan and announces its own.
  HP1 only the CAV at rank 2 plans (on the leader's future); the other CAVs run feedback.
  L1F/L1FC CACC with the head's trajectory known: the first CAV of a string anticipates on the Newell forecast of its
           HDV predecessor, a CAV behind a CAV on the real future of the CAV ahead; L1FC kept inside the band (D70).
  HFchain  anticipation chain (fair control for the plan chain): every CAV of the string anticipates slowdowns with
           the exact future of its predecessor (the leader for rank 2, the CAV ahead otherwise); HFchainC capped at
           the upper spacing band.
  Level 1 (RSU knows the leader's whole trajectory; strings anywhere):
  L1P1 the first CAV of each string plans AKS on a Newell forecast of its HDV predecessor (newell_provider); the other
       CAVs run feedback.
  L1P  as L1P1, and every CAV behind a CAV plans on its predecessor's announced plan (plan chain).
  Level 2 (RSU broadcasts only the leader's new speed and its onset):
  L2P  the first CAV of each string forecasts its predecessor as a KS1 to the new speed starting on the departure front
       (template_provider) and plans on it; CAVs behind a CAV read announced plans (plan chain).
  Level 4 (errors): E_<spec> is L1P with an erroneous report of the leader (leader_info): E_t+1.0 reported 1 s late,
       E_v-0.5 new speed reported 0.5 m/s low, E_0 exact.
  V2V state sharing: V{k}P plan chain whose string head forecasts its predecessor from the current speed of the
       vehicle k positions ahead of it (v2v_provider; k = 'a' for all vehicles up to the leader); V{k}F every CAV
       anticipates slowdowns with that forecast (no plans).
Episode of each vehicle: from 5 s before its earliest departure over the compared scenarios (first |a| > 0.05 m/s^2 or
speed 0.2 m/s from the initial one) to 5 s after its latest settling (after it has made half of the leader's speed
change: speed within 0.5 m/s of its predecessor, |a| <= 0.3 m/s^2, and for a CAV the spacing deviation in the band,
held 3 s). Platoon cost = sum of the vehicles' episode costs.
Run: python platoon.py check|sweep [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import single_cav as pt
from eval_identify import leader_transitions
from lib import A_MAX, A_MIN, DT, J_MAX, L_EFF, RES, cost, idm_acc, load_idm, load_run
from lib import CFG_DIR  # noqa: E402  (calibrated parameters, ../config)
from rollout import _filter, jerk, roll_cav
from transition import POST, candidates

import json
import os
import schedule                       # noqa: E402
import random_driver as HN             # noqa: E402
import metrics as MT               # noqa: E402

_IDM = None


def _idm_per_vehicle():
    """IDM fitted to each recorded driver (config/idm_params.json). Derived from the recorded platoon and not
    distributed; only the replay of the recorded experiment needs it."""
    global _IDM
    if _IDM is None:
        _IDM = load_idm()
    return _IDM
CFG, RS = pt.CFG, pt.RS
# 2026-11-09 (all off by default, which reproduces the earlier results):
HDV_DET_POP = os.environ.get("HDV_DET_POP", "0") == "1"  # drivers that follow their model (no noise) use the population
                                                         # IDM, the model the CAVs forecast with (the deterministic part)
NOISE_PRM = os.environ.get("NOISE_PRM")                  # JSON file with the noise parameters sg, tg, sa, ta
if os.environ.get("PLATOON_PRE"):                        # start of the run before the head's change, s (20 s before)
    pt.K_ON = int(round(float(os.environ["PLATOON_PRE"]) / DT))
POST = float(os.environ.get("PLATOON_POST", POST))      # end of the run after the head's change, s (capped by the next)
N = int(os.environ.get("PLATOON_N", 24))                 # followers; ranks beyond 25 reuse the calibrated IDMs in turn
SCEN = tuple(os.environ.get("PLATOON_SCEN", "A,B,C3,C4").split(","))
T_PRIOR = 9.5                                            # typical transition duration of an IDM predecessor (D40)
HOP = 1                                                  # relay delay per CAV hop (steps)
WS_SCHED = os.environ.get("WS_SCHED", "d")             # D73: "d" exact recursion (Eq. dp), "rollout" the earlier scheme
WS_ANT = os.environ.get("WS_ANT", "0") == "1"           # D73: the schedule's options are AKS plans or feedback only
MARGIN = os.environ.get("MARGIN", "1") == "1"           # D75: planning margin for the forecast error behind HDVs
_MT = None
# Section 5.3 (2026-11-06): recorded initial states, the commitment gate, the base controller in the commit test, MPC.
INIT = None                       # {rank: (recorded gap, recorded speed)} at the start; a CAV starts at its own spacing
GATE = os.environ.get("PLATOON_GATE", "0") == "1"       # plan only before the predecessor has covered GO_FRAC of its
GO_FRAC = 0.1                                            # change (Section 3.2, as single_cav_runs.py)
BASE_TAIL = os.environ.get("PLATOON_BASE_TAIL", "0") == "1"   # commit test: the base controller (cacc, mpc) drives
_MPC = {"obj": None}                                     # before and after a plan, not the ACC of roll_cav


def pred_level(vp):
    """Causal speed of the predecessor and the level of its last plateau (as single_cav_runs.py)."""
    vs = np.convolve(np.r_[np.full(9, vp[0]), vp], np.ones(10) / 10, mode="valid")
    a = np.abs(np.r_[0.0, np.diff(vs)]) / DT
    lev = np.full(len(vp), vs[0])
    for k in range(len(vp)):
        lev[k] = vs[k - 19:k + 1].mean() if k >= 19 and a[k - 19:k + 1].max() <= 0.3 else (lev[k - 1] if k else vs[0])
    return vs, lev


def acc_pred(vp):
    """Acceleration of a forecast as a connected vehicle would broadcast it (1-s smoothed speed differentiated)."""
    vs = np.convolve(np.pad(vp, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]
    return np.r_[np.diff(vs) / DT, 0.0]


def mpc_obj():
    if _MPC["obj"] is None:
        from mpc import MPC
        w = json.loads((CFG_DIR / "mpc_weights.json").read_text())
        _MPC["obj"] = MPC(RS, **w)
    return _MPC["obj"]


def base_roll(law, xf, vf, x0, v0, u0, uf0=None):
    """Motion of a CAV on its base controller behind the forecast (xf, vf): acc (roll_cav), cacc (Ploeg, with the
    acceleration of the forecast as its feedforward) or mpc (re-solved every TM on the forecast, held after its end)."""
    H = len(xf)
    if law == "acc":
        xx, vv, aa, _ = roll_cav(CFG, xf[None], vf[None], x0, v0, u0)
        return xx[0], vv[0], aa[0]
    x, v, a = np.zeros(H), np.zeros(H), np.zeros(H)
    x[0], v[0] = x0, v0
    if law == "cacc":
        apf, uf = acc_pred(vf), (u0 if uf0 is None else uf0)
        for i in range(H - 1):
            g = xf[i] - x[i] - L_EFF
            hk = h_sr(v[i])
            q = PLOEG_KP * (g - RS.sr(v[i])) + PLOEG_KD * ((vf[i] - v[i]) - hk * (a[i - 1] if i else u0)) + \
                apf[max(i - PLOEG_LAG, 0)]
            uf = float(np.clip(uf + DT * (q - uf) / hk, A_MIN, A_MAX))
            a[i] = uf
            v[i + 1] = max(v[i] + uf * DT, 0.0)
            x[i + 1] = x[i] + 0.5 * (v[i] + v[i + 1]) * DT
    else:
        from mpc import TM, sample
        mp = mpc_obj()
        Nm = int(np.ceil(pt.H_S / TM))
        m_ = int(round(TM / DT))
        Hn = Nm * m_ + 1
        vx = np.r_[vf, np.full(Hn, vf[-1])]
        xe = np.r_[xf, xf[-1] + vf[-1] * DT * np.arange(1, Hn + 1)]
        aa, j0, vmax = u0, 0.0, v0
        for i in range(H - 1):
            if i % m_ == 0:
                idx = i + np.arange(Hn)
                j, _, _ = mp.solve(x[i], v[i], aa, sample(xe[idx], 0, Nm, Hn), sample(vx[idx], 0, Nm, Hn), vmax)
                j0 = float(j[0])
            aa = float(np.clip(aa + j0 * DT, A_MIN, A_MAX))
            a[i] = aa
            v[i + 1] = max(v[i] + aa * DT, 0.0)
            x[i + 1] = x[i] + 0.5 * (v[i] + v[i + 1]) * DT
            vmax = max(vmax, v[i + 1])
    a[-1] = a[-2]
    return x, v, a


def with_tail(law, xf, vf, px, pv, pa, e):
    """A planned trajectory (ideal tracking up to e) followed by the base controller instead of the ACC."""
    if law == "acc" or e >= len(xf) - 2:
        return px, pv, pa
    tx, tv, ta = base_roll(law, xf[e:], vf[e:], px[e], pv[e], pa[e - 1], uf0=pa[e - 1])
    return np.r_[px[:e], tx], np.r_[pv[:e], tv], np.r_[pa[:e], ta]


def margin_of(provider):
    """D75: planning margin (m) for a forecast that crosses n HDVs (provider.n_hdv; 0 for exact trajectories, announced
    plans and previews), from config/buffer_table.json (calibrated on runs 1, 2, 4, 5)."""
    global _MT
    n = getattr(provider, "n_hdv", 0) if provider is not None else 0
    if not MARGIN or n <= 0:
        return 0.0
    if _MT is None:
        f = CFG_DIR / os.environ.get("MARGIN_TABLE", "buffer_table.json")
        _MT = json.loads(f.read_text()) if f.exists() else {"n": [0], "m": [0.0]}
    return float(np.interp(n, _MT["n"], _MT["m"]))


PLAN_TGT = os.environ.get("PLAN_TGT", "1") == "1"       # D69: behind a CAV, target = end speed of its announced plan
CACC_P = float(os.environ.get("CACC_P", "4"))           # CACC (HFchain): preview averaged by the anticipation (s)
C4_LAT = 3                                               # C4/C5: broadcast this many steps after the leader's onset
SETTLE_RULE = os.environ.get("SETTLE_RULE", "last")       # end of a vehicle's window: "last" or "first" settling
FC_CHECK = float(os.environ.get("FC_CHECK", "2"))         # >0: commit only if the forecast made FC_CHECK s ago matched
FC_TOL = float(os.environ.get("FC_TOL", "0.5"))           #     the predecessor's observed speed within FC_TOL m/s, and
FC_DROP = float(os.environ.get("FC_DROP", "99"))
# 2026-11-10: exit rule. A committed CAV leaves its plan once the predecessor runs slower than the
# forecast the plan rests on by EXIT_DV m/s for EXIT_HOLD s while the gap is below the plan's reference, or once the CAV
# ahead announces that it has left its own plan; it then returns to its base controller (jerk <= J_MAX, as the base
# controller itself) and announces the exit, so the CAV behind leaves its plan too. Off by default.
EXIT_RULE = os.environ.get("EXIT_RULE", "0") == "1"
EXIT_DV = float(os.environ.get("EXIT_DV", "1.0"))
EXIT_HOLD = float(os.environ.get("EXIT_HOLD", "0.5"))
# 2026-11-10: EXIT_MODE=risk (default) triggers on danger instead of on a speed mismatch: the
# predecessor is extrapolated EXIT_H s ahead from its current speed and its deceleration (1-s mean, braking only, down
# to a stop), the CAV on its plan (then its end speed); the CAV leaves its plan once the predicted spacing deviation
# falls below the borrowing limit -R_b(v). EXIT_MODE=speed is the first version above.
# EXIT_MODE=trend (default, 2026-11-10): the predecessor is predicted as the forecast the plan rests on plus its present
# shortfall below that forecast and the trend of the shortfall (1-s change, growing only), so that braking the forecast
# expects does not count; the same danger test as risk.
# EXIT_MODE=pguard (2026-11-10): no exit. The predecessor is predicted as in trend, the CAV on its
# plan from where it is; the lower edge of the band guard is applied to the predicted spacing deviation over EXIT_H s,
# so the command fades towards the base controller as soon as the band edge is predicted to be reached, and returns to
# the plan when the danger passes. Nothing is announced: the CAV behind sees the motion itself.
EXIT_MODE = os.environ.get("EXIT_MODE", "trend")
# 2026-11-10: TRK_SAT=1 limits the tracking command to [A_MIN, A_MAX] like the base controller; PSI_FADE > D2 fades
# the plan's weight with the departure psi (Section 3.4) from 1 at psi = D2 to 0 at psi = PSI_FADE, so a plan the CAV
# has left and that cannot be rescheduled (forecast check) no longer commands it. Both off by default.
TRK_SAT = os.environ.get("TRK_SAT", "0") == "1"
PSI_FADE = float(os.environ.get("PSI_FADE", "0"))
REL_KEEP = os.environ.get("REL_KEEP", "0") == "1"   # 2026-11-10: replans after the buffer release keep rel_up
PG_MIN = os.environ.get("PG_MIN", "0") == "1"       # 2026-11-10: the predictive guard blends towards min(feedback, plan)
EXIT_H = float(os.environ.get("EXIT_H", "4.0"))


def _danger(xp_k, vp_k, ap_k, x_path, v_path, vq=None):
    """True if the spacing deviation, with the predecessor braking on at ap_k (<= 0) from (xp_k, vp_k) (or on the given
    speeds vq) and the CAV on (x_path, v_path) from now on, falls below the borrowing limit -R_b(v) within the paths'
    length."""
    n = len(x_path)
    t = np.arange(n) * DT
    if vq is None:
        vq = np.maximum(vp_k + ap_k * t, 0.0)
    xq = xp_k + np.r_[0.0, np.cumsum(0.5 * (vq[1:] + vq[:-1]) * DT)]
    dev = xq - x_path - L_EFF - RS.sr(v_path)
    return bool(np.any(dev < -RS.rb(v_path)))


def _pweight(xp_k, vq, x_path, v_path):
    """Predictive band guard (EXIT_MODE=pguard): the lower-edge weight of pt.band_weight evaluated on the predicted
    spacing deviation (predecessor on vq from xp_k, CAV on (x_path, v_path)), its least value over the horizon."""
    xq = xp_k + np.r_[0.0, np.cumsum(0.5 * (vq[1:] + vq[:-1]) * DT)]
    dev = xq - x_path - L_EFF - RS.sr(v_path)
    rb = RS.rb(v_path)
    return float(np.clip(np.min((dev + rb) / np.maximum(0.05, np.minimum(1.0, 0.5 * rb))), 0.0, 1.0))
ETA_DEP = 1.05                                           # departure front of all-HDV IDM platoons, s per vehicle
                                                         # (median over the 26 events, results/platoon_eta.json)
# Uncertainty study (2026-10-26, Section 2.4). All three are off by default, which reproduces the earlier results.
HDV_NOISE = os.environ.get("HDV_NOISE", "0") == "1"      # HDVs with the calibrated spacing drift and acceleration
                                                         # noise (random_driver.py, best row of config/noise_grid.csv)
FC_INJECT = os.environ.get("FC_INJECT", "0") == "1"      # IDM-chain forecasts carry the part of the measured timing
                                                         # error that the stochastic HDVs do not produce (forecast_error_sim.py)
FC_PROTECT = float(os.environ.get("FC_PROTECT", "0"))    # > 0: time protection b = z sd - mu of the measured error with
                                                         # z = FC_PROTECT (Eq. protnormal); "cantelli" alpha via z
_NZ = {"prm": None, "key": (0, 0, 0), "tab": None}       # noise parameters, (seed, run, event), error table
PROTECT_LOO = os.environ.get("PROTECT_LOO", "0") == "1"  # protection from the error measured on the other four runs
PROTECT_Z = float(os.environ.get("PROTECT_Z", "1.28"))   # z of the protected scenarios L1Pb / L1PCb (10 % risk)
# Recorded HDVs ahead (2026-10-27): layout rec<n> keeps ranks 2..n+1 as recorded (followers never react to the vehicles
# behind them, so the record stays valid whatever the CAVs do); four CAVs at ranks n+2..n+5 and 20 stochastic HDVs
# behind them (the platoon is lengthened to N = n + 24 followers).
REC_NS = (2, 4, 6)                                       # vehicle 8 is missing from runs 3-5 (rank 8 there is car 9)
# CACC (2026-10-26): Ploeg et al. (2011), h u' + u = kp e + kd e' + u_(i-1)(t - theta), on the r-Safety spacing, with
# the time gap h the local slope of S_r; published gains, V2V delay one step (0.1 s).
PLOEG_KP, PLOEG_KD, PLOEG_LAG = 0.2, 0.7, 1


def h_sr(v):
    """Slope of the r-Safety spacing dS_r/dv (s), the time gap of the CACC at speed v."""
    return max(RS.tau_h + RS.r * v / RS.a_ref, 0.1)


def _rng(tag, m):
    """Random stream of one vehicle (or one forecast) in one episode; the same in every scenario of the episode."""
    return np.random.default_rng([*_NZ["key"], tag, m])


def err_stats(src, kind, n):
    """Median and robust sd of the timing error e = t#_actual - t#_forecast through n HDVs (s), from
    config/forecast_error_table.json: 'recorded' (the experiment), 'inject' (recorded minus what the simulator produces)."""
    if _NZ["tab"] is None:
        _NZ["tab"] = json.loads((CFG_DIR / "forecast_error_table.json").read_text())
    t = _NZ["tab"][src][kind]
    return float(np.interp(n, t["n"], t["mu"])), float(np.interp(n, t["n"], t["sd"]))


def err_stats_loo(kind, n, run):
    """As err_stats('recorded'), from results/wave_forecast.csv without the run being simulated."""
    key = ("loo", run, kind)
    if key not in _NZ:
        from lib import rsd
        d = pd.read_csv(RES / "wave_forecast.csv")
        d = d[(d.run != run) & (d.kind == kind)]
        ns = [k for k in sorted(d.n.unique()) if np.isfinite(rsd(d[d.n == k].e_tsharp))]
        _NZ[key] = (ns, [float(d[d.n == k].e_tsharp.median()) for k in ns], [float(rsd(d[d.n == k].e_tsharp)) for k in ns])
    ns, mu, sd = _NZ[key]
    if n > ns[-1]:                                        # 2026-11-11 (source_distance_sweep): beyond the recorded chains (n <= 8),
        na = np.asarray(ns, float)                        # sd = sigma_tau sqrt(n) (Eq. errorn, rho = 0) and a mean that
        st = float(np.sum(np.asarray(sd) * np.sqrt(na)) / np.sum(na))      # grows with n, both fitted on n <= 8
        sl = float(np.sum(np.asarray(mu) * na) / np.sum(na ** 2))
        return sl * n, st * np.sqrt(n)
    return float(np.interp(n, ns, mu)), float(np.interp(n, ns, sd))


def protection(n):
    """Signed time shift (s) of the forecast that a plan is made on (Section 2.4): the predecessor's centre is taken b s
    earlier in a slowdown and b s later in a speed-up, the side on which the follower would borrow spacing."""
    z = PROTECT_Z if _POP.get("prot") else FC_PROTECT
    if z <= 0 or n <= 0:
        return 0.0
    kind = _POP["kind"]
    mu, sd = err_stats_loo(kind, n, _NZ["key"][1]) if PROTECT_LOO else err_stats("recorded", kind, n)
    return max(0.0, z * sd - mu) if kind == "decel" else -max(0.0, z * sd + mu)


def _gbuf(xf0, xf, gref):
    """Extra spacing (m) that the buffer leaves if the predecessor comes as forecast rather than b s early: the unshifted
    forecast's position less the shifted one, over the plan's horizon (>= 0 on the side the buffer is taken; 0 without
    a buffer)."""
    d = np.zeros(len(gref)) if xf0 is xf else np.asarray(xf0, float)[:len(gref)] - np.asarray(xf, float)[:len(gref)]
    return np.maximum(np.r_[d, np.full(len(gref) - len(d), d[-1] if len(d) else 0.0)], 0.0)


def _gbuf_r(released, rel_up, xf0, xf, gref):
    """Buffer kept by a new plan. Before the release: _gbuf on the unshifted forecast. After it (REL_KEEP=1, 2026-11-10):
    the lag the release plan still held (rel_up), so a later replan keeps the spacing the buffer added instead of
    reading it as a band violation (without it the band guard handed over to feedback, which closed that spacing by
    accelerating towards a braking predecessor and then braked at A_MIN)."""
    if released:
        return np.full(len(gref), rel_up) if REL_KEEP else _gbuf(xf, xf, gref)
    return _gbuf(xf0, xf, gref)


def _gerr(plan, i, g):
    """Spacing error of a plan (2026-11-09): the plan is made on the forecast shifted by the buffer
    b, and the spacing the buffer adds when the predecessor comes on time is kept, not closed. Zero between the planned
    spacing and the planned spacing plus that extra; below it, the error as before; above it, the excess only."""
    lo = plan["gref"][i]
    hi = lo + (plan["gbuf"][i] if "gbuf" in plan else 0.0)
    return g - lo if g < lo else (g - hi if g > hi else 0.0)


def _tshift(vq, s):
    """vq(t + s) on the same grid, held at the ends (s > 0: the forecast moved earlier)."""
    if s == 0:
        return vq
    k = np.arange(len(vq))
    return np.interp(np.clip(k + s / DT, 0, len(vq) - 1), k, vq)


def idm_of(m):
    return _idm_per_vehicle()[2 + (m - 2) % 24]


# ------------------------------------------------------------------------------------------------------------ layouts
def layouts():
    out = {}
    for n in ((4, 8) if N <= 24 else (8, 16)):
        sp = tuple(int(round(k * (N + 1) / (n + 1))) + 1 for k in range(1, n + 1))
        pr = tuple(sorted({r + d for r in (int(round(k * (N + 1) / (n // 2 + 1))) + 1 for k in range(1, n // 2 + 1))
                           for d in (0, 1)}))
        q = n // 4 if N > 24 else n // 2                 # strings: two strings (25 vehicles) or four strings (50)
        starts = (6, 16) if N <= 24 else (6, 18, 30, 42)
        st = tuple(r for s0 in starts for r in range(s0, s0 + q))
        fr = tuple(range(2, 2 + n))
        out.update({f"spread{n}": sp, f"pairs{n}": pr, f"strings{n}": st, f"front{n}": fr})
    if N <= 24:
        out.update({"mid4": tuple(range(10, 14)), "mid8": tuple(range(10, 18))})
        for h in (2, 3, 4, 6, 8, 10, 14, 18):           # one string of 4 CAVs, head at rank h (h - 2 HDVs ahead)
            out[f"s4@{h}"] = tuple(range(h, h + 4))
        out.update({"1x8": tuple(range(6, 14)), "2x4": (6, 7, 8, 9, 16, 17, 18, 19),
                    "4x2": (4, 5, 9, 10, 14, 15, 19, 20)})
        for k in (1, 2, 4, 6, 8):                         # D81 penetration: k CAVs behind the head / behind 4 HDVs
            out[f"fk{k}"], out[f"dk{k}"] = tuple(range(2, 2 + k)), tuple(range(6, 6 + k))
    for h in range(2, N - 2):                             # 2026-11-11 (source_distance_sweep): one string of 4 CAVs at any rank
        out.setdefault(f"s4@{h}", tuple(range(h, h + 4)))
    for n in REC_NS:                                      # recorded HDVs ahead (set_layout lengthens the platoon)
        out[f"rec{n}"] = tuple(range(n + 2, n + 6))
    return out


LAYOUTS = layouts()


# ------------------------------------------------------------------------------------------------------- forecasts
def observe_provider(xp, vp):
    """Onboard sensing of an HDV predecessor: a held new level (2 s, 1-s smoothed) becomes a constant forecast."""
    vs = np.convolve(vp, np.ones(10) / 10, mode="full")[:len(vp)]
    acc = np.abs(np.r_[0.0, np.diff(vs)]) / DT

    def f(k, H):
        if k < 40 or acc[k - 20:k + 1].max() > 0.3:
            return None
        L = float(vs[k - 20:k + 1].mean())
        tau = np.arange(H) * DT
        return np.full(H, L), xp[k] + L * tau, H - 1
    return f


_UG = np.linspace(0, 1, 1001)
_HG = 3 * _UG ** 2 - 2 * _UG ** 3


def template_provider(xp, vp, v_end, k_known, k_start=None):
    """End speed known from step k_known (relayed or broadcast). Before the predecessor leaves its old level it is
    forecast to start a KS1 transition to v_end, of typical duration T_PRIOR, at the step k_start predicted from the
    departure front (first departure observed upstream + ETA_DEP per vehicle); once it has left, the KS1 continues from
    its observed progress."""
    v_old = float(vp[0])

    def f(k, H):
        if k < k_known or abs(v_end - v_old) < 1e-6:
            return None
        tau = np.arange(H) * DT
        if abs(vp[k] - v_old) < 0.5:
            if k_start is None:
                return None
            u = np.clip((tau - (k_start - k) * DT) / T_PRIOR, 0, 1)
            vf = v_old + (v_end - v_old) * (3 * u ** 2 - 2 * u ** 3)
            xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
            return vf, xf, H - 1
        prog = float(np.clip((vp[k] - v_old) / (v_end - v_old), 0, 0.999))
        u0 = float(np.interp(prog, _HG, _UG))
        u = np.clip(u0 + tau / T_PRIOR, 0, 1)
        vf = v_old + (v_end - v_old) * (3 * u ** 2 - 2 * u ** 3)
        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
        return vf, xf, H - 1
    return f


def exact_provider(xp, vp):
    """Level 0: the predecessor's whole future is known (a CAV head announcing its trajectory, or an RSU)."""
    K = len(xp)

    def f(k, H):
        i = np.minimum(k + np.arange(H), K - 1)
        return vp[i], xp[i], H - 1
    return f


FORECAST = os.environ.get("FORECAST", "idm")             # D79: HDVs ahead forecast by the population IDM ("newell": D49)
_POP = {"th": None}                                      # population IDM of the run being simulated (set by run_platoon)


def idm_chain_provider(vl, xp, vp, m_pred):
    """D79: an RSU knows the head's whole trajectory; the HDV at rank m_pred is forecast by passing that trajectory through
    the m_pred - 1 HDVs with the population IDM (fitted on the other four runs; the model the schedule uses for the HDVs
    behind), from equilibrium, then shifted to start from the HDV's observed speed (2-s blend); positions are integrated
    from its observed position."""
    th = _POP["th"]
    K = len(vl)
    v0 = float(vl[0])
    xh = np.concatenate([[0.0], np.cumsum(0.5 * (vl[1:] + vl[:-1]) * DT)])
    xq, vq = xh, vl
    for _ in range(m_pred - 1):
        gap0 = (th["s0"] + v0 * th["T"]) / np.sqrt(1 - (v0 / th["v0"]) ** 4)
        x = np.zeros(K); v = np.zeros(K)
        x[0], v[0] = xq[0] - L_EFF - gap0, v0
        for k in range(K - 1):
            a = idm_acc(th, xq[k] - x[k] - L_EFF, v[k], vq[k])
            v[k + 1] = max(v[k] + a * DT, 0.0)
            x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
        xq, vq = x, v
    s_inj = 0.0
    if FC_INJECT and m_pred > 1:                           # the forecast centre is off by e ~ N(mu, sd) of the table
        mu, sd = err_stats("inject", _POP["kind"], m_pred - 1)
        s_inj = mu + sd * float(_rng(1, m_pred).standard_normal())
    vq0 = _tshift(vq, s_inj)

    def f(k, H, shift=0.0):
        vq = _tshift(vq0, shift) if shift else vq0
        tau = np.arange(H) * DT
        i = np.clip(k + np.arange(H), 0, K - 1)
        vf = np.maximum(vq[i] + (vp[k] - vq[min(k, K - 1)]) * np.exp(-tau / 2.0), 0.0)
        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
        return vf, xf, H - 1
    f.n_hdv = m_pred - 1
    f.protectable, f.s_inj, f.vq0 = True, s_inj, vq0
    return f


def newell_provider(vl, xp, vp, m_pred):
    if FORECAST == "idm" and _POP["th"] is not None and m_pred > 1:
        return idm_chain_provider(vl, xp, vp, m_pred)
    """Level 1: an RSU knows the leader's whole trajectory; the HDV at rank m_pred is forecast by Newell's shifted copy,
    v(t + tau) ~ v_leader(t + tau - (m_pred - 1) ETA_DEP), starting from its observed speed and blended into the copy
    over 2 s; positions are integrated from its observed position."""
    K = len(vl)
    sh = int(round((m_pred - 1) * ETA_DEP / DT))

    def f(k, H):
        tau = np.arange(H) * DT
        i = np.clip(k - sh + np.arange(H), 0, K - 1)
        copy = vl[i]
        copy = copy + (vp[k] - copy[0]) * np.exp(-tau / 2.0)
        vf = np.maximum(copy, 0.0)
        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
        return vf, xf, H - 1
    f.n_hdv = m_pred - 1                                   # HDVs between the head and the CAV (D75)
    return f


def distributed_provider(v_src, ann_src, xp, vp, n_shift, lat=1, x_src=None):
    """Distributed sources: the nearest CAV ahead (n_shift vehicles ahead of the predecessor) shares its past speed and
    its latest announced plan. D79: that known trajectory is passed through the n_shift - 1 HDVs in between and the
    predecessor with the population IDM (recomputed when a plan is announced and at least every second); with
    FORECAST="newell" the predecessor is Newell's copy of the CAV shifted by n_shift * ETA_DEP. Either way the forecast
    starts from the predecessor's observed speed (2-s blend) and positions are integrated from now."""
    K = len(v_src)
    sh = int(round(n_shift * ETA_DEP / DT))
    cache = {"key": None, "vq": None}

    def src(k, H):                                        # the source CAV's speed as known at step k, steps 0..k+H
        out = np.empty(k + H)
        out[:k + 1] = v_src[:k + 1]
        cur = [p for p in ann_src if p[0] <= k - lat]
        if cur:
            k0, _, vr, e = cur[-1]
            idx = np.arange(k + 1, k + H) - k0
            e = min(e, len(vr))
            out[k + 1:] = np.where(idx < e, vr[np.minimum(idx, e - 1)], vr[e - 1])
        else:
            out[k + 1:] = v_src[k]
        return out

    def chain(k, H):
        th = _POP["th"]
        s_ = src(k, H)
        n = len(s_)
        xs = np.empty(n)
        xs[:k + 1] = x_src[:k + 1]
        xs[k + 1:] = x_src[k] + np.cumsum(0.5 * (s_[k + 1:] + s_[k:-1]) * DT)
        v0 = float(s_[0])
        xq, vq = xs, s_
        for _ in range(n_shift):
            gap0 = (th["s0"] + v0 * th["T"]) / np.sqrt(1 - (v0 / th["v0"]) ** 4)
            x = np.zeros(n); v = np.zeros(n)
            x[0], v[0] = xq[0] - L_EFF - gap0, v0
            for i in range(n - 1):
                a = idm_acc(th, xq[i] - x[i] - L_EFF, v[i], vq[i])
                v[i + 1] = max(v[i] + a * DT, 0.0)
                x[i + 1] = x[i] + 0.5 * (v[i] + v[i + 1]) * DT
            xq, vq = x, v
        return vq

    def f(k, H, shift=0.0):                               # shift: buffer time b (2026-11-11, 6.3), as idm_chain_provider
        tau = np.arange(H) * DT
        if FORECAST == "idm" and _POP["th"] is not None and x_src is not None:
            n_ann = len([p for p in ann_src if p[0] <= k - lat and p[3] >= 0])
            if cache["key"] is None or cache["key"][0] != n_ann or k - cache["key"][1] >= 10:
                cache["key"], cache["vq"], cache["H"] = (n_ann, k), chain(k, H), H
                f.vq0 = cache["vq"]                       # unshifted forecast, read when the buffer is released
            vq = _tshift(cache["vq"], shift) if shift else cache["vq"]
            copy = vq[np.clip(k + np.arange(H), 0, len(vq) - 1)]
        else:
            s_ = src(k, H)
            copy = s_[np.clip(k - sh + np.arange(H), 0, k + H - 1)]
        vf = np.maximum(copy + (vp[k] - copy[0]) * np.exp(-tau / 2.0), 0.0)
        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
        return vf, xf, H - 1
    f.n_hdv = n_shift                                     # D75
    f.protectable, f.s_inj, f.vq0 = True, 0.0, None      # 2026-11-11 (6.3): buffer through the n_shift HDVs
    return f


def v2v_provider(v_src, xp, vp, k_ahead):
    """V2V state sharing: the current speed of the vehicle k_ahead positions in front of the predecessor is received;
    the predecessor is forecast as its Newell copy shifted by k_ahead * ETA_DEP (only up to now: preview k_ahead * ETA_DEP
    s, then held), starting from the predecessor's observed speed (2-s blend); positions integrated from now."""
    K = len(v_src)
    L = int(round(k_ahead * ETA_DEP / DT))

    def f(k, H):
        tau = np.arange(H) * DT
        idx = np.clip(k - L + np.arange(H), 0, k)
        copy = v_src[idx] + (vp[k] - v_src[max(k - L, 0)]) * np.exp(-tau / 2.0)
        vf = np.maximum(copy, 0.0)
        xf = xp[k] + np.concatenate([[0], np.cumsum(0.5 * (vf[1:] + vf[:-1]) * DT)])
        return vf, xf, H - 1
    f.n_hdv = k_ahead                                     # D75
    return f


def leader_info(vl, lead_v2, spec):
    """Erroneous report of the leader's speed: 't+1.0' reports it 1.0 s late (negative: early); 'v-0.5' reports a new
    speed 0.5 m/s lower, the error growing with the transition's progress; '0' exact."""
    if spec == "0":
        return vl
    kind, val = spec[0], float(spec[1:])
    K = len(vl)
    if kind == "t":
        idx = np.clip(np.arange(K) - int(round(val / DT)), 0, K - 1)
        return vl[idx]
    v_old = float(vl[0])
    prog = np.clip((vl - v_old) / (lead_v2 - v_old), 0, 1) if abs(lead_v2 - v_old) > 1e-6 else np.zeros(K)
    return vl + val * prog


def first_of(*provs):
    def f(k, H):
        for p in provs:
            r = p(k, H) if p is not None else None
            if r is not None:
                return r
        return None
    return f


def held_level(vp, v_old):
    """First step at which an observed vehicle has held a new level (>= 2 m/s from v_old) for 2 s, and that level."""
    vs = np.convolve(vp, np.ones(10) / 10, mode="full")[:len(vp)]
    acc = np.abs(np.r_[0.0, np.diff(vs)]) / DT
    for k in range(40, len(vp)):
        L = float(vs[k - 20:k + 1].mean())
        if acc[k - 20:k + 1].max() <= 0.3 and abs(L - v_old) >= 2.0:
            return k, L
    return None


def announce_provider(ann, lat=1):
    """Announced plans of a CAV predecessor: (commit step, x, v, end index) from the commit step on; after the end the
    predecessor is taken to hold its end speed."""
    def f(k, H):
        cur = [p for p in ann if p[0] <= k - lat]
        if not cur or cur[-1][3] < 0:                     # nothing announced, or the CAV ahead has left its plan
            return None
        k0, xr, vr, e = cur[-1]
        i = k - k0 + np.arange(H)
        e = min(e, len(vr)) - 1
        ve, xe = vr[e], xr[e]
        vf = np.where(i <= e, vr[np.minimum(i, e)], ve)
        xf = np.where(i <= e, xr[np.minimum(i, e)], xe + ve * (i - e) * DT)
        return vf, xf, H - 1

    def end_speed(k):                                     # the announced plan states where it ends
        cur = [p for p in ann if p[0] <= k - lat]
        if not cur or cur[-1][3] < 0:
            return None
        _, _, vr, e = cur[-1]
        return float(vr[min(e, len(vr)) - 1])
    def exited(k):                                        # the CAV ahead has announced that it left its plan
        cur = [p for p in ann if p[0] <= k - lat]
        return bool(cur) and cur[-1][3] < 0
    f.end_speed = end_speed
    f.exited = exited
    return f


WS_H = None                                              # wave scheduling looks ahead to the end of the window
WS_NCAND = 8                                             # plan candidates kept for the platoon evaluation


class WaveSched:
    """Platoon look-ahead for one CAV (wave scheduling, ch. 3). The vehicles behind the CAV are simulated from the
    start of the window with a prediction model: HDVs on their IDM (ideal tier: their own parameters; realistic tier:
    the population IDM), CAVs on feedback. rollout_cost returns, per option, the cost of the CAV and of every vehicle
    behind it over [k0, k1)."""

    def __init__(self, down, v0, x_gap0, tail=None):
        self.down, self.v0, self.x_gap0 = down, v0, x_gap0     # down: list of (is_cav, th, gap0)
        self.tail = tail          # (leader speed known to the scheduler, rank of the CAV) or None: hold the last speed

    def rollout_cost(self, ex, ev, ea, k0, k1):
        n = ex.shape[0]
        tot = np.zeros(n)
        seg = ea[:, k0:k1]
        tot += np.sum(seg ** 2, 1) * DT + pt.CJ * np.sum((np.diff(seg, axis=1) / DT) ** 2, 1) * DT
        xp, vp = ex[:, :k1], ev[:, :k1]
        for item in self.down:
            is_cav, th, gap0 = item[:3]
            v0r = item[3] if len(item) > 3 else self.v0        # 5.3: recorded initial speed of that vehicle
            x = np.zeros((n, k1)); v = np.zeros((n, k1)); a = np.zeros((n, k1))
            x[:, 0], v[:, 0] = xp[:, 0] - L_EFF - gap0, v0r
            up = np.zeros(n)
            for k in range(k1 - 1):
                g = xp[:, k] - x[:, k] - L_EFF
                if is_cav:
                    u = np.clip(CFG.kg * (g - RS.sr(v[:, k])) + CFG.kd * (vp[:, k] - v[:, k]), A_MIN, A_MAX)
                    u = np.clip(u, up - J_MAX * DT, up + J_MAX * DT)
                else:
                    u = idm_acc(th, g, v[:, k], vp[:, k])
                a[:, k] = u
                up = u
                v[:, k + 1] = np.maximum(v[:, k] + u * DT, 0.0)
                x[:, k + 1] = x[:, k] + 0.5 * (v[:, k] + v[:, k + 1]) * DT
            seg = a[:, k0:k1 - 1]
            tot += np.sum(seg ** 2, 1) * DT + pt.CJ * np.sum((np.diff(seg, axis=1) / DT) ** 2, 1) * DT
            xp, vp = x, v
        return tot


def roll_f(xf, vf, x0, v0, u0, P=4.0, tau_f=1.0):
    """Predicted trajectory of the anticipating feedback (simulate_cav_f) behind a forecast (xf, vf)."""
    H = len(xf)
    n = int(round(P / DT))
    x = np.zeros(H); v = np.zeros(H); a = np.zeros(H)
    x[0], v[0] = x0, v0
    off1 = off = 0.0
    up = u0
    for k in range(H - 1):
        g = xf[k] - x[k] - L_EFF
        raw = min(0.0, float(np.mean(vf[k:k + n + 1])) - float(vf[k]))
        off1 += (raw - off1) * DT / tau_f
        off += (off1 - off) * DT / tau_f
        u = float(np.clip(CFG.kg * (g - RS.sr(v[k])) + CFG.kd * (vf[k] + off - v[k]), A_MIN, A_MAX))
        u = float(np.clip(u, up - J_MAX * DT, up + J_MAX * DT))
        a[k] = u
        up = u
        v[k + 1] = max(v[k] + u * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    a[-1] = a[-2]
    return x, v, a


# --------------------------------------------------------------------------------------------------- CAV controller
def simulate_cav(xp, vp, x0, v0, provider=None, slack=0.0, ws=None, prov_f=None, exec_law="acc", ff=None, mpcx=None):
    """single_cav.simulate with a pluggable forecast; returns the acceleration and the announced plans.
    With ws (WaveSched) the choice at each decision instant is made on the platoon cost over WS_H s among: feedback,
    anticipating feedback (if prov_f is given) and the WS_NCAND best plan candidates.
    exec_law="cacc": the feedback underneath is the CACC (h u' + u = q, feedforward ff = the predecessor's
    acceleration, received PLOEG_LAG steps late) instead of the ACC; a committed plan is tracked by the same tracking
    law (Eq. trk) in both cases, and the band guard and the handover blend towards the CACC command."""
    K = len(xp)
    uf = 0.0
    x = np.zeros(K); v = np.zeros(K); a = np.zeros(K)
    x[0], v[0] = x0, v0
    plan, basis, last_plan, ann = None, None, -10 ** 6, []
    hand, n_h = None, int(round(pt.HAND_S / DT))
    ant, off1, off, nP = False, 0.0, 0.0, int(round(4.0 / DT))
    choices = []
    hist, n_chk = {}, int(round(FC_CHECK / DT))
    last_anchor, force_due, last_rp = -10 ** 6, False, -10 ** 6
    released, rel_up = False, 0.0                          # buffer released (2026-11-09), lag it may still hold
    n_slow = 0                                             # steps the predecessor has run slower than the plan's forecast
    marg = margin_of(provider)
    s_prot = protection(getattr(provider, "n_hdv", 0)) if getattr(provider, "protectable", False) else 0.0
    if GATE:
        vs_p, lev_p = pred_level(vp)
    if exec_law == "mpc":                                  # 5.3: MPC base, forecast mpcx["fc"](k) -> (x, v) from k on
        from mpc import TM as _TM, sample as _smp
        mp_, Nm = mpc_obj(), int(np.ceil(pt.H_S / _TM))
        m_ = int(round(_TM / DT)); Hn = Nm * m_ + 1
        mpcx["store"] = {}
    st = dict(b=s_prot, s_inj=getattr(provider, "s_inj", 0.0),n_re=0, n_back=0, n_rp=0, n_rp_fail=0, n_guard=0, n_rel=0, n_rel_fail=0, n_exit=0, n_exit_ahead=0, n_exit_hand=0, k_exit=np.nan, n_pg=0, n_psi=0, w_pg_min=1.0, k_pg=np.nan, end_dv=np.nan, end_dd=np.nan, v_end=np.nan, plan_dev=0.0, D_hist=np.zeros(200, int),
              wlog=np.full(K, np.nan))   # 2026-11-11: weight of the plan in the command at each step (nan: no plan)
    for k in range(K - 1):
        g = xp[k] - x[k] - L_EFF
        if provider is not None and k >= 10:
            H = min(int(pt.H_S / DT), K - k)
            fc = provider(k, H) if H > 60 else None
            if fc is not None:
                vf, xf, ie = fc
                if FC_CHECK > 0:
                    hist[k] = vf[:n_chk + 1].copy()
                    old = hist.get(k - n_chk)
                    valid = old is not None and len(old) == n_chk + 1 and \
                        float(np.abs(old - vp[k - n_chk:k + 1]).max()) <= FC_TOL
                else:
                    valid = True
                tgt = pt.first_plateau(vf, ie, v[k])
                xf0 = xf                                    # the unshifted forecast (2026-11-09, buffer dead band)
                if s_prot:                                  # time protection: plans, re-anchors and replans use the
                    vf, xf, _ = provider(k, H, shift=s_prot)   # forecast shifted by b; the check above does not
                end = provider.end_speed(k) if PLAN_TGT and hasattr(provider, "end_speed") else None
                if end is not None:                         # behind a CAV: its announced end speed, not a detected
                    tgt = end if abs(end - v[k]) >= CFG.gate else None   # plateau (D69: the plateau rule stops short)
                if tgt is None:
                    tgt = basis
                if tgt is not None:
                    due = (plan is None and (basis is None or abs(tgt - basis) > pt.RETGT)) or \
                          (plan is not None and abs(tgt - basis) > pt.RETGT) or \
                          (plan is None and pt.RETRY_S > 0 and k - last_plan >= int(round(pt.RETRY_S / DT))
                           and abs(v[k] - v0) < pt.RETRY_DEP) or force_due     # only before the CAV has departed
                    if plan is not None:
                        i = k - plan["k0"]
                        if i < plan["e"]:
                            D = np.hypot(_gerr(plan, i, g) / CFG.sig_g, (v[k] - plan["v"][i]) / CFG.sig_v)
                            due = due or D >= CFG.D2
                    if GATE and plan is None:               # not once the predecessor's change is under way
                        prog = (vs_p[k] - lev_p[k]) / (tgt - lev_p[k]) if abs(tgt - lev_p[k]) > 0.1 else 0.0
                        due = due and prog <= GO_FRAC
                    if due and valid and k - last_plan >= 20 and abs(tgt - v[k]) >= CFG.gate:
                        last_plan, basis, force_due = k, tgt, False
                        u0 = a[k - 1]
                        c = candidates(CFG, tgt, xf, vf, x[k], v[k], a0=u0, j_rel=pt.J_REL, margin=marg, up_extra=rel_up)
                        _, _, af_fb, _ = roll_cav(CFG, xf[None], vf[None], x[k], v[k], u0)
                        tl = BASE_TAIL and exec_law in ("cacc", "mpc")
                        if tl:                                  # the base controller as it would drive on the forecast
                            bx, bv, ba = base_roll(exec_law, xf, vf, x[k], v[k], u0, uf0=(uf if exec_law == "cacc" else None))
                            af_fb = ba[None]
                        J_fb = float(cost(af_fb, jerk(af_fb), pt.CJ)[0])
                        new = None
                        if ws is None:
                            if c is not None:
                                n = len(c["e"])
                                ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
                                _, _, aa, _ = roll_cav(CFG, np.repeat(xf[None], n, 0), np.repeat(vf[None], n, 0), x[k],
                                                       v[k], u0, ref=ref, ideal=True)
                                J = cost(aa, jerk(aa), pt.CJ)
                                b = int(np.argmin(J))
                                if tl:                              # the base controller after the plan
                                    _, _, ta = with_tail(exec_law, xf, vf, c["x"][b], c["v"][b], aa[b], int(c["e"][b]))
                                    J = J.copy(); J[b] = float(cost(ta[None], jerk(ta[None]), pt.CJ)[0])
                                if J[b] < J_fb * (1 + slack):
                                    new = dict(k0=k, vf=vf.copy(), x=c["x"][b], v=c["v"][b], a=c["a"][b], gref=c["gref"][b],
                                               e=int(c["e"][b]), gbuf=_gbuf_r(released, rel_up, xf0, xf, c["gref"][b]))
                                    ann.append((k, c["x"][b], c["v"][b], int(c["e"][b])))
                        else:                                       # wave scheduling: platoon cost decides
                            Hh = len(xf)
                            xfb, vfb, _, _ = roll_cav(CFG, xf[None], vf[None], x[k], v[k], u0)
                            opts = [("fb", None, bx, bv, ba) if tl else ("fb", None, xfb[0], vfb[0], af_fb[0])]
                            if prov_f is not None and WS_ANT:
                                fx, fv, fa = roll_f(xf, vf, x[k], v[k], u0)
                                opts.append(("F", None, fx, fv, fa))
                            if c is not None:
                                n = len(c["e"])
                                ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
                                px, pv, aa, _ = roll_cav(CFG, np.repeat(xf[None], n, 0), np.repeat(vf[None], n, 0),
                                                         x[k], v[k], u0, ref=ref, ideal=True)
                                J = cost(aa, jerk(aa), pt.CJ)
                                for b in np.argsort(J)[:WS_NCAND]:
                                    if tl:
                                        opts.append(("plan", int(b), *with_tail(exec_law, xf, vf, px[b], pv[b], aa[b],
                                                                               int(c["e"][b]))))
                                    else:
                                        opts.append(("plan", int(b), px[b], pv[b], aa[b]))
                            k1 = K                                  # look ahead to the end of the window: after the
                            pad = K - k - Hh                        # forecast the CAV is taken to hold its speed
                            def ext(o):
                                xo, vo, ao = o[2], o[3], o[4]
                                if pad > 0:
                                    tt = np.arange(1, pad + 1) * DT
                                    if ws.tail is None:
                                        vt = np.full(pad, vo[-1])
                                    else:                           # Newell copy of the leader known to the scheduler
                                        vlk, rk = ws.tail
                                        sh = int(round((rk - 1) * ETA_DEP / DT))
                                        idx = np.clip(k + Hh + np.arange(1, pad + 1) - sh, 0, len(vlk) - 1)
                                        cp = vlk[idx]
                                        vt = np.maximum(cp + (vo[-1] - cp[0]) * np.exp(-tt / 3.0), 0.0)
                                    xo = np.r_[xo, xo[-1] + np.cumsum(0.5 * (np.r_[vo[-1], vt[:-1]] + vt) * DT)]
                                    ao = np.r_[ao, np.diff(np.r_[vo[-1], vt]) / DT]
                                    vo = np.r_[vo, vt]
                                return xo, vo, ao
                            ext_o = [ext(o) for o in opts]
                            ex = np.stack([np.r_[x[:k], e_[0]] for e_ in ext_o])
                            evv = np.stack([np.r_[v[:k], e_[1]] for e_ in ext_o])
                            ea = np.stack([np.r_[a[:k], e_[2]] for e_ in ext_o])
                            Jp = schedule.tree_cost(ws, ex, evv, ea, k, k1) if getattr(ws, "mode", "r") in ("d", "c") \
                                else ws.rollout_cost(ex, evv, ea, k, k1)
                            best = opts[int(np.argmin(Jp))]
                            choices.append((k, best[0]))
                            ant = best[0] == "F"
                            if best[0] == "plan":
                                b = best[1]
                                new = dict(k0=k, vf=vf.copy(), x=c["x"][b], v=c["v"][b], a=c["a"][b], gref=c["gref"][b],
                                           e=int(c["e"][b]), gbuf=_gbuf_r(released, rel_up, xf0, xf, c["gref"][b]))
                                ann.append((k, c["x"][b], c["v"][b], int(c["e"][b])))
                        if plan is not None and new is None and n_h:
                            hand = (k, float(plan["v"][min(k - plan["k0"], plan["e"] - 1)]))
                        plan = new
                        if plan is not None:
                            hand = None
                if s_prot and not released and GATE and plan is not None and k - plan["k0"] < plan["e"] \
                        and float(np.max(plan.get("gbuf", 0.0))) > 0:
                    # 2026-11-09: the buffer is released once the predecessor's change is under
                    # way (GO_FRAC of it covered), since its timing is then observed. The forecast is moved to the
                    # observed timing (the unshifted forecast's own GO_FRAC point placed at now) and the rest of the
                    # transition is replanned from the realised state (pt.replan: same end speed, whole band, free end
                    # time, least effort), so the spacing the buffer added is spent rather than closed afterwards.
                    v_end = float(plan["v"][plan["e"] - 1])
                    prog = (vs_p[k] - lev_p[k]) / (v_end - lev_p[k]) if abs(v_end - lev_p[k]) > 0.1 else 0.0
                    if prog >= GO_FRAC:
                        released = True
                        vq0 = provider.vq0
                        k0p = plan["k0"]
                        p0 = (vq0[k0p:] - vq0[k0p]) / (v_end - vq0[k0p]) if abs(v_end - vq0[k0p]) > 0.1 else None
                        kc = int(np.argmax(p0 >= GO_FRAC)) + k0p if p0 is not None and np.any(p0 >= GO_FRAC) else None
                        new = None
                        if kc is not None:
                            s_prot = (kc - k) * DT              # from now on, plans use the observed timing
                            vfo, xfo, _ = provider(k, H, shift=s_prot)
                            rel_up = float(np.max(plan["gbuf"][k - plan["k0"]:]))
                            new = pt.replan(plan, k, xfo, vfo, x[k], v[k], a[k - 1], up_extra=rel_up)
                        if new is not None:
                            new["vf"], new["gup"] = vfo.copy(), rel_up    # the band of the release plan
                            plan = new
                            ann.append((k, new["x"], new["v"], new["e"]))
                            st["n_rel"] += 1
                        else:                                 # no admissible arc: keep the plan, its buffer still kept
                            st["n_rel_fail"] += 1
                        s_prot = s_prot if kc is not None else 0.0
                if pt.REANCHOR and plan is not None and k - plan["k0"] < plan["e"] and k - last_anchor >= 10:
                    i = k - plan["k0"]
                    D = np.hypot(_gerr(plan, i, g) / CFG.sig_g, (v[k] - plan["v"][i]) / CFG.sig_v)
                    if CFG.D1 <= D < CFG.D2:                  # D74: moderate deviation, re-anchor (Section 3.4)
                        last_anchor = k
                        new = pt.reanchor(plan, k, xf, vf, x[k], v[k], a[k - 1], margin=marg)
                        if new is not None:
                            new["vf"] = vf.copy()
                            plan = new
                            ann.append((k, new["x"], new["v"], new["e"]))
                            st["n_re"] += 1
                        else:
                            force_due = True
                            st["n_back"] += 1
                if pt.REPLAN and marg > 0 and plan is not None and k - plan["k0"] < plan["e"] and k - last_rp >= 10 and \
                        pt.leaves_margin(g, v[k], max(float(v[:k + 1].max()), float(np.max(plan["v"]))), marg):
                    last_rp = k                               # D77: execution entered the margin, replan an AKS
                    new = pt.replan(plan, k, xf, vf, x[k], v[k], a[k - 1])
                    if new is not None:                       # over the whole band, free end time, announced
                        new["vf"] = vf.copy()
                        ann.append((k, new["x"], new["v"], new["e"]))
                        st["n_rp"] += 1
                    else:                                     # no admissible arc: back to feedback
                        st["n_rp_fail"] += 1
                        if n_h:
                            hand = (k, float(plan["v"][min(k - plan["k0"], plan["e"] - 1)]))
                    plan = new
        vr = float(vp[k])
        if ant and prov_f is not None:                      # anticipating feedback chosen by wave scheduling
            fcf = prov_f(k, nP + 1)
            raw = min(0.0, float(np.mean(fcf[0][:nP + 1])) - vr) if fcf is not None else 0.0
            off1 += (raw - off1) * DT
            off += (off1 - off) * DT
            vr += off
        u = float(np.clip(CFG.kg * (g - RS.sr(v[k])) + CFG.kd * (vr - v[k]), A_MIN, A_MAX))
        if exec_law == "mpc":                               # MPC base: re-solved at every sample on its forecast
            fx, fv = mpcx["fc"](k)
            vx_ = np.r_[fv, np.full(Hn, fv[-1])][:Hn]
            xe_ = np.r_[fx, fx[-1] + fv[-1] * DT * np.arange(1, Hn + 1)][:Hn]
            vmx = max(float(v[:k + 1].max()), float(np.max(plan["v"])) if plan is not None else 0.0)
            jm, vpm, _ = mp_.solve(x[k], v[k], a[k - 1] if k else 0.0, _smp(xe_, 0, Nm, Hn), _smp(vx_, 0, Nm, Hn), vmx)
            u = float(np.clip((a[k - 1] if k else 0.0) + jm[0] * DT, A_MIN, A_MAX))
            xg = x[k] + np.cumsum(0.5 * (np.r_[v[k], vpm[:-1]] + vpm) * _TM)
            mpcx["store"][k] = (np.r_[x[k], xg], np.r_[v[k], vpm])   # broadcast: its predicted motion on the TM grid
        if exec_law == "cacc":                              # CACC feedback: one step of its command filter
            hk = h_sr(v[k])
            q = PLOEG_KP * (g - RS.sr(v[k])) + PLOEG_KD * ((vp[k] - v[k]) - hk * (a[k - 1] if k else 0.0)) + \
                float(ff[max(k - PLOEG_LAG, 0)])
            u = float(np.clip(uf + DT * (q - uf) / hk, A_MIN, A_MAX))
        jcap = J_MAX
        if plan is not None and FC_CHECK > 0:
            i = k - plan["k0"]
            if i < len(plan["vf"]) and abs(float(vp[k]) - float(plan["vf"][i])) > FC_DROP:
                if n_h:
                    hand = (k, float(plan["v"][min(i, plan["e"] - 1)]))
                plan = None
        w_pg = 1.0                                          # predictive band guard weight (EXIT_MODE=pguard)
        if EXIT_RULE and plan is not None:
            i = k - plan["k0"]
            if EXIT_MODE == "speed":
                slow = i < len(plan["vf"]) and float(plan["vf"][i]) - float(vp[k]) > EXIT_DV and \
                    i < plan["e"] and _gerr(plan, i, g) < 0
            else:                                           # risk: the realised predecessor against the plan
                nH = int(round(EXIT_H / DT))
                e_ = plan["e"]
                jj = i + np.arange(nH)
                vpl = np.where(jj < e_, plan["v"][np.minimum(jj, e_ - 1)], plan["v"][e_ - 1])
                xpl = np.where(jj < e_, plan["x"][np.minimum(jj, e_ - 1)],
                               plan["x"][e_ - 1] + plan["v"][e_ - 1] * (jj - e_ + 1) * DT)
                xpl = xpl + (x[k] - xpl[0])                 # from the CAV's realised position
                if EXIT_MODE == "risk":
                    ap = min(0.0, (float(vp[k]) - float(vp[max(k - 10, 0)])) / (min(k, 10) * DT)) if k else 0.0
                    slow = _danger(float(xp[k]), float(vp[k]), ap, xpl, vpl)
                else:                                       # trend: forecast + shortfall + its growth
                    vfp = plan["vf"]
                    jf = np.minimum(jj, len(vfp) - 1)
                    e0 = float(vp[k]) - float(vfp[min(i, len(vfp) - 1)])
                    if e0 < 0 and i >= 10:
                        e1 = float(vp[k - 10]) - float(vfp[min(i - 10, len(vfp) - 1)])
                        de = min(0.0, e0 - e1)              # change of the shortfall over the last second
                        vq = np.maximum(vfp[jf] + e0 + de * np.arange(nH) * DT, 0.0)
                        if EXIT_MODE == "pguard":
                            w_pg = _pweight(float(xp[k]), vq, xpl, vpl)
                            slow = False
                        else:
                            slow = _danger(float(xp[k]), float(vp[k]), 0.0, xpl, vpl, vq=vq)
                    else:
                        slow = False
            n_slow = n_slow + 1 if slow else 0
            ahead = EXIT_MODE != "pguard" and provider is not None and getattr(provider, "exited", None) is not None \
                and provider.exited(k)
            if n_slow * DT >= EXIT_HOLD or ahead:
                st["n_exit"] += 1
                st["n_exit_ahead"] += bool(ahead)
                if np.isnan(st["k_exit"]):
                    st["k_exit"] = k
                plan, hand, n_slow = None, None, 0         # straight back to the base controller
                ann.append((k, None, None, -1))            # and tell the CAV behind
        if EXIT_RULE and plan is None and hand is not None and (
                (EXIT_MODE == "speed" and hand[1] - float(vp[k]) > EXIT_DV and v[k] > vp[k]) or
                (EXIT_MODE != "speed" and _danger(float(xp[k]), float(vp[k]),
                                                  min(0.0, (float(vp[k]) - float(vp[max(k - 10, 0)])) / 1.0),
                                                  x[k] + hand[1] * np.arange(int(round(EXIT_H / DT))) * DT,
                                                  np.full(int(round(EXIT_H / DT)), hand[1])))):
            hand = None                                     # the plan has ended, but the predecessor runs below the end
            st["n_exit_hand"] += 1                          # speed the handover holds: straight to the base controller
        if plan is not None:
            i = k - plan["k0"]
            if i < plan["e"]:
                u_fb = u
                u = plan["a"][i] + CFG.kg * _gerr(plan, i, g) + CFG.kv * (plan["v"][i] - v[k])
                st["plan_dev"] = max(st["plan_dev"], abs(float(g - plan["gref"][i])))   # spacing off the plan (m)
                Dk = np.hypot(_gerr(plan, i, g) / CFG.sig_g, (v[k] - plan["v"][i]) / CFG.sig_v)
                st["D_hist"][min(int(Dk / 0.1), 199)] += 1      # distribution of D while tracking (Section 3.4)
                if pt.BAND_GUARD:                          # D71: last resort, same band rule as the band-limited CACC
                    vmx = max(float(v[:k + 1].max()), float(np.max(plan["v"])))
                    xb = float(plan["gbuf"][i]) if "gbuf" in plan else float(plan.get("gup", rel_up))   # buffer kept
                    st["n_guard"] += pt.band_weight(g, v[k], vmx, xb) < 0.999
                    w_psi = float(np.clip((PSI_FADE - Dk) / (PSI_FADE - CFG.D2), 0.0, 1.0)) if PSI_FADE > CFG.D2 else 1.0
                    if w_pg < 1.0 or w_psi < 1.0:          # predictive guard / departure fade: the lowest weight
                        if TRK_SAT:
                            u = float(np.clip(u, A_MIN, A_MAX))
                        wb = min(pt.band_weight(g, v[k], vmx, xb), w_pg, w_psi)
                        st["wlog"][k] = wb
                        # PG_MIN (2026-11-10): a safety hand-over never asks for more than the plan. Without it, feedback
                        # with a large gap (the spacing the plan kept) accelerated towards a braking predecessor.
                        u_to = min(u_fb, u) if (PG_MIN and w_pg < 1.0) else u_fb
                        u = wb * u + (1.0 - wb) * u_to
                        st["n_pg"] += w_pg < 1.0
                        st["n_psi"] += w_psi < 1.0
                        st["w_pg_min"] = min(st["w_pg_min"], w_pg)
                        if np.isnan(st["k_pg"]) and w_pg < 1.0:
                            st["k_pg"] = k
                    else:
                        st["wlog"][k] = pt.band_weight(g, v[k], vmx, xb)
                        u = pt.band_guard(u, u_fb, g, v[k], vmx, xb)
                if TRK_SAT:
                    u = float(np.clip(u, A_MIN, A_MAX))
            else:
                if n_h:
                    hand = (k, float(plan["v"][plan["e"] - 1]))
                st.update(end_dv=float(v[k] - vp[k]), end_dd=float(g - RS.sr(v[k])), v_end=float(plan["v"][plan["e"] - 1]))
                plan = None
        if plan is None and hand is not None:
            w = 1.0 - (k - hand[0]) / n_h
            if w <= 0:
                hand = None
            else:
                u_tr = CFG.kg * (g - RS.sr(v[k])) + CFG.kv * (hand[1] - v[k])
                u = float(np.clip(w * u_tr + (1 - w) * u, A_MIN, A_MAX))
                jcap = pt.HAND_J
        u = float(np.clip(u, a[k - 1] - jcap * DT, a[k - 1] + jcap * DT)) if k > 0 else u
        u = float(_filter(np.array([u]), np.array([g]), np.array([v[k]]), np.array([vp[k]]))[0][0])
        a[k] = u
        if exec_law == "cacc":                              # no wind-up: the filter state is the applied command
            uf = u
        v[k + 1] = max(v[k] + u * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    a[-1] = a[-2]
    simulate_cav.last_choices = choices
    st["k_commit"] = ann[0][0] if ann else np.nan          # D81: information time of the CAV (first commitment)
    simulate_cav.last_stats = st
    return x, v, a, ann


def simulate_cav_f(xp, vp, x0, v0, provider, P=4.0, tau_f=1.0, cap=False):
    """Feedback that anticipates slowdowns (F=P of D43) with a forecast instead of an exact preview: the predecessor
    speed in the feedback law is its current speed plus an anticipation offset, min(0, forecast mean over the next P s
    minus current speed), passed through two first-order low-pass filters of time constant tau_f (D47: the forecast
    jumps when the predecessor starts). The feedback itself keeps the jerk limit of B (j_max)."""
    K = len(xp)
    x = np.zeros(K); v = np.zeros(K); a = np.zeros(K)
    x[0], v[0] = x0, v0
    n = int(round(P / DT))
    off1 = off = 0.0
    for k in range(K - 1):
        g = xp[k] - x[k] - L_EFF
        fc = provider(k, n + 1) if provider is not None else None
        raw = min(0.0, float(np.mean(fc[0][:n + 1])) - float(vp[k])) if fc is not None else 0.0
        off1 += (raw - off1) * DT / tau_f
        off += (off1 - off) * DT / tau_f
        o_eff = off
        if cap:                                            # D53: fade the anticipation out at the upper spacing band
            o_eff *= float(np.clip((RS.rb(float(v[:k + 1].max())) - (g - RS.sr(v[k]))) / 1.0, 0.0, 1.0))
        vr = float(vp[k]) + o_eff
        u = float(np.clip(CFG.kg * (g - RS.sr(v[k])) + CFG.kd * (vr - v[k]), A_MIN, A_MAX))
        u = float(np.clip(u, a[k - 1] - J_MAX * DT, a[k - 1] + J_MAX * DT)) if k > 0 else u
        u = float(_filter(np.array([u]), np.array([g]), np.array([v[k]]), np.array([vp[k]]))[0][0])
        a[k] = u
        v[k + 1] = max(v[k] + u * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    a[-1] = a[-2]
    return x, v, a, []


def idm_follow(th, xp, vp, x0, v0):
    K = len(xp)
    x = np.zeros(K); v = np.zeros(K); a = np.zeros(K)
    x[0], v[0] = x0, v0
    for k in range(K - 1):
        a[k] = idm_acc(th, xp[k] - x[k] - L_EFF, v[k], vp[k])
        v[k + 1] = max(v[k] + a[k] * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    a[-1] = a[-2]
    return x, v, a


# ------------------------------------------------------------------------------------------------------- platoon run
def leader(run, i):
    t, ids, x, v = load_run(run)
    trs = leader_transitions(v[1])
    k_end = trs[i][0]
    k_next = trs[i + 1][0] if i + 1 < len(trs) else len(t)
    ks, ke = k_end - pt.K_ON, min(k_end + int(POST / DT), k_next - 50, len(t) - 1)
    return x[1][ks:ke] - x[1][ks], v[1][ks:ke], trs[i][2] - trs[i][1]


def recorded(run, i, n):
    """Ranks 2..n+1 of the recorded platoon over the leader's window, positions relative to the leader's start."""
    t, ids, x, v = load_run(run)
    trs = leader_transitions(v[1])
    k_end = trs[i][0]
    k_next = trs[i + 1][0] if i + 1 < len(trs) else len(t)
    ks, ke = k_end - pt.K_ON, min(k_end + int(POST / DT), k_next - 50, len(t) - 1)
    return {m: (x[ids[m - 1]][ks:ke] - x[1][ks], v[ids[m - 1]][ks:ke]) for m in range(2, n + 2)}


_N0 = N


def set_layout(lay):
    """Platoon length for a layout: N = n + 24 followers for rec<n> (20 HDVs behind the CAVs), the default otherwise."""
    global N
    N = int(lay[3:]) + 24 if lay.startswith("rec") else _N0


def _wsc(wsm):
    """Wave-scheduling choices of the CAV just simulated: one letter per decision (f feedback, F anticipate, p plan)."""
    return "".join({"fb": "f", "F": "F", "plan": "p"}[c[1]] for c in simulate_cav.last_choices) if wsm is not None else ""


def run_platoon(xl, vl, cav, scen, lead_v2=None, pop=None, rec=None):
    """Trajectories of ranks 1..N+1 (rank 1 = recorded leader) for one scenario; also the CAV plans.
    pop: population IDM of the run (HDV model of the schedule and, D79, of the forecast of the HDVs ahead).
    C3: every CAV relays the end speed it knows (its own observation of its predecessor, or relayed) to the CAVs behind;
    C4: the leader broadcasts its new speed 0.3 s after its onset.
    rec: recorded trajectories {rank: (x, v)} of the HDVs right behind the leader; the simulated vehicles behind them
    start in equilibrium at the last recorded vehicle's initial speed."""
    _POP["th"] = pop
    _POP["kind"] = "decel" if lead_v2 is not None and lead_v2 < vl[0] else "accel"
    if HDV_NOISE and _NZ["prm"] is None:
        if NOISE_PRM:
            _NZ["prm"] = {k: float(v) for k, v in json.loads(open(NOISE_PRM).read()).items() if k in ("sg", "tg", "sa", "ta")}
        else:
            best = pd.read_csv(CFG_DIR / "noise_grid.csv").iloc[0]
            _NZ["prm"] = {k: float(best[k]) for k in ("sg", "tg", "sa", "ta")}
    K, v0 = len(xl), float(vl[0])
    X, V, A = {1: xl}, {1: vl}, {1: np.r_[0.0, np.diff(vl)] / DT}
    ann, info = {}, {}
    MPX = {}                                               # 5.3: predictions broadcast by the MPC CAVs
    tier = None                                            # "<scen>+WSi" / "+WSr": wave scheduling, ideal / realistic
    if "+WS" in scen:
        scen, tier = scen.split("+WS")
    _POP["prot"] = scen in ("L1Pb", "L1PCb", "LDPb")      # time protection with z = PROTECT_Z (LDPb: 2026-11-11, 6.3)
    if _POP["prot"]:
        scen = scen[:-1]
    cap_c5 = scen in ("C5C", "C6C")                        # D70: C5/C6 with the anticipation kept inside the band
    if cap_c5:
        scen = scen[:-1]
    # knowledge carried down the CAVs: (step known, end speed, observed departure step, rank of that vehicle)
    know = (pt.K_ON + C4_LAT, float(lead_v2), pt.K_ON, 1) if scen in ("C4", "C5") else None
    stats, prev = {}, None
    for m in range(2, N + 2):
        if prev is not None and getattr(simulate_cav, "last_stats", None) is not None:
            stats[prev] = simulate_cav.last_stats            # execution statistics of the CAV just simulated
        simulate_cav.last_stats, prev = None, m
        if rec and m in rec:                               # recorded HDV ahead of the CAVs
            X[m], V[m] = rec[m]
            A[m] = np.r_[np.diff(V[m]) / DT, 0.0]
            A[m][-1] = A[m][-2]
            continue
        if rec and m == max(rec) + 1:
            v0 = float(V[m - 1][0])
        is_cav = m in cav and scen != "A"
        wsm = None
        th = pop if (HDV_DET_POP and not HDV_NOISE and not is_cav and pop is not None) else idm_of(m)
        gap0 = RS.sr(v0) if is_cav else (th["s0"] + v0 * th["T"]) / np.sqrt(1 - (v0 / th["v0"]) ** 4)
        x0 = X[m - 1][0] - L_EFF - gap0
        if INIT is not None:                               # 5.3: recorded speed and gap; a CAV at its own spacing
            g_r, v0 = INIT[m]
            gap0 = RS.sr(v0) if is_cav else g_r
            x0 = X[m - 1][0] - L_EFF - gap0
        if not is_cav:
            if HDV_NOISE:                                  # stochastic HDV, starting at its own equilibrium gap
                X[m], V[m] = HN.follow(th, X[m - 1], V[m - 1], x0 if INIT is not None else None, v0, _NZ["prm"],
                                       _rng(0, m))
                A[m] = np.r_[np.diff(V[m]) / DT, 0.0]
                A[m][-1] = A[m][-2]
            else:
                X[m], V[m], A[m] = idm_follow(th, X[m - 1], V[m - 1], x0, v0)
            continue
        wsm = None
        if tier is not None:
            down = []
            for r in range(m + 1, N + 2):
                thr = idm_of(r)
                g0 = RS.sr(v0) if r in cav else (thr["s0"] + v0 * thr["T"]) / np.sqrt(1 - (v0 / thr["v0"]) ** 4)
                if INIT is not None:                       # 5.3: the scheduler starts them where they are
                    g_r, v_r = INIT[r]
                    down.append((r in cav, thr if tier == "i" else pop, RS.sr(v_r) if r in cav else g_r, v_r))
                else:
                    down.append((r in cav, thr if tier == "i" else pop, g0))
            full_head = scen in ("HP", "HPC", "HPM", "HP1", "L1P", "L1PC", "L1P1", "LDP") or scen.startswith("E_")
            wsm = WaveSched(down, v0, None, tail=(leader_info(V[1], float(lead_v2), scen[2:]) if scen.startswith("E_")
                                                  else V[1], m) if full_head else None)
            wsm.mode = tier if tier in ("d", "c") else WS_SCHED   # D73: exact recursion of Eq. (dp) by default
            wsm.rank = m                                   # ("r"/"i" give the HDV model; schedule.py)
        if (m - 1) in cav:                                 # CACC feedforward: the predecessor's acceleration over V2V;
            ffm = A[m - 1]                                 # a connected head broadcasts its measured acceleration
        elif m - 1 == 1:                                   # (0.5-s causal mean); an HDV broadcasts nothing
            ffm = np.convolve(A[1], np.ones(5) / 5, mode="full")[:K]
        else:
            ffm = np.zeros(K)
        # Behind an HDV (no V2V) the CACC falls back to the r-Safety ACC (2026-10-27): without the feedforward, Ploeg's
        # law is string unstable at every gain on the r-Safety time gaps (string_check.py).
        cacc = "cacc" if (m - 1) in cav or m - 1 == 1 else "acc"
        if scen in ("PM", "HPM"):                          # 5.3: MPC base; the first CAV previews the head's whole
            def fc_m(k, m=m):                              # future, a CAV behind a CAV the plan it announced or, without
                if m - 1 == 1 or (m - 1) not in cav:      # one, the motion its MPC predicts (broadcast one step late)
                    return (X[m - 1][k:], V[m - 1][k:]) if m - 1 == 1 else \
                        (X[m - 1][k] + V[m - 1][k] * DT * np.arange(K - k), np.full(K - k, V[m - 1][k]))
                if scen == "HPM":                          # anchored at the predecessor's measured position
                    f = announce_provider(ann[m - 1])(k, K - k)
                    if f is not None:
                        return f[1] - f[1][0] + X[m - 1][k], f[0]
                st_ = MPX[m - 1].get(k - 1) or MPX[m - 1].get(k)
                from mpc import TM as _TM
                tg = np.arange(len(st_[0])) * _TM - DT    # prediction made one step earlier
                tq = np.arange(K - k) * DT
                vq = np.interp(tq, tg, st_[1])
                xq = np.interp(tq, tg, st_[0])
                beyond = tq > tg[-1]
                xq[beyond] = st_[0][-1] + st_[1][-1] * (tq[beyond] - tg[-1])
                return xq - xq[0] + X[m - 1][k], vq
            mpcx = {"fc": fc_m}
            if scen == "PM":
                prov = None
            else:
                prov = exact_provider(X[1], V[1]) if m - 1 == 1 else (announce_provider(ann[m - 1]) if (m - 1) in cav else None)
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm if scen == "HPM" else None,
                                                    prov_f=None, exec_law="mpc", mpcx=mpcx)
            MPX[m] = mpcx["store"]
            info[m] = ("MPC" if scen == "PM" else ("leader known" if m - 1 == 1 else "announced"), len(ann[m]),
                       _wsc(wsm) if scen == "HPM" else "")
            continue
        if scen == "PC":                                   # CACC (Ploeg et al. 2011) on the r-Safety spacing, no plans
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, None, exec_law=cacc, ff=ffm)
            info[m] = ("CACC", 0)
            continue
        if scen.startswith("E_"):                          # level 4: erroneous head information, plan chain
            vhat = leader_info(V[1], float(lead_v2), scen[2:])
            if (m - 1) not in cav:
                prov, src = newell_provider(vhat, X[m - 1], V[m - 1], m - 1), "head info with error"
            else:
                prov, src = announce_provider(ann[m - 1]), "announced"
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None))
            info[m] = (src, len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen.startswith("V") and scen[-1] in "PF" and len(scen) <= 4:   # V2V state sharing: V{k}P / V{k}F, k or 'a'
            kk = scen[1:-1]
            k_ahead = (m - 2) if kk == "a" else min(int(kk), m - 2)
            src_rank = (m - 1) - k_ahead
            prov = v2v_provider(V[src_rank], X[m - 1], V[m - 1], k_ahead) if k_ahead >= 1 else None
            if scen[-1] == "F":
                X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, prov)
                info[m] = (f"V2V k={k_ahead}, anticipate", 0)
            else:
                if (m - 1) in cav:
                    prov = announce_provider(ann[m - 1])
                X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None))
                info[m] = (f"V2V k={k_ahead}" if (m - 1) not in cav else "announced", len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen == "L2P":                                  # level 2: RSU broadcasts only the end speed and the onset
            if (m - 1) not in cav:
                k_start = int(round(pt.K_ON + ((m - 1) - 1) * ETA_DEP / DT))
                prov, src = template_provider(X[m - 1], V[m - 1], float(lead_v2), pt.K_ON + 3, k_start), "end + onset"
            else:
                prov, src = announce_provider(ann[m - 1]), "announced"
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None))
            info[m] = (src, len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen == "LDFC":                                 # D80: band-limited CACC with the same distributed sources
            if (m - 1) in cav:
                prov = exact_provider(X[m - 1], V[m - 1])
            else:
                up = [j for j in cav if j < m - 1]
                prov = distributed_provider(V[j := max(up)], ann.get(j, []), X[m - 1], V[m - 1], (m - 1) - j, x_src=X[j]) \
                    if up else (exact_provider(X[1], V[1]) if m - 1 == 1 else newell_provider(V[1], X[m - 1], V[m - 1], m - 1))
            X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, prov, P=CACC_P, cap=True)
            info[m] = ("anticipation, nearest source, capped", 0)
            continue
        if scen == "LDP":                                  # distributed sources: forecast from the nearest CAV ahead
            if (m - 1) in cav:
                prov, src = announce_provider(ann[m - 1]), "announced"
            else:
                up = [j for j in cav if j < m - 1]
                if up:
                    j = max(up)
                    prov, src = distributed_provider(V[j], ann[j], X[m - 1], V[m - 1], (m - 1) - j, x_src=X[j]), "nearest CAV"
                else:
                    prov, src = (exact_provider(X[1], V[1]), "leader known") if m - 1 == 1 else \
                        (newell_provider(V[1], X[m - 1], V[m - 1], m - 1), "Newell forecast")
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None))
            info[m] = (src, len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen in ("L1P", "L1PC", "L1P1"):                # level 1: RSU knows the leader; HDVs ahead forecast by Newell
            if (m - 1) not in cav:                         # L1PC: the plan chain of L1P executed by the CACC
                prov, src = (exact_provider(X[1], V[1]), "leader known") if m - 1 == 1 else \
                    (newell_provider(V[1], X[m - 1], V[m - 1], m - 1), "Newell forecast")
            elif scen in ("L1P", "L1PC"):
                prov, src = announce_provider(ann[m - 1]), "announced"
            else:
                prov, src = None, "none"
            X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None),
                                                    exec_law=cacc if scen == "L1PC" else "acc", ff=ffm)
            info[m] = (src, len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen in ("L1F", "L1FC") or scen.startswith("EFC_"):   # D70: CACC with the head's trajectory known (RSU):
            if (m - 1) in cav:                             # the first CAV of a string anticipates on the Newell forecast of
                prov = exact_provider(X[m - 1], V[m - 1])  # its HDV predecessor, a CAV behind a CAV on the real future of
            elif scen.startswith("EFC_"):                  # the CAV ahead (as HFchain); L1FC kept inside the band.
                prov = newell_provider(leader_info(V[1], float(lead_v2), scen[4:]), X[m - 1], V[m - 1], m - 1)
            elif m - 1 == 1:                               # EFC_<spec>: as L1FC with an erroneous report (as E_<spec>)
                prov = exact_provider(X[1], V[1])
            else:
                prov = newell_provider(V[1], X[m - 1], V[m - 1], m - 1)
            X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, prov, P=CACC_P, cap=(scen != "L1F"))
            info[m] = ("anticipation on the head's trajectory" + (" capped" if scen == "L1FC" else ""), 0)
            continue
        if scen in ("HFchain", "HFchainC"):                # D53: anticipation chain, each CAV knows its predecessor's
            prov = exact_provider(X[m - 1], V[m - 1]) if (m - 1 == 1 or (m - 1) in cav) else None   # real future
            X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, prov, P=CACC_P, cap=(scen == "HFchainC"))
            info[m] = ("anticipation chain" + (" capped" if scen == "HFchainC" else ""), 0)
            continue
        if scen in ("HF", "HP", "HPC", "HP1"):             # level 0: informed head (leader's future known to rank 2);
            if m - 1 == 1:                                 # HPC: the plan chain of HP executed by the CACC
                prov = exact_provider(X[1], V[1])
            elif (m - 1) in cav and scen in ("HP", "HPC"):
                prov = announce_provider(ann[m - 1])
            else:
                prov = None
            if scen == "HF":
                X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, prov)
            else:
                X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, ws=wsm, prov_f=(prov if wsm is not None and (m - 1) not in cav else None),
                                                        exec_law=cacc if scen == "HPC" else "acc", ff=ffm)
            info[m] = ("leader known" if m - 1 == 1 else ("announced" if prov else "none"), len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            continue
        if scen in ("C3", "C4", "C5", "C6"):
            obs = held_level(V[m - 1], v0)                 # own observation of the predecessor
            dep = np.where(np.abs(V[m - 1] - v0) > 0.5)[0]
            mine = (obs[0], obs[1], int(dep[0]), m - 1) if obs and len(dep) else None
            cand = [z for z in (know, mine) if z is not None]
            k_know = min(cand, key=lambda z: z[0]) if cand else None
            p_ann = announce_provider(ann[m - 1]) if (m - 1) in cav else None
            p_tpl = None
            if k_know:
                k_start = int(round(k_know[2] + ((m - 1) - k_know[3]) * ETA_DEP / DT))
                p_tpl = template_provider(X[m - 1], V[m - 1], k_know[1], k_know[0], k_start)
            if scen in ("C5", "C6"):                       # same information, used to anticipate within feedback
                X[m], V[m], A[m], ann[m] = simulate_cav_f(X[m - 1], V[m - 1], x0, v0, p_tpl, cap=cap_c5)
            else:
                X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, first_of(p_ann, p_tpl), ws=wsm, prov_f=(p_tpl if wsm is not None and (m - 1) not in cav else None))
            info[m] = (f"end known at {(k_know[0] - pt.K_ON) * DT:.1f} s" if k_know else "no end info", len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
            if k_know is not None:
                know = (k_know[0] + HOP, k_know[1], k_know[2], k_know[3])
            continue
        prov, slack, src = None, 0.0, "feedback"
        if scen in ("C1", "C2"):
            if (m - 1) in cav:
                prov, src = announce_provider(ann[m - 1]), "announced"
            else:
                prov, src = observe_provider(X[m - 1], V[m - 1]), "observed"
                if scen == "C2" and (m + 1) in cav:
                    slack, src = np.inf, "observed, plans for its follower"
        X[m], V[m], A[m], ann[m] = simulate_cav(X[m - 1], V[m - 1], x0, v0, prov, slack)
        info[m] = (src, len(ann[m]), _wsc(wsm) if 'wsm' in locals() else '')
    if prev is not None and getattr(simulate_cav, "last_stats", None) is not None:
        stats[prev] = simulate_cav.last_stats
    info["_stats"] = stats
    return X, V, A, info


def windows(runs, cav, dv_lead):
    """Per-vehicle episode windows over the compared scenarios (dict scen -> (X, V, A))."""
    out = {}
    for m in range(2, N + 2):
        k0s, k1s, ok = [], [], True
        for s, (X, V, A, _) in runs.items():
            v, a, K = V[m], A[m], len(V[m])
            dep = np.where((np.abs(a) > 0.05) | (np.abs(v - v[0]) > 0.2))[0]
            k0s.append(int(dep[0]) if len(dep) else K)
            half = np.where(np.abs(v - v[0]) >= 0.5 * abs(dv_lead))[0]
            if not len(half):
                ok = False; k1s.append(K); continue
            cond = (np.abs(v - V[m - 1]) <= 0.5) & (np.abs(a) <= 0.3)
            if m in cav and s != "A":
                d = X[m - 1] - X[m] - L_EFF - RS.sr(v)
                cond &= (d >= -RS.rb(v) - 1e-6) & (d <= RS.rb(max(v[0], v[-1])) + 1e-6)
            if SETTLE_RULE == "first":                         # first time the conditions hold for 3 s
                w = np.lib.stride_tricks.sliding_window_view(cond, 30).all(1)
                idx = np.where(w[half[0]:])[0]
                if len(idx):
                    k1s.append(half[0] + int(idx[0]) + 30)
                else:
                    ok = False; k1s.append(K)
            else:                                              # last settling: the conditions hold from there on
                bad = np.where(~cond[half[0]:])[0]
                s0 = half[0] + (int(bad[-1]) + 1 if len(bad) else 0)
                if s0 + 30 <= K:
                    k1s.append(s0 + 30)
                else:
                    ok = False; k1s.append(K)
        K = len(runs["A"][1][m])
        out[m] = (max(min(k0s) - 50, 0), min(max(k1s) + 50, K), ok)
    return out


def job(args):
    run, i, lay = args[:3]
    seed = args[3] if len(args) > 3 else 0
    _NZ["key"] = (seed, run, i)
    xl, vl, dv = leader(run, i)
    cav = LAYOUTS[lay]
    set_layout(lay)
    rec = recorded(run, i, int(lay[3:])) if lay.startswith("rec") else None
    from transition import CAL
    runs = {s: run_platoon(xl, vl, cav, s, lead_v2=vl[0] + dv, pop=CAL["pop_idm"][str(run)], rec=rec) for s in SCEN}
    ref = {s: runs[s] for s in ("A", "B") if s in runs}
    v_new, v_bar = float(vl[0] + dv), max(float(vl[0]), float(vl[0] + dv))
    sgn = 1.0 if dv > 0 else -1.0
    a_lead = np.r_[0.0, np.diff(vl)] / DT
    E_lead, peak_lead = MT.band_energy(a_lead, 0.04, 0.1), float(np.max(sgn * a_lead))
    rows = []
    for s, (X, V, A, info) in runs.items():
        win = windows({**ref, s: runs[s]}, cav, dv)          # D76: window over A, B and this scenario only
        st = info.get("_stats", {})
        for m in range(2, N + 2):
            k0, k1, ok = win[m]
            J = {}
            for r_, R in (("", runs[s]), ("_A", runs.get("A")), ("_B", runs.get("B"))):
                if R is None:
                    continue
                a = R[2][m][k0:k1]
                J[r_] = (float(np.sum(a ** 2) * DT), float(pt.CJ * np.sum(jerk(a[None])[0] ** 2) * DT))
            d = X[m - 1] - X[m] - L_EFF
            dev = d - RS.sr(V[m])
            c = st.get(m, {})
            fu = MT.fuel(V[m], A[m])                     # metrics.py, over the whole simulated window
            ex = MT.extent(None, X[m], V[m], float(vl[0]), v_new)
            fast = V[m] > 2.0
            rows.append(dict(run=run, ev=i, seed=seed, layout=lay, rank=m, scen=s, cav=m in cav,
                             behind_cav=(m - 1) in cav, settled=ok, k0=k0, k1=k1, Ja=J[""][0], Jj=J[""][1],
                             J=sum(J[""]), J_A=sum(J["_A"]) if "_A" in J else np.nan,
                             J_B=sum(J["_B"]) if "_B" in J else np.nan,
                             gap_min=float(d[k0:k1].min()), info=info.get(m, ("", 0))[0],
                             n_plans=info.get(m, ("", 0))[1], kind="decel" if dv < 0 else "accel",
                             ws=(info.get(m, ("", 0, ""))[2] if len(info.get(m, ("", 0))) > 2 else ""),
                             lag=float(dev.max() / RS.rb(v_bar)), low_m=float(max(0.0, -(dev + RS.rb(V[m])).min())),
                             plan_dev=c.get("plan_dev", np.nan), n_re=c.get("n_re", 0), n_rp=c.get("n_rp", 0), n_rp_fail=c.get("n_rp_fail", 0), n_back=c.get("n_back", 0), n_guard=c.get("n_guard", 0),
                             end_dv=c.get("end_dv", np.nan), end_dd=c.get("end_dd", np.nan),
                             tgt_err=(c.get("v_end", np.nan) - v_new),
                             b=c.get("b", np.nan), s_inj=c.get("s_inj", np.nan), k_commit=c.get("k_commit", np.nan),
                             fuel_g=fu["fuel_g"], dist_m=fu["dist_m"], tract=MT.tractive_energy(V[m], A[m]),
                             E1025=MT.band_energy(A[m], 0.04, 0.1), E_lead=E_lead, peak=float(np.max(sgn * A[m])),
                             peak_lead=peak_lead, dur=ex["dur"], dist_tr=ex["dist"], x_end=float(X[m][-1]),
                             gap_all=float(d.min()), tgap_all=float((d[fast] / V[m][fast]).min()) if fast.any() else np.nan))
    return rows


def jobs():
    out = []
    lays = os.environ.get("PLATOON_LAYOUTS")
    lays = lays.split(",") if lays else list(LAYOUTS)
    runs = [int(r) for r in os.environ.get("PLATOON_RUNS", "1,2,3,4,5").split(",")]   # D75: calibration subsets
    for run in runs:
        t, ids, x, v = load_run(run)
        out += [(run, i, lay) for i in range(len(leader_transitions(v[1]))) for lay in lays]
    seeds = os.environ.get("NOISE_SEEDS")                 # uncertainty study: one job per seed
    if seeds:
        out = [(*j, int(s)) for j in out for s in seeds.split(",")]
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        rows = job((3, 0, "strings4"))
        d = pd.DataFrame(rows)
        p = d.pivot_table(index="rank", columns="scen", values="J")
        p["cav"] = d.groupby("rank").cav.first()
        p["info"] = d[d.scen == "C2"].set_index("rank").info
        p["plans_C1"] = d[d.scen == "C1"].set_index("rank").n_plans
        p["plans_C2"] = d[d.scen == "C2"].set_index("rank").n_plans
        pd.set_option("display.width", 200)
        print(LAYOUTS)
        print(p.round(3).to_string())
        print("platoon totals:", d.groupby("scen").J.sum().round(2).to_dict(), " settled:", d.groupby("rank").settled.first().mean())
    else:
        nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        with ProcessPoolExecutor(nw) as ex:
            d = pd.DataFrame([r for rows in ex.map(job, jobs(), chunksize=1) for r in rows])
        d.to_csv(RES / os.environ.get("PLATOON_OUT", "platoon.csv"), index=False)
        print("rows", len(d))
