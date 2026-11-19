"""Shared pieces of the platoon experiment on the BJTU platoon data (exploratory round 1).

Units: m, s, m/s. Positions are antenna positions; the net gap is dx - L_EFF.
HDVs follow an IDM calibrated per vehicle (calibrate_idm.py). CAVs regulate to the r-Safety spacing with the feedback
law C0 (Eq. c0 of the paper) or track a committed AKS transition (Eq. trk) under a simple safety filter.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"                                 # outputs of the experiments
CFG_DIR = ROOT / "config"                               # calibrated parameters and tables
DATA = pathlib.Path(os.environ.get("W2S_DATA", ROOT / "data" / "bjtu"))   # BJTU platoon runs NS_run{1..5}.csv
DT, L_EFF = 0.1, 4.5
KMH = 1 / 3.6

# vehicle limits (as in the paper's parameter table)
A_MIN, A_MAX, J_MAX, G_MIN = -3.5, 2.0, 5.0, 2.0


# ----------------------------------------------------------------------------------------------------------------- data
def load_run(run):
    """t (s), ids (platoon order), x {id: m}, v {id: m/s}; gaps in a record are filled by linear interpolation."""
    df = pd.read_csv(DATA / f"NS_run{run}.csv")
    t = np.arange(len(df)) * DT
    x, v, ids = {}, {}, []
    for k in range(1, 26):
        vc, xc = ("1#V(km/h)", "1#S(m)") if k == 1 else (f"{k}#V", f"{k}#S")
        if vc not in df or df[vc].isna().all():
            continue
        vv, xx = df[vc].values.astype(float), df[xc].values.astype(float)
        i = np.arange(len(vv))
        for arr in (vv, xx):
            m = np.isfinite(arr)
            arr[~m] = np.interp(i[~m], i[m], arr[m])
        x[k], v[k] = xx, vv * KMH
        ids.append(k)
    return t, ids, x, v


# ------------------------------------------------------------------------------------------------------------------ IDM
def idm_acc(th, gap, v, vp):
    """IDM acceleration; th is a dict of arrays or floats (v0, T, s0, a, b)."""
    gap = np.maximum(gap, 0.1)
    s_star = th["s0"] + np.maximum(0.0, v * th["T"] + v * (v - vp) / (2 * np.sqrt(th["a"] * th["b"])))
    return th["a"] * (1 - (np.maximum(v, 0) / th["v0"]) ** 4 - (s_star / gap) ** 2)


def load_idm():
    return {int(k): v for k, v in json.loads((CFG_DIR / "idm_params.json").read_text())["per_vehicle"].items()}


# ------------------------------------------------------------------------------------------------------------- r-Safety
class RSafety:
    def __init__(self, d0, tau_h, r, a_ref):
        self.d0, self.tau_h, self.r, self.a_ref = d0, tau_h, r, a_ref

    def rb(self, v):
        return self.r * v * v / (2 * self.a_ref)

    def sr(self, v):
        return self.d0 + self.tau_h * v + self.rb(v)


def c0_acc(rs, gap, v, vp, kg, kd):
    return np.clip(kg * (gap - rs.sr(v)) + kd * (vp - v), A_MIN, A_MAX)


def safety_filter(u, gap, v, vp, b_lead=3.5, b_own=3.0):
    """Keep the gap above G_MIN when the predecessor brakes at b_lead and the CAV, after one step, at b_own."""
    v1 = np.maximum(v + u * DT, 0)
    g1 = gap + (vp - 0.5 * (v + v1)) * DT
    need = G_MIN + v1 ** 2 / (2 * b_own) - vp ** 2 / (2 * b_lead)
    if np.all(g1 >= need):
        return u
    # largest u that satisfies the condition, by bisection on a scalar (called per vehicle)
    lo, hi = A_MIN - 2.0, u
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        v1 = max(v + mid * DT, 0); g1 = gap + (vp - 0.5 * (v + v1)) * DT
        if g1 >= G_MIN + v1 ** 2 / (2 * b_own) - vp ** 2 / (2 * b_lead):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------------------------------------- AKS shapes
from numpy.polynomial import Polynomial as Poly  # noqa: E402

SHAPES = {
    "KS1": Poly([0., 0., 3., -2.]),
    "KS1A_early": Poly([0., 0., 6., -8., 3.]),
    "KS1A_late": Poly([0., 0., 0., 4., -3.]),
    "KS1A_center": Poly([0., 0., 0., 10., -15., 6.]),
}
SHAPE_INFO = {k: dict(h=h, H=h.integ(), dh=h.deriv(), d2h=h.deriv(2), rho=float(h.integ()(1.0))) for k, h in SHAPES.items()}


def cost(a, j, cj=0.1):
    """J = sum (a^2 + cj j^2) dt over the arrays (last axis = time)."""
    return np.sum(a ** 2 + cj * j ** 2, axis=-1) * DT


def platoon_ratio(d):
    """Platoon cost of every scenario relative to the all-HDV platoon, per change (index run, ev, kind). Uses J_A, the
    all-HDV cost over each scenario's own window (D76), when the file has it; otherwise the A rows."""
    g = d.groupby(["run", "ev", "kind", "scen"])
    J = g.J.sum().unstack("scen")
    if "J_A" in d.columns and d.J_A.notna().all():
        return J.div(g.J_A.sum().unstack("scen"))
    return J.div(J["A"], axis=0)


def rsd(x):
    """Robust standard deviation: the interquartile range over 1.349 (NaN for fewer than five values)."""
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349
