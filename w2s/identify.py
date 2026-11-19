"""T5: when does a follower know where its predecessor's transition ends?

Criterion B (main): after the predecessor departs from its plateau, fit an AKS speed profile v1 + dv F((t - t0)/T) to
the speed observed so far (shapes KS1 and KS1A with rho 0.4, 0.5, 0.6; dv in closed form for each T; T on a grid with
a lognormal prior on T given |dv| learnt from the other runs). The follower knows the end speed v1 + dv at the first
time the predicted end speed has stayed within +-1 km/h over the last 1 s.
Criterion C (lower bound): the predecessor is on a new plateau, |a| <= 0.3 m/s^2 held for 1 s.
Only information available at time t is used (causal smoothing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib import DT, KMH, SHAPE_INFO

TRANS_CSV = Path(__file__).resolve().parent.parent / "data" / "e1a2_transitions.csv"   # leader transitions (AKS_DP_Method e1a2)
T_GRID = np.arange(1.0, 40.0, 0.2)
SIG_V = 0.3                    # speed noise (m/s) of the likelihood
B_DEPART = 2.0                 # departure from the plateau that starts the fit (m/s; D3 of the decision log)
FLAT_SD, FLAT_W = 0.5, 5.0      # a reference plateau: speed std below FLAT_SD over FLAT_W seconds
V_MAX = 22.0
STABLE_KMH, STABLE_S = 1.0, 1.0
A_FLAT, FLAT_S = 0.3, 1.0


def duration_prior(exclude_run):
    d = pd.read_csv(TRANS_CSV)
    d = d[(d.V1 > 1) & (d.V2 > 1) & (d.run != exclude_run)]
    x, y = np.log(np.abs(d.V2 - d.V1)), np.log(d.T_s)
    b = np.polyfit(x, y, 1)
    return dict(a=b[1], b=b[0], sd=float((y - np.polyval(b, x)).std()))


def causal_ma(v, n):
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    out[:n - 1] = c[1:n] / np.arange(1, n)
    return out


def departure(v, k_from, k_max, b=B_DEPART):
    """Departure from the most recent flat plateau: the reference level is the mean of the latest FLAT_W-s window
    whose speed std is below FLAT_SD (updated while the vehicle is flat); the departure kd is the first index in
    [k_from, k_max) where the causal 0.5-s mean differs from that level by b; k0 is the last index before kd within
    0.5 m/s of the level."""
    vm = causal_ma(v, 5)
    W = int(round(FLAT_W / DT))
    level, k_flat = None, None
    for k in range(max(k_from - 300, W), k_max):
        win = v[k - W:k]
        if win.std() < FLAT_SD:
            level, k_flat = float(win.mean()), k
        if k >= k_from and level is not None and abs(vm[k] - level) >= b:
            near = np.where(np.abs(v[k_flat - W:k] - level) <= 0.5)[0]
            k0 = k_flat - W + (near[-1] if len(near) else W)
            return k, k0, level
    return None, None, None


def fit_end_speed(v_obs, v1, prior, sign, margin=1.0):
    """MAP end speed from the observed part (samples from t0): v1 + dv of the best (shape, T), with dv of the sign of
    the departure and the end speed in [0, V_MAX]. Once the vehicle has stopped moving further in the direction of
    the change (its last-1-s mean acceleration no longer has that sign), the end speed may not lie beyond the most
    extreme observed speed by more than `margin` (m/s): it has passed its extreme and is settling or recovering."""
    n = len(v_obs)
    slope = (v_obs[-1] - v_obs[max(0, n - 11)]) / max(min(n - 1, 10) * DT, DT)
    passed = sign * slope <= 0.1
    if sign < 0:
        lo_end, hi_end = (max(0.0, v_obs.min() - margin) if passed else 0.0), v1
    else:
        lo_end, hi_end = v1, (min(V_MAX, v_obs.max() + margin) if passed else V_MAX)
    t = np.arange(n) * DT
    y = v_obs - v1
    best, best_cost = None, np.inf
    for sh in SHAPE_INFO.values():
        u = np.clip(t[None, :] / T_GRID[:, None], 0, 1)
        F = sh["h"](u)                                            # (nT, n)
        FF = (F * F).sum(1)
        dv = np.where(FF > 1e-9, (F * y[None, :]).sum(1) / np.maximum(FF, 1e-9), 0.0)
        sse = ((y[None, :] - dv[:, None] * F) ** 2).sum(1)
        dv = np.clip(dv, lo_end - v1, hi_end - v1)
        dv = np.where(sign * dv > 0, dv, 0.0)
        sse = ((y[None, :] - dv[:, None] * F) ** 2).sum(1)
        lp = (np.log(T_GRID) - prior["a"] - prior["b"] * np.log(np.maximum(np.abs(dv), 0.1))) ** 2 / (2 * prior["sd"] ** 2)
        c = sse / (2 * SIG_V ** 2) + lp
        i = int(np.argmin(c))
        if c[i] < best_cost:
            best_cost, best = c[i], v1 + dv[i]
    return best


def identify(v, k_from, prior, k_max, stable_s=STABLE_S):
    """Departure, B and C identification indices, and the predicted end speed at B."""
    kd, k0, level = departure(v, k_from, k_max)
    if kd is None:
        return None
    sign = np.sign(v[kd] - level)
    hist, kB, vB = [], None, None
    m = int(round(stable_s / DT))
    for k in range(kd, k_max):
        vhat = fit_end_speed(v[k0:k + 1], level, prior, sign)
        hist.append(vhat)
        if len(hist) > m:
            w = np.array(hist[-m - 1:])
            if np.all(np.abs(w - w[-1]) <= STABLE_KMH * KMH):
                kB, vB = k, vhat
                break
    a = np.gradient(causal_ma(v, 10)) / DT
    m = int(round(FLAT_S / DT))
    kC = None
    for k in range(kd + m, k_max):
        if np.all(np.abs(a[k - m:k + 1]) <= A_FLAT):
            kC = k
            break
    return dict(kd=kd, k0=k0, level=level, kB=kB, vB=vB, kC=kC)
