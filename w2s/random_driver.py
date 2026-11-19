"""Stochastic human driver for the uncertainty study (2026-10-25).

The intelligent driver model with two slowly varying random terms, each an Ornstein-Uhlenbeck process:
  g(t)    an offset of the desired spacing, s* = s0 + g + v T + v dv / (2 sqrt(a b))  (spacing indifference). A change
          of g across a transition moves the centre of the driver's pulse by about dg / |dv| (Proposition 1), which is
          the timing scatter measured in the recorded platoon (it falls with the size of the change);
  eta(t)  an additive acceleration noise, the driver's own fluctuation on a plateau.
OU process with standard deviation sigma and correlation time tau:  z_{k+1} = z_k - z_k dt / tau + sigma sqrt(2 dt / tau) xi.
With sigma_g = sigma_a = 0 the model is the deterministic IDM.  The switch HDV_NOISE of platoon uses this module.
"""
import numpy as np

from lib import DT, L_EFF


def ou(n, sigma, tau, rng):
    z = np.zeros(n)
    if sigma <= 0:
        return z
    z[0] = rng.normal(0, sigma)
    c, s = DT / tau, sigma * np.sqrt(2 * DT / tau)
    xi = rng.normal(0, 1, n)
    for k in range(n - 1):
        z[k + 1] = z[k] - z[k] * c + s * xi[k]
    return z


def follow(th, xp, vp, x0, v0, prm=None, rng=None):
    """Trajectory (x, v) of a driver behind a predecessor (xp, vp), from (x0, v0). prm = dict(sg, tg, sa, ta).
    x0=None starts the driver at its own equilibrium gap at v0, its spacing offset included."""
    K = len(xp)
    x, v = np.zeros(K), np.zeros(K)
    if prm is None:
        g = eta = np.zeros(K)
    else:
        g, eta = ou(K, prm["sg"], prm["tg"], rng), ou(K, prm["sa"], prm["ta"], rng)
    s0, T, a, b, vf = th["s0"], th["T"], th["a"], th["b"], th["v0"]
    if x0 is None:
        x0 = xp[0] - L_EFF - (max(s0 + g[0], 0.5) + v0 * T) / np.sqrt(1 - (v0 / vf) ** 4)
    x[0], v[0] = x0, v0
    sab = 2 * np.sqrt(a * b)
    for k in range(K - 1):
        gap = max(xp[k] - x[k] - L_EFF, 0.1)
        ss = max(s0 + g[k], 0.5) + max(0.0, v[k] * T + v[k] * (v[k] - vp[k]) / sab)
        acc = a * (1 - (max(v[k], 0) / vf) ** 4 - (ss / gap) ** 2) + eta[k]
        v[k + 1] = max(v[k] + acc * DT, 0.0)
        x[k + 1] = x[k] + 0.5 * (v[k] + v[k + 1]) * DT
    return x, v
