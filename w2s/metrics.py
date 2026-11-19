"""Evaluation metrics beyond the acceleration cost (agreed 2026-10-26), in four groups:

  energy     fuel rate of a light-duty vehicle from the simplified models of the CIRCLES energy team, reduced from
             Argonne's Autonomie vehicle models (Lee et al. 2021; Khoudari et al. 2023; MATLAB release
             github.com/CIRCLES-consortium/CIRCLES-ENERGY-MODELS, version 1.0, 2023-09-30), ported line by line;
             fuel per distance (L/100 km, MPG); vehicle specific power (Jimenez-Palacios 1999) and the positive
             tractive energy per distance it implies, as a model-free check;
  stability  energy gain from the predecessor in a band of periods (Section 2.3, |G|^2), and the gain of the peak
             acceleration in the direction of the change (an L-infinity measure);
  extent     duration and distance of the transition (10 % to 90 % of the speed change), the area of extra spacing
             (m s) and the position deficit at the end of the window, in seconds;
  safety     smallest gap, smallest time gap, deepest borrowing beyond the braking buffer.
All functions take trajectories at the simulation step DT on a flat road.
"""
import numpy as np

from lib import DT

GASOLINE_G_PER_L = 745.0                                   # density of gasoline
L_PER_GAL, M_PER_MILE = 3.78541, 1609.344

# Fitted parameters of the simplified models (fuel_model_<name>_simplified.m)
VEH = {
    "midBase": dict(mass=1743, f_idle=0.1271, b1=3.9218, b2=48.9891, b3=0.00013964, b4=8.9036, b5=6.1892, b6=0.10048,
                    c0=0.19829, c1=0.021122, c2=0.0, c3=2.7801e-05, p0=0.23956, p1=0.0080592, p2=0.0027737,
                    q0=0.0, q1=0.050556, vc=5.07, beta0=0.1271, a0=-0.15742, a1=-0.00037876, a3=-0.00022957),
}


def fuel_rate(v, a, veh="midBase"):
    """Fuel rate (g/s) on a flat road, infeasible accelerations projected to the feasible boundary (the MATLAB
    default), as fuel_model_<veh>_simplified(v, a, 0, true)."""
    p = VEH[veh]
    v = np.maximum(np.asarray(v, float), 0.0)
    a = np.asarray(a, float)
    ma = np.minimum(p["b1"], p["b2"] / np.maximum(v, 1e-12) - p["b3"] * v ** 2)
    a = np.minimum(a, ma)
    aplus = np.maximum(a, -(p["p0"] + p["p1"] * v + p["p2"] * v ** 2) / (2 * (p["q0"] + p["q1"] * np.maximum(v, 1e-12))))
    f = (p["c0"] + p["c1"] * v + p["c2"] * v ** 2 + p["c3"] * v ** 3 + p["p0"] * a + p["p1"] * a * v
         + p["p2"] * a * v ** 2 + p["q0"] * aplus ** 2 + p["q1"] * aplus ** 2 * v)
    f = np.maximum(f, (v <= p["vc"]) * p["beta0"])
    a_brake = p["a0"] + p["a1"] * v + p["a3"] * v ** 2
    f = np.where((v > p["vc"]) & (a <= a_brake), 0.0, f)
    return np.where((v < 0.1) & (np.abs(a) < 0.01), p["f_idle"], f)


def fuel(v, a, veh="midBase"):
    """Fuel (g), distance (m), L/100 km and miles per gallon over the given samples."""
    g = float(np.sum(fuel_rate(v, a, veh)) * DT)
    d = float(np.sum(np.maximum(v, 0)) * DT)
    litres = g / GASOLINE_G_PER_L
    return dict(fuel_g=g, dist_m=d, l100=100e3 * litres / d if d > 0 else np.nan,
                mpg=(d / M_PER_MILE) / (litres / L_PER_GAL) if litres > 0 else np.nan)


def vsp(v, a):
    """Vehicle specific power (kW/t) of a light-duty vehicle on a flat road (Jimenez-Palacios 1999)."""
    return v * (1.1 * a + 0.132) + 0.000302 * v ** 3


def tractive_energy(v, a, mass=1743.0):
    """Positive tractive energy per distance (kJ/km): what the engine must deliver; braking losses count as waste."""
    e = float(np.sum(np.maximum(vsp(v, a), 0.0)) * DT) * mass / 1000.0      # kJ
    d = float(np.sum(np.maximum(v, 0)) * DT)
    return 1000.0 * e / d if d > 0 else np.nan


def band_energy(a, lo, hi):
    """Energy of an acceleration segment in the band lo <= f < hi (Hz), Hann window (as in wave_measure.py).
    Kept for the earlier results; superseded for transitions by band_energy_ev (2026-11-09)."""
    seg = (a - np.mean(a)) * np.hanning(len(a))
    F, f = np.fft.rfft(seg) * DT, np.fft.rfftfreq(len(a), DT)
    m = (f >= lo) & (f < hi)
    return float(2 * np.sum(np.abs(F[m]) ** 2) * (f[1] - f[0]))


def settled_from(v, v_new, k_from, tol_v=0.5, tol_a=0.3, sustain=10.0):
    """First index from k_from on at which a vehicle has reached the new level for good enough: from there its 1-s
    averages stay within tol_v of v_new with |a| <= tol_a for at least `sustain` seconds (2026-11-09:
    later small swings on the new plateau, e.g. a recorded head creeping at 9-12 km/h, are not part of the transition).
    None if this never happens."""
    def sm(y):
        return np.convolve(np.pad(y, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]
    v = np.asarray(v, float)
    ok = (np.abs(sm(v) - v_new) <= tol_v) & (np.abs(sm(np.r_[np.diff(v) / DT, 0.0])) <= tol_a)
    n, run = int(round(sustain / DT)), 0
    for k in range(max(k_from, 0), len(ok)):
        run = run + 1 if ok[k] else 0
        if run >= n:
            return k - n + 1
    return None


def completed_at(v, v_new, k_from, tol=1.5 / 3.6):
    """First index from k_from on at which the 1-s average speed comes within tol of the new level (the vehicle has
    completed its change, even if it swings on afterwards). None if never."""
    v = np.asarray(v, float)
    vs = np.convolve(np.pad(v, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]
    k = np.where(np.abs(vs[max(k_from, 0):] - v_new) <= tol)[0]
    return int(k[0]) + max(k_from, 0) if len(k) else None


TAIL_UNSETTLED = 30.0       # s after the last vehicle completes, when the platoon never settles before the next change


def event_end(vs, v_new, k_from, K, margin=5.0):
    """End of the common window of one change over the vehicles' speeds vs (2026-11-09): 5 s
    after the last vehicle settles (settled_from); if some vehicle never settles before the record ends (a head that
    creeps on at 9-12 km/h keeps the platoon swinging), 30 s after the last vehicle completes its change (longer than
    the longest period of the 10-25 s band). Returns (k_end, how): how = 'settled', 'unsettled' or 'overlap' (the
    window would run past the record, i.e. into the next change)."""
    m = int(margin / DT)
    ks = [settled_from(v, v_new, k_from) for v in vs]
    if all(k is not None for k in ks) and max(ks) + m <= K:
        return max(ks) + m, "settled"
    kc = [completed_at(v, v_new, k_from) for v in vs]
    if all(k is not None for k in kc):
        k_end = max(kc) + int(TAIL_UNSETTLED / DT)
        if k_end <= K:
            return k_end, "unsettled"
    return K, "overlap"


def steady_before(vs, v_old, k_open, span=10.0, tol_v=0.5, tol_a=0.3):
    """True if every vehicle holds the old level (1-s averages within tol_v of v_old, |a| <= tol_a) over the `span`
    seconds before the window opens (2026-11-09): otherwise the platoon is still swinging from
    the previous change, and the change overlaps it."""
    def sm(y):
        return np.convolve(np.pad(y, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]
    k0 = k_open - int(round(span / DT))
    if k0 < 0:
        return False
    for v in vs:
        v = np.asarray(v, float)
        vv, aa = sm(v)[k0:k_open], sm(np.r_[np.diff(v) / DT, 0.0])[k0:k_open]
        if np.any(np.abs(vv - v_old) > tol_v) or np.any(np.abs(aa) > tol_a):
            return False
    return True


N_FFT = 16384                                           # 1638.4 s at DT = 0.1 s, longer than any window


def band_energy_ev(a, lo, hi, n_fft=N_FFT):
    """Energy of the acceleration of one transition in the band lo <= f < hi (Hz):
    over the event's common window, which opens and closes where the platoon is steady, so the pulse is kept as it is,
    with no taper (a Hann taper weights a pulse by its position in the window) and no mean removal; the segment is
    padded with zeros to a fixed length, so that the band holds the same frequencies whatever the window's length.
    The RMS over the window is sqrt(E / T), T the window's length."""
    a = np.asarray(a, float)
    F, f = np.fft.rfft(a, n=max(n_fft, len(a))) * DT, np.fft.rfftfreq(max(n_fft, len(a)), DT)
    m = (f >= lo) & (f < hi)
    return float(2 * np.sum(np.abs(F[m]) ** 2) * (f[1] - f[0]))


BANDS = {"4-10s": (0.1, 0.25), "10-25s": (0.04, 0.1)}


def stability(a_pred, a, sign):
    """Band energy gains E/E_pred, and the gain of the peak acceleration in the direction of the change
    (sign = -1 for a slowdown, +1 for a speed-up)."""
    out = {f"gain_{b}": band_energy(a, *lh) / max(band_energy(a_pred, *lh), 1e-12) for b, lh in BANDS.items()}
    out["peak_gain"] = float(np.max(sign * a) / max(np.max(sign * a_pred), 1e-9))
    out["peak"] = float(np.max(sign * a))
    return out


def extent(t, x, v, v_old, v_new):
    """Duration (s) and distance (m) of the transition, from 10 % to 90 % of the change (1-s moving average)."""
    vs = np.convolve(np.pad(v, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]   # no zero padding at the ends
    s = np.sign(v_new - v_old)
    prog = s * (vs - v_old) / abs(v_new - v_old)
    i10, i90 = np.where(prog >= 0.1)[0], np.where(prog >= 0.9)[0]
    if not len(i10) or not len(i90):
        return dict(dur=np.nan, dist=np.nan)
    k0 = int(i10[0])
    k1 = int(i90[i90 >= k0][0]) if np.any(i90 >= k0) else len(v) - 1
    return dict(dur=(k1 - k0) * DT, dist=float(x[k1] - x[k0]))


def extent_back(t, x, v, v_old, v_new):
    """Duration (s) and distance (m) of the transition, searched back from where the vehicle first covers 90 % of the
    change: the last crossing of 50 % before it, then the last crossing of 10 % before that (1-s moving average).
    Unlike extent(), a start away from the old level, or a dip that crosses 10 % before the wave arrives, does not move
    the start (2026-11-06). NaN if the vehicle never covers 90 % of the change."""
    vs = np.convolve(np.pad(v, 5, mode="edge"), np.ones(10) / 10, mode="same")[5:-5]
    p = np.sign(v_new - v_old) * (vs - v_old) / abs(v_new - v_old)
    if not np.any(p >= 0.9):
        return dict(dur=np.nan, dist=np.nan)
    k90 = int(np.argmax(p >= 0.9))
    below = np.where(p[:k90 + 1] < 0.5)[0]
    k50 = int(below[-1]) + 1 if len(below) else 0
    below = np.where(p[:k50 + 1] < 0.1)[0]
    k10 = int(below[-1]) + 1 if len(below) else 0
    return dict(dur=(k90 - k10) * DT, dist=float(x[k90] - x[k10]))


def safety(g, v, dev, rb):
    """Smallest gap (m), smallest time gap (s, at speeds above 2 m/s), deepest borrowing beyond the buffer (m)."""
    fast = v > 2.0
    return dict(gap_min=float(g.min()), tgap_min=float((g[fast] / v[fast]).min()) if fast.any() else np.nan,
                borrow=float(max(0.0, -(dev + rb).min())))


def ttc(g, v, vp, thr=3.0, dt=DT):
    """Time to collision (2026-11-08): gap over closing speed, where the follower is faster than its predecessor and
    moves above 2 m/s. Returns the smallest TTC (s, inf if it never closes in) and the time exposed below thr s (TET,
    Minderhoud and Bovy 2001)."""
    dv = v - vp
    ok = (dv > 0.05) & (v > 2.0)
    T = np.full(len(g), np.inf)
    T[ok] = g[ok] / dv[ok]
    return dict(ttc_min=float(T.min()), tet=float((T < thr).sum() * dt))
