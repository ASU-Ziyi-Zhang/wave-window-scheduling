"""Evaluate T5 criteria B and C on recorded adjacent pairs of the 26 moving leader transitions.

For each pair (follower = CAV candidate, predecessor = recorded vehicle id - 1), the follower watches its
predecessor's speed. Reported per pair: departure time, identification times B and C relative to the predecessor's
departure, the elapsed share of the predecessor's own transition at B, and the end-speed error at B.
Run: python eval_identify.py -> results/t5_identify.csv
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from identify import duration_prior, identify
from lib import DT, KMH, RES, load_run


def leader_transitions(v):
    """Moving-to-moving leader transitions: (index where the old plateau ends, old level, new level) in m/s."""
    vl = v * 3.6
    W, lev, i = 100, [], 0
    while i + W < len(vl):
        seg = vl[i:i + W]
        if seg.max() - seg.min() < 3.0:
            j = i + W
            while j < len(vl) and abs(vl[j] - seg.mean()) < 2.0:
                j += 1
            lev.append((i, j, vl[i:j].mean())); i = j
        else:
            i += 10
    out = []
    for k in range(len(lev) - 1):
        a, b = lev[k][2], lev[k + 1][2]
        if abs(b - a) > 5 and min(a, b) > 2:
            out.append((lev[k][1], a * KMH, b * KMH))
    return out


def run_one(run):
    t, ids, x, v = load_run(run)
    prior = duration_prior(run)
    rows = []
    for ev, (k_end, v1, v2) in enumerate(leader_transitions(v[1])):
        for vid in ids[1:]:
            p = vid - 1
            if p not in ids:
                continue
            vp = v[p]
            k_from = max(0, k_end - 50)
            k_max = min(len(vp) - 1, k_end + 900)
            r = identify(vp, k_from, prior, k_max)
            if r is None:
                continue
            kd = r["kd"]
            v_fin = v2                                   # every vehicle ends at the leader's new plateau
            band = np.where(np.abs(vp[kd:k_max] - v_fin) <= 1.5 * KMH)[0]
            kE = kd + band[0] if len(band) else None
            row = dict(run=run, event=ev, kind="decel" if v2 < v1 else "accel", v1_kmh=v1 * 3.6, v2_kmh=v2 * 3.6,
                       follower=vid, predecessor=p, rank_pred=ids.index(p),
                       t_depart=(kd - r["k0"]) * DT, T_pred=None if kE is None else (kE - r["k0"]) * DT)
            for c in ("B", "C"):
                k = r[f"k{c}"]
                row[f"t{c}"] = None if k is None else (k - r["k0"]) * DT          # seconds after the departure start
                row[f"share{c}"] = None if (k is None or kE is None) else (k - r["k0"]) / max(kE - r["k0"], 1)
            row["errB_kmh"] = None if r["vB"] is None else (r["vB"] - v_fin) * 3.6
            rows.append(row)
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        rows = [r for rr in ex.map(run_one, range(1, 6)) for r in rr]
    d = pd.DataFrame(rows)
    d.to_csv(RES / "t5_identify.csv", index=False)
    print("pairs", len(d), "events", d.groupby(["run", "event"]).ngroups)
    for kind, g in d.groupby("kind"):
        print(f"\n{kind}: n={len(g)}")
        for c in ("B", "C"):
            s = g[f"share{c}"].dropna()
            print(f"  {c}: found {g[f't{c}'].notna().mean():.0%}; time after start median {g[f't{c}'].median():.1f} s; "
                  f"share of predecessor's transition median {s.median():.2f} [IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}]")
        e = g.errB_kmh.dropna()
        print(f"  B end-speed error: median |e| {e.abs().median():.1f} km/h, 90th {e.abs().quantile(.9):.1f}, "
              f"within 2 km/h {(e.abs() <= 2).mean():.0%}")
        print("  predecessor duration median", round(g.T_pred.median(), 1), "s")
