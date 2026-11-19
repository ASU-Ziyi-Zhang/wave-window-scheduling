"""One CAV behind a human-driven predecessor that slows down, without any recorded data.

The predecessor follows a synthetic deceleration from 47 to 19 km/h. The CAV starts on the r-Safety spacing and
drives either under its base controller alone (ACC, layer "feedback") or with wave-window scheduling layered on it
(layer "inf": the CAV is told its predecessor's whole future). For both, the script reports the duration of the CAV's
transition (10 % to 90 % of the change), the energy of its acceleration over the transition, and the ratio of that
energy to the predecessor's, which is the energy the CAV passes on.

Run from the repository root:  python examples/single_cav_example.py  [--plot]
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "w2s"))
from lib import DT, L_EFF                      # noqa: E402
from single_cav import K_ON, RS, simulate      # noqa: E402

KMH = 3.6


def predecessor(v1=47 / KMH, v2=19 / KMH, T=8.0, pre=20.0, post=90.0):
    """Synthetic predecessor: constant speed, a smooth (KS1) change of speed lasting T seconds, constant speed."""
    t = np.arange(0.0, pre + T + post, DT)
    xi = np.clip((t - pre) / T, 0.0, 1.0)
    v = v1 + (v2 - v1) * (3 * xi ** 2 - 2 * xi ** 3)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
    return x, v


def transition(v, v1, v2):
    """Duration and acceleration energy of the central 10-90 % part of a change of speed."""
    p = (v - v1) / (v2 - v1)
    k10 = int(np.argmax(p >= 0.1))
    k90 = int(np.argmax(p >= 0.9))
    a = np.diff(v) / DT
    return (k90 - k10) * DT, float(np.sum(a[k10:k90] ** 2) * DT)


def main(plot=False):
    xp, vp = predecessor()
    v1, v2 = vp[0], vp[-1]
    ev = dict(xp=xp, vp=vp, x0=xp[0] - L_EFF - RS.sr(v1), v0=v1, lead_dv=v2 - v1)
    Tp, Ep = transition(vp, v1, v2)
    print(f"predecessor: {v1 * KMH:.0f} -> {v2 * KMH:.0f} km/h, transition {Tp:.1f} s")
    out = {}
    for name, layer in (("ACC alone", "feedback"), ("ACC + W2S", "inf")):
        a, plans, gap_min, dev_min = simulate(ev, layer)
        v = v1 + np.concatenate([[0.0], np.cumsum(a[:-1]) * DT])
        T, E = transition(v, v1, v2)
        out[name] = v
        print(f"{name:10s}: transition {T:5.1f} s, energy passed on / received {E / Ep:4.2f}, "
              f"smallest gap {gap_min:5.1f} m, plans committed {len(plans)}")
    if plot:
        import matplotlib.pyplot as plt
        t = (np.arange(len(vp)) - K_ON) * DT
        plt.plot(t, vp * KMH, "k:", label="predecessor")
        for name, v in out.items():
            plt.plot(t, v * KMH, label=name)
        plt.xlabel("time from the predecessor's change (s)")
        plt.ylabel("speed (km/h)")
        plt.legend()
        plt.xlim(-10, 60)
        plt.show()


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
