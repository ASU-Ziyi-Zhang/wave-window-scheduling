"""Round-2 simulator (baseline scenario D15, identification D17): time-stepped, with information, relaying,
scheduling, re-anchoring and rescheduling.

Platoon: 25 vehicles (ids 1..25, rank = id - 1); rank 0 is the recorded leader; HDVs follow their calibrated IDM;
CAVs run C0 on the r-Safety spacing, or track a committed transition (Eq. trk) under the safety filter.
Information (cfg.info):
  'relay'     a CAV identifies the end speed of an HDV predecessor from an empirical identification path (D17 option 3)
              and relays its estimate and its plans downstream, tau_c per hop; a CAV directly behind a CAV gets the
              predecessor's plan by V2V.
  'local'     identification only, no relaying.
  'broadcast' every CAV knows the leader's future trajectory from the leader's departure + latency (upper bound).
Modes: 'hdv', 'feedback', 'independent', 'sequential', 'joint'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from eval_identify import leader_transitions
from identify import duration_prior, fit_end_speed
from lib import (A_MAX, A_MIN, DT, G_MIN, J_MAX, KMH, L_EFF, RES, SHAPE_INFO, RSafety, cost, idm_acc, load_idm,
                 load_run)
from lib import CFG_DIR  # noqa: E402  (calibrated parameters, ../config)
from rollout import _filter, jerk, roll_cav, roll_idm

CAL = json.loads((CFG_DIR / "population_idm.json").read_text())
RS = RSafety(**{k: CAL["rsafety"][k] for k in ("d0", "tau_h", "r", "a_ref")})
_IDP = None


def _idp():
    """Recorded identification paths (config/identification_paths.npz). They are derived from the recorded platoon
    and are not distributed; only the experiments that replay that data need them."""
    global _IDP
    if _IDP is None:
        _IDP = np.load(CFG_DIR / "identification_paths.npz")
    return _IDP
PRE, POST = 20.0, 90.0


@dataclass
class Cfg:
    mode: str = "joint"
    info: str = "relay"
    cav: tuple = ()
    tau_c: float = 0.1            # per-hop relay delay (s)
    latency: float = 0.3          # planning and actuation latency added to every information time (s)
    kg: float = 0.50              # D18: k_s of the r-Safety paper (K = 0.50); speed gains raised for string stability
    kd: float = 1.80
    kv: float = 1.80
    gate: float = 2.0             # D18: plan only if the estimated speed change is at least this (m/s)
    hold: float = 2.0             # hysteresis: an estimate change reschedules a CAV at most once per `hold` seconds
    sig_g: float = 1.0
    sig_v: float = 0.5
    D1: float = 1.0
    D2: float = 3.0
    retime: float = 1.0           # tolerance on the end time when re-anchoring (s)
    cj: float = 0.1
    s_step: float = 0.5
    n_keep: int = 6
    beam: int = 150
    horizon: float = 60.0         # forecast horizon (s)
    seed: int = 0
    rs: RSafety = field(default_factory=lambda: RS)
    dG: float | None = None       # constant lag limit (diagnostic); None: use band_up
    band_up: str = "rb_max"       # lag limit (D24: max(R_b(v1), R_b(v2))): 'rb_v1' R_b at the transition's start speed (D23), 'rb_v' R_b(v) (D14),
                                  # 'rb_max' max(R_b(v1), R_b(v2)) (sensitivity)
    forecast: str = "template"    # predecessor forecast: 'template' (direct predecessor: Newell copy of the nearest
                                  # upstream reference shifted by eta per vehicle, or its own fitted transition);
                                  # 'pop' / 'true': car-following chain with the population / each HDV's own IDM
    eta: float = 1.0              # time for the wave to pass one vehicle (s/veh), Section 2.2
    oracle_end: bool = False      # diagnostic: once informed, the CAV holds the true end speed
    corroborate: bool = False     # commit only when a locally identified end speed is corroborated (D26)
    info_rule: str = "plateau"    # D28: targets are plateaus the observed vehicles actually reach (chained segments);
                                  # 'extrapolate': criterion B extrapolates the final end speed (D17)
    plateau_hold: float = 1.0     # a plateau: 1-s smoothed |a| <= 0.3 m/s^2 held this long (s)
    corro_frac: float = 0.5       # ... the predecessor has covered this share of the estimated change
    corro_hold: float = 2.0       # ... and the estimate has stayed within 1 km/h for this long (s)


# ------------------------------------------------------------------------------------------------------------- event
class Event:
    def __init__(self, run, idx, human_tau=0.0, eq_start=False):
        t, ids, x, v = load_run(run)
        k_end, v1, v2 = leader_transitions(v[1])[idx]
        trs = leader_transitions(v[1])
        k_next = trs[idx + 1][0] if idx + 1 < len(trs) else len(t)
        self.run, self.idx, self.v1, self.v2 = run, idx, v1, v2
        self.kind = "decel" if v2 < v1 else "accel"
        k0 = k_end - int(PRE / DT)
        k1 = min(k_end + int(POST / DT), k_next - 50, len(t) - 1)
        self.K = k1 - k0
        self.t = (np.arange(self.K) - int(PRE / DT)) * DT          # time since the leader's departure
        self.k_on = int(PRE / DT)
        self.xL, self.vL = x[1][k0:k1] - x[1][k0], v[1][k0:k1]
        X0, V0 = np.zeros(25), np.zeros(25)
        for vid in range(1, 26):
            if vid in ids:
                X0[vid - 1], V0[vid - 1] = x[vid][k0] - x[1][k0], v[vid][k0]
        for vid in range(2, 26):                               # unrecorded vehicles: between their neighbours
            if vid not in ids:
                X0[vid - 1] = 0.5 * (X0[vid - 2] + X0[vid]) if vid < 25 else X0[vid - 2] - 15
                V0[vid - 1] = 0.5 * (V0[vid - 2] + V0[vid]) if vid < 25 else V0[vid - 2]
        self.X0, self.V0 = X0, V0
        self.rec = {}                                          # recorded (x, v, a) of every recorded follower
        for vid in ids[1:]:
            xr, vr = x[vid][k0:k1] - x[1][k0], v[vid][k0:k1]
            vs = np.convolve(vr, np.ones(10) / 10, mode="same")
            self.rec[vid - 1] = (xr, vr, np.gradient(vs) / DT)
        if human_tau == 0.0:
            idm = load_idm()
        else:
            scan = json.loads((RES / "didm_scan.json").read_text())[str(human_tau)]
            idm = {int(k): v for k, v in scan.items()}
        self.human_tau = human_tau
        self.idm_true = [idm[m + 1] for m in range(1, 25)]
        self.idm_pop = CAL["pop_idm"][str(run)]
        self.prior = duration_prior(run)
        if eq_start:                                           # diagnostic: every follower in equilibrium at v_L(0)
            v0 = self.vL[0]
            for m in range(1, 25):
                th = self.idm_true[m - 1]
                gap = (th["s0"] + v0 * th["T"]) / np.sqrt(1 - (v0 / th["v0"]) ** 4)
                self.X0[m], self.V0[m] = self.X0[m - 1] - L_EFF - gap, v0


def even_ranks(n, N=24):
    return tuple(sorted({int(round(k * (N + 1) / (n + 1))) for k in range(1, n + 1)}))


# -------------------------------------------------------------------------------------------------------- candidates
def _generate(cfg, v2, xp, vp, x0, v0, start_only=False):
    """Admissible AKS transitions (KS1, KS1A rho 0.4/0.5/0.6, KS2) from (x0, v0) behind the forecast (xp, vp), as
    arrays over candidates; column 0 is the planning instant."""
    K = len(xp)
    rs = cfg.rs
    tt = np.arange(K) * DT
    Tg = np.arange(1.0, 45.0, 0.1)
    s_idx = np.array([0]) if start_only else np.arange(0, K, int(round(cfg.s_step / DT)))
    rows = []
    for name, sh in SHAPE_INFO.items():
        rho = sh["rho"]
        EF = s_idx[:, None] + Tg[None, :] / DT
        valid = EF < K - 1
        xpe = np.interp(np.where(valid, EF, K - 1).ravel(), np.arange(K), xp).reshape(EF.shape)
        f = x0 + v0 * s_idx[:, None] * DT + Tg[None, :] * (v0 + (v2 - v0) * rho) - (xpe - L_EFF - rs.sr(v2))
        f = np.where(valid, f, np.nan)
        cond = (f[:, :-1] < 0) & (f[:, 1:] >= 0) if v0 >= v2 else (f[:, :-1] > 0) & (f[:, 1:] <= 0)
        has = cond.any(1); idx = np.argmax(cond, 1)
        for r in np.where(has)[0]:
            i = idx[r]
            T = Tg[i] + (Tg[i + 1] - Tg[i]) * (-f[r, i]) / (f[r, i + 1] - f[r, i])
            rows.append((name, int(s_idx[r]), float(T), None))
    # KS2: two KS1 primitives through a junction speed vJ that closes the displacement
    for si in s_idx[::2]:
        for T in np.arange(3.0, 30.0, 1.0):
            e = si + T / DT
            if e >= K - 1:
                break
            dx = np.interp(e, np.arange(K), xp) - L_EFF - rs.sr(v2) - (x0 + v0 * si * DT)
            for al in (0.3, 0.5, 0.7):
                vJ = 2 * dx / T - al * v0 - (1 - al) * v2
                lo, hi = min(v0, v2), max(v0, v2)
                if vJ < 0 or (lo <= vJ <= hi) or abs(vJ - np.clip(vJ, lo, hi)) > 0.5 * abs(v2 - v0) + 1.0:
                    continue
                rows.append(("KS2", int(si), float(T), (al, float(vJ))))
    if not rows:
        return None
    C = len(rows)
    X = np.empty((C, K)); V = np.empty((C, K)); A = np.zeros((C, K)); E = np.empty(C, int)
    h1 = SHAPE_INFO["KS1"]
    for c, (name, si, T, extra) in enumerate(rows):
        s = si * DT
        if name != "KS2":
            sh = SHAPE_INFO[name]
            u = np.clip((tt - s) / T, 0, 1)
            V[c] = v0 + (v2 - v0) * sh["h"](u)
            inside = (tt >= s) & (tt <= s + T)
            A[c, inside] = (v2 - v0) * sh["dh"](u[inside]) / T
        else:
            al, vJ = extra
            T1, T2 = al * T, (1 - al) * T
            u1 = np.clip((tt - s) / T1, 0, 1); u2 = np.clip((tt - s - T1) / T2, 0, 1)
            V[c] = np.where(tt < s + T1, v0 + (vJ - v0) * h1["h"](u1), vJ + (v2 - vJ) * h1["h"](u2))
            in1 = (tt >= s) & (tt < s + T1); in2 = (tt >= s + T1) & (tt <= s + T)
            A[c, in1] = (vJ - v0) * h1["dh"](u1[in1]) / T1
            A[c, in2] = (v2 - vJ) * h1["dh"](u2[in2]) / T2
        X[c] = x0 + np.concatenate([[0], np.cumsum(0.5 * (V[c, 1:] + V[c, :-1]) * DT)])
        E[c] = min(int(np.ceil((s + T) / DT)), K - 1)
    return rows, X, V, A, E


def candidates(cfg, v2, xp, vp, x0, v0, start_only=False, a0=0.0, j_rel=None, margin=0.0, up_extra=0.0):
    """Admissible AKS transitions (KS1, KS1A rho 0.4/0.5/0.6, KS2) from (x0, v0, a0) behind the forecast (xp, vp), as
    arrays over candidates; column 0 is the planning instant. A non-zero current acceleration a0 is first brought to
    zero by the release primitive a = a0 (1 - u)^2 (Appendix A) over tau_in = max(0.5 s, 2|a0|/j_max), so the reference
    starts at the vehicle's acceleration. margin (m, D75) narrows the band on both sides for an uncertain forecast."""
    K = len(xp)
    rs = cfg.rs
    n_in = 0
    if abs(a0) > 0.05:
        tau = max(0.5, 2 * abs(a0) / (j_rel or J_MAX))           # j_rel: jerk of the release (default j_max)
        n_in = int(np.ceil(tau / DT)); tau = n_in * DT
        t_in = np.arange(n_in + 1) * DT; u = t_in / tau
        a_rel = a0 * (1 - u) ** 2
        v_rel = v0 + a0 * tau * (1 - (1 - u) ** 3) / 3
        x_rel = x0 + v0 * t_in + a0 * tau / 3 * (t_in - tau * (1 - (1 - u) ** 4) / 4)
        if v_rel[-1] < 0 or n_in >= K - 50:
            return None
        g = _generate(cfg, v2, xp[n_in:], vp[n_in:], x_rel[-1], v_rel[-1], start_only)
        if g is None:
            return None
        rows, Xc, Vc, Ac, Ec = g
        C = len(rows)
        X = np.concatenate([np.repeat(x_rel[None, :-1], C, 0), Xc], 1)
        V = np.concatenate([np.repeat(v_rel[None, :-1], C, 0), Vc], 1)
        A = np.concatenate([np.repeat(a_rel[None, :-1], C, 0), Ac], 1)
        E = Ec + n_in
    else:
        g = _generate(cfg, v2, xp, vp, x0, v0, start_only)
        if g is None:
            return None
        rows, X, V, A, E = g
        C = len(rows)
    Jk = jerk(A)
    G = xp[None, :] - X - L_EFF
    D = G - rs.sr(V)
    mask = np.arange(K)[None, :] <= E[:, None]
    if cfg.dG is not None:
        up = np.full_like(V, cfg.dG)
    elif cfg.band_up == "rb_v":
        up = rs.rb(V)
    elif cfg.band_up == "rb_max":
        up = np.full_like(V, max(rs.rb(v0), rs.rb(v2)))
    else:
        up = np.full_like(V, rs.rb(v0))                         # D23: the buffer held at the start of the transition
    up = np.maximum(up - margin + up_extra, float(D[0, 0]))   # D75: margin for the forecast error of the predecessor;
                                                              # up_extra: a buffer's spacing kept (2026-11-09)
    lowb = np.minimum(-rs.rb(V) + margin, float(D[0, 0]))     # a borrowing already present at release is allowed
    if getattr(cfg, "low_off", False):
        lowb = np.full_like(V, -1e9)
    ok = ((np.where(mask, D - lowb, np.inf).min(1) >= -1e-6) & (np.where(mask, up - D, np.inf).min(1) >= -1e-6)
          & (np.where(mask, G - G_MIN, np.inf).min(1) >= 0) & (np.where(mask, A, 0).min(1) >= A_MIN)
          & (np.where(mask, A, 0).max(1) <= A_MAX) & (np.abs(np.where(mask, Jk, 0))[:, 1:].max(1) <= J_MAX + 1e-6)
          & (np.abs(G[np.arange(C), E] - rs.sr(V[np.arange(C), E])) < 0.5))
    if getattr(cfg, "diag", None) is not None:                 # diagnostic: which requirement removes candidates
        cfg.diag.update(n=C, low=int((np.where(mask, D - lowb, np.inf).min(1) >= -1e-6).sum()),
                        up=int((np.where(mask, up - D, np.inf).min(1) >= -1e-6).sum()),
                        gap=int((np.where(mask, G - G_MIN, np.inf).min(1) >= 0).sum()),
                        acc=int(((np.where(mask, A, 0).min(1) >= A_MIN) & (np.where(mask, A, 0).max(1) <= A_MAX)).sum()),
                        jerk=int((np.abs(np.where(mask, Jk, 0))[:, 1:].max(1) <= J_MAX + 1e-6).sum()),
                        closure=int((np.abs(G[np.arange(C), E] - rs.sr(V[np.arange(C), E])) < 0.5).sum()))
    keep = np.where(ok)[0]
    if len(keep) == 0:
        return None
    return dict(x=X[keep], v=V[keep], a=A[keep], e=E[keep], gref=G[keep],
                shape=[rows[i][0] for i in keep], s=np.array([rows[i][1] * DT + n_in * DT for i in keep]),
                T=np.array([rows[i][2] for i in keep]))


# ---------------------------------------------------------------------------------------------- identification
def settle_acc(v, k, n=10, w=20):
    """Acceleration over the last w steps from the speed smoothed by a causal n-step (1 s) moving average: the raw
    10 Hz difference of recorded speed carries about 0.3 m/s^2 of noise, the size of the settling threshold."""
    seg = v[max(0, k - w - n + 1):k + 1]
    if len(seg) < w + n:
        return np.array([])
    sm = np.convolve(seg, np.ones(n) / n, mode="valid")
    return np.diff(sm) / DT


class SrcB:
    """Criterion B on an observed speed series (a recorded vehicle or a CAV). est[k] is the estimate held at step k
    (NaN before the first stable one); settled[k] marks a vehicle steady (|a| <= 0.3 m/s^2 for 2 s) at least 2 m/s
    away from its old plateau, whose speed is then taken as the end speed."""
    def __init__(self, K, prior, plateau_only=False, hold=20):
        self.est = np.full(K, np.nan); self.settled = np.zeros(K, bool)
        self.prior, self.flat, self.kd, self.k0, self.level, self.cur, self.hist = prior, None, None, None, None, None, []
        self.held, self.plateau_only, self.hold = None, plateau_only, hold

    def update(self, k, v, r=None):
        W = 50
        if self.kd is None:
            if k >= W:
                win = v[k - W:k]
                if win.std() < 0.5:
                    self.flat = float(win.mean())
                vm = v[max(0, k - 4):k + 1].mean()
                if self.flat is not None and abs(vm - self.flat) >= 2.0:
                    self.kd, self.level, self.sign = k, self.flat, np.sign(vm - self.flat)
                    self.hist, self.cur = [], None
                    near = np.where(np.abs(v[:k] - self.level) <= 0.5)[0]
                    self.k0 = int(near[-1]) if len(near) else k - 20
            return
        a = settle_acc(v, k, w=self.hold)                             # 1-s causal smoothing first
        if len(a) >= self.hold and np.all(np.abs(a) <= 0.3) and abs(v[k] - self.level) >= 2.0:
            self.held = float(np.mean(v[k - 9:k + 1]))                 # the latest plateau reached
        if self.held is not None:                                      # held until a new plateau replaces it
            self.est[k], self.settled[k] = self.held, True
            return
        if k - self.kd >= 50 and v[k - 50:k].std() < 0.5 and abs(v[k] - self.level) < 2.0:
            self.kd = None                                             # back on the old plateau: re-arm
            return
        if self.plateau_only:
            return
        if (k - self.kd) % 5 == 0:
            self.hist.append(fit_end_speed(v[self.k0:k + 1], self.level, self.prior, self.sign))
            w = np.array(self.hist[-3:])
            if len(w) == 3 and np.all(np.abs(w - w[-1]) <= 1.0 * KMH):
                self.cur = float(w[-1])
        if self.cur is not None:
            self.est[k] = self.cur


class SrcPath:
    """Identification of a simulated HDV from an empirical path of recorded identification errors (D17, option 3)."""
    def __init__(self, K, sim):
        self.est = np.full(K, np.nan); self.settled = np.zeros(K, bool)
        self.sim, self.flat, self.kd, self.k0, self.level, self.cur = sim, None, None, None, None, None
        self.held = None

    def update(self, k, v, r=None):
        ev = self.sim.ev
        W = 50
        if self.kd is None:
            if k >= W:
                win = v[k - W:k]
                if win.std() < 0.5:
                    self.flat = float(win.mean())
                vm = v[max(0, k - 4):k + 1].mean()
                if self.flat is not None and abs(vm - self.flat) >= 2.0:
                    self.kd, self.level = k, self.flat
                    near = np.where(np.abs(v[:k] - self.level) <= 0.5)[0]
                    self.k0 = int(near[-1]) if len(near) else k - 20
                    self.path = self.sim._sample_path(int(ev.v2 < self.level), ev.v2 - self.level)
            return
        a = settle_acc(v, k)                                           # 1-s causal smoothing first
        if len(a) >= 20 and np.all(np.abs(a) <= 0.3) and abs(v[k] - self.level) >= 2.0:
            self.held = float(np.mean(v[k - 9:k + 1]))
        if self.held is not None:
            self.est[k], self.settled[k] = self.held, True
            return
        if k - self.kd >= 50 and v[k - 50:k].std() < 0.5 and abs(v[k] - self.level) < 2.0:
            self.kd = None; self.cur = None
            return
        i = min(k - self.kd, len(self.path) - 1)
        e = self.path[max(0, i - 10):i + 1]
        if i >= 10 and not np.any(np.isnan(e)):
            vh = ev.v2 + e * (ev.v2 - self.level)
            if np.all(np.abs(vh - vh[-1]) <= 1.0 * KMH):
                self.cur = float(vh[-1])
        if self.cur is not None:
            self.est[k] = self.cur


# ---------------------------------------------------------------------------------------------------- the simulator
class Sim:
    def __init__(self, ev: Event, cfg: Cfg):
        self.ev, self.cfg = ev, cfg
        self.rng = np.random.default_rng(cfg.seed)
        K = ev.K
        self.X = np.zeros((K, 25)); self.V = np.zeros((K, 25)); self.A = np.zeros((K, 25))
        self.X[0], self.V[0] = ev.X0, ev.V0
        self.cav = set(cfg.cav) if cfg.mode != "hdv" else set()
        self.first = min(cfg.cav) if cfg.cav else 25
        self.replay = []                                       # recorded vehicles ahead of the first CAV
        for r in range(1, self.first):                         # every recorded vehicle ahead of the first CAV;
            if r in ev.rec:                                    # unrecorded ones there are simulated HDVs
                self.replay.append(r)
        for r in self.replay:
            self.X[0, r], self.V[0, r] = ev.rec[r][0][0], ev.rec[r][1][0]
        self._init_sources = cfg.mode in ("independent", "sequential", "joint") and cfg.info != "broadcast"
        self.plan = {}            # rank -> dict(k0, x, v, a, gref, e, shape, J)
        self.est = {}             # rank -> current end-speed estimate
        self.tI = {}              # rank -> step of the information time
        self.basis = {}           # rank -> estimate used by the last schedule
        self.src = {}
        self.local = {}           # rank -> identification state of an HDV predecessor
        self.inbox = []           # (deliver_step, rank, est, plans_known)
        self.relayed = {}         # rank -> plans of upstream CAVs known by relay {u: plan}
        self.closed = set()
        self.log = []
        self.n = dict(schedule=0, reanchor=0, escalate=0, filter=0)
        self.next_id = 0
        self.sources = {}         # key -> (identification source, observed rank)
        self.last_est_trig = {}   # rank -> step of the last rescheduling caused by a changed estimate
        self.corro = {}           # rank -> whether its end speed is corroborated (D26)
        self.fused = {}           # rank -> fused estimate per step
        self.fused_settled = {}   # rank -> whether that estimate rests on a settled vehicle
        self.fused_rank = {}      # rank -> rank of the vehicle whose plateau the estimate is (plateau rule)
        self.last_re = {}
        self.u_prev = np.zeros(25)

    # ------------------------------------------------------------------ information
    def _sample_path(self, decel, dv):
        IDP = _idp()
        cand = np.where((IDP["decel"] == decel) & (IDP["run"] != self.ev.run))[0]
        near = cand[np.argsort(np.abs(np.abs(IDP["dv"][cand]) - abs(dv)))[:25]]
        return IDP["e"][self.rng.choice(near)]

    def _src(self, key, kind, r):
        if key not in self.sources:
            if kind == "B":
                src = SrcB(self.ev.K, self.ev.prior, plateau_only=self.cfg.info_rule == "plateau",
                           hold=int(round(self.cfg.plateau_hold / DT)))
            else:
                src = SrcPath(self.ev.K, self)
            self.sources[key] = src, r
        return self.sources[key]

    def _sources_of(self, m):
        """(source, delay in steps) pairs that CAV m reads: its own observation of its predecessor, and, with
        relaying, the trajectories and fused estimates of the CAVs ahead of it."""
        cfg = self.cfg
        p = m - 1
        out = []
        if p in self.replay or p in self.cav or cfg.info_rule == "plateau":
            out.append((self._src(("B", p), "B", p), 0))           # observed speed: plateaus (and criterion B)
        else:
            out.append((self._src(("path", m), "path", p), 0))     # simulated HDV: empirical identification path
        if cfg.info == "relay":
            ups = [u for u in sorted(self.cav) if u < m]
            d_c = int(round(cfg.tau_c / DT))
            for i, u in enumerate(reversed(ups)):
                hops = i + 1
                if u != p:
                    out.append((self._src(("B", u), "B", u), hops * d_c))   # the CAV's own relayed trajectory
                out.append((("fused", u), hops * d_c))                     # its fused estimate
        return out

    def update_info(self, k):
        """Fuse, for every CAV, the estimates available at step k: a settled observed speed if any (median),
        otherwise the median of the stable estimates. Returns the CAVs whose window opened or changed."""
        cfg, ev = self.cfg, self.ev
        trig = set()
        lat = int(round(cfg.latency / DT))
        for (src, r) in list(self.sources.values()):
            src.update(k, self.V[:, r], r)
        for m in sorted(self.cav):
            self.fused.setdefault(m, np.full(ev.K, np.nan))
            for msg in [q for q in self.inbox if q[0] <= k and q[1] == m]:
                self.inbox.remove(msg)
                sig = lambda d: tuple(sorted((u, q["id"]) for u, q in (d or {}).items()))
                if sig(msg[3]) != sig(self.relayed.get(m)):
                    trig.add(m)
                self.relayed[m] = msg[3]
            if cfg.info == "broadcast":
                new = ev.v2 if k >= ev.k_on + lat else None
                self.src[m] = "broadcast"
            else:
                vals, settled, ranked = [], [], []
                for item, d in self._sources_of(m):
                    j = k - d
                    if j < 0:
                        continue
                    if isinstance(item, tuple) and item[0] == "fused":
                        val = self.fused.get(item[1], np.full(ev.K, np.nan))[j]
                        if np.isfinite(val):
                            (settled if self.fused_settled.get(item[1], np.zeros(ev.K, bool))[j] else vals).append(val)
                            ranked.append((self.fused_rank.get(item[1], item[1]), val))
                        continue
                    src, r_obs = item
                    if np.isfinite(src.est[j]):
                        (settled if src.settled[j] else vals).append(src.est[j])
                        ranked.append((r_obs, src.est[j]))
                if cfg.info_rule == "plateau":                     # the plateau furthest ahead in the platoon
                    if ranked:
                        rank_used, new = min(ranked)
                        self.fused_rank[m] = rank_used
                    else:
                        new = None
                else:
                    new = float(np.median(settled)) if settled else (float(np.median(vals)) if vals else None)
                if new is not None:
                    self.src[m] = "settled" if settled else ("fused" if len(vals) > 1 else "local")
            if new is None:
                continue
            if cfg.oracle_end:
                new = ev.v2
            self.fused[m][k] = new
            self.fused_settled.setdefault(m, np.zeros(ev.K, bool))[k] = self.src.get(m) == "settled"
            if cfg.corroborate and m in self.tI:
                ok = self._corroborated(k, m)
                if ok and not self.corro.get(m, False):
                    trig.add(m)
                    self.log.append((ev.t[k], m, "corroborated", f"end {new * 3.6:.1f} km/h"))
                self.corro[m] = ok
            if m not in self.tI:
                self.est[m] = new
                self.tI[m] = k
                trig.add(m)
                self.log.append((ev.t[k], m, "informed", f"{self.src[m]}, end {new * 3.6:.1f} km/h"))
                self._relay(m, k)
            else:
                self.est[m] = new
                if m in self.basis and abs(new - self.basis[m]) > 1.0 * KMH \
                        and k - self.last_est_trig.get(m, -10 ** 6) >= int(round(cfg.hold / DT)):
                    self.last_est_trig[m] = k
                    trig.add(m)
                    self.closed.discard(m)
                    self.log.append((ev.t[k], m, "estimate changed", f"{self.basis[m] * 3.6:.1f} -> {new * 3.6:.1f} km/h"))
        return trig

    def _corroborated(self, k, m):
        """D26: an end speed that only comes from identifying a human predecessor is acted on once the predecessor
        has covered `corro_frac` of the estimated change in its direction and the estimate has held within 1 km/h
        for `corro_hold` s. Exact information (broadcast, a settled vehicle, a committed plan ahead) needs no check."""
        cfg = self.cfg
        if not cfg.corroborate or cfg.info == "broadcast" or self.src.get(m) == "settled":
            return True
        if self.relayed.get(m) or ((m - 1) in self.cav and (m - 1) in self.plan):
            return True
        p = m - 1
        src = self.sources.get(("B", p), self.sources.get(("path", m), (None,)))[0]
        if src is None or src.kd is None or m not in self.est:
            return False
        lvl, est, vp = src.level, self.est[m], self.V[k, p]
        if np.sign(est - lvl) != np.sign(vp - lvl) or abs(vp - lvl) < cfg.corro_frac * abs(est - lvl):
            return False
        w = self.fused[m][max(0, k - int(round(cfg.corro_hold / DT))):k + 1]
        return bool(np.all(np.isfinite(w)) and (w.max() - w.min()) <= 1.0 * KMH)

    def _t_ready(self, m, k):
        return k                                                   # latency is applied where information is created

    def _relay(self, m, k):
        if self.cfg.info != "relay":
            return
        down = [c for c in sorted(self.cav) if c > m]
        if not down:
            return
        nxt = down[0]
        plans = dict(self.relayed.get(m, {}))
        if m in self.plan:
            plans[m] = self.plan[m]
        self.inbox.append((k + int(round(self.cfg.tau_c / DT)), nxt, None, plans))

    # ------------------------------------------------------------------ forecasts
    def _template(self, x0, v0, vend, K):
        """Forecast of an HDV whose end speed is believed to be vend: KS1 from its current speed with the prior
        duration."""
        dv = vend - v0
        T = max(2.0, float(np.exp(self.ev.prior["a"] + self.ev.prior["b"] * np.log(max(abs(dv), 0.1)))))
        tt = np.arange(K) * DT
        u = np.clip(tt / T, 0, 1)
        v = v0 + dv * SHAPE_INFO["KS1"]["h"](u)
        x = x0 + np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
        return x, v

    def _ctl(self):
        return self.cfg                                            # roll_cav reads kg, kd, kv, rs

    def forecast_chain(self, k, m_from, m_to, xp, vp, plans, H):
        """Forecast ranks m_from..m_to from step k behind predecessor rows (B, H): CAVs with a plan in `plans`
        follow it (ideal tracking), other CAVs run C0, HDVs the population IDM. Returns rows of m_to and the
        summed cost."""
        B = xp.shape[0]
        Jsum = np.zeros(B)
        self._first_rows = None
        for m in range(m_from, m_to + 1):
            x0, v0, u0 = self.X[k, m], self.V[k, m], self.A[max(k - 1, 0), m]
            if m in self.cav:
                ref = plans.get(m)
                if ref is not None and not isinstance(ref.get("e"), np.ndarray):
                    ref = self._as_rows(ref, k, H, B)
                x, v, a, _ = roll_cav(self._ctl(), xp, vp, x0, v0, u0, ref=ref, ideal=True)
            else:
                th = self.ev.idm_true[m - 1] if self.cfg.forecast == "true" else self.ev.idm_pop
                x, v, a = roll_idm(th, xp, vp, np.full(B, x0), np.full(B, v0))
            Jsum += cost(a, jerk(a), self.cfg.cj)
            if self._first_rows is None:
                self._first_rows = (x, v)
            xp, vp = x, v
        return xp, vp, Jsum

    def _as_rows(self, p, k, H, B):
        """A committed plan (absolute start p['k0']) cut to the window [k, k + H) and repeated over B rows; beyond
        its stored length the plan cruises at its last speed."""
        off = k - p["k0"]
        n = len(p["x"])
        idx = np.arange(off, off + H)
        last = np.minimum(idx, n - 1)
        extra = np.maximum(idx - (n - 1), 0) * DT
        x = p["x"][last] + p["v"][-1] * extra
        v = p["v"][last]
        a = np.where(idx < n, p["a"][last], 0.0)
        gref = p["gref"][last]
        rep = lambda z: np.repeat(z[None], B, 0)
        return dict(x=rep(x), v=rep(v), a=rep(a), gref=rep(gref), e=np.full(B, max(p["e"] - off, -1)))

    def newell_copy(self, k, p, u, v_future, H):
        """Predecessor p copies the speed of the upstream reference u shifted by (p - u) eta; v_future is u's
        forecast from step k (length H); before k the reference's realised speed is used. The present difference
        between p and the shifted reference decays over 2 s."""
        d = int(round((p - u) * self.cfg.eta / DT))
        idx = k - d + np.arange(H)
        hist = self.V[np.clip(idx, 0, k), u] if u > 0 else self.ev.vL[np.clip(idx, 0, self.ev.K - 1)]
        fut = v_future[np.clip(idx - k, 0, H - 1)]
        vr = np.where(idx < k, hist, fut)
        tt = np.arange(H) * DT
        v = np.maximum(vr + (self.V[k, p] - vr[0]) * np.exp(-tt / 2.0), 0)
        x = self.X[k, p] + np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
        return x, v

    def own_fit(self, k, m, H):
        """Predecessor transition fitted with its end speed known (estimate of CAV m): shape and duration from the
        speed observed since its departure; KS1 with the prior duration if it has not departed."""
        p = m - 1
        src = self.sources.get(("B", p), self.sources.get(("path", m), (None,)))[0]
        vend = self.est[m]
        if src is None or src.kd is None:
            return self._template(self.X[k, p], self.V[k, p], vend, H)
        k0, lvl = src.k0, src.level
        y = self.V[k0:k + 1, p] - lvl
        dv = vend - lvl
        t_obs = np.arange(len(y)) * DT
        best = None
        for sh in SHAPE_INFO.values():
            for T in np.arange(max(2.0, t_obs[-1] * 0.5), max(40.0, t_obs[-1] + 20.0), 0.2):
                sse = np.sum((y - dv * sh["h"](np.clip(t_obs / T, 0, 1))) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, sh, T)
        _, sh, T = best
        tt = t_obs[-1] + np.arange(H) * DT
        v = lvl + dv * sh["h"](np.clip(tt / T, 0, 1))
        v = v + (self.V[k, p] - v[0]) * np.exp(-np.arange(H) * DT / 2.0)
        x = self.X[k, p] + np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
        return x, v

    def front_forecast(self, k, m, H, use_plans=True):
        """Forecast of the predecessor of CAV m from step k, with the information CAV m holds."""
        cfg = self.cfg
        if cfg.forecast == "oracle" and (m - 1) in self.replay:       # diagnostic: the recorded future
            xr, vr, _ = self.ev.rec[m - 1]
            idx = k + np.arange(H)
            last = min(self.ev.K - 1, len(xr) - 1)
            x = np.where(idx <= last, xr[np.minimum(idx, last)], xr[last] + vr[last] * (idx - last) * DT)
            return x, vr[np.minimum(idx, last)]
        if cfg.forecast in ("template", "oracle"):
            p = m - 1
            if p in self.cav and p in self.plan and use_plans:
                pr = self._as_rows(self.plan[p], k, H, 1)
                return pr["x"][0], pr["v"][0]
            if cfg.info == "broadcast":
                vL = self.ev.vL[np.clip(k + np.arange(H), 0, self.ev.K - 1)]
                return self.newell_copy(k, p, 0, vL, H)
            known = self.relayed.get(m, {}) if (use_plans and cfg.info == "relay") else {}
            known = {u: self.plan.get(u, q) for u, q in known.items()}
            if known:
                u = max(known)
                return self.newell_copy(k, p, u, self._as_rows(known[u], k, H, 1)["v"][0], H)
            return self.own_fit(k, m, H)
        if cfg.info == "broadcast":
            xL = self.ev.xL[k:k + H]; vL = self.ev.vL[k:k + H]
            if len(xL) < H:
                xL = np.concatenate([xL, xL[-1] + vL[-1] * DT * np.arange(1, H - len(xL) + 1)])
                vL = np.concatenate([vL, np.full(H - len(vL), vL[-1])])
            plans = {u: self.plan[u] for u in self.plan if u < m} if use_plans else {}
            xp, vp, _ = self.forecast_chain(k, 1, m - 1, xL[None], vL[None], plans, H) if m > 1 else (xL[None], vL[None], 0)
            return xp[0], vp[0]
        known = self.relayed.get(m, {}) if (use_plans and cfg.info == "relay") else {}
        known = {u: self.plan.get(u, p) for u, p in known.items()}
        if (m - 1) in self.cav and (m - 1) in self.plan and use_plans:
            known[m - 1] = self.plan[m - 1]
        if known:
            u = max(known)                                         # nearest upstream CAV with a known plan
            pr = self._as_rows(known[u], k, H, 1)
            xs, vs = pr["x"][0], pr["v"][0]
            if u == m - 1:
                return xs, vs
            xp, vp, _ = self.forecast_chain(k, u + 1, m - 1, xs[None], vs[None], {}, H)
            return xp[0], vp[0]
        return self._template(self.X[k, m - 1], self.V[k, m - 1], self.est[m], H)

    # ------------------------------------------------------------------ scheduling
    def _own_best(self, k, m, xp, vp, H, force=False):
        cfg = self.cfg
        x0, v0, u0 = self.X[k, m], self.V[k, m], self.A[max(k - 1, 0), m]
        xf, vf, af, _ = roll_cav(cfg, xp[None], vp[None], x0, v0, u0)
        J_fb = float(cost(af, jerk(af), cfg.cj)[0])
        c = candidates(cfg, self.est[m], xp, vp, x0, v0, a0=u0) \
            if abs(self.est[m] - v0) >= cfg.gate and self._corroborated(k, m) else None
        if c is None:
            return None, J_fb, 0
        ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
        n = len(c["e"])
        _, _, aa, _ = roll_cav(cfg, np.repeat(xp[None], n, 0), np.repeat(vp[None], n, 0), x0, v0, u0, ref=ref, ideal=True)
        J = cost(aa, jerk(aa), cfg.cj)
        i = int(np.argmin(J))
        if J[i] >= J_fb and not force:
            return None, J_fb, n
        return dict(k0=k, x=c["x"][i], v=c["v"][i], a=c["a"][i], gref=c["gref"][i], e=int(c["e"][i]),
                    shape=c["shape"][i], s=float(c["s"][i]), T=float(c["T"][i]), J=float(J[i])), J_fb, n

    def schedule(self, k, trig):
        """Solve the schedule for the open CAVs at or behind the first triggered one; plans ahead of it are kept."""
        cfg = self.cfg
        H = min(int(cfg.horizon / DT), self.ev.K - k)
        open_all = [m for m in sorted(self.cav) if m in self.tI and m not in self.closed]
        trig = trig & set(open_all)
        for m in trig:
            self.basis[m] = self.est[m]
        if H < 50 or not trig:
            return
        first = min(trig)
        open_ = [m for m in open_all if m >= first]
        self.n["schedule"] += 1
        if cfg.mode in ("independent", "sequential"):
            todo = sorted(trig) if cfg.mode == "independent" else open_
            for m in todo:
                xp, vp = self.front_forecast(k, m, H, use_plans=(cfg.mode == "sequential"))
                p, J_fb, n = self._own_best(k, m, xp, vp, H)
                self._commit(k, m, p, n, J_fb)
            return
        # joint: tree over the open CAVs; the first open CAV's predecessor from its own information
        c1 = open_[0]
        xp, vp = self.front_forecast(k, c1, H)
        branches = [dict(xp=xp, vp=vp, J=0.0, choice={})]
        bounds = open_[1:] + [24 + 1]
        info = {}
        for j, m in enumerate(open_):
            nxt = []
            for br in branches:
                x0, v0, u0 = self.X[k, m], self.V[k, m], self.A[max(k - 1, 0), m]
                c = candidates(cfg, self.est[m], br["xp"], br["vp"], x0, v0, a0=u0) \
                    if abs(self.est[m] - v0) >= cfg.gate and self._corroborated(k, m) else None
                opts = [None]
                if c is not None:
                    ref = dict(x=c["x"], v=c["v"], a=c["a"], gref=c["gref"], e=c["e"])
                    nC = len(c["e"])
                    _, _, aa, _ = roll_cav(cfg, np.repeat(br["xp"][None], nC, 0), np.repeat(br["vp"][None], nC, 0),
                                           x0, v0, u0, ref=ref, ideal=True)
                    J = cost(aa, jerk(aa), cfg.cj)
                    pick = set(np.argsort(J)[:3].tolist())
                    for sh in set(c["shape"]):
                        ii = [i for i in range(nC) if c["shape"][i] == sh]
                        pick |= {min(ii, key=lambda i: J[i])}
                    pick = sorted(pick, key=lambda i: J[i])[:cfg.n_keep]
                    opts += [dict(k0=k, x=c["x"][i], v=c["v"][i], a=c["a"][i], gref=c["gref"][i], e=int(c["e"][i]),
                                  shape=c["shape"][i], s=float(c["s"][i]), T=float(c["T"][i]), J=float(J[i]))
                             for i in pick]
                    info[m] = nC
                nO = len(opts)
                comb = dict(x=np.zeros((nO, H)), v=np.zeros((nO, H)), a=np.zeros((nO, H)), gref=np.zeros((nO, H)),
                            e=np.full(nO, -1))
                for o, p in enumerate(opts):
                    if p is not None:
                        for q in ("x", "v", "a", "gref"):
                            comb[q][o] = p[q]
                        comb["e"][o] = p["e"]
                xq, vq, Js = self.forecast_chain(k, m, bounds[j] - 1, np.repeat(br["xp"][None], nO, 0),
                                                 np.repeat(br["vp"][None], nO, 0), {m: comb}, H)
                if cfg.forecast == "template" and bounds[j] - 1 > m and bounds[j] <= 24:
                    vm_rows = self._first_rows[1]
                    pairs = [self.newell_copy(k, bounds[j] - 1, m, vm_rows[o], H) for o in range(nO)]
                    xq = np.array([q[0] for q in pairs]); vq = np.array([q[1] for q in pairs])
                for o, p in enumerate(opts):
                    ch = dict(br["choice"]); ch[m] = p
                    nxt.append(dict(xp=xq[o], vp=vq[o], J=br["J"] + float(Js[o]), choice=ch))
            branches = sorted(nxt, key=lambda b: b["J"])[:cfg.beam]
        best = branches[0]
        for m in open_:
            self._commit(k, m, best["choice"][m], info.get(m, 0), None)

    def _commit(self, k, m, p, n_adm, J_fb):
        ev = self.ev
        self.basis[m] = self.est[m]
        if p is None:
            if m in self.plan:
                self.log.append((ev.t[k], m, "back to feedback", f"{n_adm} admissible"))
            self.plan.pop(m, None)
        else:
            p["id"] = self.next_id; self.next_id += 1
            self.plan[m] = p
            self.log.append((ev.t[k], m, "commit", f"{p['shape']} start +{p['s']:.1f} s, T {p['T']:.1f} s, "
                                                   f"{n_adm} admissible"))
        self._relay(m, k)

    # ------------------------------------------------------------------ execution
    def control(self, k):
        cfg, ev = self.cfg, self.ev
        X, V, A = self.X, self.V, self.A
        u = np.zeros(25)
        hdv = [m for m in range(1, 25) if m not in self.cav]
        j = max(k - int(round(ev.human_tau / DT)), 0)                 # human reaction delay
        for m in hdv:
            u[m] = idm_acc(ev.idm_true[m - 1], X[j, m - 1] - X[j, m] - L_EFF, V[j, m], V[j, m - 1])
        escalate = set()
        for m in sorted(self.cav):
            g = X[k, m - 1] - X[k, m] - L_EFF
            un = float(np.clip(cfg.kg * (g - cfg.rs.sr(V[k, m])) + cfg.kd * (V[k, m - 1] - V[k, m]), A_MIN, A_MAX))
            p = self.plan.get(m)
            if p is not None:
                i = k - p["k0"]
                if i < p["e"]:
                    D = np.hypot((g - p["gref"][i]) / cfg.sig_g, (V[k, m] - p["v"][i]) / cfg.sig_v)
                    if D >= cfg.D2:
                        escalate.add(m)
                    elif D >= cfg.D1 and k - self.last_re.get(m, -99) >= 10:
                        self.last_re[m] = k
                        if not self._reanchor(k, m):
                            escalate.add(m)
                    p = self.plan.get(m)
                    i = k - p["k0"]
                    un = p["a"][i] + cfg.kg * (g - p["gref"][i]) + cfg.kv * (p["v"][i] - V[k, m])
                else:
                    self.plan.pop(m)
                    self.log.append((ev.t[k], m, "completed", ""))
            if m in self.tI and m not in self.plan and m in self.est and abs(V[k, m] - self.est[m]) < 0.3 \
                    and k >= 20 and np.all(np.abs(A[k - 20:k, m]) <= 0.3):
                self.closed.add(m)
            un = float(np.clip(un, self.u_prev[m] - J_MAX * DT, self.u_prev[m] + J_MAX * DT))
            uf, bad = _filter(np.array([un]), np.array([g]), np.array([V[k, m]]), np.array([V[k, m - 1]]))
            self.n["filter"] += int(bad[0])
            u[m] = uf[0]
        return u, escalate

    def _reanchor(self, k, m):
        """Moderate deviation: a new arc from the realised state to the same end state, end time within cfg.retime."""
        cfg = self.cfg
        p = self.plan[m]
        H = min(int(cfg.horizon / DT), self.ev.K - k)
        xp, vp = self.front_forecast(k, m, H)
        c = candidates(cfg, self.est[m], xp, vp, self.X[k, m], self.V[k, m], start_only=True, a0=self.A[max(k - 1, 0), m])
        if c is None:
            return False
        e_old = p["k0"] + p["e"]
        ok = np.where(np.abs((k + c["e"]) - e_old) * DT <= cfg.retime)[0]
        if len(ok) == 0:
            return False
        J = np.array([np.sum(c["a"][i] ** 2) for i in ok])
        i = int(ok[np.argmin(J)])
        self.plan[m] = dict(k0=k, x=c["x"][i], v=c["v"][i], a=c["a"][i], gref=c["gref"][i], e=int(c["e"][i]),
                            shape=c["shape"][i], s=0.0, T=float(c["T"][i]), J=None, id=self.next_id)
        self.next_id += 1
        self.n["reanchor"] += 1
        self.log.append((self.ev.t[k], m, "re-anchor", f"{c['shape'][i]} T {c['T'][i]:.1f} s"))
        return True

    # ------------------------------------------------------------------ main loop
    def run(self):
        ev, cfg = self.ev, self.cfg
        pending = set()
        if self._init_sources:
            for m in sorted(self.cav):
                self._sources_of(m)
        for k in range(ev.K - 1):
            if cfg.mode in ("independent", "sequential", "joint"):
                trig = self.update_info(k) | pending
                pending = set()
                if trig:
                    self.schedule(k, trig)
            u, esc = self.control(k)
            if esc:
                self.n["escalate"] += len(esc)
                for m in esc:
                    self.log.append((ev.t[k], m, "sent back", "large deviation or no re-anchor"))
                    self.plan.pop(m, None)
                pending |= esc
            u[0] = (ev.vL[k + 1] - ev.vL[k]) / DT
            self.A[k] = u
            self.u_prev = u.copy()
            self.V[k + 1] = np.maximum(self.V[k] + u * DT, 0)
            self.X[k + 1] = self.X[k] + 0.5 * (self.V[k] + self.V[k + 1]) * DT
            self.X[k + 1, 0], self.V[k + 1, 0] = ev.xL[k + 1], ev.vL[k + 1]
            for r in self.replay:
                self.X[k + 1, r], self.V[k + 1, r] = ev.rec[r][0][k + 1], ev.rec[r][1][k + 1]
                self.A[k, r] = ev.rec[r][2][k]
        self.A[-1] = self.A[-2]
        return self

    def metrics(self):
        ev = self.ev
        k0 = ev.k_on
        J = cost(self.A[k0:, 1:].T, jerk(self.A[k0:, 1:].T), self.cfg.cj)
        f = self.first                                              # ranks ahead of it are the record in every mode
        dev = (self.V[k0:, 1:] - ev.v2) * (1 if ev.kind == "accel" else -1)        # beyond the end speed
        gap = self.X[k0:, :-1] - self.X[k0:, 1:] - L_EFF
        return dict(J_tot=float(J[f - 1:].sum()), J_rank=J.round(3).tolist(),
                    excursion_rank_kmh=(np.maximum(dev.max(0), 0) * 3.6).round(2).tolist(),
                    tail_excursion_kmh=float(max(dev[:, -1].max(), 0) * 3.6), min_gap=float(gap.min()),
                    **self.n, n_commit=sum(1 for q in self.log if q[2] == "commit"))
